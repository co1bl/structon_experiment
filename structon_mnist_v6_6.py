"""
Structon Vision v6.6 - Proper WHAT/WHERE Separation
====================================================

核心修复：真正分离特征(WHAT)和位置(WHERE)

之前的问题：
- 特征和位置混在一起
- 分割基于位置（路径追踪）
- 分类基于位置上的方向分布
- 没有位置无关的特征表示

v6.6的设计：

1. WHAT (特征 - 位置无关):
   - 不是"这个位置的方向是什么"
   - 而是"这个区域的Gabor响应模式是什么"
   - 用特征向量表示，可以在任何位置匹配

2. WHERE (位置 - 特征无关):
   - 不参与特征计算
   - 只用于计算原子间的空间关系
   - 锚点：center, top, bottom, left, right

3. 原子检测策略：
   - 把图像分成固定的网格区域（不是追踪路径）
   - 每个区域计算特征向量（WHAT）
   - 每个区域记录位置（WHERE）
   - 用特征聚类找原子，不是用空间聚类

4. 拓扑特征（更稳定）：
   - 端点数量
   - 交叉点数量
   - 闭合区域数量

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Types
# =============================================================================

class AtomType(Enum):
    """原子类型 - 基于特征模式，不是位置"""
    HORIZONTAL = "H"      # 横向笔画
    VERTICAL = "V"        # 纵向笔画
    DIAGONAL_UP = "DU"    # 斜向上 /
    DIAGONAL_DOWN = "DD"  # 斜向下 \
    CURVE_LEFT = "CL"     # 左弯曲
    CURVE_RIGHT = "CR"    # 右弯曲
    JUNCTION = "J"        # 交叉/连接点
    ENDPOINT = "E"        # 端点
    EMPTY = "empty"


class SpatialRelation(Enum):
    """空间关系 - 纯位置，不涉及特征"""
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bot_left"
    BOTTOM_RIGHT = "bot_right"


# =============================================================================
# 2. Data Structures - WHAT/WHERE 分离
# =============================================================================

@dataclass
class FeatureVector:
    """
    WHAT - 纯特征，位置无关
    
    这个向量描述"看起来像什么"，不管它在哪里
    """
    orientation_histogram: np.ndarray  # 方向直方图 [8维]
    intensity: float                    # 平均强度
    edge_density: float                # 边缘密度
    curvature: float                   # 曲率（方向变化程度）
    
    def to_vector(self) -> np.ndarray:
        return np.concatenate([
            self.orientation_histogram,
            [self.intensity, self.edge_density, self.curvature]
        ])
    
    def similarity(self, other: 'FeatureVector') -> float:
        v1 = self.to_vector()
        v2 = other.to_vector()
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))


@dataclass
class SpatialPosition:
    """
    WHERE - 纯位置，特征无关
    
    这个结构描述"在哪里"，不管它是什么
    """
    region: SpatialRelation  # 在图像的哪个区域
    center_x: float
    center_y: float
    
    def relative_to(self, other: 'SpatialPosition') -> str:
        """计算相对于另一个位置的方向"""
        dy = self.center_y - other.center_y
        dx = self.center_x - other.center_x
        
        if abs(dy) < 3 and abs(dx) < 3:
            return "same"
        
        if abs(dy) > abs(dx):
            return "below" if dy > 0 else "above"
        else:
            return "right" if dx > 0 else "left"


@dataclass
class Atom:
    """
    原子 = WHAT + WHERE
    
    特征和位置分开存储，分开使用
    """
    what: FeatureVector      # 这是什么（特征）
    where: SpatialPosition   # 在哪里（位置）
    atom_type: AtomType      # 分类结果（从what推断）
    
    def __repr__(self):
        return f"Atom({self.atom_type.value}@{self.where.region.value})"


@dataclass(frozen=True)
class StructuralSignature:
    """
    结构签名
    
    WHAT层面：有哪些类型的原子
    WHERE层面：原子间的空间关系
    """
    # WHAT: 原子类型的多重集（允许重复）
    atom_types: Tuple[str, ...]
    
    # WHERE: 空间关系集合
    spatial_relations: FrozenSet[str]
    
    # 拓扑特征（最稳定）
    n_endpoints: int
    n_junctions: int
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 原子类型相似度
        set1 = set(self.atom_types)
        set2 = set(other.atom_types)
        type_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0
        
        # 空间关系相似度
        rel_sim = (len(self.spatial_relations & other.spatial_relations) / 
                   len(self.spatial_relations | other.spatial_relations)
                   if self.spatial_relations | other.spatial_relations else 0.5)
        
        # 拓扑相似度（最重要）
        endpoint_match = 1.0 if self.n_endpoints == other.n_endpoints else 0.5
        junction_match = 1.0 if self.n_junctions == other.n_junctions else 0.5
        topo_sim = (endpoint_match + junction_match) / 2
        
        # 拓扑特征权重最高
        return 0.25 * type_sim + 0.25 * rel_sim + 0.5 * topo_sim


# =============================================================================
# 3. Feature Extractor (WHAT)
# =============================================================================

class FeatureExtractor:
    """
    特征提取器 - 只负责 WHAT
    
    给定一个图像区域，计算其特征向量
    特征是位置无关的：同样的笔画在不同位置应该有相同的特征
    """
    
    def __init__(self):
        self.n_orientations = 8
        self.orientations = np.linspace(0, np.pi, self.n_orientations, endpoint=False)
        self.filters = self._create_filters()
    
    def _create_filters(self) -> List[np.ndarray]:
        """创建方向滤波器"""
        filters = []
        size = 5
        half = size // 2
        
        for theta in self.orientations:
            kernel = np.zeros((size, size), dtype=np.float32)
            for y in range(-half, half + 1):
                for x in range(-half, half + 1):
                    x_theta = x * np.cos(theta) + y * np.sin(theta)
                    y_theta = -x * np.sin(theta) + y * np.cos(theta)
                    kernel[y + half, x + half] = np.exp(-x_theta**2 / 4) * np.cos(2 * np.pi * y_theta / 4)
            kernel -= kernel.mean()
            norm = np.linalg.norm(kernel)
            if norm > 1e-8:
                kernel /= norm
            filters.append(kernel)
        
        return filters
    
    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """卷积"""
        kh, kw = kernel.shape
        ih, iw = image.shape
        pad = kh // 2
        padded = np.pad(image, pad, mode='constant')
        output = np.zeros((ih, iw), dtype=np.float32)
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return output
    
    def extract_region_features(self, image: np.ndarray, 
                                 y1: int, y2: int, x1: int, x2: int) -> FeatureVector:
        """
        提取一个区域的特征 (WHAT)
        
        这些特征描述"这个区域看起来像什么"
        与区域的绝对位置无关
        """
        region = image[y1:y2, x1:x2]
        
        if region.size == 0 or region.max() < 0.1:
            return FeatureVector(
                orientation_histogram=np.zeros(self.n_orientations),
                intensity=0.0,
                edge_density=0.0,
                curvature=0.0
            )
        
        # 1. 方向直方图（位置无关）
        responses = []
        for filt in self.filters:
            if region.shape[0] >= filt.shape[0] and region.shape[1] >= filt.shape[1]:
                resp = self._convolve(region, filt)
                responses.append(np.mean(np.abs(resp)))
            else:
                responses.append(0.0)
        
        orientation_hist = np.array(responses)
        total = orientation_hist.sum()
        if total > 1e-8:
            orientation_hist /= total
        
        # 2. 强度（位置无关）
        intensity = float(np.mean(region))
        
        # 3. 边缘密度（位置无关）
        edge_density = float(np.mean(region > 0.3))
        
        # 4. 曲率 = 方向变化程度（位置无关）
        # 高曲率 = 方向分散，低曲率 = 方向集中
        if total > 1e-8:
            curvature = 1.0 - np.max(orientation_hist)  # 0=直线, 1=圆
        else:
            curvature = 0.0
        
        return FeatureVector(
            orientation_histogram=orientation_hist,
            intensity=intensity,
            edge_density=edge_density,
            curvature=curvature
        )
    
    def classify_atom_type(self, features: FeatureVector) -> AtomType:
        """
        根据特征分类原子类型
        
        这个分类只看 WHAT，不看 WHERE
        """
        if features.edge_density < 0.05:
            return AtomType.EMPTY
        
        hist = features.orientation_histogram
        max_idx = np.argmax(hist)
        max_val = hist[max_idx]
        
        # 高曲率 = 弯曲或交叉
        if features.curvature > 0.6:
            # 检查是否有多个强方向（交叉）
            strong_dirs = np.sum(hist > 0.15)
            if strong_dirs >= 3:
                return AtomType.JUNCTION
            else:
                # 弯曲方向
                if max_idx in [0, 4]:  # 水平
                    return AtomType.CURVE_LEFT if hist[2] > hist[6] else AtomType.CURVE_RIGHT
                else:
                    return AtomType.CURVE_LEFT
        
        # 低曲率 = 直线
        if max_val > 0.3:
            # 强主方向
            angle = self.orientations[max_idx]
            
            if angle < np.pi/8 or angle > 7*np.pi/8:
                return AtomType.VERTICAL
            elif np.pi/8 <= angle < 3*np.pi/8:
                return AtomType.DIAGONAL_UP
            elif 3*np.pi/8 <= angle < 5*np.pi/8:
                return AtomType.HORIZONTAL
            elif 5*np.pi/8 <= angle < 7*np.pi/8:
                return AtomType.DIAGONAL_DOWN
        
        return AtomType.CURVE_LEFT  # 默认


# =============================================================================
# 4. Topology Analyzer (拓扑特征 - 最稳定)
# =============================================================================

class TopologyAnalyzer:
    """
    拓扑分析器
    
    提取位置无关的拓扑特征：
    - 端点数量
    - 交叉点数量
    - 闭合区域数量
    
    这些特征对变形非常稳定
    """
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
    
    def analyze(self, image: np.ndarray) -> Tuple[int, int]:
        """分析图像的拓扑特征"""
        # 二值化
        binary = (image > self.threshold).astype(np.uint8)
        
        # 计算端点和交叉点
        n_endpoints = 0
        n_junctions = 0
        
        h, w = binary.shape
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if binary[y, x] == 0:
                    continue
                
                # 计算8邻域连通性
                neighbors = [
                    binary[y-1, x-1], binary[y-1, x], binary[y-1, x+1],
                    binary[y, x-1],                   binary[y, x+1],
                    binary[y+1, x-1], binary[y+1, x], binary[y+1, x+1]
                ]
                
                n_neighbors = sum(neighbors)
                
                # 计算crossing number
                # 环绕邻居，计算0->1的跳变次数
                ring = [
                    binary[y-1, x], binary[y-1, x+1], binary[y, x+1], binary[y+1, x+1],
                    binary[y+1, x], binary[y+1, x-1], binary[y, x-1], binary[y-1, x-1]
                ]
                
                crossings = 0
                for i in range(8):
                    if ring[i] == 0 and ring[(i+1) % 8] == 1:
                        crossings += 1
                
                # 端点：只有1个邻居
                if n_neighbors == 1 or crossings == 1:
                    n_endpoints += 1
                
                # 交叉点：3个或更多分支
                if crossings >= 3:
                    n_junctions += 1
        
        # 归一化（大概数值）
        n_endpoints = min(n_endpoints // 3, 4)  # 最多4个端点
        n_junctions = min(n_junctions // 3, 3)  # 最多3个交叉点
        
        return n_endpoints, n_junctions


# =============================================================================
# 5. Spatial Analyzer (WHERE)
# =============================================================================

class SpatialAnalyzer:
    """
    空间分析器 - 只负责 WHERE
    
    把图像分成3x3网格，分析每个区域
    """
    
    def __init__(self):
        self.grid_size = 3
        self.regions = [
            SpatialRelation.TOP_LEFT, SpatialRelation.TOP, SpatialRelation.TOP_RIGHT,
            SpatialRelation.LEFT, SpatialRelation.CENTER, SpatialRelation.RIGHT,
            SpatialRelation.BOTTOM_LEFT, SpatialRelation.BOTTOM, SpatialRelation.BOTTOM_RIGHT
        ]
    
    def get_region_bounds(self, image_h: int, image_w: int, region_idx: int) -> Tuple[int, int, int, int]:
        """获取区域边界"""
        row = region_idx // self.grid_size
        col = region_idx % self.grid_size
        
        cell_h = image_h // self.grid_size
        cell_w = image_w // self.grid_size
        
        y1 = row * cell_h
        y2 = (row + 1) * cell_h if row < self.grid_size - 1 else image_h
        x1 = col * cell_w
        x2 = (col + 1) * cell_w if col < self.grid_size - 1 else image_w
        
        return y1, y2, x1, x2
    
    def create_position(self, region_idx: int, y1: int, y2: int, x1: int, x2: int) -> SpatialPosition:
        """创建位置对象"""
        return SpatialPosition(
            region=self.regions[region_idx],
            center_x=(x1 + x2) / 2,
            center_y=(y1 + y2) / 2
        )


# =============================================================================
# 6. Atom Detector (组合 WHAT + WHERE)
# =============================================================================

class AtomDetector:
    """
    原子检测器
    
    对每个空间区域：
    1. 提取特征 (WHAT)
    2. 记录位置 (WHERE)
    3. 组合成原子
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.spatial_analyzer = SpatialAnalyzer()
        self.topology_analyzer = TopologyAnalyzer()
    
    def detect(self, image: np.ndarray) -> Tuple[List[Atom], int, int]:
        """检测原子并返回拓扑特征"""
        h, w = image.shape
        atoms = []
        
        # 拓扑分析
        n_endpoints, n_junctions = self.topology_analyzer.analyze(image)
        
        # 对每个空间区域
        for idx in range(9):
            y1, y2, x1, x2 = self.spatial_analyzer.get_region_bounds(h, w, idx)
            
            # 提取 WHAT
            features = self.feature_extractor.extract_region_features(image, y1, y2, x1, x2)
            atom_type = self.feature_extractor.classify_atom_type(features)
            
            # 跳过空区域
            if atom_type == AtomType.EMPTY:
                continue
            
            # 记录 WHERE
            position = self.spatial_analyzer.create_position(idx, y1, y2, x1, x2)
            
            # 组合成原子
            atoms.append(Atom(
                what=features,
                where=position,
                atom_type=atom_type
            ))
        
        return atoms, n_endpoints, n_junctions


# =============================================================================
# 7. Relation Analyzer
# =============================================================================

class RelationAnalyzer:
    """分析原子间的空间关系（纯 WHERE）"""
    
    def compute_relations(self, atoms: List[Atom]) -> List[str]:
        """计算原子间的空间关系"""
        relations = []
        
        for i, a1 in enumerate(atoms):
            for j, a2 in enumerate(atoms):
                if i >= j:
                    continue
                
                rel = a1.where.relative_to(a2.where)
                if rel != "same":
                    # 格式：类型1_关系_类型2
                    rel_str = f"{a1.atom_type.value}_{rel}_{a2.atom_type.value}"
                    relations.append(rel_str)
        
        return relations
    
    def build_signature(self, atoms: List[Atom], n_endpoints: int, n_junctions: int) -> StructuralSignature:
        """构建结构签名"""
        # WHAT: 原子类型（按区域排序以保持一致性）
        sorted_atoms = sorted(atoms, key=lambda a: (a.where.center_y, a.where.center_x))
        atom_types = tuple(a.atom_type.value for a in sorted_atoms)
        
        # WHERE: 空间关系
        relations = self.compute_relations(atoms)
        
        return StructuralSignature(
            atom_types=atom_types,
            spatial_relations=frozenset(relations),
            n_endpoints=n_endpoints,
            n_junctions=n_junctions
        )


# =============================================================================
# 8. Memory & Vision System
# =============================================================================

class StructuralMemory:
    def __init__(self):
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
    
    def learn(self, signature: StructuralSignature, label: str):
        self.label_signatures[label].append(signature)
    
    def recognize(self, signature: StructuralSignature) -> Tuple[Optional[str], float]:
        best_label = None
        best_score = 0.0
        
        for label, sigs in self.label_signatures.items():
            for stored in sigs:
                sim = signature.similarity(stored)
                if sim > best_score:
                    best_score = sim
                    best_label = label
        
        return (best_label, best_score) if best_score > 0.2 else (None, 0.0)
    
    def analyze(self):
        print("\n=== 结构签名分析 (WHAT/WHERE分离) ===")
        for label in sorted(self.label_signatures.keys()):
            sigs = self.label_signatures[label]
            print(f"\n数字 {label}:")
            
            # 拓扑特征统计
            endpoint_dist = defaultdict(int)
            junction_dist = defaultdict(int)
            
            for sig in sigs:
                endpoint_dist[sig.n_endpoints] += 1
                junction_dist[sig.n_junctions] += 1
            
            print(f"  端点数: {dict(endpoint_dist)}")
            print(f"  交叉点: {dict(junction_dist)}")
            
            # 原子类型统计
            type_counts = defaultdict(int)
            for sig in sigs:
                for t in sig.atom_types:
                    type_counts[t] += 1
            print(f"  原子类型: {dict(type_counts)}")


class VisionSystem:
    def __init__(self):
        self.atom_detector = AtomDetector()
        self.relation_analyzer = RelationAnalyzer()
        self.memory = StructuralMemory()
    
    def train(self, image: np.ndarray, label: str):
        atoms, n_endpoints, n_junctions = self.atom_detector.detect(image)
        signature = self.relation_analyzer.build_signature(atoms, n_endpoints, n_junctions)
        self.memory.learn(signature, label)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        atoms, n_endpoints, n_junctions = self.atom_detector.detect(image)
        signature = self.relation_analyzer.build_signature(atoms, n_endpoints, n_junctions)
        label, conf = self.memory.recognize(signature)
        return label or "unknown", conf
    
    def debug_image(self, image: np.ndarray) -> dict:
        atoms, n_endpoints, n_junctions = self.atom_detector.detect(image)
        relations = self.relation_analyzer.compute_relations(atoms)
        
        return {
            'atoms': [(a.atom_type.value, a.where.region.value) for a in atoms],
            'relations': relations,
            'topology': {'endpoints': n_endpoints, 'junctions': n_junctions}
        }


# =============================================================================
# 9. MNIST & Experiment
# =============================================================================

def load_mnist(data_dir='./mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
               "https://ossci-datasets.s3.amazonaws.com/mnist/"]
    files = {'train_images': 'train-images-idx3-ubyte.gz',
             'train_labels': 'train-labels-idx1-ubyte.gz',
             'test_images': 't10k-images-idx3-ubyte.gz',
             'test_labels': 't10k-labels-idx1-ubyte.gz'}
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            for mirror in mirrors:
                try:
                    urllib.request.urlretrieve(mirror + filename, filepath)
                    break
                except: continue
    
    def load_images(path):
        with gzip.open(path, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c).astype(np.float32) / 255
    
    def load_labels(path):
        with gzip.open(path, 'rb') as f:
            struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    return (load_images(os.path.join(data_dir, files['train_images'])),
            load_labels(os.path.join(data_dir, files['train_labels'])),
            load_images(os.path.join(data_dir, files['test_images'])),
            load_labels(os.path.join(data_dir, files['test_labels'])))


def run_experiment(n_train_per_class=10, n_test=500, verbose=True):
    print("=" * 60)
    print("Structon Vision v6.6 - WHAT/WHERE Separation")
    print("=" * 60)
    print("\n核心改进：")
    print("  WHAT: 区域的特征向量（位置无关）")
    print("  WHERE: 区域的空间位置（特征无关）")
    print("  拓扑: 端点数、交叉点数（最稳定）")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = VisionSystem()
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.memory.analyze()
    
    print("\n测试中...")
    results = {str(d): {'c': 0, 't': 0} for d in range(10)}
    test_idx = np.random.choice(len(test_images), n_test, replace=False)
    
    t0 = time.time()
    for i, idx in enumerate(test_idx):
        true = str(test_labels[idx])
        pred, _ = system.predict(test_images[idx])
        results[true]['t'] += 1
        if pred == true:
            results[true]['c'] += 1
        
        if verbose and (i+1) % 100 == 0:
            acc = sum(r['c'] for r in results.values()) / (i+1) * 100
            print(f"  {i+1}/{n_test}: {acc:.1f}%")
    
    print(f"测试完成: {time.time()-t0:.1f}秒")
    
    total_c = sum(r['c'] for r in results.values())
    total_t = sum(r['t'] for r in results.values())
    
    print(f"\n总准确率: {total_c/total_t*100:.1f}%")
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        acc = r['c']/r['t']*100 if r['t'] else 0
        print(f"  {d}: {acc:.1f}% ({r['c']}/{r['t']})")
    
    return system


def debug_digits(n=3):
    print("\n=== 调试: WHAT/WHERE 分离 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"  样本{i+1}: 拓扑={info['topology']}")
            print(f"    原子: {info['atoms']}")
            if info['relations']:
                print(f"    关系: {info['relations'][:5]}...")  # 只显示前5个


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        debug_digits(3)
    else:
        run_experiment(args.train_per_class, args.test)
