"""
Structon Vision v6.5 - Orientation-Based Segmentation
======================================================

v6.4的问题：
- 连通分量检测太激进，所有激活点连成一个blob
- 每个数字只检测到1个原子，无法建立结构关系

v6.5的核心改进：
- 不用空间连通性分割，而是用方向不连续性分割
- 沿着笔画追踪，当方向突然变化时（比如转角），就是一个新原子的开始
- 这样数字"4"会被分成多个原子：竖线、横线、斜线

分割策略：
1. 把激活点按笔画路径排序
2. 沿路径走，计算相邻点的方向差
3. 当方向差超过阈值（比如45°），就切分成新的segment
4. 每个segment分析为一个原子

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, FrozenSet, Set
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
    LINE_H = "line_h"      # 横线 (80-100°)
    LINE_V = "line_v"      # 竖线 (0-20° or 160-180°)
    LINE_D1 = "line_d1"    # 斜线 / (30-60°)
    LINE_D2 = "line_d2"    # 斜线 \ (120-150°)
    CURVE = "curve"        # 曲线（方向渐变）
    CORNER = "corner"      # 转角点
    STROKE = "stroke"      # 通用笔画


class RelationType(Enum):
    ABOVE = "above"
    BELOW = "below"  
    LEFT = "left"
    RIGHT = "right"
    CONNECTED_TOP = "conn_top"      # 连接在顶部
    CONNECTED_BOTTOM = "conn_bot"   # 连接在底部
    CONNECTED_LEFT = "conn_left"    # 连接在左侧
    CONNECTED_RIGHT = "conn_right"  # 连接在右侧
    CROSSING = "crossing"


# =============================================================================
# 2. Data Structures
# =============================================================================

@dataclass
class Activation:
    x: float
    y: float
    strength: float
    orientation: int  # 0-179


@dataclass
class Anchors:
    center: Tuple[float, float]
    head: Tuple[float, float]   # 起点
    tail: Tuple[float, float]   # 终点


@dataclass
class AtomInstance:
    atom_type: AtomType
    anchors: Anchors
    activations: List[Activation]
    mean_orientation: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.anchors.center
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        xs = [a.x for a in self.activations]
        ys = [a.y for a in self.activations]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class AtomRelation:
    atom1_idx: int
    atom2_idx: int
    atom1_type: AtomType
    atom2_type: AtomType
    relation: RelationType
    
    def to_string(self) -> str:
        return f"{self.atom1_type.value}_{self.relation.value}_{self.atom2_type.value}"


@dataclass(frozen=True)
class StructuralSignature:
    atom_types: Tuple[str, ...]  # 有序的原子类型列表
    relations: FrozenSet[str]
    n_atoms: int
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 原子类型序列相似度 (考虑顺序)
        common_types = set(self.atom_types) & set(other.atom_types)
        all_types = set(self.atom_types) | set(other.atom_types)
        type_sim = len(common_types) / len(all_types) if all_types else 0
        
        # 关系相似度
        common_rels = self.relations & other.relations
        all_rels = self.relations | other.relations
        rel_sim = len(common_rels) / len(all_rels) if all_rels else 0.5
        
        # 原子数量相似度
        count_sim = 1.0 - abs(self.n_atoms - other.n_atoms) / max(self.n_atoms, other.n_atoms, 1)
        
        return 0.4 * type_sim + 0.4 * rel_sim + 0.2 * count_sim


# =============================================================================
# 3. V1 Layer
# =============================================================================

class V1GaborBank:
    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size
        self.orientations = [0, 22, 45, 67, 90, 112, 135, 157]
        self.filters = self._create_gabor_bank()
        self.n_orientations = len(self.orientations)
    
    def _create_gabor_kernel(self, theta, sigma=2.0, lambd=4.0, gamma=0.5):
        size = self.kernel_size
        half = size // 2
        kernel = np.zeros((size, size), dtype=np.float32)
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                x_theta = x * np.cos(theta) + y * np.sin(theta)
                y_theta = -x * np.sin(theta) + y * np.cos(theta)
                gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
                sinusoid = np.cos(2 * np.pi * x_theta / lambd)
                kernel[y + half, x + half] = gaussian * sinusoid
        kernel -= kernel.mean()
        norm = np.linalg.norm(kernel)
        return kernel / norm if norm > 1e-8 else kernel
    
    def _create_gabor_bank(self):
        return [self._create_gabor_kernel(np.deg2rad(theta)) for theta in self.orientations]
    
    def _convolve(self, image, kernel):
        kh, kw = kernel.shape
        ih, iw = image.shape
        pad = kh // 2
        padded = np.pad(image, pad, mode='constant')
        output = np.zeros((ih, iw), dtype=np.float32)
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return output
    
    def extract_activations(self, image: np.ndarray, threshold: float = 0.3, step: int = 2) -> List[Activation]:
        responses = [np.abs(self._convolve(image, k)) for k in self.filters]
        h, w = image.shape
        
        global_max = max(r.max() for r in responses)
        if global_max < 1e-8:
            return []
        
        responses = [r / global_max for r in responses]
        activations = []
        
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                vals = [responses[i][y, x] for i in range(self.n_orientations)]
                max_val = max(vals)
                
                if max_val < threshold:
                    continue
                
                best_idx = vals.index(max_val)
                
                activations.append(Activation(
                    x=float(x),
                    y=float(y),
                    strength=max_val,
                    orientation=self.orientations[best_idx]
                ))
        
        return activations


# =============================================================================
# 4. Stroke Segmenter - 核心改进
# =============================================================================

class StrokeSegmenter:
    """
    笔画分割器
    
    策略：沿笔画追踪，在方向不连续处切分
    """
    
    def __init__(self, angle_threshold: float = 40.0, min_segment_length: int = 4):
        self.angle_threshold = angle_threshold  # 切分阈值（度）
        self.min_segment_length = min_segment_length
    
    def segment(self, activations: List[Activation]) -> List[List[Activation]]:
        """把激活点分割成多个笔画段"""
        if len(activations) < self.min_segment_length:
            return [activations] if activations else []
        
        # 1. 把激活点排序成路径
        path = self._trace_path(activations)
        
        # 2. 找切分点
        cut_points = self._find_cut_points(path)
        
        # 3. 切分
        segments = self._split_at_cuts(path, cut_points)
        
        # 4. 过滤太短的段
        segments = [s for s in segments if len(s) >= self.min_segment_length]
        
        return segments if segments else [activations]
    
    def _trace_path(self, activations: List[Activation]) -> List[Activation]:
        """把激活点排序成连续路径（最近邻）"""
        if len(activations) <= 2:
            return activations
        
        # 从最上方的点开始（更符合书写习惯）
        remaining = list(range(len(activations)))
        start_idx = min(remaining, key=lambda i: (activations[i].y, activations[i].x))
        
        path = [activations[start_idx]]
        remaining.remove(start_idx)
        
        while remaining:
            current = path[-1]
            # 找最近的未访问点
            nearest = min(remaining, key=lambda i: 
                         (activations[i].x - current.x)**2 + (activations[i].y - current.y)**2)
            path.append(activations[nearest])
            remaining.remove(nearest)
        
        return path
    
    def _find_cut_points(self, path: List[Activation]) -> List[int]:
        """找方向突变点"""
        cuts = []
        
        # 计算每个点的局部方向变化
        window = 3  # 用前后几个点计算方向
        
        for i in range(window, len(path) - window):
            # 计算进入方向
            dx_in = path[i].x - path[i - window].x
            dy_in = path[i].y - path[i - window].y
            angle_in = np.degrees(np.arctan2(dy_in, dx_in))
            
            # 计算离开方向  
            dx_out = path[i + window].x - path[i].x
            dy_out = path[i + window].y - path[i].y
            angle_out = np.degrees(np.arctan2(dy_out, dx_out))
            
            # 计算方向差
            angle_diff = abs(angle_out - angle_in)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            if angle_diff > self.angle_threshold:
                cuts.append(i)
        
        return cuts
    
    def _split_at_cuts(self, path: List[Activation], cuts: List[int]) -> List[List[Activation]]:
        """在切分点处分割路径"""
        if not cuts:
            return [path]
        
        segments = []
        prev = 0
        
        for cut in cuts:
            if cut > prev:
                segments.append(path[prev:cut])
            prev = cut
        
        if prev < len(path):
            segments.append(path[prev:])
        
        return segments


# =============================================================================
# 5. Atom Detector
# =============================================================================

class AtomDetector:
    """原子检测器"""
    
    def __init__(self):
        self.segmenter = StrokeSegmenter(angle_threshold=40.0, min_segment_length=4)
    
    def detect_atoms(self, activations: List[Activation]) -> List[AtomInstance]:
        if len(activations) < 4:
            return []
        
        # 分割成笔画段
        segments = self.segmenter.segment(activations)
        
        # 每个段分析为一个原子
        atoms = []
        for seg in segments:
            atom = self._analyze_segment(seg)
            if atom:
                atoms.append(atom)
        
        return atoms
    
    def _analyze_segment(self, activations: List[Activation]) -> Optional[AtomInstance]:
        """分析一个笔画段"""
        if len(activations) < 3:
            return None
        
        # 计算几何特征
        xs = [a.x for a in activations]
        ys = [a.y for a in activations]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        
        # 计算平均方向
        orientations = [a.orientation for a in activations]
        
        # 处理方向的循环性 (0° 和 180° 接近)
        sin_sum = sum(np.sin(np.radians(2 * o)) for o in orientations)
        cos_sum = sum(np.cos(np.radians(2 * o)) for o in orientations)
        mean_orient = np.degrees(np.arctan2(sin_sum, cos_sum)) / 2
        if mean_orient < 0:
            mean_orient += 180
        
        # 方向标准差
        orient_var = np.var([min(abs(o - mean_orient), 180 - abs(o - mean_orient)) for o in orientations])
        
        # 分类
        if orient_var < 300:  # 方向一致 -> 直线
            atom_type = self._classify_line(mean_orient)
        else:  # 方向变化 -> 曲线
            atom_type = AtomType.CURVE
        
        # 锚点
        head_idx = min(range(len(activations)), key=lambda i: (activations[i].y, activations[i].x))
        tail_idx = max(range(len(activations)), key=lambda i: (activations[i].y, activations[i].x))
        
        anchors = Anchors(
            center=(cx, cy),
            head=(activations[head_idx].x, activations[head_idx].y),
            tail=(activations[tail_idx].x, activations[tail_idx].y)
        )
        
        return AtomInstance(
            atom_type=atom_type,
            anchors=anchors,
            activations=activations,
            mean_orientation=mean_orient
        )
    
    def _classify_line(self, orient: float) -> AtomType:
        """根据方向分类直线类型"""
        # orient 在 0-180 范围
        if orient < 20 or orient >= 160:
            return AtomType.LINE_V  # 竖线
        elif 70 <= orient < 110:
            return AtomType.LINE_H  # 横线
        elif 20 <= orient < 70:
            return AtomType.LINE_D1  # / 斜线
        else:  # 110-160
            return AtomType.LINE_D2  # \ 斜线


# =============================================================================
# 6. Relation Analyzer
# =============================================================================

class RelationAnalyzer:
    """关系分析器"""
    
    def __init__(self, connect_threshold: float = 5.0):
        self.connect_threshold = connect_threshold
    
    def compute_relations(self, atoms: List[AtomInstance]) -> List[AtomRelation]:
        relations = []
        
        for i, a1 in enumerate(atoms):
            for j, a2 in enumerate(atoms):
                if i >= j:
                    continue
                
                rel = self._classify_relation(i, j, a1, a2)
                if rel:
                    relations.append(rel)
        
        return relations
    
    def _classify_relation(self, i: int, j: int, a1: AtomInstance, a2: AtomInstance) -> Optional[AtomRelation]:
        """分类两个原子的关系"""
        
        # 检查连接（端点接近）
        connections = self._check_connections(a1, a2)
        if connections:
            return AtomRelation(i, j, a1.atom_type, a2.atom_type, connections[0])
        
        # 检查空间关系
        rel = self._spatial_relation(a1, a2)
        if rel:
            return AtomRelation(i, j, a1.atom_type, a2.atom_type, rel)
        
        return None
    
    def _check_connections(self, a1: AtomInstance, a2: AtomInstance) -> List[RelationType]:
        """检查两个原子是否在端点处连接"""
        connections = []
        
        # a1 的 tail 连接 a2 的 head
        d_tail_head = self._dist(a1.anchors.tail, a2.anchors.head)
        if d_tail_head < self.connect_threshold:
            # 判断连接方向
            dy = a2.anchors.head[1] - a1.anchors.tail[1]
            dx = a2.anchors.head[0] - a1.anchors.tail[0]
            connections.append(self._dir_to_relation(dx, dy))
        
        # a1 的 head 连接 a2 的 tail
        d_head_tail = self._dist(a1.anchors.head, a2.anchors.tail)
        if d_head_tail < self.connect_threshold:
            dy = a1.anchors.head[1] - a2.anchors.tail[1]
            dx = a1.anchors.head[0] - a2.anchors.tail[0]
            connections.append(self._dir_to_relation(dx, dy))
        
        # 中心点接近（交叉）
        d_centers = self._dist(a1.anchors.center, a2.anchors.center)
        if d_centers < self.connect_threshold:
            connections.append(RelationType.CROSSING)
        
        return connections
    
    def _dir_to_relation(self, dx: float, dy: float) -> RelationType:
        """方向转关系类型"""
        if abs(dy) > abs(dx):
            return RelationType.CONNECTED_BOTTOM if dy > 0 else RelationType.CONNECTED_TOP
        else:
            return RelationType.CONNECTED_RIGHT if dx > 0 else RelationType.CONNECTED_LEFT
    
    def _spatial_relation(self, a1: AtomInstance, a2: AtomInstance) -> Optional[RelationType]:
        """空间位置关系"""
        dy = a1.center[1] - a2.center[1]
        dx = a1.center[0] - a2.center[0]
        
        if abs(dy) > 5 or abs(dx) > 5:
            if abs(dy) > abs(dx):
                return RelationType.ABOVE if dy < 0 else RelationType.BELOW
            else:
                return RelationType.LEFT if dx < 0 else RelationType.RIGHT
        
        return None
    
    def _dist(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def build_signature(self, atoms: List[AtomInstance], relations: List[AtomRelation]) -> StructuralSignature:
        # 按空间位置排序原子类型（从上到下，从左到右）
        sorted_atoms = sorted(atoms, key=lambda a: (a.center[1], a.center[0]))
        atom_types = tuple(a.atom_type.value for a in sorted_atoms)
        
        relation_strs = frozenset(r.to_string() for r in relations)
        
        return StructuralSignature(
            atom_types=atom_types,
            relations=relation_strs,
            n_atoms=len(atoms)
        )


# =============================================================================
# 7. Memory & Vision System
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
        print("\n=== 结构签名分析 ===")
        for label in sorted(self.label_signatures.keys()):
            sigs = self.label_signatures[label]
            print(f"\n数字 {label} ({len(sigs)} 样本):")
            
            # 统计原子类型
            type_counts = defaultdict(int)
            rel_counts = defaultdict(int)
            n_atoms_dist = defaultdict(int)
            
            for sig in sigs:
                for t in sig.atom_types:
                    type_counts[t] += 1
                for r in sig.relations:
                    rel_counts[r] += 1
                n_atoms_dist[sig.n_atoms] += 1
            
            print(f"  原子数分布: {dict(n_atoms_dist)}")
            print(f"  原子类型: {dict(type_counts)}")
            if rel_counts:
                top_rels = sorted(rel_counts.items(), key=lambda x: -x[1])[:3]
                print(f"  主要关系: {dict(top_rels)}")


class VisionSystem:
    def __init__(self):
        self.v1 = V1GaborBank()
        self.atom_detector = AtomDetector()
        self.relation_analyzer = RelationAnalyzer()
        self.memory = StructuralMemory()
    
    def train(self, image: np.ndarray, label: str):
        activations = self.v1.extract_activations(image)
        atoms = self.atom_detector.detect_atoms(activations)
        relations = self.relation_analyzer.compute_relations(atoms)
        signature = self.relation_analyzer.build_signature(atoms, relations)
        self.memory.learn(signature, label)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        activations = self.v1.extract_activations(image)
        atoms = self.atom_detector.detect_atoms(activations)
        relations = self.relation_analyzer.compute_relations(atoms)
        signature = self.relation_analyzer.build_signature(atoms, relations)
        label, conf = self.memory.recognize(signature)
        return label or "unknown", conf
    
    def debug_image(self, image: np.ndarray) -> dict:
        activations = self.v1.extract_activations(image)
        atoms = self.atom_detector.detect_atoms(activations)
        relations = self.relation_analyzer.compute_relations(atoms)
        
        return {
            'n_activations': len(activations),
            'n_atoms': len(atoms),
            'atoms': [(a.atom_type.value, f"orient={a.mean_orientation:.0f}°", 
                      f"pos=({a.center[0]:.0f},{a.center[1]:.0f})") for a in atoms],
            'relations': [r.to_string() for r in relations]
        }


# =============================================================================
# 8. MNIST & Experiment
# =============================================================================

MNIST_MIRRORS = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
                 "https://ossci-datasets.s3.amazonaws.com/mnist/"]
MNIST_FILES = {'train_images': 'train-images-idx3-ubyte.gz',
               'train_labels': 'train-labels-idx1-ubyte.gz',
               'test_images': 't10k-images-idx3-ubyte.gz',
               'test_labels': 't10k-labels-idx1-ubyte.gz'}

def download_mnist(data_dir='./mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            for mirror in MNIST_MIRRORS:
                try:
                    urllib.request.urlretrieve(mirror + filename, filepath)
                    break
                except: continue
    print("MNIST ready.")

def load_mnist(data_dir='./mnist_data'):
    download_mnist(data_dir)
    def load_images(path):
        with gzip.open(path, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c).astype(np.float32) / 255
    def load_labels(path):
        with gzip.open(path, 'rb') as f:
            struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    return (load_images(os.path.join(data_dir, MNIST_FILES['train_images'])),
            load_labels(os.path.join(data_dir, MNIST_FILES['train_labels'])),
            load_images(os.path.join(data_dir, MNIST_FILES['test_images'])),
            load_labels(os.path.join(data_dir, MNIST_FILES['test_labels'])))


def run_experiment(n_train_per_class=10, n_test=500, verbose=True):
    print("=" * 60)
    print("Structon Vision v6.5 - Orientation-Based Segmentation")
    print("=" * 60)
    print("\n核心改进：按方向不连续性分割笔画，而不是空间连通性")
    print("目标：检测多个原子，建立结构关系")
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 选择训练样本
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"\n训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = VisionSystem()
    
    # 训练
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.memory.analyze()
    
    # 测试
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
    
    print(f"\n测试完成: {time.time()-t0:.1f}秒")
    
    # 结果
    total_c = sum(r['c'] for r in results.values())
    total_t = sum(r['t'] for r in results.values())
    
    print(f"\n总准确率: {total_c/total_t*100:.1f}%")
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        print(f"  {d}: {r['c']/r['t']*100 if r['t'] else 0:.1f}% ({r['c']}/{r['t']})")
    
    return system


def debug_digits(n=3):
    print("\n=== 调试 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        for idx in indices:
            info = system.debug_image(train_images[idx])
            print(f"  样本: {info['n_atoms']}原子")
            for a in info['atoms']:
                print(f"    {a}")
            if info['relations']:
                print(f"    关系: {info['relations']}")


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
