"""
Structon Vision v7.3 - Resonance-Based Emergence
=================================================

v7.2的问题：
- 完整特征集必须完全匹配才能涌现
- 太严格，导致很多模式无法涌现
- 例如：8的两个HOLE没有涌现为独特模式

v7.3的改进：
- 部分匹配 / 共振涌现
- 核心特征重复就可以涌现
- 更接近真实的LRM - 不是精确匹配，而是"共振"

关键机制：
1. 特征分组：按类型分组 (HOLE, EP, JC, EDGE)
2. 核心签名：提取核心特征组合
3. 共振匹配：核心特征相似就算匹配
4. 层次涌现：核心模式涌现后，组合成更高层

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, FrozenSet, Counter
from dataclasses import dataclass, field
from collections import defaultdict, Counter as CounterType
import os
import gzip
import struct
import urllib.request
import time
import hashlib


# =============================================================================
# 1. Basic Types
# =============================================================================

@dataclass(frozen=True)
class AtomicFeature:
    """Level 0: 原子特征"""
    feature_type: str  # 'EP', 'JC', 'E_H', 'E_V', 'E_D1', 'E_D2', 'HOLE'
    position: str      # 'TL', 'T', 'TR', 'L', 'C', 'R', 'BL', 'B', 'BR', 'GLOBAL'
    
    def __str__(self):
        return f"{self.feature_type}@{self.position}"
    
    @property
    def category(self) -> str:
        """特征类别"""
        if self.feature_type == 'HOLE':
            return 'topology'
        elif self.feature_type in ('EP', 'JC'):
            return 'topology'
        else:
            return 'edge'


@dataclass
class CoreSignature:
    """
    核心签名 - 用于共振匹配
    
    不是完整特征集，而是关键特征的统计
    """
    n_holes: int
    n_endpoints: int
    n_junctions: int
    
    # 端点位置分布
    endpoint_positions: FrozenSet[str]
    
    # 交叉点位置分布
    junction_positions: FrozenSet[str]
    
    # 边缘方向统计
    edge_histogram: Tuple[int, int, int, int]  # H, V, D1, D2
    
    def __hash__(self):
        return hash((self.n_holes, self.n_endpoints, self.n_junctions,
                    self.endpoint_positions, self.junction_positions,
                    self.edge_histogram))
    
    def __eq__(self, other):
        return (isinstance(other, CoreSignature) and
                self.n_holes == other.n_holes and
                self.n_endpoints == other.n_endpoints and
                self.n_junctions == other.n_junctions and
                self.endpoint_positions == other.endpoint_positions and
                self.junction_positions == other.junction_positions and
                self.edge_histogram == other.edge_histogram)
    
    def resonance(self, other: 'CoreSignature') -> float:
        """
        计算与另一个签名的共振度
        
        共振 = 核心特征的相似程度
        """
        # 拓扑特征必须匹配 (权重最高)
        if self.n_holes != other.n_holes:
            hole_sim = 0.0
        else:
            hole_sim = 1.0
        
        ep_diff = abs(self.n_endpoints - other.n_endpoints)
        ep_sim = 1.0 / (1.0 + ep_diff)
        
        jc_diff = abs(self.n_junctions - other.n_junctions)
        jc_sim = 1.0 / (1.0 + jc_diff)
        
        # 位置匹配
        if self.endpoint_positions and other.endpoint_positions:
            ep_pos_inter = len(self.endpoint_positions & other.endpoint_positions)
            ep_pos_union = len(self.endpoint_positions | other.endpoint_positions)
            ep_pos_sim = ep_pos_inter / ep_pos_union if ep_pos_union > 0 else 1.0
        else:
            ep_pos_sim = 1.0 if self.endpoint_positions == other.endpoint_positions else 0.5
        
        if self.junction_positions and other.junction_positions:
            jc_pos_inter = len(self.junction_positions & other.junction_positions)
            jc_pos_union = len(self.junction_positions | other.junction_positions)
            jc_pos_sim = jc_pos_inter / jc_pos_union if jc_pos_union > 0 else 1.0
        else:
            jc_pos_sim = 1.0 if self.junction_positions == other.junction_positions else 0.5
        
        # 边缘直方图相似度 (余弦相似度)
        h1 = np.array(self.edge_histogram)
        h2 = np.array(other.edge_histogram)
        norm1 = np.linalg.norm(h1)
        norm2 = np.linalg.norm(h2)
        if norm1 > 0 and norm2 > 0:
            edge_sim = float(np.dot(h1, h2) / (norm1 * norm2))
        else:
            edge_sim = 1.0 if np.array_equal(h1, h2) else 0.0
        
        # 加权组合
        return (0.35 * hole_sim + 
                0.15 * ep_sim + 
                0.15 * jc_sim + 
                0.10 * ep_pos_sim +
                0.10 * jc_pos_sim +
                0.15 * edge_sim)


@dataclass
class EmergedPattern:
    """涌现的模式"""
    pattern_id: str
    level: int
    core_signature: CoreSignature
    frequency: int = 0
    
    def __hash__(self):
        return hash(self.pattern_id)


# =============================================================================
# 2. Resonant Memory with Partial Matching
# =============================================================================

class ResonantMemory:
    """
    共振记忆 - 支持部分匹配
    
    关键改进：
    - 不要求完全匹配
    - 核心特征相似就可以共振
    - 共振度超过阈值就认为是同一模式
    """
    
    def __init__(self, emergence_threshold: int = 2, resonance_threshold: float = 0.85):
        self.emergence_threshold = emergence_threshold
        self.resonance_threshold = resonance_threshold
        
        # 存储涌现的模式
        self.patterns: List[EmergedPattern] = []
        
        # 签名到模式的映射 (用于精确匹配)
        self.signature_to_pattern: Dict[CoreSignature, EmergedPattern] = {}
        
        # 观察计数 (按核心签名)
        self.observation_count: Dict[CoreSignature, int] = defaultdict(int)
        
        self.pattern_counter = 0
    
    def observe(self, signature: CoreSignature, level: int) -> Optional[EmergedPattern]:
        """
        观察一个核心签名
        
        1. 检查是否与已有模式共振
        2. 如果共振 → 返回已有模式
        3. 如果不共振 → 记录观察，检查是否应该涌现
        """
        # 1. 检查是否与已有模式共振
        best_match = None
        best_resonance = 0.0
        
        for pattern in self.patterns:
            res = signature.resonance(pattern.core_signature)
            if res > best_resonance:
                best_resonance = res
                best_match = pattern
        
        if best_resonance >= self.resonance_threshold:
            # 共振成功，增加频率
            best_match.frequency += 1
            return best_match
        
        # 2. 没有共振，记录新观察
        self.observation_count[signature] += 1
        count = self.observation_count[signature]
        
        # 3. 检查是否应该涌现
        if count >= self.emergence_threshold:
            pattern = self._create_pattern(signature, level, count)
            self.patterns.append(pattern)
            self.signature_to_pattern[signature] = pattern
            return pattern
        
        return None
    
    def _create_pattern(self, signature: CoreSignature, level: int, freq: int) -> EmergedPattern:
        """创建新涌现的模式"""
        self.pattern_counter += 1
        
        # 基于核心特征生成有意义的ID
        id_parts = []
        if signature.n_holes > 0:
            id_parts.append(f"H{signature.n_holes}")
        if signature.n_endpoints > 0:
            id_parts.append(f"E{signature.n_endpoints}")
        if signature.n_junctions > 0:
            id_parts.append(f"J{signature.n_junctions}")
        
        if not id_parts:
            id_parts.append("X")
        
        pattern_id = f"L{level}_{'_'.join(id_parts)}_{self.pattern_counter}"
        
        return EmergedPattern(
            pattern_id=pattern_id,
            level=level,
            core_signature=signature,
            frequency=freq
        )
    
    def find_best_match(self, signature: CoreSignature) -> Optional[Tuple[EmergedPattern, float]]:
        """找最匹配的模式"""
        best_match = None
        best_resonance = 0.0
        
        for pattern in self.patterns:
            res = signature.resonance(pattern.core_signature)
            if res > best_resonance:
                best_resonance = res
                best_match = pattern
        
        if best_match and best_resonance >= self.resonance_threshold * 0.8:  # 稍微放宽匹配
            return (best_match, best_resonance)
        return None
    
    def get_stats(self) -> dict:
        return {
            'n_patterns': len(self.patterns),
            'patterns': [
                {
                    'id': p.pattern_id,
                    'freq': p.frequency,
                    'holes': p.core_signature.n_holes,
                    'endpoints': p.core_signature.n_endpoints,
                    'junctions': p.core_signature.n_junctions
                }
                for p in sorted(self.patterns, key=lambda x: -x.frequency)[:10]
            ]
        }


# =============================================================================
# 3. Feature Detector (Fixed Level 0)
# =============================================================================

class FeatureDetector:
    """Level 0: 固定特征检测"""
    
    def __init__(self):
        self.regions = ['TL', 'T', 'TR', 'L', 'C', 'R', 'BL', 'B', 'BR']
    
    def detect(self, image: np.ndarray) -> Tuple[List[AtomicFeature], CoreSignature]:
        """检测原子特征并生成核心签名"""
        h, w = image.shape
        features = []
        
        # 全局拓扑
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(image)
        
        # 添加HOLE特征
        for _ in range(n_holes):
            features.append(AtomicFeature('HOLE', 'GLOBAL'))
        
        # 区域分析
        endpoint_positions = set()
        junction_positions = set()
        edge_counts = [0, 0, 0, 0]  # H, V, D1, D2
        
        for idx, region_name in enumerate(self.regions):
            row, col = idx // 3, idx % 3
            y1, y2 = row * h // 3, (row + 1) * h // 3
            x1, x2 = col * w // 3, (col + 1) * w // 3
            
            region = image[y1:y2, x1:x2]
            density = np.mean(region > 0.3)
            
            if density < 0.05:
                continue
            
            # 边缘方向
            direction = self._detect_direction(region)
            if direction:
                features.append(AtomicFeature(direction, region_name))
                if direction == 'E_H':
                    edge_counts[0] += 1
                elif direction == 'E_V':
                    edge_counts[1] += 1
                elif direction == 'E_D1':
                    edge_counts[2] += 1
                elif direction == 'E_D2':
                    edge_counts[3] += 1
            
            # 端点
            region_eps = [(y, x) for y, x in endpoints if y1 <= y < y2 and x1 <= x < x2]
            for _ in region_eps:
                features.append(AtomicFeature('EP', region_name))
                endpoint_positions.add(region_name)
            
            # 交叉点
            region_jcs = [(y, x) for y, x in junctions if y1 <= y < y2 and x1 <= x < x2]
            for _ in region_jcs:
                features.append(AtomicFeature('JC', region_name))
                junction_positions.add(region_name)
        
        # 生成核心签名
        core_sig = CoreSignature(
            n_holes=n_holes,
            n_endpoints=len(endpoints),
            n_junctions=len(junctions),
            endpoint_positions=frozenset(endpoint_positions),
            junction_positions=frozenset(junction_positions),
            edge_histogram=tuple(edge_counts)
        )
        
        return features, core_sig
    
    def _detect_direction(self, region: np.ndarray) -> Optional[str]:
        if region.shape[0] < 3 or region.shape[1] < 3:
            return None
        
        gy = np.abs(region[2:, :] - region[:-2, :]).sum()
        gx = np.abs(region[:, 2:] - region[:, :-2]).sum()
        gd1 = np.abs(region[2:, 2:] - region[:-2, :-2]).sum()
        gd2 = np.abs(region[2:, :-2] - region[:-2, 2:]).sum()
        
        gradients = [gx, gy, gd1, gd2]
        max_idx = np.argmax(gradients)
        max_val = gradients[max_idx]
        
        if max_val < 0.5:
            return None
        
        return ['E_V', 'E_H', 'E_D1', 'E_D2'][max_idx]
    
    def _skeletonize(self, image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        binary = (image > threshold).astype(np.uint8)
        skeleton = binary.copy()
        h, w = skeleton.shape
        
        def neighbors(y, x):
            return [
                skeleton[y-1, x], skeleton[y-1, x+1], skeleton[y, x+1], skeleton[y+1, x+1],
                skeleton[y+1, x], skeleton[y+1, x-1], skeleton[y, x-1], skeleton[y-1, x-1]
            ]
        
        def transitions(n):
            n = n + [n[0]]
            return sum(n[i] == 0 and n[i+1] == 1 for i in range(8))
        
        changed = True
        iterations = 0
        
        while changed and iterations < 50:
            changed = False
            iterations += 1
            
            to_remove = []
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if skeleton[y, x] == 0:
                        continue
                    P = neighbors(y, x)
                    B = sum(P)
                    A = transitions(P)
                    if 2 <= B <= 6 and A == 1 and P[0]*P[2]*P[4] == 0 and P[2]*P[4]*P[6] == 0:
                        to_remove.append((y, x))
            
            for y, x in to_remove:
                skeleton[y, x] = 0
                changed = True
            
            to_remove = []
            for y in range(1, h-1):
                for x in range(1, w-1):
                    if skeleton[y, x] == 0:
                        continue
                    P = neighbors(y, x)
                    B = sum(P)
                    A = transitions(P)
                    if 2 <= B <= 6 and A == 1 and P[0]*P[2]*P[6] == 0 and P[0]*P[4]*P[6] == 0:
                        to_remove.append((y, x))
            
            for y, x in to_remove:
                skeleton[y, x] = 0
                changed = True
        
        return skeleton
    
    def _find_topology_points(self, skeleton: np.ndarray) -> Tuple[List, List]:
        h, w = skeleton.shape
        endpoints = []
        junctions = []
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                ring = [
                    skeleton[y-1, x], skeleton[y-1, x+1], skeleton[y, x+1], skeleton[y+1, x+1],
                    skeleton[y+1, x], skeleton[y+1, x-1], skeleton[y, x-1], skeleton[y-1, x-1]
                ]
                
                crossings = sum(ring[i] != ring[(i+1) % 8] for i in range(8)) // 2
                
                if crossings == 1:
                    endpoints.append((y, x))
                elif crossings >= 3:
                    junctions.append((y, x))
        
        return endpoints, junctions
    
    def _count_holes(self, image: np.ndarray, threshold: float = 0.3) -> int:
        binary = (image > threshold).astype(np.uint8)
        h, w = binary.shape
        
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary
        
        visited = np.zeros_like(padded, dtype=bool)
        
        def flood_fill(start_y, start_x):
            queue = [(start_y, start_x)]
            visited[start_y, start_x] = True
            while queue:
                y, x = queue.pop(0)
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                        if padded[ny, nx] == 0 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
        
        flood_fill(0, 0)
        
        n_holes = 0
        for y in range(1, h + 1):
            for x in range(1, w + 1):
                if padded[y, x] == 0 and not visited[y, x]:
                    queue = [(y, x)]
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.pop(0)
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                                if padded[ny, nx] == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
                    n_holes += 1
        
        return n_holes


# =============================================================================
# 4. Vision System
# =============================================================================

class StructonVisionSystem:
    """
    Structon视觉系统 v7.3
    
    - Level 0: 固定特征检测
    - Level 1+: 共振记忆涌现
    """
    
    def __init__(self, emergence_threshold: int = 2, resonance_threshold: float = 0.85):
        self.detector = FeatureDetector()
        self.memory = ResonantMemory(emergence_threshold, resonance_threshold)
        
        # 标签记忆
        self.label_memory: Dict[str, List[Tuple[CoreSignature, Optional[EmergedPattern]]]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> Tuple[CoreSignature, Optional[EmergedPattern]]:
        """处理图像"""
        features, core_sig = self.detector.detect(image)
        pattern = self.memory.observe(core_sig, level=1)
        return core_sig, pattern
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        core_sig, pattern = self.process(image)
        self.label_memory[label].append((core_sig, pattern))
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        core_sig, pattern = self.process(image)
        
        best_label = None
        best_score = 0.0
        
        for label, stored_items in self.label_memory.items():
            for stored_sig, stored_pattern in stored_items:
                # 计算共振度
                score = core_sig.resonance(stored_sig)
                
                # 如果都有涌现模式且匹配，加分
                if pattern and stored_pattern and pattern.pattern_id == stored_pattern.pattern_id:
                    score = min(1.0, score + 0.2)
                
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试"""
        features, core_sig = self.detector.detect(image)
        _, pattern = self.process(image)
        
        return {
            'n_features': len(features),
            'features': [str(f) for f in features[:15]],
            'core': {
                'holes': core_sig.n_holes,
                'endpoints': core_sig.n_endpoints,
                'junctions': core_sig.n_junctions,
                'ep_positions': list(core_sig.endpoint_positions),
                'jc_positions': list(core_sig.junction_positions),
                'edge_hist': core_sig.edge_histogram
            },
            'pattern': pattern.pattern_id if pattern else None
        }
    
    def print_emerged_patterns(self):
        """打印涌现的模式"""
        print("\n=== 涌现的模式 (共振记忆) ===")
        stats = self.memory.get_stats()
        print(f"共 {stats['n_patterns']} 个模式涌现:")
        for p in stats['patterns']:
            print(f"  {p['id']}: freq={p['freq']}, holes={p['holes']}, "
                  f"endpoints={p['endpoints']}, junctions={p['junctions']}")
    
    def print_label_signatures(self):
        """打印每个标签的核心特征"""
        print("\n=== 各数字的核心特征 ===")
        for label in sorted(self.label_memory.keys()):
            items = self.label_memory[label]
            
            # 统计
            holes = [s.n_holes for s, _ in items]
            endpoints = [s.n_endpoints for s, _ in items]
            junctions = [s.n_junctions for s, _ in items]
            patterns = [p.pattern_id if p else None for _, p in items]
            
            print(f"\n数字 {label}:")
            print(f"  空洞: {CounterType(holes)}")
            print(f"  端点: {CounterType(endpoints)}")
            print(f"  交叉: {CounterType(junctions)}")
            print(f"  涌现模式: {CounterType(patterns)}")


# =============================================================================
# 5. MNIST Experiment
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
    print("Structon Vision v7.3 - Resonance-Based Emergence")
    print("=" * 60)
    print("\n核心改进:")
    print("  - 部分匹配 / 共振涌现")
    print("  - 核心特征相似就可以共振")
    print("  - 不要求完全匹配")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = StructonVisionSystem(emergence_threshold=2, resonance_threshold=0.85)
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.print_emerged_patterns()
        system.print_label_signatures()
    
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
    print("\n=== 调试: 共振涌现 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = StructonVisionSystem(emergence_threshold=2, resonance_threshold=0.85)
    
    # 训练
    print("训练中...")
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:5]
        for idx in indices:
            system.train(train_images[idx], str(digit))
    
    system.print_emerged_patterns()
    
    # 详细分析
    print("\n=== 各数字详细分析 ===")
    for digit in range(10):
        print(f"\n{'='*40}")
        print(f"数字 {digit}")
        print('='*40)
        
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"\n  样本 {i+1}:")
            print(f"    核心: holes={info['core']['holes']}, "
                  f"ep={info['core']['endpoints']}, jc={info['core']['junctions']}")
            print(f"    EP位置: {info['core']['ep_positions']}")
            print(f"    JC位置: {info['core']['jc_positions']}")
            print(f"    涌现: {info['pattern']}")


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
