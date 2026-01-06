"""
Structon Vision v7.2 - Fixed Features + Emergent Growth via LRM
================================================================

核心架构：

Level 0: 固定基础特征 (像人类V1皮层)
  - Gabor方向检测
  - 端点检测
  - 交叉点检测
  - 空洞检测
  这些是"字母表" - 固定不变

Level 1+: 通过LRM涌现
  - 当原子组合频繁出现 → 自动创建新structon
  - 没有硬编码规则
  - 模式通过观察涌现

关键原则：
1. Local rule: 只看局部关系
2. Global emergence: 复杂模式通过频率涌现
3. LRM: 共振记忆决定什么成为新概念

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
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
    """
    Level 0: 原子特征 (固定, 不学习)
    
    这些是基础"字母", 由固定算法检测
    """
    feature_type: str  # 'EP', 'JC', 'E_H', 'E_V', 'E_D1', 'E_D2', 'HOLE'
    position: str      # 'TL', 'T', 'TR', 'L', 'C', 'R', 'BL', 'B', 'BR', 'GLOBAL'
    
    def __str__(self):
        return f"{self.feature_type}@{self.position}"


@dataclass
class StructonPattern:
    """
    Level 1+: 涌现的结构模式
    
    由LRM通过观察创建, 不是硬编码
    """
    pattern_id: str                           # 自动生成的ID
    level: int                                # 层级
    components: FrozenSet[str]                # 组成成分 (可以是原子或其他pattern)
    frequency: int = 0                        # 观察次数
    
    def __hash__(self):
        return hash(self.pattern_id)
    
    def __eq__(self, other):
        return isinstance(other, StructonPattern) and self.pattern_id == other.pattern_id


# =============================================================================
# 2. Local Resonant Memory (LRM)
# =============================================================================

class LocalResonantMemory:
    """
    局部共振记忆
    
    核心机制：
    1. observe(): 观察一个模式
    2. 如果模式已存在 → 增加频率 (共振)
    3. 如果模式频繁出现 → 提升为正式structon
    4. 新模式通过频率涌现, 不是硬编码
    """
    
    def __init__(self, emergence_threshold: int = 3, resonance_threshold: float = 0.8):
        self.emergence_threshold = emergence_threshold  # 出现几次后成为正式概念
        self.resonance_threshold = resonance_threshold  # 相似度阈值
        
        # 存储: signature → StructonPattern
        self.patterns: Dict[FrozenSet[str], StructonPattern] = {}
        
        # 频率计数
        self.observation_count: Dict[FrozenSet[str], int] = defaultdict(int)
        
        # 已涌现的模式 (频率超过阈值)
        self.emerged_patterns: Dict[FrozenSet[str], StructonPattern] = {}
        
        # 模式命名计数器
        self.pattern_counter = 0
    
    def observe(self, components: FrozenSet[str], level: int) -> Optional[StructonPattern]:
        """
        观察一个组件组合
        
        Returns:
            如果该组合已涌现为structon, 返回它
            否则返回None (但会记录观察)
        """
        # 1. 记录观察
        self.observation_count[components] += 1
        count = self.observation_count[components]
        
        # 2. 检查是否已涌现
        if components in self.emerged_patterns:
            pattern = self.emerged_patterns[components]
            pattern.frequency = count
            return pattern
        
        # 3. 检查是否应该涌现 (频率超过阈值)
        if count >= self.emergence_threshold:
            pattern = self._create_pattern(components, level, count)
            self.emerged_patterns[components] = pattern
            return pattern
        
        # 4. 尚未涌现
        return None
    
    def _create_pattern(self, components: FrozenSet[str], level: int, freq: int) -> StructonPattern:
        """创建新的涌现模式"""
        self.pattern_counter += 1
        
        # 自动生成ID (基于组件的hash)
        component_str = "+".join(sorted(components))
        short_hash = hashlib.md5(component_str.encode()).hexdigest()[:6]
        pattern_id = f"P{level}_{self.pattern_counter}_{short_hash}"
        
        return StructonPattern(
            pattern_id=pattern_id,
            level=level,
            components=components,
            frequency=freq
        )
    
    def find_similar(self, components: FrozenSet[str]) -> Optional[Tuple[StructonPattern, float]]:
        """
        找相似的已涌现模式
        
        用于泛化: 即使不完全匹配, 也能识别相似模式
        """
        best_match = None
        best_score = 0.0
        
        for sig, pattern in self.emerged_patterns.items():
            score = self._compute_similarity(components, sig)
            if score > best_score and score >= self.resonance_threshold:
                best_score = score
                best_match = pattern
        
        if best_match:
            return (best_match, best_score)
        return None
    
    def _compute_similarity(self, c1: FrozenSet[str], c2: FrozenSet[str]) -> float:
        """计算两个组件集合的相似度 (Jaccard)"""
        if not c1 or not c2:
            return 0.0
        intersection = len(c1 & c2)
        union = len(c1 | c2)
        return intersection / union if union > 0 else 0.0
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'total_observations': len(self.observation_count),
            'emerged_patterns': len(self.emerged_patterns),
            'pattern_details': {
                p.pattern_id: {'freq': p.frequency, 'components': list(p.components)[:5]}
                for p in list(self.emerged_patterns.values())[:10]
            }
        }


# =============================================================================
# 3. Level 0: Fixed Feature Detectors
# =============================================================================

class FixedFeatureDetector:
    """
    Level 0: 固定特征检测器
    
    这些算法是固定的, 不学习, 像人类V1皮层
    """
    
    def __init__(self):
        self.regions = ['TL', 'T', 'TR', 'L', 'C', 'R', 'BL', 'B', 'BR']
    
    def detect(self, image: np.ndarray) -> List[AtomicFeature]:
        """检测所有原子特征"""
        features = []
        h, w = image.shape
        
        # 1. 全局拓扑特征
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(image)
        
        # 添加全局特征
        for _ in range(n_holes):
            features.append(AtomicFeature('HOLE', 'GLOBAL'))
        
        # 2. 区域特征
        for idx, region_name in enumerate(self.regions):
            row, col = idx // 3, idx % 3
            y1, y2 = row * h // 3, (row + 1) * h // 3
            x1, x2 = col * w // 3, (col + 1) * w // 3
            
            region = image[y1:y2, x1:x2]
            skeleton_region = skeleton[y1:y2, x1:x2]
            
            # 检查区域密度
            density = np.mean(region > 0.3)
            if density < 0.05:
                continue
            
            # 检测该区域的边缘方向
            direction = self._detect_direction(region)
            if direction:
                features.append(AtomicFeature(direction, region_name))
            
            # 检查该区域是否有端点
            region_endpoints = [(y, x) for y, x in endpoints 
                               if y1 <= y < y2 and x1 <= x < x2]
            for _ in region_endpoints:
                features.append(AtomicFeature('EP', region_name))
            
            # 检查该区域是否有交叉点
            region_junctions = [(y, x) for y, x in junctions 
                               if y1 <= y < y2 and x1 <= x < x2]
            for _ in region_junctions:
                features.append(AtomicFeature('JC', region_name))
        
        return features
    
    def _detect_direction(self, region: np.ndarray) -> Optional[str]:
        """检测区域的主方向"""
        if region.shape[0] < 3 or region.shape[1] < 3:
            return None
        
        # 简单梯度
        gy = np.abs(region[2:, :] - region[:-2, :]).sum()
        gx = np.abs(region[:, 2:] - region[:, :-2]).sum()
        
        # 对角线梯度
        gd1 = np.abs(region[2:, 2:] - region[:-2, :-2]).sum()  # /
        gd2 = np.abs(region[2:, :-2] - region[:-2, 2:]).sum()  # \
        
        gradients = [gx, gy, gd1, gd2]
        max_idx = np.argmax(gradients)
        max_val = gradients[max_idx]
        
        if max_val < 0.5:
            return None
        
        directions = ['E_V', 'E_H', 'E_D1', 'E_D2']
        return directions[max_idx]
    
    def _skeletonize(self, image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Zhang-Suen骨架化"""
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
        """找端点和交叉点"""
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
        """计算空洞数量"""
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
# 4. Hierarchical Structon System with LRM
# =============================================================================

class StructonVisionSystem:
    """
    层次化Structon视觉系统
    
    Level 0: 固定特征检测 (不学习)
    Level 1+: 通过LRM涌现 (学习)
    """
    
    def __init__(self, n_levels: int = 3, emergence_threshold: int = 2):
        self.n_levels = n_levels
        self.feature_detector = FixedFeatureDetector()
        
        # 每层都有自己的LRM
        self.lrm_layers = [
            LocalResonantMemory(emergence_threshold=emergence_threshold)
            for _ in range(n_levels)
        ]
        
        # 标签记忆: label → List[FrozenSet[str]]
        self.label_memory: Dict[str, List[FrozenSet[str]]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> Tuple[FrozenSet[str], List[Optional[StructonPattern]]]:
        """
        处理图像
        
        Returns:
            (最终特征集, 各层涌现的模式)
        """
        # Level 0: 固定特征检测
        atomic_features = self.feature_detector.detect(image)
        current_components = frozenset(str(f) for f in atomic_features)
        
        emerged_patterns = []
        
        # Level 1+: 通过LRM涌现
        for level, lrm in enumerate(self.lrm_layers):
            # 观察当前组件组合
            pattern = lrm.observe(current_components, level + 1)
            emerged_patterns.append(pattern)
            
            # 如果涌现了新模式, 将其ID加入组件
            if pattern:
                current_components = frozenset(
                    list(current_components) + [pattern.pattern_id]
                )
        
        return current_components, emerged_patterns
    
    def train(self, image: np.ndarray, label: str):
        """训练: 观察并记录"""
        final_components, _ = self.process(image)
        self.label_memory[label].append(final_components)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测: 找最相似的标签"""
        final_components, _ = self.process(image)
        
        best_label = None
        best_score = 0.0
        
        for label, stored_patterns in self.label_memory.items():
            for stored in stored_patterns:
                # 计算相似度
                score = self._compute_similarity(final_components, stored)
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def _compute_similarity(self, c1: FrozenSet[str], c2: FrozenSet[str]) -> float:
        """计算组件集合的相似度"""
        if not c1 or not c2:
            return 0.0
        
        # 分离原子特征和涌现模式
        atoms1 = set(c for c in c1 if not c.startswith('P'))
        atoms2 = set(c for c in c2 if not c.startswith('P'))
        patterns1 = set(c for c in c1 if c.startswith('P'))
        patterns2 = set(c for c in c2 if c.startswith('P'))
        
        # 原子特征相似度 (Jaccard)
        if atoms1 or atoms2:
            atom_intersection = len(atoms1 & atoms2)
            atom_union = len(atoms1 | atoms2)
            atom_sim = atom_intersection / atom_union if atom_union > 0 else 0
        else:
            atom_sim = 1.0
        
        # 涌现模式相似度
        if patterns1 or patterns2:
            pattern_intersection = len(patterns1 & patterns2)
            pattern_union = len(patterns1 | patterns2)
            pattern_sim = pattern_intersection / pattern_union if pattern_union > 0 else 0
        else:
            pattern_sim = 1.0
        
        # 加权组合 (涌现模式权重更高, 因为更高层抽象)
        return 0.4 * atom_sim + 0.6 * pattern_sim
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试: 显示处理过程"""
        atomic_features = self.feature_detector.detect(image)
        final_components, emerged = self.process(image)
        
        return {
            'n_atomic_features': len(atomic_features),
            'atomic_features': [str(f) for f in atomic_features],
            'emerged_patterns': [
                {'level': i+1, 'pattern': p.pattern_id if p else None, 'freq': p.frequency if p else 0}
                for i, p in enumerate(emerged)
            ],
            'final_components': list(final_components)[:20]
        }
    
    def get_lrm_stats(self) -> dict:
        """获取LRM统计"""
        stats = {}
        for i, lrm in enumerate(self.lrm_layers):
            stats[f'level_{i+1}'] = lrm.get_stats()
        return stats
    
    def print_emerged_patterns(self):
        """打印已涌现的模式"""
        print("\n=== 已涌现的模式 (通过LRM) ===")
        for i, lrm in enumerate(self.lrm_layers):
            print(f"\nLevel {i+1}: {len(lrm.emerged_patterns)} 个模式涌现")
            for sig, pattern in list(lrm.emerged_patterns.items())[:5]:
                components_str = ', '.join(list(pattern.components)[:5])
                print(f"  {pattern.pattern_id}: freq={pattern.frequency}, components=[{components_str}...]")


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
    print("Structon Vision v7.2 - Fixed Features + Emergent LRM")
    print("=" * 60)
    print("\n架构:")
    print("  Level 0: 固定特征 (Gabor, EP, JC, HOLE)")
    print("  Level 1+: 通过LRM涌现 (无硬编码规则)")
    print("\n原则:")
    print("  - 基础特征固定 (像V1皮层)")
    print("  - 高层模式通过观察涌现")
    print("  - LRM决定什么成为新概念")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = StructonVisionSystem(n_levels=3, emergence_threshold=2)
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.print_emerged_patterns()
        
        # 显示每个数字学到的模式
        print("\n=== 每个数字的特征 ===")
        for label in sorted(system.label_memory.keys()):
            patterns = system.label_memory[label]
            # 找共同的原子特征
            if patterns:
                common = set.intersection(*[set(p) for p in patterns]) if len(patterns) > 1 else set(patterns[0])
                atoms = sorted([c for c in common if not c.startswith('P')])[:8]
                emerged = sorted([c for c in common if c.startswith('P')])
                print(f"数字 {label}:")
                print(f"  共同原子: {atoms}")
                print(f"  涌现模式: {emerged}")
    
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


def debug_digits(n=2):
    print("\n=== 调试: 涌现过程 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = StructonVisionSystem(n_levels=3, emergence_threshold=2)
    
    # 先训练一些样本
    print("训练中...")
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:5]
        for idx in indices:
            system.train(train_images[idx], str(digit))
    
    # 显示涌现的模式
    system.print_emerged_patterns()
    
    # 显示每个数字的详细分析
    print("\n=== 各数字分析 ===")
    for digit in range(10):
        print(f"\n{'='*40}")
        print(f"数字 {digit}")
        print('='*40)
        
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"\n  样本 {i+1}:")
            print(f"    原子特征 ({info['n_atomic_features']}): {info['atomic_features'][:10]}...")
            for ep in info['emerged_patterns']:
                if ep['pattern']:
                    print(f"    Level {ep['level']}: {ep['pattern']} (freq={ep['freq']})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        debug_digits(2)
    else:
        run_experiment(args.train_per_class, args.test)
