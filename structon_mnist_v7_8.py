"""
Structon Vision v7.8 - True Free Emergence
==========================================

v7.7的问题：
- 人为设计每层看什么特征
- Level 1 只看拓扑，Level 2 加位置，等等
- 这不是真正的涌现，是预设的层级

v7.8的核心改变：
- 所有特征都进入 Level 1
- LRM 自己发现哪些组合频繁
- 频繁组合涌现为新模式，成为下一层的输入
- 没有人为设计什么特征去什么层

真正的自由涌现：
1. Level 1 看所有原子特征
2. 频繁的原子组合 → 涌现为 L1 模式
3. Level 2 看所有 L1 模式 + 所有原子特征
4. 频繁的 L1+原子组合 → 涌现为 L2 模式
5. 以此类推...

关键：系统自己发现层级结构，而不是人设计

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import os
import gzip
import struct
import urllib.request
import time
import hashlib


# =============================================================================
# 1. Atomic Feature (单个原子特征)
# =============================================================================

@dataclass(frozen=True)
class Atom:
    """单个原子特征 - 不可分割的最小单位"""
    name: str
    
    def __str__(self):
        return self.name
    
    def __hash__(self):
        return hash(self.name)


# =============================================================================
# 2. Local Resonant Memory - True Free Version
# =============================================================================

class FreeLocalResonantMemory:
    """
    自由涌现的局部共振记忆
    
    - 接收任意原子集合
    - 统计哪些组合频繁出现
    - 频繁组合涌现为新的命名模式
    - 新模式可以作为更高层的输入
    """
    
    def __init__(self, level: int, emergence_threshold: int = 3):
        self.level = level
        self.emergence_threshold = emergence_threshold
        
        # 观察计数: frozenset of atoms → count
        self.observations: Dict[FrozenSet[str], int] = defaultdict(int)
        
        # 涌现的模式: frozenset → pattern_name
        self.emerged_patterns: Dict[FrozenSet[str], str] = {}
        
        # 反向映射: pattern_name → frozenset
        self.pattern_to_atoms: Dict[str, FrozenSet[str]] = {}
        
        self.counter = 0
    
    def observe(self, atoms: FrozenSet[str]) -> Tuple[Optional[str], FrozenSet[str]]:
        """
        观察一组原子
        
        Returns:
            (emerged_pattern_name, atoms_that_didnt_emerge)
        """
        if not atoms:
            return None, frozenset()
        
        self.observations[atoms] += 1
        count = self.observations[atoms]
        
        # 已经涌现
        if atoms in self.emerged_patterns:
            return self.emerged_patterns[atoms], frozenset()
        
        # 检查是否应该涌现
        if count >= self.emergence_threshold:
            self.counter += 1
            pattern_name = f"L{self.level}P{self.counter}"
            self.emerged_patterns[atoms] = pattern_name
            self.pattern_to_atoms[pattern_name] = atoms
            return pattern_name, frozenset()
        
        # 没有涌现，返回原子
        return None, atoms
    
    def find_partial_matches(self, atoms: FrozenSet[str]) -> List[Tuple[str, FrozenSet[str], float]]:
        """
        找部分匹配的已涌现模式
        
        Returns:
            [(pattern_name, matched_atoms, match_ratio), ...]
        """
        matches = []
        
        for pattern_atoms, pattern_name in self.emerged_patterns.items():
            # 计算交集
            intersection = atoms & pattern_atoms
            if intersection:
                # 匹配比例 = 交集 / 模式大小
                ratio = len(intersection) / len(pattern_atoms)
                if ratio >= 0.5:  # 至少50%匹配
                    matches.append((pattern_name, intersection, ratio))
        
        return sorted(matches, key=lambda x: -x[2])  # 按匹配度排序
    
    def get_stats(self) -> dict:
        """获取统计"""
        return {
            'n_observations': len(self.observations),
            'n_emerged': len(self.emerged_patterns),
            'patterns': [
                {
                    'name': name,
                    'atoms': list(atoms)[:5],  # 只显示前5个
                    'n_atoms': len(atoms),
                    'freq': self.observations[atoms]
                }
                for atoms, name in sorted(
                    self.emerged_patterns.items(),
                    key=lambda x: -self.observations[x[0]]
                )[:10]
            ]
        }


# =============================================================================
# 3. Hierarchical Free Emergence System
# =============================================================================

class FreeEmergenceSystem:
    """
    自由涌现系统
    
    - 所有原子特征进入系统
    - 多层LRM，每层自由发现模式
    - 下层涌现的模式成为上层的输入
    """
    
    def __init__(self, n_levels: int = 4, emergence_threshold: int = 3):
        self.n_levels = n_levels
        self.emergence_threshold = emergence_threshold
        
        # 每层的LRM
        self.lrms = [
            FreeLocalResonantMemory(level=i+1, emergence_threshold=emergence_threshold)
            for i in range(n_levels)
        ]
        
        # 特征提取器
        self.extractor = FeatureExtractor()
        
        # 标签记忆: label → [(层级模式列表, 原子特征)]
        self.label_memory: Dict[str, List[Tuple[List[Optional[str]], FrozenSet[str]]]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> Tuple[List[Optional[str]], FrozenSet[str]]:
        """
        处理图像 - 自由涌现
        
        Returns:
            (每层涌现的模式列表, 原始原子特征)
        """
        # Level 0: 提取所有原子特征
        atoms = self.extractor.extract_atoms(image)
        original_atoms = atoms.copy()
        
        emerged_patterns = []
        
        # 逐层处理
        current_input = atoms
        
        for level, lrm in enumerate(self.lrms):
            # 观察当前输入
            pattern, remaining = lrm.observe(current_input)
            emerged_patterns.append(pattern)
            
            if pattern:
                # 涌现了！模式成为下一层的一部分
                # 下一层输入 = 涌现的模式 + 原始原子 (让上层也能看到原子)
                current_input = frozenset([pattern]) | original_atoms
            else:
                # 没涌现，尝试找部分匹配
                matches = lrm.find_partial_matches(current_input)
                if matches:
                    best_pattern, _, _ = matches[0]
                    emerged_patterns[-1] = best_pattern  # 用部分匹配
                    current_input = frozenset([best_pattern]) | original_atoms
                else:
                    # 完全没匹配，保持原子
                    current_input = original_atoms
        
        return emerged_patterns, original_atoms
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        patterns, atoms = self.process(image)
        self.label_memory[label].append((patterns, atoms))
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        patterns, atoms = self.process(image)
        
        best_label = None
        best_score = 0.0
        
        for label, stored_list in self.label_memory.items():
            for stored_patterns, stored_atoms in stored_list:
                score = self._compute_similarity(patterns, atoms, stored_patterns, stored_atoms)
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def _compute_similarity(self, patterns1, atoms1, patterns2, atoms2) -> float:
        """计算相似度"""
        score = 0.0
        
        # 层级模式匹配 (高层权重更高)
        for i, (p1, p2) in enumerate(zip(patterns1, patterns2)):
            weight = (i + 1) / len(patterns1)  # L1: 0.25, L2: 0.5, L3: 0.75, L4: 1.0
            if p1 and p2 and p1 == p2:
                score += weight * 0.6  # 60% 来自模式匹配
        
        # 原子特征匹配 (Jaccard)
        if atoms1 and atoms2:
            intersection = len(atoms1 & atoms2)
            union = len(atoms1 | atoms2)
            jaccard = intersection / union if union > 0 else 0
            score += 0.4 * jaccard  # 40% 来自原子匹配
        
        return score
    
    def print_emergence_stats(self):
        """打印涌现统计"""
        print("\n" + "=" * 70)
        print("自由涌现统计 (Free Emergence)")
        print("=" * 70)
        
        for i, lrm in enumerate(self.lrms):
            stats = lrm.get_stats()
            print(f"\nLevel {i+1}: {stats['n_emerged']} 模式从 {stats['n_observations']} 种观察中涌现")
            
            if stats['patterns']:
                print("  涌现的模式:")
                for p in stats['patterns'][:5]:
                    atoms_str = ', '.join(p['atoms'])
                    if p['n_atoms'] > 5:
                        atoms_str += f", ... (+{p['n_atoms']-5} more)"
                    print(f"    {p['name']}: freq={p['freq']}, n_atoms={p['n_atoms']}")
                    print(f"      atoms: [{atoms_str}]")
    
    def print_label_patterns(self):
        """打印各标签的涌现模式"""
        print("\n" + "=" * 70)
        print("各数字的涌现模式")
        print("=" * 70)
        
        for label in sorted(self.label_memory.keys()):
            items = self.label_memory[label]
            
            # 统计每层的模式分布
            level_counters = [Counter() for _ in range(self.n_levels)]
            
            for patterns, _ in items:
                for i, p in enumerate(patterns):
                    level_counters[i][p or "None"] += 1
            
            print(f"\n数字 {label}:")
            for i, counter in enumerate(level_counters):
                top_patterns = counter.most_common(3)
                print(f"  L{i+1}: {top_patterns}")


# =============================================================================
# 4. Feature Extractor - Returns Atoms
# =============================================================================

class FeatureExtractor:
    """特征提取器 - 返回原子集合"""
    
    def extract_atoms(self, image: np.ndarray) -> FrozenSet[str]:
        """提取所有原子特征，返回字符串集合"""
        atoms = set()
        h, w = image.shape
        
        # 预处理
        binary = (image > 0.3).astype(np.uint8)
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(image)
        
        # === 拓扑原子 ===
        atoms.add(f"holes={n_holes}")
        atoms.add(f"endpoints={len(endpoints)}")
        atoms.add(f"junctions={len(junctions)}")
        
        # 闭合性
        if len(endpoints) == 0 and n_holes >= 1:
            atoms.add("closed")
        else:
            atoms.add("open")
        
        # === 端点位置原子 ===
        for y, x in endpoints:
            if y < h / 3:
                atoms.add("ep_top")
            elif y < 2 * h / 3:
                atoms.add("ep_mid")
            else:
                atoms.add("ep_bot")
            
            if x < w / 3:
                atoms.add("ep_left")
            elif x < 2 * w / 3:
                atoms.add("ep_center")
            else:
                atoms.add("ep_right")
        
        # === 交叉位置原子 ===
        for y, x in junctions:
            if y < h / 3:
                atoms.add("jc_top")
            elif y < 2 * h / 3:
                atoms.add("jc_mid")
            else:
                atoms.add("jc_bot")
        
        # === 边缘方向原子 ===
        # 检查各区域的主方向
        regions = [
            ("top", 0, h//3, 0, w),
            ("mid", h//3, 2*h//3, 0, w),
            ("bot", 2*h//3, h, 0, w),
            ("left", 0, h, 0, w//3),
            ("center", 0, h, w//3, 2*w//3),
            ("right", 0, h, 2*w//3, w),
        ]
        
        for name, y1, y2, x1, x2 in regions:
            region = image[y1:y2, x1:x2]
            if np.mean(region > 0.3) > 0.05:
                direction = self._get_direction(region)
                if direction:
                    atoms.add(f"{direction}_{name}")
        
        # === 角点原子 ===
        corners = self._detect_corners(skeleton)
        if corners:
            atoms.add(f"corners={len(corners)}")
            for y, x in corners:
                if y < h / 3:
                    atoms.add("corner_top")
                elif y > 2 * h / 3:
                    atoms.add("corner_bot")
        
        # === 终止方向原子 ===
        for y, x in endpoints:
            direction = self._get_endpoint_direction(skeleton, y, x, h, w)
            if direction:
                atoms.add(f"term_{direction}")
        
        # === 开口方向原子 ===
        left_mass = np.sum(binary[:, :w//2])
        right_mass = np.sum(binary[:, w//2:])
        top_mass = np.sum(binary[:h//2, :])
        bot_mass = np.sum(binary[h//2:, :])
        
        if right_mass > left_mass * 1.8:
            atoms.add("open_left")
        if left_mass > right_mass * 1.8:
            atoms.add("open_right")
        if bot_mass > top_mass * 1.8:
            atoms.add("open_top")
        if top_mass > bot_mass * 1.8:
            atoms.add("open_bot")
        
        # === 几何原子 ===
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            
            if bbox_h > bbox_w * 1.5:
                atoms.add("tall")
            elif bbox_w > bbox_h * 1.3:
                atoms.add("wide")
            else:
                atoms.add("square")
        
        # 密度
        density = np.sum(binary) / (h * w)
        if density < 0.08:
            atoms.add("sparse")
        elif density > 0.20:
            atoms.add("dense")
        else:
            atoms.add("medium_density")
        
        # === 特殊模式原子 ===
        # 尾巴检测 (有洞 + 有端点)
        if n_holes >= 1 and len(endpoints) >= 1:
            ep_top = any(y < h/3 for y, x in endpoints)
            ep_bot = any(y > 2*h/3 for y, x in endpoints)
            if ep_top:
                atoms.add("tail_top")
            if ep_bot:
                atoms.add("tail_bot")
        
        return frozenset(atoms)
    
    def _get_direction(self, region: np.ndarray) -> Optional[str]:
        """检测区域主方向"""
        if region.shape[0] < 3 or region.shape[1] < 3:
            return None
        
        gy = np.abs(region[1:, :] - region[:-1, :]).sum()
        gx = np.abs(region[:, 1:] - region[:, :-1]).sum()
        
        if gy > gx * 1.3:
            return "H"  # 水平边缘 (垂直梯度大)
        elif gx > gy * 1.3:
            return "V"  # 垂直边缘 (水平梯度大)
        return None
    
    def _detect_corners(self, skeleton: np.ndarray) -> List[Tuple[int, int]]:
        """检测角点"""
        h, w = skeleton.shape
        corners = []
        
        for y in range(2, h - 2):
            for x in range(2, w - 2):
                if skeleton[y, x] == 0:
                    continue
                
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0
                
                if np.sum(neighbors) == 2:
                    positions = np.argwhere(neighbors)
                    if len(positions) == 2:
                        p1, p2 = positions
                        v1 = p1 - np.array([1, 1])
                        v2 = p2 - np.array([1, 1])
                        dot = np.dot(v1, v2)
                        if dot > 0.3:
                            corners.append((y, x))
        
        return corners
    
    def _get_endpoint_direction(self, skeleton: np.ndarray, y: int, x: int, h: int, w: int) -> Optional[str]:
        """获取端点方向"""
        for dy, dx, direction in [(-1, 0, 'down'), (1, 0, 'up'),
                                   (0, -1, 'right'), (0, 1, 'left')]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                return direction
        return None
    
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
            
            for phase in [0, 1]:
                to_remove = []
                for y in range(1, h-1):
                    for x in range(1, w-1):
                        if skeleton[y, x] == 0:
                            continue
                        P = neighbors(y, x)
                        B = sum(P)
                        A = transitions(P)
                        
                        if phase == 0:
                            cond = P[0]*P[2]*P[4] == 0 and P[2]*P[4]*P[6] == 0
                        else:
                            cond = P[0]*P[2]*P[6] == 0 and P[0]*P[4]*P[6] == 0
                        
                        if 2 <= B <= 6 and A == 1 and cond:
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
        
        queue = [(0, 0)]
        visited[0, 0] = True
        while queue:
            y, x = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                    if padded[ny, nx] == 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
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


def run_experiment(n_train_per_class=10, n_test=500, n_levels=4, emergence_threshold=3, verbose=True):
    print("=" * 70)
    print("Structon Vision v7.8 - True Free Emergence")
    print("=" * 70)
    print("\n自由涌现架构:")
    print("  - 所有原子特征进入 Level 1")
    print("  - LRM 自己发现频繁组合")
    print("  - 频繁组合涌现为新模式")
    print("  - 新模式成为更高层的输入")
    print("  - 没有人为设计层级结构")
    print(f"\n层数: {n_levels}, 涌现阈值: {emergence_threshold}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = FreeEmergenceSystem(n_levels=n_levels, emergence_threshold=emergence_threshold)
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.print_emergence_stats()
        system.print_label_patterns()
    
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


def debug_atoms(n=3):
    """调试原子特征"""
    print("\n=== 调试: 原子特征 ===")
    train_images, train_labels, _, _ = load_mnist()
    extractor = FeatureExtractor()
    
    for digit in range(10):
        print(f"\n{'='*50}")
        print(f"数字 {digit}")
        print('='*50)
        
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            atoms = extractor.extract_atoms(train_images[idx])
            print(f"\n  样本 {i+1}: {len(atoms)} 原子")
            print(f"    {sorted(atoms)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--levels', type=int, default=4)
    parser.add_argument('--emergence-threshold', type=int, default=3)
    parser.add_argument('--debug-atoms', action='store_true')
    args = parser.parse_args()
    
    if args.debug_atoms:
        debug_atoms(3)
    else:
        run_experiment(args.train_per_class, args.test, args.levels, args.emergence_threshold)
