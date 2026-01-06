"""
Structon Vision v7.7 - True Hierarchical Emergence
===================================================

v7.6的问题：
- 只有一层LRM，只是统计特征组合频率
- 更多数据只是更好的Level 1统计，不是更深的结构

v7.7的核心改变：
- 每一层的涌现模式成为下一层的"原子"
- 真正的层级涌现：Level N 输出 → Level N+1 输入
- 每个structon都有自己的LRM

架构：
  Level 0: 固定基础特征 (不变)
  Level 1 LRM: 原子 → 基础模式 (LOOP, STROKE, CORNER, ...)
  Level 2 LRM: 基础模式 + 位置 → 结构模式 (LOOP_TOP, STROKE_WITH_H, ...)
  Level 3 LRM: 结构模式 → 对象模式 (数字签名)

关键：
- Level 1 模式有名字，成为 Level 2 的输入
- Level 2 模式有名字，成为 Level 3 的输入
- 真正的递归组合

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


# =============================================================================
# 1. Level 0: Fixed Atomic Features (same as v7.6)
# =============================================================================

@dataclass
class AtomicFeatures:
    """Level 0: 固定原子特征"""
    
    # 拓扑
    n_holes: int = 0
    n_endpoints: int = 0
    n_junctions: int = 0
    
    # 端点位置
    ep_top: bool = False
    ep_center: bool = False
    ep_bottom: bool = False
    
    # 交叉位置
    jc_top: bool = False
    jc_center: bool = False
    jc_bottom: bool = False
    
    # 结构
    top_horizontal: bool = False
    bottom_horizontal: bool = False
    center_vertical: bool = False
    center_horizontal: bool = False
    
    # 曲率
    n_sharp_corners: int = 0
    has_top_corner: bool = False
    
    # 终止方向
    ep_terminates_up: bool = False
    ep_terminates_down: bool = False
    ep_terminates_left: bool = False
    ep_terminates_right: bool = False
    
    # 开口
    has_curve_open_left: bool = False
    has_curve_open_right: bool = False
    
    # 连通性
    is_closed: bool = False
    
    # 特殊
    has_tail_top: bool = False
    has_tail_bottom: bool = False


# =============================================================================
# 2. Local Resonant Memory (LRM) - Core Component
# =============================================================================

class LocalResonantMemory:
    """
    局部共振记忆 - 每个Structon层的核心
    
    功能：
    1. 观察输入模式
    2. 如果模式频繁出现 → 涌现为新概念并命名
    3. 返回涌现的概念名称，作为上层的输入
    """
    
    def __init__(self, level: int, name_prefix: str, emergence_threshold: int = 3):
        self.level = level
        self.name_prefix = name_prefix
        self.emergence_threshold = emergence_threshold
        
        # 观察计数: signature → count
        self.observations: Dict[str, int] = defaultdict(int)
        
        # 涌现的模式: signature → pattern_name
        self.emerged: Dict[str, str] = {}
        
        # 模式计数器
        self.counter = 0
    
    def observe(self, signature: str) -> Optional[str]:
        """
        观察一个签名
        
        Returns:
            如果涌现，返回模式名称
            否则返回 None
        """
        self.observations[signature] += 1
        count = self.observations[signature]
        
        # 已经涌现
        if signature in self.emerged:
            return self.emerged[signature]
        
        # 检查是否应该涌现
        if count >= self.emergence_threshold:
            self.counter += 1
            pattern_name = f"{self.name_prefix}{self.counter}"
            self.emerged[signature] = pattern_name
            return pattern_name
        
        return None
    
    def get_emerged_patterns(self) -> List[Tuple[str, str, int]]:
        """获取所有涌现的模式"""
        return [
            (name, sig, self.observations[sig])
            for sig, name in sorted(self.emerged.items(), 
                                   key=lambda x: -self.observations[x[0]])
        ]


# =============================================================================
# 3. Hierarchical Structon System
# =============================================================================

class HierarchicalStructonSystem:
    """
    层次化Structon系统 - 真正的多层涌现
    
    Level 0: 固定原子特征提取
    Level 1: 拓扑模式涌现 (基于 holes, endpoints, junctions, closed)
    Level 2: 位置模式涌现 (基于 Level 1 + 端点/交叉位置)
    Level 3: 结构模式涌现 (基于 Level 2 + 边缘方向)
    Level 4: 完整模式涌现 (基于 Level 3 + 所有细节)
    """
    
    def __init__(self, emergence_threshold: int = 3):
        # 每层都有自己的LRM
        self.level1_lrm = LocalResonantMemory(1, "TOPO_", emergence_threshold)
        self.level2_lrm = LocalResonantMemory(2, "POS_", emergence_threshold)
        self.level3_lrm = LocalResonantMemory(3, "STRUCT_", emergence_threshold)
        self.level4_lrm = LocalResonantMemory(4, "OBJ_", emergence_threshold)
        
        # 特征提取器
        self.extractor = FeatureExtractor()
        
        # 标签记忆
        self.label_memory: Dict[str, List[Tuple[str, str, str, str, AtomicFeatures]]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> Tuple[str, str, str, str, AtomicFeatures]:
        """
        处理图像，通过多层LRM
        
        Returns:
            (level1_pattern, level2_pattern, level3_pattern, level4_pattern, raw_features)
        """
        # Level 0: 提取原子特征
        features = self.extractor.extract(image)
        
        # === Level 1: 拓扑模式 ===
        # 输入: 基本拓扑特征
        level1_sig = self._make_level1_signature(features)
        level1_pattern = self.level1_lrm.observe(level1_sig)
        
        # === Level 2: 位置模式 ===
        # 输入: Level 1 模式 + 位置信息
        level2_sig = self._make_level2_signature(level1_pattern or level1_sig, features)
        level2_pattern = self.level2_lrm.observe(level2_sig)
        
        # === Level 3: 结构模式 ===
        # 输入: Level 2 模式 + 边缘方向信息
        level3_sig = self._make_level3_signature(level2_pattern or level2_sig, features)
        level3_pattern = self.level3_lrm.observe(level3_sig)
        
        # === Level 4: 完整模式 ===
        # 输入: Level 3 模式 + 所有细节
        level4_sig = self._make_level4_signature(level3_pattern or level3_sig, features)
        level4_pattern = self.level4_lrm.observe(level4_sig)
        
        return (
            level1_pattern or level1_sig,
            level2_pattern or level2_sig,
            level3_pattern or level3_sig,
            level4_pattern or level4_sig,
            features
        )
    
    def _make_level1_signature(self, f: AtomicFeatures) -> str:
        """Level 1: 拓扑签名"""
        parts = [f"H{f.n_holes}", f"E{f.n_endpoints}", f"J{f.n_junctions}"]
        if f.is_closed:
            parts.append("CL")
        return "_".join(parts)
    
    def _make_level2_signature(self, level1: str, f: AtomicFeatures) -> str:
        """Level 2: Level 1 + 位置"""
        parts = [level1]
        
        # 端点位置
        ep_pos = []
        if f.ep_top:
            ep_pos.append("T")
        if f.ep_center:
            ep_pos.append("C")
        if f.ep_bottom:
            ep_pos.append("B")
        if ep_pos:
            parts.append(f"EP{''.join(ep_pos)}")
        
        # 交叉位置
        if f.jc_center:
            parts.append("JCc")
        if f.jc_top:
            parts.append("JCt")
        if f.jc_bottom:
            parts.append("JCb")
        
        # 尾巴位置
        if f.has_tail_top:
            parts.append("TailT")
        if f.has_tail_bottom:
            parts.append("TailB")
        
        return "_".join(parts)
    
    def _make_level3_signature(self, level2: str, f: AtomicFeatures) -> str:
        """Level 3: Level 2 + 边缘方向"""
        parts = [level2]
        
        if f.top_horizontal:
            parts.append("Htop")
        if f.bottom_horizontal:
            parts.append("Hbot")
        if f.center_vertical:
            parts.append("Vctr")
        if f.center_horizontal:
            parts.append("Hctr")
        
        if f.n_sharp_corners > 0:
            parts.append(f"SC{f.n_sharp_corners}")
        
        return "_".join(parts)
    
    def _make_level4_signature(self, level3: str, f: AtomicFeatures) -> str:
        """Level 4: Level 3 + 细节"""
        parts = [level3]
        
        # 终止方向
        if f.ep_terminates_left:
            parts.append("tL")
        if f.ep_terminates_right:
            parts.append("tR")
        
        # 开口
        if f.has_curve_open_left:
            parts.append("oL")
        if f.has_curve_open_right:
            parts.append("oR")
        
        return "_".join(parts)
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        result = self.process(image)
        self.label_memory[label].append(result)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        l1, l2, l3, l4, features = self.process(image)
        
        best_label = None
        best_score = 0.0
        
        for label, stored_list in self.label_memory.items():
            for stored in stored_list:
                s_l1, s_l2, s_l3, s_l4, s_features = stored
                score = self._compute_similarity(
                    (l1, l2, l3, l4, features),
                    (s_l1, s_l2, s_l3, s_l4, s_features)
                )
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def _compute_similarity(self, current: Tuple, stored: Tuple) -> float:
        """计算相似度 - 层级加权"""
        l1, l2, l3, l4, f1 = current
        s_l1, s_l2, s_l3, s_l4, f2 = stored
        
        score = 0.0
        
        # Level 4 匹配 (最高层, 权重最高)
        if l4 == s_l4:
            score += 0.35
        
        # Level 3 匹配
        if l3 == s_l3:
            score += 0.25
        
        # Level 2 匹配
        if l2 == s_l2:
            score += 0.20
        
        # Level 1 匹配 (基础拓扑)
        if l1 == s_l1:
            score += 0.15
        
        # 原子特征匹配 (细节补充)
        atom_score = self._compute_atom_similarity(f1, f2)
        score += 0.05 * atom_score
        
        return score
    
    def _compute_atom_similarity(self, f1: AtomicFeatures, f2: AtomicFeatures) -> float:
        """计算原子特征相似度"""
        matches = 0
        total = 0
        
        # 拓扑
        if f1.n_holes == f2.n_holes:
            matches += 2  # 重要
        total += 2
        
        if f1.n_endpoints == f2.n_endpoints:
            matches += 1
        total += 1
        
        # 位置
        for attr in ['ep_top', 'ep_center', 'ep_bottom', 
                     'jc_center', 'top_horizontal', 'center_vertical']:
            if getattr(f1, attr) == getattr(f2, attr):
                matches += 1
            total += 1
        
        return matches / total if total > 0 else 0
    
    def print_emergence_stats(self):
        """打印涌现统计"""
        print("\n" + "=" * 60)
        print("层次化涌现统计")
        print("=" * 60)
        
        for level, lrm in [(1, self.level1_lrm), (2, self.level2_lrm),
                           (3, self.level3_lrm), (4, self.level4_lrm)]:
            patterns = lrm.get_emerged_patterns()
            print(f"\nLevel {level} ({lrm.name_prefix}): {len(patterns)} 个模式涌现")
            for name, sig, count in patterns[:8]:
                # 简化显示
                short_sig = sig[:50] + "..." if len(sig) > 50 else sig
                print(f"  {name}: freq={count}")
                print(f"    ← {short_sig}")
    
    def print_label_analysis(self):
        """打印各标签的层级模式"""
        print("\n" + "=" * 60)
        print("各数字的层级模式分布")
        print("=" * 60)
        
        for label in sorted(self.label_memory.keys()):
            items = self.label_memory[label]
            
            l1_counter = Counter(item[0] for item in items)
            l2_counter = Counter(item[1] for item in items)
            l3_counter = Counter(item[2] for item in items)
            l4_counter = Counter(item[3] for item in items)
            
            print(f"\n数字 {label}:")
            print(f"  L1 (拓扑): {l1_counter.most_common(3)}")
            print(f"  L2 (位置): {l2_counter.most_common(2)}")
            print(f"  L3 (结构): {l3_counter.most_common(2)}")
            print(f"  L4 (完整): {l4_counter.most_common(2)}")


# =============================================================================
# 4. Feature Extractor (same as v7.6)
# =============================================================================

class FeatureExtractor:
    """特征提取器"""
    
    def extract(self, image: np.ndarray) -> AtomicFeatures:
        """提取原子特征"""
        f = AtomicFeatures()
        h, w = image.shape
        
        # 骨架化和拓扑
        binary = (image > 0.3).astype(np.uint8)
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        
        f.n_holes = self._count_holes(image)
        f.n_endpoints = len(endpoints)
        f.n_junctions = len(junctions)
        
        # 端点位置
        for y, x in endpoints:
            if y < h / 3:
                f.ep_top = True
            elif y < 2 * h / 3:
                f.ep_center = True
            else:
                f.ep_bottom = True
        
        # 交叉位置
        for y, x in junctions:
            if y < h / 3:
                f.jc_top = True
            elif y < 2 * h / 3:
                f.jc_center = True
            else:
                f.jc_bottom = True
        
        # 边缘方向
        f.top_horizontal = self._has_horizontal_in_region(image, 0, h//3)
        f.bottom_horizontal = self._has_horizontal_in_region(image, 2*h//3, h)
        f.center_vertical = self._has_vertical_in_region(image, h//3, 2*h//3, w//3, 2*w//3)
        f.center_horizontal = self._has_horizontal_in_region(image, h//3, 2*h//3)
        
        # 角点
        corners = self._detect_corners(skeleton)
        f.n_sharp_corners = len(corners)
        f.has_top_corner = any(y < h/3 for y, x in corners)
        
        # 终止方向
        for y, x in endpoints:
            direction = self._get_endpoint_direction(skeleton, y, x)
            if direction == 'up':
                f.ep_terminates_up = True
            elif direction == 'down':
                f.ep_terminates_down = True
            elif direction == 'left':
                f.ep_terminates_left = True
            elif direction == 'right':
                f.ep_terminates_right = True
        
        # 开口
        openings = self._detect_openings(image)
        f.has_curve_open_left = 'left' in openings
        f.has_curve_open_right = 'right' in openings
        
        # 连通性
        f.is_closed = (f.n_endpoints == 0 and f.n_holes >= 1)
        
        # 特殊模式
        if f.n_holes >= 1 and f.n_endpoints >= 1:
            if f.ep_top:
                f.has_tail_top = True
            if f.ep_bottom:
                f.has_tail_bottom = True
        
        return f
    
    def _has_horizontal_in_region(self, image: np.ndarray, y1: int, y2: int) -> bool:
        """检查区域是否有水平边"""
        region = image[y1:y2, :]
        if region.size == 0 or np.mean(region > 0.3) < 0.05:
            return False
        
        gy = np.abs(region[1:, :] - region[:-1, :]).sum()
        gx = np.abs(region[:, 1:] - region[:, :-1]).sum()
        
        return gy > gx * 1.3
    
    def _has_vertical_in_region(self, image: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> bool:
        """检查区域是否有垂直边"""
        region = image[y1:y2, x1:x2]
        if region.size == 0 or np.mean(region > 0.3) < 0.05:
            return False
        
        gy = np.abs(region[1:, :] - region[:-1, :]).sum()
        gx = np.abs(region[:, 1:] - region[:, :-1]).sum()
        
        return gx > gy * 1.3
    
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
                        if dot > 0.3:  # 尖角
                            corners.append((y, x))
        
        return corners
    
    def _get_endpoint_direction(self, skeleton: np.ndarray, y: int, x: int) -> str:
        """获取端点方向"""
        h, w = skeleton.shape
        
        for dy, dx, direction in [(-1, 0, 'down'), (1, 0, 'up'),
                                   (0, -1, 'right'), (0, 1, 'left'),
                                   (-1, -1, 'down'), (-1, 1, 'down'),
                                   (1, -1, 'up'), (1, 1, 'up')]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                return direction
        return ''
    
    def _detect_openings(self, image: np.ndarray) -> Set[str]:
        """检测开口"""
        openings = set()
        h, w = image.shape
        binary = (image > 0.3).astype(np.uint8)
        
        left_half = binary[:, :w//2]
        right_half = binary[:, w//2:]
        
        if np.sum(right_half) > np.sum(left_half) * 2:
            openings.add('left')
        if np.sum(left_half) > np.sum(right_half) * 2:
            openings.add('right')
        
        return openings
    
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


def run_experiment(n_train_per_class=10, n_test=500, emergence_threshold=3, verbose=True):
    print("=" * 60)
    print("Structon Vision v7.7 - True Hierarchical Emergence")
    print("=" * 60)
    print("\n层次化LRM架构:")
    print("  Level 0: 固定原子特征")
    print("  Level 1 LRM: 原子 → 拓扑模式 (TOPO_)")
    print("  Level 2 LRM: 拓扑 + 位置 → 位置模式 (POS_)")
    print("  Level 3 LRM: 位置 + 边缘 → 结构模式 (STRUCT_)")
    print("  Level 4 LRM: 结构 + 细节 → 完整模式 (OBJ_)")
    print(f"\n涌现阈值: {emergence_threshold}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = HierarchicalStructonSystem(emergence_threshold=emergence_threshold)
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.print_emergence_stats()
        system.print_label_analysis()
    
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


def debug_emergence(n_samples=50):
    """调试涌现过程"""
    print("\n=== 调试: 层次涌现过程 ===")
    train_images, train_labels, _, _ = load_mnist()
    
    system = HierarchicalStructonSystem(emergence_threshold=3)
    
    print(f"训练 {n_samples} 样本/类...")
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_samples]
        for idx in indices:
            system.train(train_images[idx], str(digit))
    
    system.print_emergence_stats()
    system.print_label_analysis()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--emergence-threshold', type=int, default=3)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        debug_emergence(50)
    else:
        run_experiment(args.train_per_class, args.test, args.emergence_threshold)
