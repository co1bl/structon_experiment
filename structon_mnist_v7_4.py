"""
Structon Vision v7.4 - Deep Hierarchical Emergence
===================================================

v7.3的问题：
- 只有Level 1涌现
- 没有层级组合
- 高层抽象缺失

v7.4的改进：
- 多层LRM，每层都可以涌现
- Level 1模式作为Level 2的输入
- 递归组合，真正的层级结构

架构：
  Level 0: 固定原子特征 (EP, JC, HOLE, Edges)
  Level 1: 局部模式涌现 (端点组合, 边缘组合)
  Level 2: 结构模式涌现 (环+尾巴, 交叉结构)
  Level 3: 对象模式涌现 (完整数字结构)

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
# 1. Core Types
# =============================================================================

@dataclass(frozen=True)
class AtomicFeature:
    """Level 0: 原子特征"""
    feature_type: str
    position: str
    
    def __str__(self):
        return f"{self.feature_type}@{self.position}"


@dataclass(frozen=True)
class PatternSignature:
    """
    模式签名 - 用于任意层级
    
    可以包含原子特征或低层模式ID
    """
    level: int
    components: FrozenSet[str]  # 组件字符串集合
    
    # 关键统计特征
    n_holes: int = 0
    n_endpoints: int = 0
    n_junctions: int = 0
    
    # 位置特征
    has_top: bool = False
    has_bottom: bool = False
    has_center: bool = False
    
    def __hash__(self):
        return hash((self.level, self.components, self.n_holes, 
                    self.n_endpoints, self.n_junctions,
                    self.has_top, self.has_bottom, self.has_center))
    
    def resonance(self, other: 'PatternSignature') -> float:
        """计算共振度"""
        if self.level != other.level:
            return 0.0
        
        # 拓扑特征匹配
        if self.n_holes != other.n_holes:
            topo_sim = 0.3  # 洞数不同，大幅降分
        else:
            topo_sim = 1.0
        
        ep_diff = abs(self.n_endpoints - other.n_endpoints)
        ep_sim = 1.0 / (1.0 + ep_diff * 0.5)
        
        jc_diff = abs(self.n_junctions - other.n_junctions)
        jc_sim = 1.0 / (1.0 + jc_diff * 0.5)
        
        # 位置特征匹配
        pos_match = 0
        pos_total = 3
        if self.has_top == other.has_top:
            pos_match += 1
        if self.has_bottom == other.has_bottom:
            pos_match += 1
        if self.has_center == other.has_center:
            pos_match += 1
        pos_sim = pos_match / pos_total
        
        # 组件匹配 (Jaccard)
        if self.components and other.components:
            inter = len(self.components & other.components)
            union = len(self.components | other.components)
            comp_sim = inter / union if union > 0 else 0
        else:
            comp_sim = 0.5
        
        # 加权组合 - 拓扑最重要
        return (0.30 * topo_sim + 
                0.15 * ep_sim + 
                0.15 * jc_sim + 
                0.20 * pos_sim +
                0.20 * comp_sim)


@dataclass
class EmergedPattern:
    """涌现的模式"""
    pattern_id: str
    level: int
    signature: PatternSignature
    frequency: int = 1
    
    def __hash__(self):
        return hash(self.pattern_id)
    
    def __str__(self):
        return self.pattern_id


# =============================================================================
# 2. Layer-wise Resonant Memory
# =============================================================================

class LayerMemory:
    """单层的共振记忆"""
    
    def __init__(self, level: int, emergence_threshold: int = 2, 
                 resonance_threshold: float = 0.80):
        self.level = level
        self.emergence_threshold = emergence_threshold
        self.resonance_threshold = resonance_threshold
        
        self.patterns: List[EmergedPattern] = []
        self.observation_count: Dict[PatternSignature, int] = defaultdict(int)
        self.pattern_counter = 0
    
    def observe(self, signature: PatternSignature) -> Optional[EmergedPattern]:
        """观察一个签名，返回匹配/涌现的模式"""
        
        # 1. 检查是否与已有模式共振
        best_match = None
        best_res = 0.0
        
        for pattern in self.patterns:
            res = signature.resonance(pattern.signature)
            if res > best_res:
                best_res = res
                best_match = pattern
        
        if best_res >= self.resonance_threshold:
            best_match.frequency += 1
            return best_match
        
        # 2. 没有共振，记录观察
        self.observation_count[signature] += 1
        count = self.observation_count[signature]
        
        # 3. 检查是否应该涌现
        if count >= self.emergence_threshold:
            pattern = self._create_pattern(signature, count)
            self.patterns.append(pattern)
            return pattern
        
        return None
    
    def _create_pattern(self, sig: PatternSignature, freq: int) -> EmergedPattern:
        """创建新模式"""
        self.pattern_counter += 1
        
        # 基于特征生成ID
        parts = [f"L{self.level}"]
        if sig.n_holes > 0:
            parts.append(f"H{sig.n_holes}")
        if sig.n_endpoints > 0:
            parts.append(f"E{sig.n_endpoints}")
        if sig.n_junctions > 0:
            parts.append(f"J{sig.n_junctions}")
        
        # 位置标记
        pos_parts = []
        if sig.has_top:
            pos_parts.append("T")
        if sig.has_center:
            pos_parts.append("C")
        if sig.has_bottom:
            pos_parts.append("B")
        if pos_parts:
            parts.append("".join(pos_parts))
        
        parts.append(str(self.pattern_counter))
        
        return EmergedPattern(
            pattern_id="_".join(parts),
            level=self.level,
            signature=sig,
            frequency=freq
        )
    
    def get_stats(self) -> List[dict]:
        """获取统计"""
        return [
            {
                'id': p.pattern_id,
                'freq': p.frequency,
                'holes': p.signature.n_holes,
                'endpoints': p.signature.n_endpoints,
                'junctions': p.signature.n_junctions,
                'positions': f"{'T' if p.signature.has_top else ''}"
                            f"{'C' if p.signature.has_center else ''}"
                            f"{'B' if p.signature.has_bottom else ''}"
            }
            for p in sorted(self.patterns, key=lambda x: -x.frequency)
        ]


# =============================================================================
# 3. Multi-Level Resonant Memory System
# =============================================================================

class HierarchicalMemory:
    """
    层次化共振记忆
    
    多层LRM，每层的输出作为下一层的输入
    """
    
    def __init__(self, n_levels: int = 3, emergence_threshold: int = 2):
        self.n_levels = n_levels
        self.layers = [
            LayerMemory(level=i+1, emergence_threshold=emergence_threshold)
            for i in range(n_levels)
        ]
    
    def process(self, atomic_features: List[AtomicFeature]) -> List[Optional[EmergedPattern]]:
        """
        处理原子特征，通过多层LRM
        
        Returns:
            每层涌现的模式列表
        """
        emerged = []
        
        # 从原子特征提取统计
        n_holes = sum(1 for f in atomic_features if f.feature_type == 'HOLE')
        n_endpoints = sum(1 for f in atomic_features if f.feature_type == 'EP')
        n_junctions = sum(1 for f in atomic_features if f.feature_type == 'JC')
        
        positions = set(f.position for f in atomic_features 
                       if f.feature_type in ('EP', 'JC'))
        has_top = any(p in ('TL', 'T', 'TR') for p in positions)
        has_bottom = any(p in ('BL', 'B', 'BR') for p in positions)
        has_center = 'C' in positions
        
        # Level 1: 原子 → 基础模式
        current_components = frozenset(str(f) for f in atomic_features)
        
        sig1 = PatternSignature(
            level=1,
            components=current_components,
            n_holes=n_holes,
            n_endpoints=n_endpoints,
            n_junctions=n_junctions,
            has_top=has_top,
            has_bottom=has_bottom,
            has_center=has_center
        )
        
        pattern1 = self.layers[0].observe(sig1)
        emerged.append(pattern1)
        
        # Level 2: 组合更高层特征
        if pattern1:
            # 包含Level 1模式ID + 位置信息
            level2_components = frozenset([
                pattern1.pattern_id,
                f"top={has_top}",
                f"center={has_center}",
                f"bottom={has_bottom}"
            ])
            
            sig2 = PatternSignature(
                level=2,
                components=level2_components,
                n_holes=n_holes,
                n_endpoints=n_endpoints,
                n_junctions=n_junctions,
                has_top=has_top,
                has_bottom=has_bottom,
                has_center=has_center
            )
            
            pattern2 = self.layers[1].observe(sig2)
            emerged.append(pattern2)
            
            # Level 3: 最高层抽象
            if pattern2 and self.n_levels >= 3:
                level3_components = frozenset([
                    pattern1.pattern_id,
                    pattern2.pattern_id,
                    f"H{n_holes}",
                    f"E{n_endpoints}",
                    f"J{n_junctions}"
                ])
                
                sig3 = PatternSignature(
                    level=3,
                    components=level3_components,
                    n_holes=n_holes,
                    n_endpoints=n_endpoints,
                    n_junctions=n_junctions,
                    has_top=has_top,
                    has_bottom=has_bottom,
                    has_center=has_center
                )
                
                pattern3 = self.layers[2].observe(sig3)
                emerged.append(pattern3)
            else:
                if self.n_levels >= 3:
                    emerged.append(None)
        else:
            emerged.extend([None] * (self.n_levels - 1))
        
        return emerged
    
    def print_all_patterns(self):
        """打印所有层的涌现模式"""
        print("\n=== 层次化涌现模式 ===")
        for i, layer in enumerate(self.layers):
            stats = layer.get_stats()
            print(f"\nLevel {i+1}: {len(stats)} 个模式")
            for s in stats[:8]:  # 只显示前8个
                print(f"  {s['id']}: freq={s['freq']}, "
                      f"H{s['holes']}E{s['endpoints']}J{s['junctions']} "
                      f"pos={s['positions']}")


# =============================================================================
# 4. Feature Detector
# =============================================================================

class FeatureDetector:
    """Level 0: 固定特征检测"""
    
    def __init__(self):
        self.regions = ['TL', 'T', 'TR', 'L', 'C', 'R', 'BL', 'B', 'BR']
    
    def detect(self, image: np.ndarray) -> List[AtomicFeature]:
        """检测原子特征"""
        h, w = image.shape
        features = []
        
        # 全局拓扑
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(image)
        
        # 添加HOLE
        for _ in range(n_holes):
            features.append(AtomicFeature('HOLE', 'GLOBAL'))
        
        # 区域分析
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
            
            # 端点
            for y, x in endpoints:
                if y1 <= y < y2 and x1 <= x < x2:
                    features.append(AtomicFeature('EP', region_name))
            
            # 交叉点
            for y, x in junctions:
                if y1 <= y < y2 and x1 <= x < x2:
                    features.append(AtomicFeature('JC', region_name))
        
        return features
    
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
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """Structon视觉系统 v7.4"""
    
    def __init__(self, n_levels: int = 3, emergence_threshold: int = 2):
        self.detector = FeatureDetector()
        self.memory = HierarchicalMemory(n_levels, emergence_threshold)
        
        # 标签记忆: label → [(features, [patterns])]
        self.label_memory: Dict[str, List[Tuple[List[AtomicFeature], List[Optional[EmergedPattern]]]]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> Tuple[List[AtomicFeature], List[Optional[EmergedPattern]]]:
        """处理图像"""
        features = self.detector.detect(image)
        patterns = self.memory.process(features)
        return features, patterns
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        features, patterns = self.process(image)
        self.label_memory[label].append((features, patterns))
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        features, patterns = self.process(image)
        
        # 提取当前图像的核心特征
        n_holes = sum(1 for f in features if f.feature_type == 'HOLE')
        n_endpoints = sum(1 for f in features if f.feature_type == 'EP')
        n_junctions = sum(1 for f in features if f.feature_type == 'JC')
        
        positions = set(f.position for f in features if f.feature_type in ('EP', 'JC'))
        has_top = any(p in ('TL', 'T', 'TR') for p in positions)
        has_bottom = any(p in ('BL', 'B', 'BR') for p in positions)
        has_center = 'C' in positions
        
        best_label = None
        best_score = 0.0
        
        for label, stored_items in self.label_memory.items():
            for stored_features, stored_patterns in stored_items:
                # 提取存储的核心特征
                s_holes = sum(1 for f in stored_features if f.feature_type == 'HOLE')
                s_endpoints = sum(1 for f in stored_features if f.feature_type == 'EP')
                s_junctions = sum(1 for f in stored_features if f.feature_type == 'JC')
                
                s_positions = set(f.position for f in stored_features if f.feature_type in ('EP', 'JC'))
                s_has_top = any(p in ('TL', 'T', 'TR') for p in s_positions)
                s_has_bottom = any(p in ('BL', 'B', 'BR') for p in s_positions)
                s_has_center = 'C' in s_positions
                
                # 计算相似度
                # 拓扑匹配
                if n_holes != s_holes:
                    topo_score = 0.3
                else:
                    topo_score = 1.0
                
                ep_score = 1.0 / (1.0 + abs(n_endpoints - s_endpoints) * 0.3)
                jc_score = 1.0 / (1.0 + abs(n_junctions - s_junctions) * 0.3)
                
                # 位置匹配
                pos_match = 0
                if has_top == s_has_top:
                    pos_match += 1
                if has_bottom == s_has_bottom:
                    pos_match += 1
                if has_center == s_has_center:
                    pos_match += 1
                pos_score = pos_match / 3
                
                # 模式匹配 (高层更重要)
                pattern_score = 0.0
                pattern_count = 0
                for i, (p1, p2) in enumerate(zip(patterns, stored_patterns)):
                    weight = (i + 1) / len(patterns)  # 高层权重更高
                    if p1 and p2 and p1.pattern_id == p2.pattern_id:
                        pattern_score += weight
                    pattern_count += weight
                
                if pattern_count > 0:
                    pattern_score /= pattern_count
                else:
                    pattern_score = 0.5
                
                # 总分
                score = (0.25 * topo_score + 
                        0.15 * ep_score + 
                        0.10 * jc_score +
                        0.20 * pos_score + 
                        0.30 * pattern_score)
                
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试"""
        features, patterns = self.process(image)
        
        return {
            'n_features': len(features),
            'features': [str(f) for f in features[:12]],
            'patterns': [p.pattern_id if p else None for p in patterns]
        }
    
    def print_analysis(self):
        """打印分析"""
        self.memory.print_all_patterns()
        
        print("\n=== 各数字的特征分布 ===")
        for label in sorted(self.label_memory.keys()):
            items = self.label_memory[label]
            
            # 统计
            holes = []
            endpoints = []
            junctions = []
            pattern_counts = [Counter() for _ in range(3)]
            
            for features, patterns in items:
                holes.append(sum(1 for f in features if f.feature_type == 'HOLE'))
                endpoints.append(sum(1 for f in features if f.feature_type == 'EP'))
                junctions.append(sum(1 for f in features if f.feature_type == 'JC'))
                
                for i, p in enumerate(patterns):
                    if i < 3:
                        pattern_counts[i][p.pattern_id if p else 'None'] += 1
            
            print(f"\n数字 {label}:")
            print(f"  空洞: {Counter(holes)}")
            print(f"  端点: {Counter(endpoints)}")
            print(f"  交叉: {Counter(junctions)}")
            for i, pc in enumerate(pattern_counts):
                if pc:
                    top_patterns = pc.most_common(3)
                    print(f"  L{i+1}模式: {top_patterns}")


# =============================================================================
# 6. MNIST Experiment
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
    print("Structon Vision v7.4 - Deep Hierarchical Emergence")
    print("=" * 60)
    print("\n架构:")
    print("  Level 0: 固定原子特征")
    print("  Level 1: 基础模式涌现")
    print("  Level 2: 结构模式涌现")
    print("  Level 3: 对象模式涌现")
    print("\n每层的输出作为下一层的输入")
    
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
        system.print_analysis()
    
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
    print("\n=== 调试: 深层涌现 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = StructonVisionSystem(n_levels=3, emergence_threshold=2)
    
    # 训练
    print("训练中...")
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:5]
        for idx in indices:
            system.train(train_images[idx], str(digit))
    
    system.memory.print_all_patterns()
    
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
            print(f"    特征: {info['features'][:8]}...")
            print(f"    涌现: L1={info['patterns'][0]}, L2={info['patterns'][1]}, L3={info['patterns'][2] if len(info['patterns']) > 2 else None}")


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
