"""
Structon Vision v7.5 - Independent Multi-Level Emergence
=========================================================

v7.4的问题：
- 级联依赖：Level N+1 只有在 Level N 涌现后才处理
- 导致大量样本在所有层都是 None
- 准确率反而下降

v7.5的修复：
- 每层独立处理，不依赖下层涌现
- 每层观察数据的不同方面：
  - Level 1: 拓扑特征 (holes, endpoints, junctions 数量)
  - Level 2: 位置特征 (端点/交叉在哪里)
  - Level 3: 形状特征 (边缘方向分布)
- 最终匹配综合所有层的模式

核心原则：
- 独立涌现：每层都能独立发现模式
- 多视角：不同层看数据的不同方面
- 组合匹配：预测时综合所有层

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
# 1. Signatures for Different Levels
# =============================================================================

@dataclass(frozen=True)
class TopologySignature:
    """Level 1: 拓扑签名 - 全局结构"""
    n_holes: int
    n_endpoints: int
    n_junctions: int
    
    def __str__(self):
        return f"H{self.n_holes}E{self.n_endpoints}J{self.n_junctions}"
    
    def resonance(self, other: 'TopologySignature') -> float:
        if self.n_holes != other.n_holes:
            return 0.3  # 洞数不同，严重不匹配
        
        ep_sim = 1.0 / (1.0 + abs(self.n_endpoints - other.n_endpoints))
        jc_sim = 1.0 / (1.0 + abs(self.n_junctions - other.n_junctions))
        
        return 0.5 + 0.25 * ep_sim + 0.25 * jc_sim


@dataclass(frozen=True)
class PositionSignature:
    """Level 2: 位置签名 - 端点和交叉点在哪里"""
    endpoint_positions: FrozenSet[str]  # {'T', 'B', 'C', 'TL', ...}
    junction_positions: FrozenSet[str]
    
    # 简化的位置特征
    has_top_ep: bool
    has_bottom_ep: bool
    has_center_ep: bool
    has_center_jc: bool
    
    def __str__(self):
        parts = []
        if self.has_top_ep:
            parts.append("Etop")
        if self.has_bottom_ep:
            parts.append("Ebot")
        if self.has_center_ep:
            parts.append("Ectr")
        if self.has_center_jc:
            parts.append("Jctr")
        return "_".join(parts) if parts else "empty"
    
    def resonance(self, other: 'PositionSignature') -> float:
        score = 0.0
        total = 4
        
        if self.has_top_ep == other.has_top_ep:
            score += 1
        if self.has_bottom_ep == other.has_bottom_ep:
            score += 1
        if self.has_center_ep == other.has_center_ep:
            score += 1
        if self.has_center_jc == other.has_center_jc:
            score += 1
        
        return score / total


@dataclass(frozen=True)
class ShapeSignature:
    """Level 3: 形状签名 - 边缘方向分布"""
    dominant_direction: str  # 'H', 'V', 'D1', 'D2', 'mixed'
    has_horizontal_top: bool
    has_vertical_center: bool
    edge_balance: str  # 'horizontal', 'vertical', 'diagonal', 'balanced'
    
    def __str__(self):
        parts = [self.dominant_direction]
        if self.has_horizontal_top:
            parts.append("Htop")
        if self.has_vertical_center:
            parts.append("Vctr")
        parts.append(self.edge_balance[:3])
        return "_".join(parts)
    
    def resonance(self, other: 'ShapeSignature') -> float:
        score = 0.0
        
        if self.dominant_direction == other.dominant_direction:
            score += 0.3
        if self.has_horizontal_top == other.has_horizontal_top:
            score += 0.25
        if self.has_vertical_center == other.has_vertical_center:
            score += 0.2
        if self.edge_balance == other.edge_balance:
            score += 0.25
        
        return score


# =============================================================================
# 2. Emerged Patterns
# =============================================================================

@dataclass
class EmergedPattern:
    """涌现的模式"""
    pattern_id: str
    level: int
    frequency: int = 1
    
    def __hash__(self):
        return hash(self.pattern_id)


# =============================================================================
# 3. Layer Memory (Independent)
# =============================================================================

class LayerMemory:
    """单层共振记忆 - 独立运作"""
    
    def __init__(self, level: int, emergence_threshold: int = 2, 
                 resonance_threshold: float = 0.75):
        self.level = level
        self.emergence_threshold = emergence_threshold
        self.resonance_threshold = resonance_threshold
        
        self.patterns: Dict[str, EmergedPattern] = {}  # signature_str → pattern
        self.observation_count: Dict[str, int] = defaultdict(int)
        self.pattern_counter = 0
    
    def observe(self, signature, signature_str: str) -> Optional[EmergedPattern]:
        """观察签名，返回涌现的模式"""
        
        # 1. 精确匹配检查
        if signature_str in self.patterns:
            self.patterns[signature_str].frequency += 1
            return self.patterns[signature_str]
        
        # 2. 共振匹配检查
        for stored_str, pattern in self.patterns.items():
            # 简单字符串相似度作为共振
            if self._string_similarity(signature_str, stored_str) >= self.resonance_threshold:
                pattern.frequency += 1
                return pattern
        
        # 3. 记录新观察
        self.observation_count[signature_str] += 1
        count = self.observation_count[signature_str]
        
        # 4. 检查是否涌现
        if count >= self.emergence_threshold:
            self.pattern_counter += 1
            pattern_id = f"L{self.level}_{signature_str}_{self.pattern_counter}"
            pattern = EmergedPattern(pattern_id, self.level, count)
            self.patterns[signature_str] = pattern
            return pattern
        
        return None
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """简单的字符串相似度"""
        if s1 == s2:
            return 1.0
        
        # 基于共同部分
        parts1 = set(s1.split('_'))
        parts2 = set(s2.split('_'))
        
        if not parts1 or not parts2:
            return 0.0
        
        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_patterns(self) -> List[dict]:
        return [
            {'id': p.pattern_id, 'freq': p.frequency}
            for p in sorted(self.patterns.values(), key=lambda x: -x.frequency)
        ]


# =============================================================================
# 4. Feature Extractor
# =============================================================================

class FeatureExtractor:
    """特征提取器 - 生成各层签名"""
    
    def __init__(self):
        self.regions = ['TL', 'T', 'TR', 'L', 'C', 'R', 'BL', 'B', 'BR']
        self.top_regions = {'TL', 'T', 'TR'}
        self.bottom_regions = {'BL', 'B', 'BR'}
        self.center_region = {'C'}
    
    def extract(self, image: np.ndarray) -> Tuple[TopologySignature, PositionSignature, ShapeSignature]:
        """提取三层签名"""
        h, w = image.shape
        
        # 骨架化和拓扑分析
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(image)
        
        # === Level 1: 拓扑签名 ===
        topo_sig = TopologySignature(
            n_holes=n_holes,
            n_endpoints=len(endpoints),
            n_junctions=len(junctions)
        )
        
        # === Level 2: 位置签名 ===
        ep_positions = self._classify_positions(endpoints, h, w)
        jc_positions = self._classify_positions(junctions, h, w)
        
        has_top_ep = bool(ep_positions & self.top_regions)
        has_bottom_ep = bool(ep_positions & self.bottom_regions)
        has_center_ep = bool(ep_positions & self.center_region)
        has_center_jc = bool(jc_positions & self.center_region)
        
        pos_sig = PositionSignature(
            endpoint_positions=frozenset(ep_positions),
            junction_positions=frozenset(jc_positions),
            has_top_ep=has_top_ep,
            has_bottom_ep=has_bottom_ep,
            has_center_ep=has_center_ep,
            has_center_jc=has_center_jc
        )
        
        # === Level 3: 形状签名 ===
        edge_counts = self._count_edges_by_direction(image)
        
        # 主方向
        directions = ['H', 'V', 'D1', 'D2']
        max_idx = np.argmax(edge_counts)
        if edge_counts[max_idx] > sum(edge_counts) * 0.4:
            dominant = directions[max_idx]
        else:
            dominant = 'mixed'
        
        # 边缘平衡
        h_count = edge_counts[0]
        v_count = edge_counts[1]
        d_count = edge_counts[2] + edge_counts[3]
        
        if h_count > v_count * 1.5 and h_count > d_count:
            balance = 'horizontal'
        elif v_count > h_count * 1.5 and v_count > d_count:
            balance = 'vertical'
        elif d_count > h_count and d_count > v_count:
            balance = 'diagonal'
        else:
            balance = 'balanced'
        
        # 顶部是否有水平边
        has_h_top = self._has_direction_in_region(image, 'H', 'top')
        has_v_center = self._has_direction_in_region(image, 'V', 'center')
        
        shape_sig = ShapeSignature(
            dominant_direction=dominant,
            has_horizontal_top=has_h_top,
            has_vertical_center=has_v_center,
            edge_balance=balance
        )
        
        return topo_sig, pos_sig, shape_sig
    
    def _classify_positions(self, points: List[Tuple], h: int, w: int) -> Set[str]:
        """将点分类到区域"""
        positions = set()
        
        for y, x in points:
            row = 0 if y < h/3 else (1 if y < 2*h/3 else 2)
            col = 0 if x < w/3 else (1 if x < 2*w/3 else 2)
            
            region_names = [
                ['TL', 'T', 'TR'],
                ['L', 'C', 'R'],
                ['BL', 'B', 'BR']
            ]
            positions.add(region_names[row][col])
        
        return positions
    
    def _count_edges_by_direction(self, image: np.ndarray) -> List[int]:
        """统计各方向边缘数量"""
        h, w = image.shape
        counts = [0, 0, 0, 0]  # H, V, D1, D2
        
        for idx, region_name in enumerate(self.regions):
            row, col = idx // 3, idx % 3
            y1, y2 = row * h // 3, (row + 1) * h // 3
            x1, x2 = col * w // 3, (col + 1) * w // 3
            
            region = image[y1:y2, x1:x2]
            if np.mean(region > 0.3) < 0.05:
                continue
            
            direction = self._detect_direction(region)
            if direction == 'E_H':
                counts[0] += 1
            elif direction == 'E_V':
                counts[1] += 1
            elif direction == 'E_D1':
                counts[2] += 1
            elif direction == 'E_D2':
                counts[3] += 1
        
        return counts
    
    def _has_direction_in_region(self, image: np.ndarray, direction: str, region: str) -> bool:
        """检查特定区域是否有特定方向"""
        h, w = image.shape
        
        if region == 'top':
            area = image[0:h//3, :]
        elif region == 'center':
            area = image[h//3:2*h//3, w//3:2*w//3]
        elif region == 'bottom':
            area = image[2*h//3:, :]
        else:
            return False
        
        if np.mean(area > 0.3) < 0.05:
            return False
        
        detected = self._detect_direction(area)
        
        if direction == 'H':
            return detected == 'E_H'
        elif direction == 'V':
            return detected == 'E_V'
        
        return False
    
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
    """Structon视觉系统 v7.5 - 独立多层涌现"""
    
    def __init__(self, emergence_threshold: int = 2):
        self.extractor = FeatureExtractor()
        
        # 三层独立记忆
        self.topo_memory = LayerMemory(1, emergence_threshold, 0.9)   # 拓扑要求精确
        self.pos_memory = LayerMemory(2, emergence_threshold, 0.75)   # 位置可以宽松
        self.shape_memory = LayerMemory(3, emergence_threshold, 0.7)  # 形状最宽松
        
        # 标签记忆
        self.label_memory: Dict[str, List[Tuple]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> Tuple[
        TopologySignature, PositionSignature, ShapeSignature,
        Optional[EmergedPattern], Optional[EmergedPattern], Optional[EmergedPattern]
    ]:
        """处理图像 - 三层独立"""
        topo_sig, pos_sig, shape_sig = self.extractor.extract(image)
        
        # 每层独立涌现
        topo_pattern = self.topo_memory.observe(topo_sig, str(topo_sig))
        pos_pattern = self.pos_memory.observe(pos_sig, str(pos_sig))
        shape_pattern = self.shape_memory.observe(shape_sig, str(shape_sig))
        
        return topo_sig, pos_sig, shape_sig, topo_pattern, pos_pattern, shape_pattern
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        result = self.process(image)
        self.label_memory[label].append(result)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        topo_sig, pos_sig, shape_sig, topo_p, pos_p, shape_p = self.process(image)
        
        best_label = None
        best_score = 0.0
        
        for label, stored_items in self.label_memory.items():
            for stored in stored_items:
                s_topo, s_pos, s_shape, s_topo_p, s_pos_p, s_shape_p = stored
                
                # 三层相似度
                topo_score = topo_sig.resonance(s_topo)
                pos_score = pos_sig.resonance(s_pos)
                shape_score = shape_sig.resonance(s_shape)
                
                # 模式匹配加分
                pattern_bonus = 0.0
                if topo_p and s_topo_p and topo_p.pattern_id == s_topo_p.pattern_id:
                    pattern_bonus += 0.15
                if pos_p and s_pos_p and pos_p.pattern_id == s_pos_p.pattern_id:
                    pattern_bonus += 0.10
                if shape_p and s_shape_p and shape_p.pattern_id == s_shape_p.pattern_id:
                    pattern_bonus += 0.05
                
                # 加权总分 - 拓扑最重要
                score = (0.45 * topo_score + 
                        0.30 * pos_score + 
                        0.15 * shape_score +
                        pattern_bonus)
                
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试"""
        topo_sig, pos_sig, shape_sig, topo_p, pos_p, shape_p = self.process(image)
        
        return {
            'L1_topo': str(topo_sig),
            'L2_pos': str(pos_sig),
            'L3_shape': str(shape_sig),
            'patterns': {
                'L1': topo_p.pattern_id if topo_p else None,
                'L2': pos_p.pattern_id if pos_p else None,
                'L3': shape_p.pattern_id if shape_p else None
            }
        }
    
    def print_analysis(self):
        """打印分析"""
        print("\n=== 层次化涌现模式 ===")
        
        print(f"\nLevel 1 (拓扑): {len(self.topo_memory.patterns)} 个模式")
        for p in self.topo_memory.get_patterns()[:10]:
            print(f"  {p['id']}: freq={p['freq']}")
        
        print(f"\nLevel 2 (位置): {len(self.pos_memory.patterns)} 个模式")
        for p in self.pos_memory.get_patterns()[:10]:
            print(f"  {p['id']}: freq={p['freq']}")
        
        print(f"\nLevel 3 (形状): {len(self.shape_memory.patterns)} 个模式")
        for p in self.shape_memory.get_patterns()[:10]:
            print(f"  {p['id']}: freq={p['freq']}")
        
        print("\n=== 各数字的特征分布 ===")
        for label in sorted(self.label_memory.keys()):
            items = self.label_memory[label]
            
            topo_patterns = Counter()
            pos_patterns = Counter()
            shape_patterns = Counter()
            
            for item in items:
                topo_patterns[str(item[0])] += 1
                pos_patterns[str(item[1])] += 1
                shape_patterns[str(item[2])] += 1
            
            print(f"\n数字 {label}:")
            print(f"  拓扑: {topo_patterns.most_common(3)}")
            print(f"  位置: {pos_patterns.most_common(3)}")
            print(f"  形状: {shape_patterns.most_common(2)}")


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
    print("Structon Vision v7.5 - Independent Multi-Level Emergence")
    print("=" * 60)
    print("\n三层独立涌现:")
    print("  Level 1: 拓扑 (holes, endpoints, junctions)")
    print("  Level 2: 位置 (端点/交叉在哪里)")
    print("  Level 3: 形状 (边缘方向分布)")
    print("\n每层独立处理，不依赖其他层")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = StructonVisionSystem(emergence_threshold=2)
    
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


def debug_digits(n=3):
    print("\n=== 调试: 独立多层涌现 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = StructonVisionSystem(emergence_threshold=2)
    
    # 训练
    print("训练中...")
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:5]
        for idx in indices:
            system.train(train_images[idx], str(digit))
    
    system.print_analysis()
    
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
            print(f"    L1 拓扑: {info['L1_topo']} → {info['patterns']['L1']}")
            print(f"    L2 位置: {info['L2_pos']} → {info['patterns']['L2']}")
            print(f"    L3 形状: {info['L3_shape']} → {info['patterns']['L3']}")


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
