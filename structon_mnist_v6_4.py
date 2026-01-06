"""
Structon Vision v6.4 - Fixed Atom Detection
=============================================

v6.3的问题：
- Loop检测器太激进，把所有数字都识别成一个Loop
- 原因：std_dist < mean_dist * 0.5 这个条件太宽松
- 所有数字的激活点都大致围绕图像中心分布，都被误判为Loop

v6.4的修复：
1. 更严格的Loop检测：
   - 需要真正的封闭轮廓，不只是围绕中心分布
   - 检查激活点是否形成环形
   
2. 改变检测顺序：
   - 先检测直线（最简单）
   - 再检测曲线（基于方向变化）
   - 最后检测Loop（需要闭合）

3. 使用连通分量分析：
   - 先把激活点分成连通的组
   - 对每个组单独分析

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, FrozenSet, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Atom Types
# =============================================================================

class AtomType(Enum):
    LINE_H = "line_h"
    LINE_V = "line_v"
    LINE_D1 = "line_d1"
    LINE_D2 = "line_d2"
    CURVE_C = "curve_c"
    CURVE_D = "curve_d"
    CURVE_U = "curve_u"
    CURVE_N = "curve_n"
    LOOP = "loop"
    STROKE = "stroke"  # 通用笔画（无法确定类型时）
    UNKNOWN = "unknown"


class RelationType(Enum):
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    INSIDE = "inside"
    CONTAINS = "contains"
    CONNECTED = "connected"
    CROSSING = "crossing"
    ADJACENT = "adjacent"


# =============================================================================
# 2. Data Structures
# =============================================================================

@dataclass
class Activation:
    x: float
    y: float
    strength: float
    orientation: int
    feature_response: np.ndarray


@dataclass
class Anchors:
    center: Tuple[float, float]
    head: Tuple[float, float]
    tail: Tuple[float, float]


@dataclass
class AtomInstance:
    atom_type: AtomType
    anchors: Anchors
    activations: List[Activation]
    confidence: float
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.anchors.center
    
    @property 
    def bbox(self) -> Tuple[float, float, float, float]:
        if not self.activations:
            return (self.anchors.center[0], self.anchors.center[1],
                    self.anchors.center[0], self.anchors.center[1])
        xs = [a.x for a in self.activations]
        ys = [a.y for a in self.activations]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class AtomRelation:
    atom1_type: AtomType
    atom2_type: AtomType
    relation: RelationType
    
    def to_string(self) -> str:
        return f"{self.atom1_type.value}_{self.relation.value}_{self.atom2_type.value}"


@dataclass(frozen=True)
class StructuralSignature:
    atom_types: FrozenSet[AtomType]
    relations: FrozenSet[str]
    atom_count: int  # 原子数量也是签名的一部分
    
    def __hash__(self):
        return hash((self.atom_types, self.relations, self.atom_count))
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 原子类型匹配
        type_intersection = len(self.atom_types & other.atom_types)
        type_union = len(self.atom_types | other.atom_types)
        type_sim = type_intersection / type_union if type_union > 0 else 0
        
        # 关系匹配
        rel_intersection = len(self.relations & other.relations)
        rel_union = len(self.relations | other.relations)
        rel_sim = rel_intersection / rel_union if rel_union > 0 else 0
        
        # 原子数量匹配
        count_diff = abs(self.atom_count - other.atom_count)
        count_sim = 1.0 / (1.0 + count_diff)
        
        return 0.3 * type_sim + 0.5 * rel_sim + 0.2 * count_sim


# =============================================================================
# 3. V1 Layer
# =============================================================================

class V1GaborBank:
    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size
        self.orientations = [0, 22, 45, 67, 90, 112, 135, 157]  # 8方向
        self.scales = [1.0, 1.5]
        self.filters = self._create_gabor_bank()
        self.n_features = len(self.filters)
        self.n_orientations = len(self.orientations)
    
    def _create_gabor_kernel(self, theta, sigma, lambd, gamma=0.5):
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
        filters = []
        for scale in self.scales:
            for theta_deg in self.orientations:
                theta = np.deg2rad(theta_deg)
                kernel = self._create_gabor_kernel(theta, 2.0*scale, 4.0*scale)
                filters.append(kernel)
        return filters
    
    def _convolve(self, image, kernel):
        kh, kw = kernel.shape
        ih, iw = image.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros((ih, iw), dtype=np.float32)
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return output
    
    def extract_activations(self, image: np.ndarray, threshold: float = 0.3, step: int = 2) -> List[Activation]:
        response_maps = [np.abs(self._convolve(image, kernel)) for kernel in self.filters]
        h, w = image.shape
        
        global_max = max(r.max() for r in response_maps)
        if global_max < 1e-8:
            return []
        
        norm_maps = [r / global_max for r in response_maps]
        activations = []
        
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                feature_vec = np.array([norm_maps[i][y, x] for i in range(self.n_features)])
                max_response = feature_vec.max()
                
                if max_response < threshold:
                    continue
                
                best_idx = np.argmax(feature_vec[:self.n_orientations])  # 只看第一个scale
                dominant_orient = self.orientations[best_idx]
                
                activations.append(Activation(
                    x=float(x),
                    y=float(y),
                    strength=max_response,
                    orientation=dominant_orient,
                    feature_response=feature_vec
                ))
        
        return activations


# =============================================================================
# 4. Atom Detector (完全重写)
# =============================================================================

class AtomDetector:
    """
    重写的原子检测器
    
    策略：
    1. 先找连通分量（笔画段）
    2. 对每个连通分量分析其形状
    3. 根据形状特征分类
    """
    
    def __init__(self, image_size: float = 28.0):
        self.image_size = image_size
        self.connection_radius = 4.0  # 连通性半径
    
    def detect_atoms(self, activations: List[Activation]) -> List[AtomInstance]:
        if len(activations) < 3:
            return []
        
        # 1. 找连通分量
        components = self._find_connected_components(activations)
        
        atoms = []
        for comp_indices in components:
            comp_acts = [activations[i] for i in comp_indices]
            
            if len(comp_acts) < 3:
                continue
            
            # 2. 分析每个分量的形状
            atom = self._analyze_component(comp_acts)
            if atom:
                atoms.append(atom)
        
        # 3. 检查是否有Loop（需要检查整体闭合性）
        if len(atoms) >= 2:
            loop = self._check_for_loop(atoms, activations)
            if loop:
                # 如果检测到闭合圈，替换掉组成它的原子
                atoms = [a for a in atoms if a.atom_type not in 
                        [AtomType.CURVE_C, AtomType.CURVE_D, AtomType.CURVE_U, AtomType.CURVE_N]]
                atoms.append(loop)
        
        return atoms
    
    def _find_connected_components(self, activations: List[Activation]) -> List[List[int]]:
        """找连通分量"""
        n = len(activations)
        visited = [False] * n
        components = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            component = []
            queue = [i]
            visited[i] = True
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                ca = activations[current]
                
                for j in range(n):
                    if visited[j]:
                        continue
                    
                    ja = activations[j]
                    dist = np.sqrt((ca.x - ja.x)**2 + (ca.y - ja.y)**2)
                    
                    if dist <= self.connection_radius:
                        visited[j] = True
                        queue.append(j)
            
            components.append(component)
        
        return components
    
    def _analyze_component(self, activations: List[Activation]) -> Optional[AtomInstance]:
        """分析一个连通分量的形状"""
        
        # 计算基本几何特征
        xs = [a.x for a in activations]
        ys = [a.y for a in activations]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # 计算方向变化
        sorted_acts = self._sort_by_path(activations)
        orientations = [a.orientation for a in sorted_acts]
        
        # 计算总方向变化
        total_rotation = 0
        for i in range(1, len(orientations)):
            diff = orientations[i] - orientations[i-1]
            if diff > 90:
                diff -= 180
            elif diff < -90:
                diff += 180
            total_rotation += diff
        
        # 计算方向一致性（标准差）
        orient_std = np.std(orientations) if len(orientations) > 1 else 0
        
        # 计算aspect ratio
        aspect = width / (height + 1e-8)
        
        # 分类
        atom_type = AtomType.STROKE
        
        # 直线检测：方向变化小，方向一致性高
        if abs(total_rotation) < 30 and orient_std < 30:
            # 判断是哪种直线
            dominant_orient = np.median(orientations)
            
            if dominant_orient < 22 or dominant_orient >= 157:
                atom_type = AtomType.LINE_V
            elif 67 <= dominant_orient < 112:
                atom_type = AtomType.LINE_H
            elif 22 <= dominant_orient < 67:
                atom_type = AtomType.LINE_D1
            else:  # 112-157
                atom_type = AtomType.LINE_D2
        
        # 曲线检测：有明显的方向变化
        elif abs(total_rotation) > 40:
            # 根据形状和开口方向判断曲线类型
            if aspect > 1.2:  # 横向延展
                if cy < (min_y + max_y) / 2 + 2:
                    atom_type = AtomType.CURVE_N  # 拱形
                else:
                    atom_type = AtomType.CURVE_U  # U形
            elif aspect < 0.8:  # 纵向延展
                if cx < (min_x + max_x) / 2:
                    atom_type = AtomType.CURVE_D  # 右开口
                else:
                    atom_type = AtomType.CURVE_C  # 左开口
            else:
                # 正方形区域内的曲线，看旋转方向
                if total_rotation > 0:
                    atom_type = AtomType.CURVE_C
                else:
                    atom_type = AtomType.CURVE_D
        
        # 构建锚点
        anchors = self._compute_anchors(activations, atom_type)
        
        return AtomInstance(
            atom_type=atom_type,
            anchors=anchors,
            activations=activations,
            confidence=0.7
        )
    
    def _check_for_loop(self, atoms: List[AtomInstance], 
                        all_activations: List[Activation]) -> Optional[AtomInstance]:
        """
        检查是否形成闭合圈
        
        条件：
        1. 有多个曲线原子
        2. 这些曲线首尾相连
        3. 形成一个封闭区域
        """
        curves = [a for a in atoms if a.atom_type in 
                 [AtomType.CURVE_C, AtomType.CURVE_D, AtomType.CURVE_U, AtomType.CURVE_N]]
        
        if len(curves) < 2:
            return None
        
        # 检查曲线是否首尾相连形成闭合
        # 简化方法：检查所有曲线的激活点是否形成一个环
        all_curve_acts = []
        for c in curves:
            all_curve_acts.extend(c.activations)
        
        if len(all_curve_acts) < 8:
            return None
        
        # 检查是否形成封闭环
        # 方法：计算质心，检查点是否均匀分布在质心周围
        xs = [a.x for a in all_curve_acts]
        ys = [a.y for a in all_curve_acts]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # 计算每个点到质心的角度
        angles = []
        for a in all_curve_acts:
            angle = np.arctan2(a.y - cy, a.x - cx)
            angles.append(angle)
        
        angles = sorted(angles)
        
        # 检查角度覆盖：如果形成闭合圈，角度应该覆盖接近360度
        angle_coverage = angles[-1] - angles[0]
        
        # 检查角度分布的均匀性
        angle_gaps = []
        for i in range(1, len(angles)):
            angle_gaps.append(angles[i] - angles[i-1])
        
        max_gap = max(angle_gaps) if angle_gaps else np.pi
        
        # 闭合圈条件：覆盖大部分角度，且没有大的空隙
        if angle_coverage > np.pi * 1.5 and max_gap < np.pi * 0.5:
            anchors = Anchors(
                center=(cx, cy),
                head=(cx, min(ys)),
                tail=(cx, max(ys))
            )
            
            return AtomInstance(
                atom_type=AtomType.LOOP,
                anchors=anchors,
                activations=all_curve_acts,
                confidence=0.8
            )
        
        return None
    
    def _sort_by_path(self, activations: List[Activation]) -> List[Activation]:
        """按路径排序"""
        if len(activations) <= 2:
            return activations
        
        # 从最左上的点开始
        start_idx = min(range(len(activations)), 
                       key=lambda i: (activations[i].y, activations[i].x))
        
        sorted_acts = [activations[start_idx]]
        remaining = set(range(len(activations))) - {start_idx}
        
        while remaining:
            current = sorted_acts[-1]
            nearest = min(remaining, 
                         key=lambda i: (activations[i].x - current.x)**2 + 
                                      (activations[i].y - current.y)**2)
            sorted_acts.append(activations[nearest])
            remaining.remove(nearest)
        
        return sorted_acts
    
    def _compute_anchors(self, activations: List[Activation], atom_type: AtomType) -> Anchors:
        """计算锚点"""
        xs = [a.x for a in activations]
        ys = [a.y for a in activations]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        if atom_type in [AtomType.LINE_V, AtomType.LINE_D1, AtomType.LINE_D2]:
            # 竖向或斜向线：head在上，tail在下
            head_idx = min(range(len(activations)), key=lambda i: activations[i].y)
            tail_idx = max(range(len(activations)), key=lambda i: activations[i].y)
        else:
            # 横向线或曲线：head在左，tail在右
            head_idx = min(range(len(activations)), key=lambda i: activations[i].x)
            tail_idx = max(range(len(activations)), key=lambda i: activations[i].x)
        
        return Anchors(
            center=(cx, cy),
            head=(activations[head_idx].x, activations[head_idx].y),
            tail=(activations[tail_idx].x, activations[tail_idx].y)
        )


# =============================================================================
# 5. Relation Analyzer
# =============================================================================

class RelationAnalyzer:
    def __init__(self, distance_threshold: float = 6.0):
        self.distance_threshold = distance_threshold
    
    def compute_relations(self, atoms: List[AtomInstance]) -> List[AtomRelation]:
        relations = []
        
        for i, a1 in enumerate(atoms):
            for j, a2 in enumerate(atoms):
                if i >= j:
                    continue
                
                rel = self._classify_relation(a1, a2)
                if rel:
                    relations.append(rel)
        
        return relations
    
    def _classify_relation(self, a1: AtomInstance, a2: AtomInstance) -> Optional[AtomRelation]:
        # 包含关系
        if a2.atom_type == AtomType.LOOP and self._is_inside(a1, a2):
            return AtomRelation(a1.atom_type, a2.atom_type, RelationType.INSIDE)
        if a1.atom_type == AtomType.LOOP and self._is_inside(a2, a1):
            return AtomRelation(a1.atom_type, a2.atom_type, RelationType.CONTAINS)
        
        # 连接关系
        if self._is_connected(a1, a2):
            rel_type = self._get_relative_position(a1, a2)
            return AtomRelation(a1.atom_type, a2.atom_type, rel_type)
        
        # 相邻关系
        if self._is_adjacent(a1, a2):
            rel_type = self._get_relative_position(a1, a2)
            return AtomRelation(a1.atom_type, a2.atom_type, rel_type)
        
        return None
    
    def _is_inside(self, a1: AtomInstance, a2: AtomInstance) -> bool:
        bbox = a2.bbox
        c = a1.center
        margin = 3
        return (bbox[0] + margin < c[0] < bbox[2] - margin and
                bbox[1] + margin < c[1] < bbox[3] - margin)
    
    def _is_connected(self, a1: AtomInstance, a2: AtomInstance) -> bool:
        min_dist = self._min_anchor_distance(a1, a2)
        return min_dist < self.distance_threshold
    
    def _is_adjacent(self, a1: AtomInstance, a2: AtomInstance) -> bool:
        min_dist = self._min_anchor_distance(a1, a2)
        return min_dist < self.distance_threshold * 2
    
    def _min_anchor_distance(self, a1: AtomInstance, a2: AtomInstance) -> float:
        min_d = float('inf')
        for p1 in [a1.anchors.center, a1.anchors.head, a1.anchors.tail]:
            for p2 in [a2.anchors.center, a2.anchors.head, a2.anchors.tail]:
                d = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_d = min(min_d, d)
        return min_d
    
    def _get_relative_position(self, a1: AtomInstance, a2: AtomInstance) -> RelationType:
        c1 = a1.center
        c2 = a2.center
        
        dy = c1[1] - c2[1]
        dx = c1[0] - c2[0]
        
        threshold = 4
        
        if abs(dy) > abs(dx) + 2:
            return RelationType.ABOVE if dy < -threshold else RelationType.BELOW if dy > threshold else RelationType.CONNECTED
        elif abs(dx) > abs(dy) + 2:
            return RelationType.LEFT if dx < -threshold else RelationType.RIGHT if dx > threshold else RelationType.CONNECTED
        else:
            return RelationType.ADJACENT
    
    def build_signature(self, atoms: List[AtomInstance]) -> StructuralSignature:
        atom_types = frozenset(a.atom_type for a in atoms)
        relations = self.compute_relations(atoms)
        relation_strings = frozenset(r.to_string() for r in relations)
        
        return StructuralSignature(
            atom_types=atom_types,
            relations=relation_strings,
            atom_count=len(atoms)
        )


# =============================================================================
# 6. Structural Memory
# =============================================================================

class StructuralMemory:
    def __init__(self):
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
        self.signature_to_label: Dict[StructuralSignature, str] = {}
    
    def learn(self, signature: StructuralSignature, label: str):
        self.label_signatures[label].append(signature)
        if signature not in self.signature_to_label:
            self.signature_to_label[signature] = label
    
    def recognize(self, signature: StructuralSignature) -> Tuple[Optional[str], float]:
        # 精确匹配
        if signature in self.signature_to_label:
            return self.signature_to_label[signature], 1.0
        
        # 相似度匹配
        best_label = None
        best_score = 0.0
        
        for label, signatures in self.label_signatures.items():
            for stored_sig in signatures:
                sim = signature.similarity(stored_sig)
                if sim > best_score:
                    best_score = sim
                    best_label = label
        
        if best_score < 0.2:
            return None, best_score
        
        return best_label, best_score
    
    def analyze(self):
        print("\n=== 结构签名分析 ===")
        for label, sigs in sorted(self.label_signatures.items()):
            print(f"\n数字 {label} ({len(sigs)} 个样本):")
            
            type_counts = defaultdict(int)
            rel_counts = defaultdict(int)
            atom_counts = defaultdict(int)
            
            for sig in sigs:
                for t in sig.atom_types:
                    type_counts[t.value] += 1
                for r in sig.relations:
                    rel_counts[r] += 1
                atom_counts[sig.atom_count] += 1
            
            print(f"  原子类型分布:")
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                print(f"    {t}: {c}次")
            
            if rel_counts:
                print(f"  关系分布:")
                for r, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:5]:
                    print(f"    {r}: {c}次")
            
            print(f"  原子数量分布: {dict(atom_counts)}")


# =============================================================================
# 7. Vision System
# =============================================================================

class VisionSystem:
    def __init__(self):
        self.v1 = V1GaborBank(kernel_size=5)
        self.atom_detector = AtomDetector(image_size=28.0)
        self.relation_analyzer = RelationAnalyzer(distance_threshold=6.0)
        self.memory = StructuralMemory()
    
    def train(self, image: np.ndarray, label: str):
        activations = self.v1.extract_activations(image)
        atoms = self.atom_detector.detect_atoms(activations)
        signature = self.relation_analyzer.build_signature(atoms)
        self.memory.learn(signature, label)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        activations = self.v1.extract_activations(image)
        atoms = self.atom_detector.detect_atoms(activations)
        signature = self.relation_analyzer.build_signature(atoms)
        label, conf = self.memory.recognize(signature)
        return label or "unknown", conf
    
    def debug_image(self, image: np.ndarray) -> dict:
        activations = self.v1.extract_activations(image)
        atoms = self.atom_detector.detect_atoms(activations)
        relations = self.relation_analyzer.compute_relations(atoms)
        signature = self.relation_analyzer.build_signature(atoms)
        
        return {
            'n_activations': len(activations),
            'n_atoms': len(atoms),
            'atoms': [(a.atom_type.value, f"({a.center[0]:.1f}, {a.center[1]:.1f})") for a in atoms],
            'relations': [r.to_string() for r in relations],
            'signature': {
                'types': [t.value for t in signature.atom_types],
                'relations': list(signature.relations),
                'atom_count': signature.atom_count
            }
        }


# =============================================================================
# 8. MNIST Loading
# =============================================================================

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
]

MNIST_FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}


def download_mnist(data_dir: str = './mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in MNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            for mirror in MNIST_MIRRORS:
                try:
                    urllib.request.urlretrieve(mirror + filename, filepath)
                    break
                except:
                    continue
    print("MNIST data ready.")


def load_mnist_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0


def load_mnist_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir: str = './mnist_data'):
    download_mnist(data_dir)
    return (
        load_mnist_images(os.path.join(data_dir, MNIST_FILES['train_images'])),
        load_mnist_labels(os.path.join(data_dir, MNIST_FILES['train_labels'])),
        load_mnist_images(os.path.join(data_dir, MNIST_FILES['test_images'])),
        load_mnist_labels(os.path.join(data_dir, MNIST_FILES['test_labels']))
    )


# =============================================================================
# 9. Experiment
# =============================================================================

def run_experiment(n_train_per_class: int = 10, n_test: int = 500, verbose: bool = True):
    print("=" * 60)
    print("Structon Vision v6.4 - Fixed Atom Detection")
    print("=" * 60)
    print("\nv6.3的问题：Loop检测太激进，所有数字都被识别为Loop")
    print("v6.4的修复：")
    print("  - 先找连通分量，再分析形状")
    print("  - 更严格的Loop检测（需要真正的闭合）")
    print("  - 基于方向变化分类直线/曲线")
    
    # 加载数据
    print("\n加载MNIST数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 选择训练样本
    train_indices = []
    for digit in range(10):
        digit_indices = np.where(train_labels == digit)[0][:n_train_per_class]
        train_indices.extend(digit_indices)
    
    total_train = len(train_indices)
    n_test = min(n_test, len(test_images))
    
    print(f"训练样本: {total_train} (每类 {n_train_per_class} 个)")
    print(f"测试样本: {n_test}")
    
    # 创建系统
    system = VisionSystem()
    
    # 训练
    print("\n" + "=" * 40)
    print("训练中...")
    print("=" * 40)
    
    start_time = time.time()
    
    for i, idx in enumerate(train_indices):
        system.train(train_images[idx], str(train_labels[idx]))
        if verbose and (i + 1) % 20 == 0:
            print(f"  已训练 {i+1}/{total_train}")
    
    train_time = time.time() - start_time
    print(f"\n训练完成，用时 {train_time:.2f}秒")
    
    if verbose:
        system.memory.analyze()
    
    # 测试
    print("\n" + "=" * 40)
    print("测试中...")
    print("=" * 40)
    
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    start_time = time.time()
    
    for i, idx in enumerate(test_indices):
        true_label = str(test_labels[idx])
        pred_label, conf = system.predict(test_images[idx])
        
        results[true_label]['total'] += 1
        if pred_label == true_label:
            results[true_label]['correct'] += 1
        
        if verbose and (i + 1) % 100 == 0:
            current_acc = sum(r['correct'] for r in results.values()) / (i + 1) * 100
            print(f"  已测试 {i+1}/{n_test} - 当前准确率: {current_acc:.1f}%")
    
    test_time = time.time() - start_time
    
    # 结果
    print("\n" + "=" * 40)
    print("结果")
    print("=" * 40)
    
    total_correct = sum(r['correct'] for r in results.values())
    total_count = sum(r['total'] for r in results.values())
    
    print(f"\n总准确率: {total_correct/total_count*100:.1f}%")
    print(f"随机基线: 10.0%")
    
    print(f"\n各数字准确率:")
    for digit in range(10):
        r = results[str(digit)]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  数字 {digit}: {acc:5.1f}% ({r['correct']}/{r['total']})")
    
    print(f"\n时间: 训练 {train_time:.2f}秒, 测试 {test_time:.2f}秒")
    
    return system, results


def debug_digits(n_per_digit: int = 3):
    """调试：检查每个数字的结构分析"""
    print("\n" + "=" * 60)
    print("数字结构调试")
    print("=" * 60)
    
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n{'='*40}")
        print(f"数字 {digit}")
        print('='*40)
        
        indices = np.where(train_labels == digit)[0][:n_per_digit]
        
        for i, idx in enumerate(indices):
            analysis = system.debug_image(train_images[idx])
            
            print(f"\n  样本 {i+1}:")
            print(f"    激活点: {analysis['n_activations']}")
            print(f"    原子数: {analysis['n_atoms']}")
            print(f"    原子: {analysis['atoms']}")
            print(f"    关系: {analysis['relations']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Structon Vision v6.4')
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug', action='store_true', help='只运行调试')
    
    args = parser.parse_args()
    
    if args.debug:
        debug_digits(n_per_digit=3)
    else:
        run_experiment(
            n_train_per_class=args.train_per_class,
            n_test=args.test,
            verbose=not args.quiet
        )
