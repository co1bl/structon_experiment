"""
Structon Vision v6.3 - Structural Relations (The Brain)
========================================================

核心修复：从"词袋模型"回归到"结构化模型"

v6.2的问题：
- 只统计原子类型的直方图（有几个圈、几条线）
- 丢失了空间关系（圈在线的上面还是下面？）
- 导致无法区分 6 和 9

v6.3的改进：
- 复活连接器：计算原子间的空间关系
- 结构签名 = {原子类型} + {原子间关系}
- 基于结构匹配，而不是统计匹配

数字签名示例：
- 6: {Loop, Line_V} + {Line_V_above_Loop}
- 9: {Loop, Line_V} + {Line_V_below_Loop}  
- 8: {Loop, Loop} + {Loop_above_Loop}

目标：用 100 个样本（每类10张）达到 80%+ 准确率

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
    """原子类型"""
    # 直线类
    LINE_H = "line_h"
    LINE_V = "line_v"
    LINE_D1 = "line_d1"  # /
    LINE_D2 = "line_d2"  # \
    
    # 曲线类
    CURVE_C = "curve_c"   # 左开口 C
    CURVE_D = "curve_d"   # 右开口 反C
    CURVE_U = "curve_u"   # U形
    CURVE_N = "curve_n"   # 拱形
    
    # 特殊类
    LOOP = "loop"
    ENDPOINT = "endpoint"
    
    UNKNOWN = "unknown"


class RelationType(Enum):
    """关系类型"""
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    INSIDE = "inside"
    CONTAINS = "contains"
    CONNECTED = "connected"
    CROSSING = "crossing"


# =============================================================================
# 2. Data Structures
# =============================================================================

@dataclass
class Activation:
    """V1激活点"""
    x: float
    y: float
    strength: float
    orientation: int
    feature_response: np.ndarray


@dataclass
class Anchors:
    """空间锚点"""
    center: Tuple[float, float]
    head: Tuple[float, float]
    tail: Tuple[float, float]
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """边界框 (min_x, min_y, max_x, max_y)"""
        xs = [self.center[0], self.head[0], self.tail[0]]
        ys = [self.center[1], self.head[1], self.tail[1]]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class AtomInstance:
    """检测到的原子实例"""
    atom_type: AtomType
    anchors: Anchors
    activations: List[Activation]
    confidence: float
    feature_signature: np.ndarray
    
    @property
    def center(self) -> Tuple[float, float]:
        return self.anchors.center
    
    @property 
    def bbox(self) -> Tuple[float, float, float, float]:
        """更精确的边界框"""
        if not self.activations:
            return self.anchors.bbox
        xs = [a.x for a in self.activations]
        ys = [a.y for a in self.activations]
        return (min(xs), min(ys), max(xs), max(ys))


@dataclass
class AtomRelation:
    """原子间关系"""
    atom1_type: AtomType
    atom2_type: AtomType
    relation: RelationType
    
    def __hash__(self):
        return hash((self.atom1_type, self.atom2_type, self.relation))
    
    def __eq__(self, other):
        return (self.atom1_type == other.atom1_type and 
                self.atom2_type == other.atom2_type and 
                self.relation == other.relation)
    
    def to_string(self) -> str:
        return f"{self.atom1_type.value}_{self.relation.value}_{self.atom2_type.value}"


@dataclass(frozen=True)
class StructuralSignature:
    """
    结构签名 = 原子类型集合 + 关系集合
    
    这是模式的本质，不是统计量
    """
    atom_types: FrozenSet[AtomType]
    relations: FrozenSet[str]  # 关系字符串集合
    
    def __hash__(self):
        return hash((self.atom_types, self.relations))
    
    def similarity(self, other: 'StructuralSignature') -> float:
        """计算两个结构签名的相似度"""
        # 原子类型匹配
        type_intersection = len(self.atom_types & other.atom_types)
        type_union = len(self.atom_types | other.atom_types)
        type_sim = type_intersection / type_union if type_union > 0 else 0
        
        # 关系匹配
        rel_intersection = len(self.relations & other.relations)
        rel_union = len(self.relations | other.relations)
        rel_sim = rel_intersection / rel_union if rel_union > 0 else 0
        
        # 关系更重要（区分 6 和 9 的关键）
        return 0.3 * type_sim + 0.7 * rel_sim


# =============================================================================
# 3. V1 Layer
# =============================================================================

class V1GaborBank:
    """V1 Gabor滤波器组"""
    
    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size
        self.orientations = [0, 20, 40, 60, 80, 100, 120, 140, 160]
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
    
    def compute_response_maps(self, image: np.ndarray) -> List[np.ndarray]:
        return [np.abs(self._convolve(image, kernel)) for kernel in self.filters]
    
    def compute_orientation_map(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算方向图和强度图"""
        response_maps = self.compute_response_maps(image)
        h, w = image.shape
        
        orientation_map = np.zeros((h, w), dtype=np.float32)
        strength_map = np.zeros((h, w), dtype=np.float32)
        
        for y in range(h):
            for x in range(w):
                max_response = -1
                best_orient = 0
                for feat_idx in range(self.n_features):
                    val = response_maps[feat_idx][y, x]
                    if val > max_response:
                        max_response = val
                        orient_idx = feat_idx % self.n_orientations
                        best_orient = self.orientations[orient_idx]
                
                orientation_map[y, x] = best_orient
                strength_map[y, x] = max_response
        
        return orientation_map, strength_map
    
    def extract_activations(self, image: np.ndarray, threshold: float = 0.25, step: int = 2) -> List[Activation]:
        """提取激活点"""
        response_maps = self.compute_response_maps(image)
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
                
                best_idx = np.argmax(feature_vec)
                dominant_orient = self.orientations[best_idx % self.n_orientations]
                
                activations.append(Activation(
                    x=float(x),
                    y=float(y),
                    strength=max_response,
                    orientation=dominant_orient,
                    feature_response=feature_vec
                ))
        
        return activations


# =============================================================================
# 4. Atom Detector (眼睛 - 保持不变)
# =============================================================================

class AtomDetector:
    """原子检测器"""
    
    def __init__(self, image_size: float = 28.0):
        self.image_size = image_size
        self.min_activations = 3
        self.curve_rotation_threshold = 50
    
    def detect_atoms(self, activations: List[Activation], 
                     strength_map: np.ndarray) -> List[AtomInstance]:
        """检测所有原子"""
        if len(activations) < self.min_activations:
            return []
        
        atoms = []
        used: Set[int] = set()
        
        # 1. 检测闭合圈
        loop = self._detect_loop(activations, used)
        if loop:
            atoms.append(loop)
        
        # 2. 检测曲线
        curves = self._detect_curves(activations, used)
        atoms.extend(curves)
        
        # 3. 检测直线
        lines = self._detect_lines(activations, used)
        atoms.extend(lines)
        
        return atoms
    
    def _detect_loop(self, activations: List[Activation], used: Set[int]) -> Optional[AtomInstance]:
        """检测闭合圈"""
        if len(activations) < 6:
            return None
        
        cx = np.mean([a.x for a in activations])
        cy = np.mean([a.y for a in activations])
        
        distances = [np.sqrt((a.x - cx)**2 + (a.y - cy)**2) for a in activations]
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # 均匀分布在质心周围
        if std_dist < mean_dist * 0.5 and mean_dist > 2:
            orientations = [a.orientation for a in activations]
            unique_orients = len(set(o // 40 for o in orientations))
            
            if unique_orients >= 3:
                for i in range(len(activations)):
                    used.add(i)
                
                anchors = Anchors(
                    center=(cx, cy),
                    head=(cx, cy - mean_dist),
                    tail=(cx, cy + mean_dist)
                )
                
                return AtomInstance(
                    atom_type=AtomType.LOOP,
                    anchors=anchors,
                    activations=activations.copy(),
                    confidence=0.8,
                    feature_signature=self._compute_feature(activations)
                )
        
        return None
    
    def _detect_curves(self, activations: List[Activation], used: Set[int]) -> List[AtomInstance]:
        """检测曲线"""
        curves = []
        available = [i for i in range(len(activations)) if i not in used]
        
        if len(available) < 4:
            return curves
        
        clusters = self._spatial_cluster([activations[i] for i in available], radius=6)
        
        for cluster_indices in clusters:
            if len(cluster_indices) < 4:
                continue
            
            cluster_acts = [activations[available[i]] for i in cluster_indices]
            curve_type = self._classify_curve(cluster_acts)
            
            if curve_type != AtomType.UNKNOWN:
                for i in cluster_indices:
                    used.add(available[i])
                
                anchors = self._compute_curve_anchors(cluster_acts, curve_type)
                
                curves.append(AtomInstance(
                    atom_type=curve_type,
                    anchors=anchors,
                    activations=cluster_acts,
                    confidence=0.7,
                    feature_signature=self._compute_feature(cluster_acts)
                ))
        
        return curves
    
    def _classify_curve(self, activations: List[Activation]) -> AtomType:
        """分类曲线类型"""
        if len(activations) < 4:
            return AtomType.UNKNOWN
        
        sorted_acts = self._sort_by_path(activations)
        orientations = [a.orientation for a in sorted_acts]
        
        # 计算总旋转
        total_rotation = 0
        for i in range(1, len(orientations)):
            diff = orientations[i] - orientations[i-1]
            if diff > 90:
                diff -= 180
            elif diff < -90:
                diff += 180
            total_rotation += diff
        
        if abs(total_rotation) < self.curve_rotation_threshold:
            return AtomType.UNKNOWN
        
        # 计算边界和质心
        cx = np.mean([a.x for a in activations])
        cy = np.mean([a.y for a in activations])
        
        min_x = min(a.x for a in activations)
        max_x = max(a.x for a in activations)
        min_y = min(a.y for a in activations)
        max_y = max(a.y for a in activations)
        
        width = max_x - min_x + 1e-8
        height = max_y - min_y + 1e-8
        
        cx_rel = (cx - min_x) / width
        cy_rel = (cy - min_y) / height
        
        # 判断开口方向
        if width > height * 0.7:
            if cy_rel < 0.4:
                return AtomType.CURVE_N
            elif cy_rel > 0.6:
                return AtomType.CURVE_U
        
        if height > width * 0.7:
            if cx_rel < 0.4:
                return AtomType.CURVE_D
            elif cx_rel > 0.6:
                return AtomType.CURVE_C
        
        return AtomType.CURVE_C if total_rotation > 0 else AtomType.CURVE_D
    
    def _detect_lines(self, activations: List[Activation], used: Set[int]) -> List[AtomInstance]:
        """检测直线"""
        lines = []
        available = [i for i in range(len(activations)) if i not in used]
        
        if len(available) < 3:
            return lines
        
        direction_groups: Dict[int, List[int]] = defaultdict(list)
        
        for i in available:
            a = activations[i]
            bucket = (a.orientation + 22) // 45 * 45 % 180
            direction_groups[bucket].append(i)
        
        for direction, indices in direction_groups.items():
            if len(indices) < 3:
                continue
            
            acts = [activations[i] for i in indices]
            
            if self._is_collinear(acts):
                for i in indices:
                    used.add(i)
                
                line_type = self._classify_line(direction)
                anchors = self._compute_line_anchors(acts)
                
                lines.append(AtomInstance(
                    atom_type=line_type,
                    anchors=anchors,
                    activations=acts,
                    confidence=0.8,
                    feature_signature=self._compute_feature(acts)
                ))
        
        return lines
    
    def _classify_line(self, direction: int) -> AtomType:
        if direction in [0, 180]:
            return AtomType.LINE_V
        elif direction == 90:
            return AtomType.LINE_H
        elif direction == 45:
            return AtomType.LINE_D1
        elif direction == 135:
            return AtomType.LINE_D2
        return AtomType.LINE_H
    
    def _spatial_cluster(self, activations: List[Activation], radius: float) -> List[List[int]]:
        """空间聚类"""
        n = len(activations)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            cluster = [i]
            visited[i] = True
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                ca = activations[current]
                
                for j in range(n):
                    if visited[j]:
                        continue
                    
                    ja = activations[j]
                    dist = np.sqrt((ca.x - ja.x)**2 + (ca.y - ja.y)**2)
                    
                    if dist < radius:
                        visited[j] = True
                        cluster.append(j)
                        queue.append(j)
            
            if len(cluster) >= 3:
                clusters.append(cluster)
        
        return clusters
    
    def _sort_by_path(self, activations: List[Activation]) -> List[Activation]:
        """按路径排序"""
        if len(activations) <= 2:
            return activations
        
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
    
    def _is_collinear(self, activations: List[Activation], tolerance: float = 3.0) -> bool:
        """检查共线"""
        if len(activations) < 3:
            return True
        
        p1 = (activations[0].x, activations[0].y)
        p2 = (activations[-1].x, activations[-1].y)
        
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length < 1e-8:
            return True
        
        for a in activations[1:-1]:
            dist = abs(dy * a.x - dx * a.y + p2[0]*p1[1] - p2[1]*p1[0]) / length
            if dist > tolerance:
                return False
        
        return True
    
    def _compute_line_anchors(self, activations: List[Activation]) -> Anchors:
        sorted_acts = sorted(activations, key=lambda a: (a.x, a.y))
        head = (sorted_acts[0].x, sorted_acts[0].y)
        tail = (sorted_acts[-1].x, sorted_acts[-1].y)
        center = ((head[0] + tail[0]) / 2, (head[1] + tail[1]) / 2)
        return Anchors(center=center, head=head, tail=tail)
    
    def _compute_curve_anchors(self, activations: List[Activation], curve_type: AtomType) -> Anchors:
        cx = np.mean([a.x for a in activations])
        cy = np.mean([a.y for a in activations])
        
        min_x_act = min(activations, key=lambda a: a.x)
        max_x_act = max(activations, key=lambda a: a.x)
        min_y_act = min(activations, key=lambda a: a.y)
        max_y_act = max(activations, key=lambda a: a.y)
        
        if curve_type == AtomType.CURVE_C:
            return Anchors(
                center=(min_x_act.x, cy),
                head=(max_x_act.x, min_y_act.y),
                tail=(max_x_act.x, max_y_act.y)
            )
        elif curve_type == AtomType.CURVE_D:
            return Anchors(
                center=(max_x_act.x, cy),
                head=(min_x_act.x, min_y_act.y),
                tail=(min_x_act.x, max_y_act.y)
            )
        elif curve_type == AtomType.CURVE_U:
            return Anchors(
                center=(cx, max_y_act.y),
                head=(min_x_act.x, min_y_act.y),
                tail=(max_x_act.x, min_y_act.y)
            )
        elif curve_type == AtomType.CURVE_N:
            return Anchors(
                center=(cx, min_y_act.y),
                head=(min_x_act.x, max_y_act.y),
                tail=(max_x_act.x, max_y_act.y)
            )
        else:
            return Anchors(center=(cx, cy), 
                          head=(min_x_act.x, min_y_act.y),
                          tail=(max_x_act.x, max_y_act.y))
    
    def _compute_feature(self, activations: List[Activation]) -> np.ndarray:
        if not activations:
            return np.zeros(18, dtype=np.float32)
        
        total_weight = sum(a.strength for a in activations)
        if total_weight < 1e-8:
            return activations[0].feature_response
        
        aggregated = np.zeros_like(activations[0].feature_response)
        for a in activations:
            aggregated += a.feature_response * a.strength
        aggregated /= total_weight
        
        norm = np.linalg.norm(aggregated)
        return aggregated / norm if norm > 1e-8 else aggregated


# =============================================================================
# 5. Relation Analyzer (大脑 - 核心新增)
# =============================================================================

class RelationAnalyzer:
    """
    关系分析器 - 计算原子间的空间关系
    
    这是区分 6 和 9 的关键！
    """
    
    def __init__(self, distance_threshold: float = 8.0):
        self.distance_threshold = distance_threshold
    
    def compute_relations(self, atoms: List[AtomInstance]) -> List[AtomRelation]:
        """计算所有原子对之间的关系"""
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
        """分类两个原子之间的关系"""
        
        # 1. 检查包含关系
        if self._is_inside(a1, a2):
            return AtomRelation(a1.atom_type, a2.atom_type, RelationType.INSIDE)
        if self._is_inside(a2, a1):
            return AtomRelation(a1.atom_type, a2.atom_type, RelationType.CONTAINS)
        
        # 2. 检查连接关系
        if self._is_connected(a1, a2):
            # 3. 确定相对位置
            rel_type = self._get_relative_position(a1, a2)
            return AtomRelation(a1.atom_type, a2.atom_type, rel_type)
        
        # 4. 检查交叉
        if self._is_crossing(a1, a2):
            return AtomRelation(a1.atom_type, a2.atom_type, RelationType.CROSSING)
        
        return None
    
    def _is_inside(self, a1: AtomInstance, a2: AtomInstance) -> bool:
        """检查 a1 是否在 a2 内部"""
        if a2.atom_type != AtomType.LOOP:
            return False
        
        # a1 的中心是否在 a2 的边界框内
        bbox = a2.bbox
        c = a1.center
        
        margin = 2
        return (bbox[0] + margin < c[0] < bbox[2] - margin and
                bbox[1] + margin < c[1] < bbox[3] - margin)
    
    def _is_connected(self, a1: AtomInstance, a2: AtomInstance) -> bool:
        """检查两个原子是否连接"""
        # 方法1：边界框重叠或接近
        bbox1 = a1.bbox
        bbox2 = a2.bbox
        
        # 扩展边界框
        margin = self.distance_threshold
        
        expanded1 = (bbox1[0] - margin, bbox1[1] - margin, 
                     bbox1[2] + margin, bbox1[3] + margin)
        
        # 检查重叠
        if (expanded1[2] < bbox2[0] or bbox2[2] < expanded1[0] or
            expanded1[3] < bbox2[1] or bbox2[3] < expanded1[1]):
            return False
        
        # 方法2：最近点距离
        min_dist = self._min_distance(a1, a2)
        return min_dist < self.distance_threshold
    
    def _min_distance(self, a1: AtomInstance, a2: AtomInstance) -> float:
        """计算两个原子之间的最小距离"""
        min_d = float('inf')
        
        # 锚点之间的距离
        for p1 in [a1.anchors.center, a1.anchors.head, a1.anchors.tail]:
            for p2 in [a2.anchors.center, a2.anchors.head, a2.anchors.tail]:
                d = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
                min_d = min(min_d, d)
        
        return min_d
    
    def _is_crossing(self, a1: AtomInstance, a2: AtomInstance) -> bool:
        """检查两个原子是否交叉"""
        # 简单方法：中心距离很近但不是连接
        c1 = a1.center
        c2 = a2.center
        
        center_dist = np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
        return center_dist < self.distance_threshold * 0.5
    
    def _get_relative_position(self, a1: AtomInstance, a2: AtomInstance) -> RelationType:
        """获取 a1 相对于 a2 的位置"""
        c1 = a1.center
        c2 = a2.center
        
        dy = c1[1] - c2[1]  # 正 = a1 在下, 负 = a1 在上
        dx = c1[0] - c2[0]  # 正 = a1 在右, 负 = a1 在左
        
        threshold = 3  # 最小差异阈值
        
        # 判断主要方向
        if abs(dy) > abs(dx):
            # 垂直关系为主
            if dy < -threshold:
                return RelationType.ABOVE
            elif dy > threshold:
                return RelationType.BELOW
        else:
            # 水平关系为主
            if dx < -threshold:
                return RelationType.LEFT
            elif dx > threshold:
                return RelationType.RIGHT
        
        return RelationType.CONNECTED
    
    def build_signature(self, atoms: List[AtomInstance]) -> StructuralSignature:
        """构建结构签名"""
        atom_types = frozenset(a.atom_type for a in atoms)
        
        relations = self.compute_relations(atoms)
        relation_strings = frozenset(r.to_string() for r in relations)
        
        return StructuralSignature(
            atom_types=atom_types,
            relations=relation_strings
        )


# =============================================================================
# 6. Structural Memory (基于结构匹配)
# =============================================================================

class StructuralMemory:
    """
    结构化记忆 - 基于结构签名匹配
    
    不是统计平均，而是结构对比
    """
    
    def __init__(self):
        # 每个标签存储其结构签名集合
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
        
        # 索引：签名 -> 标签（用于快速查找）
        self.signature_to_label: Dict[StructuralSignature, str] = {}
    
    def learn(self, signature: StructuralSignature, label: str):
        """学习一个结构签名"""
        self.label_signatures[label].append(signature)
        
        # 尝试添加到索引（如果签名唯一）
        if signature not in self.signature_to_label:
            self.signature_to_label[signature] = label
    
    def recognize(self, signature: StructuralSignature) -> Tuple[Optional[str], float]:
        """基于结构签名识别"""
        
        # 1. 精确匹配
        if signature in self.signature_to_label:
            return self.signature_to_label[signature], 1.0
        
        # 2. 相似度匹配
        best_label = None
        best_score = 0.0
        
        for label, signatures in self.label_signatures.items():
            for stored_sig in signatures:
                sim = signature.similarity(stored_sig)
                if sim > best_score:
                    best_score = sim
                    best_label = label
        
        # 阈值
        if best_score < 0.3:
            return None, best_score
        
        return best_label, best_score
    
    def analyze(self):
        """分析存储的结构签名"""
        print("\n=== 结构签名分析 ===")
        for label, sigs in sorted(self.label_signatures.items()):
            print(f"\n数字 {label} ({len(sigs)} 个样本):")
            
            # 统计常见的原子类型组合
            type_sets = defaultdict(int)
            relation_sets = defaultdict(int)
            
            for sig in sigs:
                type_key = tuple(sorted(t.value for t in sig.atom_types))
                type_sets[type_key] += 1
                
                for rel in sig.relations:
                    relation_sets[rel] += 1
            
            print(f"  常见原子组合:")
            for types, count in sorted(type_sets.items(), key=lambda x: -x[1])[:3]:
                print(f"    {types}: {count}次")
            
            print(f"  常见关系:")
            for rel, count in sorted(relation_sets.items(), key=lambda x: -x[1])[:3]:
                print(f"    {rel}: {count}次")


# =============================================================================
# 7. Vision System
# =============================================================================

class VisionSystem:
    """视觉系统"""
    
    def __init__(self):
        self.v1 = V1GaborBank(kernel_size=5)
        self.atom_detector = AtomDetector(image_size=28.0)
        self.relation_analyzer = RelationAnalyzer(distance_threshold=8.0)
        self.memory = StructuralMemory()
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        activations = self.v1.extract_activations(image)
        _, strength_map = self.v1.compute_orientation_map(image)
        
        atoms = self.atom_detector.detect_atoms(activations, strength_map)
        signature = self.relation_analyzer.build_signature(atoms)
        
        self.memory.learn(signature, label)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        activations = self.v1.extract_activations(image)
        _, strength_map = self.v1.compute_orientation_map(image)
        
        atoms = self.atom_detector.detect_atoms(activations, strength_map)
        signature = self.relation_analyzer.build_signature(atoms)
        
        label, conf = self.memory.recognize(signature)
        return label or "unknown", conf
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试：返回图像的完整分析"""
        activations = self.v1.extract_activations(image)
        _, strength_map = self.v1.compute_orientation_map(image)
        
        atoms = self.atom_detector.detect_atoms(activations, strength_map)
        relations = self.relation_analyzer.compute_relations(atoms)
        signature = self.relation_analyzer.build_signature(atoms)
        
        return {
            'n_activations': len(activations),
            'atoms': [(a.atom_type.value, a.center) for a in atoms],
            'relations': [r.to_string() for r in relations],
            'signature': {
                'types': [t.value for t in signature.atom_types],
                'relations': list(signature.relations)
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
                    url = mirror + filename
                    print(f"  Trying {mirror}...")
                    urllib.request.urlretrieve(url, filepath)
                    print(f"  Saved to {filepath}")
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue
    
    print("MNIST data ready.")


def load_mnist_images(filepath: str) -> np.ndarray:
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0


def load_mnist_labels(filepath: str) -> np.ndarray:
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir: str = './mnist_data'):
    download_mnist(data_dir)
    
    train_images = load_mnist_images(os.path.join(data_dir, MNIST_FILES['train_images']))
    train_labels = load_mnist_labels(os.path.join(data_dir, MNIST_FILES['train_labels']))
    test_images = load_mnist_images(os.path.join(data_dir, MNIST_FILES['test_images']))
    test_labels = load_mnist_labels(os.path.join(data_dir, MNIST_FILES['test_labels']))
    
    return train_images, train_labels, test_images, test_labels


# =============================================================================
# 9. Experiment
# =============================================================================

def run_experiment(n_train_per_class: int = 10, n_test: int = 500, verbose: bool = True):
    """
    运行实验
    
    关键改变：默认只用每类10个样本！
    如果结构化方法有效，这应该足够了。
    """
    print("=" * 60)
    print("Structon Vision v6.3 - Structural Relations")
    print("=" * 60)
    print("\n核心改进：")
    print("  - 基于结构签名匹配，不是统计直方图")
    print("  - 签名 = {原子类型} + {原子间关系}")
    print("  - 例如：6 = {Loop, Line_V} + {Line_V_above_Loop}")
    print(f"\n目标：用 {n_train_per_class * 10} 个样本达到 80%+ 准确率")
    
    # 加载数据
    print("\n加载MNIST数据...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 选择训练样本：每类 n_train_per_class 个
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
        image = train_images[idx]
        label = str(train_labels[idx])
        system.train(image, label)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  已训练 {i+1}/{total_train}")
    
    train_time = time.time() - start_time
    print(f"\n训练完成，用时 {train_time:.2f}秒")
    
    # 分析结构签名
    if verbose:
        system.memory.analyze()
    
    # 测试
    print("\n" + "=" * 40)
    print("测试中...")
    print("=" * 40)
    
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    
    # 随机选择测试样本
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    start_time = time.time()
    
    for i, idx in enumerate(test_indices):
        image = test_images[idx]
        true_label = str(test_labels[idx])
        
        pred_label, conf = system.predict(image)
        
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
    print(f"训练样本: {total_train}")
    
    print(f"\n各数字准确率:")
    for digit in range(10):
        r = results[str(digit)]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  数字 {digit}: {acc:5.1f}% ({r['correct']}/{r['total']})")
    
    print(f"\n时间:")
    print(f"  训练: {train_time:.2f}秒")
    print(f"  测试: {test_time:.2f}秒")
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)
    
    return system, results


def debug_digits(system: VisionSystem, train_images: np.ndarray, train_labels: np.ndarray):
    """调试：检查每个数字的结构分析"""
    print("\n=== 数字结构调试 ===")
    
    for digit in range(10):
        idx = np.where(train_labels == digit)[0][0]
        image = train_images[idx]
        
        analysis = system.debug_image(image)
        
        print(f"\n数字 {digit}:")
        print(f"  激活点数: {analysis['n_activations']}")
        print(f"  原子: {analysis['atoms']}")
        print(f"  关系: {analysis['relations']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Structon Vision v6.3 MNIST Test')
    parser.add_argument('--train-per-class', type=int, default=10, 
                        help='每类训练样本数 (默认: 10)')
    parser.add_argument('--test', type=int, default=500, 
                        help='测试样本数 (默认: 500)')
    parser.add_argument('--quiet', action='store_true', help='减少输出')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    
    args = parser.parse_args()
    
    system, results = run_experiment(
        n_train_per_class=args.train_per_class,
        n_test=args.test,
        verbose=not args.quiet
    )
    
    if args.debug:
        train_images, train_labels, _, _ = load_mnist()
        debug_digits(system, train_images, train_labels)
