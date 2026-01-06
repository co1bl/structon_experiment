"""
Structon Vision v7.0 - Recursive Hierarchical Composition
==========================================================

核心哲学：
1. Local rule, global emergence - 局部规则，全局涌现
2. Local RM, global learn - 局部共振记忆，全局学习
3. Local grow, global adaptation - 局部生长，全局适应

架构：
- Level 0: 像素激活 (Gabor响应)
- Level 1: 原子 (边缘、曲线、端点)
- Level 2: 组合 (弧、角、线段)
- Level 3: 结构 (环、交叉)
- Level 4: 对象 (数字)

关键机制：
- promote(): 当子节点组合频繁出现时，创建新的高层Structon
- 递归组合: CompositeNode可以包含CompositeNode
- 局部共振记忆: 每一层都有自己的记忆

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Core Structon Types
# =============================================================================

class AtomType(Enum):
    """Level 1: 原子类型"""
    # 边缘方向
    EDGE_H = "E_H"      # 水平边缘
    EDGE_V = "E_V"      # 垂直边缘
    EDGE_D1 = "E_D1"    # 斜边 /
    EDGE_D2 = "E_D2"    # 斜边 \
    
    # 特殊点
    ENDPOINT = "EP"      # 端点
    JUNCTION = "JC"      # 交叉点
    
    EMPTY = "empty"


class RelationType(Enum):
    """空间关系类型"""
    ABOVE = "above"
    BELOW = "below"
    LEFT = "left"
    RIGHT = "right"
    ADJACENT = "adj"      # 相邻
    CONNECTED = "conn"    # 连接
    INSIDE = "inside"     # 内部
    SURROUND = "surround" # 环绕


# =============================================================================
# 2. Structon Node - 递归结构的核心
# =============================================================================

@dataclass
class StructonNode:
    """
    Structon节点 - 可以是原子或组合
    
    关键特性：
    - 递归结构：children可以是任何级别的StructonNode
    - 自描述：通过signature描述自己的结构
    - 位置无关：signature不包含绝对位置
    """
    level: int                              # 层级 (0=pixel, 1=atom, 2+=composite)
    node_type: str                          # 类型标识
    children: List['StructonNode'] = field(default_factory=list)
    
    # 空间信息 (WHERE)
    center: Tuple[float, float] = (0, 0)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x1,y1,x2,y2
    
    # 结构信息 (WHAT) - 位置无关
    child_types: FrozenSet[str] = field(default_factory=frozenset)
    relations: FrozenSet[str] = field(default_factory=frozenset)
    
    # 统计信息
    activation_count: int = 0
    
    def __hash__(self):
        return hash((self.level, self.node_type, self.child_types, self.relations))
    
    def __eq__(self, other):
        if not isinstance(other, StructonNode):
            return False
        return (self.level == other.level and 
                self.node_type == other.node_type and
                self.child_types == other.child_types and
                self.relations == other.relations)
    
    @property
    def signature(self) -> Tuple:
        """位置无关的结构签名"""
        return (self.level, self.node_type, self.child_types, self.relations)
    
    def is_atomic(self) -> bool:
        return self.level == 1
    
    def is_composite(self) -> bool:
        return self.level > 1


# =============================================================================
# 3. Local Resonant Memory - 每层的局部记忆
# =============================================================================

class LocalResonantMemory:
    """
    局部共振记忆
    
    每一层都有自己的记忆，存储该层级的模式
    """
    
    def __init__(self, resonance_threshold: float = 0.8):
        self.patterns: Dict[Tuple, StructonNode] = {}  # signature -> prototype
        self.frequency: Dict[Tuple, int] = defaultdict(int)  # signature -> count
        self.resonance_threshold = resonance_threshold
    
    def store(self, node: StructonNode):
        """存储模式"""
        sig = node.signature
        self.frequency[sig] += 1
        
        if sig not in self.patterns:
            self.patterns[sig] = node
    
    def resonates(self, signature: Tuple) -> bool:
        """检查是否与已知模式共振"""
        return signature in self.patterns
    
    def get_pattern(self, signature: Tuple) -> Optional[StructonNode]:
        """获取匹配的模式"""
        return self.patterns.get(signature)
    
    def get_frequency(self, signature: Tuple) -> int:
        """获取模式出现频率"""
        return self.frequency.get(signature, 0)
    
    def find_similar(self, node: StructonNode) -> Optional[Tuple[StructonNode, float]]:
        """找最相似的模式"""
        best_match = None
        best_score = 0
        
        for sig, pattern in self.patterns.items():
            score = self._compute_similarity(node, pattern)
            if score > best_score:
                best_score = score
                best_match = pattern
        
        if best_score >= self.resonance_threshold:
            return (best_match, best_score)
        return None
    
    def _compute_similarity(self, n1: StructonNode, n2: StructonNode) -> float:
        """计算两个节点的相似度"""
        if n1.level != n2.level:
            return 0.0
        
        # 类型匹配
        type_match = 1.0 if n1.node_type == n2.node_type else 0.0
        
        # 子类型匹配
        if n1.child_types and n2.child_types:
            intersection = len(n1.child_types & n2.child_types)
            union = len(n1.child_types | n2.child_types)
            child_match = intersection / union if union > 0 else 0
        else:
            child_match = 1.0 if n1.child_types == n2.child_types else 0
        
        # 关系匹配
        if n1.relations and n2.relations:
            intersection = len(n1.relations & n2.relations)
            union = len(n1.relations | n2.relations)
            rel_match = intersection / union if union > 0 else 0
        else:
            rel_match = 1.0 if n1.relations == n2.relations else 0
        
        return 0.3 * type_match + 0.35 * child_match + 0.35 * rel_match


# =============================================================================
# 4. Relation Detector - 检测节点间的空间关系
# =============================================================================

class RelationDetector:
    """检测两个节点之间的空间关系"""
    
    def __init__(self, adjacency_threshold: float = 5.0):
        self.adjacency_threshold = adjacency_threshold
    
    def detect(self, n1: StructonNode, n2: StructonNode) -> List[str]:
        """检测两个节点间的所有关系"""
        relations = []
        
        # 位置关系
        dy = n1.center[1] - n2.center[1]
        dx = n1.center[0] - n2.center[0]
        
        if abs(dy) > 3:
            relations.append("above" if dy < 0 else "below")
        if abs(dx) > 3:
            relations.append("left" if dx < 0 else "right")
        
        # 距离关系
        dist = np.sqrt(dx**2 + dy**2)
        if dist < self.adjacency_threshold:
            relations.append("adjacent")
        
        # 连接关系 (边界框重叠)
        if self._boxes_overlap(n1.bbox, n2.bbox):
            relations.append("connected")
        
        # 包含关系
        if self._is_inside(n1.bbox, n2.bbox):
            relations.append("inside")
        elif self._is_inside(n2.bbox, n1.bbox):
            relations.append("surround")
        
        return relations
    
    def _boxes_overlap(self, b1, b2) -> bool:
        """检查两个边界框是否重叠"""
        return not (b1[2] < b2[0] or b2[2] < b1[0] or 
                   b1[3] < b2[1] or b2[3] < b1[1])
    
    def _is_inside(self, inner, outer) -> bool:
        """检查inner是否在outer内部"""
        margin = 2
        return (inner[0] >= outer[0] - margin and 
                inner[2] <= outer[2] + margin and
                inner[1] >= outer[1] - margin and 
                inner[3] <= outer[3] + margin)


# =============================================================================
# 5. Structon Layer - 单层处理
# =============================================================================

class StructonLayer:
    """
    Structon层 - 处理一个层级的节点
    
    功能：
    1. 接收下层节点
    2. 尝试组合成更高层节点
    3. promote() 频繁出现的组合
    """
    
    def __init__(self, level: int, promote_threshold: int = 2):
        self.level = level
        self.memory = LocalResonantMemory()
        self.relation_detector = RelationDetector()
        self.promote_threshold = promote_threshold
        
        # 已知的组合类型
        self.known_composites: Dict[FrozenSet, str] = {}
    
    def process(self, children: List[StructonNode]) -> List[StructonNode]:
        """
        处理子节点，尝试组合成更高层节点
        """
        if not children:
            return []
        
        # 1. 找可以组合的节点组
        groups = self._find_composable_groups(children)
        
        # 2. 对每个组尝试promote
        promoted = []
        used_children = set()
        
        for group in groups:
            composite = self._try_promote(group)
            if composite:
                promoted.append(composite)
                for child in group:
                    used_children.add(id(child))
        
        # 3. 未被组合的节点也要保留
        remaining = [c for c in children if id(c) not in used_children]
        
        return promoted + remaining
    
    def _find_composable_groups(self, nodes: List[StructonNode]) -> List[List[StructonNode]]:
        """找可以组合的节点组（基于空间邻近）"""
        groups = []
        used = set()
        
        for i, n1 in enumerate(nodes):
            if i in used:
                continue
            
            group = [n1]
            used.add(i)
            
            for j, n2 in enumerate(nodes):
                if j in used:
                    continue
                
                # 检查是否邻近
                relations = self.relation_detector.detect(n1, n2)
                if 'adjacent' in relations or 'connected' in relations:
                    group.append(n2)
                    used.add(j)
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _try_promote(self, children: List[StructonNode]) -> Optional[StructonNode]:
        """
        尝试将子节点组合提升为更高层节点
        
        这是Structon的核心机制：
        - 如果这个组合在记忆中存在，直接返回
        - 如果这个组合足够频繁，创建新的复合节点
        """
        # 计算组合的结构签名
        child_types = frozenset(c.node_type for c in children)
        
        # 计算子节点间的关系
        relations = set()
        for i, c1 in enumerate(children):
            for j, c2 in enumerate(children):
                if i >= j:
                    continue
                rels = self.relation_detector.detect(c1, c2)
                for r in rels:
                    relations.add(f"{c1.node_type}_{r}_{c2.node_type}")
        
        relations = frozenset(relations)
        
        # 确定组合类型名称
        composite_type = self._get_composite_type(child_types, relations)
        
        # 创建候选节点
        candidate = StructonNode(
            level=self.level,
            node_type=composite_type,
            children=children,
            center=self._compute_center(children),
            bbox=self._compute_bbox(children),
            child_types=child_types,
            relations=relations
        )
        
        # 检查记忆
        if self.memory.resonates(candidate.signature):
            # 已知模式，直接返回
            self.memory.store(candidate)
            return candidate
        
        # 检查频率
        self.memory.store(candidate)
        if self.memory.get_frequency(candidate.signature) >= self.promote_threshold:
            # 频繁出现，正式创建
            return candidate
        
        return candidate  # 返回但可能不够稳定
    
    def _get_composite_type(self, child_types: FrozenSet[str], 
                           relations: FrozenSet[str]) -> str:
        """
        根据子类型和关系确定组合类型
        
        这里可以定义已知的组合模式
        """
        key = (child_types, relations)
        
        if key in self.known_composites:
            return self.known_composites[key]
        
        # 自动生成类型名称
        types_str = "+".join(sorted(child_types))
        
        # 检测特殊模式
        if self._is_loop_pattern(child_types, relations):
            return "LOOP"
        if self._is_corner_pattern(child_types, relations):
            return "CORNER"
        if self._is_arc_pattern(child_types, relations):
            return "ARC"
        
        return f"C_{types_str}"
    
    def _is_loop_pattern(self, child_types, relations) -> bool:
        """检测是否是环形模式"""
        # 环：多个边缘首尾相连，形成闭合
        has_edges = any('E_' in t for t in child_types)
        has_surround = any('surround' in r for r in relations)
        return has_edges and has_surround
    
    def _is_corner_pattern(self, child_types, relations) -> bool:
        """检测是否是角模式"""
        # 角：两条不同方向的边缘相连
        edge_types = [t for t in child_types if 'E_' in t]
        has_connected = any('connected' in r for r in relations)
        return len(edge_types) >= 2 and has_connected
    
    def _is_arc_pattern(self, child_types, relations) -> bool:
        """检测是否是弧形模式"""
        # 弧：同向边缘相邻排列
        edge_types = [t for t in child_types if 'E_' in t]
        has_adjacent = any('adjacent' in r for r in relations)
        return len(edge_types) >= 2 and has_adjacent
    
    def _compute_center(self, nodes: List[StructonNode]) -> Tuple[float, float]:
        """计算中心点"""
        if not nodes:
            return (0, 0)
        cx = np.mean([n.center[0] for n in nodes])
        cy = np.mean([n.center[1] for n in nodes])
        return (cx, cy)
    
    def _compute_bbox(self, nodes: List[StructonNode]) -> Tuple[float, float, float, float]:
        """计算边界框"""
        if not nodes:
            return (0, 0, 0, 0)
        x1 = min(n.bbox[0] for n in nodes)
        y1 = min(n.bbox[1] for n in nodes)
        x2 = max(n.bbox[2] for n in nodes)
        y2 = max(n.bbox[3] for n in nodes)
        return (x1, y1, x2, y2)


# =============================================================================
# 6. Atomic Detector - Level 1 检测
# =============================================================================

class AtomicDetector:
    """
    原子检测器 - 从像素到原子
    
    检测Level 1的基本元素：边缘、端点、交叉点
    """
    
    def __init__(self):
        self.orientations = [0, 45, 90, 135]
        self.filters = self._create_filters()
    
    def _create_filters(self) -> List[np.ndarray]:
        """创建方向滤波器"""
        filters = []
        size = 5
        half = size // 2
        
        for theta_deg in self.orientations:
            theta = np.deg2rad(theta_deg)
            kernel = np.zeros((size, size), dtype=np.float32)
            for y in range(-half, half + 1):
                for x in range(-half, half + 1):
                    x_t = x * np.cos(theta) + y * np.sin(theta)
                    y_t = -x * np.sin(theta) + y * np.cos(theta)
                    kernel[y + half, x + half] = np.exp(-x_t**2/4) * np.cos(2*np.pi*y_t/4)
            kernel -= kernel.mean()
            norm = np.linalg.norm(kernel)
            if norm > 1e-8:
                kernel /= norm
            filters.append(kernel)
        
        return filters
    
    def _convolve(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        kh, kw = kernel.shape
        ih, iw = image.shape
        pad = kh // 2
        padded = np.pad(image, pad, mode='constant')
        output = np.zeros((ih, iw), dtype=np.float32)
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return output
    
    def detect(self, image: np.ndarray) -> List[StructonNode]:
        """检测原子节点"""
        h, w = image.shape
        
        # 1. 计算各方向响应
        responses = [np.abs(self._convolve(image, f)) for f in self.filters]
        
        # 归一化
        max_resp = max(r.max() for r in responses)
        if max_resp < 1e-8:
            return []
        responses = [r / max_resp for r in responses]
        
        # 2. 骨架化以找端点和交叉点
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        
        # 3. 划分区域并检测边缘类型
        atoms = []
        cell_size = 6  # 每个cell的大小
        
        for y in range(0, h - cell_size, cell_size // 2):  # 重叠
            for x in range(0, w - cell_size, cell_size // 2):
                y2 = min(y + cell_size, h)
                x2 = min(x + cell_size, w)
                
                # 检查该区域的响应
                region_responses = [r[y:y2, x:x2].mean() for r in responses]
                max_idx = np.argmax(region_responses)
                max_val = region_responses[max_idx]
                
                if max_val < 0.2:
                    continue
                
                # 确定原子类型
                atom_type = self._classify_edge(max_idx, region_responses)
                
                # 检查是否有特殊点
                has_endpoint = any(y <= py < y2 and x <= px < x2 for py, px in endpoints)
                has_junction = any(y <= py < y2 and x <= px < x2 for py, px in junctions)
                
                if has_junction:
                    atom_type = "JC"
                elif has_endpoint:
                    atom_type = "EP"
                
                atoms.append(StructonNode(
                    level=1,
                    node_type=atom_type,
                    center=((x + x2) / 2, (y + y2) / 2),
                    bbox=(x, y, x2, y2),
                    activation_count=1
                ))
        
        return atoms
    
    def _classify_edge(self, max_idx: int, responses: List[float]) -> str:
        """根据响应分类边缘类型"""
        types = ["E_H", "E_D1", "E_V", "E_D2"]  # 对应 0, 45, 90, 135 度
        return types[max_idx]
    
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


# =============================================================================
# 7. Hierarchical Structon System
# =============================================================================

class HierarchicalStructonSystem:
    """
    层次化Structon系统
    
    多层处理：
    Level 1: Atomic (边缘、端点)
    Level 2: Composite (弧、角)
    Level 3: Structure (环、交叉结构)
    Level 4: Object (数字)
    """
    
    def __init__(self, n_levels: int = 4):
        self.n_levels = n_levels
        self.atomic_detector = AtomicDetector()
        self.layers = [StructonLayer(level=i+2) for i in range(n_levels - 1)]
        
        # 顶层记忆 (存储完整对象)
        self.object_memory: Dict[str, List[StructonNode]] = defaultdict(list)
    
    def process(self, image: np.ndarray) -> StructonNode:
        """
        处理图像，返回最高层的结构表示
        """
        # Level 1: 检测原子
        atoms = self.atomic_detector.detect(image)
        
        if not atoms:
            return StructonNode(level=0, node_type="EMPTY")
        
        # Level 2+: 递归组合
        current_nodes = atoms
        
        for layer in self.layers:
            if len(current_nodes) < 2:
                break
            current_nodes = layer.process(current_nodes)
        
        # 创建顶层对象节点
        object_node = self._create_object_node(current_nodes)
        
        return object_node
    
    def _create_object_node(self, nodes: List[StructonNode]) -> StructonNode:
        """创建顶层对象节点"""
        if not nodes:
            return StructonNode(level=self.n_levels, node_type="EMPTY")
        
        # 收集所有节点的类型
        all_types = set()
        all_relations = set()
        
        for node in nodes:
            all_types.add(node.node_type)
            all_relations.update(node.relations)
        
        # 计算节点间的顶层关系
        relation_detector = RelationDetector()
        for i, n1 in enumerate(nodes):
            for j, n2 in enumerate(nodes):
                if i >= j:
                    continue
                rels = relation_detector.detect(n1, n2)
                for r in rels:
                    all_relations.add(f"{n1.node_type}_{r}_{n2.node_type}")
        
        # 确定对象类型
        obj_type = self._infer_object_type(all_types, all_relations, nodes)
        
        return StructonNode(
            level=self.n_levels,
            node_type=obj_type,
            children=nodes,
            center=self._compute_center(nodes),
            bbox=self._compute_bbox(nodes),
            child_types=frozenset(all_types),
            relations=frozenset(all_relations)
        )
    
    def _infer_object_type(self, types: Set[str], relations: Set[str], 
                          nodes: List[StructonNode]) -> str:
        """推断对象类型"""
        # 统计特征
        n_loops = sum(1 for n in nodes if 'LOOP' in n.node_type)
        n_endpoints = sum(1 for n in nodes if n.node_type == 'EP')
        n_junctions = sum(1 for n in nodes if n.node_type == 'JC')
        has_corner = any('CORNER' in n.node_type for n in nodes)
        
        # 基于特征推断
        if n_loops == 2:
            return "DOUBLE_LOOP"  # 可能是8
        elif n_loops == 1:
            return "SINGLE_LOOP"  # 可能是0, 6, 9
        elif n_junctions > 0 and has_corner:
            return "JUNCTION_CORNER"  # 可能是4
        elif n_endpoints == 2 and n_junctions == 0:
            return "SIMPLE_STROKE"  # 可能是1, 7
        else:
            return "COMPLEX"
    
    def _compute_center(self, nodes: List[StructonNode]) -> Tuple[float, float]:
        if not nodes:
            return (0, 0)
        return (np.mean([n.center[0] for n in nodes]),
                np.mean([n.center[1] for n in nodes]))
    
    def _compute_bbox(self, nodes: List[StructonNode]) -> Tuple[float, float, float, float]:
        if not nodes:
            return (0, 0, 0, 0)
        return (min(n.bbox[0] for n in nodes),
                min(n.bbox[1] for n in nodes),
                max(n.bbox[2] for n in nodes),
                max(n.bbox[3] for n in nodes))
    
    def train(self, image: np.ndarray, label: str):
        """训练：存储对象模式"""
        obj_node = self.process(image)
        self.object_memory[label].append(obj_node)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测：找最匹配的对象"""
        obj_node = self.process(image)
        
        best_label = None
        best_score = 0.0
        
        for label, patterns in self.object_memory.items():
            for pattern in patterns:
                score = self._compute_similarity(obj_node, pattern)
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def _compute_similarity(self, n1: StructonNode, n2: StructonNode) -> float:
        """计算两个对象节点的相似度"""
        # 类型匹配
        type_match = 1.0 if n1.node_type == n2.node_type else 0.3
        
        # 子类型匹配
        if n1.child_types and n2.child_types:
            intersection = len(n1.child_types & n2.child_types)
            union = len(n1.child_types | n2.child_types)
            child_match = intersection / union if union > 0 else 0
        else:
            child_match = 0.5
        
        # 关系匹配
        if n1.relations and n2.relations:
            # 只比较关系类型，不比较具体节点
            r1_types = set(r.split('_')[1] for r in n1.relations if '_' in r)
            r2_types = set(r.split('_')[1] for r in n2.relations if '_' in r)
            intersection = len(r1_types & r2_types)
            union = len(r1_types | r2_types)
            rel_match = intersection / union if union > 0 else 0
        else:
            rel_match = 0.5
        
        return 0.4 * type_match + 0.3 * child_match + 0.3 * rel_match
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试：显示处理过程"""
        atoms = self.atomic_detector.detect(image)
        
        # 统计原子类型
        atom_types = defaultdict(int)
        for a in atoms:
            atom_types[a.node_type] += 1
        
        # 处理各层
        layer_info = []
        current_nodes = atoms
        
        for i, layer in enumerate(self.layers):
            if len(current_nodes) < 2:
                break
            current_nodes = layer.process(current_nodes)
            layer_info.append({
                'level': i + 2,
                'n_nodes': len(current_nodes),
                'types': [n.node_type for n in current_nodes]
            })
        
        # 顶层
        obj_node = self._create_object_node(current_nodes)
        
        return {
            'n_atoms': len(atoms),
            'atom_types': dict(atom_types),
            'layers': layer_info,
            'object_type': obj_node.node_type,
            'child_types': list(obj_node.child_types),
            'n_relations': len(obj_node.relations)
        }


# =============================================================================
# 8. MNIST Experiment
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
    print("Structon Vision v7.0 - Hierarchical Recursive Composition")
    print("=" * 60)
    print("\n核心原理：")
    print("  1. Local rule, global emergence")
    print("  2. Local RM, global learn")
    print("  3. Local grow, global adaptation")
    print("\n层次结构：")
    print("  Level 1: Atoms (边缘、端点、交叉)")
    print("  Level 2: Composites (弧、角)")
    print("  Level 3: Structures (环、交叉结构)")
    print("  Level 4: Objects (数字)")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = HierarchicalStructonSystem(n_levels=4)
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    # 分析每个数字的模式
    if verbose:
        print("\n=== 学习到的模式 ===")
        for label in sorted(system.object_memory.keys()):
            patterns = system.object_memory[label]
            types = defaultdict(int)
            for p in patterns:
                types[p.node_type] += 1
            print(f"数字 {label}: {dict(types)}")
    
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
    print("\n=== 调试: 层次结构 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = HierarchicalStructonSystem(n_levels=4)
    
    for digit in range(10):
        print(f"\n{'='*40}")
        print(f"数字 {digit}")
        print('='*40)
        
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"\n  样本 {i+1}:")
            print(f"    原子: {info['n_atoms']} 个, 类型: {info['atom_types']}")
            for layer in info['layers']:
                print(f"    Level {layer['level']}: {layer['n_nodes']} 节点, 类型: {layer['types'][:5]}...")
            print(f"    对象类型: {info['object_type']}")
            print(f"    子类型: {info['child_types'][:5]}...")


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
