"""
Structon Vision v7.6 - Comprehensive Basic Features
====================================================

核心思想：
- 人类V1皮层有所有可能的基础特征，冗余存在
- LRM自然会选择哪些特征重要
- 不用担心冗余，让涌现机制处理

Level 0 特征大全：
1. 拓扑特征
   - holes, endpoints, junctions (数量)
   - endpoint/junction 位置 (9个区域)

2. 边缘方向特征
   - 每个区域的主方向 (H, V, D1, D2)
   - 全局方向直方图

3. 曲率特征
   - 角点检测 (sharp corner)
   - 曲线检测 (smooth curve)
   - 角点位置

4. 笔画终止特征
   - 端点的笔画方向 (向上/下/左/右 终止)
   - 端点邻域的边缘方向

5. 特殊结构特征
   - 顶部水平线 (7, 5)
   - 底部水平线 (2, 5)
   - 中心垂直线 (1)
   - 中心交叉 (4, 8)

6. 曲线开口方向
   - 开口朝左/右/上/下
   - 基于质心偏移

7. 连通性特征
   - 是否闭合
   - 连通分量数

8. 几何特征
   - 宽高比
   - 填充密度
   - 重心位置

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
# 1. Comprehensive Feature Set
# =============================================================================

@dataclass
class ComprehensiveFeatures:
    """所有可能的基础特征"""
    
    # === 拓扑特征 ===
    n_holes: int = 0
    n_endpoints: int = 0
    n_junctions: int = 0
    
    # 端点位置 (9区域)
    ep_top_left: bool = False
    ep_top: bool = False
    ep_top_right: bool = False
    ep_left: bool = False
    ep_center: bool = False
    ep_right: bool = False
    ep_bottom_left: bool = False
    ep_bottom: bool = False
    ep_bottom_right: bool = False
    
    # 交叉点位置
    jc_top: bool = False
    jc_center: bool = False
    jc_bottom: bool = False
    
    # === 边缘方向特征 ===
    # 全局方向统计
    n_horizontal: int = 0
    n_vertical: int = 0
    n_diagonal1: int = 0  # /
    n_diagonal2: int = 0  # \
    dominant_direction: str = 'mixed'
    
    # 特定区域方向
    top_horizontal: bool = False      # 顶部有水平线 (7, 5)
    bottom_horizontal: bool = False   # 底部有水平线 (2)
    center_vertical: bool = False     # 中心有垂直线 (1)
    center_horizontal: bool = False   # 中心有水平线 (4)
    
    # === 曲率特征 ===
    n_sharp_corners: int = 0          # 尖角数量 (7)
    n_smooth_curves: int = 0          # 平滑曲线数量 (2, 3, 6, 8, 9, 0)
    has_top_corner: bool = False      # 顶部有角 (7)
    has_bottom_corner: bool = False   # 底部有角
    
    # === 笔画终止特征 ===
    # 端点的笔画方向
    ep_terminates_up: bool = False
    ep_terminates_down: bool = False
    ep_terminates_left: bool = False
    ep_terminates_right: bool = False
    
    # 顶部端点方向
    top_ep_direction: str = ''        # 'up', 'down', 'left', 'right'
    # 底部端点方向
    bottom_ep_direction: str = ''
    
    # === 曲线开口特征 ===
    has_curve_open_left: bool = False   # 3, 部分9
    has_curve_open_right: bool = False  # 2, 部分6
    has_curve_open_up: bool = False     # 部分结构
    has_curve_open_down: bool = False   # 部分结构
    
    # === 连通性特征 ===
    is_closed: bool = False             # 是否闭合 (0, 8)
    n_components: int = 1               # 连通分量数
    
    # === 几何特征 ===
    aspect_ratio: str = 'normal'        # 'tall', 'wide', 'normal'
    fill_density: str = 'medium'        # 'sparse', 'medium', 'dense'
    center_of_mass_x: str = 'center'    # 'left', 'center', 'right'
    center_of_mass_y: str = 'center'    # 'top', 'center', 'bottom'
    
    # === 特殊模式 ===
    has_loop_top: bool = False          # 9
    has_loop_bottom: bool = False       # 6
    has_loop_center: bool = False       # 8 的两个环
    has_tail_top: bool = False          # 6 的尾巴在上
    has_tail_bottom: bool = False       # 9 的尾巴在下
    
    def to_signature_tuple(self) -> Tuple:
        """转换为可哈希的签名"""
        return (
            # 拓扑
            self.n_holes, self.n_endpoints, self.n_junctions,
            # 端点位置
            self.ep_top_left, self.ep_top, self.ep_top_right,
            self.ep_left, self.ep_center, self.ep_right,
            self.ep_bottom_left, self.ep_bottom, self.ep_bottom_right,
            # 交叉位置
            self.jc_top, self.jc_center, self.jc_bottom,
            # 边缘方向
            self.dominant_direction,
            self.top_horizontal, self.bottom_horizontal,
            self.center_vertical, self.center_horizontal,
            # 曲率
            self.n_sharp_corners, self.n_smooth_curves,
            self.has_top_corner, self.has_bottom_corner,
            # 终止方向
            self.top_ep_direction, self.bottom_ep_direction,
            # 曲线开口
            self.has_curve_open_left, self.has_curve_open_right,
            # 连通性
            self.is_closed,
            # 几何
            self.aspect_ratio, self.fill_density,
            self.center_of_mass_x, self.center_of_mass_y,
            # 特殊模式
            self.has_loop_top, self.has_loop_bottom,
            self.has_tail_top, self.has_tail_bottom,
        )
    
    def to_string(self) -> str:
        """生成可读的签名字符串"""
        parts = []
        
        # 拓扑
        parts.append(f"H{self.n_holes}E{self.n_endpoints}J{self.n_junctions}")
        
        # 端点位置
        ep_pos = []
        if self.ep_top or self.ep_top_left or self.ep_top_right:
            ep_pos.append("T")
        if self.ep_center or self.ep_left or self.ep_right:
            ep_pos.append("M")
        if self.ep_bottom or self.ep_bottom_left or self.ep_bottom_right:
            ep_pos.append("B")
        if ep_pos:
            parts.append(f"EP{''.join(ep_pos)}")
        
        # 交叉位置
        if self.jc_center:
            parts.append("JCctr")
        
        # 特殊结构
        if self.top_horizontal:
            parts.append("Htop")
        if self.bottom_horizontal:
            parts.append("Hbot")
        if self.center_vertical:
            parts.append("Vctr")
        
        # 曲率
        if self.n_sharp_corners > 0:
            parts.append(f"SC{self.n_sharp_corners}")
        if self.n_smooth_curves > 0:
            parts.append(f"CV{self.n_smooth_curves}")
        
        # 曲线开口
        if self.has_curve_open_left:
            parts.append("OpnL")
        if self.has_curve_open_right:
            parts.append("OpnR")
        
        # 闭合
        if self.is_closed:
            parts.append("Closed")
        
        # 特殊模式
        if self.has_loop_top:
            parts.append("LoopT")
        if self.has_loop_bottom:
            parts.append("LoopB")
        if self.has_tail_top:
            parts.append("TailT")
        if self.has_tail_bottom:
            parts.append("TailB")
        
        return "_".join(parts)


# =============================================================================
# 2. Comprehensive Feature Extractor
# =============================================================================

class ComprehensiveFeatureExtractor:
    """提取所有可能的基础特征"""
    
    def __init__(self):
        self.regions_3x3 = [
            ('TL', 0, 1/3, 0, 1/3),
            ('T', 0, 1/3, 1/3, 2/3),
            ('TR', 0, 1/3, 2/3, 1),
            ('L', 1/3, 2/3, 0, 1/3),
            ('C', 1/3, 2/3, 1/3, 2/3),
            ('R', 1/3, 2/3, 2/3, 1),
            ('BL', 2/3, 1, 0, 1/3),
            ('B', 2/3, 1, 1/3, 2/3),
            ('BR', 2/3, 1, 2/3, 1),
        ]
    
    def extract(self, image: np.ndarray) -> ComprehensiveFeatures:
        """提取所有特征"""
        f = ComprehensiveFeatures()
        h, w = image.shape
        
        # === 预处理 ===
        binary = (image > 0.3).astype(np.uint8)
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        
        # === 1. 拓扑特征 ===
        f.n_holes = self._count_holes(image)
        f.n_endpoints = len(endpoints)
        f.n_junctions = len(junctions)
        
        # 端点位置
        for y, x in endpoints:
            ry, rx = y / h, x / w
            if ry < 1/3:
                if rx < 1/3: f.ep_top_left = True
                elif rx < 2/3: f.ep_top = True
                else: f.ep_top_right = True
            elif ry < 2/3:
                if rx < 1/3: f.ep_left = True
                elif rx < 2/3: f.ep_center = True
                else: f.ep_right = True
            else:
                if rx < 1/3: f.ep_bottom_left = True
                elif rx < 2/3: f.ep_bottom = True
                else: f.ep_bottom_right = True
        
        # 交叉点位置
        for y, x in junctions:
            ry = y / h
            if ry < 1/3: f.jc_top = True
            elif ry < 2/3: f.jc_center = True
            else: f.jc_bottom = True
        
        # === 2. 边缘方向特征 ===
        edge_counts = {'H': 0, 'V': 0, 'D1': 0, 'D2': 0}
        region_directions = {}
        
        for name, y1f, y2f, x1f, x2f in self.regions_3x3:
            y1, y2 = int(y1f * h), int(y2f * h)
            x1, x2 = int(x1f * w), int(x2f * w)
            region = image[y1:y2, x1:x2]
            
            if np.mean(region > 0.3) < 0.05:
                continue
            
            direction = self._detect_direction(region)
            if direction:
                region_directions[name] = direction
                edge_counts[direction] += 1
        
        f.n_horizontal = edge_counts['H']
        f.n_vertical = edge_counts['V']
        f.n_diagonal1 = edge_counts['D1']
        f.n_diagonal2 = edge_counts['D2']
        
        # 主方向
        max_dir = max(edge_counts, key=edge_counts.get)
        if edge_counts[max_dir] > sum(edge_counts.values()) * 0.4:
            f.dominant_direction = max_dir
        else:
            f.dominant_direction = 'mixed'
        
        # 特定区域方向
        f.top_horizontal = region_directions.get('T') == 'H' or region_directions.get('TL') == 'H' or region_directions.get('TR') == 'H'
        f.bottom_horizontal = region_directions.get('B') == 'H' or region_directions.get('BL') == 'H' or region_directions.get('BR') == 'H'
        f.center_vertical = region_directions.get('C') == 'V'
        f.center_horizontal = region_directions.get('C') == 'H'
        
        # === 3. 曲率特征 ===
        corners, curves = self._detect_curvature(skeleton, image)
        f.n_sharp_corners = len(corners)
        f.n_smooth_curves = len(curves)
        
        for y, x in corners:
            if y < h / 3:
                f.has_top_corner = True
            elif y > 2 * h / 3:
                f.has_bottom_corner = True
        
        # === 4. 笔画终止特征 ===
        for y, x in endpoints:
            direction = self._get_stroke_direction_at_endpoint(skeleton, y, x)
            if direction == 'up':
                f.ep_terminates_up = True
            elif direction == 'down':
                f.ep_terminates_down = True
            elif direction == 'left':
                f.ep_terminates_left = True
            elif direction == 'right':
                f.ep_terminates_right = True
            
            # 顶部/底部端点方向
            if y < h / 3:
                f.top_ep_direction = direction
            elif y > 2 * h / 3:
                f.bottom_ep_direction = direction
        
        # === 5. 曲线开口特征 ===
        curve_openings = self._detect_curve_openings(image)
        f.has_curve_open_left = 'left' in curve_openings
        f.has_curve_open_right = 'right' in curve_openings
        f.has_curve_open_up = 'up' in curve_openings
        f.has_curve_open_down = 'down' in curve_openings
        
        # === 6. 连通性特征 ===
        f.is_closed = (f.n_endpoints == 0 and f.n_holes >= 1)
        f.n_components = self._count_components(binary)
        
        # === 7. 几何特征 ===
        # 边界框
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            
            if bbox_h > bbox_w * 1.5:
                f.aspect_ratio = 'tall'
            elif bbox_w > bbox_h * 1.5:
                f.aspect_ratio = 'wide'
            else:
                f.aspect_ratio = 'normal'
        
        # 填充密度
        density = np.sum(binary) / (h * w)
        if density < 0.1:
            f.fill_density = 'sparse'
        elif density > 0.25:
            f.fill_density = 'dense'
        else:
            f.fill_density = 'medium'
        
        # 重心
        if np.sum(binary) > 0:
            cy, cx = np.argwhere(binary).mean(axis=0)
            f.center_of_mass_x = 'left' if cx < w * 0.4 else ('right' if cx > w * 0.6 else 'center')
            f.center_of_mass_y = 'top' if cy < h * 0.4 else ('bottom' if cy > h * 0.6 else 'center')
        
        # === 8. 特殊模式 ===
        # 检测环的位置 (基于空洞和质心)
        if f.n_holes >= 1:
            loop_positions = self._detect_loop_positions(image)
            f.has_loop_top = 'top' in loop_positions
            f.has_loop_bottom = 'bottom' in loop_positions
            f.has_loop_center = 'center' in loop_positions
        
        # 检测尾巴 (有环 + 端点)
        if f.n_holes >= 1 and f.n_endpoints >= 1:
            if f.ep_top or f.ep_top_left or f.ep_top_right:
                f.has_tail_top = True
            if f.ep_bottom or f.ep_bottom_left or f.ep_bottom_right:
                f.has_tail_bottom = True
        
        return f
    
    def _detect_direction(self, region: np.ndarray) -> Optional[str]:
        """检测区域主方向"""
        if region.shape[0] < 3 or region.shape[1] < 3:
            return None
        
        gy = np.abs(region[2:, :] - region[:-2, :]).sum()
        gx = np.abs(region[:, 2:] - region[:, :-2]).sum()
        gd1 = np.abs(region[2:, 2:] - region[:-2, :-2]).sum()
        gd2 = np.abs(region[2:, :-2] - region[:-2, 2:]).sum()
        
        gradients = {'V': gx, 'H': gy, 'D1': gd1, 'D2': gd2}
        max_key = max(gradients, key=gradients.get)
        
        if gradients[max_key] < 0.5:
            return None
        
        return max_key
    
    def _detect_curvature(self, skeleton: np.ndarray, image: np.ndarray) -> Tuple[List, List]:
        """检测角点和曲线"""
        h, w = skeleton.shape
        corners = []
        curves = []
        
        for y in range(2, h - 2):
            for x in range(2, w - 2):
                if skeleton[y, x] == 0:
                    continue
                
                # 获取3x3邻域
                neighbors = skeleton[y-1:y+2, x-1:x+2].copy()
                neighbors[1, 1] = 0
                n_neighbors = np.sum(neighbors)
                
                if n_neighbors == 2:
                    # 检查是角还是曲线
                    positions = np.argwhere(neighbors)
                    if len(positions) == 2:
                        p1, p2 = positions
                        # 计算两个邻居之间的角度
                        v1 = p1 - np.array([1, 1])
                        v2 = p2 - np.array([1, 1])
                        dot = np.dot(v1, v2)
                        
                        if dot < -0.5:  # 接近180度 - 直线
                            pass
                        elif dot > 0.5:  # 接近0度 - 尖角
                            corners.append((y, x))
                        else:  # 90度左右 - 曲线或角
                            # 检查是否是平滑曲线
                            if self._is_smooth_curve(image, y, x):
                                curves.append((y, x))
                            else:
                                corners.append((y, x))
        
        return corners, curves
    
    def _is_smooth_curve(self, image: np.ndarray, y: int, x: int) -> bool:
        """检查点是否是平滑曲线的一部分"""
        h, w = image.shape
        
        # 获取更大的邻域检查平滑度
        y1, y2 = max(0, y - 3), min(h, y + 4)
        x1, x2 = max(0, x - 3), min(w, x + 4)
        
        region = image[y1:y2, x1:x2]
        
        # 计算梯度变化 - 平滑曲线梯度变化小
        if region.shape[0] < 3 or region.shape[1] < 3:
            return False
        
        gy = np.abs(region[1:, :] - region[:-1, :])
        gx = np.abs(region[:, 1:] - region[:, :-1])
        
        # 梯度变化的标准差 - 低表示平滑
        smoothness = np.std(gy) + np.std(gx)
        
        return smoothness < 0.3
    
    def _get_stroke_direction_at_endpoint(self, skeleton: np.ndarray, y: int, x: int) -> str:
        """获取端点处笔画的方向"""
        h, w = skeleton.shape
        
        # 找到相邻的骨架点
        for dy, dx, direction in [(-1, 0, 'down'), (1, 0, 'up'), 
                                   (0, -1, 'right'), (0, 1, 'left'),
                                   (-1, -1, 'down-right'), (-1, 1, 'down-left'),
                                   (1, -1, 'up-right'), (1, 1, 'up-left')]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                # 笔画从(ny,nx)指向(y,x)，所以终止方向是反向
                if direction in ['up', 'up-left', 'up-right']:
                    return 'up'
                elif direction in ['down', 'down-left', 'down-right']:
                    return 'down'
                elif direction in ['left']:
                    return 'left'
                elif direction in ['right']:
                    return 'right'
        
        return ''
    
    def _detect_curve_openings(self, image: np.ndarray) -> Set[str]:
        """检测曲线开口方向"""
        openings = set()
        h, w = image.shape
        binary = (image > 0.3).astype(np.uint8)
        
        # 分析上下左右四个方向的空白
        # 如果某侧大部分是空白，但中心有内容，可能是开口
        
        # 检查左侧开口 (右侧有内容，左侧空)
        left_half = binary[:, :w//2]
        right_half = binary[:, w//2:]
        if np.sum(right_half) > np.sum(left_half) * 2:
            openings.add('left')
        if np.sum(left_half) > np.sum(right_half) * 2:
            openings.add('right')
        
        # 检查上下开口
        top_half = binary[:h//2, :]
        bottom_half = binary[h//2:, :]
        if np.sum(bottom_half) > np.sum(top_half) * 2:
            openings.add('up')
        if np.sum(top_half) > np.sum(bottom_half) * 2:
            openings.add('down')
        
        return openings
    
    def _detect_loop_positions(self, image: np.ndarray) -> Set[str]:
        """检测环的位置"""
        positions = set()
        h, w = image.shape
        
        # 简单方法：找到内部空白区域的位置
        binary = (image > 0.3).astype(np.uint8)
        
        # 从边界flood fill找到外部
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary
        
        external = np.zeros_like(padded, dtype=bool)
        queue = [(0, 0)]
        external[0, 0] = True
        
        while queue:
            cy, cx = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                    if padded[ny, nx] == 0 and not external[ny, nx]:
                        external[ny, nx] = True
                        queue.append((ny, nx))
        
        # 找内部空白
        internal = (padded == 0) & (~external)
        
        # 确定位置
        internal_points = np.argwhere(internal[1:-1, 1:-1])
        if len(internal_points) > 0:
            mean_y = internal_points[:, 0].mean()
            if mean_y < h / 3:
                positions.add('top')
            elif mean_y > 2 * h / 3:
                positions.add('bottom')
            else:
                positions.add('center')
        
        return positions
    
    def _count_components(self, binary: np.ndarray) -> int:
        """计算连通分量数"""
        h, w = binary.shape
        visited = np.zeros_like(binary, dtype=bool)
        n_components = 0
        
        for y in range(h):
            for x in range(w):
                if binary[y, x] and not visited[y, x]:
                    # BFS
                    queue = [(y, x)]
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.pop(0)
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dy == 0 and dx == 0:
                                    continue
                                ny, nx = cy + dy, cx + dx
                                if 0 <= ny < h and 0 <= nx < w:
                                    if binary[ny, nx] and not visited[ny, nx]:
                                        visited[ny, nx] = True
                                        queue.append((ny, nx))
                    n_components += 1
        
        return n_components
    
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
# 3. Multi-Level Memory with Comprehensive Features
# =============================================================================

class LayerMemory:
    """层级记忆"""
    
    def __init__(self, level: int, emergence_threshold: int = 2):
        self.level = level
        self.emergence_threshold = emergence_threshold
        self.patterns: Dict[str, int] = defaultdict(int)  # signature → frequency
        self.pattern_ids: Dict[str, str] = {}  # signature → pattern_id
        self.counter = 0
    
    def observe(self, signature: str) -> Optional[str]:
        """观察签名，返回涌现的模式ID"""
        self.patterns[signature] += 1
        
        if self.patterns[signature] >= self.emergence_threshold:
            if signature not in self.pattern_ids:
                self.counter += 1
                self.pattern_ids[signature] = f"L{self.level}_{self.counter}"
            return self.pattern_ids[signature]
        
        return None
    
    def get_stats(self) -> List[Tuple[str, int]]:
        return sorted(self.patterns.items(), key=lambda x: -x[1])[:15]


class StructonVisionSystem:
    """Structon视觉系统 v7.6"""
    
    def __init__(self, emergence_threshold: int = 2):
        self.extractor = ComprehensiveFeatureExtractor()
        self.memory = LayerMemory(1, emergence_threshold)
        
        # 标签记忆
        self.label_memory: Dict[str, List[ComprehensiveFeatures]] = defaultdict(list)
    
    def train(self, image: np.ndarray, label: str):
        """训练"""
        features = self.extractor.extract(image)
        signature = features.to_string()
        self.memory.observe(signature)
        self.label_memory[label].append(features)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        features = self.extractor.extract(image)
        
        best_label = None
        best_score = 0.0
        
        for label, stored_list in self.label_memory.items():
            for stored in stored_list:
                score = self._compute_similarity(features, stored)
                if score > best_score:
                    best_score = score
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def _compute_similarity(self, f1: ComprehensiveFeatures, f2: ComprehensiveFeatures) -> float:
        """计算特征相似度"""
        score = 0.0
        weights_total = 0.0
        
        # === 拓扑 (最重要) ===
        # 空洞数
        if f1.n_holes == f2.n_holes:
            score += 0.20
        weights_total += 0.20
        
        # 端点数
        ep_diff = abs(f1.n_endpoints - f2.n_endpoints)
        score += 0.10 / (1 + ep_diff)
        weights_total += 0.10
        
        # 交叉点数
        jc_diff = abs(f1.n_junctions - f2.n_junctions)
        score += 0.08 / (1 + jc_diff)
        weights_total += 0.08
        
        # === 端点位置 ===
        ep_match = 0
        ep_total = 0
        for attr in ['ep_top', 'ep_top_left', 'ep_top_right', 
                     'ep_center', 'ep_left', 'ep_right',
                     'ep_bottom', 'ep_bottom_left', 'ep_bottom_right']:
            if getattr(f1, attr) == getattr(f2, attr):
                ep_match += 1
            ep_total += 1
        score += 0.12 * (ep_match / ep_total)
        weights_total += 0.12
        
        # === 交叉位置 ===
        jc_match = 0
        for attr in ['jc_top', 'jc_center', 'jc_bottom']:
            if getattr(f1, attr) == getattr(f2, attr):
                jc_match += 1
        score += 0.06 * (jc_match / 3)
        weights_total += 0.06
        
        # === 特殊结构 ===
        struct_match = 0
        for attr in ['top_horizontal', 'bottom_horizontal', 'center_vertical', 'center_horizontal']:
            if getattr(f1, attr) == getattr(f2, attr):
                struct_match += 1
        score += 0.10 * (struct_match / 4)
        weights_total += 0.10
        
        # === 曲率特征 ===
        if f1.n_sharp_corners == f2.n_sharp_corners:
            score += 0.05
        weights_total += 0.05
        
        curve_diff = abs(f1.n_smooth_curves - f2.n_smooth_curves)
        score += 0.05 / (1 + curve_diff)
        weights_total += 0.05
        
        # === 笔画终止 ===
        term_match = 0
        for attr in ['ep_terminates_up', 'ep_terminates_down', 'ep_terminates_left', 'ep_terminates_right']:
            if getattr(f1, attr) == getattr(f2, attr):
                term_match += 1
        score += 0.06 * (term_match / 4)
        weights_total += 0.06
        
        # === 曲线开口 ===
        open_match = 0
        for attr in ['has_curve_open_left', 'has_curve_open_right']:
            if getattr(f1, attr) == getattr(f2, attr):
                open_match += 1
        score += 0.06 * (open_match / 2)
        weights_total += 0.06
        
        # === 闭合性 ===
        if f1.is_closed == f2.is_closed:
            score += 0.05
        weights_total += 0.05
        
        # === 特殊模式 ===
        special_match = 0
        for attr in ['has_loop_top', 'has_loop_bottom', 'has_tail_top', 'has_tail_bottom']:
            if getattr(f1, attr) == getattr(f2, attr):
                special_match += 1
        score += 0.07 * (special_match / 4)
        weights_total += 0.07
        
        return score / weights_total if weights_total > 0 else 0
    
    def debug_image(self, image: np.ndarray) -> dict:
        """调试"""
        f = self.extractor.extract(image)
        return {
            'signature': f.to_string(),
            'topology': f"H{f.n_holes}E{f.n_endpoints}J{f.n_junctions}",
            'ep_pos': {
                'top': f.ep_top or f.ep_top_left or f.ep_top_right,
                'center': f.ep_center,
                'bottom': f.ep_bottom or f.ep_bottom_left or f.ep_bottom_right
            },
            'jc_pos': {'top': f.jc_top, 'center': f.jc_center, 'bottom': f.jc_bottom},
            'structure': {
                'top_H': f.top_horizontal,
                'bottom_H': f.bottom_horizontal,
                'center_V': f.center_vertical
            },
            'curvature': f"corners={f.n_sharp_corners}, curves={f.n_smooth_curves}",
            'termination': {
                'up': f.ep_terminates_up, 'down': f.ep_terminates_down,
                'left': f.ep_terminates_left, 'right': f.ep_terminates_right
            },
            'curve_open': {
                'left': f.has_curve_open_left, 'right': f.has_curve_open_right
            },
            'special': {
                'loop_top': f.has_loop_top, 'loop_bottom': f.has_loop_bottom,
                'tail_top': f.has_tail_top, 'tail_bottom': f.has_tail_bottom,
                'closed': f.is_closed
            }
        }
    
    def print_analysis(self):
        """打印分析"""
        print("\n=== 涌现的模式 ===")
        for sig, freq in self.memory.get_stats():
            print(f"  {sig}: {freq}")
        
        print("\n=== 各数字的特征分布 ===")
        for label in sorted(self.label_memory.keys()):
            features_list = self.label_memory[label]
            
            # 统计关键特征
            holes = Counter(f.n_holes for f in features_list)
            closed = sum(1 for f in features_list if f.is_closed)
            top_h = sum(1 for f in features_list if f.top_horizontal)
            center_v = sum(1 for f in features_list if f.center_vertical)
            open_left = sum(1 for f in features_list if f.has_curve_open_left)
            open_right = sum(1 for f in features_list if f.has_curve_open_right)
            loop_top = sum(1 for f in features_list if f.has_loop_top)
            loop_bottom = sum(1 for f in features_list if f.has_loop_bottom)
            
            print(f"\n数字 {label}:")
            print(f"  空洞: {dict(holes)}")
            print(f"  闭合: {closed}/{len(features_list)}")
            print(f"  顶部水平: {top_h}, 中心垂直: {center_v}")
            print(f"  开口左: {open_left}, 开口右: {open_right}")
            print(f"  环顶: {loop_top}, 环底: {loop_bottom}")


# =============================================================================
# 4. MNIST Experiment
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
    print("Structon Vision v7.6 - Comprehensive Basic Features")
    print("=" * 60)
    print("\n完整特征集:")
    print("  1. 拓扑: holes, endpoints, junctions + 位置")
    print("  2. 边缘方向: 全局 + 特定区域 (顶部水平, 中心垂直)")
    print("  3. 曲率: 角点, 平滑曲线")
    print("  4. 笔画终止: 端点方向")
    print("  5. 曲线开口: 左/右/上/下")
    print("  6. 连通性: 闭合, 分量数")
    print("  7. 几何: 宽高比, 密度, 重心")
    print("  8. 特殊模式: 环位置, 尾巴位置")
    
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


def debug_digits(n=2):
    print("\n=== 调试: 完整特征 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = StructonVisionSystem()
    
    for digit in range(10):
        print(f"\n{'='*50}")
        print(f"数字 {digit}")
        print('='*50)
        
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"\n  样本 {i+1}:")
            print(f"    签名: {info['signature']}")
            print(f"    拓扑: {info['topology']}")
            print(f"    端点位置: {info['ep_pos']}")
            print(f"    结构: {info['structure']}")
            print(f"    曲率: {info['curvature']}")
            print(f"    终止方向: {info['termination']}")
            print(f"    曲线开口: {info['curve_open']}")
            print(f"    特殊: {info['special']}")


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
