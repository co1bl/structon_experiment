"""
Structon Vision v7.10 - Rich State Space + Reward-Based LRM
============================================================

结合两个版本的优点：
- v7.6: 丰富的语义特征 (state space)
- v7.9: 基于奖励的 LRM + promote 机制

State Space (特征向量):
- 拓扑: holes, endpoints, junctions, is_closed
- 位置: endpoint/junction 位置 (9区域)
- 结构: top_horizontal, bottom_horizontal, center_vertical
- 曲率: sharp_corners, smooth_curves
- 终止方向: up, down, left, right
- 曲线开口: open_left, open_right
- 几何: aspect_ratio, fill_density
- 特殊: loop_top, loop_bottom, tail_top, tail_bottom

Action Space:
- 10 个动作 (数字 0-9)

Reward:
- +1: 预测正确
- -1: 预测错误

LRM 机制:
- 存储 state → action 映射
- reward 调整 strength
- strength 影响匹配分数
- 满了 → promote (freeze + wrap)

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import copy
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. State Space: Comprehensive Feature Extractor
# =============================================================================

class StateExtractor:
    """
    提取丰富的语义特征作为 State Space
    
    输出: 归一化的特征向量 (约30维)
    """
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取状态向量"""
        h, w = image.shape
        features = []
        
        # 预处理
        binary = (image > 0.3).astype(np.uint8)
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(binary)
        
        # === 1. 拓扑特征 (归一化) ===
        features.append(n_holes / 3.0)              # 0-2 holes typical
        features.append(len(endpoints) / 5.0)       # 0-4 endpoints typical
        features.append(len(junctions) / 3.0)       # 0-2 junctions typical
        features.append(1.0 if (len(endpoints) == 0 and n_holes >= 1) else 0.0)  # is_closed
        
        # === 2. 端点位置 (9区域, binary) ===
        ep_regions = [0.0] * 9  # TL, T, TR, L, C, R, BL, B, BR
        for y, x in endpoints:
            ry, rx = y / h, x / w
            row = 0 if ry < 1/3 else (1 if ry < 2/3 else 2)
            col = 0 if rx < 1/3 else (1 if rx < 2/3 else 2)
            ep_regions[row * 3 + col] = 1.0
        features.extend(ep_regions)
        
        # === 3. 交叉点位置 (3区域: top, center, bottom) ===
        jc_regions = [0.0, 0.0, 0.0]
        for y, x in junctions:
            ry = y / h
            if ry < 1/3:
                jc_regions[0] = 1.0
            elif ry < 2/3:
                jc_regions[1] = 1.0
            else:
                jc_regions[2] = 1.0
        features.extend(jc_regions)
        
        # === 4. 边缘方向特征 ===
        top_h = self._has_horizontal(image, 0, h//3)
        bot_h = self._has_horizontal(image, 2*h//3, h)
        ctr_v = self._has_vertical(image, h//3, 2*h//3, w//3, 2*w//3)
        ctr_h = self._has_horizontal(image, h//3, 2*h//3)
        
        features.append(1.0 if top_h else 0.0)
        features.append(1.0 if bot_h else 0.0)
        features.append(1.0 if ctr_v else 0.0)
        features.append(1.0 if ctr_h else 0.0)
        
        # === 5. 曲率特征 ===
        corners = self._detect_corners(skeleton)
        features.append(len(corners) / 5.0)  # n_corners normalized
        features.append(1.0 if any(y < h/3 for y, x in corners) else 0.0)  # top_corner
        features.append(1.0 if any(y > 2*h/3 for y, x in corners) else 0.0)  # bot_corner
        
        # === 6. 笔画终止方向 ===
        term_up, term_down, term_left, term_right = False, False, False, False
        for y, x in endpoints:
            direction = self._get_endpoint_direction(skeleton, y, x)
            if direction == 'up': term_up = True
            elif direction == 'down': term_down = True
            elif direction == 'left': term_left = True
            elif direction == 'right': term_right = True
        
        features.append(1.0 if term_up else 0.0)
        features.append(1.0 if term_down else 0.0)
        features.append(1.0 if term_left else 0.0)
        features.append(1.0 if term_right else 0.0)
        
        # === 7. 曲线开口 ===
        left_mass = np.sum(binary[:, :w//2])
        right_mass = np.sum(binary[:, w//2:])
        top_mass = np.sum(binary[:h//2, :])
        bot_mass = np.sum(binary[h//2:, :])
        
        total_mass = left_mass + right_mass + 1e-9
        features.append(1.0 if right_mass > left_mass * 1.8 else 0.0)  # open_left
        features.append(1.0 if left_mass > right_mass * 1.8 else 0.0)  # open_right
        
        # === 8. 几何特征 ===
        # Aspect ratio
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        if rows.any() and cols.any():
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            bbox_h = y_max - y_min + 1
            bbox_w = x_max - x_min + 1
            aspect = bbox_h / (bbox_w + 1e-9)
            features.append(min(aspect / 2.0, 1.0))  # normalized aspect
        else:
            features.append(0.5)
        
        # Fill density
        density = np.sum(binary) / (h * w)
        features.append(density * 5.0)  # scale to ~0-1 range
        
        # Center of mass
        if np.sum(binary) > 0:
            cy, cx = np.argwhere(binary).mean(axis=0)
            features.append(cx / w)  # x position 0-1
            features.append(cy / h)  # y position 0-1
        else:
            features.append(0.5)
            features.append(0.5)
        
        # === 9. 特殊模式 ===
        # Loop position (基于 hole 位置)
        loop_top, loop_bot = False, False
        if n_holes >= 1:
            loop_pos = self._detect_loop_position(binary)
            loop_top = loop_pos == 'top'
            loop_bot = loop_pos == 'bottom'
        
        features.append(1.0 if loop_top else 0.0)
        features.append(1.0 if loop_bot else 0.0)
        
        # Tail position (有 loop + 有 endpoint)
        tail_top = n_holes >= 1 and any(y < h/3 for y, x in endpoints)
        tail_bot = n_holes >= 1 and any(y > 2*h/3 for y, x in endpoints)
        features.append(1.0 if tail_top else 0.0)
        features.append(1.0 if tail_bot else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _has_horizontal(self, image: np.ndarray, y1: int, y2: int) -> bool:
        """检查区域是否有水平边"""
        region = image[y1:y2, :]
        if region.size == 0 or np.mean(region > 0.3) < 0.05:
            return False
        gy = np.abs(region[1:, :] - region[:-1, :]).sum()
        gx = np.abs(region[:, 1:] - region[:, :-1]).sum()
        return gy > gx * 1.3
    
    def _has_vertical(self, image: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> bool:
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
                        if dot > 0.3:
                            corners.append((y, x))
        return corners
    
    def _get_endpoint_direction(self, skeleton: np.ndarray, y: int, x: int) -> str:
        """获取端点方向"""
        h, w = skeleton.shape
        for dy, dx, direction in [(-1, 0, 'down'), (1, 0, 'up'),
                                   (0, -1, 'right'), (0, 1, 'left')]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skeleton[ny, nx]:
                return direction
        return ''
    
    def _detect_loop_position(self, binary: np.ndarray) -> str:
        """检测 loop 位置"""
        h, w = binary.shape
        
        # 找内部空白区域
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary
        
        visited = np.zeros_like(padded, dtype=bool)
        queue = [(0, 0)]
        visited[0, 0] = True
        while queue:
            cy, cx = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                    if padded[ny, nx] == 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        # 找内部空白点
        internal = []
        for y in range(1, h + 1):
            for x in range(1, w + 1):
                if padded[y, x] == 0 and not visited[y, x]:
                    internal.append((y - 1, x - 1))
        
        if not internal:
            return 'center'
        
        mean_y = np.mean([p[0] for p in internal])
        if mean_y < h / 3:
            return 'top'
        elif mean_y > 2 * h / 3:
            return 'bottom'
        return 'center'
    
    def _skeletonize(self, image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
        """Zhang-Suen 骨架化"""
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
    
    def _count_holes(self, binary: np.ndarray) -> int:
        """计算空洞数量"""
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
# 2. Memory Entry
# =============================================================================

@dataclass
class MemoryEntry:
    """LRM 记忆条目: state → action"""
    state: np.ndarray         # 状态向量
    action: str               # 动作 (数字标签)
    strength: float = 1.0     # 强度 (reward 调节)
    access_count: int = 0
    success_count: int = 0
    
    def reinforce(self, reward: float):
        """根据 reward 调整强度"""
        self.access_count += 1
        if reward > 0:
            self.success_count += 1
            self.strength = min(2.0, self.strength + 0.15 * reward)
        else:
            self.strength = max(0.1, self.strength + 0.1 * reward)


# =============================================================================
# 3. Local Resonant Memory (LRM)
# =============================================================================

class LRM:
    """局部共振记忆"""
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.85):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.entries: List[MemoryEntry] = []
        self.frozen = False
    
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity
    
    def size(self) -> int:
        return len(self.entries)
    
    def freeze(self):
        self.frozen = True
    
    def query(self, state: np.ndarray) -> Tuple[Optional[int], float]:
        """查询最佳匹配"""
        if not self.entries:
            return None, 0.0
        
        best_idx = None
        best_score = -1.0
        
        for i, entry in enumerate(self.entries):
            sim = self._cosine(state, entry.state)
            score = sim * entry.strength
            if score > best_score:
                best_score = score
                best_idx = i
        
        if best_score >= self.similarity_threshold:
            return best_idx, best_score
        return None, best_score
    
    def get_action(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(self.entries):
            return self.entries[idx].action
        return None
    
    def reinforce(self, idx: int, reward: float):
        if 0 <= idx < len(self.entries):
            self.entries[idx].reinforce(reward)
    
    def add(self, state: np.ndarray, action: str) -> int:
        if self.frozen:
            return -1
        entry = MemoryEntry(state=state.copy(), action=action)
        self.entries.append(entry)
        return len(self.entries) - 1
    
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# 4. Structon
# =============================================================================

class Structon:
    """Structon - 分形智能单元"""
    
    _id_counter = 0
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.85):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.lrm = LRM(capacity=capacity, similarity_threshold=similarity_threshold)
        self.children: List['Structon'] = []
        
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        
        self.total_queries = 0
        self.local_hits = 0
    
    @property
    def frozen(self) -> bool:
        return self.lrm.frozen
    
    def freeze(self):
        self.lrm.freeze()
    
    def query(self, state: np.ndarray) -> Tuple[Optional[str], float, 'Structon']:
        """查询 - 返回 action"""
        self.total_queries += 1
        
        # 本层查询
        idx, score = self.lrm.query(state)
        if idx is not None:
            self.local_hits += 1
            action = self.lrm.get_action(idx)
            return action, score, self
        
        # 路由到子节点
        if self.children:
            best_action = None
            best_score = -1.0
            best_structon = None
            
            for child in self.children:
                action, child_score, responding = child.query(state)
                if child_score > best_score:
                    best_score = child_score
                    best_action = action
                    best_structon = responding
            
            if best_action is not None:
                return best_action, best_score, best_structon
        
        return None, score, self
    
    def learn(self, state: np.ndarray, action: str, reward: float) -> Tuple[str, 'Structon']:
        """学习 - 基于 reward"""
        
        # Case 1: 我是 wrapper
        if self.children:
            # 先查 frozen children
            for child in self.children:
                if child.frozen:
                    idx, score = child.lrm.query(state)
                    if idx is not None:
                        child.lrm.reinforce(idx, reward)
                        return "reinforced_frozen", self
            
            # frozen 不认识，给 active sibling
            for i, child in enumerate(self.children):
                if not child.frozen:
                    status, returned = child.learn(state, action, reward)
                    if "promoted" in status:
                        self.children[i] = returned
                    return status, self
            
            return "no_active_child", self
        
        # Case 2: 我是叶子
        if self.frozen:
            return "frozen_leaf", self
        
        # 本层查询
        idx, score = self.lrm.query(state)
        
        if idx is not None:
            self.lrm.reinforce(idx, reward)
            return "reinforced", self
        
        # 无匹配 + 正 reward → 添加
        if reward > 0:
            if not self.lrm.is_full():
                self.lrm.add(state, action)
                return "added", self
            else:
                new_wrapper = self.promote()
                for child in new_wrapper.children:
                    if not child.frozen:
                        child.lrm.add(state, action)
                        break
                return "promoted_and_added", new_wrapper
        
        return "no_match_negative", self
    
    def promote(self) -> 'Structon':
        """晋升：冻结 → sibling → 包裹"""
        self.freeze()
        
        sibling = Structon(
            capacity=self.capacity,
            similarity_threshold=self.similarity_threshold
        )
        
        wrapper = Structon(
            capacity=self.capacity,
            similarity_threshold=self.similarity_threshold
        )
        wrapper.children = [self, sibling]
        
        return wrapper
    
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def total_entries(self) -> int:
        count = self.lrm.size()
        for child in self.children:
            count += child.total_entries()
        return count
    
    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def hit_rate(self) -> float:
        return self.local_hits / self.total_queries if self.total_queries > 0 else 0.0
    
    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        icon = "❄️" if self.frozen else "🔥"
        children_info = f" children:{len(self.children)}" if self.children else ""
        
        print(f"{prefix}{icon} {self.id} (mem:{self.lrm.size()}/{self.capacity}, hit:{self.hit_rate()*100:.0f}%){children_info}")
        
        for i, entry in enumerate(self.lrm.entries[:5]):
            print(f"{prefix}  └─ {i}: [{entry.action}] str={entry.strength:.2f} "
                  f"acc={entry.access_count} suc={entry.success_count}")
        if self.lrm.size() > 5:
            print(f"{prefix}  └─ ... ({self.lrm.size() - 5} more)")
        
        for child in self.children:
            child.print_tree(indent + 1)


# =============================================================================
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """Structon 视觉系统"""
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.85):
        self.extractor = StateExtractor()
        self.root = Structon(capacity=capacity, similarity_threshold=similarity_threshold)
        
        self.train_count = 0
        self.correct_count = 0
    
    def predict(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        state = self.extractor.extract(image)
        action, score, _ = self.root.query(state)
        return action, score
    
    def train_one(self, image: np.ndarray, true_label: str) -> Tuple[str, bool]:
        self.train_count += 1
        
        state = self.extractor.extract(image)
        predicted, score, _ = self.root.query(state)
        
        if predicted == true_label:
            reward = 1.0
            correct = True
            self.correct_count += 1
        elif predicted is None:
            reward = 1.0
            correct = False
        else:
            reward = -1.0
            correct = False
        
        status, returned = self.root.learn(state, true_label, reward)
        
        if "promoted" in status:
            self.root = returned
        
        return status, correct
    
    def train_accuracy(self) -> float:
        return self.correct_count / self.train_count if self.train_count > 0 else 0.0
    
    def print_stats(self):
        print(f"\n=== Structon Vision System ===")
        print(f"训练样本: {self.train_count}")
        print(f"训练准确率: {self.train_accuracy()*100:.1f}%")
        print(f"树深度: {self.root.depth()}")
        print(f"总节点数: {self.root.count_nodes()}")
        print(f"总记忆条目: {self.root.total_entries()}")
        print(f"\n=== 树结构 ===")
        self.root.print_tree()


# =============================================================================
# 6. MNIST
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
                except:
                    continue
    
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


def run_experiment(n_train=100, n_test=500, capacity=10, threshold=0.85, verbose=True):
    print("=" * 70)
    print("Structon Vision v7.10 - Rich State Space + Reward-Based LRM")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, threshold={threshold}")
    print(f"训练: {n_train}, 测试: {n_test}")
    print("\n状态空间 (~35维):")
    print("  拓扑, 端点位置, 交叉位置, 边缘方向, 曲率,")
    print("  终止方向, 曲线开口, 几何, 特殊模式")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(capacity=capacity, similarity_threshold=threshold)
    
    print(f"\n增量训练 {n_train} 个样本...")
    train_indices = np.random.choice(len(train_images), n_train, replace=False)
    
    t0 = time.time()
    promote_count = 0
    
    for i, idx in enumerate(train_indices):
        status, correct = system.train_one(train_images[idx], str(train_labels[idx]))
        
        if "promoted" in status:
            promote_count += 1
            if verbose:
                print(f"  [{i}] PROMOTE! depth={system.root.depth()}, nodes={system.root.count_nodes()}")
        
        if verbose and (i + 1) % 50 == 0:
            print(f"  [{i+1}/{n_train}] 训练准确率: {system.train_accuracy()*100:.1f}%")
    
    print(f"\n训练完成: {time.time()-t0:.1f}秒, Promote: {promote_count}次")
    
    system.print_stats()
    
    # 测试
    print(f"\n测试 {n_test} 个样本...")
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    
    t0 = time.time()
    for idx in test_indices:
        predicted, _ = system.predict(test_images[idx])
        true_label = str(test_labels[idx])
        
        results[true_label]['total'] += 1
        if predicted == true_label:
            results[true_label]['correct'] += 1
    
    print(f"测试完成: {time.time()-t0:.1f}秒")
    
    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    
    print(f"\n总准确率: {total_correct/total_samples*100:.1f}%")
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  {d}: {acc:.1f}% ({r['correct']}/{r['total']})")
    
    return system


def debug_states(n=3):
    """调试状态空间"""
    print("\n=== 调试: 状态空间 ===")
    train_images, train_labels, _, _ = load_mnist()
    extractor = StateExtractor()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        
        states = []
        for idx in indices:
            state = extractor.extract(train_images[idx])
            states.append(state)
            print(f"  样本: {len(state)}维, holes={state[0]*3:.0f}, eps={state[1]*5:.0f}, "
                  f"closed={state[3]:.0f}, top_H={state[16]:.0f}")
        
        if len(states) >= 2:
            sim = np.dot(states[0], states[1]) / (np.linalg.norm(states[0]) * np.linalg.norm(states[1]))
            print(f"  样本间相似度: {sim:.3f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        debug_states(3)
    else:
        run_experiment(args.train, args.test, args.capacity, args.threshold)
