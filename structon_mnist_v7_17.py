"""
Structon Vision v7.11 - 完整多巴胺驱动分形学习
==============================================

核心机制：
1. 多巴胺驱动：dopamine = actual - expected
2. 门控学习：只有 |δ| > threshold 才学习
3. 预期习惯化：重复暴露 → 预期稳定 → 不再惊讶
4. 分形生长：满了 → 冻结 → 复制(继承) → 包裹
5. 按类别训练：先学完一类，再学下一类

这才是真正的 Structon：
- 局部规则（多巴胺）
- 全局涌现（分形结构）
- 增量学习（无灾难性遗忘）

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. State Space: Feature Extractor
# =============================================================================

class StateExtractor:
    """提取状态向量"""
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape
        features = []
        
        binary = (image > 0.3).astype(np.uint8)
        skeleton = self._skeletonize(image)
        endpoints, junctions = self._find_topology_points(skeleton)
        n_holes = self._count_holes(binary)
        
        # 拓扑
        features.append(n_holes / 3.0)
        features.append(len(endpoints) / 5.0)
        features.append(len(junctions) / 3.0)
        features.append(1.0 if (len(endpoints) == 0 and n_holes >= 1) else 0.0)
        
        # 端点位置 (9区域)
        ep_regions = [0.0] * 9
        for y, x in endpoints:
            ry, rx = y / h, x / w
            row = 0 if ry < 1/3 else (1 if ry < 2/3 else 2)
            col = 0 if rx < 1/3 else (1 if rx < 2/3 else 2)
            ep_regions[row * 3 + col] = 1.0
        features.extend(ep_regions)
        
        # 交叉点位置 (3区域)
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
        
        # 边缘方向
        features.append(1.0 if self._has_horizontal(image, 0, h//3) else 0.0)
        features.append(1.0 if self._has_horizontal(image, 2*h//3, h) else 0.0)
        features.append(1.0 if self._has_vertical(image, h//3, 2*h//3, w//3, 2*w//3) else 0.0)
        features.append(1.0 if self._has_horizontal(image, h//3, 2*h//3) else 0.0)
        
        # 密度分布
        third_h = h // 3
        features.append(float(np.mean(image[:third_h] > 0.3)))
        features.append(float(np.mean(image[third_h:2*third_h] > 0.3)))
        features.append(float(np.mean(image[2*third_h:] > 0.3)))
        
        # 质心
        if np.sum(binary) > 0:
            cy, cx = np.argwhere(binary).mean(axis=0)
            features.append(cx / w)
            features.append(cy / h)
        else:
            features.append(0.5)
            features.append(0.5)
        
        return np.array(features, dtype=np.float32)
    
    def _has_horizontal(self, image, y1, y2):
        region = image[y1:y2, :]
        if region.size == 0 or np.mean(region > 0.3) < 0.05:
            return False
        gy = np.abs(region[1:, :] - region[:-1, :]).sum()
        gx = np.abs(region[:, 1:] - region[:, :-1]).sum()
        return gy > gx * 1.3
    
    def _has_vertical(self, image, y1, y2, x1, x2):
        region = image[y1:y2, x1:x2]
        if region.size == 0 or np.mean(region > 0.3) < 0.05:
            return False
        gy = np.abs(region[1:, :] - region[:-1, :]).sum()
        gx = np.abs(region[:, 1:] - region[:, :-1]).sum()
        return gx > gy * 1.3
    
    def _skeletonize(self, image, threshold=0.3):
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
    
    def _find_topology_points(self, skeleton):
        h, w = skeleton.shape
        endpoints, junctions = [], []
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
    
    def _count_holes(self, binary):
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
# 2. Memory Entry - 带预期
# =============================================================================

@dataclass
class MemoryEntry:
    """LRM 记忆条目"""
    pattern: np.ndarray
    action: str
    expectation: float = 0.0   # 预期感觉（会习惯化）
    strength: float = 1.0
    access_count: int = 0


# =============================================================================
# 3. LRM - 多巴胺驱动
# =============================================================================

class LRM:
    """局部共振记忆 - 多巴胺驱动"""
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.85):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.entries: List[MemoryEntry] = []
        self.frozen = False
        
        # 多巴胺参数
        self.dopamine_threshold = 0.3  # 惊讶阈值
        self.learning_rate = 0.2
        self.habituation_rate = 0.1    # 习惯化速率
    
    def query(self, pattern: np.ndarray) -> Tuple[Optional[int], float]:
        """共振查询"""
        if not self.entries:
            return None, 0.0
        
        best_idx = None
        best_sim = -1.0
        
        for i, entry in enumerate(self.entries):
            sim = self._cosine(pattern, entry.pattern)
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        
        if best_idx is not None and best_sim >= self.similarity_threshold:
            self.entries[best_idx].access_count += 1
            return best_idx, best_sim
        
        return None, best_sim
    
    def get_action(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(self.entries):
            return self.entries[idx].action
        return None
    
    def get_expectation(self, idx: int) -> float:
        if 0 <= idx < len(self.entries):
            return self.entries[idx].expectation
        return 0.0
    
    def update_with_dopamine(self, idx: int, dopamine: float):
        """用多巴胺更新记忆"""
        if self.frozen or idx < 0 or idx >= len(self.entries):
            return
        
        entry = self.entries[idx]
        
        # 更新强度
        entry.strength += self.learning_rate * dopamine
        entry.strength = max(0.1, min(2.0, entry.strength))
        
        # 习惯化：预期向实际靠拢
        # 如果 dopamine > 0，说明实际比预期好，预期应该提高
        # 如果 dopamine < 0，说明实际比预期差，预期应该降低
        entry.expectation += self.habituation_rate * dopamine
        entry.expectation = max(-1.0, min(1.0, entry.expectation))
    
    def add(self, pattern: np.ndarray, action: str, expectation: float = 0.0) -> int:
        if self.frozen:
            return -1
        if len(self.entries) >= self.capacity:
            return -1  # 满了，不能添加
        entry = MemoryEntry(
            pattern=self._normalize(pattern).copy(),
            action=action,
            expectation=expectation
        )
        self.entries.append(entry)
        return len(self.entries) - 1
    
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity
    
    def size(self) -> int:
        return len(self.entries)
    
    def freeze(self):
        self.frozen = True
    
    def clone(self) -> 'LRM':
        """深度复制"""
        new_lrm = LRM(self.capacity, self.similarity_threshold)
        new_lrm.entries = [
            MemoryEntry(
                pattern=e.pattern.copy(),
                action=e.action,
                expectation=e.expectation,
                strength=e.strength,
                access_count=0  # 重置访问计数
            )
            for e in self.entries
        ]
        new_lrm.frozen = False
        return new_lrm
    
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v
    
    def _cosine(self, a, b):
        return float(np.dot(self._normalize(a), self._normalize(b)))


# =============================================================================
# 4. Structon - 分形单元
# =============================================================================

class Structon:
    """Structon - 多巴胺驱动的分形智能单元"""
    
    _id_counter = 0
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.85,
                 dopamine_threshold: float = 0.3, full_dim: int = 25, key_dim: int = 8,
                 projector: Optional[np.ndarray] = None):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.lrm = LRM(capacity=capacity, similarity_threshold=similarity_threshold)
        self.lrm.dopamine_threshold = dopamine_threshold
        self.children: List['Structon'] = []
        
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.dopamine_threshold = dopamine_threshold
        
        # 随机投影：将 full_dim 投影到 key_dim
        self.full_dim = full_dim
        self.key_dim = key_dim
        
        if projector is not None:
            # 继承投影矩阵
            self.projector = projector.copy()
        else:
            # 生成新的随机投影矩阵
            self.projector = np.random.randn(full_dim, key_dim).astype(np.float32)
            # 列归一化
            self.projector /= (np.linalg.norm(self.projector, axis=0, keepdims=True) + 1e-8)
        
        # 统计
        self.total_queries = 0
        self.local_hits = 0
        self.learns = 0
        self.skipped = 0  # 门控跳过次数
    
    def _local(self, pattern: np.ndarray) -> np.ndarray:
        """随机投影到局部空间"""
        key = pattern.astype(np.float32) @ self.projector
        norm = np.linalg.norm(key)
        if norm > 1e-8:
            key /= norm
        return key
    
    @property
    def frozen(self) -> bool:
        return self.lrm.frozen
    
    def freeze(self):
        self.lrm.freeze()
    
    def query(self, pattern: np.ndarray) -> Tuple[Optional[str], float, Optional[int], 'Structon']:
        """
        查询
        返回: (action, similarity, entry_idx, responding_structon)
        """
        self.total_queries += 1
        
        # 本层共振 - 用投影后的特征
        local_pattern = self._local(pattern)
        idx, sim = self.lrm.query(local_pattern)
        if idx is not None:
            self.local_hits += 1
            action = self.lrm.get_action(idx)
            return action, sim, idx, self
        
        # 路由到 children
        if self.children:
            best_action = None
            best_sim = -1.0
            best_idx = None
            best_structon = None
            
            for child in self.children:
                action, child_sim, child_idx, responding = child.query(pattern)  # 传完整 pattern
                if child_sim > best_sim:
                    best_sim = child_sim
                    best_action = action
                    best_idx = child_idx
                    best_structon = responding
            
            if best_action is not None:
                return best_action, best_sim, best_idx, best_structon
        
        return None, sim, None, self
    
    def learn(self, pattern: np.ndarray, true_action: str) -> Tuple[str, 'Structon']:
        """
        多巴胺驱动学习
        
        惊讶 = 预测和现实不符
        - 没见过 → 惊讶 → 存新记忆
        - 预测对了 → 不惊讶 → 小强化
        - 预测错了 → 惊讶 → 弱化错误 + 存正确
        """
        # 1. 查询：我预测是什么？
        predicted_action, sim, idx, responding = self.query(pattern)
        
        # 2. 判断惊讶并学习
        if idx is None:
            # === 没见过 → 惊讶！→ 存新记忆 ===
            return self._add_new_memory(pattern, true_action)
        
        if predicted_action == true_action:
            # === 预测对了 → 不惊讶 → 小强化 ===
            self.skipped += 1
            responding.lrm.update_with_dopamine(idx, 0.1)
            return "correct_reinforced", self
        
        # === 预测错了 → 惊讶！→ 弱化错误 + 存正确 ===
        self.learns += 1
        responding.lrm.update_with_dopamine(idx, -1.0)
        
        # 找到正确的地方存储正确记忆
        return self._add_new_memory(pattern, true_action)
    
    def _add_new_memory(self, pattern: np.ndarray, action: str) -> Tuple[str, 'Structon']:
        """添加新记忆，处理容量和 promote"""
        self.learns += 1
        
        if self.children:
            # 我是 wrapper → 找 active sibling
            for i, child in enumerate(self.children):
                if not child.frozen:
                    status, returned = child._add_new_memory(pattern, action)
                    if "promoted" in status:
                        self.children[i] = returned
                    return status, self
            
            # 没有 active child？不应该发生
            return "no_active_child", self
        
        # 我是叶子
        if self.frozen:
            return "frozen_cannot_add", self
        
        # 用局部特征存储
        local_pattern = self._local(pattern)
        
        # 检查容量
        if self.lrm.size() < self.capacity:
            self.lrm.add(local_pattern, action, expectation=0.0)
            return "added", self
        
        # 满了 → promote
        new_wrapper = self.promote()
        
        # 在 sibling（非冻结的那个）中添加 - sibling 用自己的 _local()
        for child in new_wrapper.children:
            if not child.frozen:
                child_local = child._local(pattern)
                child.lrm.add(child_local, action, expectation=0.0)
                child.learns += 1
                break
        
        return "promoted_and_added", new_wrapper
    
    def promote(self) -> 'Structon':
        """
        晋升：冻结 → 创建空 sibling → 包裹
        
        frozen: 保留所有记忆（保护旧知识）
        sibling: 空的！从零开始学新东西
        关键：sibling 和 wrapper 继承相同的投影矩阵
        """
        # 1. 冻结自己
        self.freeze()
        
        # 2. 创建空的 sibling（继承相同的投影矩阵！）
        sibling = Structon(
            capacity=self.capacity,
            similarity_threshold=self.similarity_threshold,
            dopamine_threshold=self.dopamine_threshold,
            full_dim=self.full_dim,
            key_dim=self.key_dim,
            projector=self.projector  # 继承投影矩阵
        )
        
        # 3. 创建 wrapper（也继承相同的投影矩阵！）
        wrapper = Structon(
            capacity=self.capacity,
            similarity_threshold=self.similarity_threshold,
            dopamine_threshold=self.dopamine_threshold,
            full_dim=self.full_dim,
            key_dim=self.key_dim,
            projector=self.projector  # 继承投影矩阵
        )
        
        # 4. 建立关系
        wrapper.children = [self, sibling]
        
        return wrapper
    
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def total_entries(self) -> int:
        count = self.lrm.size()
        for child in self.children:
            count += child.total_entries()
        return count
    
    def total_learns(self) -> int:
        count = self.learns
        for child in self.children:
            count += child.total_learns()
        return count
    
    def total_skipped(self) -> int:
        count = self.skipped
        for child in self.children:
            count += child.total_skipped()
        return count
    
    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        icon = "❄️" if self.frozen else "🔥"
        
        print(f"{prefix}{icon} {self.id} (mem:{self.lrm.size()}/{self.capacity}, "
              f"proj:{self.full_dim}→{self.key_dim})")
        
        for i, entry in enumerate(self.lrm.entries[:3]):
            print(f"{prefix}  └─ [{entry.action}] exp={entry.expectation:.2f} "
                  f"str={entry.strength:.2f} acc={entry.access_count}")
        if self.lrm.size() > 3:
            print(f"{prefix}  └─ ... ({self.lrm.size() - 3} more)")
        
        for child in self.children:
            child.print_tree(indent + 1)


# =============================================================================
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """Structon 视觉系统 - 多巴胺驱动"""
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.85,
                 dopamine_threshold: float = 0.3, full_dim: int = 25, key_dim: int = 8):
        self.extractor = StateExtractor()
        self.root = Structon(
            capacity=capacity,
            similarity_threshold=similarity_threshold,
            dopamine_threshold=dopamine_threshold,
            full_dim=full_dim,
            key_dim=key_dim
        )
        
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.dopamine_threshold = dopamine_threshold
        self.full_dim = full_dim
        self.key_dim = key_dim
        
        self.train_count = 0
        self.correct_count = 0
        self.promote_count = 0
    
    def predict(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        state = self.extractor.extract(image)
        action, sim, _, _ = self.root.query(state)
        return action, sim
    
    def train_one(self, image: np.ndarray, true_action: str) -> Tuple[str, bool]:
        """训练一个样本"""
        self.train_count += 1
        
        state = self.extractor.extract(image)
        
        # 先预测
        predicted, sim, idx, responding = self.root.query(state)
        correct = (predicted == true_action)
        if correct:
            self.correct_count += 1
        
        # 学习
        status, returned = self.root.learn(state, true_action)
        
        if "promoted" in status:
            self.root = returned
            self.promote_count += 1
        
        return status, correct
    
    def train_accuracy(self) -> float:
        return self.correct_count / self.train_count if self.train_count > 0 else 0.0
    
    def print_stats(self):
        print(f"\n=== Structon Vision System ===")
        print(f"训练样本: {self.train_count}")
        print(f"训练准确率: {self.train_accuracy()*100:.1f}%")
        print(f"Promote 次数: {self.promote_count}")
        print(f"树深度: {self.root.depth()}")
        print(f"总节点数: {self.root.count_nodes()}")
        print(f"总记忆条目: {self.root.total_entries()}")
        print(f"总学习次数: {self.root.total_learns()}")
        print(f"总跳过次数: {self.root.total_skipped()}")
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


# =============================================================================
# 7. 按类别训练
# =============================================================================

def run_experiment(n_per_class=20, n_test=500, capacity=10, threshold=0.85,
                   dopamine_threshold=0.3, key_dim=8, verbose=True):
    """
    按类别训练实验
    
    关键：先学完一类，再学下一类
    """
    print("=" * 70)
    print("Structon Vision v7.17 - 随机投影 + 多巴胺驱动分形学习")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  capacity={capacity}, threshold={threshold}")
    print(f"  dopamine_threshold={dopamine_threshold}")
    print(f"  key_dim={key_dim} (25D → {key_dim}D 随机投影)")
    print(f"  每类训练: {n_per_class}, 测试: {n_test}")
    print("\n机制:")
    print("  1. 随机投影：每个Structon用投影矩阵编码特征")
    print("  2. 多巴胺门控：惊讶才学习")
    print("  3. 按类别训练：先学完一类再学下一类")
    print("  4. 分形生长：promote时继承投影矩阵")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        capacity=capacity,
        similarity_threshold=threshold,
        dopamine_threshold=dopamine_threshold,
        full_dim=25,
        key_dim=key_dim
    )
    
    print(f"\n=== 按类别增量训练 ===")
    t0 = time.time()
    
    for digit in range(10):
        # 获取该类别的样本
        indices = np.where(train_labels == digit)[0][:n_per_class]
        
        print(f"\n--- 学习数字 {digit} ({len(indices)} 样本) ---")
        
        digit_correct = 0
        digit_learns = 0
        digit_skipped = 0
        
        for i, idx in enumerate(indices):
            status, correct = system.train_one(train_images[idx], str(digit))
            
            if correct:
                digit_correct += 1
            if "added" in status or "updated" in status:
                digit_learns += 1
            if "skipped" in status:
                digit_skipped += 1
        
        print(f"  准确率: {digit_correct}/{len(indices)} ({digit_correct/len(indices)*100:.0f}%)")
        print(f"  学习: {digit_learns}, 跳过(习惯化): {digit_skipped}")
        print(f"  当前树: depth={system.root.depth()}, nodes={system.root.count_nodes()}, "
              f"entries={system.root.total_entries()}")
    
    print(f"\n训练完成: {time.time()-t0:.1f}秒")
    
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试 {n_test} 样本 ===")
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
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


def run_mixed_comparison(n_per_class=20, n_test=500, capacity=10):
    """
    对比实验：按类别 vs 混合训练
    """
    print("=" * 70)
    print("对比实验：按类别训练 vs 混合训练")
    print("=" * 70)
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 准备数据
    all_indices = []
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        all_indices.extend([(idx, digit) for idx in indices])
    
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    def test_system(system):
        correct = 0
        for idx in test_indices:
            predicted, _ = system.predict(test_images[idx])
            if predicted == str(test_labels[idx]):
                correct += 1
        return correct / len(test_indices) * 100
    
    # 实验 A：按类别训练
    print("\n--- A: 按类别训练 ---")
    system_a = StructonVisionSystem(capacity=capacity)
    
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        for idx in indices:
            system_a.train_one(train_images[idx], str(digit))
    
    acc_a = test_system(system_a)
    print(f"准确率: {acc_a:.1f}%")
    print(f"树: depth={system_a.root.depth()}, nodes={system_a.root.count_nodes()}")
    
    # 实验 B：混合训练
    print("\n--- B: 混合训练 ---")
    system_b = StructonVisionSystem(capacity=capacity)
    
    np.random.shuffle(all_indices)
    for idx, digit in all_indices:
        system_b.train_one(train_images[idx], str(digit))
    
    acc_b = test_system(system_b)
    print(f"准确率: {acc_b:.1f}%")
    print(f"树: depth={system_b.root.depth()}, nodes={system_b.root.count_nodes()}")
    
    # 结果
    print("\n=== 结论 ===")
    print(f"按类别: {acc_a:.1f}%")
    print(f"混合:   {acc_b:.1f}%")
    print(f"差异:   {acc_a - acc_b:+.1f}%")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=20)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--dopamine', type=float, default=0.3)
    parser.add_argument('--key-dim', type=int, default=8, help='投影后的维度')
    parser.add_argument('--compare', action='store_true', help='对比按类别vs混合')
    args = parser.parse_args()
    
    if args.compare:
        run_mixed_comparison(args.per_class, args.test, args.capacity)
    else:
        run_experiment(
            n_per_class=args.per_class,
            n_test=args.test,
            capacity=args.capacity,
            threshold=args.threshold,
            dopamine_threshold=args.dopamine,
            key_dim=args.key_dim
        )
