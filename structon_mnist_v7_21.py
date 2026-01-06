"""
Structon Vision v7.21 - 真正的 Bottom-Up 分形架构
=================================================

核心修复：
1. 动态晋升：记忆满了 → 冻结 → 创建 Wrapper → 继续生长
2. 路由即动作：Wrapper 学习 "送给哪个 child"，不学标签
3. 强化反馈：child 预测对了 → 强化路由；错了 → 弱化路由

设计哲学：
- Wrapper 不知道底层数字是几
- Wrapper 只知道 "这个 pattern 送给 Child_A 通常得到好结果"
- 结构自动生长，知识永不丢失

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. MNIST 加载
# =============================================================================

def load_mnist(path='./mnist_data'):
    """加载 MNIST 数据集"""
    os.makedirs(path, exist_ok=True)
    
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    for name, filename in files.items():
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    def read_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num, rows, cols)
    
    def read_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    train_images = read_images(os.path.join(path, files['train_images']))
    train_labels = read_labels(os.path.join(path, files['train_labels']))
    test_images = read_images(os.path.join(path, files['test_images']))
    test_labels = read_labels(os.path.join(path, files['test_labels']))
    
    return train_images, train_labels, test_images, test_labels


# =============================================================================
# 2. 特征提取器
# =============================================================================

class StateExtractor:
    """提取 MNIST 图像的状态特征"""
    
    def __init__(self):
        self.feature_dim = 25
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取 25 维特征"""
        img = image.astype(np.float32) / 255.0
        binary = (img > 0.3).astype(np.float32)
        
        features = []
        
        # 1. 拓扑特征 (4D)
        n_holes = self._count_holes(binary)
        endpoints = self._find_endpoints(binary)
        junctions = self._find_junctions(binary)
        is_closed = 1.0 if n_holes > 0 else 0.0
        
        features.extend([
            n_holes / 3.0,
            len(endpoints) / 5.0,
            len(junctions) / 3.0,
            is_closed
        ])
        
        # 2. 端点位置 (9D)
        ep_regions = np.zeros(9)
        for y, x in endpoints:
            ry, rx = min(2, y // 10), min(2, x // 10)
            ep_regions[ry * 3 + rx] = 1.0
        features.extend(ep_regions)
        
        # 3. 交叉点位置 (3D)
        jc_regions = np.zeros(3)
        for y, x in junctions:
            region = min(2, y // 10)
            jc_regions[region] = 1.0
        features.extend(jc_regions)
        
        # 4. 边缘方向 (4D)
        h_top = np.sum(binary[:10, :]) / (10 * 28)
        h_bottom = np.sum(binary[18:, :]) / (10 * 28)
        v_left = np.sum(binary[:, :10]) / (28 * 10)
        v_right = np.sum(binary[:, 18:]) / (28 * 10)
        features.extend([h_top, h_bottom, v_left, v_right])
        
        # 5. 密度分布 (3D)
        density_top = np.sum(binary[:9, :]) / (9 * 28)
        density_mid = np.sum(binary[9:18, :]) / (9 * 28)
        density_bottom = np.sum(binary[18:, :]) / (10 * 28)
        features.extend([density_top, density_mid, density_bottom])
        
        # 6. 质心 (2D)
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            cx, cy = np.mean(xs) / 28.0, np.mean(ys) / 28.0
        else:
            cx, cy = 0.5, 0.5
        features.extend([cx, cy])
        
        return np.array(features, dtype=np.float32)
    
    def _count_holes(self, binary: np.ndarray) -> int:
        from collections import deque
        
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        visited = np.zeros_like(padded, dtype=bool)
        
        queue = deque([(0, 0)])
        visited[0, 0] = True
        
        while queue:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < padded.shape[0] and 0 <= nx < padded.shape[1]:
                    if not visited[ny, nx] and padded[ny, nx] == 0:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        holes = 0
        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if padded[y, x] == 0 and not visited[y, x]:
                    holes += 1
                    queue = deque([(y, x)])
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.popleft()
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if not visited[ny, nx] and padded[ny, nx] == 0:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
        
        return holes
    
    def _find_endpoints(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        endpoints = []
        skeleton = binary.copy()
        
        for y in range(1, binary.shape[0] - 1):
            for x in range(1, binary.shape[1] - 1):
                if skeleton[y, x] > 0:
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbors == 1:
                        endpoints.append((y, x))
        
        return endpoints[:5]
    
    def _find_junctions(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        junctions = []
        skeleton = binary.copy()
        
        for y in range(1, binary.shape[0] - 1):
            for x in range(1, binary.shape[1] - 1):
                if skeleton[y, x] > 0:
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbors >= 3:
                        junctions.append((y, x))
        
        return junctions[:3]


# =============================================================================
# 3. 共振记忆 (Resonant Memory)
# =============================================================================

class ResonantMemory:
    """
    共振记忆：基于余弦相似度的无梯度学习
    
    存储 key → Q-values 映射
    Q-values 长度 = n_actions（children 数量）
    """
    
    def __init__(
        self,
        key_dim: int,
        n_actions: int,
        capacity: int = 50,
        temperature: float = 0.1,
        lr: float = 0.3,
        similarity_threshold: float = 0.7
    ):
        self.key_dim = key_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.temperature = temperature
        self.lr = lr
        self.similarity_threshold = similarity_threshold
        
        # 记忆存储
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.access_counts: List[int] = []
        
        # 冻结状态
        self.frozen: bool = False
        
        # 统计
        self.total_queries = 0
        self.total_updates = 0
    
    def query(self, key: np.ndarray) -> Tuple[np.ndarray, float, int]:
        """
        共振查询
        
        Returns:
            q_values: 各 action 的 Q 值
            confidence: 置信度
            best_match_idx: 最佳匹配的记忆索引（-1 表示无匹配）
        """
        self.total_queries += 1
        
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0, -1
        
        # 计算相似度
        key_matrix = np.array(self.keys)
        scores = key_matrix @ key
        
        best_idx = int(np.argmax(scores))
        max_score = float(scores[best_idx])
        
        # Softmax 加权
        exp_scores = np.exp((scores - max_score) / self.temperature)
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        # 加权求和得到 Q-values
        value_matrix = np.array(self.values)
        q_values = weights @ value_matrix
        
        # 更新访问计数
        self.access_counts[best_idx] += 1
        
        # 置信度
        confidence = max(0.0, min(1.0, (max_score - 0.3) / 0.5))
        
        return q_values.astype(np.float32), confidence, best_idx
    
    def remember(
        self,
        key: np.ndarray,
        action: int,
        reward: float
    ) -> str:
        """
        写入记忆（TD 更新）
        
        Args:
            key: 编码后的 pattern
            action: 选择的动作（child 索引）
            reward: 奖励（+1 正确，-1 错误，0 中性）
        
        Returns:
            'update' / 'new' / 'frozen'
        """
        if self.frozen:
            return 'frozen'
        
        self.total_updates += 1
        
        # 查找相似记忆
        best_idx = -1
        best_score = -1.0
        
        if len(self.keys) > 0:
            key_matrix = np.array(self.keys)
            scores = key_matrix @ key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score < self.similarity_threshold:
                best_idx = -1
        
        if best_idx >= 0:
            # 更新现有记忆的 Q-value
            old_q = self.values[best_idx][action]
            # TD 更新：Q = Q + lr * (reward - Q)
            self.values[best_idx][action] = old_q + self.lr * (reward - old_q)
            self.access_counts[best_idx] += 1
            return 'update'
        else:
            # 容量管理
            if len(self.keys) >= self.capacity:
                min_idx = int(np.argmin(self.access_counts))
                self.keys.pop(min_idx)
                self.values.pop(min_idx)
                self.access_counts.pop(min_idx)
            
            # 创建新记忆
            new_q = np.zeros(self.n_actions, dtype=np.float32)
            new_q[action] = reward
            
            self.keys.append(key.copy())
            self.values.append(new_q)
            self.access_counts.append(1)
            return 'new'
    
    def freeze(self):
        self.frozen = True
    
    def unfreeze(self):
        self.frozen = False
    
    @property
    def size(self) -> int:
        return len(self.keys)
    
    def is_full(self) -> bool:
        return self.size >= self.capacity


# =============================================================================
# 4. 统一的 Structon（支持 Bottom-Up 生长）
# =============================================================================

class Structon:
    """
    统一的 Structon - 自相似分形单元
    
    三种角色：
    - Actuator: children 是字符串（动作），无 LRM
    - Atomic: children 是 Structon，LRM 学习路由
    - Composite: 冻结的 Atomic，只路由不学习
    
    关键：Wrapper 不知道底层标签，只学习路由策略
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        children: List[Union[str, 'Structon']],
        capacity: int = 50,
        key_dim: int = 16,
        full_dim: int = 25,
        temperature: float = 0.1,
        lr: float = 0.3,
        similarity_threshold: float = 0.7,
        projector: np.ndarray = None
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.children = children
        self.n_actions = len(children)
        
        # 参数存储
        self.capacity = capacity
        self.key_dim = key_dim
        self.full_dim = full_dim
        self.temperature = temperature
        self.lr = lr
        self.similarity_threshold = similarity_threshold
        
        # 随机投影
        if projector is not None:
            self.projector = projector.copy()
        else:
            self.projector = np.random.randn(full_dim, key_dim).astype(np.float32)
            self.projector /= (np.linalg.norm(self.projector, axis=0, keepdims=True) + 1e-8)
        
        # 共振记忆（只有非 Actuator 才需要）
        if self.is_actuator():
            self.lrm = None
        else:
            self.lrm = ResonantMemory(
                key_dim=key_dim,
                n_actions=self.n_actions,
                capacity=capacity,
                temperature=temperature,
                lr=lr,
                similarity_threshold=similarity_threshold
            )
        
        # 统计
        self.total_queries = 0
        self.promotes = 0
    
    def _encode(self, pattern: np.ndarray) -> np.ndarray:
        """随机投影编码"""
        key = pattern.astype(np.float32) @ self.projector
        norm = np.linalg.norm(key)
        if norm > 1e-8:
            key /= norm
        return key
    
    @property
    def frozen(self) -> bool:
        return self.lrm.frozen if self.lrm else False
    
    def freeze(self):
        if self.lrm:
            self.lrm.freeze()
    
    def unfreeze(self):
        if self.lrm:
            self.lrm.unfreeze()
    
    def is_actuator(self) -> bool:
        """是否是执行器（children 全是字符串）"""
        return all(isinstance(c, str) for c in self.children)
    
    def is_atomic(self) -> bool:
        """是否是学习中的中间节点"""
        return not self.is_actuator() and not self.frozen
    
    def is_composite(self) -> bool:
        """是否是冻结的中间节点"""
        return not self.is_actuator() and self.frozen
    
    def query(self, pattern: np.ndarray) -> Tuple[int, float]:
        """
        查询：选择哪个 child
        
        Returns:
            action_idx: 选择的 child 索引
            confidence: 置信度
        """
        self.total_queries += 1
        
        if self.is_actuator():
            # Actuator：随机或默认选择
            return 0, 1.0
        
        key = self._encode(pattern)
        q_values, confidence, _ = self.lrm.query(key)
        action_idx = int(np.argmax(q_values))
        return action_idx, confidence
    
    def execute(self, pattern: np.ndarray) -> Tuple[str, float]:
        """
        执行：递归查询直到得到动作
        
        Returns:
            action: 最终选择的动作（字符串）
            confidence: 置信度
        """
        if self.is_actuator():
            # Actuator：直接返回唯一的 action
            return self.children[0], 1.0
        
        action_idx, confidence = self.query(pattern)
        child = self.children[action_idx]
        
        if isinstance(child, str):
            return child, confidence
        else:
            return child.execute(pattern)
    
    def learn(
        self,
        pattern: np.ndarray,
        reward: float,
        chosen_idx: int
    ) -> str:
        """
        学习路由策略（不是学标签！）
        
        Args:
            pattern: 输入模式
            reward: 奖励（来自下游反馈）
            chosen_idx: 选择的 child 索引
        
        Returns:
            状态信息
        """
        if self.lrm is None or self.frozen:
            return 'skip'
        
        key = self._encode(pattern)
        return self.lrm.remember(key, chosen_idx, reward)
    
    def should_promote(self) -> bool:
        """是否需要晋升"""
        if self.lrm is None:
            return False
        return self.lrm.is_full() and not self.frozen
    
    def promote(self) -> 'Structon':
        """
        晋升：冻结自己，创建 Wrapper
        
        返回新的 Wrapper（应该替换 self 成为新的 root）
        """
        self.promotes += 1
        
        # 1. 冻结自己
        self.freeze()
        
        # 2. 创建空的 sibling（不同的投影！）
        sibling = Structon(
            children=self.children.copy() if self.is_actuator() else [c for c in self.children],
            capacity=self.capacity,
            key_dim=self.key_dim,
            full_dim=self.full_dim,
            temperature=self.temperature,
            lr=self.lr,
            similarity_threshold=self.similarity_threshold,
            projector=None  # 新的随机投影
        )
        
        # 3. 创建 Wrapper
        wrapper = Structon(
            children=[self, sibling],  # 2 个 children
            capacity=self.capacity,
            key_dim=self.key_dim,
            full_dim=self.full_dim,
            temperature=self.temperature,
            lr=self.lr,
            similarity_threshold=self.similarity_threshold,
            projector=self.projector.copy()  # 继承投影（保持路由一致）
        )
        
        return wrapper
    
    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            if isinstance(child, Structon):
                count += child.count_nodes()
        return count
    
    def depth(self) -> int:
        if self.is_actuator():
            return 1
        
        max_child_depth = 0
        for child in self.children:
            if isinstance(child, Structon):
                max_child_depth = max(max_child_depth, child.depth())
        
        return 1 + max_child_depth
    
    def total_memories(self) -> int:
        count = self.lrm.size if self.lrm else 0
        for child in self.children:
            if isinstance(child, Structon):
                count += child.total_memories()
        return count
    
    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        
        if self.is_actuator():
            icon = "⚡"
            role = "Actuator"
            mem_info = f"actions={self.children}"
        elif self.frozen:
            icon = "❄️"
            role = "Composite"
            mem_info = f"mem:{self.lrm.size}/{self.capacity}"
        else:
            icon = "🔥"
            role = "Atomic"
            mem_info = f"mem:{self.lrm.size}/{self.capacity}"
        
        print(f"{prefix}{icon} {self.id} ({role}) [{mem_info}]")
        
        for child in self.children:
            if isinstance(child, Structon):
                child.print_tree(indent + 1)


# =============================================================================
# 5. Vision System（支持 Bottom-Up 生长）
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统 - MNIST 分类
    
    关键改变：
    1. 从 10 个 Actuator 开始
    2. 学习时通过强化反馈训练路由
    3. 满了自动 promote
    """
    
    def __init__(
        self,
        capacity: int = 50,
        key_dim: int = 16,
        full_dim: int = 25,
        temperature: float = 0.1,
        lr: float = 0.3,
        similarity_threshold: float = 0.7
    ):
        self.extractor = StateExtractor()
        
        # 创建 10 个 Actuator（每个代表一个数字）
        actuators = []
        for i in range(10):
            act = Structon(
                children=[str(i)],  # 单一动作
                capacity=capacity,
                key_dim=key_dim,
                full_dim=full_dim
            )
            actuators.append(act)
        
        # 创建 root（路由到 10 个 Actuator）
        self.root = Structon(
            children=actuators,
            capacity=capacity,
            key_dim=key_dim,
            full_dim=full_dim,
            temperature=temperature,
            lr=lr,
            similarity_threshold=similarity_threshold
        )
        
        # 参数
        self.capacity = capacity
        self.key_dim = key_dim
        self.full_dim = full_dim
        
        # 统计
        self.train_count = 0
        self.correct_count = 0
        self.promote_count = 0
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        state = self.extractor.extract(image)
        return self.root.execute(state)
    
    def train_one(
        self,
        image: np.ndarray,
        true_label: str,
        explore_rate: float = 0.1
    ) -> Tuple[bool, str]:
        """
        训练一个样本
        
        关键：用强化学习方式训练路由
        """
        self.train_count += 1
        state = self.extractor.extract(image)
        
        # 1. 查询路由决策
        chosen_idx, confidence = self.root.query(state)
        
        # 2. 探索：有一定概率随机选择
        if np.random.random() < explore_rate:
            chosen_idx = np.random.randint(self.root.n_actions)
        
        # 3. 执行选择的 child
        child = self.root.children[chosen_idx]
        predicted, _ = child.execute(state)
        
        # 4. 计算奖励
        correct = (predicted == true_label)
        if correct:
            self.correct_count += 1
            reward = 1.0
        else:
            reward = -0.5  # 惩罚但不要太重
        
        # 5. 学习路由决策
        status = self.root.learn(state, reward, chosen_idx)
        
        # 6. 检查是否需要 promote
        if self.root.should_promote():
            self.root = self.root.promote()
            self.promote_count += 1
            status = f"promoted ({self.promote_count})"
        
        return correct, status
    
    def print_stats(self):
        print(f"\n=== Structon Vision System ===")
        print(f"训练样本: {self.train_count}")
        if self.train_count > 0:
            print(f"训练准确率: {self.correct_count/self.train_count*100:.1f}%")
        print(f"晋升次数: {self.promote_count}")
        print(f"节点数: {self.root.count_nodes()}")
        print(f"深度: {self.root.depth()}")
        print(f"总记忆: {self.root.total_memories()}")
        
        print(f"\n=== 树结构 ===")
        self.root.print_tree()


# =============================================================================
# 6. 实验
# =============================================================================

def run_experiment(
    n_per_class: int = 50,
    n_test: int = 500,
    capacity: int = 50,
    key_dim: int = 16,
    temperature: float = 0.1,
    lr: float = 0.3,
    similarity_threshold: float = 0.7,
    explore_rate: float = 0.1,
    epochs: int = 3
):
    """运行实验"""
    print("=" * 70)
    print("Structon Vision v7.21 - 真正的 Bottom-Up 分形架构")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  capacity={capacity}, key_dim={key_dim}")
    print(f"  temperature={temperature}, lr={lr}")
    print(f"  similarity_threshold={similarity_threshold}")
    print(f"  explore_rate={explore_rate}, epochs={epochs}")
    print(f"  每类训练: {n_per_class}, 测试: {n_test}")
    
    print("\n核心改变:")
    print("  1. 动态晋升：记忆满了 → 冻结 → 创建 Wrapper")
    print("  2. 路由学习：Wrapper 学习'送给哪个 child'，不学标签")
    print("  3. 强化反馈：child 预测对了 → 强化路由")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        capacity=capacity,
        key_dim=key_dim,
        full_dim=25,
        temperature=temperature,
        lr=lr,
        similarity_threshold=similarity_threshold
    )
    
    # 收集训练样本
    train_indices = []
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        train_indices.extend(indices)
    
    total_samples = len(train_indices)
    
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # 打乱顺序
        np.random.shuffle(train_indices)
        
        epoch_correct = 0
        t0 = time.time()
        
        for i, idx in enumerate(train_indices):
            correct, status = system.train_one(
                train_images[idx],
                str(train_labels[idx]),
                explore_rate=explore_rate
            )
            if correct:
                epoch_correct += 1
            
            if (i + 1) % 100 == 0:
                acc = epoch_correct / (i + 1) * 100
                mem = system.root.lrm.size if system.root.lrm else 0
                print(f"  训练 {i+1}/{total_samples}, "
                      f"准确率: {acc:.1f}%, "
                      f"记忆: {mem}/{capacity}, "
                      f"晋升: {system.promote_count}")
        
        print(f"Epoch {epoch+1} 完成: {time.time()-t0:.1f}秒, "
              f"准确率: {epoch_correct/total_samples*100:.1f}%")
        
        # 逐渐降低探索率
        explore_rate *= 0.7
    
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试 {n_test} 样本 ===")
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    t0 = time.time()
    for idx in test_indices:
        predicted, confidence = system.predict(test_images[idx])
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


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=50)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--key-dim', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--explore', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=3)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        key_dim=args.key_dim,
        temperature=args.temperature,
        lr=args.lr,
        similarity_threshold=args.threshold,
        explore_rate=args.explore,
        epochs=args.epochs
    )
