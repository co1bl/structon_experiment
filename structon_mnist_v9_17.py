#!/usr/bin/env python3
"""
Structon Vision v9.17 - Structon Network as Deep Learning

核心思想：
- 单个 Structon = 微型 NN (特征变换) + LRM (决策)
- 多个 Structon 连接 = 深度网络
- 信号流经的路径 = 网络的深度
- 深度从路由中涌现！

架构：
传统 DL:  Layer1 → Layer2 → Layer3 (固定深度)
Structon: S_a → S_b → S_c → ... (动态深度，路由决定)

每个 Structon:
┌─────────────────────────┐
│  输入 features          │
│         ↓               │
│  ┌─────────────────┐    │
│  │   微型 NN       │    │
│  │  (Hebbian学习)  │    │
│  │  in → hidden    │    │
│  └────────┬────────┘    │
│           ↓             │
│    transformed_features │
│           ↓             │
│  ┌─────────────────┐    │
│  │      LRM        │    │
│  │  (RL 决策)      │    │
│  │  SELF / 路由    │    │
│  └─────────────────┘    │
│           ↓             │
│  输出到下一个 Structon   │
└─────────────────────────┘

信号流: 
  输入 → S1 → S2 → S5 → S3(SELF) → 输出
              ↓
         4层深度网络！
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Optional
import argparse

# =============================================================================
# 1. MNIST 加载
# =============================================================================
def load_mnist():
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    data = {}
    mnist_dir = os.path.expanduser('~/.mnist')
    os.makedirs(mnist_dir, exist_ok=True)
    
    for key, filename in files.items():
        filepath = os.path.join(mnist_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                f.read(16)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            else:
                f.read(8)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

# =============================================================================
# 2. 简单特征提取（作为第一层输入）
# =============================================================================
class StateExtractor:
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        self.output_dim = None
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                grid[i, j] = np.mean(block)
        
        # Contrast Binary
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                features.append(-1.0 if grid[i, j] < self.threshold else 1.0)
        
        # 水平差异
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                features.append((grid[i, j] - grid[i, j+1]) * 2)
        
        # 垂直差异
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                features.append((grid[i, j] - grid[i+1, j]) * 2)
        
        # 对角线
        diag1 = np.mean([grid[i, i] for i in range(self.grid_size)])
        diag2 = np.mean([grid[i, self.grid_size-1-i] for i in range(self.grid_size)])
        features.extend([(diag1 - 0.5) * 2, (diag2 - 0.5) * 2, (diag1 - diag2) * 2])
        
        # 四象限
        mid = self.grid_size // 2
        q1, q2 = np.mean(grid[:mid, :mid]), np.mean(grid[:mid, mid:])
        q3, q4 = np.mean(grid[mid:, :mid]), np.mean(grid[mid:, mid:])
        features.extend([(q1-q4)*2, (q2-q3)*2, (q1+q4-q2-q3)*2, ((q1+q2)-(q3+q4))*2])
        
        # 边缘
        top, bottom = np.mean(grid[0, :]), np.mean(grid[-1, :])
        left, right = np.mean(grid[:, 0]), np.mean(grid[:, -1])
        features.extend([(top-bottom)*2, (left-right)*2, (top-0.5)*2, (bottom-0.5)*2])
        
        state = np.array(features, dtype=np.float32)
        
        if self.output_dim is None:
            self.output_dim = len(state)
        
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state

# =============================================================================
# 3. 微型 NN (Hebbian 学习)
# =============================================================================
class MicroNN:
    """
    微型神经网络 - 单层 + ReLU
    使用 Hebbian 学习（本地，无 backprop）
    
    功能：特征变换
    输入维度 = 输出维度（保持一致，方便信号流动）
    """
    
    def __init__(self, dim: int, compression: float = 0.5):
        self.dim = dim
        self.hidden_dim = max(16, int(dim * compression))
        
        # 权重初始化
        self.W1 = np.random.randn(dim, self.hidden_dim).astype(np.float32) * 0.1
        self.W2 = np.random.randn(self.hidden_dim, dim).astype(np.float32) * 0.1
        
        # Hebbian 学习率
        self.lr = 0.01
        
        # 缓存（用于 Hebbian 更新）
        self._input = None
        self._hidden = None
        self._output = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self._input = x
        
        # 隐藏层
        h = x @ self.W1
        h = np.maximum(0, h)  # ReLU
        self._hidden = h
        
        # 输出层
        out = h @ self.W2
        out = np.tanh(out)  # 输出归一化到 [-1, 1]
        self._output = out
        
        # 归一化
        norm = np.linalg.norm(out)
        if norm > 1e-6:
            out = out / norm
        
        return out
    
    def hebbian_learn(self, is_positive: bool):
        """
        Hebbian 学习
        正样本：加强激活的连接
        负样本：减弱激活的连接
        """
        if self._input is None or self._hidden is None:
            return
        
        sign = 1.0 if is_positive else -1.0
        
        # 更新 W1: 输入-隐藏
        # "一起激活，一起加强"
        active_hidden = (self._hidden > 0).astype(np.float32)
        delta_W1 = sign * self.lr * np.outer(self._input, active_hidden * self._hidden)
        self.W1 += delta_W1
        
        # 权重衰减（防止爆炸）
        self.W1 *= 0.999
        
        # 更新 W2: 隐藏-输出
        if self._output is not None:
            delta_W2 = sign * self.lr * np.outer(self._hidden, self._output)
            self.W2 += delta_W2
            self.W2 *= 0.999

# =============================================================================
# 4. LRM (决策记忆)
# =============================================================================
class LRM:
    """
    Local Resonant Memory - 用于决策
    本质是本地 RL (Q-learning)
    """
    
    def __init__(self, n_actions: int, capacity=300, top_k=5):
        self.n_actions = n_actions
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.85

    def query(self, features) -> Tuple[np.ndarray, float]:
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.array([np.dot(k, features) for k in self.keys])
        
        k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        valid_mask = top_scores > 0.1
        if not np.any(valid_mask):
            return np.zeros(self.n_actions), 0.0
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        weights = top_scores ** 2
        weights /= np.sum(weights)
        
        q_values = np.zeros(self.n_actions)
        for idx, w in zip(top_indices, weights):
            q_values += w * self.values[idx]
        
        return q_values, float(np.max(top_scores))

    def update(self, features, action: int, reward: float):
        if action >= self.n_actions:
            return
            
        if self.keys:
            scores = np.array([np.dot(k, features) for k in self.keys])
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                self.access_counts[best_idx] += 1
                return

        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
            
        self.keys.append(features.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        
    @property
    def size(self):
        return len(self.keys)

# =============================================================================
# 5. Structon = 微型 NN + LRM
# =============================================================================
class Structon:
    """
    Structon = 微型 NN + LRM
    
    微型 NN: 特征变换（Hebbian 学习）
    LRM: 决策（RL 学习）
    
    信号流经多个 Structon = 深度网络
    """
    _id_counter = 0
    
    def __init__(self, label: str, variant_id: int, feature_dim: int,
                 capacity=300, n_connections=5):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.variant_id = variant_id
        self.full_label = f"{label}_{variant_id}"
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        # 微型 NN（特征变换）
        self.micro_nn = MicroNN(feature_dim)
        self.feature_dim = feature_dim
        
        # LRM（决策）- 延迟初始化
        self.lrm: Optional[LRM] = None
        self.capacity = capacity
        
        self.confidence_threshold = 0.5
        
        self.stats = {
            'hebbian_positive': 0,
            'hebbian_negative': 0,
            'self_confident': 0,
            'self_uncertain': 0,
            'route_accepted': 0,
            'route_continued': 0,
            'route_deadend': 0,
            'route_revisit': 0,
        }
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
        n_actions = 1 + len(self.connections)
        self.lrm = LRM(n_actions=n_actions, capacity=self.capacity)

    def transform(self, features: np.ndarray) -> np.ndarray:
        """用微型 NN 变换特征"""
        return self.micro_nn.forward(features)
    
    def get_q_values(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """获取决策 Q 值"""
        if self.lrm is None:
            return np.zeros(1), 0.0
        transformed = self.transform(features)
        return self.lrm.query(transformed)

    def decide(self, features: np.ndarray, visited_ids: set) -> Tuple[int, Optional['Structon'], float, np.ndarray]:
        """
        决策并返回变换后的特征
        返回: (action, next_structon, confidence, transformed_features)
        """
        if self.lrm is None:
            return 0, None, 0.0, features
        
        # 变换特征
        transformed = self.transform(features)
        
        # 查询 LRM
        q_values, resonance = self.lrm.query(transformed)
        
        # 屏蔽已访问
        adjusted_q = q_values.copy()
        for i, conn in enumerate(self.connections):
            if conn.id in visited_ids:
                adjusted_q[i + 1] = -100.0
        
        best_action = int(np.argmax(adjusted_q))
        confidence = adjusted_q[best_action]
        
        if best_action == 0:
            return 0, None, confidence, transformed
        else:
            return best_action, self.connections[best_action - 1], confidence, transformed

    # =========================================================================
    # 学习方法
    # =========================================================================
    
    def learn_hebbian(self, is_positive: bool):
        """Hebbian 学习微型 NN"""
        self.micro_nn.hebbian_learn(is_positive)
        if is_positive:
            self.stats['hebbian_positive'] += 1
        else:
            self.stats['hebbian_negative'] += 1
    
    def learn_self_confident(self, features: np.ndarray, confidence: float):
        transformed = self.transform(features)
        if confidence > self.confidence_threshold:
            self.lrm.update(transformed, 0, 1.2)
            self.stats['self_confident'] += 1
        else:
            self.lrm.update(transformed, 0, 0.3)
            self.stats['self_uncertain'] += 1
    
    def learn_route_accepted(self, features: np.ndarray, action: int, downstream_confidence: float):
        transformed = self.transform(features)
        reward = 0.5 + 0.5 * min(downstream_confidence / self.confidence_threshold, 1.0)
        self.lrm.update(transformed, action, reward)
        self.stats['route_accepted'] += 1
    
    def learn_route_continued(self, features: np.ndarray, action: int):
        transformed = self.transform(features)
        self.lrm.update(transformed, action, 0.2)
        self.stats['route_continued'] += 1
    
    def learn_route_deadend(self, features: np.ndarray, action: int):
        transformed = self.transform(features)
        self.lrm.update(transformed, action, -0.8)
        self.stats['route_deadend'] += 1
    
    def learn_route_revisit(self, features: np.ndarray, action: int):
        transformed = self.transform(features)
        self.lrm.update(transformed, action, -1.0)
        self.stats['route_revisit'] += 1

    def train_supervised(self, features: np.ndarray, is_me: bool):
        """监督训练"""
        # 1. 微型 NN Hebbian 学习
        _ = self.transform(features)  # 前向传播
        self.learn_hebbian(is_me)
        
        # 2. LRM 学习
        transformed = self.transform(features)
        
        if is_me:
            self.lrm.update(transformed, 0, 1.5)
            for i in range(1, self.lrm.n_actions):
                self.lrm.update(transformed, i, -0.5)
        else:
            self.lrm.update(transformed, 0, -0.8)
            for i in range(1, self.lrm.n_actions):
                self.lrm.update(transformed, i, 0.3)

    @property
    def memory_size(self):
        return self.lrm.size if self.lrm else 0

# =============================================================================
# 6. Vision System - Structon Network as Deep Learning
# =============================================================================
class StructonVisionSystem:
    """
    Structon 网络 = 深度学习
    
    信号流经的路径决定了"深度"
    每个 Structon 是一"层"的计算
    """
    
    def __init__(self, n_variants=10, capacity=300, n_connections=5):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structons: Dict[str, List[Structon]] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        self.n_variants = n_variants
        self.feature_dim = None
        
    def build(self, labels, sample_image):
        sample_state = self.extractor.extract(sample_image)
        self.feature_dim = len(sample_state)
        
        n_total = len(labels) * self.n_variants
        print(f"\n=== 创建 Structon v9.17 (Network as Deep Learning, N={n_total}) ===")
        print(f"特征维度: {self.feature_dim}")
        print(f"每个数字 {self.n_variants} 个 Structon")
        Structon._id_counter = 0
        
        self.structons = []
        self.label_to_structons = {label: [] for label in labels}
        
        for label in labels:
            for v in range(self.n_variants):
                s = Structon(
                    label, v, self.feature_dim,
                    self.capacity, self.n_connections
                )
                self.structons.append(s)
                self.label_to_structons[label].append(s)
        
        print(f"  总共 {len(self.structons)} 个 Structon")
        print(f"  每个 Structon = 微型 NN ({self.feature_dim}→{self.structons[0].micro_nn.hidden_dim}→{self.feature_dim}) + LRM")
            
        print(f"\n设置稀疏连接 (每个 {self.n_connections} 条)...")
        for s in self.structons:
            s.set_connections(self.structons)
        
        print("  连接示例:")
        for s in self.structons[:3]:
            conns = [c.full_label for c in s.connections[:3]]
            print(f"    {s.full_label} → {conns}...")
            
    def train_epoch(self, samples, n_negatives=5):
        np.random.shuffle(samples)
        
        for img, label, variant_idx in samples:
            state = self.extractor.extract(img)
            
            correct_s = self.label_to_structons[label][variant_idx]
            for _ in range(2):
                correct_s.train_supervised(state, is_me=True)
            
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_supervised(state, is_me=False)

    def predict(self, image, max_hops=30, local_learning=False) -> Tuple[str, List[str], int]:
        """
        路由预测
        
        信号流: 输入 → S1 → S2 → S5 → S3(SELF) → 输出
        每经过一个 Structon，特征被变换一次
        路径长度 = 网络深度！
        """
        state = self.extractor.extract(image)
        features = state  # 初始特征
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        queries = 0
        
        for step in range(max_hops):
            visited.add(current.id)
            path.append(current.full_label)
            queries += 1
            
            # 决策（同时变换特征）
            action, next_s, confidence, transformed = current.decide(features, visited)
            
            if action == 0:  # SELF
                if local_learning:
                    current.learn_self_confident(features, confidence)
                    current.learn_hebbian(True)  # 正样本
                return current.label, path, queries
            
            # 路由
            if next_s is not None:
                if next_s.id in visited:
                    if local_learning:
                        current.learn_route_revisit(features, action)
                    
                    found_alt = False
                    for i, conn in enumerate(current.connections):
                        if conn.id not in visited:
                            next_s = conn
                            action = i + 1
                            found_alt = True
                            break
                    
                    if not found_alt:
                        next_s = None
                
                if next_s is not None:
                    # 获取下游信息
                    downstream_q, _ = next_s.get_q_values(transformed)
                    downstream_self_q = downstream_q[0]
                    
                    downstream_visited = visited | {next_s.id}
                    has_path = any(c.id not in downstream_visited for c in next_s.connections)
                    
                    if local_learning:
                        if downstream_self_q > next_s.confidence_threshold:
                            current.learn_route_accepted(features, action, downstream_self_q)
                        elif has_path:
                            current.learn_route_continued(features, action)
                        else:
                            current.learn_route_deadend(features, action)
                    
                    # 传递变换后的特征给下一个 Structon！
                    features = transformed
                    current = next_s
                    continue
            
            # 跳转
            all_unvisited = [s for s in self.structons if s.id not in visited]
            if all_unvisited:
                current = np.random.choice(all_unvisited)
                path.append("JUMP")
            else:
                break
        
        return current.label, path, queries

    def train_local_epoch(self, samples):
        np.random.shuffle(samples)
        for img, label, variant_idx in samples:
            self.predict(img, local_learning=True)

    def predict_voting(self, image) -> Tuple[str, Dict[str, float], int]:
        state = self.extractor.extract(image)
        
        votes = {}
        queries = 0
        
        for s in self.structons:
            queries += 1
            if s.lrm is None:
                continue
            
            q_values, _ = s.get_q_values(state)
            self_q = q_values[0]
            
            if s.label not in votes:
                votes[s.label] = 0
            if self_q > 0:
                votes[s.label] += self_q
        
        if not votes or max(votes.values()) <= 0:
            all_scores = {}
            for s in self.structons:
                q_values, _ = s.get_q_values(state)
                if s.label not in all_scores:
                    all_scores[s.label] = []
                all_scores[s.label].append(q_values[0])
            for label in all_scores:
                votes[label] = max(all_scores[label])
        
        best = max(votes, key=votes.get)
        return best, votes, queries

    def print_stats(self):
        print("\n" + "=" * 70)
        print(f"Structon Vision v9.17 - Network as Deep Learning (N={len(self.structons)})")
        print("=" * 70)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"每个数字: {self.n_variants} 个变体")
        print(f"连接数/节点: {self.n_connections}")
        
        total_mem = sum(s.memory_size for s in self.structons)
        print(f"总 LRM 记忆: {total_mem}")
        
        total_heb_pos = sum(s.stats['hebbian_positive'] for s in self.structons)
        total_heb_neg = sum(s.stats['hebbian_negative'] for s in self.structons)
        print(f"总 Hebbian 学习: 正={total_heb_pos}, 负={total_heb_neg}")
        
        print("\n=== 按数字统计 ===")
        print(f"{'数字':<6} {'LRM':<8} {'Heb+':<8} {'Heb-':<8} {'Self+':<8}")
        print("-" * 45)
        for label in sorted(self.label_to_structons.keys()):
            structons = self.label_to_structons[label]
            total_mem = sum(s.memory_size for s in structons)
            total_heb_pos = sum(s.stats['hebbian_positive'] for s in structons)
            total_heb_neg = sum(s.stats['hebbian_negative'] for s in structons)
            total_self = sum(s.stats['self_confident'] for s in structons)
            print(f"  {label:<4} {total_mem:<8} {total_heb_pos:<8} {total_heb_neg:<8} {total_self:<8}")

# =============================================================================
# 7. 主实验
# =============================================================================
def run_experiment(
    n_variants: int = 10,
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 300,
    epochs: int = 30,
    n_connections: int = 5,
    n_negatives: int = 5,
    local_epochs: int = 10
):
    n_total = 10 * n_variants
    print("=" * 70)
    print(f"Structon Vision v9.17 - Network as Deep Learning (N={n_total})")
    print("=" * 70)
    print("\n核心思想:")
    print("  单个 Structon = 微型 NN (Hebbian) + LRM (RL)")
    print("  Structon 网络 = 深度学习")
    print("  路径长度 = 网络深度 (动态涌现!)")
    print(f"\n参数: capacity={capacity}, 每类样本={n_per_class}, 连接数={n_connections}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        n_variants=n_variants,
        capacity=capacity,
        n_connections=n_connections
    )
    system.build([str(i) for i in range(10)], train_images[0])
    
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:n_per_class]
        for i, idx in enumerate(idxs):
            variant_idx = i % n_variants
            samples.append((train_images[idx], str(d), variant_idx))
    
    print(f"\n训练样本: {len(samples)}")
    
    # 阶段1：监督训练
    print(f"\n=== 阶段1: 监督训练 ({epochs} epochs) ===")
    t0 = time.time()
    
    for ep in range(epochs):
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (ep + 1) % 5 == 0:
            correct = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, _, _ = system.predict_voting(test_images[i])
                if pred == str(test_labels[i]):
                    correct += 1
            print(f"  轮次 {ep+1}: 投票准确率={correct/check_n*100:.1f}%")
    
    print(f"\n监督训练: {time.time()-t0:.1f}秒")
    
    # 阶段2：本地学习
    print(f"\n=== 阶段2: 本地学习 ({local_epochs} epochs) ===")
    print("信号流经 Structon 网络，深度动态涌现！")
    t1 = time.time()
    
    for ep in range(local_epochs):
        system.train_local_epoch(samples)
        
        if (ep + 1) % 2 == 0:
            correct = 0
            path_lens = []
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, path, _ = system.predict(test_images[i])
                path_lens.append(len(path))
                if pred == str(test_labels[i]):
                    correct += 1
            avg_depth = np.mean(path_lens)
            print(f"  轮次 {ep+1}: 路由准确率={correct/check_n*100:.1f}%, 平均深度={avg_depth:.1f}")
    
    print(f"\n本地学习: {time.time()-t1:.1f}秒")
    
    system.print_stats()
    
    # 最终测试
    print(f"\n=== 最终测试 ===")
    test_idxs = np.random.choice(len(test_images), n_test, replace=False)
    
    # 路由测试
    correct_route = 0
    route_queries = 0
    path_lengths = []
    
    t_route = time.time()
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, path, queries = system.predict(test_images[idx])
        route_queries += queries
        path_lengths.append(len(path))
        if pred == true_label:
            correct_route += 1
    t_route = time.time() - t_route
    
    print(f"\n路由预测 (深度网络):")
    print(f"  准确率: {correct_route/n_test*100:.1f}%")
    print(f"  平均查询: {route_queries/n_test:.1f}")
    print(f"  平均深度: {np.mean(path_lengths):.2f}")
    print(f"  最大深度: {max(path_lengths)}")
    print(f"  最小深度: {min(path_lengths)}")
    print(f"  时间: {t_route:.2f}秒")
    
    # 投票测试
    correct_vote = 0
    vote_queries = 0
    
    t_vote = time.time()
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, _, queries = system.predict_voting(test_images[idx])
        vote_queries += queries
        if pred == true_label:
            correct_vote += 1
    t_vote = time.time() - t_vote
    
    print(f"\n投票预测 (单层):")
    print(f"  准确率: {correct_vote/n_test*100:.1f}%")
    print(f"  平均查询: {vote_queries/n_test:.1f}")
    print(f"  时间: {t_vote:.2f}秒")
    
    # 对比
    print("\n=== 对比 ===")
    print(f"{'方法':<12} {'准确率':<10} {'查询数':<10} {'深度':<10}")
    print("-" * 45)
    print(f"{'路由(深度)':<12} {correct_route/n_test*100:<10.1f}% {route_queries/n_test:<10.1f} {np.mean(path_lengths):<10.1f}")
    print(f"{'投票(单层)':<12} {correct_vote/n_test*100:<10.1f}% {vote_queries/n_test:<10.1f} {'1.0':<10}")
    
    # 示例路径
    print("\n=== 示例路径 (信号流 = 深度网络) ===")
    for i in range(5):
        idx = test_idxs[i]
        pred, path, queries = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        depth = len(path)
        path_str = ' → '.join(path[:6])
        if len(path) > 6:
            path_str += f" ..."
        print(f"  真实={true}, 预测={pred} {status}, 深度={depth}, 路径: {path_str}")
    
    return system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=int, default=10)
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=300)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--connections', type=int, default=5)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--local-epochs', type=int, default=10)
    args = parser.parse_args()
    
    run_experiment(
        n_variants=args.variants,
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives,
        local_epochs=args.local_epochs
    )
