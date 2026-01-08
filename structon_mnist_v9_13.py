#!/usr/bin/env python3
"""
Structon Vision v9.13 - Auto Sense + Connection as Action

核心改进：
1. 自动感知：每次决策前，自动获取邻居的 SELF Q 值
2. Extended State：将邻居共鸣编码进 state
3. Connection as Action：决策仍然通过 LRM 学习

state = [原始特征, 邻居0的SELF_Q, 邻居1的SELF_Q, 邻居2的SELF_Q]

这样：
- 感知是自动的（不是动作）
- 但感知结果影响学习（通过 extended_state）
- 保持 "connection as action" 的纯粹性
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
# 2. 特征提取
# =============================================================================
class StateExtractor:
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        
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
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state

# =============================================================================
# 3. SELF LRM - 只用于 SELF 判断（不含邻居信息）
# =============================================================================
class SelfLRM:
    """
    专门用于 SELF 判断的 LRM
    输入是原始 state（不含邻居信息）
    """
    
    def __init__(self, capacity=300, top_k=5):
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []  # 只存 SELF 的 Q 值（标量）
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88

    def query(self, state) -> Tuple[float, float]:
        """返回 (self_q, confidence)"""
        if not self.keys:
            return 0.0, 0.0
        
        scores = np.array([np.dot(k, state) for k in self.keys])
        
        k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        valid_mask = top_scores > 0.1
        if not np.any(valid_mask):
            return 0.0, 0.0
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        weights = top_scores ** 2
        weights /= np.sum(weights)
        
        self_q = sum(w * self.values[idx] for idx, w in zip(top_indices, weights))
        
        return float(self_q), float(np.max(top_scores))

    def update(self, state, reward: float):
        """更新 SELF Q 值"""
        if self.keys:
            scores = np.array([np.dot(k, state) for k in self.keys])
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx]
                self.values[best_idx] = old_q + self.learning_rate * (reward - old_q)
                self.access_counts[best_idx] += 1
                return

        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
            
        self.keys.append(state.copy())
        self.values.append(reward)
        self.access_counts.append(1)
        
    @property
    def size(self):
        return len(self.keys)

# =============================================================================
# 4. Routing LRM - 用于路由决策（含邻居信息）
# =============================================================================
class RoutingLRM:
    """
    路由决策 LRM
    输入是 extended_state（原始特征 + 邻居 SELF Q 值）
    输出是每个动作的 Q 值
    
    动作空间：
    - 动作 0: SELF
    - 动作 1~N: 路由到连接 0~(N-1)
    """
    
    def __init__(self, n_connections: int, capacity=300, top_k=5):
        self.n_actions = 1 + n_connections
        self.n_connections = n_connections
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.85  # 稍低，因为 extended_state 更长

    def query(self, extended_state) -> Tuple[np.ndarray, float]:
        """查询所有动作的 Q 值"""
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.array([np.dot(k, extended_state) for k in self.keys])
        
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

    def update(self, extended_state, action: int, reward: float):
        """更新路由 Q 值"""
        if action >= self.n_actions:
            return
            
        if self.keys:
            scores = np.array([np.dot(k, extended_state) for k in self.keys])
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
            
        self.keys.append(extended_state.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        
    @property
    def size(self):
        return len(self.keys)

# =============================================================================
# 5. Structon - 自动感知 + Connection as Action
# =============================================================================
class Structon:
    """
    Structon with Auto Sense
    
    两个 LRM：
    1. self_lrm: 判断 SELF（输入：原始 state）
    2. routing_lrm: 路由决策（输入：extended_state = state + 邻居共鸣）
    """
    _id_counter = 0
    
    def __init__(self, label: str, capacity=300, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        # SELF 判断（用原始 state）
        self.self_lrm = SelfLRM(capacity=capacity)
        
        # 路由决策（用 extended_state）
        self.routing_lrm: Optional[RoutingLRM] = None
        
        # 本地学习参数
        self.confidence_threshold = 0.5
        
        # 统计
        self.stats = {
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
        
        # 初始化路由 LRM
        self.routing_lrm = RoutingLRM(
            n_connections=len(self.connections),
            capacity=300,
            top_k=5
        )

    def get_self_q(self, state) -> float:
        """获取 SELF Q 值（供邻居感知）"""
        self_q, _ = self.self_lrm.query(state)
        return self_q

    def build_extended_state(self, state) -> np.ndarray:
        """
        构建 extended_state
        = 原始 state + 邻居的 SELF Q 值（归一化）
        """
        # 获取邻居共鸣
        neighbor_qs = []
        for conn in self.connections:
            q = conn.get_self_q(state)
            neighbor_qs.append(q)
        
        # 归一化邻居 Q 值到 [-1, 1] 范围
        neighbor_qs = np.array(neighbor_qs, dtype=np.float32)
        max_abs = np.max(np.abs(neighbor_qs)) if len(neighbor_qs) > 0 else 1.0
        if max_abs > 1e-6:
            neighbor_qs = neighbor_qs / max_abs
        
        # 拼接
        extended = np.concatenate([state, neighbor_qs])
        
        # 重新归一化整个 extended_state
        norm = np.linalg.norm(extended)
        if norm > 1e-6:
            extended = extended / norm
        
        return extended

    def decide(self, state, visited_ids: set) -> Tuple[int, Optional['Structon'], float]:
        """
        决策：使用 extended_state
        """
        if self.routing_lrm is None:
            return 0, None, 0.0
        
        # 构建 extended_state（自动感知邻居）
        extended_state = self.build_extended_state(state)
        
        # 查询路由 Q 值
        q_values, confidence = self.routing_lrm.query(extended_state)
        
        # 屏蔽已访问的连接
        adjusted_q = q_values.copy()
        for i, conn in enumerate(self.connections):
            if conn.id in visited_ids:
                adjusted_q[i + 1] = -100.0
        
        best_action = int(np.argmax(adjusted_q))
        best_q = adjusted_q[best_action]
        
        if best_action == 0:
            return 0, None, best_q
        else:
            return best_action, self.connections[best_action - 1], best_q

    def update_routing(self, state, action: int, reward: float):
        """更新路由（使用 extended_state）"""
        if self.routing_lrm is not None:
            extended_state = self.build_extended_state(state)
            self.routing_lrm.update(extended_state, action, reward)

    # =========================================================================
    # 本地学习方法
    # =========================================================================
    
    def learn_self_confident(self, state, confidence: float):
        """本地规则：高置信度说 SELF"""
        if confidence > self.confidence_threshold:
            self.self_lrm.update(state, 1.2)
            self.update_routing(state, 0, 1.2)
            self.stats['self_confident'] += 1
        else:
            self.self_lrm.update(state, 0.3)
            self.update_routing(state, 0, 0.3)
            self.stats['self_uncertain'] += 1
    
    def learn_route_accepted(self, state, action: int, downstream_confidence: float):
        """本地规则：下游接受了（说 SELF）"""
        reward = 0.5 + 0.5 * min(downstream_confidence / self.confidence_threshold, 1.0)
        self.update_routing(state, action, reward)
        self.stats['route_accepted'] += 1
    
    def learn_route_continued(self, state, action: int):
        """本地规则：下游继续传递"""
        self.update_routing(state, action, 0.2)
        self.stats['route_continued'] += 1
    
    def learn_route_deadend(self, state, action: int):
        """本地规则：死路"""
        self.update_routing(state, action, -0.8)
        self.stats['route_deadend'] += 1
    
    def learn_route_revisit(self, state, action: int):
        """本地规则：回头"""
        self.update_routing(state, action, -1.0)
        self.stats['route_revisit'] += 1

    def train_self_supervised(self, state, is_me: bool):
        """监督训练 SELF"""
        if is_me:
            self.self_lrm.update(state, 1.5)
            self.update_routing(state, 0, 1.5)
            # 惩罚路由出去
            for i in range(1, self.routing_lrm.n_actions):
                self.update_routing(state, i, -0.5)
        else:
            self.self_lrm.update(state, -0.8)
            self.update_routing(state, 0, -0.8)
            # 鼓励路由出去
            for i in range(1, self.routing_lrm.n_actions):
                self.update_routing(state, i, 0.3)

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"
    
    @property
    def memory_size(self):
        routing_size = self.routing_lrm.size if self.routing_lrm else 0
        return self.self_lrm.size + routing_size

# =============================================================================
# 6. Vision System
# =============================================================================
class StructonVisionSystem:
    """
    Structon Vision System v9.13 - Auto Sense + Connection as Action
    """
    
    def __init__(self, capacity=300, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        print("\n=== 创建 Structon v9.13 (Auto Sense) ===")
        Structon._id_counter = 0
        
        self.structons = []
        self.label_to_structon = {}
        
        for label in labels:
            s = Structon(label, self.capacity, self.n_connections)
            self.structons.append(s)
            self.label_to_structon[label] = s
            print(f"  + {s.id} label='{label}'")
            
        print("\n设置稀疏连接...")
        for s in self.structons:
            s.set_connections(self.structons)
            print(f"  {s.id} ({s.label}) → {s.get_connections_str()}")
            
    def train_epoch(self, samples, n_negatives=3):
        """监督训练 SELF"""
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            correct_s = self.label_to_structon[label]
            for _ in range(2):
                correct_s.train_self_supervised(state, is_me=True)
            
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_self_supervised(state, is_me=False)

    def predict(self, image, max_hops=20, local_learning=False) -> Tuple[str, List[str]]:
        """路由预测 + 本地学习"""
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for step in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            # 决策（自动感知邻居）
            action, next_s, confidence = current.decide(state, visited)
            
            if action == 0:  # SELF
                if local_learning:
                    current.learn_self_confident(state, confidence)
                return current.label, path
            
            # 路由
            if next_s is not None:
                if next_s.id in visited:
                    if local_learning:
                        current.learn_route_revisit(state, action)
                    
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
                    # 查看下游响应
                    downstream_self_q = next_s.get_self_q(state)
                    downstream_visited = visited | {next_s.id}
                    has_path = any(c.id not in downstream_visited for c in next_s.connections)
                    
                    if local_learning:
                        if downstream_self_q > next_s.confidence_threshold:
                            current.learn_route_accepted(state, action, downstream_self_q)
                        elif has_path:
                            current.learn_route_continued(state, action)
                        else:
                            current.learn_route_deadend(state, action)
                    
                    current = next_s
                    continue
            
            # 跳转
            all_unvisited = [s for s in self.structons if s.id not in visited]
            if all_unvisited:
                current = np.random.choice(all_unvisited)
                path.append("JUMP")
            else:
                break
        
        return current.label, path

    def train_local_epoch(self, samples):
        """本地学习训练"""
        np.random.shuffle(samples)
        for img, label in samples:
            self.predict(img, local_learning=True)

    def predict_voting(self, image) -> Tuple[str, Dict[str, float]]:
        """投票预测（基于 SELF Q 值）"""
        state = self.extractor.extract(image)
        
        votes = {}
        details = {}
        
        for s in self.structons:
            self_q = s.get_self_q(state)
            details[s.label] = self_q
            
            if self_q > 0:
                votes[s.label] = votes.get(s.label, 0) + self_q + 1
        
        if not votes:
            best = max(details, key=details.get)
            return best, details
        
        best = max(votes, key=votes.get)
        return best, details

    def print_stats(self):
        print("\n" + "=" * 70)
        print("Structon Vision v9.13 - Auto Sense + Connection as Action")
        print("=" * 70)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        
        total_mem = sum(s.memory_size for s in self.structons)
        print(f"总记忆: {total_mem}")
        
        print("\n=== 本地学习统计 ===")
        print(f"{'ID':<5} {'Lbl':<4} {'Mem':<5} "
              f"{'Self+':<7} {'Self?':<7} "
              f"{'Accpt':<7} {'Cont':<7} {'Dead':<6} {'Revis':<6}")
        print("-" * 70)
        for s in self.structons:
            st = s.stats
            print(f"{s.id:<5} {s.label:<4} {s.memory_size:<5} "
                  f"{st['self_confident']:<7} {st['self_uncertain']:<7} "
                  f"{st['route_accepted']:<7} {st['route_continued']:<7} "
                  f"{st['route_deadend']:<6} {st['route_revisit']:<6}")

# =============================================================================
# 7. 二元分类测试
# =============================================================================
def test_binary_classification(n_train=100, n_test=200):
    print("\n" + "=" * 60)
    print("二元分类测试")
    print("=" * 60)
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    extractor = StateExtractor()
    
    results = []
    
    for target_digit in range(10):
        lrm = SelfLRM(capacity=300)
        
        pos_indices = np.where(train_labels == target_digit)[0][:n_train]
        neg_indices = np.where(train_labels != target_digit)[0][:n_train]
        
        for epoch in range(15):
            for idx in pos_indices:
                state = extractor.extract(train_images[idx])
                lrm.update(state, 1.5)
            
            np.random.shuffle(neg_indices)
            for idx in neg_indices[:n_train//2]:
                state = extractor.extract(train_images[idx])
                lrm.update(state, -0.8)
        
        pos_test = np.where(test_labels == target_digit)[0][:n_test]
        neg_test = np.where(test_labels != target_digit)[0][:n_test]
        
        pos_correct = sum(1 for idx in pos_test 
                        if lrm.query(extractor.extract(test_images[idx]))[0] > 0)
        neg_correct = sum(1 for idx in neg_test 
                        if lrm.query(extractor.extract(test_images[idx]))[0] <= 0)
        
        yes_rate = pos_correct / len(pos_test) * 100
        no_rate = neg_correct / len(neg_test) * 100
        
        results.append((target_digit, yes_rate, no_rate, lrm.size))
        print(f"  数字 {target_digit}: YES={yes_rate:.1f}%, NO={no_rate:.1f}%, mem={lrm.size}")
    
    avg_yes = np.mean([r[1] for r in results])
    avg_no = np.mean([r[2] for r in results])
    print(f"\n平均: YES={avg_yes:.1f}%, NO={avg_no:.1f}%, 平衡={(avg_yes+avg_no)/2:.1f}%")
    
    return results

# =============================================================================
# 8. 主实验
# =============================================================================
def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 300,
    epochs: int = 30,
    n_connections: int = 3,
    n_negatives: int = 3,
    local_epochs: int = 10
):
    print("=" * 70)
    print("Structon Vision v9.13 - Auto Sense + Connection as Action")
    print("=" * 70)
    print("\n核心改进:")
    print("- 自动感知：每次决策前获取邻居 SELF Q 值")
    print("- Extended State：邻居共鸣编码进 state")
    print("- Connection as Action：决策仍通过 LRM 学习")
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    
    test_binary_classification()
    
    print("\n" + "=" * 70)
    print("系统测试")
    print("=" * 70)
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(capacity=capacity, n_connections=n_connections)
    system.build([str(i) for i in range(10)])
    
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:n_per_class]
        for i in idxs:
            samples.append((train_images[i], str(d)))
    
    print(f"\n训练样本: {len(samples)}")
    
    # 阶段1：监督训练
    print(f"\n=== 阶段1: 监督训练 SELF ({epochs} epochs) ===")
    t0 = time.time()
    
    for ep in range(epochs):
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (ep + 1) % 5 == 0:
            correct = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, _ = system.predict_voting(test_images[i])
                if pred == str(test_labels[i]):
                    correct += 1
            print(f"  轮次 {ep+1}: 投票准确率 {correct/check_n*100:.1f}%")
    
    print(f"\n监督训练: {time.time()-t0:.1f}秒")
    
    # 阶段2：本地学习
    print(f"\n=== 阶段2: 本地学习 ({local_epochs} epochs) ===")
    print("自动感知邻居共鸣，纯本地学习！")
    t1 = time.time()
    
    for ep in range(local_epochs):
        system.train_local_epoch(samples)
        
        if (ep + 1) % 2 == 0:
            correct = 0
            path_lens = []
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, path = system.predict(test_images[i])
                path_lens.append(len(path))
                if pred == str(test_labels[i]):
                    correct += 1
            print(f"  轮次 {ep+1}: 路由准确率={correct/check_n*100:.1f}%, 平均路径={np.mean(path_lens):.1f}")
    
    print(f"\n本地学习: {time.time()-t1:.1f}秒")
    
    system.print_stats()
    
    # 最终测试
    print(f"\n=== 最终测试 ===")
    test_idxs = np.random.choice(len(test_images), n_test, replace=False)
    
    # 路由测试
    path_lengths = []
    jump_counts = []
    correct_route = 0
    
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, path = system.predict(test_images[idx])
        path_lengths.append(len(path))
        jump_counts.append(path.count("JUMP"))
        if pred == true_label:
            correct_route += 1
    
    print(f"\n路由预测:")
    print(f"  准确率: {correct_route/n_test*100:.1f}%")
    print(f"  平均路径: {np.mean(path_lengths):.2f}")
    print(f"  平均跳转: {np.mean(jump_counts):.2f}")
    
    # 投票测试
    correct_vote = 0
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, _ = system.predict_voting(test_images[idx])
        if pred == true_label:
            correct_vote += 1
    
    print(f"\n投票预测:")
    print(f"  准确率: {correct_vote/n_test*100:.1f}%")
    
    # 示例路径
    print("\n=== 示例路径 ===")
    for i in range(5):
        idx = test_idxs[i]
        pred, path = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        path_str = ' → '.join(path[:10])
        if len(path) > 10:
            path_str += f" ... ({len(path)} hops)"
        print(f"  真实={true}, 预测={pred} {status}, 路径: {path_str}")
    
    return system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=300)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--connections', type=int, default=3)
    parser.add_argument('--negatives', type=int, default=3)
    parser.add_argument('--local-epochs', type=int, default=10)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives,
        local_epochs=args.local_epochs
    )
