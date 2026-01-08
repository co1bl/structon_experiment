#!/usr/bin/env python3
"""
Structon Vision v9.7 - Unified Routing: Connection as Action

核心思想：YES 也是路由 —— 路由到自己！

统一路由模型：
- 动作 0: 路由到自己 (YES，我认领这个输入)
- 动作 1: 路由到连接[0]
- 动作 2: 路由到连接[1]
- 动作 3: 路由到连接[2]
- ...

惩罚机制（本地规则）：
- 如果路由到已访问的节点 → 惩罚
- 如果最终预测正确 → 奖励整条路径

这样 connection 和 action 在同一个 LRM 中统一学习！
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
# 2. 特征提取 - Contrast Binary + Structure
# =============================================================================
class StateExtractor:
    """
    最佳特征组合：Contrast Binary + Structure
    """
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        # 1. 计算 7x7 grid 值
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                grid[i, j] = np.mean(block)
        
        # 2. Contrast Binary: 暗=-1, 亮=+1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] < self.threshold:
                    features.append(-1.0)
                else:
                    features.append(1.0)
        
        # 3. 水平差异（相邻列）
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                diff = grid[i, j] - grid[i, j+1]
                features.append(diff * 2)
        
        # 4. 垂直差异（相邻行）
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                diff = grid[i, j] - grid[i+1, j]
                features.append(diff * 2)
        
        # 5. 对角线
        diag1 = np.mean([grid[i, i] for i in range(self.grid_size)])
        diag2 = np.mean([grid[i, self.grid_size-1-i] for i in range(self.grid_size)])
        features.append((diag1 - 0.5) * 2)
        features.append((diag2 - 0.5) * 2)
        features.append((diag1 - diag2) * 2)
        
        # 6. 四象限
        mid = self.grid_size // 2
        q1 = np.mean(grid[:mid, :mid])
        q2 = np.mean(grid[:mid, mid:])
        q3 = np.mean(grid[mid:, :mid])
        q4 = np.mean(grid[mid:, mid:])
        
        features.append((q1 - q4) * 2)
        features.append((q2 - q3) * 2)
        features.append((q1 + q4 - q2 - q3) * 2)
        features.append(((q1 + q2) - (q3 + q4)) * 2)
        
        # 7. 边缘
        top = np.mean(grid[0, :])
        bottom = np.mean(grid[-1, :])
        left = np.mean(grid[:, 0])
        right = np.mean(grid[:, -1])
        
        features.append((top - bottom) * 2)
        features.append((left - right) * 2)
        features.append((top - 0.5) * 2)
        features.append((bottom - 0.5) * 2)
        
        state = np.array(features, dtype=np.float32)
        
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        
        return state

# =============================================================================
# 3. Unified Routing LRM
# =============================================================================
class UnifiedRoutingLRM:
    """
    统一路由记忆
    
    动作空间：
    - 动作 0: SELF (路由到自己，即 YES)
    - 动作 1~N: 路由到连接 0~(N-1)
    
    这样 YES/NO 和连接选择在同一个 LRM 中学习！
    """
    
    def __init__(self, n_connections: int, capacity=300, top_k=5):
        # 动作数 = 1 (SELF) + n_connections
        self.n_actions = 1 + n_connections
        self.n_connections = n_connections
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88
        
        # 动作索引常量
        self.SELF = 0  # 路由到自己 = YES

    def query(self, state) -> Tuple[np.ndarray, float]:
        """查询所有路由动作的 Q 值"""
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.array([np.dot(k, state) for k in self.keys])
        
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

    def update(self, state, action: int, reward: float):
        """更新路由记忆"""
        if action >= self.n_actions:
            return
            
        if self.keys:
            scores = np.array([np.dot(k, state) for k in self.keys])
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
            
        self.keys.append(state.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        
    @property
    def size(self):
        return len(self.keys)

# =============================================================================
# 4. Structon - 统一路由
# =============================================================================
class Structon:
    """
    Structon with Unified Routing
    
    所有决策统一为路由：
    - SELF (动作0): 认领输入，路由到自己
    - 连接i (动作i+1): 路由到第i个连接
    """
    _id_counter = 0
    
    def __init__(self, label: str, capacity=300, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        # 统一路由记忆（延迟初始化）
        self.routing_lrm: Optional[UnifiedRoutingLRM] = None
        
        # 统计
        self.stats = {
            'self_rewards': 0,
            'self_punishments': 0,
            'route_rewards': 0,
            'route_punishments': 0
        }
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
        # 初始化统一路由记忆
        self.routing_lrm = UnifiedRoutingLRM(
            n_connections=len(self.connections),
            capacity=300,
            top_k=5
        )

    def decide(self, state, visited_ids: set) -> Tuple[int, Optional['Structon'], float]:
        """
        统一决策
        
        返回：
        - action: 动作索引
        - next_structon: 下一个Structon（如果是SELF则为None）
        - q_value: 选择的Q值
        """
        if self.routing_lrm is None:
            return 0, None, 0.0
        
        q_values, confidence = self.routing_lrm.query(state)
        
        # 调整Q值：惩罚已访问的连接
        adjusted_q = q_values.copy()
        for i, conn in enumerate(self.connections):
            action_idx = i + 1  # 连接动作从1开始
            if conn.id in visited_ids:
                adjusted_q[action_idx] = -10.0  # 强惩罚已访问
        
        # 选择最佳动作
        best_action = int(np.argmax(adjusted_q))
        best_q = adjusted_q[best_action]
        
        if best_action == 0:
            # SELF: 路由到自己
            return 0, None, best_q
        else:
            # 路由到连接
            conn_idx = best_action - 1
            return best_action, self.connections[conn_idx], best_q

    def update_routing(self, state, action: int, reward: float):
        """更新路由记忆"""
        if self.routing_lrm is not None:
            self.routing_lrm.update(state, action, reward)
            
            # 统计
            if action == 0:  # SELF
                if reward > 0:
                    self.stats['self_rewards'] += 1
                else:
                    self.stats['self_punishments'] += 1
            else:  # 路由到连接
                if reward > 0:
                    self.stats['route_rewards'] += 1
                else:
                    self.stats['route_punishments'] += 1

    def train_self(self, state, is_me: bool):
        """
        训练 SELF 动作（兼容旧接口）
        
        is_me=True: 应该路由到自己 (SELF奖励，其他惩罚)
        is_me=False: 不应该路由到自己 (SELF惩罚)
        """
        if self.routing_lrm is None:
            return
            
        if is_me:
            # 正样本：奖励SELF
            self.routing_lrm.update(state, 0, 1.5)
            # 惩罚其他路由
            for i in range(1, self.routing_lrm.n_actions):
                self.routing_lrm.update(state, i, -0.5)
        else:
            # 负样本：惩罚SELF
            self.routing_lrm.update(state, 0, -0.8)
            # 轻微鼓励路由出去
            for i in range(1, self.routing_lrm.n_actions):
                self.routing_lrm.update(state, i, 0.3)

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"
    
    def get_connection_index(self, target: 'Structon') -> int:
        """获取连接的动作索引（从1开始）"""
        for i, conn in enumerate(self.connections):
            if conn.id == target.id:
                return i + 1  # 动作索引 = 连接索引 + 1
        return -1

    @property
    def memory_size(self):
        return self.routing_lrm.size if self.routing_lrm else 0

# =============================================================================
# 5. Vision System
# =============================================================================
class StructonVisionSystem:
    """
    Structon Vision System v9.7 - Unified Routing
    """
    
    def __init__(self, capacity=300, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
        # 路由奖惩参数
        self.reward_correct_self = 1.5      # 正确认领
        self.punishment_wrong_self = -1.0   # 错误认领
        self.reward_good_route = 0.8        # 好的路由（导向正确结果）
        self.punishment_bad_route = -0.5    # 坏的路由
        self.punishment_revisit = -1.5      # 回头惩罚
        
    def build(self, labels):
        print("\n=== 创建 Structon v9.7 (Unified Routing) ===")
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
        """训练一个 epoch（分类训练）"""
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 正样本：训练正确的Structon认领
            correct_s = self.label_to_structon[label]
            for _ in range(2):
                correct_s.train_self(state, is_me=True)
            
            # 负样本：训练其他Structon不认领
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_self(state, is_me=False)

    def predict(self, image, max_hops=20, train_routing=False, true_label=None) -> Tuple[str, List[str]]:
        """
        路由预测
        
        参数：
            train_routing: 是否训练路由
            true_label: 真实标签（用于训练）
        """
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        # 记录路由历史 [(structon, action, state), ...]
        routing_history = []
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            # 统一决策
            action, next_s, q_value = current.decide(state, visited)
            
            if action == 0:  # SELF - 认领
                # 记录这个决策
                routing_history.append((current, 0, state.copy()))
                
                # 训练路由
                if train_routing and true_label is not None:
                    self._update_routing_history(
                        routing_history, 
                        success=(current.label == true_label)
                    )
                
                return current.label, path
            
            # 路由到连接
            if next_s is not None:
                # 检查是否回头（本地惩罚）
                if next_s.id in visited:
                    # 立即惩罚
                    conn_action = current.get_connection_index(next_s)
                    if conn_action > 0:
                        current.update_routing(state, conn_action, self.punishment_revisit)
                    
                    # 选择其他未访问的
                    for i, conn in enumerate(current.connections):
                        if conn.id not in visited:
                            next_s = conn
                            action = i + 1
                            break
                    else:
                        next_s = None
                
                if next_s is not None:
                    # 记录路由决策
                    routing_history.append((current, action, state.copy()))
                    current = next_s
                    continue
            
            # 没有可用连接，跳转
            all_unvisited = [s for s in self.structons if s.id not in visited]
            if all_unvisited:
                current = np.random.choice(all_unvisited)
                path.append("JUMP")
            else:
                break
        
        # 超时或无法继续
        if train_routing and routing_history:
            self._update_routing_history(routing_history, success=False)
        
        return current.label, path
    
    def _update_routing_history(self, history, success: bool):
        """更新路由历史"""
        if not history:
            return
        
        n = len(history)
        
        for i, (structon, action, state) in enumerate(history):
            # 距离终点的步数
            steps_to_end = n - 1 - i
            decay = 0.85 ** steps_to_end
            
            if success:
                if action == 0:  # 最后的SELF决策
                    reward = self.reward_correct_self
                else:  # 中间的路由决策
                    reward = self.reward_good_route * decay
            else:
                if action == 0:  # 错误的SELF
                    reward = self.punishment_wrong_self
                else:  # 导致失败的路由
                    reward = self.punishment_bad_route * decay
            
            structon.update_routing(state, action, reward)

    def train_routing_epoch(self, samples):
        """专门训练路由的 epoch"""
        np.random.shuffle(samples)
        
        for img, label in samples:
            self.predict(img, train_routing=True, true_label=label)

    def predict_voting(self, image) -> Tuple[str, Dict[str, float]]:
        """投票预测（基于SELF的Q值）"""
        state = self.extractor.extract(image)
        
        votes = {}
        details = {}
        
        for s in self.structons:
            if s.routing_lrm is None:
                continue
                
            q_values, _ = s.routing_lrm.query(state)
            self_q = q_values[0]  # SELF 的 Q 值
            
            details[s.label] = self_q
            
            if self_q > 0:  # 正Q值 = 想要认领
                votes[s.label] = votes.get(s.label, 0) + self_q + 1
        
        if not votes:
            # 没人想认领，选 SELF Q 值最高的
            best = max(details, key=details.get)
            return best, details
        
        best = max(votes, key=votes.get)
        return best, details

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.7 - Unified Routing")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        
        total_mem = sum(s.memory_size for s in self.structons)
        print(f"总路由记忆: {total_mem}")
        
        print("\n=== 各 Structon 统计 ===")
        print(f"{'ID':<6} {'Label':<6} {'Mem':<6} {'Self+':<8} {'Self-':<8} {'Route+':<8} {'Route-':<8}")
        print("-" * 60)
        for s in self.structons:
            print(f"{s.id:<6} {s.label:<6} {s.memory_size:<6} "
                  f"{s.stats['self_rewards']:<8} {s.stats['self_punishments']:<8} "
                  f"{s.stats['route_rewards']:<8} {s.stats['route_punishments']:<8}")

# =============================================================================
# 6. 二元分类测试
# =============================================================================
def test_binary_classification(n_train=100, n_test=200):
    """测试每个 Structon 的二元分类能力"""
    print("\n" + "=" * 60)
    print("二元分类测试 (Unified Routing)")
    print("=" * 60)
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    extractor = StateExtractor()
    
    results = []
    
    for target_digit in range(10):
        # 创建单个Structon（只有SELF动作）
        lrm = UnifiedRoutingLRM(n_connections=0, capacity=300)
        
        # 训练
        pos_indices = np.where(train_labels == target_digit)[0][:n_train]
        neg_indices = np.where(train_labels != target_digit)[0][:n_train]
        
        for epoch in range(15):
            for idx in pos_indices:
                state = extractor.extract(train_images[idx])
                lrm.update(state, 0, 1.5)  # SELF = YES
            
            np.random.shuffle(neg_indices)
            for idx in neg_indices[:n_train//2]:
                state = extractor.extract(train_images[idx])
                lrm.update(state, 0, -0.8)  # SELF = NO
        
        # 测试
        pos_test = np.where(test_labels == target_digit)[0][:n_test]
        neg_test = np.where(test_labels != target_digit)[0][:n_test]
        
        pos_correct = 0
        for idx in pos_test:
            state = extractor.extract(test_images[idx])
            q, _ = lrm.query(state)
            if q[0] > 0:  # SELF Q > 0 means YES
                pos_correct += 1
        
        neg_correct = 0
        for idx in neg_test:
            state = extractor.extract(test_images[idx])
            q, _ = lrm.query(state)
            if q[0] <= 0:  # SELF Q <= 0 means NO
                neg_correct += 1
        
        yes_rate = pos_correct / len(pos_test) * 100
        no_rate = neg_correct / len(neg_test) * 100
        
        results.append((target_digit, yes_rate, no_rate, lrm.size))
        print(f"  数字 {target_digit}: YES={yes_rate:.1f}%, NO={no_rate:.1f}%, mem={lrm.size}")
    
    avg_yes = np.mean([r[1] for r in results])
    avg_no = np.mean([r[2] for r in results])
    print(f"\n平均: YES={avg_yes:.1f}%, NO={avg_no:.1f}%, 平衡={(avg_yes+avg_no)/2:.1f}%")
    
    return results

# =============================================================================
# 7. 主实验
# =============================================================================
def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 300,
    epochs: int = 30,
    n_connections: int = 3,
    n_negatives: int = 3,
    routing_epochs: int = 10
):
    print("=" * 70)
    print("Structon Vision v9.7 - Unified Routing: Connection as Action")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    print(f"分类训练: {epochs} epochs, 路由训练: {routing_epochs} epochs")
    
    # 二元分类测试
    test_binary_classification()
    
    print("\n" + "=" * 70)
    print("系统测试")
    print("=" * 70)
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(capacity=capacity, n_connections=n_connections)
    system.build([str(i) for i in range(10)])
    
    # 准备样本
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:n_per_class]
        for i in idxs:
            samples.append((train_images[i], str(d)))
    
    print(f"\n训练样本: {len(samples)}")
    
    # 阶段1：分类训练
    print(f"\n=== 阶段1: 分类训练 ({epochs} epochs) ===")
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
    
    print(f"\n分类训练: {time.time()-t0:.1f}秒")
    
    # 阶段2：路由训练
    print(f"\n=== 阶段2: 路由训练 ({routing_epochs} epochs) ===")
    t1 = time.time()
    
    for ep in range(routing_epochs):
        system.train_routing_epoch(samples)
        
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
            avg_len = np.mean(path_lens)
            print(f"  轮次 {ep+1}: 路由准确率={correct/check_n*100:.1f}%, 平均路径={avg_len:.1f}")
    
    print(f"\n路由训练: {time.time()-t1:.1f}秒")
    
    system.print_stats()
    
    # 最终测试
    print(f"\n=== 最终测试 ===")
    test_idxs = np.random.choice(len(test_images), n_test, replace=False)
    
    # 路由测试
    stats_route = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
    path_lengths = []
    jump_counts = []
    
    correct_route = 0
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, path = system.predict(test_images[idx])
        stats_route[true_label]['total'] += 1
        path_lengths.append(len(path))
        jump_counts.append(path.count("JUMP"))
        if pred == true_label:
            correct_route += 1
            stats_route[true_label]['correct'] += 1
    
    print(f"\n路由预测:")
    print(f"  准确率: {correct_route/n_test*100:.1f}%")
    print(f"  平均路径: {np.mean(path_lengths):.2f}")
    print(f"  平均跳转: {np.mean(jump_counts):.2f}")
    
    # 投票测试
    stats_vote = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
    correct_vote = 0
    
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, _ = system.predict_voting(test_images[idx])
        stats_vote[true_label]['total'] += 1
        if pred == true_label:
            correct_vote += 1
            stats_vote[true_label]['correct'] += 1
    
    print(f"\n投票预测:")
    print(f"  准确率: {correct_vote/n_test*100:.1f}%")
    
    # 各数字比较
    print("\n各数字准确率:")
    print(f"{'数字':<6} {'路由':<12} {'投票':<12}")
    print("-" * 30)
    for d in range(10):
        s1 = stats_route[str(d)]
        s2 = stats_vote[str(d)]
        acc1 = s1['correct']/s1['total']*100 if s1['total'] > 0 else 0
        acc2 = s2['correct']/s2['total']*100 if s2['total'] > 0 else 0
        print(f"  {d}     {acc1:>6.1f}%      {acc2:>6.1f}%")
    
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
    parser.add_argument('--routing-epochs', type=int, default=10)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives,
        routing_epochs=args.routing_epochs
    )
