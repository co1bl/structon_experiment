#!/usr/bin/env python3
"""
Structon Vision v9.14 - Connection as Action

基于 v9.6，核心改进：
- 路由不再是随机的，而是可学习的动作
- 统一 LRM：动作 0 = SELF，动作 1~N = 路由到邻居
- 保持简洁，不加额外复杂性

改变：
v9.6: LRM 只学 YES/NO，路由随机选择
v9.14: LRM 学习 SELF + 所有路由动作

不能回头仍然是硬规则（简单高效）
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
# 2. 特征提取 - Contrast Binary + Structure（与 v9.6 相同）
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
# 3. Unified LRM - Connection as Action
# =============================================================================
class UnifiedLRM:
    """
    统一 LRM：SELF + 路由动作
    
    动作空间：
    - 动作 0: SELF（认领输入）
    - 动作 1~N: 路由到连接 0~(N-1)
    """
    
    def __init__(self, n_connections: int, capacity=300, top_k=5):
        self.n_actions = 1 + n_connections  # SELF + connections
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88

    def query(self, state) -> Tuple[np.ndarray, float]:
        """查询所有动作的 Q 值"""
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
        """更新 Q 值"""
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
# 4. Structon - Connection as Action
# =============================================================================
class Structon:
    """
    Structon with Connection as Action
    
    决策统一为：
    - SELF (动作0): 认领输入
    - CONNECTION_i (动作i+1): 路由到第i个连接
    """
    _id_counter = 0
    
    def __init__(self, label: str, capacity=300, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        # 延迟初始化（等连接设置好）
        self.lrm: Optional[UnifiedLRM] = None
        self.capacity = capacity
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
        # 初始化统一 LRM
        self.lrm = UnifiedLRM(
            n_connections=len(self.connections),
            capacity=self.capacity
        )

    def decide(self, state, visited_ids: set) -> Tuple[int, Optional['Structon'], float]:
        """
        决策
        返回: (action, next_structon, q_value)
        """
        if self.lrm is None:
            return 0, None, 0.0
        
        q_values, confidence = self.lrm.query(state)
        
        # 屏蔽已访问的连接（硬规则）
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

    def train_supervised(self, state, is_me: bool):
        """
        监督训练
        is_me=True: 应该说 SELF
        is_me=False: 不应该说 SELF
        """
        if self.lrm is None:
            return
            
        if is_me:
            # 正样本：奖励 SELF，惩罚路由
            self.lrm.update(state, 0, 1.5)
            for i in range(1, self.lrm.n_actions):
                self.lrm.update(state, i, -0.5)
        else:
            # 负样本：惩罚 SELF，鼓励路由
            self.lrm.update(state, 0, -0.8)
            for i in range(1, self.lrm.n_actions):
                self.lrm.update(state, i, 0.3)

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"
    
    @property
    def memory_size(self):
        return self.lrm.size if self.lrm else 0

# =============================================================================
# 5. Vision System
# =============================================================================
class StructonVisionSystem:
    """Structon Vision System v9.14 - Connection as Action"""
    
    def __init__(self, capacity=300, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        print("\n=== 创建 Structon v9.14 (Connection as Action) ===")
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
        """监督训练"""
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 正样本
            correct_s = self.label_to_structon[label]
            for _ in range(2):
                correct_s.train_supervised(state, is_me=True)
            
            # 负样本
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_supervised(state, is_me=False)

    def predict(self, image, max_hops=20) -> Tuple[str, List[str]]:
        """路由预测"""
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            action, next_s, confidence = current.decide(state, visited)
            
            if action == 0:  # SELF
                return current.label, path
            
            if next_s is not None:
                current = next_s
            else:
                # 没有可用连接，跳转
                all_unvisited = [s for s in self.structons if s.id not in visited]
                if all_unvisited:
                    current = np.random.choice(all_unvisited)
                    path.append("JUMP")
                else:
                    break
        
        return current.label, path

    def predict_voting(self, image) -> Tuple[str, Dict[str, float]]:
        """投票预测（基于 SELF Q 值）"""
        state = self.extractor.extract(image)
        
        votes = {}
        details = {}
        
        for s in self.structons:
            if s.lrm is None:
                continue
            q_values, _ = s.lrm.query(state)
            self_q = q_values[0]  # SELF 的 Q 值
            details[s.label] = self_q
            
            if self_q > 0:
                votes[s.label] = votes.get(s.label, 0) + self_q + 1
        
        if not votes:
            best = max(details, key=details.get)
            return best, details
        
        best = max(votes, key=votes.get)
        return best, details

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.14 - Connection as Action")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        total = sum(s.memory_size for s in self.structons)
        print(f"总记忆: {total}")
        
        print("\n=== 各 Structon ===")
        for s in self.structons:
            print(f"  {s.id} ['{s.label}'] mem:{s.memory_size}")

# =============================================================================
# 6. 二元分类测试
# =============================================================================
def test_binary_classification(n_train=100, n_test=200):
    print("\n" + "=" * 60)
    print("二元分类测试")
    print("=" * 60)
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    extractor = StateExtractor()
    
    results = []
    
    for target_digit in range(10):
        # 用 UnifiedLRM，但只看 SELF 动作
        lrm = UnifiedLRM(n_connections=0, capacity=300)
        
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
        
        pos_test = np.where(test_labels == target_digit)[0][:n_test]
        neg_test = np.where(test_labels != target_digit)[0][:n_test]
        
        pos_correct = sum(1 for idx in pos_test 
                        if lrm.query(extractor.extract(test_images[idx]))[0][0] > 0)
        neg_correct = sum(1 for idx in neg_test 
                        if lrm.query(extractor.extract(test_images[idx]))[0][0] <= 0)
        
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
    n_negatives: int = 3
):
    print("=" * 70)
    print("Structon Vision v9.14 - Connection as Action")
    print("=" * 70)
    print("\n核心改进: 路由从随机变成可学习的动作")
    print("- 动作 0: SELF（认领）")
    print("- 动作 1~N: 路由到邻居")
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
    
    print(f"\n=== 训练 ({epochs} epochs) ===")
    t0 = time.time()
    
    for ep in range(epochs):
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (ep + 1) % 5 == 0:
            # 测试投票
            correct_vote = 0
            # 测试路由
            correct_route = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred_vote, _ = system.predict_voting(test_images[i])
                pred_route, _ = system.predict(test_images[i])
                if pred_vote == str(test_labels[i]):
                    correct_vote += 1
                if pred_route == str(test_labels[i]):
                    correct_route += 1
            print(f"  轮次 {ep+1}: 投票={correct_vote/check_n*100:.1f}%, 路由={correct_route/check_n*100:.1f}%")
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
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
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives
    )
