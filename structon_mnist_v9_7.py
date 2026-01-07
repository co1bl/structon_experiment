#!/usr/bin/env python3
"""
Structon Vision v9.7 - 双 LRM 架构

核心设计：
  每个 Structon 有两个 LRM：
  1. Recognition LRM: 是不是我的？[NO, YES]
  2. Routing LRM: 如果不是，传给哪个邻居？[去A, 去B, 去C]

完全符合 Structon 哲学：
  - 局部规则 ✓
  - 局部学习 ✓
  - 全局涌现 ✓

训练方式：
  - 执行完整路径
  - 根据最终结果，反向给每个节点局部奖励
  - 每个节点只更新自己的 LRM
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
    """Contrast Binary + Structure 特征"""
    
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        # 计算 grid
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
        features.append((diag1 - 0.5) * 2)
        features.append((diag2 - 0.5) * 2)
        features.append((diag1 - diag2) * 2)
        
        # 四象限
        mid = self.grid_size // 2
        q1 = np.mean(grid[:mid, :mid])
        q2 = np.mean(grid[:mid, mid:])
        q3 = np.mean(grid[mid:, :mid])
        q4 = np.mean(grid[mid:, mid:])
        
        features.append((q1 - q4) * 2)
        features.append((q2 - q3) * 2)
        features.append((q1 + q4 - q2 - q3) * 2)
        features.append(((q1 + q2) - (q3 + q4)) * 2)
        
        # 边缘
        features.append((np.mean(grid[0, :]) - np.mean(grid[-1, :])) * 2)
        features.append((np.mean(grid[:, 0]) - np.mean(grid[:, -1])) * 2)
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state

# =============================================================================
# 3. LRM
# =============================================================================
class LRM:
    """Local Resonant Memory"""
    
    def __init__(self, n_actions=2, capacity=200, top_k=5):
        self.n_actions = n_actions
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88

    def query(self, state) -> Tuple[np.ndarray, float]:
        """查询 Q 值"""
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

    def update(self, state, action, reward):
        """更新记忆"""
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
# 4. Structon - 双 LRM
# =============================================================================
class Structon:
    """
    双 LRM Structon
    
    - Recognition LRM: [NO, YES]
    - Routing LRM: [去邻居0, 去邻居1, 去邻居2]
    """
    _id_counter = 0
    
    def __init__(self, label: str, capacity=200, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        # 双 LRM
        self.recognition_lrm = LRM(n_actions=2, capacity=capacity)  # [NO, YES]
        self.routing_lrm = LRM(n_actions=n_connections, capacity=capacity)  # [去A, 去B, 去C]
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others + [None] * (self.n_connections - len(others))
    
    def decide_recognition(self, state, explore=False, epsilon=0.1) -> Tuple[int, float]:
        """识别决策：是我的吗？"""
        q, score = self.recognition_lrm.query(state)
        
        if explore and np.random.random() < epsilon:
            action = np.random.randint(0, 2)
        else:
            action = int(np.argmax(q))
        
        return action, q[action] if q[action] != 0 else score
    
    def decide_routing(self, state, visited_ids: set, explore=False, epsilon=0.15) -> Tuple[int, Optional['Structon']]:
        """
        路由决策：传给哪个邻居？
        
        - 排除已访问的邻居
        - 从剩余中选择
        """
        q, _ = self.routing_lrm.query(state)
        
        # 创建可用邻居的 mask
        available = []
        for i, conn in enumerate(self.connections):
            if conn is not None and conn.id not in visited_ids:
                available.append(i)
        
        if not available:
            # 没有可用邻居，返回 None
            return -1, None
        
        if explore and np.random.random() < epsilon:
            # 探索：随机选一个可用的
            action = np.random.choice(available)
        else:
            # 利用：选 Q 值最高的可用邻居
            best_action = available[0]
            best_q = q[available[0]]
            for a in available[1:]:
                if q[a] > best_q:
                    best_q = q[a]
                    best_action = a
            action = best_action
        
        return action, self.connections[action]
    
    def learn_recognition(self, state, is_me: bool, claimed: bool):
        """
        学习识别
        
        is_me: 真实标签是否是我
        claimed: 是否声称是我的（选了 YES）
        """
        if is_me:
            # 应该说 YES
            self.recognition_lrm.update(state, 1, 1.5)   # YES 好
            self.recognition_lrm.update(state, 0, -0.8)  # NO 不好
        else:
            # 应该说 NO
            self.recognition_lrm.update(state, 0, 0.8)   # NO 好
            self.recognition_lrm.update(state, 1, -0.8)  # YES 不好
    
    def learn_routing(self, state, action: int, success: bool):
        """
        学习路由
        
        action: 选择了哪个邻居
        success: 最终是否成功找到正确答案
        """
        if action < 0 or action >= self.n_connections:
            return
            
        if success:
            self.routing_lrm.update(state, action, 1.0)  # 这条路通了！
        else:
            self.routing_lrm.update(state, action, -0.3)  # 这条路不通
    
    def get_connections_str(self):
        labels = []
        for c in self.connections:
            labels.append(c.label if c else "?")
        return f"[{', '.join(labels)}]"

# =============================================================================
# 5. Vision System
# =============================================================================
class StructonVisionSystem:
    """双 LRM 视觉系统"""
    
    def __init__(self, capacity=200, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        print("\n=== 创建 Structon v9.7 (双 LRM) ===")
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
    
    def execute_path(self, state: np.ndarray, explore=False, max_hops=15) -> Tuple[str, List[Tuple]]:
        """
        执行完整路径
        
        返回：
          - 最终预测标签
          - 路径记录 [(structon, recognition_action, routing_action, next_structon), ...]
        """
        current = np.random.choice(self.structons)
        visited = set()
        path = []
        
        for _ in range(max_hops):
            visited.add(current.id)
            
            # 1. 识别决策
            rec_action, rec_conf = current.decide_recognition(state, explore)
            
            if rec_action == 1:  # YES - 声称是我的
                path.append((current, rec_action, -1, None))
                return current.label, path
            
            # 2. 路由决策
            route_action, next_structon = current.decide_routing(state, visited, explore)
            path.append((current, rec_action, route_action, next_structon))
            
            if next_structon is None:
                # 死胡同，随机跳跃
                unvisited = [s for s in self.structons if s.id not in visited]
                if unvisited:
                    next_structon = np.random.choice(unvisited)
                else:
                    # 全部访问完了
                    return current.label, path
            
            current = next_structon
        
        # 达到最大跳数
        return current.label, path
    
    def train_single(self, image: np.ndarray, true_label: str):
        """
        训练单个样本
        
        1. 执行路径（带探索）
        2. 根据结果给每个节点反馈
        """
        state = self.extractor.extract(image)
        
        # 执行路径
        predicted, path = self.execute_path(state, explore=True)
        success = (predicted == true_label)
        
        # 给路径上每个节点反馈
        for structon, rec_action, route_action, next_s in path:
            is_me = (structon.label == true_label)
            
            # 识别学习
            structon.learn_recognition(state, is_me, rec_action == 1)
            
            # 路由学习（只有当选了 NO 时才学习路由）
            if rec_action == 0 and route_action >= 0:
                structon.learn_routing(state, route_action, success)
        
        # 额外：直接训练正确的 Structon
        correct_structon = self.label_to_structon.get(true_label)
        if correct_structon:
            correct_structon.learn_recognition(state, True, False)
        
        return success
    
    def train_epoch(self, samples):
        """训练一个 epoch"""
        np.random.shuffle(samples)
        correct = 0
        
        for img, label in samples:
            if self.train_single(img, label):
                correct += 1
        
        return correct / len(samples)
    
    def predict(self, image, max_hops=15) -> Tuple[str, List[str]]:
        """预测（不探索）"""
        state = self.extractor.extract(image)
        predicted, path = self.execute_path(state, explore=False, max_hops=max_hops)
        
        # 提取路径标签
        path_labels = []
        for structon, rec_action, route_action, next_s in path:
            path_labels.append(structon.label)
        
        return predicted, path_labels
    
    def predict_voting(self, image) -> Tuple[str, Dict[str, float]]:
        """投票预测"""
        state = self.extractor.extract(image)
        
        votes = {}
        details = {}
        
        for s in self.structons:
            q, _ = s.recognition_lrm.query(state)
            details[s.label] = q[1]  # YES 的 Q 值
            
            if np.argmax(q) == 1:  # YES
                votes[s.label] = votes.get(s.label, 0) + q[1] + 1
        
        if not votes:
            best = max(details, key=details.get)
            return best, details
        
        best = max(votes, key=votes.get)
        return best, details
    
    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.7 - 双 LRM")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        
        total_rec = sum(s.recognition_lrm.size for s in self.structons)
        total_route = sum(s.routing_lrm.size for s in self.structons)
        print(f"识别记忆: {total_rec}")
        print(f"路由记忆: {total_route}")
        
        print("\n=== 各 Structon ===")
        for s in self.structons:
            print(f"  {s.id} ['{s.label}'] rec:{s.recognition_lrm.size} route:{s.routing_lrm.size} → {s.get_connections_str()}")

# =============================================================================
# 6. 实验
# =============================================================================
def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 200,
    epochs: int = 30,
    n_connections: int = 3
):
    print("=" * 70)
    print("Structon Vision v9.7 - 双 LRM (识别 + 路由)")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    print("\n核心设计:")
    print("  - Recognition LRM: 是不是我的？ [NO, YES]")
    print("  - Routing LRM: 传给哪个邻居？ [去A, 去B, 去C]")
    print("  - 完全局部学习！")
    
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
    
    print(f"\n=== 训练 ===")
    t0 = time.time()
    
    for ep in range(epochs):
        train_acc = system.train_epoch(samples)
        
        if (ep + 1) % 5 == 0:
            # 测试
            correct = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, _ = system.predict_voting(test_images[i])
                if pred == str(test_labels[i]):
                    correct += 1
            print(f"  轮次 {ep+1}: 训练={train_acc*100:.1f}%, 测试={correct/check_n*100:.1f}%")
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试 - 路由
    print(f"\n=== 测试（学习的路由）===")
    test_idxs = np.random.choice(len(test_images), n_test, replace=False)
    
    stats = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
    path_lengths = []
    
    correct1 = 0
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, path = system.predict(test_images[idx])
        path_lengths.append(len(path))
        stats[true_label]['total'] += 1
        if pred == true_label:
            correct1 += 1
            stats[true_label]['correct'] += 1
    
    print(f"准确率: {correct1/n_test*100:.1f}%")
    print(f"平均路径长度: {np.mean(path_lengths):.1f}")
    
    # 测试 - 投票
    print(f"\n=== 测试（投票）===")
    correct2 = 0
    stats2 = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
    
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, _ = system.predict_voting(test_images[idx])
        stats2[true_label]['total'] += 1
        if pred == true_label:
            correct2 += 1
            stats2[true_label]['correct'] += 1
    
    print(f"准确率: {correct2/n_test*100:.1f}%")
    
    print("\n各数字准确率:")
    print(f"{'数字':<6} {'路由':<12} {'投票':<12}")
    print("-" * 30)
    for d in range(10):
        s1 = stats[str(d)]
        s2 = stats2[str(d)]
        acc1 = s1['correct']/s1['total']*100 if s1['total'] > 0 else 0
        acc2 = s2['correct']/s2['total']*100 if s2['total'] > 0 else 0
        print(f"  {d}     {acc1:>6.1f}%      {acc2:>6.1f}%")
    
    # 示例路径
    print("\n=== 示例路径 ===")
    for i in range(8):
        idx = test_idxs[i]
        pred, path = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        path_str = ' → '.join(path[:10])
        if len(path) > 10:
            path_str += f" ..."
        print(f"  真实={true}, 预测={pred} {status}, 路径({len(path)}): {path_str}")
    
    return system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=200)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--connections', type=int, default=3)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        epochs=args.max_epochs,
        n_connections=args.connections
    )
