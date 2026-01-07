#!/usr/bin/env python3
"""
Structon Vision v9.2 - Local Reward "Hot Potato"

核心思想：
- 放弃全局路径优化
- 纯局部奖励：
  - 如果是我的图：奖励 "Is Me" (Action 0)
  - 如果不是我的图：奖励 "Go Structon" (Action > 0)
- 路由策略：近似随机游走 (Random Walk)，依靠 "Is Me" 的高准确率来终止路径。

训练策略：
- 每个样本只训练：
  1. 正确的 Structon（学习"是我的"）
  2. 随机几个错误的 Structon（学习"传走"）
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Optional, Dict
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
# 2. 特征提取 (41维)
# =============================================================================
class StateExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        features = []
        h, w = img.shape
        
        # 5x5 Grid (25维)
        bh, bw = h // 5, w // 5
        for i in range(5):
            for j in range(5):
                features.append(np.mean(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]))
        
        # 水平投影 (5维)
        for i in range(5): 
            features.append(np.mean(img[i*(h//5):(i+1)*(h//5), :]))
        
        # 垂直投影 (5维)
        for j in range(5): 
            features.append(np.mean(img[:, j*(w//5):(j+1)*(w//5)]))
        
        # 结构特征 (6维)
        binary = (img > 0.3).astype(np.uint8)
        features.append(np.mean(binary))  # 总密度
        features.append(np.mean(binary[:h//2, :]) - np.mean(binary[h//2:, :]))  # 上下差
        features.append(np.mean(binary[:, :w//2]) - np.mean(binary[:, w//2:]))  # 左右差
        features.append(np.mean(binary[h//4:3*h//4, w//4:3*w//4]))  # 中心密度
        features.append(np.mean(binary[2:5, :]))   # 顶部
        features.append(np.mean(binary[-5:-2, :]))  # 底部
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6: 
            state = state / norm
        return state

# =============================================================================
# 3. LRM (Local Resonant Memory)
# =============================================================================
class LRM:
    def __init__(self, state_dim=41, n_actions=4, capacity=200):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        
        # 随机投影矩阵
        self.projection = np.random.randn(state_dim, 16).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.92

    def _compute_key(self, state):
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6: 
            key /= norm
        return key

    def query(self, state):
        if not self.keys: 
            return np.zeros(self.n_actions), 0.0
        
        key = self._compute_key(state)
        scores = np.array(self.keys) @ key
        
        # Softmax-like weighting
        weights = np.maximum(scores, 0) ** 3
        if np.sum(weights) < 1e-6: 
            return np.zeros(self.n_actions), 0.0
        weights /= np.sum(weights)
        
        q_values = np.zeros(self.n_actions)
        for i, w in enumerate(weights):
            if w > 0.01: 
                q_values += w * self.values[i]
        return q_values, float(np.max(scores))

    def update(self, state, action, reward):
        key = self._compute_key(state)
        
        # 1. 尝试更新现有记忆
        if self.keys:
            scores = np.array(self.keys) @ key
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                self.access_counts[best_idx] += 1
                return

        # 2. 写入新记忆
        if len(self.keys) >= self.capacity:
            # 移除访问最少的
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
            
        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward 
        self.keys.append(key)
        self.values.append(new_q)
        self.access_counts.append(1)
    
    @property
    def size(self): 
        return len(self.keys)

# =============================================================================
# 4. Structon
# =============================================================================
class Structon:
    _id_counter = 0
    
    def __init__(self, label: str, n_connections=3, capacity=200):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        # Action 0 = Is Me
        # Action 1..N = Go Connection i
        self.lrm = LRM(n_actions=1+n_connections, capacity=capacity)
        
    def set_connections(self, all_structons):
        """随机选择连接"""
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
    def predict(self, state) -> int:
        """贪婪选择动作"""
        q, _ = self.lrm.query(state)
        return int(np.argmax(q))
    
    def train_local(self, state, is_my_label: bool, epsilon: float = 0.2):
        """
        纯局部训练逻辑
        
        规则：
        - 是我的 + 选"是我的" → +1.0
        - 是我的 + 选"传走" → -1.0
        - 不是我的 + 选"传走" → +0.5
        - 不是我的 + 选"是我的" → -1.0
        """
        q_values, _ = self.lrm.query(state)
        chosen_action = int(np.argmax(q_values))
        
        # 探索：偶尔随机选一个动作
        if np.random.rand() < epsilon:
            chosen_action = np.random.randint(0, self.lrm.n_actions)

        if is_my_label:
            # 目标：Action 0 (Is Me)
            if chosen_action == 0:
                reward = 1.0       # 答对了！
            else:
                reward = -1.0      # 传走了？惩罚！
        else:
            # 目标：Action > 0 (Go Structon)
            if chosen_action == 0:
                reward = -1.0      # 冒领？重罚！
            else:
                reward = 0.5       # 传走了，好（不管去哪）
                
        self.lrm.update(state, chosen_action, reward)
    
    def get_connections_str(self) -> str:
        return f"[{', '.join(c.label for c in self.connections)}]"

# =============================================================================
# 5. Vision System
# =============================================================================
class StructonVisionSystem:
    def __init__(self, n_connections=3, capacity=200):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.n_connections = n_connections
        self.capacity = capacity
        
    def build(self, labels):
        """创建所有 Structon 并设置连接"""
        print("\n=== 创建 Structon ===")
        Structon._id_counter = 0  # 重置计数器
        
        self.structons = []
        self.label_to_structon = {}
        
        for label in labels:
            s = Structon(label, self.n_connections, self.capacity)
            self.structons.append(s)
            self.label_to_structon[label] = s
            print(f"  + {s.id} label='{label}'")
        
        print("\n设置稀疏连接...")
        for s in self.structons:
            s.set_connections(self.structons)
            print(f"  {s.id} ({s.label}) → {s.get_connections_str()}")
            
    def train_epoch(self, samples, n_negatives=3):
        """
        训练一个 epoch
        
        每个样本：
        1. 训练正确的 Structon（学习"是我的"）
        2. 随机训练几个错误的 Structon（学习"传走"）
        """
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 1. 训练正确的 Structon
            correct_s = self.label_to_structon[label]
            correct_s.train_local(state, is_my_label=True)
            
            # 2. 随机抽几个错误的来训练 "传走"
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_local(state, is_my_label=False)

    def predict(self, image, max_hops=15) -> Tuple[str, List[str]]:
        """
        预测（随机入口，随机游走）
        """
        state = self.extractor.extract(image)
        
        # 随机入口
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            action = current.predict(state)
            
            if action == 0:
                # Structon 说: "是我的!"
                return current.label, path
            
            # Action > 0: 传给邻居
            next_idx = action - 1
            if next_idx < len(current.connections):
                current = current.connections[next_idx]
            else:
                break
                
        # 跳跃耗尽，返回最后位置
        return current.label, path

    def predict_voting(self, image, max_hops=15) -> Tuple[str, float]:
        """
        从所有起点预测，投票
        """
        state = self.extractor.extract(image)
        
        votes = {}
        for entry in self.structons:
            current = entry
            visited = set()
            
            for _ in range(max_hops):
                if current.id in visited:
                    break
                visited.add(current.id)
                
                action = current.predict(state)
                
                if action == 0:
                    votes[current.label] = votes.get(current.label, 0) + 1
                    break
                
                next_idx = action - 1
                if next_idx < len(current.connections):
                    current = current.connections[next_idx]
                else:
                    votes[current.label] = votes.get(current.label, 0) + 1
                    break
            else:
                votes[current.label] = votes.get(current.label, 0) + 1
        
        if not votes:
            return "?", 0.0
        
        best = max(votes, key=votes.get)
        confidence = votes[best] / len(self.structons)
        return best, confidence

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.2 - Hot Potato")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        total_mem = sum(s.lrm.size for s in self.structons)
        print(f"总记忆: {total_mem}")
        
        print("\n=== 各 Structon ===")
        for s in self.structons:
            print(f"  {s.id} ['{s.label}'] mem:{s.lrm.size} → {s.get_connections_str()}")

# =============================================================================
# 6. 实验
# =============================================================================
def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 200,
    max_epochs: int = 30,
    n_connections: int = 3,
    n_negatives: int = 3
):
    print("=" * 70)
    print("Structon Vision v9.2 - Local Rewards (Hot Potato)")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    print(f"每样本负样本数: {n_negatives}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(n_connections=n_connections, capacity=capacity)
    system.build([str(i) for i in range(10)])
    
    # 准备样本
    samples = []
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        for idx in indices:
            samples.append((train_images[idx], str(digit)))
    
    print(f"\n训练样本: {len(samples)}")
    
    print(f"\n=== 训练 (Hot Potato) ===")
    t0 = time.time()
    
    for epoch in range(max_epochs):
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (epoch + 1) % 5 == 0:
            # 快速测试
            correct = 0
            test_subset = np.random.choice(len(test_images), 200, replace=False)
            for idx in test_subset:
                pred, _ = system.predict(test_images[idx])
                if pred == str(test_labels[idx]):
                    correct += 1
            print(f"  轮次 {epoch+1}: {correct/200*100:.1f}%")
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试 - 随机入口
    print(f"\n=== 测试（随机入口）===")
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    correct1 = 0
    for idx in test_indices:
        pred, path = system.predict(test_images[idx])
        if pred == str(test_labels[idx]):
            correct1 += 1
    print(f"准确率: {correct1/n_test*100:.1f}%")
    
    # 测试 - 全起点投票
    print(f"\n=== 测试（全起点投票）===")
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    
    for idx in test_indices:
        pred, conf = system.predict_voting(test_images[idx])
        true_label = str(test_labels[idx])
        results[true_label]['total'] += 1
        if pred == true_label:
            results[true_label]['correct'] += 1
    
    total_correct = sum(r['correct'] for r in results.values())
    print(f"准确率: {total_correct/n_test*100:.1f}%")
    
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        if r['total'] > 0:
            print(f"  {d}: {r['correct']/r['total']*100:.1f}%")
    
    # 显示几个预测路径
    print("\n=== 示例路径 ===")
    for i in range(5):
        idx = test_indices[i]
        pred, path = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        print(f"  真实={true}, 预测={pred} {status}, 路径: {' → '.join(path)}")
    
    return system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=200)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--connections', type=int, default=3)
    parser.add_argument('--negatives', type=int, default=3)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        max_epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives
    )
