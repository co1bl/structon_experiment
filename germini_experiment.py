#!/usr/bin/env python3
"""
Structon Vision v9.5 - Final Architecture

关键特性 (Key Features):
1. Zero-Mean Features: 输入去均值化 [-0.5, 0.5]，解决 LRM 坍缩问题。
2. Top-K LRM: 仅基于最相似的 K 个记忆做决策，提高信噪比。
3. Binary Brain: 每个 Structon 只做二元判断 (YES/NO)。
4. Taboo Routing: 禁忌搜索路由，防止死循环。
5. Broadcast Training: 正负样本混合广播训练，防止灾难性遗忘。
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Set
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
# 2. 特征提取 (Zero-Mean 55维)
# =============================================================================
class StateExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        # [核心修复] Zero-Mean: 范围从 [0, 1] 变为 [-0.5, 0.5]
        img = image.astype(np.float32) / 255.0 - 0.5
        
        features = []
        h, w = img.shape
        
        # 1. 5x5 Grid (25维)
        bh, bw = h // 5, w // 5
        for i in range(5):
            for j in range(5):
                features.append(np.mean(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]))
        
        # 2. 投影 (10维)
        for i in range(5): features.append(np.mean(img[i*(h//5):(i+1)*(h//5), :]))
        for j in range(5): features.append(np.mean(img[:, j*(w//5):(j+1)*(w//5)]))
        
        # 3. 结构特征 (基于阈值后的二值图，同样去均值)
        # 先恢复到 0..1 做二值化，再减去 0.5
        img_norm = img + 0.5
        binary = (img_norm > 0.3).astype(np.float32) - 0.5 
        
        features.append(np.mean(binary))
        features.append(np.mean(binary[:h//2, :w//2]))
        features.append(np.mean(binary[:h//2, w//2:]))
        features.append(np.mean(binary[h//2:, :w//2]))
        features.append(np.mean(binary[h//2:, w//2:]))
        features.append(np.mean(binary[:h//2, :]) - np.mean(binary[h//2:, :]))
        features.append(np.mean(binary[:, :w//2]) - np.mean(binary[:, w//2:]))
        features.append(np.mean(binary[h//4:3*h//4, w//4:3*w//4]))
        features.append(np.mean(binary[h//3:2*h//3, w//3:2*w//3]))
        features.append(np.mean(binary[2:5, :]))
        features.append(np.mean(binary[h//2-2:h//2+2, :]))
        features.append(np.mean(binary[-5:-2, :]))
        features.append(np.mean(binary[:, w//2-2:w//2+2]))
        features.append(np.mean(binary[:, 2:5]))
        features.append(np.mean(binary[:, -5:-2]))
        
        # 对角线
        diag1 = np.mean([binary[i, i] for i in range(min(h, w))])
        diag2 = np.mean([binary[i, w-1-i] for i in range(min(h, w))])
        features.append(diag1)
        features.append(diag2)
        features.append(diag1 - diag2)
        features.append(np.mean(binary[0, :]))
        features.append(np.mean(binary[-1, :]))
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6: state = state / norm
        return state

# =============================================================================
# 3. LRM (Top-K & Zero-Mean Optimized)
# =============================================================================
class LRM:
    def __init__(self, state_dim=55, capacity=200):
        self.state_dim = state_dim
        self.n_actions = 2 # 0=NO, 1=YES
        self.capacity = capacity
        
        # [优化] 更高维度的投影，分散特征
        self.key_dim = 64
        self.projection = np.random.randn(state_dim, self.key_dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.5
        # [优化] Zero-Mean 下，不相关的向量相似度接近 0，正相关接近 1。
        # 0.6 是一个比较安全的“正相关”门槛。
        self.similarity_threshold = 0.6 

    def _compute_key(self, state):
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6: key /= norm
        return key

    def query(self, state, k=5) -> Tuple[np.ndarray, float]:
        """Top-K Query"""
        if not self.keys: return np.zeros(self.n_actions), 0.0
        
        key = self._compute_key(state)
        scores = np.dot(self.keys, key)
        
        # 选出 Top K
        if len(scores) <= k:
            indices = np.arange(len(scores))
        else:
            indices = np.argpartition(scores, -k)[-k:]
        
        best_score = float(np.max(scores))
        
        # 加权投票 (Softmax-like)
        q_values = np.zeros(self.n_actions)
        w_sum = 0.0
        
        for i in indices:
            score = scores[i]
            # [关键] 只有正相关的记忆才有资格投票
            if score > 0.2: 
                w = score ** 3  # 锐化权重
                q_values += w * self.values[i]
                w_sum += w
        
        if w_sum > 1e-6:
            q_values /= w_sum
            
        return q_values, best_score

    def update(self, state, action, reward):
        key = self._compute_key(state)
        
        if self.keys:
            scores = np.dot(self.keys, key)
            best_idx = int(np.argmax(scores))
            
            # 如果非常相似，则更新旧记忆 (Hebbian-like update)
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx]
                target_q = np.zeros(self.n_actions)
                target_q[action] = reward
                
                # 如果这个记忆之前是错的 (比如存了 NO 但现在是 YES)，我们需要大幅修正
                # 这里简单使用 learning rate
                self.values[best_idx] = old_q + self.learning_rate * (target_q - old_q)
                self.access_counts[best_idx] += 1
                return

        # 否则创建新记忆
        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
            
        self.keys.append(key)
        self.values.append(new_q)
        self.access_counts.append(1)
        
    @property
    def size(self): return len(self.keys)

# =============================================================================
# 4. Structon (Binary Agent)
# =============================================================================
class Structon:
    _id_counter = 0
    def __init__(self, label: str, capacity=200, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections = []
        self.lrm = LRM(capacity=capacity)
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
    def decide(self, state) -> Tuple[int, float]:
        """Returns action (0/1) and confidence of action 1"""
        q, score = self.lrm.query(state, k=5)
        
        # 简单的 Argmax 策略
        action = int(np.argmax(q))
        
        # 置信度：如果是 YES，取 q[1]；如果是 NO，置信度设为 0 (或者 q[1]本身)
        # 这里为了投票方便，直接返回 q[1] (YES 的概率)
        yes_confidence = q[1]
        
        return action, yes_confidence

    def train_binary(self, state, is_me: bool):
        target = 1 if is_me else 0
        reward = 1.0
        self.lrm.update(state, target, reward)

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"

# =============================================================================
# 5. Vision System (Taboo Routing)
# =============================================================================
class StructonVisionSystem:
    def __init__(self, capacity=200, n_connections=3):
        self.extractor = StateExtractor()
        self.structons = []
        self.label_to_structon = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        Structon._id_counter = 0
        self.structons = []
        self.label_to_structon = {}
        for l in labels:
            s = Structon(l, self.capacity, self.n_connections)
            self.structons.append(s)
            self.label_to_structon[l] = s
            print(f"  + {s.id} label='{l}'")
        
        print("\n设置稀疏连接...")
        for s in self.structons:
            s.set_connections(self.structons)
            print(f"  {s.id} ({s.label}) → {s.get_connections_str()}")

    def train_epoch(self, samples, n_negatives=5):
        np.random.shuffle(samples)
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 1. 训练正样本 (YES)
            self.label_to_structon[label].train_binary(state, True)
            
            # 2. 训练负样本 (NO) - 随机抽几个
            # [关键] 负样本数量不要太多，以免淹没正样本的记忆，但也不能太少
            others = [s for s in self.structons if s.label != label]
            negs = np.random.choice(others, n_negatives, replace=False)
            for s in negs:
                s.train_binary(state, False)

    def predict_routing(self, image, max_hops=20):
        """禁忌路由预测"""
        state = self.extractor.extract(image)
        
        # 随机入口
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            # 1. 问 Structon
            action, conf = current.decide(state)
            
            # 2. YES -> 捕获
            if action == 1: 
                return current.label, path
            
            # 3. NO -> 禁忌搜索
            candidates = [c for c in current.connections if c.id not in visited]
            
            if candidates:
                # 简单随机游走，也可以优化为“选择最不确定的邻居”
                current = np.random.choice(candidates)
            else:
                # 死胡同 -> 随机跳跃
                unvisited = [s for s in self.structons if s.id not in visited]
                if unvisited:
                    current = np.random.choice(unvisited)
                    path.append("JUMP")
                else:
                    break # 全网遍历完毕
                    
        return current.label, path

    def predict_voting(self, image):
        """全网投票 (模拟无限跳跃)"""
        state = self.extractor.extract(image)
        votes = {}
        
        best_no_structon = None
        best_no_conf = -1.0
        
        for s in self.structons:
            action, yes_prob = s.decide(state)
            
            # 如果 Structon 认为是 YES，记录票数
            if action == 1:
                votes[s.label] = votes.get(s.label, 0) + yes_prob
            else:
                # 记录“即使是NO，但也没那么NO”的节点作为备选
                if yes_prob > best_no_conf:
                    best_no_conf = yes_prob
                    best_no_structon = s.label
        
        if not votes:
            # 如果没一个人说是，就选“最不抗拒”的那个
            return best_no_structon if best_no_structon else "?", 0.0
            
        best = max(votes, key=votes.get)
        return best, votes[best]

    def print_stats(self):
        print("\n=== Stats ===")
        for s in self.structons:
            print(f"  {s.id} [{s.label}] Mem:{s.lrm.size}")

# =============================================================================
# 6. Run
# =============================================================================
def run_experiment(capacity=200, epochs=30, connections=3, n_negatives=5):
    print("="*70)
    print(f"Structon v9.5 - Zero-Mean & Top-K")
    print("="*70)
    
    images, labels, t_images, t_labels = load_mnist()
    
    system = StructonVisionSystem(capacity=capacity, n_connections=connections)
    system.build([str(i) for i in range(10)])
    
    # 准备数据：每类 100 个
    samples = []
    for d in range(10):
        idxs = np.where(labels == d)[0][:100]
        for i in idxs: samples.append((images[i], str(d)))
        
    print(f"Training on {len(samples)} samples...")
    t0 = time.time()
    
    for ep in range(epochs):
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (ep+1) % 5 == 0:
            corr = 0
            check = 200
            idxs = np.random.choice(len(t_images), check)
            for i in idxs:
                pred, _ = system.predict_voting(t_images[i])
                if pred == str(t_labels[i]): corr += 1
            print(f"  Epoch {ep+1}: {corr/check*100:.1f}% (Voting Acc)")

    print(f"Done in {time.time()-t0:.1f}s")
    system.print_stats()
    
    # 测试 1: 禁忌路由 (Taboo Routing)
    print("\n=== Testing (Routing) ===")
    corr = 0
    total = 1000
    idxs = np.random.choice(len(t_images), total, replace=False)
    
    stats = {str(i): {'c':0, 't':0} for i in range(10)}
    
    for i in idxs:
        lbl = str(t_labels[i])
        pred, path = system.predict_routing(t_images[i])
        stats[lbl]['t'] += 1
        if pred == lbl:
            corr += 1
            stats[lbl]['c'] += 1
            
    print(f"Routing Accuracy: {corr/total*100:.1f}%")
    for d in range(10):
        s = stats[str(d)]
        if s['t'] > 0:
            print(f"  {d}: {s['c']/s['t']*100:.1f}%")
            
    # 测试 2: 全网投票 (Voting)
    print("\n=== Testing (Voting) ===")
    corr_v = 0
    for i in idxs:
        pred, _ = system.predict_voting(t_images[i])
        if pred == str(t_labels[i]): corr_v += 1
    print(f"Voting Accuracy: {corr_v/total*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=200)
    parser.add_argument('--connections', type=int, default=3)
    parser.add_argument('--negatives', type=int, default=5)
    args = parser.parse_args()
    
    run_experiment(
        capacity=args.capacity, 
        connections=args.connections,
        n_negatives=args.negatives
    )
