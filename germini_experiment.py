#!/usr/bin/env python3
"""
Structon Vision v9.6 - High Threshold & Z-Score

修复 (Fixes):
1. 阈值提升 (Threshold Boost): 从 0.6 提升到 0.95。
   - 之前: 0.6 的阈值太低，导致 0 和 1 (相似度0.78) 被合并，记忆坍缩为 1。
   - 现在: 0.95 的阈值强迫 LRM 区分细微差别，防止记忆混淆。

2. Z-Score 归一化: 
   - 之前: img - 0.5 导致背景(-0.5)之间产生虚假的正相关。
   - 现在: (img - mean) / std，背景变为 ~0，消除了背景干扰。
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Set
import argparse

# ... (Load MNIST 保持不变) ...
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
            # print(f"Downloading {filename}...")
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
# 2. 特征提取 (Z-Score 优化版)
# =============================================================================
class StateExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        
        # [核心修复] Z-Score 归一化 (Per-image)
        # 让背景变成 0 (或接近0的负数)，前景变成正数
        # 这样背景*背景 ≈ 0，不再产生干扰
        mean = np.mean(img)
        std = np.std(img) + 1e-5
        img = (img - mean) / std
        
        # 限制范围，防止离群点
        img = np.clip(img, -3.0, 3.0)
        
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
        
        # 3. 结构特征 (二值化后 Z-Score)
        # 用原始数据做二值化，比较稳定
        raw_binary = (image > 80).astype(np.float32)
        b_mean = np.mean(raw_binary)
        b_std = np.std(raw_binary) + 1e-5
        binary = (raw_binary - b_mean) / b_std
        
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
# 3. LRM (High Threshold)
# =============================================================================
class LRM:
    def __init__(self, state_dim=55, capacity=200):
        self.state_dim = state_dim
        self.n_actions = 2
        self.capacity = capacity
        
        self.key_dim = 64
        self.projection = np.random.randn(state_dim, self.key_dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.5
        
        # [核心修复] 提升阈值！防止混淆
        self.similarity_threshold = 0.95

    def _compute_key(self, state):
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6: key /= norm
        return key

    def query(self, state, k=3) -> Tuple[np.ndarray, float]:
        if not self.keys: return np.zeros(self.n_actions), 0.0
        
        key = self._compute_key(state)
        scores = np.dot(self.keys, key)
        
        # Top-K
        if len(scores) <= k:
            indices = np.arange(len(scores))
        else:
            indices = np.argpartition(scores, -k)[-k:]
        
        best_score = float(np.max(scores))
        
        q_values = np.zeros(self.n_actions)
        w_sum = 0.0
        
        for i in indices:
            score = scores[i]
            # 只有非常确信的才投票
            if score > 0.5: 
                w = score ** 4  # 更激进的加权
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
            
            # 只有极其相似才合并
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx]
                target_q = np.zeros(self.n_actions)
                target_q[action] = reward
                self.values[best_idx] = old_q + self.learning_rate * (target_q - old_q)
                self.access_counts[best_idx] += 1
                return

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
# 4. Structon
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
        q, score = self.lrm.query(state, k=3)
        action = int(np.argmax(q))
        yes_confidence = q[1]
        return action, yes_confidence

    def train_binary(self, state, is_me: bool):
        target = 1 if is_me else 0
        reward = 1.0
        self.lrm.update(state, target, reward)

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"

# =============================================================================
# 5. Vision System
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
            
        for s in self.structons:
            s.set_connections(self.structons)

    def train_epoch(self, samples, n_negatives=5):
        np.random.shuffle(samples)
        for img, label in samples:
            state = self.extractor.extract(img)
            self.label_to_structon[label].train_binary(state, True)
            others = [s for s in self.structons if s.label != label]
            negs = np.random.choice(others, n_negatives, replace=False)
            for s in negs: s.train_binary(state, False)

    def predict_voting(self, image):
        state = self.extractor.extract(image)
        votes = {}
        best_no_structon = None
        best_no_conf = -1.0
        
        for s in self.structons:
            action, yes_prob = s.decide(state)
            if action == 1:
                votes[s.label] = votes.get(s.label, 0) + yes_prob
            else:
                if yes_prob > best_no_conf:
                    best_no_conf = yes_prob
                    best_no_structon = s.label
        
        if not votes:
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
    print(f"Structon v9.6 - Z-Score & 0.95 Threshold")
    print("="*70)
    
    images, labels, t_images, t_labels = load_mnist()
    
    system = StructonVisionSystem(capacity=capacity, n_connections=connections)
    system.build([str(i) for i in range(10)])
    
    # 扩大训练集: 每类 200 个
    samples = []
    for d in range(10):
        idxs = np.where(labels == d)[0][:200]
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
            print(f"  Epoch {ep+1}: {corr/check*100:.1f}%")

    print(f"Done in {time.time()-t0:.1f}s")
    system.print_stats()
    
    print("\n=== Testing (Voting) ===")
    corr = 0
    total = 1000
    idxs = np.random.choice(len(t_images), total, replace=False)
    for i in idxs:
        pred, _ = system.predict_voting(t_images[i])
        if pred == str(t_labels[i]): corr += 1
    print(f"Accuracy: {corr/total*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--capacity', type=int, default=300)
    args = parser.parse_args()
    run_experiment(capacity=args.capacity)
