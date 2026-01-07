#!/usr/bin/env python3
"""
Structon Vision Test - 修复版
修复了归一化、阈值和训练顺序问题
"""

import numpy as np
import gzip
import os
import urllib.request

# ... (Load MNIST function remains the same) ...
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
            urllib.request.urlretrieve(base_url + filename, filepath)
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                f.read(16)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            else:
                f.read(8)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

class OneHotExtractor:
    def __init__(self, grid_size=7, n_bins=4):
        self.grid_size = grid_size
        self.n_bins = n_bins
        self.dim = grid_size * grid_size * n_bins
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                value = np.mean(block)
                one_hot = np.zeros(self.n_bins)
                bin_idx = min(int(value * self.n_bins), self.n_bins - 1)
                one_hot[bin_idx] = 1.0
                features.extend(one_hot)
        
        state = np.array(features, dtype=np.float32)
        # [修复 1] 强制归一化！
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state

class ZeroMeanExtractor:
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        self.dim = grid_size * grid_size
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0 - 0.5
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                features.append(np.mean(block))
        
        state = np.array(features, dtype=np.float32)
        # Zero-Mean 已经包含归一化
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state

class LRM:
    def __init__(self, state_dim, capacity=300, threshold=0.9):
        self.n_actions = 2
        self.capacity = capacity
        self.state_dim = state_dim
        self.keys = []
        self.values = []
        self.learning_rate = 0.3
        # [修复 2] 支持自定义阈值
        self.similarity_threshold = threshold

    def query(self, state):
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.dot(self.keys, state)
        best_score = float(np.max(scores))
        
        # Top-K weighted voting
        k = min(5, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        
        q_values = np.zeros(self.n_actions)
        w_sum = 0
        for idx in top_indices:
            s = scores[idx]
            if s > 0.5: # 只有正相关的才参与投票
                w = s ** 2
                q_values += w * self.values[idx]
                w_sum += w
        
        if w_sum > 0:
            q_values /= w_sum
            
        return q_values, best_score

    def update(self, state, action, reward):
        if self.keys:
            scores = np.dot(self.keys, state)
            best_idx = int(np.argmax(scores))
            # 只有相似度足够高才更新
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                return

        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
            
        self.keys.append(state)
        self.values.append(new_q)
    
    @property
    def size(self):
        return len(self.keys)

def test_extractor(extractor, name, threshold, n_train=100, n_test=200):
    print(f"\n测试: {name} (阈值={threshold})")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    results = []
    
    for target_digit in range(10):
        lrm = LRM(state_dim=extractor.dim, capacity=300, threshold=threshold)
        
        # 准备数据
        pos_indices = np.where(train_labels == target_digit)[0][:n_train]
        neg_indices = np.where(train_labels != target_digit)[0][:n_train] # 保持 1:1 平衡
        
        # [修复 3] 混合训练数据
        samples = []
        for idx in pos_indices: samples.append((train_images[idx], 1)) # YES
        for idx in neg_indices: samples.append((train_images[idx], 0)) # NO
        
        for epoch in range(10):
            np.random.shuffle(samples) # 打乱顺序！
            for img, label in samples:
                state = extractor.extract(img)
                if label == 1:
                    lrm.update(state, 1, 1.0)
                else:
                    lrm.update(state, 0, 1.0)
        
        # 测试
        pos_test = np.where(test_labels == target_digit)[0][:n_test]
        neg_test = np.where(test_labels != target_digit)[0][:n_test]
        
        pos_correct = sum(1 for idx in pos_test if np.argmax(lrm.query(extractor.extract(test_images[idx]))[0]) == 1)
        neg_correct = sum(1 for idx in neg_test if np.argmax(lrm.query(extractor.extract(test_images[idx]))[0]) == 0)
        
        yes_rate = pos_correct / len(pos_test) * 100
        no_rate = neg_correct / len(neg_test) * 100
        
        results.append((target_digit, yes_rate, no_rate, lrm.size))
        print(f"  数字 {target_digit}: YES={yes_rate:.1f}%, NO={no_rate:.1f}%, mem={lrm.size}")
    
    avg_yes = np.mean([r[1] for r in results])
    avg_no = np.mean([r[2] for r in results])
    print(f"平均: YES={avg_yes:.1f}%, NO={avg_no:.1f}%")

if __name__ == "__main__":
    # One-Hot 归一化后，相似度通常较低（因为只有局部重叠），阈值设低一点
    test_extractor(OneHotExtractor(7, 4), "One-Hot (Normalized)", threshold=0.65)
    
    # Zero-Mean 相似度很高，阈值必须非常高才能区分
    test_extractor(ZeroMeanExtractor(7), "Zero-Mean", threshold=0.92)
