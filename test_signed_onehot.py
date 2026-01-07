#!/usr/bin/env python3
"""
测试 Signed One-Hot 特征

核心思想：
- 普通 One-Hot: [1,0,0,0] - 所有值都是正的
- Signed One-Hot: [-1,0,0,+1] - 有正有负

这样不同数字的特征向量可以指向相反方向，
相似度可以是负数，区分度更高！
"""

import numpy as np
import gzip
import os
import urllib.request

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


class SignedOneHotExtractor:
    """
    Signed One-Hot 特征提取器
    
    每个格子的值被编码成带符号的 one-hot：
    - 很暗 (< 0.2):  [-1, 0, 0, 0]
    - 较暗 (0.2-0.4): [0, -0.5, 0, 0]
    - 较亮 (0.4-0.6): [0, 0, +0.5, 0]
    - 很亮 (> 0.6):  [0, 0, 0, +1]
    
    这样：
    - 暗的格子产生负值
    - 亮的格子产生正值
    - 不同数字会有正负差异
    """
    def __init__(self, grid_size=7, n_bins=4):
        self.grid_size = grid_size
        self.n_bins = n_bins
        self.dim = grid_size * grid_size * n_bins
        
        # 每个 bin 的符号和强度
        # bin 0, 1 是负的（暗），bin 2, 3 是正的（亮）
        self.bin_values = [-1.0, -0.5, 0.5, 1.0]
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                value = np.mean(block)
                
                # Signed One-Hot
                one_hot = np.zeros(self.n_bins)
                bin_idx = min(int(value * self.n_bins), self.n_bins - 1)
                one_hot[bin_idx] = self.bin_values[bin_idx]
                features.extend(one_hot)
        
        state = np.array(features, dtype=np.float32)
        
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        
        return state


class ContrastOneHotExtractor:
    """
    对比 One-Hot：更极端的正负编码
    
    - 暗 (< 0.3): -1
    - 亮 (> 0.3): +1
    
    只有 2 个 bin，但对比更强烈
    """
    def __init__(self, grid_size=7, threshold=0.3):
        self.grid_size = grid_size
        self.threshold = threshold
        self.dim = grid_size * grid_size  # 每个格子只有 1 个值
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                value = np.mean(block)
                
                # 二值化：暗=-1，亮=+1
                if value < self.threshold:
                    features.append(-1.0)
                else:
                    features.append(1.0)
        
        state = np.array(features, dtype=np.float32)
        
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        
        return state


class ZeroMeanPlusStructureExtractor:
    """
    Zero-Mean + 结构特征
    
    结合连续特征和结构差异
    """
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        # grid + 差异特征
        self.dim = grid_size * grid_size + 20
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0 - 0.5  # Zero-Mean
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        # Grid 特征
        grid_values = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                v = np.mean(block)
                features.append(v)
                grid_values.append(v)
        
        grid = np.array(grid_values).reshape(self.grid_size, self.grid_size)
        
        # 结构差异特征（相邻格子的差）
        # 水平差异
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                features.append(grid[i, j] - grid[i, j+1])
        
        # 垂直差异
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                features.append(grid[i, j] - grid[i+1, j])
        
        # 对角线
        features.append(np.mean([grid[i, i] for i in range(self.grid_size)]))
        features.append(np.mean([grid[i, self.grid_size-1-i] for i in range(self.grid_size)]))
        
        # 四象限差异
        mid = self.grid_size // 2
        features.append(np.mean(grid[:mid, :mid]) - np.mean(grid[mid:, mid:]))  # 左上-右下
        features.append(np.mean(grid[:mid, mid:]) - np.mean(grid[mid:, :mid]))  # 右上-左下
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state


class LRM:
    """LRM for testing"""
    def __init__(self, state_dim, capacity=300):
        self.n_actions = 2
        self.capacity = capacity
        self.state_dim = state_dim
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.90

    def query(self, state):
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.array([np.dot(k, state) for k in self.keys])
        
        # Top-5
        k = min(5, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        # 只用正相似度
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


def test_extractor(extractor, name, n_train=100, n_test=200):
    """测试一种特征提取器"""
    print(f"\n{'='*60}")
    print(f"测试: {name}")
    print(f"特征维度: {extractor.dim}")
    print(f"{'='*60}")
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    results = []
    
    for target_digit in range(10):
        lrm = LRM(state_dim=extractor.dim, capacity=300)
        
        # 训练
        pos_indices = np.where(train_labels == target_digit)[0][:n_train]
        neg_indices = np.where(train_labels != target_digit)[0][:n_train]
        
        for epoch in range(15):
            for idx in pos_indices:
                state = extractor.extract(train_images[idx])
                lrm.update(state, 1, 1.5)
                lrm.update(state, 0, -0.8)
            
            np.random.shuffle(neg_indices)
            for idx in neg_indices[:n_train//2]:
                state = extractor.extract(train_images[idx])
                lrm.update(state, 0, 0.8)
                lrm.update(state, 1, -0.5)
        
        # 测试
        pos_test = np.where(test_labels == target_digit)[0][:n_test]
        neg_test = np.where(test_labels != target_digit)[0][:n_test]
        
        pos_correct = sum(
            1 for idx in pos_test
            if np.argmax(lrm.query(extractor.extract(test_images[idx]))[0]) == 1
        )
        
        neg_correct = sum(
            1 for idx in neg_test
            if np.argmax(lrm.query(extractor.extract(test_images[idx]))[0]) == 0
        )
        
        yes_rate = pos_correct / len(pos_test) * 100
        no_rate = neg_correct / len(neg_test) * 100
        
        results.append((target_digit, yes_rate, no_rate, lrm.size))
        print(f"  数字 {target_digit}: YES={yes_rate:.1f}%, NO={no_rate:.1f}%, mem={lrm.size}")
    
    avg_yes = np.mean([r[1] for r in results])
    avg_no = np.mean([r[2] for r in results])
    balanced = (avg_yes + avg_no) / 2
    print(f"\n平均: YES={avg_yes:.1f}%, NO={avg_no:.1f}%, 平衡={balanced:.1f}%")
    
    return avg_yes, avg_no, balanced


def analyze_similarity(extractor, name):
    """分析相似度"""
    print(f"\n{'='*60}")
    print(f"相似度分析: {name}")
    print(f"{'='*60}")
    
    train_images, train_labels, _, _ = load_mnist()
    
    digit_features = {}
    for d in range(10):
        indices = np.where(train_labels == d)[0][:100]
        features = [extractor.extract(train_images[idx]) for idx in indices]
        digit_features[d] = np.mean(features, axis=0)
    
    print("\n数字间相似度:")
    print("    ", end="")
    for d in range(10):
        print(f"  {d}  ", end="")
    print()
    
    all_sims = []
    for d1 in range(10):
        print(f" {d1}: ", end="")
        for d2 in range(10):
            sim = np.dot(digit_features[d1], digit_features[d2])
            if d1 == d2:
                print(f" --- ", end="")
            else:
                print(f"{sim:5.2f}", end="")
                all_sims.append(sim)
        print()
    
    print(f"\n非对角线相似度: 最小={min(all_sims):.2f}, 最大={max(all_sims):.2f}, 平均={np.mean(all_sims):.2f}")


if __name__ == "__main__":
    extractors = [
        (SignedOneHotExtractor(grid_size=7, n_bins=4), "Signed One-Hot (7x7, 4-bin)"),
        (ContrastOneHotExtractor(grid_size=7, threshold=0.3), "Contrast Binary (7x7)"),
        (ZeroMeanPlusStructureExtractor(grid_size=7), "Zero-Mean + Structure (7x7)"),
    ]
    
    # 相似度分析
    for extractor, name in extractors:
        analyze_similarity(extractor, name)
    
    # 性能测试
    print("\n" + "=" * 70)
    print("性能测试")
    print("=" * 70)
    
    results = []
    for extractor, name in extractors:
        yes, no, balanced = test_extractor(extractor, name)
        results.append((name, yes, no, balanced))
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print(f"{'特征':<35} {'YES%':>8} {'NO%':>8} {'平衡%':>8}")
    print("-" * 70)
    for name, yes, no, balanced in results:
        print(f"{name:<35} {yes:>7.1f}% {no:>7.1f}% {balanced:>7.1f}%")
