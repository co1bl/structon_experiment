#!/usr/bin/env python3
"""
测试 One-Hot 特征对 LRM 的效果

方案：把 5x5 grid 离散化成 one-hot
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


class OneHotExtractor:
    """
    One-Hot 特征提取器
    
    把图像分成 7x7 grid，每个格子离散化成 4 个等级
    7x7 x 4 = 196 维
    """
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
                # 计算格子的平均值
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                value = np.mean(block)
                
                # 离散化成 one-hot
                one_hot = np.zeros(self.n_bins)
                bin_idx = min(int(value * self.n_bins), self.n_bins - 1)
                one_hot[bin_idx] = 1.0
                features.extend(one_hot)
        
        return np.array(features, dtype=np.float32)


class ZeroMeanExtractor:
    """Zero-Mean 连续特征（对比用）"""
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
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state


class LRM:
    """简单的 LRM 用于测试"""
    def __init__(self, state_dim, capacity=300):
        self.n_actions = 2
        self.capacity = capacity
        self.state_dim = state_dim
        
        self.keys = []
        self.values = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.75

    def query(self, state):
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        # 直接用点积作为相似度
        scores = np.array([np.dot(k, state) for k in self.keys])
        
        # Top-5
        k = min(5, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        valid_mask = top_scores > 0
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
        # 找最相似的
        if self.keys:
            scores = np.array([np.dot(k, state) for k in self.keys])
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                return

        # 新记忆
        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
            
        self.keys.append(state.copy())
        self.values.append(new_q)
    
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
            # 正样本
            for idx in pos_indices:
                state = extractor.extract(train_images[idx])
                lrm.update(state, 1, 1.5)
                lrm.update(state, 0, -0.8)
            
            # 负样本
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
    
    # 总结
    avg_yes = np.mean([r[1] for r in results])
    avg_no = np.mean([r[2] for r in results])
    print(f"\n平均: YES={avg_yes:.1f}%, NO={avg_no:.1f}%")
    
    return results


def analyze_similarity(extractor, name):
    """分析不同数字之间的特征相似度"""
    print(f"\n{'='*60}")
    print(f"相似度分析: {name}")
    print(f"{'='*60}")
    
    train_images, train_labels, _, _ = load_mnist()
    
    # 计算每个数字的平均特征
    digit_features = {}
    for d in range(10):
        indices = np.where(train_labels == d)[0][:100]
        features = [extractor.extract(train_images[idx]) for idx in indices]
        digit_features[d] = np.mean(features, axis=0)
    
    # 计算相似度矩阵
    print("\n数字间相似度（点积）:")
    print("    ", end="")
    for d in range(10):
        print(f"  {d}  ", end="")
    print()
    
    for d1 in range(10):
        print(f" {d1}: ", end="")
        for d2 in range(10):
            sim = np.dot(digit_features[d1], digit_features[d2])
            if d1 == d2:
                print(f" --- ", end="")
            else:
                print(f"{sim:5.2f}", end="")
        print()


if __name__ == "__main__":
    # 测试 One-Hot
    onehot_extractor = OneHotExtractor(grid_size=7, n_bins=4)
    
    # 测试 Zero-Mean
    zeromean_extractor = ZeroMeanExtractor(grid_size=7)
    
    # 相似度分析
    analyze_similarity(onehot_extractor, "One-Hot (7x7, 4-bin)")
    analyze_similarity(zeromean_extractor, "Zero-Mean (7x7)")
    
    # 性能测试
    print("\n" + "=" * 70)
    print("性能测试")
    print("=" * 70)
    
    test_extractor(onehot_extractor, "One-Hot (7x7, 4-bin)")
    test_extractor(zeromean_extractor, "Zero-Mean (7x7)")
