#!/usr/bin/env python3
"""
测试单个 Structon 的二元分类能力

问题：LRM 能不能学会区分 "是5" vs "不是5"？
"""

import numpy as np
import gzip
import os
import urllib.request

# MNIST 加载
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

# 特征提取
class StateExtractor:
    def extract(self, image):
        img = image.astype(np.float32) / 255.0
        features = []
        h, w = img.shape
        
        # 5x5 Grid
        bh, bw = h // 5, w // 5
        for i in range(5):
            for j in range(5):
                features.append(np.mean(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]))
        
        # 投影
        for i in range(5): 
            features.append(np.mean(img[i*(h//5):(i+1)*(h//5), :]))
        for j in range(5): 
            features.append(np.mean(img[:, j*(w//5):(j+1)*(w//5)]))
        
        # 结构
        binary = (img > 0.3).astype(np.uint8)
        features.append(np.mean(binary))
        features.append(np.mean(binary[:h//2, :]) - np.mean(binary[h//2:, :]))
        features.append(np.mean(binary[:, :w//2]) - np.mean(binary[:, w//2:]))
        features.append(np.mean(binary[h//4:3*h//4, w//4:3*w//4]))
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6: 
            state = state / norm
        return state

# LRM
class LRM:
    def __init__(self, state_dim=39, capacity=500):  # 增加容量
        self.n_actions = 2
        self.capacity = capacity
        
        self.projection = np.random.randn(state_dim, 16).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys = []
        self.values = []
        self.learning_rate = 0.5
        self.similarity_threshold = 0.85  # 降低阈值！

    def _compute_key(self, state):
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6: 
            key /= norm
        return key

    def query(self, state):
        if not self.keys: 
            return np.array([0.0, 0.0]), 0.0
        
        key = self._compute_key(state)
        scores = np.array(self.keys) @ key
        
        weights = np.maximum(scores, 0) ** 3
        w_sum = np.sum(weights)
        if w_sum < 1e-6: 
            return np.array([0.0, 0.0]), 0.0
        
        weights /= w_sum
        q_values = np.zeros(2)
        for i, w in enumerate(weights):
            if w > 0.01: 
                q_values += w * self.values[i]
        return q_values, float(np.max(scores))

    def update(self, state, action, reward):
        key = self._compute_key(state)
        
        if self.keys:
            scores = np.array(self.keys) @ key
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                return

        new_q = np.zeros(2, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            self.keys.pop(0)
            self.values.pop(0)
            
        self.keys.append(key)
        self.values.append(new_q)
    
    @property
    def size(self):
        return len(self.keys)


def test_single_structon(target_digit=5, n_train=100, n_test=200):
    """
    测试单个 Structon 识别一个数字的能力
    """
    print(f"\n{'='*60}")
    print(f"测试 Structon 识别数字 {target_digit}")
    print(f"{'='*60}")
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    extractor = StateExtractor()
    lrm = LRM(capacity=300)
    
    # 准备训练数据
    pos_indices = np.where(train_labels == target_digit)[0][:n_train]
    neg_indices = np.where(train_labels != target_digit)[0][:n_train * 2]
    
    print(f"\n训练: {len(pos_indices)} 正样本, {len(neg_indices)} 负样本")
    
    # 训练
    for epoch in range(20):
        # 正样本 - 更激进的训练
        for idx in pos_indices:
            state = extractor.extract(train_images[idx])
            for _ in range(3):  # 重复训练
                lrm.update(state, 1, 2.0)   # YES 高奖励
            lrm.update(state, 0, -1.0)  # not NO
        
        # 负样本 - 减少数量
        np.random.shuffle(neg_indices)
        for idx in neg_indices[:n_train//2]:  # 只用一半
            state = extractor.extract(train_images[idx])
            lrm.update(state, 0, 0.5)   # NO
            lrm.update(state, 1, -0.5)  # not YES（轻微惩罚）
    
    print(f"LRM 记忆数: {lrm.size}")
    
    # 测试
    print(f"\n=== 测试 ===")
    
    # 测试正样本（应该说 YES）
    pos_test = np.where(test_labels == target_digit)[0][:n_test]
    pos_correct = 0
    for idx in pos_test:
        state = extractor.extract(test_images[idx])
        q, _ = lrm.query(state)
        if np.argmax(q) == 1:  # YES
            pos_correct += 1
    
    # 测试负样本（应该说 NO）
    neg_test = np.where(test_labels != target_digit)[0][:n_test]
    neg_correct = 0
    for idx in neg_test:
        state = extractor.extract(test_images[idx])
        q, _ = lrm.query(state)
        if np.argmax(q) == 0:  # NO
            neg_correct += 1
    
    print(f"正样本（{target_digit}）识别率: {pos_correct}/{len(pos_test)} = {pos_correct/len(pos_test)*100:.1f}%")
    print(f"负样本（非{target_digit}）拒绝率: {neg_correct}/{len(neg_test)} = {neg_correct/len(neg_test)*100:.1f}%")
    
    total = pos_correct + neg_correct
    total_samples = len(pos_test) + len(neg_test)
    print(f"\n总体准确率: {total}/{total_samples} = {total/total_samples*100:.1f}%")
    
    # 分析 Q 值分布
    print(f"\n=== Q 值分析 ===")
    
    pos_yes_q = []
    neg_yes_q = []
    
    for idx in pos_test[:50]:
        state = extractor.extract(test_images[idx])
        q, _ = lrm.query(state)
        pos_yes_q.append(q[1])
    
    for idx in neg_test[:50]:
        state = extractor.extract(test_images[idx])
        q, _ = lrm.query(state)
        neg_yes_q.append(q[1])
    
    print(f"正样本的 YES Q值: 均值={np.mean(pos_yes_q):.3f}, 标准差={np.std(pos_yes_q):.3f}")
    print(f"负样本的 YES Q值: 均值={np.mean(neg_yes_q):.3f}, 标准差={np.std(neg_yes_q):.3f}")
    
    # 分离度
    separation = np.mean(pos_yes_q) - np.mean(neg_yes_q)
    print(f"分离度: {separation:.3f}")
    
    return pos_correct/len(pos_test), neg_correct/len(neg_test)


if __name__ == "__main__":
    # 测试所有数字
    results = []
    for digit in range(10):
        pos_acc, neg_acc = test_single_structon(target_digit=digit)
        results.append((digit, pos_acc, neg_acc))
    
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    for digit, pos_acc, neg_acc in results:
        print(f"  数字 {digit}: YES率={pos_acc*100:.1f}%, NO率={neg_acc*100:.1f}%")
