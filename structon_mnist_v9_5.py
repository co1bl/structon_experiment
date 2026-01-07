#!/usr/bin/env python3
"""
Structon Vision v9.5 - Zero-Mean + Top-K

核心修复：
1. Zero-Mean 特征：img - 0.5，让特征有正有负
2. Top-K 查询：只看最相似的 K 条记忆
3. 二元大脑 + 禁忌路由

这解决了"正象限陷阱"问题！
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict
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
# 2. 特征提取 - Zero-Mean 版本
# =============================================================================
class StateExtractor:
    """
    Zero-Mean 特征提取器
    
    关键改变：img = img - 0.5
    让特征有正有负，分布在原点四周
    """
    def extract(self, image: np.ndarray) -> np.ndarray:
        # 归一化到 [-0.5, 0.5]（不是 [0, 1]）
        img = image.astype(np.float32) / 255.0 - 0.5
        
        features = []
        h, w = img.shape
        
        # 1. 5x5 Grid (25维)
        bh, bw = h // 5, w // 5
        for i in range(5):
            for j in range(5):
                features.append(np.mean(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]))
        
        # 2. 投影 (10维)
        for i in range(5): 
            features.append(np.mean(img[i*(h//5):(i+1)*(h//5), :]))
        for j in range(5): 
            features.append(np.mean(img[:, j*(w//5):(j+1)*(w//5)]))
        
        # 3. 结构特征 (20维)
        # 注意：二值化用原始值，但输出仍然是 zero-mean
        binary = (img > -0.2).astype(np.float32) - 0.5  # 也做 zero-mean
        
        # 基本密度
        features.append(np.mean(binary))
        
        # 四象限
        features.append(np.mean(binary[:h//2, :w//2]))
        features.append(np.mean(binary[:h//2, w//2:]))
        features.append(np.mean(binary[h//2:, :w//2]))
        features.append(np.mean(binary[h//2:, w//2:]))
        
        # 上下左右差
        features.append(np.mean(binary[:h//2, :]) - np.mean(binary[h//2:, :]))
        features.append(np.mean(binary[:, :w//2]) - np.mean(binary[:, w//2:]))
        
        # 中心密度
        features.append(np.mean(binary[h//4:3*h//4, w//4:3*w//4]))
        features.append(np.mean(binary[h//3:2*h//3, w//3:2*w//3]))
        
        # 水平切片
        features.append(np.mean(binary[2:5, :]))
        features.append(np.mean(binary[h//2-2:h//2+2, :]))
        features.append(np.mean(binary[-5:-2, :]))
        
        # 垂直切片
        features.append(np.mean(binary[:, w//2-2:w//2+2]))
        features.append(np.mean(binary[:, 2:5]))
        features.append(np.mean(binary[:, -5:-2]))
        
        # 对角线
        diag1 = np.mean([binary[i, i] for i in range(min(h, w))])
        diag2 = np.mean([binary[i, w-1-i] for i in range(min(h, w))])
        features.append(diag1)
        features.append(diag2)
        features.append(diag1 - diag2)
        
        # 边缘
        features.append(np.mean(binary[0, :]))
        features.append(np.mean(binary[-1, :]))
        
        state = np.array(features, dtype=np.float32)
        
        # 归一化（保持方向，统一长度）
        norm = np.linalg.norm(state)
        if norm > 1e-6: 
            state = state / norm
        return state

# =============================================================================
# 3. LRM - Top-K 版本
# =============================================================================
class LRM:
    """
    Local Resonant Memory - Top-K 版本
    
    改进：
    - 只看最相似的 K 条记忆
    - 避免噪音干扰
    """
    def __init__(self, state_dim=55, capacity=300, top_k=5):
        self.n_actions = 2  # NO, YES
        self.capacity = capacity
        self.top_k = top_k
        
        # 随机投影
        self.projection = np.random.randn(state_dim, 20).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.80  # 更低的阈值

    def _compute_key(self, state):
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6: 
            key /= norm
        return key

    def query(self, state) -> Tuple[np.ndarray, float]:
        """Top-K 查询"""
        if not self.keys: 
            return np.zeros(self.n_actions), 0.0
        
        key = self._compute_key(state)
        scores = np.array(self.keys) @ key
        
        # Top-K：只看最相似的 K 条
        k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        # 只用正相似度的记忆
        valid_mask = top_scores > 0
        if not np.any(valid_mask):
            return np.zeros(self.n_actions), 0.0
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        # 加权平均
        weights = top_scores ** 2
        weights /= np.sum(weights)
        
        q_values = np.zeros(self.n_actions)
        for idx, w in zip(top_indices, weights):
            q_values += w * self.values[idx]
        
        return q_values, float(np.max(top_scores))

    def update(self, state, action, reward):
        """更新记忆"""
        key = self._compute_key(state)
        
        # 尝试更新现有记忆
        if self.keys:
            scores = np.array(self.keys) @ key
            best_idx = int(np.argmax(scores))
            if scores[best_idx] > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                self.access_counts[best_idx] += 1
                return

        # 写入新记忆
        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            # 移除最少访问的
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
            
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
    """二元大脑 Structon"""
    _id_counter = 0
    
    def __init__(self, label: str, capacity=300, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        self.lrm = LRM(capacity=capacity)
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
    def decide(self, state) -> Tuple[int, float]:
        """决策：是我的吗？"""
        q, score = self.lrm.query(state)
        action = int(np.argmax(q))
        confidence = q[action] if q[action] != 0 else score
        return action, confidence

    def train_binary(self, state, is_me: bool):
        """训练二元分类"""
        if is_me:
            self.lrm.update(state, 1, 1.5)   # YES
            self.lrm.update(state, 0, -1.0)  # not NO
        else:
            self.lrm.update(state, 0, 0.8)   # NO
            self.lrm.update(state, 1, -0.8)  # not YES

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"

# =============================================================================
# 5. Vision System
# =============================================================================
class StructonVisionSystem:
    """Structon 视觉系统 v9.5"""
    
    def __init__(self, capacity=300, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        print("\n=== 创建 Structon v9.5 ===")
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
        """训练一个 epoch"""
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 正样本多次训练
            correct_s = self.label_to_structon[label]
            for _ in range(2):
                correct_s.train_binary(state, is_me=True)
            
            # 负样本
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_binary(state, is_me=False)

    def predict(self, image, max_hops=20) -> Tuple[str, List[str]]:
        """禁忌路由预测"""
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            action, conf = current.decide(state)
            
            if action == 1:  # YES
                return current.label, path
            
            # NO → 禁忌搜索
            candidates = [c for c in current.connections if c.id not in visited]
            
            if candidates:
                current = np.random.choice(candidates)
            else:
                all_unvisited = [s for s in self.structons if s.id not in visited]
                if all_unvisited:
                    current = np.random.choice(all_unvisited)
                    path.append("JUMP")
                else:
                    break
        
        return current.label, path

    def predict_voting(self, image) -> Tuple[str, float]:
        """投票预测"""
        state = self.extractor.extract(image)
        
        votes = {}
        for s in self.structons:
            action, conf = s.decide(state)
            if action == 1:  # YES
                votes[s.label] = votes.get(s.label, 0) + conf + 1
        
        if not votes:
            # 选 YES Q 值最高的
            best_s = None
            best_q = -999
            for s in self.structons:
                q, _ = s.lrm.query(state)
                if q[1] > best_q:
                    best_q = q[1]
                    best_s = s
            return best_s.label if best_s else "?", 0.0
        
        best = max(votes, key=votes.get)
        return best, votes[best] / len(self.structons)

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.5 - Zero-Mean + Top-K")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        total = sum(s.lrm.size for s in self.structons)
        print(f"总记忆: {total}")
        
        print("\n=== 各 Structon ===")
        for s in self.structons:
            print(f"  {s.id} ['{s.label}'] mem:{s.lrm.size}")

# =============================================================================
# 6. 实验
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
    print("Structon Vision v9.5 - Zero-Mean + Top-K")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    
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
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (ep + 1) % 5 == 0:
            correct = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, _ = system.predict_voting(test_images[i])
                if pred == str(test_labels[i]): 
                    correct += 1
            print(f"  轮次 {ep+1}: {correct/check_n*100:.1f}%")
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试（禁忌路由）===")
    test_idxs = np.random.choice(len(test_images), n_test, replace=False)
    
    stats = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
    
    correct1 = 0
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, path = system.predict(test_images[idx])
        stats[true_label]['total'] += 1
        if pred == true_label:
            correct1 += 1
            stats[true_label]['correct'] += 1
    
    print(f"准确率: {correct1/n_test*100:.1f}%")
    
    print(f"\n=== 测试（投票）===")
    correct2 = 0
    for idx in test_idxs:
        pred, _ = system.predict_voting(test_images[idx])
        if pred == str(test_labels[idx]):
            correct2 += 1
    print(f"准确率: {correct2/n_test*100:.1f}%")
    
    print("\n各数字（禁忌路由）:")
    for d in range(10):
        s = stats[str(d)]
        if s['total'] > 0:
            print(f"  {d}: {s['correct']/s['total']*100:.1f}%")
    
    # 示例路径
    print("\n=== 示例路径 ===")
    for i in range(5):
        idx = test_idxs[i]
        pred, path = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        path_str = ' → '.join(path[:8])
        if len(path) > 8:
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
