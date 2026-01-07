#!/usr/bin/env python3
"""
Structon Vision v9.6 - Contrast Binary + Structure

最佳特征组合：
1. Contrast Binary: 7x7 grid，暗=-1，亮=+1
2. Structure: 相邻差异、对角线、四象限

基于测试结果：
- Contrast Binary: YES=94.1%, NO=71.0%, 平衡=82.6%
- Zero-Mean + Structure: YES=97.6%, NO=68.5%, 平衡=83.0%

组合应该能达到更高的平衡准确率！
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
# 2. 特征提取 - Contrast Binary + Structure
# =============================================================================
class StateExtractor:
    """
    最佳特征组合：Contrast Binary + Structure
    
    特征：
    1. 7x7 Contrast Binary (49维): 暗=-1, 亮=+1
    2. 水平差异 (42维): 相邻列的差
    3. 垂直差异 (42维): 相邻行的差
    4. 对角线 (2维)
    5. 四象限差异 (4维)
    6. 边缘特征 (4维)
    
    总计: ~143维
    """
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        # 1. 计算 7x7 grid 值
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                grid[i, j] = np.mean(block)
        
        # 2. Contrast Binary: 暗=-1, 亮=+1
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i, j] < self.threshold:
                    features.append(-1.0)
                else:
                    features.append(1.0)
        
        # 3. 水平差异（相邻列）
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                diff = grid[i, j] - grid[i, j+1]
                features.append(diff * 2)  # 放大差异
        
        # 4. 垂直差异（相邻行）
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                diff = grid[i, j] - grid[i+1, j]
                features.append(diff * 2)
        
        # 5. 对角线
        diag1 = np.mean([grid[i, i] for i in range(self.grid_size)])
        diag2 = np.mean([grid[i, self.grid_size-1-i] for i in range(self.grid_size)])
        features.append((diag1 - 0.5) * 2)
        features.append((diag2 - 0.5) * 2)
        features.append((diag1 - diag2) * 2)
        
        # 6. 四象限
        mid = self.grid_size // 2
        q1 = np.mean(grid[:mid, :mid])      # 左上
        q2 = np.mean(grid[:mid, mid:])      # 右上
        q3 = np.mean(grid[mid:, :mid])      # 左下
        q4 = np.mean(grid[mid:, mid:])      # 右下
        
        features.append((q1 - q4) * 2)  # 左上-右下
        features.append((q2 - q3) * 2)  # 右上-左下
        features.append((q1 + q4 - q2 - q3) * 2)  # 对角差
        features.append(((q1 + q2) - (q3 + q4)) * 2)  # 上-下
        
        # 7. 边缘
        top = np.mean(grid[0, :])
        bottom = np.mean(grid[-1, :])
        left = np.mean(grid[:, 0])
        right = np.mean(grid[:, -1])
        
        features.append((top - bottom) * 2)
        features.append((left - right) * 2)
        features.append((top - 0.5) * 2)
        features.append((bottom - 0.5) * 2)
        
        state = np.array(features, dtype=np.float32)
        
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        
        return state

# =============================================================================
# 3. LRM - Top-K
# =============================================================================
class LRM:
    """Local Resonant Memory with Top-K query"""
    
    def __init__(self, capacity=300, top_k=5):
        self.n_actions = 2  # NO, YES
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88

    def query(self, state) -> Tuple[np.ndarray, float]:
        """Top-K 查询"""
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.array([np.dot(k, state) for k in self.keys])
        
        # Top-K
        k = min(self.top_k, len(scores))
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
        """更新记忆"""
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
# 4. Structon
# =============================================================================
class Structon:
    """Binary Brain Structon"""
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
            self.lrm.update(state, 0, -0.8)  # not NO
        else:
            self.lrm.update(state, 0, 0.8)   # NO
            self.lrm.update(state, 1, -0.5)  # not YES

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"

# =============================================================================
# 5. Vision System
# =============================================================================
class StructonVisionSystem:
    """Structon Vision System v9.6"""
    
    def __init__(self, capacity=300, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        print("\n=== 创建 Structon v9.6 ===")
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

    def predict_voting(self, image) -> Tuple[str, Dict[str, float]]:
        """投票预测"""
        state = self.extractor.extract(image)
        
        votes = {}
        details = {}
        
        for s in self.structons:
            q, score = s.lrm.query(state)
            action = int(np.argmax(q))
            
            # 记录 YES 的 Q 值
            details[s.label] = q[1]
            
            if action == 1:  # YES
                votes[s.label] = votes.get(s.label, 0) + q[1] + 1
        
        if not votes:
            # 没人说 YES，选 YES Q 值最高的
            best = max(details, key=details.get)
            return best, details
        
        best = max(votes, key=votes.get)
        return best, details

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.6 - Contrast Binary + Structure")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        total = sum(s.lrm.size for s in self.structons)
        print(f"总记忆: {total}")
        
        print("\n=== 各 Structon ===")
        for s in self.structons:
            print(f"  {s.id} ['{s.label}'] mem:{s.lrm.size}")

# =============================================================================
# 6. 单独测试二元分类能力
# =============================================================================
def test_binary_classification(n_train=100, n_test=200):
    """测试每个 Structon 的二元分类能力"""
    print("\n" + "=" * 60)
    print("二元分类测试")
    print("=" * 60)
    
    train_images, train_labels, test_images, test_labels = load_mnist()
    extractor = StateExtractor()
    
    results = []
    
    for target_digit in range(10):
        lrm = LRM(capacity=300)
        
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
    print("Structon Vision v9.6 - Contrast Binary + Structure")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    
    # 先测试二元分类能力
    test_binary_classification()
    
    print("\n" + "=" * 70)
    print("系统测试")
    print("=" * 70)
    
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
    
    # 测试 - 禁忌路由
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
    print(f"{'数字':<6} {'禁忌路由':<12} {'投票':<12}")
    print("-" * 30)
    for d in range(10):
        s1 = stats[str(d)]
        s2 = stats2[str(d)]
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
