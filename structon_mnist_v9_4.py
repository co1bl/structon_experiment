#!/usr/bin/env python3
"""
Structon Vision v9.4 - 二元分类 + 禁忌搜索 + 增强特征

核心设计：
1. 二元大脑：Structon 只输出 [NO, YES]，不关心路由方向
2. 禁忌路由：强制剔除已访问节点，防止乒乓球循环
3. 动态跳跃：死胡同时随机跳到未访问节点
4. 增强特征：55维，更好区分相似数字

这是 Structon 的核心哲学：
- 局部规则（每个节点只做二元判断）
- 全局涌现（系统层面的禁忌搜索产生正确路由）
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
# 2. 特征提取 (增强版 55维)
# =============================================================================
class StateExtractor:
    """
    增强特征提取器 - 更好区分相似数字
    
    特征：
    - 5x5 Grid (25维): 整体形状
    - 投影 (10维): 行/列分布  
    - 四象限 (4维): 区分 6 vs 9
    - 差异特征 (2维): 上下/左右对称性
    - 中心密度 (2维): 区分 0 vs 8
    - 切片特征 (6维): 区分 1, 7
    - 对角线 (3维): 区分斜向特征
    - 边缘 (2维): 开口方向
    """
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
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
        binary = (img > 0.3).astype(np.uint8)
        
        # 基本密度
        features.append(np.mean(binary))
        
        # 四象限 (区分 6 vs 9)
        features.append(np.mean(binary[:h//2, :w//2]))   # 左上
        features.append(np.mean(binary[:h//2, w//2:]))   # 右上
        features.append(np.mean(binary[h//2:, :w//2]))   # 左下
        features.append(np.mean(binary[h//2:, w//2:]))   # 右下
        
        # 上下左右差
        features.append(np.mean(binary[:h//2, :]) - np.mean(binary[h//2:, :]))
        features.append(np.mean(binary[:, :w//2]) - np.mean(binary[:, w//2:]))
        
        # 中心密度 (区分 0 vs 8)
        features.append(np.mean(binary[h//4:3*h//4, w//4:3*w//4]))
        features.append(np.mean(binary[h//3:2*h//3, w//3:2*w//3]))  # 核心
        
        # 水平切片 (区分 7)
        features.append(np.mean(binary[2:5, :]))
        features.append(np.mean(binary[h//2-2:h//2+2, :]))
        features.append(np.mean(binary[-5:-2, :]))
        
        # 垂直切片 (区分 1)
        features.append(np.mean(binary[:, w//2-2:w//2+2]))
        features.append(np.mean(binary[:, 2:5]))
        features.append(np.mean(binary[:, -5:-2]))
        
        # 对角线 (区分 1 vs 7)
        diag1 = np.mean([binary[i, i] for i in range(min(h, w))])
        diag2 = np.mean([binary[i, w-1-i] for i in range(min(h, w))])
        features.append(diag1)
        features.append(diag2)
        features.append(diag1 - diag2)
        
        # 边缘
        features.append(np.mean(binary[0, :]))
        features.append(np.mean(binary[-1, :]))
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6: 
            state = state / norm
        return state

# =============================================================================
# 3. LRM (二元版)
# =============================================================================
class LRM:
    """
    Local Resonant Memory - 二元版
    
    只有两个动作：NO (0) 和 YES (1)
    """
    def __init__(self, state_dim=55, capacity=200):
        self.state_dim = state_dim
        self.n_actions = 2  # 0=NO, 1=YES
        self.capacity = capacity
        
        # 随机投影
        self.projection = np.random.randn(state_dim, 16).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.5
        self.similarity_threshold = 0.93

    def _compute_key(self, state):
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6: 
            key /= norm
        return key

    def query(self, state) -> Tuple[np.ndarray, float]:
        """查询 Q 值"""
        if not self.keys: 
            return np.zeros(self.n_actions), 0.0
        
        key = self._compute_key(state)
        scores = np.array(self.keys) @ key
        
        # 加权平均
        weights = np.maximum(scores, 0) ** 3
        w_sum = np.sum(weights)
        if w_sum < 1e-6: 
            return np.zeros(self.n_actions), 0.0
        
        weights /= w_sum
        q_values = np.zeros(self.n_actions)
        for i, w in enumerate(weights):
            if w > 0.01: 
                q_values += w * self.values[i]
        return q_values, float(np.max(scores))

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
# 4. Structon (二元决策)
# =============================================================================
class Structon:
    """
    Structon - 二元大脑
    
    只回答一个问题："这是我的吗？"
    - NO (0): 不是我的
    - YES (1): 是我的
    """
    _id_counter = 0
    
    def __init__(self, label: str, capacity=200, n_connections=3):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        self.lrm = LRM(capacity=capacity)
        
    def set_connections(self, all_structons):
        """随机选择连接"""
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
    def decide(self, state) -> Tuple[int, float]:
        """
        决策：是我的吗？
        
        Returns:
            action: 0=NO, 1=YES
            confidence: 置信度
        """
        q, score = self.lrm.query(state)
        action = int(np.argmax(q))
        confidence = q[action] if q[action] != 0 else score
        return action, confidence

    def train_binary(self, state, is_me: bool):
        """
        训练二元分类
        
        只教它一件事：识别自己
        """
        target_action = 1 if is_me else 0
        
        # 强化正确答案
        self.lrm.update(state, target_action, 1.0)
        
        # 抑制错误答案
        wrong_action = 1 - target_action
        self.lrm.update(state, wrong_action, -0.5)

    def get_connections_str(self):
        return f"[{', '.join(c.label for c in self.connections)}]"

# =============================================================================
# 5. Vision System (禁忌路由)
# =============================================================================
class StructonVisionSystem:
    """
    Structon 视觉系统 v9.4
    
    核心：二元大脑 + 禁忌路由
    """
    def __init__(self, capacity=200, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        self.capacity = capacity
        self.n_connections = n_connections
        
    def build(self, labels):
        """创建所有 Structon"""
        print("\n=== 创建 Structon v9.4 ===")
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
        """
        训练一个 epoch
        
        每个样本：
        1. 正样本：教正确 Structon 说 YES
        2. 负样本：教其他 Structon 说 NO
        """
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 1. 训练正样本 (YES)
            correct_s = self.label_to_structon[label]
            correct_s.train_binary(state, is_me=True)
            
            # 2. 训练负样本 (NO)
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_binary(state, is_me=False)

    def predict(self, image, max_hops=20) -> Tuple[str, List[str]]:
        """
        禁忌路由预测
        
        1. 随机选起点
        2. 问当前节点："是你的吗？"
        3. YES → 返回
        4. NO → 从未访问的邻居中选下一个
        5. 死胡同 → 随机跳到任意未访问节点
        """
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            # 1. 问 Structon
            action, conf = current.decide(state)
            
            # 2. YES → 返回
            if action == 1:
                return current.label, path
            
            # 3. NO → 禁忌搜索找下一个
            candidates = [c for c in current.connections if c.id not in visited]
            
            if candidates:
                # 从未访问的邻居中随机选
                current = np.random.choice(candidates)
            else:
                # 死胡同！随机跳到任意未访问节点
                all_unvisited = [s for s in self.structons if s.id not in visited]
                if all_unvisited:
                    current = np.random.choice(all_unvisited)
                    path.append("JUMP")
                else:
                    # 全部访问完了
                    break
        
        return current.label, path

    def predict_voting(self, image) -> Tuple[str, float]:
        """
        投票预测：每个 Structon 独立判断，投票
        """
        state = self.extractor.extract(image)
        
        votes = {}
        for s in self.structons:
            action, conf = s.decide(state)
            if action == 1:  # YES
                votes[s.label] = votes.get(s.label, 0) + conf + 1
        
        if not votes:
            # 没人说 YES，选置信度最高的 NO
            best_s = None
            best_conf = -999
            for s in self.structons:
                q, _ = s.lrm.query(state)
                # YES 的 Q 值越高越好（即使最终选了 NO）
                if q[1] > best_conf:
                    best_conf = q[1]
                    best_s = s
            return best_s.label if best_s else "?", 0.0
        
        best = max(votes, key=votes.get)
        return best, votes[best] / len(self.structons)

    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.4 - Binary Brain + Taboo Routing")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        total = sum(s.lrm.size for s in self.structons)
        print(f"总记忆: {total}")
        
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
    epochs: int = 30,
    n_connections: int = 3,
    n_negatives: int = 3
):
    print("=" * 70)
    print("Structon Vision v9.4 - Binary Brain + Taboo Routing")
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
            # 快速测试
            correct = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, _ = system.predict(test_images[i])
                if pred == str(test_labels[i]): 
                    correct += 1
            print(f"  轮次 {ep+1}: {correct/check_n*100:.1f}%")
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试 - 随机入口 + 禁忌路由
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
        path_str = ' → '.join(path[:10])
        if len(path) > 10:
            path_str += f" ... ({len(path)} hops)"
        print(f"  真实={true}, 预测={pred} {status}, 路径: {path_str}")
    
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
        epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives
    )
