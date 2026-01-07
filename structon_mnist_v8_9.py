#!/usr/bin/env python3
"""
Structon Vision v8.9 - Q-learning with LRM

核心设计：
- 每个 Structon = LRM + Q-learning
- 动作选择基于 Q 值
- 奖励沿路径传播（信用分配）

当预测正确：路径上所有 Structon 都获得正奖励
当预测错误：路径上所有 Structon 都获得负奖励
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
import time
import gzip
import os
import urllib.request


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
# 2. 特征提取
# =============================================================================

class StateExtractor:
    """增强特征提取器 (45维)"""
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        features = []
        h, w = img.shape
        
        # 5x5 下采样 (25维)
        bh, bw = h // 5, w // 5
        for i in range(5):
            for j in range(5):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                features.append(np.mean(block))
        
        # 水平投影 (5维)
        for i in range(5):
            features.append(np.mean(img[i*(h//5):(i+1)*(h//5), :]))
        
        # 垂直投影 (5维)
        for j in range(5):
            features.append(np.mean(img[:, j*(w//5):(j+1)*(w//5)]))
        
        # 结构特征 (10维)
        binary = (img > 0.3).astype(np.uint8)
        features.append(np.mean(binary))
        features.append(np.mean(binary[:h//2, :]) - np.mean(binary[h//2:, :]))
        features.append(np.mean(binary[:, :w//2]) - np.mean(binary[:, w//2:]))
        features.append(np.mean(binary[h//4:3*h//4, w//4:3*w//4]))
        features.append((np.mean(binary[0,:]) + np.mean(binary[-1,:]) + 
                        np.mean(binary[:,0]) + np.mean(binary[:,-1])) / 4)
        features.append(np.mean([binary[i,i] for i in range(min(h,w))]))
        features.append(np.mean([binary[i,w-1-i] for i in range(min(h,w))]))
        features.append(np.mean(binary[2:5, :]))
        features.append(np.mean(binary[h//2-2:h//2+2, :]))
        features.append(np.mean(binary[-5:-2, :]))
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state


# =============================================================================
# 3. LRM (Local Resonant Memory)
# =============================================================================

class LRM:
    """Q-learning 风格的 LRM"""
    
    def __init__(
        self,
        state_dim: int = 45,
        n_actions: int = 1,
        capacity: int = 200,
        key_dim: int = 16,
        similarity_threshold: float = 0.95,
        learning_rate: float = 0.3
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.key_dim = key_dim
        self.similarity_threshold = similarity_threshold
        self.learning_rate = learning_rate
        
        self.projection = np.random.randn(state_dim, key_dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []  # Q-values
        self.access_counts: List[int] = []
    
    def set_n_actions(self, n_actions: int):
        if n_actions <= self.n_actions:
            return
        for i in range(len(self.values)):
            old_q = self.values[i]
            new_q = np.zeros(n_actions, dtype=np.float32)
            new_q[:len(old_q)] = old_q
            self.values[i] = new_q
        self.n_actions = n_actions
    
    def _compute_key(self, state: np.ndarray) -> np.ndarray:
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6:
            key = key / norm
        return key
    
    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """查询 Q 值"""
        key = self._compute_key(state)
        
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        key_matrix = np.array(self.keys)
        scores = key_matrix @ key
        
        weights = np.maximum(scores, 0) ** 2
        weight_sum = np.sum(weights)
        
        if weight_sum < 1e-6:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        weights = weights / weight_sum
        q_values = np.zeros(self.n_actions, dtype=np.float32)
        for i, w in enumerate(weights):
            if w > 0.01:
                q_values += w * self.values[i]
        
        return q_values, float(np.max(scores))
    
    def update(self, state: np.ndarray, action: int, reward: float):
        """用奖励更新 Q 值"""
        if action >= self.n_actions:
            return
        
        key = self._compute_key(state)
        
        # 找相似记忆
        if len(self.keys) > 0:
            key_matrix = np.array(self.keys)
            scores = key_matrix @ key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score > self.similarity_threshold:
                # 更新现有记忆的 Q 值
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (reward - old_q)
                self.access_counts[best_idx] += 1
                return
        
        # 创建新记忆
        new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = reward
        
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
        
        self.keys.append(key.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
    
    @property
    def size(self) -> int:
        return len(self.keys)


# =============================================================================
# 4. Structon
# =============================================================================

class Structon:
    """
    Structon = LRM + Q-learning
    
    动作：[是我的, 去S0, 去S1, ...]
    学习：基于奖励
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        state_dim: int = 45,
        capacity: int = 200,
        key_dim: int = 16,
        epsilon: float = 0.1
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.label = label
        self.state_dim = state_dim
        self.capacity = capacity
        self.epsilon = epsilon
        
        self.lrm = LRM(
            state_dim=state_dim,
            n_actions=1,
            capacity=capacity,
            key_dim=key_dim
        )
        
        self.connections: Dict[int, 'Structon'] = {}
    
    def add_connection(self, other: 'Structon'):
        new_action_idx = len(self.connections) + 1
        self.connections[new_action_idx] = other
        self.lrm.set_n_actions(new_action_idx + 1)
    
    def select_action(self, state: np.ndarray, explore: bool = False) -> int:
        """选择动作"""
        q_values, _ = self.lrm.query(state)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, len(q_values))
        return int(np.argmax(q_values))
    
    def execute(self, state: np.ndarray, explore: bool = False,
                visited: set = None, path: list = None, max_hops: int = 10) -> Tuple[str, List]:
        """
        执行并记录路径
        
        Returns:
            result: 预测结果
            path: [(structon, action), ...] 路径
        """
        if visited is None:
            visited = set()
        if path is None:
            path = []
        
        if self.id in visited or len(visited) >= max_hops:
            return self.label, path
        
        visited.add(self.id)
        
        # 选择动作
        action = self.select_action(state, explore)
        
        # 记录到路径
        path.append((self, action))
        
        if action == 0:  # 是我的
            return self.label, path
        elif action in self.connections:
            return self.connections[action].execute(state, explore, visited, path, max_hops)
        else:
            return self.label, path
    
    def receive_reward(self, state: np.ndarray, action: int, reward: float):
        """接收奖励，更新 LRM"""
        self.lrm.update(state, action, reward)


# =============================================================================
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统 v8.9 - Q-learning with Reward
    """
    
    def __init__(
        self,
        state_dim: int = 45,
        capacity: int = 200,
        key_dim: int = 16
    ):
        self.extractor = StateExtractor()
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        self.structons: Dict[str, Structon] = {}
        
        # 入口路由 LRM
        self.entry_lrm = LRM(
            state_dim=state_dim,
            n_actions=1,
            capacity=capacity * 2,
            key_dim=key_dim
        )
        self.label_to_action: Dict[str, int] = {}
    
    def add_class(self, label: str):
        new_structon = Structon(
            label=label,
            state_dim=self.state_dim,
            capacity=self.capacity
        )
        
        for existing in self.structons.values():
            new_structon.add_connection(existing)
            existing.add_connection(new_structon)
        
        self.structons[label] = new_structon
        
        # 入口动作
        action_idx = len(self.label_to_action)
        self.label_to_action[label] = action_idx
        self.entry_lrm.set_n_actions(len(self.label_to_action))
        
        print(f"  + {new_structon.id} label='{label}', actions: {new_structon.lrm.n_actions}")
    
    def _select_entry(self, state: np.ndarray, explore: bool = False) -> Tuple[Structon, int]:
        """选择入口 Structon"""
        q_values, _ = self.entry_lrm.query(state)
        
        if explore and np.random.random() < 0.1:
            action = np.random.randint(0, len(q_values))
        else:
            action = int(np.argmax(q_values))
        
        for label, idx in self.label_to_action.items():
            if idx == action:
                return self.structons[label], action
        
        # 随机
        label = np.random.choice(list(self.structons.keys()))
        return self.structons[label], self.label_to_action[label]
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测（不探索）"""
        if not self.structons:
            return "?", 0.0
        
        state = self.extractor.extract(image)
        entry, _ = self._select_entry(state, explore=False)
        result, path = entry.execute(state, explore=False)
        
        return result, 1.0
    
    def train(self, image: np.ndarray, label: str):
        """
        训练 - 基于奖励的 Q-learning
        
        1. 选择入口（带探索）
        2. 执行（带探索），记录路径
        3. 计算奖励
        4. 沿路径传播奖励给每个 Structon
        """
        if not self.structons or label not in self.structons:
            return False
        
        state = self.extractor.extract(image)
        
        # 1. 选择入口
        entry, entry_action = self._select_entry(state, explore=True)
        
        # 2. 执行，记录路径
        result, path = entry.execute(state, explore=True)
        
        # 3. 计算奖励
        correct = (result == label)
        reward = 1.0 if correct else -0.5
        
        # 4. 入口 LRM 获得奖励
        correct_entry_action = self.label_to_action[label]
        if correct:
            self.entry_lrm.update(state, entry_action, reward)
        else:
            # 错了，强化正确的入口
            self.entry_lrm.update(state, correct_entry_action, 0.5)
            self.entry_lrm.update(state, entry_action, -0.3)
        
        # 5. 路径上每个 Structon 获得奖励
        for structon, action in path:
            structon.receive_reward(state, action, reward)
        
        # 6. 确保正确的 Structon 也学习
        correct_structon = self.structons[label]
        if correct_structon not in [s for s, _ in path]:
            # 强化它的 "是我的" 动作
            correct_structon.receive_reward(state, 0, 0.5)
        
        return correct
    
    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v8.9 - Q-learning with Reward")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"入口 LRM 记忆: {self.entry_lrm.size}")
        total_mem = sum(s.lrm.size for s in self.structons.values())
        print(f"Structon 总记忆: {total_mem}")
        
        print("\n=== 各 Structon ===")
        for label in sorted(self.structons.keys()):
            s = self.structons[label]
            print(f"  {s.id} ['{s.label}'] mem:{s.lrm.size}/{s.capacity}")


# =============================================================================
# 6. 实验
# =============================================================================

def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 200,
    max_epochs: int = 30,
    target_acc: float = 0.90
):
    print("=" * 70)
    print("Structon Vision v8.9 - Q-learning with Reward")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        state_dim=45,
        capacity=capacity
    )
    
    # 准备样本
    class_samples = {}
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        np.random.shuffle(indices)
        class_samples[digit] = [(train_images[i], str(digit)) for i in indices]
    
    print(f"\n=== 创建 Structon ===")
    for digit in range(10):
        system.add_class(str(digit))
    
    print(f"\n=== 训练 (Q-learning) ===")
    t0 = time.time()
    
    all_samples = []
    for digit in range(10):
        all_samples.extend(class_samples[digit])
    
    for epoch in range(max_epochs):
        np.random.shuffle(all_samples)
        
        correct = 0
        for img, label in all_samples:
            if system.train(img, label):
                correct += 1
        
        acc = correct / len(all_samples) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"  轮次 {epoch+1}: {acc:.1f}%")
        
        if acc >= target_acc * 100:
            print(f"  ✓ 达标! {acc:.1f}%")
            break
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试 ===")
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    for idx in test_indices:
        predicted, _ = system.predict(test_images[idx])
        true_label = str(test_labels[idx])
        results[true_label]['total'] += 1
        if predicted == true_label:
            results[true_label]['correct'] += 1
    
    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    
    print(f"\n总准确率: {total_correct/total_samples*100:.1f}%")
    for d in range(10):
        r = results[str(d)]
        if r['total'] > 0:
            print(f"  {d}: {r['correct']/r['total']*100:.1f}%")
    
    return system


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=200)
    parser.add_argument('--max-epochs', type=int, default=30)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        max_epochs=args.max_epochs
    )
