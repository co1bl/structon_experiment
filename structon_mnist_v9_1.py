#!/usr/bin/env python3
"""
Structon Vision v9.1 - 局部奖励

核心设计：
- 每个 Structon 只学习局部决策
- "是我的" vs "去别人"
- 不关心路径长度
- 不关心去哪个 Structon

局部奖励规则：
- 标签是我 + 选"是我的" → +1
- 标签是我 + 选"去别人" → -1  
- 标签不是我 + 选"去别人" → +1
- 标签不是我 + 选"是我的" → -1
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Set
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
        n_actions: int = 4,
        capacity: int = 200,
        key_dim: int = 16,
        similarity_threshold: float = 0.92,
        learning_rate: float = 0.5
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
# 4. Structon - 局部奖励
# =============================================================================

class Structon:
    """
    Structon = LRM + 局部奖励
    
    只学习：是我的 vs 去别人
    不关心去哪个，不关心路径长度
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        state_dim: int = 45,
        capacity: int = 200,
        key_dim: int = 16,
        epsilon: float = 0.15,
        n_connections: int = 3
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.label = label
        self.state_dim = state_dim
        self.capacity = capacity
        self.epsilon = epsilon
        self.n_connections = n_connections
        
        # 动作：[是我的, 去A, 去B, 去C]
        self.lrm = LRM(
            state_dim=state_dim,
            n_actions=1 + n_connections,
            capacity=capacity,
            key_dim=key_dim
        )
        
        self.connections: List[Optional['Structon']] = [None] * n_connections
    
    def set_connections(self, others: List['Structon']):
        """随机选连接"""
        available = [s for s in others if s.id != self.id]
        if len(available) >= self.n_connections:
            chosen = np.random.choice(available, self.n_connections, replace=False)
            self.connections = list(chosen)
        else:
            self.connections = available + [None] * (self.n_connections - len(available))
    
    def select_action(self, state: np.ndarray, explore: bool = False) -> int:
        """选择动作"""
        q_values, _ = self.lrm.query(state)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(0, len(q_values))
        return int(np.argmax(q_values))
    
    def learn_local(self, state: np.ndarray, true_label: str, action_taken: int):
        """
        局部学习 - 只基于"是我的"还是"不是我的"
        
        规则：
        - 标签是我 + 选"是我的" → +1
        - 标签是我 + 选"去别人" → -1
        - 标签不是我 + 选"去别人" → +1
        - 标签不是我 + 选"是我的" → -1
        """
        is_mine = (true_label == self.label)
        chose_mine = (action_taken == 0)
        
        if is_mine:
            if chose_mine:
                reward = 1.0   # 正确：是我的，选了是我的
            else:
                reward = -1.0  # 错误：是我的，但选了去别人
        else:
            if chose_mine:
                reward = -1.0  # 错误：不是我的，但选了是我的
            else:
                reward = 1.0   # 正确：不是我的，选了去别人
        
        self.lrm.update(state, action_taken, reward)
    
    def execute(self, state: np.ndarray, true_label: str = None, 
                explore: bool = False, learn: bool = False,
                visited: Set[str] = None, max_hops: int = 20) -> str:
        """
        执行
        
        如果 learn=True，每个经过的 Structon 都会学习
        """
        if visited is None:
            visited = set()
        
        if self.id in visited or len(visited) >= max_hops:
            return self.label
        
        visited.add(self.id)
        
        # 选择动作
        action = self.select_action(state, explore)
        
        # 局部学习
        if learn and true_label is not None:
            self.learn_local(state, true_label, action)
        
        if action == 0:  # 是我的
            return self.label
        else:
            conn_idx = action - 1
            if conn_idx < len(self.connections) and self.connections[conn_idx] is not None:
                return self.connections[conn_idx].execute(
                    state, true_label, explore, learn, visited, max_hops
                )
            else:
                return self.label
    
    def get_connections_str(self) -> str:
        conns = [c.label if c else "?" for c in self.connections]
        return f"[{', '.join(conns)}]"


# =============================================================================
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统 v9.1 - 局部奖励
    """
    
    def __init__(
        self,
        state_dim: int = 45,
        capacity: int = 200,
        key_dim: int = 16,
        n_connections: int = 3
    ):
        self.extractor = StateExtractor()
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        self.n_connections = n_connections
        
        self.structons: Dict[str, Structon] = {}
        self.structon_list: List[Structon] = []
    
    def add_class(self, label: str):
        new_structon = Structon(
            label=label,
            state_dim=self.state_dim,
            capacity=self.capacity,
            n_connections=self.n_connections
        )
        
        self.structons[label] = new_structon
        self.structon_list.append(new_structon)
        
        print(f"  + {new_structon.id} label='{label}'")
    
    def setup_connections(self):
        print("\n设置稀疏连接...")
        for structon in self.structon_list:
            structon.set_connections(self.structon_list)
            print(f"  {structon.id} ({structon.label}) → {structon.get_connections_str()}")
    
    def predict(self, image: np.ndarray) -> str:
        """预测（随机起点）"""
        if not self.structon_list:
            return "?"
        
        state = self.extractor.extract(image)
        entry = np.random.choice(self.structon_list)
        return entry.execute(state, explore=False, learn=False)
    
    def predict_from_all(self, image: np.ndarray) -> str:
        """从所有起点预测，投票"""
        if not self.structon_list:
            return "?"
        
        state = self.extractor.extract(image)
        
        votes = {}
        for entry in self.structon_list:
            result = entry.execute(state, explore=False, learn=False)
            votes[result] = votes.get(result, 0) + 1
        
        return max(votes, key=votes.get)
    
    def train(self, image: np.ndarray, label: str) -> bool:
        """
        训练 - 随机起点，局部学习
        
        每个经过的 Structon 都学习自己的局部决策
        """
        if not self.structon_list or label not in self.structons:
            return False
        
        state = self.extractor.extract(image)
        
        # 随机起点
        entry = np.random.choice(self.structon_list)
        
        # 执行 + 学习（每个经过的节点都学习）
        result = entry.execute(state, true_label=label, explore=True, learn=True)
        
        return result == label
    
    def train_all_entries(self, image: np.ndarray, label: str) -> int:
        """
        从所有起点训练（更充分的学习）
        """
        if not self.structon_list or label not in self.structons:
            return 0
        
        state = self.extractor.extract(image)
        correct = 0
        
        for entry in self.structon_list:
            result = entry.execute(state, true_label=label, explore=True, learn=True)
            if result == label:
                correct += 1
        
        return correct
    
    def print_stats(self):
        print("\n" + "=" * 60)
        print("Structon Vision v9.1 - 局部奖励")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        total_mem = sum(s.lrm.size for s in self.structon_list)
        print(f"总记忆: {total_mem}")
        
        print("\n=== 各 Structon ===")
        for s in self.structon_list:
            print(f"  {s.id} ['{s.label}'] mem:{s.lrm.size}/{s.capacity} → {s.get_connections_str()}")


# =============================================================================
# 6. 实验
# =============================================================================

def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 200,
    max_epochs: int = 50,
    n_connections: int = 3,
    train_all: bool = False
):
    print("=" * 70)
    print("Structon Vision v9.1 - 局部奖励")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}, 连接数={n_connections}")
    print(f"训练模式: {'全起点' if train_all else '随机起点'}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        state_dim=45,
        capacity=capacity,
        n_connections=n_connections
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
    
    system.setup_connections()
    
    print(f"\n=== 训练 (局部奖励) ===")
    t0 = time.time()
    
    all_samples = []
    for digit in range(10):
        all_samples.extend(class_samples[digit])
    
    for epoch in range(max_epochs):
        np.random.shuffle(all_samples)
        
        correct = 0
        for img, label in all_samples:
            if train_all:
                c = system.train_all_entries(img, label)
                correct += (c > 5)  # 多数起点正确
            else:
                if system.train(img, label):
                    correct += 1
        
        acc = correct / len(all_samples) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"  轮次 {epoch+1}: {acc:.1f}%")
    
    print(f"\n训练: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试（随机入口）===")
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    correct1 = 0
    for idx in test_indices:
        if system.predict(test_images[idx]) == str(test_labels[idx]):
            correct1 += 1
    print(f"准确率: {correct1/n_test*100:.1f}%")
    
    print(f"\n=== 测试（全起点投票）===")
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    
    for idx in test_indices:
        predicted = system.predict_from_all(test_images[idx])
        true_label = str(test_labels[idx])
        results[true_label]['total'] += 1
        if predicted == true_label:
            results[true_label]['correct'] += 1
    
    total_correct = sum(r['correct'] for r in results.values())
    print(f"准确率: {total_correct/n_test*100:.1f}%")
    
    print("\n各数字:")
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
    parser.add_argument('--max-epochs', type=int, default=50)
    parser.add_argument('--connections', type=int, default=3)
    parser.add_argument('--train-all', action='store_true', help='从所有起点训练')
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        max_epochs=args.max_epochs,
        n_connections=args.connections,
        train_all=args.train_all
    )
