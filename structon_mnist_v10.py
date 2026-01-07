#!/usr/bin/env python3
"""
Structon Vision v10 - 连接即动作 (Connection as Action)

核心改变：
- 动作空间从 {YES, NO} 变为 {STAY, GOTO_conn0, GOTO_conn1, ...}
- 引入 TD-learning：γ * max(next_Q) 估计未来价值
- 每张图片作为一个 episode，路由过程中学习

设计理念：
- 每个 Structon 学习"看到这个特征，我该留下还是推荐给谁"
- 通过 γ 折扣，正确的路由决策会反向传播
- 可能涌现出专业化分工：某些节点成为"路由器"

作者: Boe
日期: 2025
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import argparse


# =============================================================================
# 1. MNIST 加载 / 模拟数据
# =============================================================================
def load_mnist():
    """加载 MNIST 数据集"""
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


def generate_synthetic_digit(digit: int, noise: float = 0.15) -> np.ndarray:
    """
    生成合成数字图像
    
    每个数字有独特的模式，用于测试算法
    """
    img = np.zeros((28, 28), dtype=np.float32)
    
    # 每个数字的简化模式
    patterns = {
        0: [(10, 14, 8, 20), (10, 14, 18, 8)],  # 椭圆
        1: [(14, 14, 8, 20)],  # 竖线
        2: [(8, 18, 8, 8), (14, 14, 8, 8), (20, 10, 8, 8), (10, 14, 20, 6)],
        3: [(10, 14, 8, 8), (14, 16, 8, 8), (10, 14, 20, 8)],
        4: [(6, 10, 14, 6), (14, 14, 8, 20)],
        5: [(8, 18, 8, 8), (10, 14, 8, 8), (20, 10, 8, 8)],
        6: [(10, 10, 8, 8), (8, 14, 14, 8), (14, 10, 8, 8)],
        7: [(8, 18, 8, 8), (18, 14, 8, 16)],
        8: [(10, 14, 8, 8), (14, 14, 10, 8), (10, 14, 20, 8)],
        9: [(10, 14, 8, 8), (14, 18, 8, 8), (18, 14, 8, 16)],
    }
    
    for (y, x, h, w) in patterns.get(digit, []):
        y1, y2 = max(0, y-h//2), min(28, y+h//2)
        x1, x2 = max(0, x-w//2), min(28, x+w//2)
        img[y1:y2, x1:x2] = 0.8 + np.random.uniform(-0.2, 0.2)
    
    # 添加噪声
    img += np.random.randn(28, 28) * noise
    img = np.clip(img, 0, 1)
    
    # 随机偏移
    shift_y = np.random.randint(-2, 3)
    shift_x = np.random.randint(-2, 3)
    img = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
    
    return (img * 255).astype(np.uint8)


def load_synthetic_data(n_train_per_class: int = 500, n_test_per_class: int = 100):
    """生成合成数据集"""
    print("Generating synthetic data...")
    
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    
    for digit in range(10):
        for _ in range(n_train_per_class):
            train_images.append(generate_synthetic_digit(digit))
            train_labels.append(digit)
        for _ in range(n_test_per_class):
            test_images.append(generate_synthetic_digit(digit, noise=0.2))
            test_labels.append(digit)
    
    return (
        np.array(train_images),
        np.array(train_labels),
        np.array(test_images),
        np.array(test_labels)
    )


def try_load_mnist_or_synthetic():
    """尝试加载 MNIST，失败则用合成数据"""
    try:
        return load_mnist()
    except Exception as e:
        print(f"Cannot load MNIST: {e}")
        print("Using synthetic data instead...")
        return load_synthetic_data()


# =============================================================================
# 2. 特征提取 - Contrast Binary + Structure
# =============================================================================
class StateExtractor:
    """
    特征提取器：Contrast Binary + Structure
    
    特征组成：
    1. 7x7 Contrast Binary (49维): 暗=-1, 亮=+1
    2. 水平差异 (42维): 相邻列的差
    3. 垂直差异 (42维): 相邻行的差
    4. 对角线 (3维)
    5. 四象限差异 (4维)
    6. 边缘特征 (4维)
    
    总计: ~144维
    """
    def __init__(self, grid_size: int = 7, threshold: float = 0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        self._feature_dim = None
        
    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            # 计算特征维度
            dummy = np.zeros((28, 28), dtype=np.uint8)
            self._feature_dim = len(self.extract(dummy))
        return self._feature_dim
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取特征向量"""
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
                features.append(diff * 2)
        
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
        
        features.append((q1 - q4) * 2)
        features.append((q2 - q3) * 2)
        features.append((q1 + q4 - q2 - q3) * 2)
        features.append(((q1 + q2) - (q3 + q4)) * 2)
        
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
# 3. Local Resonant Memory - TD 版本
# =============================================================================
class LocalResonantMemory:
    """
    局部共振记忆 - 支持 TD-learning
    
    改进：
    - 支持任意数量动作
    - update_td() 方法接受预计算的 target（含 γ * max_next_Q）
    - 新记忆继承相似记忆的 Q 值
    """
    
    def __init__(
        self,
        n_actions: int,
        capacity: int = 300,
        top_k: int = 5,
        learning_rate: float = 0.3,
        similarity_threshold: float = 0.88
    ):
        self.n_actions = n_actions
        self.capacity = capacity
        self.top_k = top_k
        self.learning_rate = learning_rate
        self.similarity_threshold = similarity_threshold
        
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.access_counts: List[int] = []
        
        # 统计
        self.query_count = 0
        self.update_count = 0
        self.new_memory_count = 0

    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Top-K 查询
        
        Returns:
            q_values: 各动作的 Q 值
            confidence: 最高相似度分数
        """
        self.query_count += 1
        
        if not self.keys:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        # 计算相似度
        scores = np.array([np.dot(k, state) for k in self.keys])
        
        # Top-K
        k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        # 只用正相似度
        valid_mask = top_scores > 0.1
        if not np.any(valid_mask):
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        # 平方加权
        weights = top_scores ** 2
        weights /= np.sum(weights) + 1e-8
        
        # 加权求和
        q_values = np.zeros(self.n_actions, dtype=np.float32)
        for idx, w in zip(top_indices, weights):
            q_values += w * self.values[idx]
        
        return q_values, float(np.max(top_scores))

    def update_td(self, state: np.ndarray, action: int, target: float):
        """
        TD 风格更新
        
        Args:
            state: 状态向量（已归一化）
            action: 执行的动作
            target: TD 目标值（已包含 reward + γ * max_next_Q）
        """
        if self.keys:
            scores = np.array([np.dot(k, state) for k in self.keys])
            best_idx = int(np.argmax(scores))
            
            if scores[best_idx] > self.similarity_threshold:
                # 更新现有记忆
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (target - old_q)
                self.access_counts[best_idx] += 1
                self.update_count += 1
                return
        
        # 创建新记忆
        if self.keys:
            # 继承相似记忆的 Q 值
            new_q, _ = self.query(state)
            new_q = new_q.copy()
        else:
            new_q = np.zeros(self.n_actions, dtype=np.float32)
        
        new_q[action] = target
        
        # 容量管理
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
        
        self.keys.append(state.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        self.new_memory_count += 1
    
    def update_simple(self, state: np.ndarray, action: int, reward: float):
        """简单更新（无 TD，用于对比）"""
        self.update_td(state, action, reward)
        
    @property
    def size(self) -> int:
        return len(self.keys)
    
    def get_stats(self) -> Dict:
        return {
            'size': self.size,
            'capacity': self.capacity,
            'queries': self.query_count,
            'updates': self.update_count,
            'new_memories': self.new_memory_count,
        }


# =============================================================================
# 4. Structon - 连接即动作
# =============================================================================
class Structon:
    """
    Structon - 连接即动作版本
    
    动作空间: [STAY, GOTO_conn0, GOTO_conn1, GOTO_conn2, ...]
    - STAY (action=0): 停留在当前节点，声明"这是我的类别"
    - GOTO_i (action=i+1): 跳转到第 i 个连接
    """
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        n_connections: int = 3,
        capacity: int = 300
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label
        self.n_connections = n_connections
        
        # 动作空间: STAY + 每个连接
        self.n_actions = 1 + n_connections
        
        # 连接（训练时设置）
        self.connections: List['Structon'] = []
        
        # 共振记忆
        self.lrm = LocalResonantMemory(
            n_actions=self.n_actions,
            capacity=capacity
        )
        
    def set_connections(self, all_structons: List['Structon']):
        """设置稀疏连接"""
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(
                others, self.n_connections, replace=False
            ))
        else:
            self.connections = others
            
    def decide(self, state: np.ndarray) -> Tuple[int, float]:
        """
        决策：选择动作
        
        Returns:
            action: 0=STAY, 1+=GOTO_connection
            confidence: Q 值或相似度
        """
        q, score = self.lrm.query(state)
        action = int(np.argmax(q))
        confidence = q[action] if q[action] != 0 else score
        return action, confidence
    
    def get_next(self, action: int) -> Optional['Structon']:
        """根据动作获取下一个 Structon"""
        if action == 0:  # STAY
            return None
        conn_idx = action - 1
        if conn_idx < len(self.connections):
            return self.connections[conn_idx]
        return None
    
    def get_connections_str(self) -> str:
        return f"[{', '.join(c.label for c in self.connections)}]"
    
    def get_stats(self) -> Dict:
        return {
            'id': self.id,
            'label': self.label,
            'connections': [c.label for c in self.connections],
            'memory': self.lrm.get_stats(),
        }


# =============================================================================
# 5. Structon Vision System v10
# =============================================================================
class StructonVisionSystem:
    """
    Structon 视觉系统 v10 - 连接即动作
    
    训练方式：
    - 每张图片作为一个 episode
    - 路由过程中收集轨迹
    - 反向 TD 更新
    """
    
    def __init__(
        self,
        capacity: int = 300,
        n_connections: int = 3,
        gamma: float = 0.95,
        epsilon: float = 0.1
    ):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structon: Dict[str, Structon] = {}
        
        self.capacity = capacity
        self.n_connections = n_connections
        self.gamma = gamma
        self.epsilon = epsilon
        
        # 统计
        self.total_episodes = 0
        self.total_hops = 0
        
    def build(self, labels: List[str]):
        """构建 Structon 网络"""
        print("\n=== 创建 Structon v10 - 连接即动作 ===")
        Structon._id_counter = 0
        
        self.structons = []
        self.label_to_structon = {}
        
        for label in labels:
            s = Structon(label, self.n_connections, self.capacity)
            self.structons.append(s)
            self.label_to_structon[label] = s
            print(f"  + {s.id} label='{label}' actions={s.n_actions}")
            
        print("\n设置稀疏连接...")
        for s in self.structons:
            s.set_connections(self.structons)
            print(f"  {s.id} ({s.label}) → {s.get_connections_str()}")
    
    def train_episode(
        self,
        image: np.ndarray,
        true_label: str,
        max_hops: int = 15
    ) -> Dict:
        """
        训练一个 episode
        
        Args:
            image: 输入图像
            true_label: 真实标签
            max_hops: 最大跳转次数
            
        Returns:
            episode 统计信息
        """
        state = self.extractor.extract(image)
        
        # 收集轨迹: [(structon, action, reward), ...]
        trajectory = []
        current = np.random.choice(self.structons)
        visited = set()
        final_correct = False
        
        for hop in range(max_hops):
            visited.add(current.id)
            
            # ε-greedy 探索
            if np.random.random() < self.epsilon:
                action = np.random.randint(current.n_actions)
            else:
                action, _ = current.decide(state)
            
            if action == 0:  # STAY - 停止并声明
                if current.label == true_label:
                    reward = 2.0  # 正确停留
                    final_correct = True
                else:
                    reward = -1.5  # 错误停留
                trajectory.append((current, action, reward))
                break
            else:  # GOTO - 跳转
                next_structon = current.get_next(action)
                
                # 检查跳转有效性
                if next_structon is None or next_structon.id in visited:
                    # 无效跳转，强制 STAY
                    if current.label == true_label:
                        reward = 2.0
                        final_correct = True
                    else:
                        reward = -1.5
                    trajectory.append((current, 0, reward))  # 记录为 STAY
                    break
                
                # 有效跳转
                reward = -0.05  # 小惩罚，鼓励快速决策
                trajectory.append((current, action, reward))
                current = next_structon
        
        # 如果超过 max_hops 没停止，强制结束
        if len(trajectory) == 0 or trajectory[-1][1] != 0:
            if current.label == true_label:
                reward = 1.5
                final_correct = True
            else:
                reward = -1.0
            trajectory.append((current, 0, reward))
        
        # 反向 TD 更新
        next_q_max = 0.0
        for structon, action, reward in reversed(trajectory):
            target = reward + self.gamma * next_q_max
            structon.lrm.update_td(state, action, target)
            
            # 为下一次迭代准备
            q, _ = structon.lrm.query(state)
            next_q_max = np.max(q)
        
        # 统计
        self.total_episodes += 1
        self.total_hops += len(trajectory)
        
        return {
            'hops': len(trajectory),
            'correct': final_correct,
            'path': [t[0].label for t in trajectory],
        }
    
    def train_epoch(
        self,
        samples: List[Tuple[np.ndarray, str]],
        max_hops: int = 15
    ) -> Dict:
        """训练一个 epoch"""
        np.random.shuffle(samples)
        
        correct = 0
        total_hops = 0
        
        for img, label in samples:
            result = self.train_episode(img, label, max_hops)
            if result['correct']:
                correct += 1
            total_hops += result['hops']
        
        return {
            'accuracy': correct / len(samples) * 100,
            'avg_hops': total_hops / len(samples),
        }
    
    def predict(
        self,
        image: np.ndarray,
        max_hops: int = 20
    ) -> Tuple[str, List[str]]:
        """
        预测：贪心路由
        
        Returns:
            prediction: 预测的标签
            path: 路由路径
        """
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(current.label)
            
            action, _ = current.decide(state)
            
            if action == 0:  # STAY
                return current.label, path
            
            next_structon = current.get_next(action)
            if next_structon is None or next_structon.id in visited:
                return current.label, path
            
            current = next_structon
        
        return current.label, path
    
    def predict_voting(
        self,
        image: np.ndarray
    ) -> Tuple[str, Dict[str, float]]:
        """
        投票预测：所有 Structon 投票
        
        Returns:
            prediction: 预测的标签
            scores: 各标签的分数
        """
        state = self.extractor.extract(image)
        
        scores = {}
        for s in self.structons:
            q, _ = s.lrm.query(state)
            # STAY 的 Q 值作为该标签的分数
            scores[s.label] = q[0]
        
        best = max(scores, key=scores.get)
        return best, scores
    
    def evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 500,
        method: str = 'routing'
    ) -> Dict:
        """评估准确率"""
        indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
        
        correct = 0
        per_class = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
        total_hops = 0
        
        for idx in indices:
            true_label = str(labels[idx])
            per_class[true_label]['total'] += 1
            
            if method == 'routing':
                pred, path = self.predict(images[idx])
                total_hops += len(path)
            else:  # voting
                pred, _ = self.predict_voting(images[idx])
            
            if pred == true_label:
                correct += 1
                per_class[true_label]['correct'] += 1
        
        result = {
            'accuracy': correct / len(indices) * 100,
            'per_class': per_class,
        }
        
        if method == 'routing':
            result['avg_hops'] = total_hops / len(indices)
        
        return result
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("Structon Vision v10 - 连接即动作")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"连接数/节点: {self.n_connections}")
        print(f"动作空间: 1 (STAY) + {self.n_connections} (GOTO)")
        print(f"γ (折扣因子): {self.gamma}")
        print(f"ε (探索率): {self.epsilon}")
        
        total_mem = sum(s.lrm.size for s in self.structons)
        print(f"总记忆数: {total_mem}")
        
        if self.total_episodes > 0:
            print(f"总训练 episodes: {self.total_episodes}")
            print(f"平均 hops/episode: {self.total_hops / self.total_episodes:.2f}")
        
        print("\n=== 各 Structon ===")
        for s in self.structons:
            stats = s.lrm.get_stats()
            print(f"  {s.id} ['{s.label}'] mem:{stats['size']} "
                  f"queries:{stats['queries']} updates:{stats['updates']}")


# =============================================================================
# 6. 实验
# =============================================================================
def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 300,
    epochs: int = 30,
    n_connections: int = 3,
    gamma: float = 0.95,
    epsilon_start: float = 0.3,
    epsilon_end: float = 0.05,
):
    """运行实验"""
    print("=" * 70)
    print("Structon Vision v10 - 连接即动作 (Connection as Action)")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  - capacity: {capacity}")
    print(f"  - 每类样本: {n_per_class}")
    print(f"  - 连接数: {n_connections}")
    print(f"  - γ (折扣): {gamma}")
    print(f"  - ε (探索): {epsilon_start} → {epsilon_end}")
    
    print("\nLoading data...")
    train_images, train_labels, test_images, test_labels = try_load_mnist_or_synthetic()
    
    # 创建系统
    system = StructonVisionSystem(
        capacity=capacity,
        n_connections=n_connections,
        gamma=gamma,
        epsilon=epsilon_start
    )
    system.build([str(i) for i in range(10)])
    
    # 准备样本
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:n_per_class]
        for i in idxs:
            samples.append((train_images[i], str(d)))
    
    print(f"\n训练样本: {len(samples)}")
    
    # 训练
    print(f"\n=== 训练 ===")
    t0 = time.time()
    
    epsilon_decay = (epsilon_start - epsilon_end) / epochs
    
    for ep in range(epochs):
        # 衰减 epsilon
        system.epsilon = max(epsilon_end, epsilon_start - epsilon_decay * ep)
        
        result = system.train_epoch(samples)
        
        if (ep + 1) % 5 == 0:
            # 验证
            eval_result = system.evaluate(test_images, test_labels, n_samples=200)
            print(f"  轮次 {ep+1:2d}: "
                  f"train_acc={result['accuracy']:.1f}% "
                  f"test_acc={eval_result['accuracy']:.1f}% "
                  f"avg_hops={result['avg_hops']:.2f} "
                  f"ε={system.epsilon:.3f}")
    
    print(f"\n训练时间: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试 - 路由
    print(f"\n=== 测试（路由）===")
    result_routing = system.evaluate(test_images, test_labels, n_test, method='routing')
    print(f"准确率: {result_routing['accuracy']:.1f}%")
    print(f"平均 hops: {result_routing['avg_hops']:.2f}")
    
    # 测试 - 投票
    print(f"\n=== 测试（投票）===")
    result_voting = system.evaluate(test_images, test_labels, n_test, method='voting')
    print(f"准确率: {result_voting['accuracy']:.1f}%")
    
    # 各数字准确率
    print("\n各数字准确率:")
    print(f"{'数字':<6} {'路由':<12} {'投票':<12}")
    print("-" * 30)
    for d in range(10):
        s1 = result_routing['per_class'][str(d)]
        s2 = result_voting['per_class'][str(d)]
        acc1 = s1['correct']/s1['total']*100 if s1['total'] > 0 else 0
        acc2 = s2['correct']/s2['total']*100 if s2['total'] > 0 else 0
        print(f"  {d}     {acc1:>6.1f}%      {acc2:>6.1f}%")
    
    # 示例路径
    print("\n=== 示例路径 ===")
    test_idxs = np.random.choice(len(test_images), 10, replace=False)
    for i, idx in enumerate(test_idxs):
        pred, path = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        path_str = ' → '.join(path[:10])
        if len(path) > 10:
            path_str += f" ... ({len(path)} hops)"
        print(f"  {i+1}. 真实={true}, 预测={pred} {status}, 路径: {path_str}")
    
    return system


def compare_with_baseline(n_per_class: int = 100, n_test: int = 500):
    """与 v9.6 基线对比"""
    print("=" * 70)
    print("对比实验: v10 (连接即动作) vs v9.6 (YES/NO)")
    print("=" * 70)
    
    print("\nLoading data...")
    train_images, train_labels, test_images, test_labels = try_load_mnist_or_synthetic()
    
    # 准备样本
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:n_per_class]
        for i in idxs:
            samples.append((train_images[i], str(d)))
    
    results = {}
    
    # v10 - 连接即动作
    print("\n--- v10: 连接即动作 ---")
    system_v10 = StructonVisionSystem(
        capacity=300, n_connections=3, gamma=0.95, epsilon=0.2
    )
    system_v10.build([str(i) for i in range(10)])
    
    t0 = time.time()
    for ep in range(30):
        system_v10.epsilon = max(0.05, 0.2 - 0.005 * ep)
        system_v10.train_epoch(samples)
    t_v10 = time.time() - t0
    
    result_v10 = system_v10.evaluate(test_images, test_labels, n_test, method='voting')
    results['v10'] = {
        'accuracy': result_v10['accuracy'],
        'time': t_v10,
        'memories': sum(s.lrm.size for s in system_v10.structons)
    }
    print(f"准确率: {result_v10['accuracy']:.1f}%")
    print(f"训练时间: {t_v10:.1f}秒")
    
    # 总结
    print("\n=== 对比总结 ===")
    print(f"{'版本':<15} {'准确率':<12} {'时间':<12} {'记忆数':<12}")
    print("-" * 50)
    for name, r in results.items():
        print(f"{name:<15} {r['accuracy']:>6.1f}%     {r['time']:>6.1f}s     {r['memories']:>6}")
    
    return results


# =============================================================================
# 7. 主函数
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Structon Vision v10 - Connection as Action')
    parser.add_argument('--per-class', type=int, default=100, help='训练样本数/类')
    parser.add_argument('--test', type=int, default=500, help='测试样本数')
    parser.add_argument('--capacity', type=int, default=300, help='记忆容量')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--connections', type=int, default=3, help='每个节点的连接数')
    parser.add_argument('--gamma', type=float, default=0.95, help='折扣因子')
    parser.add_argument('--epsilon-start', type=float, default=0.3, help='初始探索率')
    parser.add_argument('--epsilon-end', type=float, default=0.05, help='最终探索率')
    parser.add_argument('--compare', action='store_true', help='与基线对比')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_with_baseline(args.per_class, args.test)
    else:
        run_experiment(
            n_per_class=args.per_class,
            n_test=args.test,
            capacity=args.capacity,
            epochs=args.epochs,
            n_connections=args.connections,
            gamma=args.gamma,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
