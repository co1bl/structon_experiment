#!/usr/bin/env python3
"""
Structon Vision v8.0 - 统一架构

核心简化：
- 不再区分 Atomic 和 Wrapper
- 每个 Structon 既是识别器又是路由器
- Structon: "这是我的吗？是→输出label，不是→交给左孩子"

结构：
S9 (label='9')
├─ [是 9] → 输出 "9"
└─ [不是 9] → S8 (label='8')
                ├─ [是 8] → 输出 "8"
                └─ [不是 8] → S7 ...
                                └─ S1 (label='1')
                                    ├─ [是 1] → 输出 "1"
                                    └─ [不是 1] → S0 (label='0')
                                                    └─ 输出 "0"（叶子）
"""

import numpy as np
from typing import Optional, Tuple, List
import time
import gzip
import os
import urllib.request


# =============================================================================
# 1. MNIST 加载
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


# =============================================================================
# 2. 特征提取
# =============================================================================

class StateExtractor:
    """简单的特征提取器：5x5 下采样"""
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        
        # 5x5 下采样
        h, w = img.shape
        bh, bw = h // 5, w // 5
        features = []
        for i in range(5):
            for j in range(5):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                features.append(np.mean(block))
        
        state = np.array(features, dtype=np.float32)
        
        # 归一化
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        
        return state


# =============================================================================
# 3. Local Resonant Memory (LRM)
# =============================================================================

class LRM:
    """
    Local Resonant Memory
    
    简化版：
    - 2 个动作：[是我的, 不是我的]
    - 基于余弦相似度匹配
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        capacity: int = 200,
        key_dim: int = 16,
        similarity_threshold: float = 0.95,
        learning_rate: float = 0.3
    ):
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        self.similarity_threshold = similarity_threshold
        self.learning_rate = learning_rate
        self.n_actions = 2  # [是, 不是]
        
        # 随机投影矩阵
        self.projection = np.random.randn(state_dim, key_dim).astype(np.float32)
        self.projection /= np.linalg.norm(self.projection, axis=0, keepdims=True)
        
        # 记忆存储
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []  # Q-values for each action
        self.access_counts: List[int] = []
        
        self.frozen = False
    
    def _compute_key(self, state: np.ndarray) -> np.ndarray:
        key = state @ self.projection
        norm = np.linalg.norm(key)
        if norm > 1e-6:
            key = key / norm
        return key
    
    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """查询状态对应的 Q 值"""
        key = self._compute_key(state)
        
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        key_matrix = np.array(self.keys)
        scores = key_matrix @ key
        
        # 加权平均
        weights = np.maximum(scores, 0) ** 2
        weight_sum = np.sum(weights)
        
        if weight_sum < 1e-6:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        weights = weights / weight_sum
        q_values = np.zeros(self.n_actions, dtype=np.float32)
        for i, w in enumerate(weights):
            if w > 0.01:
                q_values += w * self.values[i]
        
        confidence = float(np.max(scores))
        return q_values, confidence
    
    def remember(self, state: np.ndarray, action: int, target_q: float) -> str:
        """记住经验"""
        if self.frozen:
            return 'frozen'
        
        key = self._compute_key(state)
        
        # 检查是否有相似记忆
        if len(self.keys) > 0:
            key_matrix = np.array(self.keys)
            scores = key_matrix @ key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score > self.similarity_threshold:
                # 更新现有记忆
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (target_q - old_q)
                self.access_counts[best_idx] += 1
                return 'update'
        
        # 创建新记忆
        if len(self.keys) > 0:
            new_q, _ = self.query(state)
            new_q = new_q.copy()
        else:
            new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = target_q
        
        # 容量管理
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
        
        self.keys.append(key.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        
        return 'new'
    
    def freeze(self):
        self.frozen = True
    
    @property
    def size(self) -> int:
        return len(self.keys)


# =============================================================================
# 4. 统一的 Structon
# =============================================================================

class Structon:
    """
    统一的 Structon - 既是识别器又是路由器
    
    核心逻辑：
    - "这是我的吗？"
    - 是 → 输出 self.label
    - 不是 → 交给 left_child 继续判断
    
    LRM 学习：
    - action[0]: 是我的
    - action[1]: 不是我的
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        left_child: Optional['Structon'] = None,
        state_dim: int = 25,
        capacity: int = 200,
        key_dim: int = 16
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.label = label
        self.left_child = left_child  # None 表示叶子节点
        
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        self.lrm = LRM(
            state_dim=state_dim,
            capacity=capacity,
            key_dim=key_dim
        )
        
        self.frozen = False
        
        # 统计
        self.total_executes = 0
        self.total_learns = 0
        
        # 追踪学习效果
        self.history = []  # 最近的正确/错误记录
        self.history_window = 20
    
    def execute(self, state: np.ndarray) -> Tuple[Optional[str], float]:
        """
        执行判断
        
        Returns:
            label: 识别结果
            confidence: 置信度
        """
        self.total_executes += 1
        
        q_values, confidence = self.lrm.query(state)
        
        if q_values[0] > q_values[1]:  # action=0: 是我的
            return self.label, confidence
        else:  # action=1: 不是我的
            if self.left_child is not None:
                return self.left_child.execute(state)
            else:
                # 叶子节点，默认返回自己（最后的选择）
                return self.label, confidence * 0.5
    
    def learn(self, state: np.ndarray, true_label: str) -> bool:
        """
        学习
        
        Returns:
            correct: 这次判断是否正确
        """
        self.total_learns += 1
        
        is_mine = (true_label == self.label)
        
        # 先执行看当前判断
        q_values, _ = self.lrm.query(state)
        predicted_mine = (q_values[0] > q_values[1])
        
        # 计算正确性
        if is_mine:
            correct = predicted_mine  # 应该说"是我的"
        else:
            correct = not predicted_mine  # 应该说"不是我的"
        
        # 学习
        if is_mine:
            # 这是我的！强化 "是"
            self.lrm.remember(state, action=0, target_q=1.0)
            self.lrm.remember(state, action=1, target_q=-0.5)
        else:
            # 不是我的！强化 "不是"
            self.lrm.remember(state, action=1, target_q=1.0)
            self.lrm.remember(state, action=0, target_q=-0.5)
        
        # 递归让左孩子也学习
        if self.left_child is not None:
            self.left_child.learn(state, true_label)
        
        # 记录历史
        self.history.append(1 if correct else 0)
        if len(self.history) > self.history_window * 2:
            self.history = self.history[-self.history_window:]
        
        return correct
    
    def get_accuracy(self) -> float:
        """获取最近的准确率"""
        if len(self.history) < self.history_window:
            return 0.0
        recent = self.history[-self.history_window:]
        return sum(recent) / len(recent)
    
    def freeze(self):
        """冻结（停止学习）"""
        self.frozen = True
        self.lrm.freeze()
        if self.left_child is not None:
            self.left_child.freeze()
    
    def depth(self) -> int:
        """树深度"""
        if self.left_child is None:
            return 1
        return 1 + self.left_child.depth()
    
    def count_nodes(self) -> int:
        """节点总数"""
        if self.left_child is None:
            return 1
        return 1 + self.left_child.count_nodes()
    
    def total_memories(self) -> int:
        """总记忆数"""
        total = self.lrm.size
        if self.left_child is not None:
            total += self.left_child.total_memories()
        return total
    
    def print_tree(self, indent: int = 0):
        """打印树结构"""
        prefix = "  " * indent
        icon = "❄️" if self.frozen else "🔥"
        acc = self.get_accuracy() * 100
        print(f"{prefix}{icon} {self.id} [label='{self.label}'] "
              f"mem:{self.lrm.size}/{self.capacity} acc:{acc:.0f}%")
        if self.left_child is not None:
            print(f"{prefix}  └─[不是{self.label}]:")
            self.left_child.print_tree(indent + 2)


# =============================================================================
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统 v8
    
    简化版：只有一种 Structon，向上生长
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        capacity: int = 200,
        key_dim: int = 16
    ):
        self.extractor = StateExtractor()
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        self.root: Optional[Structon] = None
        self.promote_count = 0
    
    def add_class(self, label: str):
        """
        添加新类别
        
        创建新的 Structon，把旧的作为 left_child
        """
        self.promote_count += 1
        
        new_structon = Structon(
            label=label,
            left_child=self.root,  # 旧 root 变成 left_child
            state_dim=self.state_dim,
            capacity=self.capacity,
            key_dim=self.key_dim
        )
        
        self.root = new_structon
        print(f"  + 添加 Structon label='{label}', 总数: {self.promote_count}")
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        if self.root is None:
            return "?", 0.0
        
        state = self.extractor.extract(image)
        result, confidence = self.root.execute(state)
        
        return result if result else "?", confidence
    
    def train(self, image: np.ndarray, label: str) -> bool:
        """训练一个样本"""
        if self.root is None:
            return False
        
        state = self.extractor.extract(image)
        return self.root.learn(state, label)
    
    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("Structon Vision System v8.0")
        print("=" * 60)
        
        if self.root is None:
            print("(空)")
            return
        
        print(f"类别数: {self.promote_count}")
        print(f"深度: {self.root.depth()}")
        print(f"节点数: {self.root.count_nodes()}")
        print(f"总记忆: {self.root.total_memories()}")
        
        print("\n=== 树结构 ===")
        self.root.print_tree()


# =============================================================================
# 6. 实验
# =============================================================================

def run_experiment(
    n_per_class: int = 200,
    n_test: int = 500,
    capacity: int = 200,
    key_dim: int = 16,
    target_accuracy: float = 0.90,
    max_epochs: int = 30,
    min_epochs: int = 3
):
    """运行实验"""
    print("=" * 70)
    print("Structon Vision v8.0 - 统一架构")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  capacity={capacity}, key_dim={key_dim}")
    print(f"  target_accuracy={target_accuracy}")
    print(f"  max_epochs={max_epochs}, min_epochs={min_epochs}")
    print(f"  每类训练: {n_per_class}, 测试: {n_test}")
    
    print("\n核心设计:")
    print("  - 每个 Structon 既是识别器又是路由器")
    print("  - '是我的' → 输出 label")
    print("  - '不是我的' → 交给 left_child")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        state_dim=25,
        capacity=capacity,
        key_dim=key_dim
    )
    
    # 准备每个类别的样本
    class_samples = {}
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        np.random.shuffle(indices)
        class_samples[digit] = [(train_images[i], str(digit)) for i in indices]
    
    print(f"\n=== 逐类增量学习 ===")
    t0 = time.time()
    
    total_samples_used = 0
    
    for current_digit in range(10):
        print(f"\n--- 阶段 {current_digit}: 学习数字 {current_digit} ---")
        
        # 1. 添加新类别
        system.add_class(str(current_digit))
        
        # 2. 准备训练样本：所有已学数字
        train_samples = []
        for digit in range(current_digit + 1):
            train_samples.extend(class_samples[digit])
        
        print(f"  训练样本: {len(train_samples)} (数字 0-{current_digit})")
        
        # 3. 训练直到准确率达标
        epoch = 0
        best_acc = 0.0
        
        while epoch < max_epochs:
            epoch += 1
            np.random.shuffle(train_samples)
            
            epoch_correct = 0
            for img, label in train_samples:
                state = system.extractor.extract(img)
                result, conf = system.root.execute(state)
                if result == label:
                    epoch_correct += 1
                system.train(img, label)
                total_samples_used += 1
            
            acc = epoch_correct / len(train_samples) * 100
            best_acc = max(best_acc, acc)
            
            # 每 5 轮或达标时打印
            if epoch % 5 == 0 or acc >= target_accuracy * 100:
                print(f"    轮次 {epoch}: 准确率 {acc:.1f}%")
            
            # 检查是否达标
            if epoch >= min_epochs and acc >= target_accuracy * 100:
                print(f"  ✓ 达标! 准确率 {acc:.1f}% >= {target_accuracy*100}%")
                break
        
        if epoch >= max_epochs:
            print(f"  ✗ 达到最大轮数，最佳准确率 {best_acc:.1f}%")
    
    print(f"\n训练完成: {time.time()-t0:.1f}秒")
    print(f"总样本: {total_samples_used}")
    
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试 {n_test} 样本 ===")
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    t0 = time.time()
    for idx in test_indices:
        predicted, confidence = system.predict(test_images[idx])
        true_label = str(test_labels[idx])
        
        results[true_label]['total'] += 1
        if predicted == true_label:
            results[true_label]['correct'] += 1
    
    print(f"测试完成: {time.time()-t0:.1f}秒")
    
    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    
    print(f"\n总准确率: {total_correct/total_samples*100:.1f}%")
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  {d}: {acc:.1f}% ({r['correct']}/{r['total']})")
    
    return system


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-class', type=int, default=200)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=200)
    parser.add_argument('--key-dim', type=int, default=16)
    parser.add_argument('--target-acc', type=float, default=0.90)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--min-epochs', type=int, default=3)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        key_dim=args.key_dim,
        target_accuracy=args.target_acc,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs
    )
