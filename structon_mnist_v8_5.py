#!/usr/bin/env python3
"""
Structon Vision v8.1 - 全连接图结构

核心设计：
- 所有 Structon 平等，没有固定顺序
- 每个 Structon 可以跳到任何其他 Structon
- S_i 的动作: [是i, 去其他所有S]
- 连接是动作，由 LRM 学习

结构：全连接图
    S0 ←→ S1 ←→ S2 ←→ ...
     ↕  ╲  ↕  ╱  ↕
    S3 ←→ S4 ←→ S5 ←→ ...
    （所有节点互相连接）

入口：从最后创建的 Structon 开始
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
# 2. 特征提取 - 增强版
# =============================================================================

class StateExtractor:
    """
    增强特征提取器
    
    特征：
    1. 5x5 下采样 (25维)
    2. 水平/垂直投影 (10维)
    3. 结构特征：洞数、端点、交叉点等 (10维)
    
    总共 45 维
    """
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        
        features = []
        
        # 1. 5x5 下采样 (25维)
        h, w = img.shape
        bh, bw = h // 5, w // 5
        for i in range(5):
            for j in range(5):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                features.append(np.mean(block))
        
        # 2. 水平投影 (5维) - 每行的平均亮度
        for i in range(5):
            row_start = i * (h // 5)
            row_end = (i + 1) * (h // 5)
            features.append(np.mean(img[row_start:row_end, :]))
        
        # 3. 垂直投影 (5维) - 每列的平均亮度
        for j in range(5):
            col_start = j * (w // 5)
            col_end = (j + 1) * (w // 5)
            features.append(np.mean(img[:, col_start:col_end]))
        
        # 4. 结构特征 (10维)
        binary = (img > 0.3).astype(np.uint8)
        
        # 4.1 总像素比例
        features.append(np.mean(binary))
        
        # 4.2 上下比例（区分 6 和 9）
        upper_half = np.mean(binary[:h//2, :])
        lower_half = np.mean(binary[h//2:, :])
        features.append(upper_half - lower_half)
        
        # 4.3 左右比例
        left_half = np.mean(binary[:, :w//2])
        right_half = np.mean(binary[:, w//2:])
        features.append(left_half - right_half)
        
        # 4.4 中心密度（区分 0 和 8：0 中间空，8 中间满）
        center_region = binary[h//4:3*h//4, w//4:3*w//4]
        features.append(np.mean(center_region))
        
        # 4.5 边缘密度
        edge_density = (np.mean(binary[0, :]) + np.mean(binary[-1, :]) + 
                       np.mean(binary[:, 0]) + np.mean(binary[:, -1])) / 4
        features.append(edge_density)
        
        # 4.6 对角线特征（区分 1 和 7）
        diag1 = np.mean([binary[i, i] for i in range(min(h, w))])
        diag2 = np.mean([binary[i, w-1-i] for i in range(min(h, w))])
        features.append(diag1)
        features.append(diag2)
        
        # 4.7 水平线检测（顶部、中部、底部）
        features.append(np.mean(binary[2:5, :]))   # 顶部
        features.append(np.mean(binary[h//2-2:h//2+2, :]))  # 中部
        features.append(np.mean(binary[-5:-2, :]))  # 底部
        
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
    Local Resonant Memory - 支持动态动作数
    """
    
    def __init__(
        self,
        state_dim: int = 25,
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
        self.values: List[np.ndarray] = []
        self.access_counts: List[int] = []
        
        self.frozen = False
    
    def set_n_actions(self, n_actions: int):
        """设置动作数（扩展现有记忆）"""
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
        
        confidence = float(np.max(scores))
        return q_values, confidence
    
    def remember(self, state: np.ndarray, action: int, target_q: float) -> str:
        if self.frozen:
            return 'frozen'
        
        if action >= self.n_actions:
            return 'invalid_action'
        
        key = self._compute_key(state)
        
        if len(self.keys) > 0:
            key_matrix = np.array(self.keys)
            scores = key_matrix @ key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score > self.similarity_threshold:
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (target_q - old_q)
                self.access_counts[best_idx] += 1
                return 'update'
        
        if len(self.keys) > 0:
            new_q, _ = self.query(state)
            new_q = new_q.copy()
        else:
            new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = target_q
        
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
        
        self.keys.append(key.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        
        return 'new'
    
    @property
    def size(self) -> int:
        return len(self.keys)


# =============================================================================
# 4. Structon - 全连接版本
# =============================================================================

class Structon:
    """
    Structon - 全连接图节点
    
    动作空间：
    - action[0]: "是我的" → 输出 self.label
    - action[1..N]: "去其他 Structon"
    
    所有 Structon 平等，可以互相跳转
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        state_dim: int = 25,
        capacity: int = 200,
        key_dim: int = 16
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.label = label
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        # 初始只有 1 个动作：是我的
        self.lrm = LRM(
            state_dim=state_dim,
            n_actions=1,
            capacity=capacity,
            key_dim=key_dim
        )
        
        # 连接到其他 Structon 的映射
        # action_idx -> Structon
        self.connections: Dict[int, 'Structon'] = {}
        
        # 统计
        self.total_executes = 0
        self.total_learns = 0
    
    def add_connection(self, other: 'Structon'):
        """添加到另一个 Structon 的连接"""
        new_action_idx = len(self.connections) + 1
        self.connections[new_action_idx] = other
        self.lrm.set_n_actions(new_action_idx + 1)
    
    def execute(self, state: np.ndarray, visited: set = None, max_hops: int = 10) -> Tuple[str, float]:
        """执行判断"""
        self.total_executes += 1
        
        if visited is None:
            visited = set()
        
        if self.id in visited or len(visited) >= max_hops:
            return self.label, 0.1
        
        visited.add(self.id)
        
        q_values, confidence = self.lrm.query(state)
        best_action = int(np.argmax(q_values))
        
        if best_action == 0:
            return self.label, confidence
        elif best_action in self.connections:
            return self.connections[best_action].execute(state, visited, max_hops)
        else:
            return self.label, confidence * 0.5
    
    def learn(self, state: np.ndarray, true_label: str, all_structons: Dict[str, 'Structon']):
        """学习"""
        self.total_learns += 1
        
        is_mine = (true_label == self.label)
        
        if is_mine:
            # 这是我的！强化 action[0]
            self.lrm.remember(state, action=0, target_q=1.0)
            for action_idx in self.connections:
                self.lrm.remember(state, action=action_idx, target_q=-0.3)
        else:
            # 不是我的！
            self.lrm.remember(state, action=0, target_q=-0.5)
            
            # 强化跳转到正确的 Structon
            for action_idx, target_structon in self.connections.items():
                if target_structon.label == true_label:
                    self.lrm.remember(state, action=action_idx, target_q=1.0)
                else:
                    self.lrm.remember(state, action=action_idx, target_q=-0.1)


# =============================================================================
# 5. Vision System - 全连接图
# =============================================================================

class StructonVisionSystem:
    """Structon 视觉系统 v8.3 - 全连接图 + Router 入口"""
    
    def __init__(
        self,
        state_dim: int = 45,  # 增强特征：25 + 10 + 10
        capacity: int = 200,
        key_dim: int = 16
    ):
        self.extractor = StateExtractor()
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        self.structons: Dict[str, Structon] = {}
        
        # 特殊的 Router（入口），不识别任何数字，只路由
        self.router = Structon(
            label="ROUTER",  # 特殊标签，永远不会匹配
            state_dim=state_dim,
            capacity=capacity,
            key_dim=key_dim
        )
        # Router 的 action[0]（是我的）永远不会被强化
    
    def add_class(self, label: str):
        """添加新类别"""
        new_structon = Structon(
            label=label,
            state_dim=self.state_dim,
            capacity=self.capacity,
            key_dim=self.key_dim
        )
        
        # 新 Structon 与所有现有 Structon 双向连接
        for existing in self.structons.values():
            new_structon.add_connection(existing)
            existing.add_connection(new_structon)
        
        # Router 连接到新 Structon
        self.router.add_connection(new_structon)
        
        # 新 Structon 也连接回 Router（可选，允许"我不确定"）
        # new_structon.add_connection(self.router)
        
        self.structons[label] = new_structon
        
        print(f"  + {new_structon.id} label='{label}', "
              f"actions: {new_structon.lrm.n_actions}, "
              f"router_actions: {self.router.lrm.n_actions}")
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测 - 从 Router 开始"""
        state = self.extractor.extract(image)
        return self.router.execute(state)
    
    def train(self, image: np.ndarray, label: str):
        """训练 - 从 Router 开始，沿路径学习"""
        state = self.extractor.extract(image)
        
        # 1. 从 Router 执行，记录路径
        path = []
        self._execute_and_record(state, path, self.router)
        
        # 2. 沿路径学习
        for structon in path:
            structon.learn(state, label, self.structons)
        
        # 3. 确保正确的 Structon 学习
        if label in self.structons:
            correct_structon = self.structons[label]
            if correct_structon not in path:
                correct_structon.learn(state, label, self.structons)
        
        # 4. Router 学习：应该跳到正确的 Structon
        self._train_router(state, label)
    
    def _train_router(self, state: np.ndarray, label: str):
        """训练 Router 直接跳到正确的 Structon"""
        # Router 的 action[0] 是"是 ROUTER"，永远不强化
        self.router.lrm.remember(state, action=0, target_q=-1.0)
        
        # 找到指向正确 Structon 的动作
        for action_idx, target in self.router.connections.items():
            if target.label == label:
                self.router.lrm.remember(state, action=action_idx, target_q=1.0)
            else:
                self.router.lrm.remember(state, action=action_idx, target_q=-0.2)
    
    def _execute_and_record(self, state: np.ndarray, path: list, 
                            entry: Structon,
                            visited: set = None, max_hops: int = 10) -> Tuple[str, float]:
        """执行并记录路径"""
        if visited is None:
            visited = set()
        
        current = entry
        
        while current is not None and current.id not in visited and len(visited) < max_hops:
            visited.add(current.id)
            path.append(current)
            
            q_values, confidence = current.lrm.query(state)
            best_action = int(np.argmax(q_values))
            
            if best_action == 0 and current.label != "ROUTER":  # 是我的（Router 除外）
                return current.label, confidence
            elif best_action in current.connections:
                current = current.connections[best_action]
            else:
                if current.label != "ROUTER":
                    return current.label, confidence * 0.5
                else:
                    # Router 没有有效跳转，随机选一个
                    if current.connections:
                        current = list(current.connections.values())[0]
                    else:
                        return "?", 0.0
        
        if current and current.label != "ROUTER":
            return current.label, 0.1
        return "?", 0.0
    
    def print_stats(self):
        """打印统计"""
        print("\n" + "=" * 60)
        print("Structon Vision v8.3 - 全连接图 + Router")
        print("=" * 60)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"Router 记忆: {self.router.lrm.size}/{self.router.capacity}")
        total_mem = sum(s.lrm.size for s in self.structons.values())
        print(f"总记忆: {total_mem}")
        
        print("\n=== 各 Structon ===")
        for label in sorted(self.structons.keys()):
            s = self.structons[label]
            print(f"  {s.id} ['{s.label}'] mem:{s.lrm.size}/{s.capacity}")


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
    print("Structon Vision v8.1 - 全连接图")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, 每类={n_per_class}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        state_dim=45,  # 增强特征
        capacity=capacity,
        key_dim=key_dim
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
    
    print(f"\n=== 训练 ===")
    t0 = time.time()
    
    all_samples = []
    for digit in range(10):
        all_samples.extend(class_samples[digit])
    
    for epoch in range(max_epochs):
        np.random.shuffle(all_samples)
        
        correct = 0
        for img, label in all_samples:
            result, _ = system.predict(img)
            if result == label:
                correct += 1
            system.train(img, label)
        
        acc = correct / len(all_samples) * 100
        
        if (epoch + 1) % 5 == 0:
            print(f"  轮次 {epoch+1}: {acc:.1f}%")
        
        if epoch >= min_epochs - 1 and acc >= target_accuracy * 100:
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
    parser.add_argument('--per-class', type=int, default=200)
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
