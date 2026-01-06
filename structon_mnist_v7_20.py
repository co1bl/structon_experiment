"""
Structon Vision v7.20 - 统一分形架构
=====================================

核心改变：
1. 每个 Structon 结构相同（自相似）
2. LRM 存储 Q-values（长度 = n_children）
3. children 可以是 action 字符串或子 Structon
4. 混合记忆：同一个 LRM 存储多类别的路由决策

设计哲学：
- Structure is skeleton, Memory is soul
- 局部规则，全局涌现
- 适应优于优化

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. MNIST 加载
# =============================================================================

def load_mnist(path='./mnist_data'):
    """加载 MNIST 数据集"""
    os.makedirs(path, exist_ok=True)
    
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    for name, filename in files.items():
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    def read_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8)
            return images.reshape(num, rows, cols)
    
    def read_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            magic, num = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    train_images = read_images(os.path.join(path, files['train_images']))
    train_labels = read_labels(os.path.join(path, files['train_labels']))
    test_images = read_images(os.path.join(path, files['test_images']))
    test_labels = read_labels(os.path.join(path, files['test_labels']))
    
    return train_images, train_labels, test_images, test_labels


# =============================================================================
# 2. 特征提取器
# =============================================================================

class StateExtractor:
    """提取 MNIST 图像的状态特征"""
    
    def __init__(self):
        self.feature_dim = 25
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取 25 维特征"""
        img = image.astype(np.float32) / 255.0
        binary = (img > 0.3).astype(np.float32)
        
        features = []
        
        # 1. 拓扑特征 (4D)
        n_holes = self._count_holes(binary)
        endpoints = self._find_endpoints(binary)
        junctions = self._find_junctions(binary)
        is_closed = 1.0 if n_holes > 0 else 0.0
        
        features.extend([
            n_holes / 3.0,
            len(endpoints) / 5.0,
            len(junctions) / 3.0,
            is_closed
        ])
        
        # 2. 端点位置 (9D) - 9 宫格
        ep_regions = np.zeros(9)
        for y, x in endpoints:
            ry, rx = min(2, y // 10), min(2, x // 10)
            ep_regions[ry * 3 + rx] = 1.0
        features.extend(ep_regions)
        
        # 3. 交叉点位置 (3D) - 上中下
        jc_regions = np.zeros(3)
        for y, x in junctions:
            region = min(2, y // 10)
            jc_regions[region] = 1.0
        features.extend(jc_regions)
        
        # 4. 边缘方向 (4D)
        h_top = np.sum(binary[:10, :]) / (10 * 28)
        h_bottom = np.sum(binary[18:, :]) / (10 * 28)
        v_left = np.sum(binary[:, :10]) / (28 * 10)
        v_right = np.sum(binary[:, 18:]) / (28 * 10)
        features.extend([h_top, h_bottom, v_left, v_right])
        
        # 5. 密度分布 (3D)
        density_top = np.sum(binary[:9, :]) / (9 * 28)
        density_mid = np.sum(binary[9:18, :]) / (9 * 28)
        density_bottom = np.sum(binary[18:, :]) / (10 * 28)
        features.extend([density_top, density_mid, density_bottom])
        
        # 6. 质心 (2D)
        ys, xs = np.where(binary > 0)
        if len(xs) > 0:
            cx, cy = np.mean(xs) / 28.0, np.mean(ys) / 28.0
        else:
            cx, cy = 0.5, 0.5
        features.extend([cx, cy])
        
        return np.array(features, dtype=np.float32)
    
    def _count_holes(self, binary: np.ndarray) -> int:
        from collections import deque
        
        padded = np.pad(binary, 1, mode='constant', constant_values=0)
        visited = np.zeros_like(padded, dtype=bool)
        
        queue = deque([(0, 0)])
        visited[0, 0] = True
        
        while queue:
            y, x = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < padded.shape[0] and 0 <= nx < padded.shape[1]:
                    if not visited[ny, nx] and padded[ny, nx] == 0:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        holes = 0
        for y in range(1, padded.shape[0] - 1):
            for x in range(1, padded.shape[1] - 1):
                if padded[y, x] == 0 and not visited[y, x]:
                    holes += 1
                    queue = deque([(y, x)])
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.popleft()
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if not visited[ny, nx] and padded[ny, nx] == 0:
                                visited[ny, nx] = True
                                queue.append((ny, nx))
        
        return holes
    
    def _find_endpoints(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        endpoints = []
        skeleton = binary.copy()
        
        for y in range(1, binary.shape[0] - 1):
            for x in range(1, binary.shape[1] - 1):
                if skeleton[y, x] > 0:
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbors == 1:
                        endpoints.append((y, x))
        
        return endpoints[:5]
    
    def _find_junctions(self, binary: np.ndarray) -> List[Tuple[int, int]]:
        junctions = []
        skeleton = binary.copy()
        
        for y in range(1, binary.shape[0] - 1):
            for x in range(1, binary.shape[1] - 1):
                if skeleton[y, x] > 0:
                    neighbors = np.sum(skeleton[y-1:y+2, x-1:x+2]) - skeleton[y, x]
                    if neighbors >= 3:
                        junctions.append((y, x))
        
        return junctions[:3]


# =============================================================================
# 3. 共振记忆 (Resonant Memory)
# =============================================================================

class ResonantMemory:
    """
    共振记忆：基于余弦相似度的无梯度学习
    
    存储 pattern → Q-values 映射
    Q-values 长度 = n_actions（children 数量）
    """
    
    def __init__(
        self,
        key_dim: int,
        n_actions: int,
        capacity: int = 100,
        temperature: float = 0.1,
        lr: float = 0.3,
        similarity_threshold: float = 0.85
    ):
        self.key_dim = key_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.temperature = temperature
        self.lr = lr
        self.similarity_threshold = similarity_threshold
        
        # 记忆存储
        self.keys: List[np.ndarray] = []          # pattern keys
        self.values: List[np.ndarray] = []        # Q-values (n_actions 维)
        self.access_counts: List[int] = []
        
        # 冻结状态
        self.frozen: bool = False
        
        # 统计
        self.total_queries = 0
        self.total_updates = 0
    
    def query(self, key: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        共振查询
        
        Returns:
            q_values: 各 action 的 Q 值
            confidence: 置信度 (0-1)
        """
        self.total_queries += 1
        
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        # 计算相似度
        key_matrix = np.array(self.keys)
        scores = key_matrix @ key
        
        max_score = float(np.max(scores))
        
        # Softmax 加权
        exp_scores = np.exp((scores - max_score) / self.temperature)
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        # 加权求和得到 Q-values
        value_matrix = np.array(self.values)
        q_values = weights @ value_matrix
        
        # 更新访问计数
        best_idx = int(np.argmax(scores))
        self.access_counts[best_idx] += 1
        
        # 置信度
        confidence = max(0.0, min(1.0, (max_score - 0.5) * 2))
        
        return q_values.astype(np.float32), confidence
    
    def remember(
        self,
        key: np.ndarray,
        action: int,
        target: float
    ) -> str:
        """
        写入记忆
        
        Returns:
            'update' - 更新现有记忆
            'new' - 创建新记忆
            'frozen' - 冻结状态，拒绝写入
        """
        if self.frozen:
            return 'frozen'
        
        self.total_updates += 1
        
        # 查找相似记忆
        best_idx = -1
        best_score = -1.0
        
        if len(self.keys) > 0:
            key_matrix = np.array(self.keys)
            scores = key_matrix @ key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score < self.similarity_threshold:
                best_idx = -1
        
        if best_idx >= 0:
            # 更新现有记忆
            old_val = self.values[best_idx][action]
            self.values[best_idx][action] = old_val + self.lr * (target - old_val)
            self.access_counts[best_idx] += 1
            return 'update'
        else:
            # 容量管理
            if len(self.keys) >= self.capacity:
                min_idx = int(np.argmin(self.access_counts))
                self.keys.pop(min_idx)
                self.values.pop(min_idx)
                self.access_counts.pop(min_idx)
            
            # 创建新记忆
            new_q = np.zeros(self.n_actions, dtype=np.float32)
            if len(self.keys) > 0:
                # 继承当前查询结果
                new_q, _ = self.query(key)
                new_q = new_q.copy()
            new_q[action] = target
            
            self.keys.append(key.copy())
            self.values.append(new_q)
            self.access_counts.append(1)
            return 'new'
    
    def freeze(self):
        """冻结记忆"""
        self.frozen = True
    
    def unfreeze(self):
        """解冻记忆"""
        self.frozen = False
    
    @property
    def size(self) -> int:
        return len(self.keys)
    
    def get_stats(self) -> Dict:
        return {
            'size': self.size,
            'capacity': self.capacity,
            'capacity_ratio': self.size / self.capacity if self.capacity > 0 else 0,
            'total_queries': self.total_queries,
            'total_updates': self.total_updates,
            'frozen': self.frozen
        }


# =============================================================================
# 4. 统一的 Structon
# =============================================================================

class Structon:
    """
    统一的 Structon - 自相似分形单元
    
    每个 Structon：
    - 有 LRM 存储路由决策（Q-values）
    - children 可以是 action 字符串或子 Structon
    - 用随机投影编码输入
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        children: List[Union[str, 'Structon']],
        capacity: int = 100,
        key_dim: int = 16,
        full_dim: int = 25,
        temperature: float = 0.1,
        lr: float = 0.3,
        similarity_threshold: float = 0.85
    ):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.children = children
        self.n_actions = len(children)
        
        # 随机投影
        self.full_dim = full_dim
        self.key_dim = key_dim
        self.projector = np.random.randn(full_dim, key_dim).astype(np.float32)
        self.projector /= (np.linalg.norm(self.projector, axis=0, keepdims=True) + 1e-8)
        
        # 共振记忆
        self.lrm = ResonantMemory(
            key_dim=key_dim,
            n_actions=self.n_actions,
            capacity=capacity,
            temperature=temperature,
            lr=lr,
            similarity_threshold=similarity_threshold
        )
        
        # 参数存储（用于创建子节点）
        self.capacity = capacity
        self.temperature = temperature
        self.lr = lr
        self.similarity_threshold = similarity_threshold
        
        # 统计
        self.total_queries = 0
        self.total_learns = 0
    
    def _encode(self, pattern: np.ndarray) -> np.ndarray:
        """随机投影编码"""
        key = pattern.astype(np.float32) @ self.projector
        norm = np.linalg.norm(key)
        if norm > 1e-8:
            key /= norm
        return key
    
    @property
    def frozen(self) -> bool:
        return self.lrm.frozen
    
    def freeze(self):
        """冻结（停止学习，但仍可查询）"""
        self.lrm.freeze()
    
    def unfreeze(self):
        """解冻"""
        self.lrm.unfreeze()
    
    def is_leaf(self) -> bool:
        """是否是叶子节点（children 全是字符串）"""
        return all(isinstance(c, str) for c in self.children)
    
    def query(self, pattern: np.ndarray) -> Tuple[int, np.ndarray, float]:
        """
        查询：选择哪个 child
        
        Returns:
            action_idx: 选择的 child 索引
            q_values: 所有 children 的 Q 值
            confidence: 置信度
        """
        self.total_queries += 1
        key = self._encode(pattern)
        q_values, confidence = self.lrm.query(key)
        action_idx = int(np.argmax(q_values))
        return action_idx, q_values, confidence
    
    def execute(self, pattern: np.ndarray) -> Tuple[str, float]:
        """
        执行：递归查询直到得到 action
        
        Returns:
            action: 最终选择的动作（字符串）
            confidence: 置信度
        """
        action_idx, q_values, confidence = self.query(pattern)
        child = self.children[action_idx]
        
        if isinstance(child, str):
            # 叶子：返回 action
            return child, confidence
        else:
            # 中间节点：递归
            return child.execute(pattern)
    
    def learn(
        self,
        pattern: np.ndarray,
        true_action: str,
        reward: float = 1.0
    ) -> str:
        """
        学习：更新路由决策
        
        Args:
            pattern: 输入模式
            true_action: 正确的动作（字符串）
            reward: 奖励值
        
        Returns:
            状态信息
        """
        self.total_learns += 1
        key = self._encode(pattern)
        
        # 找到 true_action 对应的 child 索引
        target_idx = None
        for i, child in enumerate(self.children):
            if isinstance(child, str):
                if child == true_action:
                    target_idx = i
                    break
            else:
                # 子 Structon：检查它能否到达 true_action
                # 简化：假设每个数字最终由对应索引的 child 处理
                if str(i) == true_action:
                    target_idx = i
                    break
        
        if target_idx is None:
            # 如果 children 是数字字符串
            try:
                target_idx = int(true_action)
            except:
                return "invalid_action"
        
        if target_idx >= self.n_actions:
            return "invalid_action"
        
        # 更新 LRM
        status = self.lrm.remember(key, target_idx, reward)
        
        # 如果 child 是 Structon，递归学习
        child = self.children[target_idx]
        if isinstance(child, Structon):
            child.learn(pattern, true_action, reward)
        
        return status
    
    def predict(self, pattern: np.ndarray) -> Tuple[str, float]:
        """预测（execute 的别名）"""
        return self.execute(pattern)
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            'id': self.id,
            'n_children': self.n_actions,
            'is_leaf': self.is_leaf(),
            'frozen': self.frozen,
            'total_queries': self.total_queries,
            'total_learns': self.total_learns,
            'lrm': self.lrm.get_stats()
        }
        
        if not self.is_leaf():
            stats['children'] = []
            for child in self.children:
                if isinstance(child, Structon):
                    stats['children'].append(child.get_stats())
                else:
                    stats['children'].append({'action': child})
        
        return stats
    
    def count_nodes(self) -> int:
        """统计节点数量"""
        count = 1
        for child in self.children:
            if isinstance(child, Structon):
                count += child.count_nodes()
        return count
    
    def depth(self) -> int:
        """获取深度"""
        if self.is_leaf():
            return 1
        
        max_child_depth = 0
        for child in self.children:
            if isinstance(child, Structon):
                max_child_depth = max(max_child_depth, child.depth())
        
        return 1 + max_child_depth
    
    def total_memories(self) -> int:
        """统计总记忆数"""
        count = self.lrm.size
        for child in self.children:
            if isinstance(child, Structon):
                count += child.total_memories()
        return count
    
    def print_tree(self, indent: int = 0):
        """打印树结构"""
        prefix = "  " * indent
        
        # 图标
        if self.is_leaf():
            icon = "🌿"
            role = "Leaf"
        elif self.frozen:
            icon = "❄️"
            role = "Frozen"
        else:
            icon = "🔥"
            role = "Active"
        
        print(f"{prefix}{icon} {self.id} ({role}) "
              f"[mem:{self.lrm.size}/{self.capacity}, "
              f"children:{self.n_actions}]")
        
        # 显示部分 Q-values
        if self.lrm.size > 0:
            for i, (key, val) in enumerate(zip(self.lrm.keys[:2], self.lrm.values[:2])):
                best_action = int(np.argmax(val))
                print(f"{prefix}  └─ pattern→{best_action} "
                      f"(Q: {val[best_action]:.2f})")
            if self.lrm.size > 2:
                print(f"{prefix}  └─ ... ({self.lrm.size - 2} more)")
        
        # 递归打印子节点
        for i, child in enumerate(self.children):
            if isinstance(child, Structon):
                child.print_tree(indent + 1)


# =============================================================================
# 5. Vision System
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统 - MNIST 分类
    
    单层 Structon，10 个 children 对应 10 个数字
    """
    
    def __init__(
        self,
        capacity: int = 100,
        key_dim: int = 16,
        full_dim: int = 25,
        temperature: float = 0.1,
        lr: float = 0.3,
        similarity_threshold: float = 0.85
    ):
        self.extractor = StateExtractor()
        
        # 创建 root Structon
        # children = ["0", "1", ..., "9"]
        self.root = Structon(
            children=[str(i) for i in range(10)],
            capacity=capacity,
            key_dim=key_dim,
            full_dim=full_dim,
            temperature=temperature,
            lr=lr,
            similarity_threshold=similarity_threshold
        )
        
        self.capacity = capacity
        self.key_dim = key_dim
        self.full_dim = full_dim
        
        # 统计
        self.train_count = 0
        self.correct_count = 0
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        state = self.extractor.extract(image)
        return self.root.predict(state)
    
    def train_one(self, image: np.ndarray, true_label: str, reward: float = 1.0) -> Tuple[str, bool]:
        """训练一个样本"""
        self.train_count += 1
        
        state = self.extractor.extract(image)
        
        # 预测
        predicted, confidence = self.root.predict(state)
        correct = (predicted == true_label)
        
        if correct:
            self.correct_count += 1
        
        # 学习
        status = self.root.learn(state, true_label, reward)
        
        return status, correct
    
    def print_stats(self):
        """打印统计信息"""
        print(f"\n=== Structon Vision System ===")
        print(f"训练样本: {self.train_count}")
        if self.train_count > 0:
            print(f"训练准确率: {self.correct_count/self.train_count*100:.1f}%")
        print(f"节点数: {self.root.count_nodes()}")
        print(f"深度: {self.root.depth()}")
        print(f"总记忆: {self.root.total_memories()}")
        
        print(f"\n=== 树结构 ===")
        self.root.print_tree()


# =============================================================================
# 6. 实验
# =============================================================================

def run_experiment(
    n_per_class: int = 50,
    n_test: int = 500,
    capacity: int = 100,
    key_dim: int = 16,
    temperature: float = 0.1,
    lr: float = 0.3,
    similarity_threshold: float = 0.85
):
    """运行实验"""
    print("=" * 70)
    print("Structon Vision v7.20 - 统一分形架构")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  capacity={capacity}, key_dim={key_dim}")
    print(f"  temperature={temperature}, lr={lr}")
    print(f"  similarity_threshold={similarity_threshold}")
    print(f"  每类训练: {n_per_class}, 测试: {n_test}")
    
    print("\n核心改变:")
    print("  1. 统一 Structon：每个节点结构相同")
    print("  2. LRM 存 Q-values：长度 = n_children = 10")
    print("  3. 混合记忆：同一 LRM 存多类别路由决策")
    print("  4. 相似 pattern → 相似 Q-values（共振）")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        capacity=capacity,
        key_dim=key_dim,
        full_dim=25,
        temperature=temperature,
        lr=lr,
        similarity_threshold=similarity_threshold
    )
    
    print(f"\n=== 训练（混合顺序）===")
    t0 = time.time()
    
    # 收集训练样本（每类 n_per_class 个）
    train_indices = []
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        train_indices.extend(indices)
    
    # 打乱顺序
    np.random.shuffle(train_indices)
    
    total_samples = len(train_indices)
    correct = 0
    
    for i, idx in enumerate(train_indices):
        status, is_correct = system.train_one(
            train_images[idx],
            str(train_labels[idx])
        )
        if is_correct:
            correct += 1
        
        if (i + 1) % 100 == 0:
            print(f"  训练 {i+1}/{total_samples}, "
                  f"准确率: {correct/(i+1)*100:.1f}%, "
                  f"记忆: {system.root.lrm.size}/{capacity}")
    
    print(f"\n训练完成: {time.time()-t0:.1f}秒")
    
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
    parser.add_argument('--per-class', type=int, default=50)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=100)
    parser.add_argument('--key-dim', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--threshold', type=float, default=0.85)
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        key_dim=args.key_dim,
        temperature=args.temperature,
        lr=args.lr,
        similarity_threshold=args.threshold
    )
