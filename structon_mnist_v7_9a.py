"""
Structon Vision v7.9 - Reward-Based LRM with Promote
=====================================================

正确的 Structon 机制：

1. 查询：pattern → LRM → 最匹配的记忆 → 输出 action
2. 执行：action 被环境评估
3. 奖励：环境返回 reward (+1/-1)
4. 更新：
   - 正奖励 + 有匹配 → 强化该记忆
   - 负奖励 + 有匹配 → 弱化该记忆
   - 正奖励 + 无匹配 → 添加新记忆
5. 晋升：记忆满了 → 冻结 → 创建 sibling → 包裹

记忆不直接存标签，而是：
- 存 pattern → action 映射
- 通过 reward 调整记忆强度
- 强的记忆更容易被激活
- 弱的记忆可能被遗忘

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import copy
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Memory Entry - 存储 pattern → action 映射
# =============================================================================

@dataclass
class MemoryEntry:
    """
    LRM 中的一条记忆
    
    不是 pattern → label
    而是 pattern → action (response)
    strength 由 reward 调节
    """
    pattern: np.ndarray       # 输入特征
    action: Any               # 输出动作/响应
    strength: float = 1.0     # 记忆强度 (reward 调节)
    access_count: int = 0     # 访问次数
    success_count: int = 0    # 成功次数 (正奖励)
    
    def reinforce(self, reward: float):
        """根据 reward 调整强度"""
        self.access_count += 1
        if reward > 0:
            self.success_count += 1
            self.strength = min(2.0, self.strength + 0.1 * reward)
        else:
            self.strength = max(0.1, self.strength + 0.1 * reward)  # reward 是负的
    
    @property
    def success_rate(self) -> float:
        return self.success_count / self.access_count if self.access_count > 0 else 0.0


# =============================================================================
# 2. Local Resonant Memory (LRM) - Reward-Based
# =============================================================================

class LRM:
    """
    局部共振记忆 - 基于奖励
    
    - pattern → action 映射
    - 通过 reward 调整 strength
    - strength 影响匹配分数
    """
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.8):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.entries: List[MemoryEntry] = []
        self.frozen = False
        
        # 统计
        self.query_count = 0
        self.hit_count = 0
    
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity
    
    def size(self) -> int:
        return len(self.entries)
    
    def freeze(self):
        self.frozen = True
    
    def query(self, pattern: np.ndarray) -> Tuple[Optional[int], float]:
        """
        查询最佳匹配
        
        分数 = similarity * strength
        
        Returns:
            (index, score) 或 (None, best_score)
        """
        self.query_count += 1
        
        if not self.entries:
            return None, 0.0
        
        best_idx = None
        best_score = -1.0
        
        for i, entry in enumerate(self.entries):
            sim = self._cosine(pattern, entry.pattern)
            score = sim * entry.strength  # strength 加权
            
            if score > best_score:
                best_score = score
                best_idx = i
        
        # 检查是否超过阈值
        if best_score >= self.similarity_threshold:
            self.hit_count += 1
            return best_idx, best_score
        
        return None, best_score
    
    def get_action(self, idx: int) -> Optional[Any]:
        """获取记忆的 action"""
        if 0 <= idx < len(self.entries):
            return self.entries[idx].action
        return None
    
    def reinforce(self, idx: int, reward: float):
        """用 reward 强化/弱化记忆"""
        if 0 <= idx < len(self.entries):
            self.entries[idx].reinforce(reward)
    
    def add(self, pattern: np.ndarray, action: Any) -> int:
        """添加新记忆"""
        if self.frozen:
            return -1
        
        entry = MemoryEntry(pattern=pattern.copy(), action=action)
        self.entries.append(entry)
        return len(self.entries) - 1
    
    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    def hit_rate(self) -> float:
        return self.hit_count / self.query_count if self.query_count > 0 else 0.0


# =============================================================================
# 3. Structon - 分形单元
# =============================================================================

class Structon:
    """
    Structon - 分形智能单元
    
    流程：
    1. query(pattern) → action
    2. 环境评估 → reward
    3. learn(pattern, action, reward) → 更新记忆
    4. 记忆满 → promote()
    """
    
    _id_counter = 0
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.8):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        
        self.lrm = LRM(capacity=capacity, similarity_threshold=similarity_threshold)
        self.children: List['Structon'] = []
        
        # 配置
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        
        # 统计
        self.total_queries = 0
        self.local_hits = 0
        self.promotions = 0
    
    @property
    def frozen(self) -> bool:
        return self.lrm.frozen
    
    def freeze(self):
        self.lrm.freeze()
    
    def query(self, pattern: np.ndarray) -> Tuple[Optional[Any], float, 'Structon']:
        """
        查询 - 返回 action
        
        1. 本层 LRM 查询
        2. 有匹配 → 返回 action
        3. 无匹配 → 路由到子节点
        
        Returns:
            (action, score, responding_structon)
        """
        self.total_queries += 1
        
        # 本层查询
        idx, score = self.lrm.query(pattern)
        
        if idx is not None:
            self.local_hits += 1
            action = self.lrm.get_action(idx)
            return action, score, self
        
        # 路由到子节点
        if self.children:
            best_action = None
            best_score = -1.0
            best_structon = None
            
            for child in self.children:
                action, child_score, responding = child.query(pattern)
                if child_score > best_score:
                    best_score = child_score
                    best_action = action
                    best_structon = responding
            
            if best_action is not None:
                return best_action, best_score, best_structon
        
        # 没有匹配
        return None, score, self
    
    def learn(self, pattern: np.ndarray, action: Any, reward: float) -> Tuple[str, 'Structon']:
        """
        学习 - 基于 reward
        
        路由逻辑：
        1. 如果我有 children（是 wrapper）：
           - 先查 frozen children 是否认识
           - 认识 → 更新 frozen 的 strength
           - 不认识 → 给 active sibling 学
        2. 如果我是叶子：
           - 查本层 LRM
           - 有匹配 → reward 调整
           - 无匹配 + 正 reward → 添加
           - 满了 → promote
        
        Returns:
            (status, current_root)
        """
        # Case 1: 我是 wrapper (有 children)
        if self.children:
            # 先查所有 frozen children
            for child in self.children:
                if child.frozen:
                    idx, score = child.lrm.query(pattern)
                    if idx is not None:
                        # frozen child 认识这个 pattern
                        child.lrm.reinforce(idx, reward)
                        return "reinforced_frozen", self
            
            # frozen 不认识，交给 active sibling
            for child in self.children:
                if not child.frozen:
                    return child.learn(pattern, action, reward)
            
            # 没有 active child？不应该发生
            return "no_active_child", self
        
        # Case 2: 我是叶子
        if self.frozen:
            # 我冻结了但没有 children？不应该发生
            return "frozen_leaf", self
        
        # 本层查询
        idx, score = self.lrm.query(pattern)
        
        if idx is not None:
            # 有匹配 → 用 reward 调整
            self.lrm.reinforce(idx, reward)
            return "reinforced", self
        
        # 无匹配
        if reward > 0:
            # 正奖励 → 添加新记忆
            if not self.lrm.is_full():
                self.lrm.add(pattern, action)
                return "added", self
            else:
                # 满了 → promote
                new_root = self.promote()
                # 在新的 sibling 中添加 (sibling 是空的!)
                for child in new_root.children:
                    if not child.frozen:
                        child.lrm.add(pattern, action)
                        break
                return "promoted_and_added", new_root
        
        # 负奖励 + 无匹配 → 没什么可做的
        return "no_match_negative", self
    
    def promote(self) -> 'Structon':
        """
        晋升：冻结 → 创建 sibling → 包裹
        
        关键：sibling 是空的，不是复制！
        - frozen node: 保留已学知识
        - sibling: 学习新的、不同的 pattern
        
        这样才能形成分工：
        - frozen 处理"见过的"
        - sibling 处理"新的"
        """
        self.promotions += 1
        
        # 1. 冻结自己
        self.freeze()
        
        # 2. 创建 sibling（空的，学习新 pattern）
        sibling = Structon(
            capacity=self.capacity,
            similarity_threshold=self.similarity_threshold
        )
        # sibling 是空的！不复制记忆
        
        # 3. 创建 wrapper
        wrapper = Structon(
            capacity=self.capacity,
            similarity_threshold=self.similarity_threshold
        )
        
        # 4. 建立关系
        wrapper.children = [self, sibling]
        
        return wrapper
    
    def should_promote(self) -> bool:
        """检查是否应该 promote"""
        return self.lrm.is_full() and not self.frozen
    
    # =========================================================================
    # 统计和可视化
    # =========================================================================
    
    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)
    
    def total_entries(self) -> int:
        count = self.lrm.size()
        for child in self.children:
            count += child.total_entries()
        return count
    
    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def hit_rate(self) -> float:
        return self.local_hits / self.total_queries if self.total_queries > 0 else 0.0
    
    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        icon = "❄️" if self.frozen else "🔥"
        
        mem_info = f"mem:{self.lrm.size()}/{self.capacity}"
        hit_info = f"hit:{self.hit_rate()*100:.0f}%"
        children_info = f"children:{len(self.children)}" if self.children else ""
        
        print(f"{prefix}{icon} {self.id} ({mem_info}, {hit_info}) {children_info}")
        
        # 打印记忆条目
        for i, entry in enumerate(self.lrm.entries[:5]):  # 只显示前5条
            action_str = str(entry.action)[:10]
            print(f"{prefix}  └─ {i}: [{action_str}] str={entry.strength:.2f} "
                  f"acc={entry.access_count} suc={entry.success_count}")
        if self.lrm.size() > 5:
            print(f"{prefix}  └─ ... ({self.lrm.size() - 5} more)")
        
        # 递归打印子节点
        for child in self.children:
            child.print_tree(indent + 1)


# =============================================================================
# 4. Feature Extractor
# =============================================================================

class FeatureExtractor:
    """提取图像特征为向量"""
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """提取特征向量"""
        features = []
        h, w = image.shape
        
        # 1. 全局特征
        features.append(np.mean(image))  # 平均亮度
        features.append(np.std(image))   # 对比度
        
        # 2. 区域特征 (3x3 grid)
        for i in range(3):
            for j in range(3):
                y1, y2 = i * h // 3, (i + 1) * h // 3
                x1, x2 = j * w // 3, (j + 1) * w // 3
                region = image[y1:y2, x1:x2]
                features.append(np.mean(region))
                features.append(np.std(region))
        
        # 3. 拓扑特征
        binary = (image > 0.3).astype(np.uint8)
        n_holes = self._count_holes(binary)
        features.append(n_holes / 3.0)  # 归一化
        
        # 4. 边缘特征
        skeleton = self._simple_skeleton(binary)
        n_endpoints = self._count_endpoints(skeleton)
        features.append(n_endpoints / 10.0)  # 归一化
        
        # 5. 方向特征
        gy = np.abs(image[1:, :] - image[:-1, :]).mean()
        gx = np.abs(image[:, 1:] - image[:, :-1]).mean()
        features.append(gy)
        features.append(gx)
        
        # 6. 对称性
        h_sym = np.mean(np.abs(image - np.fliplr(image)))
        v_sym = np.mean(np.abs(image - np.flipud(image)))
        features.append(1.0 - h_sym)
        features.append(1.0 - v_sym)
        
        return np.array(features, dtype=np.float32)
    
    def _count_holes(self, binary: np.ndarray) -> int:
        h, w = binary.shape
        padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
        padded[1:-1, 1:-1] = binary
        
        visited = np.zeros_like(padded, dtype=bool)
        
        # Flood fill from border
        queue = [(0, 0)]
        visited[0, 0] = True
        while queue:
            y, x = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                    if padded[ny, nx] == 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        
        # Count internal holes
        n_holes = 0
        for y in range(1, h + 1):
            for x in range(1, w + 1):
                if padded[y, x] == 0 and not visited[y, x]:
                    # Found a hole, flood fill it
                    queue = [(y, x)]
                    visited[y, x] = True
                    while queue:
                        cy, cx = queue.pop(0)
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                                if padded[ny, nx] == 0 and not visited[ny, nx]:
                                    visited[ny, nx] = True
                                    queue.append((ny, nx))
                    n_holes += 1
        
        return n_holes
    
    def _simple_skeleton(self, binary: np.ndarray) -> np.ndarray:
        """简化的骨架化"""
        return binary  # 简化版，直接用 binary
    
    def _count_endpoints(self, skeleton: np.ndarray) -> int:
        """计算端点数"""
        h, w = skeleton.shape
        count = 0
        
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 0:
                    continue
                
                neighbors = (
                    skeleton[y-1, x-1] + skeleton[y-1, x] + skeleton[y-1, x+1] +
                    skeleton[y, x-1] + skeleton[y, x+1] +
                    skeleton[y+1, x-1] + skeleton[y+1, x] + skeleton[y+1, x+1]
                )
                
                if neighbors == 1:
                    count += 1
        
        return count


# =============================================================================
# 5. Vision System with Structon
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统
    
    流程：
    1. 提取特征
    2. 查询 Structon → 得到预测
    3. 比较真实标签 → 计算 reward
    4. 用 reward 更新 Structon
    """
    
    def __init__(self, capacity: int = 10, similarity_threshold: float = 0.75):
        self.extractor = FeatureExtractor()
        self.root = Structon(capacity=capacity, similarity_threshold=similarity_threshold)
        
        # 统计
        self.train_count = 0
        self.correct_count = 0
    
    def predict(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """预测"""
        features = self.extractor.extract(image)
        action, score, _ = self.root.query(features)
        return action, score
    
    def train_one(self, image: np.ndarray, true_label: str) -> Tuple[str, bool]:
        """
        训练一个样本
        
        1. 提取特征
        2. 查询 → 预测
        3. 计算 reward
        4. 学习
        
        Returns:
            (status, correct)
        """
        self.train_count += 1
        
        # 提取特征
        features = self.extractor.extract(image)
        
        # 查询
        predicted, score, _ = self.root.query(features)
        
        # 计算 reward
        if predicted == true_label:
            reward = 1.0
            correct = True
            self.correct_count += 1
        elif predicted is None:
            # 没有预测，给正奖励让它学习
            reward = 1.0
            correct = False
        else:
            # 预测错误
            reward = -1.0
            correct = False
        
        # 学习
        status, new_root = self.root.learn(features, true_label, reward)
        
        # 如果发生了 promote，更新 root
        if new_root != self.root:
            self.root = new_root
        
        return status, correct
    
    def train_accuracy(self) -> float:
        return self.correct_count / self.train_count if self.train_count > 0 else 0.0
    
    def print_stats(self):
        print(f"\n=== Structon Vision System ===")
        print(f"训练样本: {self.train_count}")
        print(f"训练准确率: {self.train_accuracy()*100:.1f}%")
        print(f"树深度: {self.root.depth()}")
        print(f"总节点数: {self.root.count_nodes()}")
        print(f"总记忆条目: {self.root.total_entries()}")
        print(f"\n=== 树结构 ===")
        self.root.print_tree()


# =============================================================================
# 6. MNIST Experiment
# =============================================================================

def load_mnist(data_dir='./mnist_data'):
    os.makedirs(data_dir, exist_ok=True)
    mirrors = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
               "https://ossci-datasets.s3.amazonaws.com/mnist/"]
    files = {'train_images': 'train-images-idx3-ubyte.gz',
             'train_labels': 'train-labels-idx1-ubyte.gz',
             'test_images': 't10k-images-idx3-ubyte.gz',
             'test_labels': 't10k-labels-idx1-ubyte.gz'}
    
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            for mirror in mirrors:
                try:
                    urllib.request.urlretrieve(mirror + filename, filepath)
                    break
                except:
                    continue
    
    def load_images(path):
        with gzip.open(path, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c).astype(np.float32) / 255
    
    def load_labels(path):
        with gzip.open(path, 'rb') as f:
            struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
    
    return (load_images(os.path.join(data_dir, files['train_images'])),
            load_labels(os.path.join(data_dir, files['train_labels'])),
            load_images(os.path.join(data_dir, files['test_images'])),
            load_labels(os.path.join(data_dir, files['test_labels'])))


def run_experiment(n_train=100, n_test=500, capacity=10, verbose=True):
    """
    增量学习实验
    
    - 一个样本一个样本训练
    - 观察 promote 何时发生
    - 测试泛化能力
    """
    print("=" * 70)
    print("Structon Vision v7.9 - Reward-Based Incremental Learning")
    print("=" * 70)
    print(f"\n参数: capacity={capacity}, n_train={n_train}, n_test={n_test}")
    print("\n机制:")
    print("  1. 查询 → 预测")
    print("  2. 比较真实标签 → reward (+1/-1)")
    print("  3. reward 调整记忆强度")
    print("  4. 记忆满 → promote (冻结 + 包裹)")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    # 创建系统
    system = StructonVisionSystem(capacity=capacity, similarity_threshold=0.75)
    
    print(f"\n增量训练 {n_train} 个样本...")
    
    # 随机选择训练样本
    train_indices = np.random.choice(len(train_images), n_train, replace=False)
    
    t0 = time.time()
    promote_events = []
    
    for i, idx in enumerate(train_indices):
        image = train_images[idx]
        label = str(train_labels[idx])
        
        status, correct = system.train_one(image, label)
        
        if "promoted" in status:
            promote_events.append((i, system.root.depth()))
            if verbose:
                print(f"  [{i}] PROMOTE! depth={system.root.depth()}, nodes={system.root.count_nodes()}")
        
        if verbose and (i + 1) % 20 == 0:
            print(f"  [{i+1}/{n_train}] 训练准确率: {system.train_accuracy()*100:.1f}%")
    
    train_time = time.time() - t0
    print(f"\n训练完成: {train_time:.1f}秒")
    print(f"Promote 次数: {len(promote_events)}")
    
    # 打印系统状态
    system.print_stats()
    
    # 测试
    print(f"\n测试 {n_test} 个样本...")
    test_indices = np.random.choice(len(test_images), n_test, replace=False)
    
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    
    t0 = time.time()
    for idx in test_indices:
        image = test_images[idx]
        true_label = str(test_labels[idx])
        
        predicted, _ = system.predict(image)
        
        results[true_label]['total'] += 1
        if predicted == true_label:
            results[true_label]['correct'] += 1
    
    test_time = time.time() - t0
    
    total_correct = sum(r['correct'] for r in results.values())
    total_samples = sum(r['total'] for r in results.values())
    
    print(f"\n测试完成: {test_time:.1f}秒")
    print(f"\n总准确率: {total_correct/total_samples*100:.1f}%")
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  {d}: {acc:.1f}% ({r['correct']}/{r['total']})")
    
    return system


def demo_incremental():
    """演示增量学习过程"""
    print("\n=== 增量学习演示 ===")
    print("观察 Structon 如何一步步学习和晋升\n")
    
    train_images, train_labels, _, _ = load_mnist()
    
    system = StructonVisionSystem(capacity=5, similarity_threshold=0.7)
    
    # 只用少量样本，详细展示过程
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:3]
        for idx in indices:
            image = train_images[idx]
            label = str(train_labels[idx])
            
            status, correct = system.train_one(image, label)
            
            print(f"样本 {system.train_count}: 数字={label}, status={status}, "
                  f"correct={correct}, depth={system.root.depth()}")
    
    print("\n最终结构:")
    system.root.print_tree()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=10)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    
    if args.demo:
        demo_incremental()
    else:
        run_experiment(args.train, args.test, args.capacity)
