#!/usr/bin/env python3
"""
Structon Vision v9.7 - 多局部 Structon

改变：
- 每个数字有多个 Structon（比如 10 个）
- 每个 Structon 只看部分特征
- 投票时，同一 label 的所有 Structon 加总

保持 v9.6 的核心逻辑：
- YES/NO 二元判断
- 禁忌路由
- 简单的监督学习

作者: Boe
日期: 2025
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


def generate_synthetic_digit(digit: int, noise: float = 0.15) -> np.ndarray:
    """生成合成数字图像"""
    img = np.zeros((28, 28), dtype=np.float32)
    
    patterns = {
        0: [(10, 14, 8, 20), (10, 14, 18, 8)],
        1: [(14, 14, 8, 20)],
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
    
    img += np.random.randn(28, 28) * noise
    img = np.clip(img, 0, 1)
    
    shift_y = np.random.randint(-2, 3)
    shift_x = np.random.randint(-2, 3)
    img = np.roll(np.roll(img, shift_y, axis=0), shift_x, axis=1)
    
    return (img * 255).astype(np.uint8)


def load_synthetic_data(n_train_per_class: int = 500, n_test_per_class: int = 100):
    """生成合成数据集"""
    print("Generating synthetic data...")
    
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    for digit in range(10):
        for _ in range(n_train_per_class):
            train_images.append(generate_synthetic_digit(digit))
            train_labels.append(digit)
        for _ in range(n_test_per_class):
            test_images.append(generate_synthetic_digit(digit, noise=0.2))
            test_labels.append(digit)
    
    return (np.array(train_images), np.array(train_labels),
            np.array(test_images), np.array(test_labels))


def try_load_mnist_or_synthetic():
    """尝试加载 MNIST，失败则用合成数据"""
    try:
        return load_mnist()
    except Exception as e:
        print(f"Cannot load MNIST: {e}")
        print("Using synthetic data instead...")
        return load_synthetic_data()


# =============================================================================
# 2. 特征提取
# =============================================================================
class StateExtractor:
    """特征提取器：Contrast Binary + Structure"""
    
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        self._feature_dim = None
        
    @property
    def feature_dim(self) -> int:
        if self._feature_dim is None:
            dummy = np.zeros((28, 28), dtype=np.uint8)
            self._feature_dim = len(self.extract(dummy))
        return self._feature_dim
        
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
        
        # 2. Contrast Binary
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                features.append(-1.0 if grid[i, j] < self.threshold else 1.0)
        
        # 3. 水平差异
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                features.append((grid[i, j] - grid[i, j+1]) * 2)
        
        # 4. 垂直差异
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                features.append((grid[i, j] - grid[i+1, j]) * 2)
        
        # 5. 对角线
        diag1 = np.mean([grid[i, i] for i in range(self.grid_size)])
        diag2 = np.mean([grid[i, self.grid_size-1-i] for i in range(self.grid_size)])
        features.extend([(diag1 - 0.5) * 2, (diag2 - 0.5) * 2, (diag1 - diag2) * 2])
        
        # 6. 四象限
        mid = self.grid_size // 2
        q1, q2 = np.mean(grid[:mid, :mid]), np.mean(grid[:mid, mid:])
        q3, q4 = np.mean(grid[mid:, :mid]), np.mean(grid[mid:, mid:])
        features.extend([
            (q1 - q4) * 2, (q2 - q3) * 2,
            (q1 + q4 - q2 - q3) * 2, ((q1 + q2) - (q3 + q4)) * 2
        ])
        
        # 7. 边缘
        top, bottom = np.mean(grid[0, :]), np.mean(grid[-1, :])
        left, right = np.mean(grid[:, 0]), np.mean(grid[:, -1])
        features.extend([
            (top - bottom) * 2, (left - right) * 2,
            (top - 0.5) * 2, (bottom - 0.5) * 2
        ])
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        
        return state


# =============================================================================
# 3. LRM
# =============================================================================
class LRM:
    """Local Resonant Memory"""
    
    def __init__(self, state_dim: int, capacity: int = 100, top_k: int = 5):
        self.state_dim = state_dim
        self.n_actions = 2  # NO, YES
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88

    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """查询"""
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
        
        scores = np.array([np.dot(k, state) for k in self.keys])
        
        k = min(self.top_k, len(scores))
        top_indices = np.argsort(scores)[-k:]
        top_scores = scores[top_indices]
        
        valid_mask = top_scores > 0.1
        if not np.any(valid_mask):
            return np.zeros(self.n_actions), 0.0
        
        top_indices = top_indices[valid_mask]
        top_scores = top_scores[valid_mask]
        
        weights = top_scores ** 2
        weights /= np.sum(weights) + 1e-8
        
        q_values = np.zeros(self.n_actions)
        for idx, w in zip(top_indices, weights):
            q_values += w * self.values[idx]
        
        return q_values, float(np.max(top_scores))

    def update(self, state: np.ndarray, action: int, reward: float):
        """更新"""
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
# 4. Structon - 局部特征版本
# =============================================================================
class LocalStructon:
    """局部 Structon - 只看部分特征"""
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        feature_indices: np.ndarray,
        capacity: int = 100,
        n_connections: int = 3
    ):
        LocalStructon._id_counter += 1
        self.id = f"S{LocalStructon._id_counter:03d}"
        self.label = label
        self.feature_indices = feature_indices
        self.state_dim = len(feature_indices)
        self.n_connections = n_connections
        self.connections: List['LocalStructon'] = []
        
        self.lrm = LRM(state_dim=self.state_dim, capacity=capacity)
        
    def set_connections(self, all_structons: List['LocalStructon']):
        """设置稀疏连接（只连接不同 label 的）"""
        others = [s for s in all_structons if s.label != self.label]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(
                others, self.n_connections, replace=False
            ))
        else:
            self.connections = others
        
    def extract_local(self, full_state: np.ndarray) -> np.ndarray:
        """提取局部特征"""
        local = full_state[self.feature_indices]
        norm = np.linalg.norm(local)
        if norm > 1e-6:
            local = local / norm
        return local
        
    def decide(self, full_state: np.ndarray) -> Tuple[int, float]:
        """决策：是我的吗？"""
        local_state = self.extract_local(full_state)
        q, score = self.lrm.query(local_state)
        action = int(np.argmax(q))
        confidence = q[action] if q[action] != 0 else score
        return action, confidence
    
    def get_yes_score(self, full_state: np.ndarray) -> float:
        """获取 YES 的 Q 值（用于投票）"""
        local_state = self.extract_local(full_state)
        q, _ = self.lrm.query(local_state)
        return q[1]  # YES 的 Q 值

    def train_binary(self, full_state: np.ndarray, is_me: bool):
        """训练二元分类"""
        local_state = self.extract_local(full_state)
        if is_me:
            self.lrm.update(local_state, 1, 1.5)   # YES
            self.lrm.update(local_state, 0, -0.8)  # not NO
        else:
            self.lrm.update(local_state, 0, 0.8)   # NO
            self.lrm.update(local_state, 1, -0.5)  # not YES


# =============================================================================
# 5. Vision System v9.7
# =============================================================================
class StructonVisionSystem:
    """Structon Vision System v9.7 - 多局部 Structon"""
    
    def __init__(
        self,
        n_local_per_label: int = 10,
        capacity: int = 100,
        n_connections: int = 3,
        feature_mode: str = 'random'  # 'random', 'sequential', 'overlap'
    ):
        self.extractor = StateExtractor()
        self.n_local_per_label = n_local_per_label
        self.capacity = capacity
        self.n_connections = n_connections
        self.feature_mode = feature_mode
        
        self.structons: List[LocalStructon] = []
        self.label_to_structons: Dict[str, List[LocalStructon]] = {}
        
    def _generate_feature_masks(self, total_dim: int, n_masks: int) -> List[np.ndarray]:
        """生成特征掩码"""
        if self.feature_mode == 'sequential':
            # 顺序分割
            chunk_size = total_dim // n_masks
            masks = []
            for i in range(n_masks):
                start = i * chunk_size
                end = start + chunk_size if i < n_masks - 1 else total_dim
                masks.append(np.arange(start, end))
            return masks
            
        elif self.feature_mode == 'overlap':
            # 重叠分割（每个看 50% 的特征）
            subset_size = total_dim // 2
            masks = []
            for i in range(n_masks):
                start = (i * total_dim // n_masks) % total_dim
                indices = [(start + j) % total_dim for j in range(subset_size)]
                masks.append(np.array(sorted(indices)))
            return masks
            
        else:  # random
            # 随机分割
            subset_size = max(total_dim // n_masks, 10)
            masks = []
            for _ in range(n_masks):
                indices = np.random.choice(total_dim, subset_size, replace=False)
                masks.append(np.sort(indices))
            return masks
        
    def build(self, labels: List[str]):
        """构建 Structon 网络"""
        print(f"\n=== 创建 Structon v9.7 - 多局部版本 ===")
        print(f"每个数字 {self.n_local_per_label} 个局部 Structon")
        print(f"特征分配模式: {self.feature_mode}")
        
        LocalStructon._id_counter = 0
        
        self.structons = []
        self.label_to_structons = {label: [] for label in labels}
        
        # 获取特征维度
        feature_dim = self.extractor.feature_dim
        print(f"总特征维度: {feature_dim}")
        
        # 为每个 label 创建多个局部 Structon
        feature_masks = self._generate_feature_masks(
            feature_dim, self.n_local_per_label
        )
        
        for label in labels:
            for i, mask in enumerate(feature_masks):
                s = LocalStructon(
                    label=label,
                    feature_indices=mask,
                    capacity=self.capacity,
                    n_connections=self.n_connections
                )
                self.structons.append(s)
                self.label_to_structons[label].append(s)
        
        print(f"总 Structon 数: {len(self.structons)}")
        print(f"每个局部 Structon 看 {len(feature_masks[0])} 维特征")
        
        # 设置稀疏连接
        print("\n设置稀疏连接...")
        for s in self.structons:
            s.set_connections(self.structons)
            
    def train_epoch(self, samples: List[Tuple[np.ndarray, str]], n_negatives: int = 3):
        """训练一个 epoch"""
        np.random.shuffle(samples)
        
        for img, label in samples:
            state = self.extractor.extract(img)
            
            # 正样本：该 label 的所有 Structon
            for s in self.label_to_structons[label]:
                s.train_binary(state, is_me=True)
            
            # 负样本：随机选其他 label 的部分 Structon
            other_labels = [l for l in self.label_to_structons.keys() if l != label]
            neg_labels = np.random.choice(other_labels, min(n_negatives, len(other_labels)), replace=False)
            
            for neg_label in neg_labels:
                # 每个负 label 随机选一个 Structon 训练
                neg_s = np.random.choice(self.label_to_structons[neg_label])
                neg_s.train_binary(state, is_me=False)

    def predict_voting(self, image: np.ndarray) -> Tuple[str, Dict[str, float]]:
        """投票预测"""
        state = self.extractor.extract(image)
        
        # 每个 label 的所有 Structon 投票加总
        scores = {}
        for label, structons in self.label_to_structons.items():
            total_score = sum(s.get_yes_score(state) for s in structons)
            scores[label] = total_score
        
        best = max(scores, key=scores.get)
        return best, scores
    
    def predict_routing(self, image: np.ndarray, max_hops: int = 30) -> Tuple[str, List[str]]:
        """禁忌路由预测"""
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        
        for _ in range(max_hops):
            visited.add(current.id)
            path.append(f"{current.label}")
            
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
                else:
                    break
        
        return current.label, path
    
    def evaluate(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 500,
        method: str = 'voting'
    ) -> Dict:
        """评估"""
        indices = np.random.choice(len(images), min(n_samples, len(images)), replace=False)
        
        correct = 0
        per_class = {str(i): {'correct': 0, 'total': 0} for i in range(10)}
        
        for idx in indices:
            true_label = str(labels[idx])
            per_class[true_label]['total'] += 1
            
            if method == 'voting':
                pred, _ = self.predict_voting(images[idx])
            else:
                pred, _ = self.predict_routing(images[idx])
            
            if pred == true_label:
                correct += 1
                per_class[true_label]['correct'] += 1
        
        return {
            'accuracy': correct / len(indices) * 100,
            'per_class': per_class,
        }
    
    def print_stats(self):
        """打印统计"""
        print("\n" + "=" * 60)
        print("Structon Vision v9.7 - 多局部 Structon")
        print("=" * 60)
        print(f"总 Structon 数: {len(self.structons)}")
        print(f"每个 label 的 Structon 数: {self.n_local_per_label}")
        print(f"特征分配模式: {self.feature_mode}")
        
        total_mem = sum(s.lrm.size for s in self.structons)
        print(f"总记忆数: {total_mem}")
        
        # 每个 label 的统计
        print("\n各 label 记忆分布:")
        for label in sorted(self.label_to_structons.keys()):
            structons = self.label_to_structons[label]
            mems = [s.lrm.size for s in structons]
            print(f"  {label}: {sum(mems)} 条记忆 (avg={np.mean(mems):.1f})")


# =============================================================================
# 6. 实验
# =============================================================================
def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    n_local: int = 10,
    capacity: int = 100,
    epochs: int = 30,
    feature_mode: str = 'random'
):
    """运行实验"""
    print("=" * 70)
    print("Structon Vision v9.7 - 多局部 Structon")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  - 每个 label 的 Structon 数: {n_local}")
    print(f"  - 每个 Structon 容量: {capacity}")
    print(f"  - 特征分配: {feature_mode}")
    print(f"  - 每类训练样本: {n_per_class}")
    
    print("\nLoading data...")
    train_images, train_labels, test_images, test_labels = try_load_mnist_or_synthetic()
    
    # 创建系统
    system = StructonVisionSystem(
        n_local_per_label=n_local,
        capacity=capacity,
        feature_mode=feature_mode
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
    
    for ep in range(epochs):
        system.train_epoch(samples)
        
        if (ep + 1) % 5 == 0:
            result = system.evaluate(test_images, test_labels, n_samples=200)
            print(f"  轮次 {ep+1:2d}: 准确率={result['accuracy']:.1f}%")
    
    print(f"\n训练时间: {time.time()-t0:.1f}秒")
    system.print_stats()
    
    # 测试
    print(f"\n=== 测试（投票）===")
    result = system.evaluate(test_images, test_labels, n_test, method='voting')
    print(f"准确率: {result['accuracy']:.1f}%")
    
    print("\n各数字准确率:")
    for d in range(10):
        s = result['per_class'][str(d)]
        acc = s['correct']/s['total']*100 if s['total'] > 0 else 0
        print(f"  {d}: {acc:>6.1f}%")
    
    return system


def compare_configurations():
    """对比不同配置"""
    print("=" * 70)
    print("对比实验：不同局部 Structon 配置")
    print("=" * 70)
    
    print("\nLoading data...")
    train_images, train_labels, test_images, test_labels = try_load_mnist_or_synthetic()
    
    # 准备样本
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:100]
        for i in idxs:
            samples.append((train_images[i], str(d)))
    
    configs = [
        {'n_local': 1, 'feature_mode': 'sequential', 'name': 'v9.6 基线 (1个/label)'},
        {'n_local': 5, 'feature_mode': 'sequential', 'name': '5个/label 顺序'},
        {'n_local': 10, 'feature_mode': 'sequential', 'name': '10个/label 顺序'},
        {'n_local': 10, 'feature_mode': 'random', 'name': '10个/label 随机'},
        {'n_local': 10, 'feature_mode': 'overlap', 'name': '10个/label 重叠'},
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        
        system = StructonVisionSystem(
            n_local_per_label=cfg['n_local'],
            capacity=100,
            feature_mode=cfg['feature_mode']
        )
        system.build([str(i) for i in range(10)])
        
        t0 = time.time()
        for _ in range(30):
            system.train_epoch(samples)
        train_time = time.time() - t0
        
        result = system.evaluate(test_images, test_labels, 500)
        total_mem = sum(s.lrm.size for s in system.structons)
        
        results.append({
            'name': cfg['name'],
            'accuracy': result['accuracy'],
            'time': train_time,
            'memories': total_mem,
            'n_structons': len(system.structons)
        })
        
        print(f"准确率: {result['accuracy']:.1f}%")
    
    # 汇总
    print("\n" + "=" * 70)
    print("对比汇总")
    print("=" * 70)
    print(f"{'配置':<25} {'准确率':<10} {'时间':<10} {'Structon数':<12} {'记忆数':<10}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['accuracy']:>6.1f}%   {r['time']:>6.1f}s   {r['n_structons']:>8}     {r['memories']:>8}")
    
    return results


# =============================================================================
# 7. 主函数
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Structon Vision v9.7')
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--n-local', type=int, default=10)
    parser.add_argument('--capacity', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--feature-mode', type=str, default='random',
                        choices=['random', 'sequential', 'overlap'])
    parser.add_argument('--compare', action='store_true')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_configurations()
    else:
        run_experiment(
            n_per_class=args.per_class,
            n_test=args.test,
            n_local=args.n_local,
            capacity=args.capacity,
            epochs=args.epochs,
            feature_mode=args.feature_mode
        )
