#!/usr/bin/env python3
"""
Structon Vision v9.15 - Scale Test (N=100)

扩大规模测试：
- 100 个 Structon
- 每个数字 10 个 Structon（数字的不同变体）
- 测试投票 vs 路由在大规模下的表现

预期：
- 投票: O(100) 查询
- 路由: O(log 100) ≈ O(5-7) 步

基于 v9.12 的纯本地学习
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Optional
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
# 2. 特征提取
# =============================================================================
class StateExtractor:
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        
        features = []
        
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                block = img[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
                grid[i, j] = np.mean(block)
        
        # Contrast Binary
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                features.append(-1.0 if grid[i, j] < self.threshold else 1.0)
        
        # 水平差异
        for i in range(self.grid_size):
            for j in range(self.grid_size - 1):
                features.append((grid[i, j] - grid[i, j+1]) * 2)
        
        # 垂直差异
        for i in range(self.grid_size - 1):
            for j in range(self.grid_size):
                features.append((grid[i, j] - grid[i+1, j]) * 2)
        
        # 对角线
        diag1 = np.mean([grid[i, i] for i in range(self.grid_size)])
        diag2 = np.mean([grid[i, self.grid_size-1-i] for i in range(self.grid_size)])
        features.extend([(diag1 - 0.5) * 2, (diag2 - 0.5) * 2, (diag1 - diag2) * 2])
        
        # 四象限
        mid = self.grid_size // 2
        q1, q2 = np.mean(grid[:mid, :mid]), np.mean(grid[:mid, mid:])
        q3, q4 = np.mean(grid[mid:, :mid]), np.mean(grid[mid:, mid:])
        features.extend([(q1-q4)*2, (q2-q3)*2, (q1+q4-q2-q3)*2, ((q1+q2)-(q3+q4))*2])
        
        # 边缘
        top, bottom = np.mean(grid[0, :]), np.mean(grid[-1, :])
        left, right = np.mean(grid[:, 0]), np.mean(grid[:, -1])
        features.extend([(top-bottom)*2, (left-right)*2, (top-0.5)*2, (bottom-0.5)*2])
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm
        return state

# =============================================================================
# 3. Unified Routing LRM
# =============================================================================
class UnifiedRoutingLRM:
    def __init__(self, n_connections: int, capacity=300, top_k=5):
        self.n_actions = 1 + n_connections
        self.n_connections = n_connections
        self.capacity = capacity
        self.top_k = top_k
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.learning_rate = 0.3
        self.similarity_threshold = 0.88
        self.SELF = 0

    def query(self, state) -> Tuple[np.ndarray, float]:
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
        weights /= np.sum(weights)
        
        q_values = np.zeros(self.n_actions)
        for idx, w in zip(top_indices, weights):
            q_values += w * self.values[idx]
        
        return q_values, float(np.max(top_scores))

    def update(self, state, action: int, reward: float):
        if action >= self.n_actions:
            return
            
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
    _id_counter = 0
    
    def __init__(self, label: str, variant_id: int, capacity=300, n_connections=5):
        Structon._id_counter += 1
        self.id = f"S{Structon._id_counter:03d}"
        self.label = label  # 数字 "0"-"9"
        self.variant_id = variant_id  # 变体 0-9
        self.full_label = f"{label}_{variant_id}"
        self.n_connections = n_connections
        self.connections: List['Structon'] = []
        
        self.routing_lrm: Optional[UnifiedRoutingLRM] = None
        self.confidence_threshold = 0.5
        
        self.stats = {
            'self_confident': 0,
            'self_uncertain': 0,
            'route_accepted': 0,
            'route_continued': 0,
            'route_deadend': 0,
            'route_revisit': 0,
        }
        
    def set_connections(self, all_structons):
        others = [s for s in all_structons if s.id != self.id]
        if len(others) >= self.n_connections:
            self.connections = list(np.random.choice(others, self.n_connections, replace=False))
        else:
            self.connections = others
        
        self.routing_lrm = UnifiedRoutingLRM(
            n_connections=len(self.connections),
            capacity=300,
            top_k=5
        )

    def get_q_values(self, state) -> Tuple[np.ndarray, float]:
        if self.routing_lrm is None:
            return np.zeros(1), 0.0
        return self.routing_lrm.query(state)

    def decide(self, state, visited_ids: set) -> Tuple[int, Optional['Structon'], float]:
        if self.routing_lrm is None:
            return 0, None, 0.0
        
        q_values, resonance = self.routing_lrm.query(state)
        
        adjusted_q = q_values.copy()
        for i, conn in enumerate(self.connections):
            if conn.id in visited_ids:
                adjusted_q[i + 1] = -100.0
        
        best_action = int(np.argmax(adjusted_q))
        confidence = adjusted_q[best_action]
        
        if best_action == 0:
            return 0, None, confidence
        else:
            return best_action, self.connections[best_action - 1], confidence

    def update_routing(self, state, action: int, reward: float):
        if self.routing_lrm is not None:
            self.routing_lrm.update(state, action, reward)

    # 本地学习方法
    def learn_self_confident(self, state, confidence: float):
        if confidence > self.confidence_threshold:
            self.routing_lrm.update(state, 0, 1.2)
            self.stats['self_confident'] += 1
        else:
            self.routing_lrm.update(state, 0, 0.3)
            self.stats['self_uncertain'] += 1
    
    def learn_route_accepted(self, state, action: int, downstream_confidence: float):
        reward = 0.5 + 0.5 * min(downstream_confidence / self.confidence_threshold, 1.0)
        self.routing_lrm.update(state, action, reward)
        self.stats['route_accepted'] += 1
    
    def learn_route_continued(self, state, action: int):
        self.routing_lrm.update(state, action, 0.2)
        self.stats['route_continued'] += 1
    
    def learn_route_deadend(self, state, action: int):
        self.routing_lrm.update(state, action, -0.8)
        self.stats['route_deadend'] += 1
    
    def learn_route_revisit(self, state, action: int):
        self.routing_lrm.update(state, action, -1.0)
        self.stats['route_revisit'] += 1

    def train_self_supervised(self, state, is_me: bool):
        if self.routing_lrm is None:
            return
            
        if is_me:
            self.routing_lrm.update(state, 0, 1.5)
            for i in range(1, self.routing_lrm.n_actions):
                self.routing_lrm.update(state, i, -0.5)
        else:
            self.routing_lrm.update(state, 0, -0.8)
            for i in range(1, self.routing_lrm.n_actions):
                self.routing_lrm.update(state, i, 0.3)

    @property
    def memory_size(self):
        return self.routing_lrm.size if self.routing_lrm else 0

# =============================================================================
# 5. Vision System - N=100
# =============================================================================
class StructonVisionSystem:
    def __init__(self, n_variants=10, capacity=300, n_connections=5):
        self.extractor = StateExtractor()
        self.structons: List[Structon] = []
        self.label_to_structons: Dict[str, List[Structon]] = {}  # 每个数字多个 Structon
        self.capacity = capacity
        self.n_connections = n_connections
        self.n_variants = n_variants
        
    def build(self, labels):
        n_total = len(labels) * self.n_variants
        print(f"\n=== 创建 Structon v9.15 (N={n_total}) ===")
        print(f"每个数字 {self.n_variants} 个 Structon")
        Structon._id_counter = 0
        
        self.structons = []
        self.label_to_structons = {label: [] for label in labels}
        
        for label in labels:
            for v in range(self.n_variants):
                s = Structon(label, v, self.capacity, self.n_connections)
                self.structons.append(s)
                self.label_to_structons[label].append(s)
        
        print(f"  总共 {len(self.structons)} 个 Structon")
            
        print(f"\n设置稀疏连接 (每个 {self.n_connections} 条)...")
        for s in self.structons:
            s.set_connections(self.structons)
        
        # 打印连接示例
        print("  连接示例:")
        for s in self.structons[:3]:
            conns = [c.full_label for c in s.connections[:3]]
            print(f"    {s.full_label} → {conns}...")
            
    def train_epoch(self, samples, n_negatives=5):
        """监督训练"""
        np.random.shuffle(samples)
        
        for img, label, variant_idx in samples:
            state = self.extractor.extract(img)
            
            # 正样本：训练对应的 Structon
            correct_s = self.label_to_structons[label][variant_idx]
            for _ in range(2):
                correct_s.train_self_supervised(state, is_me=True)
            
            # 负样本：随机选其他 Structon
            others = [s for s in self.structons if s.label != label]
            n_neg = min(n_negatives, len(others))
            negatives = np.random.choice(others, n_neg, replace=False)
            for s in negatives:
                s.train_self_supervised(state, is_me=False)

    def predict(self, image, max_hops=30, local_learning=False) -> Tuple[str, List[str], int]:
        """
        路由预测
        返回: (预测的数字, 路径, 查询次数)
        """
        state = self.extractor.extract(image)
        
        current = np.random.choice(self.structons)
        path = []
        visited = set()
        queries = 0
        
        for step in range(max_hops):
            visited.add(current.id)
            path.append(current.full_label)
            queries += 1
            
            action, next_s, confidence = current.decide(state, visited)
            
            if action == 0:  # SELF
                if local_learning:
                    current.learn_self_confident(state, confidence)
                return current.label, path, queries
            
            if next_s is not None:
                if next_s.id in visited:
                    if local_learning:
                        current.learn_route_revisit(state, action)
                    
                    found_alt = False
                    for i, conn in enumerate(current.connections):
                        if conn.id not in visited:
                            next_s = conn
                            action = i + 1
                            found_alt = True
                            break
                    
                    if not found_alt:
                        next_s = None
                
                if next_s is not None:
                    downstream_q, _ = next_s.get_q_values(state)
                    downstream_self_q = downstream_q[0]
                    
                    downstream_visited = visited | {next_s.id}
                    has_path = any(c.id not in downstream_visited for c in next_s.connections)
                    
                    if local_learning:
                        if downstream_self_q > next_s.confidence_threshold:
                            current.learn_route_accepted(state, action, downstream_self_q)
                        elif has_path:
                            current.learn_route_continued(state, action)
                        else:
                            current.learn_route_deadend(state, action)
                    
                    current = next_s
                    continue
            
            all_unvisited = [s for s in self.structons if s.id not in visited]
            if all_unvisited:
                current = np.random.choice(all_unvisited)
                path.append("JUMP")
            else:
                break
        
        return current.label, path, queries

    def train_local_epoch(self, samples):
        """本地学习"""
        np.random.shuffle(samples)
        for img, label, variant_idx in samples:
            self.predict(img, local_learning=True)

    def predict_voting(self, image) -> Tuple[str, Dict[str, float], int]:
        """
        投票预测
        返回: (预测的数字, 详情, 查询次数)
        """
        state = self.extractor.extract(image)
        
        votes = {}
        queries = 0
        
        for s in self.structons:
            queries += 1
            if s.routing_lrm is None:
                continue
                
            q_values, _ = s.routing_lrm.query(state)
            self_q = q_values[0]
            
            # 累加到数字的投票
            if s.label not in votes:
                votes[s.label] = 0
            if self_q > 0:
                votes[s.label] += self_q
        
        if not votes or max(votes.values()) <= 0:
            # 没有正票，选最高的
            all_scores = {}
            for s in self.structons:
                q_values, _ = s.routing_lrm.query(state)
                if s.label not in all_scores:
                    all_scores[s.label] = []
                all_scores[s.label].append(q_values[0])
            # 每个数字取最高分
            for label in all_scores:
                votes[label] = max(all_scores[label])
        
        best = max(votes, key=votes.get)
        return best, votes, queries

    def print_stats(self):
        print("\n" + "=" * 70)
        print(f"Structon Vision v9.15 - Scale Test (N={len(self.structons)})")
        print("=" * 70)
        print(f"Structon 数量: {len(self.structons)}")
        print(f"每个数字: {self.n_variants} 个变体")
        print(f"连接数/节点: {self.n_connections}")
        
        total_mem = sum(s.memory_size for s in self.structons)
        print(f"总记忆: {total_mem}")
        
        # 按数字统计
        print("\n=== 按数字统计 ===")
        print(f"{'数字':<6} {'Mem':<8} {'Self+':<8} {'Route+':<8}")
        print("-" * 35)
        for label in sorted(self.label_to_structons.keys()):
            structons = self.label_to_structons[label]
            total_mem = sum(s.memory_size for s in structons)
            total_self = sum(s.stats['self_confident'] for s in structons)
            total_route = sum(s.stats['route_accepted'] for s in structons)
            print(f"  {label:<4} {total_mem:<8} {total_self:<8} {total_route:<8}")

# =============================================================================
# 6. 主实验
# =============================================================================
def run_experiment(
    n_variants: int = 10,
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 300,
    epochs: int = 30,
    n_connections: int = 5,
    n_negatives: int = 5,
    local_epochs: int = 10
):
    n_total = 10 * n_variants
    print("=" * 70)
    print(f"Structon Vision v9.15 - Scale Test (N={n_total})")
    print("=" * 70)
    print(f"\n每个数字 {n_variants} 个 Structon，共 {n_total} 个")
    print(f"参数: capacity={capacity}, 每类样本={n_per_class}, 连接数={n_connections}")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        n_variants=n_variants,
        capacity=capacity,
        n_connections=n_connections
    )
    system.build([str(i) for i in range(10)])
    
    # 准备样本：每个样本分配给一个变体
    samples = []
    for d in range(10):
        idxs = np.where(train_labels == d)[0][:n_per_class]
        for i, idx in enumerate(idxs):
            variant_idx = i % n_variants  # 轮流分配给不同变体
            samples.append((train_images[idx], str(d), variant_idx))
    
    print(f"\n训练样本: {len(samples)}")
    
    # 阶段1：监督训练
    print(f"\n=== 阶段1: 监督训练 ({epochs} epochs) ===")
    t0 = time.time()
    
    for ep in range(epochs):
        system.train_epoch(samples, n_negatives=n_negatives)
        
        if (ep + 1) % 5 == 0:
            correct = 0
            total_queries = 0
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, _, queries = system.predict_voting(test_images[i])
                total_queries += queries
                if pred == str(test_labels[i]):
                    correct += 1
            avg_queries = total_queries / check_n
            print(f"  轮次 {ep+1}: 投票准确率={correct/check_n*100:.1f}%, 平均查询={avg_queries:.0f}")
    
    print(f"\n监督训练: {time.time()-t0:.1f}秒")
    
    # 阶段2：本地学习
    print(f"\n=== 阶段2: 本地学习 ({local_epochs} epochs) ===")
    t1 = time.time()
    
    for ep in range(local_epochs):
        system.train_local_epoch(samples)
        
        if (ep + 1) % 2 == 0:
            correct = 0
            total_queries = 0
            path_lens = []
            check_n = 200
            idxs = np.random.choice(len(test_images), check_n, replace=False)
            for i in idxs:
                pred, path, queries = system.predict(test_images[i])
                total_queries += queries
                path_lens.append(len(path))
                if pred == str(test_labels[i]):
                    correct += 1
            avg_queries = total_queries / check_n
            avg_path = np.mean(path_lens)
            print(f"  轮次 {ep+1}: 路由准确率={correct/check_n*100:.1f}%, 平均查询={avg_queries:.1f}, 平均路径={avg_path:.1f}")
    
    print(f"\n本地学习: {time.time()-t1:.1f}秒")
    
    system.print_stats()
    
    # 最终测试
    print(f"\n=== 最终测试 ===")
    test_idxs = np.random.choice(len(test_images), n_test, replace=False)
    
    # 路由测试
    correct_route = 0
    route_queries = 0
    path_lengths = []
    jump_counts = []
    
    t_route = time.time()
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, path, queries = system.predict(test_images[idx])
        route_queries += queries
        path_lengths.append(len(path))
        jump_counts.append(path.count("JUMP"))
        if pred == true_label:
            correct_route += 1
    t_route = time.time() - t_route
    
    print(f"\n路由预测:")
    print(f"  准确率: {correct_route/n_test*100:.1f}%")
    print(f"  平均查询: {route_queries/n_test:.1f}")
    print(f"  平均路径: {np.mean(path_lengths):.2f}")
    print(f"  平均跳转: {np.mean(jump_counts):.2f}")
    print(f"  总时间: {t_route:.2f}秒")
    
    # 投票测试
    correct_vote = 0
    vote_queries = 0
    
    t_vote = time.time()
    for idx in test_idxs:
        true_label = str(test_labels[idx])
        pred, _, queries = system.predict_voting(test_images[idx])
        vote_queries += queries
        if pred == true_label:
            correct_vote += 1
    t_vote = time.time() - t_vote
    
    print(f"\n投票预测:")
    print(f"  准确率: {correct_vote/n_test*100:.1f}%")
    print(f"  平均查询: {vote_queries/n_test:.1f}")
    print(f"  总时间: {t_vote:.2f}秒")
    
    # 对比
    print("\n=== 对比 ===")
    print(f"{'方法':<10} {'准确率':<10} {'查询数':<10} {'时间':<10}")
    print("-" * 40)
    print(f"{'路由':<10} {correct_route/n_test*100:<10.1f}% {route_queries/n_test:<10.1f} {t_route:<10.2f}s")
    print(f"{'投票':<10} {correct_vote/n_test*100:<10.1f}% {vote_queries/n_test:<10.1f} {t_vote:<10.2f}s")
    
    speedup = (vote_queries/n_test) / (route_queries/n_test)
    print(f"\n路由查询效率: {speedup:.1f}x 少于投票")
    
    # 示例路径
    print("\n=== 示例路径 ===")
    for i in range(5):
        idx = test_idxs[i]
        pred, path, queries = system.predict(test_images[idx])
        true = str(test_labels[idx])
        status = "✓" if pred == true else "✗"
        path_str = ' → '.join(path[:8])
        if len(path) > 8:
            path_str += f" ... ({len(path)} hops)"
        print(f"  真实={true}, 预测={pred} {status}, 查询={queries}, 路径: {path_str}")
    
    return system


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', type=int, default=10, help='每个数字的 Structon 数量')
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=300)
    parser.add_argument('--max-epochs', type=int, default=30)
    parser.add_argument('--connections', type=int, default=5)
    parser.add_argument('--negatives', type=int, default=5)
    parser.add_argument('--local-epochs', type=int, default=10)
    args = parser.parse_args()
    
    run_experiment(
        n_variants=args.variants,
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        epochs=args.max_epochs,
        n_connections=args.connections,
        n_negatives=args.negatives,
        local_epochs=args.local_epochs
    )
