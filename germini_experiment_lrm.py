"""
Structon Routing RL: 基于改进版共振记忆的学习型路由网络
====================================================

核心思想：
1. 每个 Structon 是一个独立的 Agent。
2. 动作空间：Action 0 = Claim (这是我的数字); Action 1..k = Route (传给邻居 k)。
3. 训练方式：多步轨迹回溯更新 (Trajectory Replay)。

结合了：
- improved_local_resonant_memory.py 的记忆机制
- structon_mnist_v9_6.py 的特征提取
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Any, Optional
import argparse

# =============================================================================
# 1. 改进版共振记忆 (移植自 file 1)
# =============================================================================
class ImprovedResonantMemory:
    """改进版共振记忆体 - 移植并适配"""
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        capacity: int = 100,
        key_dim: Optional[int] = None,
        temperature: float = 0.1,
        learning_rate: float = 0.1, # 降低一点 LR 以稳定路由
        decay_rate: float = 0.99,   # 记忆访问衰减
        discretize: bool = False,   # 图像特征建议不用离散化，使用投影
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.discretize = discretize
        
        # 投影矩阵：将高维特征映射到 Key 空间
        self.key_dim = key_dim or 64
        self.projector = np.random.randn(state_dim, self.key_dim).astype(np.float32)
        self.projector /= np.linalg.norm(self.projector, axis=0, keepdims=True) + 1e-8
        
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.access_counts: List[int] = []
    
    def _encode(self, state: np.ndarray) -> np.ndarray:
        # 随机投影编码
        key = state.astype(np.float32) @ self.projector
        norm = np.linalg.norm(key)
        if norm > 1e-8:
            key /= norm
        return key
    
    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        query_key = self._encode(state)
        key_bank = np.array(self.keys)
        
        scores = key_bank @ query_key
        max_score = np.max(scores)
        
        # Softmax 加权
        weights = np.exp((scores - max_score) / self.temperature)
        weights /= np.sum(weights) + 1e-8
        
        value_bank = np.array(self.values)
        q_values = weights @ value_bank
        
        # 访问计数增加 (选取权重最大的)
        top_idx = int(np.argmax(weights))
        self.access_counts[top_idx] += 1
        
        return q_values, float(max_score)
    
    def remember(self, state: np.ndarray, action: int, target_q: float) -> None:
        """更新 Q 值"""
        query_key = self._encode(state)
        
        # 衰减访问计数
        if self.decay_rate < 1.0 and len(self.access_counts) > 0:
             # 简单的定期衰减，实际实现可简化
             pass 

        # 1. 尝试更新现有记忆
        if len(self.keys) > 0:
            key_bank = np.array(self.keys)
            scores = key_bank @ query_key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score > 0.90: # 相似度阈值
                old_val = self.values[best_idx][action]
                self.values[best_idx][action] = old_val + self.learning_rate * (target_q - old_val)
                self.access_counts[best_idx] += 1
                return

        # 2. 新增记忆
        if len(self.keys) > 0:
            # 继承附近的 Q 值作为初始值，加速收敛
            new_q, _ = self.query(state)
            new_q = new_q.copy()
        else:
            new_q = np.zeros(self.n_actions, dtype=np.float32)
            
        new_q[action] = target_q # 初始赋予目标值
        
        # 容量管理：移除最少使用的
        if len(self.keys) >= self.capacity:
            forgotten_idx = int(np.argmin(self.access_counts))
            self.keys.pop(forgotten_idx)
            self.values.pop(forgotten_idx)
            self.access_counts.pop(forgotten_idx)
        
        self.keys.append(query_key)
        self.values.append(new_q)
        self.access_counts.append(1)

# =============================================================================
# 2. 特征提取 (保持 Structon v9.6 的优秀特征)
# =============================================================================
class StateExtractor:
    """Contrast Binary + Structure 特征提取"""
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        bh, bw = h // self.grid_size, w // self.grid_size
        features = []
        
        # 简化版实现，保留核心特征
        grid = np.zeros((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid[i, j] = np.mean(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
        
        # 1. Contrast (49)
        mean_val = np.mean(grid)
        for val in grid.flat:
            features.append(1.0 if val > mean_val else -1.0)
            
        # 2. Gradients (84)
        # Horizontal
        features.extend((grid[:, :-1] - grid[:, 1:]).flatten())
        # Vertical
        features.extend((grid[:-1, :] - grid[1:, :]).flatten())
        
        # 3. Global Stats
        features.append(np.mean(grid[:3, :3]) - np.mean(grid[4:, 4:])) # TL - BR
        features.append(np.mean(grid[:3, 4:]) - np.mean(grid[4:, :3])) # TR - BL
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        return state / (norm + 1e-6)

# =============================================================================
# 3. 智能路由 Structon
# =============================================================================
class RoutingStructon:
    def __init__(self, label: str, capacity=300):
        self.label = label
        self.neighbors: List['RoutingStructon'] = []
        self.memory: Optional[ImprovedResonantMemory] = None
        
    def connect(self, all_structons, n_connections=3):
        """建立连接并初始化记忆体"""
        others = [s for s in all_structons if s.label != self.label]
        # 随机选择邻居 (模拟局部连接)
        if len(others) >= n_connections:
            self.neighbors = list(np.random.choice(others, n_connections, replace=False))
        else:
            self.neighbors = others
            
        # 动作空间：Action 0 = Claim; Action 1..k = Route to neighbor[k-1]
        n_actions = 1 + len(self.neighbors)
        
        # 初始化记忆体
        self.memory = ImprovedResonantMemory(
            state_dim=135, # 约 135 维特征
            n_actions=n_actions,
            capacity=capacity,
            key_dim=64,
            temperature=0.08
        )
        
    def decide(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """Epsilon-Greedy 决策"""
        if np.random.random() < epsilon:
            return np.random.randint(self.memory.n_actions)
        
        q, _ = self.memory.query(state)
        return int(np.argmax(q))

# =============================================================================
# 4. 视觉系统与训练逻辑
# =============================================================================
class StructonRoutingSystem:
    def __init__(self, capacity=200, n_connections=3):
        self.extractor = StateExtractor()
        self.structons: List[RoutingStructon] = []
        self.n_connections = n_connections
        self.capacity = capacity
        
    def build(self, labels):
        self.structons = [RoutingStructon(l, self.capacity) for l in labels]
        for s in self.structons:
            s.connect(self.structons, self.n_connections)
        print(f"Build complete: {len(self.structons)} structons, {self.n_connections} neighbors each.")

    def train_episode(self, image: np.ndarray, true_label: str, epsilon: float = 0.1):
        """执行一次完整的路由训练 (Episode)"""
        state = self.extractor.extract(image)
        
        # 随机选择入口 Structon
        current = self.structons[np.random.randint(len(self.structons))]
        
        trajectory = [] # [(structon, action), ...]
        max_hops = 8
        done = False
        final_result = "timeout"
        
        # --- 1. 前向传播 (Forward) ---
        for _ in range(max_hops):
            action = current.decide(state, epsilon)
            trajectory.append((current, action))
            
            if action == 0: # Claim (Yes)
                if current.label == true_label:
                    final_result = "correct"
                else:
                    final_result = "wrong"
                done = True
                break
            else: # Route (No)
                # Action i -> Neighbor i-1
                neighbor_idx = action - 1
                current = current.neighbors[neighbor_idx]
        
        # --- 2. 计算奖励 (Reward Calculation) ---
        if final_result == "correct":
            final_reward = 1.0
        elif final_result == "wrong":
            final_reward = -1.0 # 惩罚抢答错误
        else: # timeout
            final_reward = -0.5 # 惩罚找不到路
            
        # --- 3. 反向传播更新 (Backward Update) ---
        # 我们使用蒙特卡洛回报：G_t = r + gamma * G_{t+1}
        running_reward = final_reward
        gamma = 0.9 # 衰减系数，让之前的节点分到的奖励少一点（鼓励短路径）
        step_penalty = -0.05 # 每多跳一步扣一点分，鼓励快速找到
        
        for structon, action in reversed(trajectory):
            # Q-Learning update: target = reward
            # 这里的 reward 是该步骤之后累积的折扣回报
            structon.memory.remember(state, action, running_reward)
            
            # 衰减并加上单步惩罚
            running_reward = running_reward * gamma + step_penalty

        return final_result

    def predict(self, image: np.ndarray) -> Tuple[str, List[str]]:
        state = self.extractor.extract(image)
        # 测试时通常从固定的或者随机的入口进入
        # 为了稳定性，可以多次随机入口投票，或者简单地随机选一个
        current = self.structons[np.random.randint(len(self.structons))]
        path = []
        
        for _ in range(15): # Max hops
            path.append(current.label)
            q, _ = current.memory.query(state)
            action = int(np.argmax(q))
            
            if action == 0: # Claim
                return current.label, path
            else:
                neighbor_idx = action - 1
                current = current.neighbors[neighbor_idx]
                
        return "unknown", path

# =============================================================================
# 5. 数据加载与主程序
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

def main():
    print("=" * 60)
    print("Structon RL Routing Experiment")
    print("动作 0: Claim (Yes) | 动作 1-N: Route to Neighbor")
    print("=" * 60)
    
    # 1. 加载数据
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    
    # 2. 构建系统
    # 减少 connections 强制它学习跳跃，增加 connections 路由更容易但表更大
    system = StructonRoutingSystem(capacity=300, n_connections=3)
    system.build([str(i) for i in range(10)])
    
    # 3. 训练
    n_epochs = 5
    batch_size = 2000 # 每个 epoch 随机采样的样本数
    
    print(f"\n开始训练 ({n_epochs} 轮)...")
    t0 = time.time()
    
    for ep in range(n_epochs):
        indices = np.random.choice(len(train_img), batch_size, replace=False)
        outcomes = {'correct': 0, 'wrong': 0, 'timeout': 0}
        
        # 随时间降低探索率 epsilon
        epsilon = max(0.05, 0.5 * (0.6 ** ep))
        
        for i in indices:
            res = system.train_episode(train_img[i], str(train_lbl[i]), epsilon)
            outcomes[res] += 1
            
        acc = outcomes['correct'] / batch_size * 100
        print(f"Epoch {ep+1}: Acc={acc:.1f}% (Correct: {outcomes['correct']}, Wrong: {outcomes['wrong']}, Timeout: {outcomes['timeout']}) | Eps={epsilon:.2f}")
    
    print(f"训练耗时: {time.time()-t0:.1f}s")
    
    # 4. 测试
    print("\n=== 测试集评估 ===")
    test_size = 500
    test_indices = np.random.choice(len(test_img), test_size, replace=False)
    correct = 0
    
    print(f"示例路径:")
    for i, idx in enumerate(test_indices):
        pred, path = system.predict(test_img[idx])
        true_lbl = str(test_lbl[idx])
        if pred == true_lbl:
            correct += 1
            
        if i < 10: # 打印前10个示例
            status = "✓" if pred == true_lbl else "✗"
            path_str = " -> ".join(path)
            print(f"  True: {true_lbl} | Pred: {pred} {status} | Path: {path_str}")
            
    print(f"\n最终准确率: {correct/test_size*100:.1f}%")
    
    # 5. 记忆统计
    print("\n记忆体统计:")
    total_mem = 0
    for s in system.structons:
        mem_size = len(s.memory.keys)
        total_mem += mem_size
        print(f"  Structon {s.label}: {mem_size} keys")
    print(f"Total Memories: {total_mem}")

if __name__ == "__main__":
    main()
