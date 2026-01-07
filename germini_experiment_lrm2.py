"""
Structon Multi-View RL (Mixture of Experts)
===========================================
核心特性：
1.  **30个智能体**：每个数字有 Global/Top/Bottom 三种专家。
2.  **关注点机制 (Concerns)**：不同的 Structon 观察图像的不同部分。
3.  **强制教学 (Teacher Forcing)**：到达正确节点时强制学习 Claim 动作。
4.  **无投影记忆**：使用原始特征直接计算相似度，提高精度。
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Any, Optional, Set

# =============================================================================
# 1. 改进版共振记忆 (无投影版本)
# =============================================================================
class ImprovedResonantMemory:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        capacity: int = 100,
        temperature: float = 0.1,
        learning_rate: float = 0.1,
        decay_rate: float = 0.99,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        
        # [修改] 移除了随机投影，直接存储原始特征以保持几何信息
        self.keys: List[np.ndarray] = []
        self.values: List[np.ndarray] = []
        self.access_counts: List[int] = []
    
    def _encode(self, state: np.ndarray) -> np.ndarray:
        # 直接透传归一化的状态
        return state
    
    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        query_key = self._encode(state)
        key_bank = np.array(self.keys)
        
        # 计算点积相似度
        scores = key_bank @ query_key
        max_score = np.max(scores)
        
        # Softmax 加权
        weights = np.exp((scores - max_score) / self.temperature)
        weights /= np.sum(weights) + 1e-8
        
        value_bank = np.array(self.values)
        q_values = weights @ value_bank
        
        top_idx = int(np.argmax(weights))
        self.access_counts[top_idx] += 1
        
        return q_values, float(max_score)
    
    def remember(self, state: np.ndarray, action: int, target_q: float) -> None:
        query_key = self._encode(state)
        
        # 1. 尝试更新现有记忆
        if len(self.keys) > 0:
            key_bank = np.array(self.keys)
            scores = key_bank @ query_key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            # 提高匹配阈值，保证专精
            if best_score > 0.92: 
                old_val = self.values[best_idx][action]
                self.values[best_idx][action] = old_val + self.learning_rate * (target_q - old_val)
                self.access_counts[best_idx] += 1
                return

        # 2. 新增记忆
        if len(self.keys) > 0:
            new_q, _ = self.query(state)
            new_q = new_q.copy()
        else:
            new_q = np.zeros(self.n_actions, dtype=np.float32)
            
        new_q[action] = target_q
        
        # 容量管理
        if len(self.keys) >= self.capacity:
            forgotten_idx = int(np.argmin(self.access_counts))
            self.keys.pop(forgotten_idx)
            self.values.pop(forgotten_idx)
            self.access_counts.pop(forgotten_idx)
        
        self.keys.append(query_key)
        self.values.append(new_q)
        self.access_counts.append(1)

# =============================================================================
# 2. 特化特征提取器 (支持关注点裁剪)
# =============================================================================
class SpecializedExtractor:
    def __init__(self, grid_size=7):
        self.grid_size = grid_size
        
    def extract(self, image: np.ndarray, concern: str = 'global') -> np.ndarray:
        """根据 concern 裁剪图片并提取特征"""
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        
        # --- 1. 视点裁剪 (Attention Logic) ---
        if concern == 'top':
            # 只看上半部分
            img = img[:h//2, :]
        elif concern == 'bottom':
            # 只看下半部分
            img = img[h//2:, :]
        elif concern == 'center':
            # 只看中心
            mh, mw = h//4, w//4
            img = img[mh:-mh, mw:-mw]
        # 'global' 使用全图
        
        # --- 2. 动态网格特征提取 ---
        # 因为图片大小变了，需要重新计算 block size
        cur_h, cur_w = img.shape
        bh, bw = max(1, cur_h // self.grid_size), max(1, cur_w // self.grid_size)
        
        features = []
        grid = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 安全切片，防止除法导致的越界
                r_start = min(i*bh, cur_h)
                r_end = min((i+1)*bh, cur_h)
                c_start = min(j*bw, cur_w)
                c_end = min((j+1)*bw, cur_w)
                
                if r_end > r_start and c_end > c_start:
                    grid[i, j] = np.mean(img[r_start:r_end, c_start:c_end])
                else:
                    grid[i, j] = 0.0
        
        # 3. 构建特征向量 (Contrast + Gradients + Global)
        mean_val = np.mean(grid)
        # Binary Contrast (49)
        for val in grid.flat:
            features.append(1.0 if val > mean_val else -1.0)
            
        # Gradients (Horizontal + Vertical)
        features.extend((grid[:, :-1] - grid[:, 1:]).flatten())
        features.extend((grid[:-1, :] - grid[1:, :]).flatten())
        
        # Quadrant diffs
        features.append(np.mean(grid[:3, :3]) - np.mean(grid[4:, 4:]))
        features.append(np.mean(grid[:3, 4:]) - np.mean(grid[4:, :3]))
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        return state / (norm + 1e-6)

# =============================================================================
# 3. 多视角 Structon
# =============================================================================
class MultiViewStructon:
    def __init__(self, label: str, concern: str, capacity=300):
        self.label = label          # e.g., "1"
        self.concern = concern      # e.g., "top"
        self.unique_id = f"{label}_{concern}" # e.g., "1_top"
        
        self.neighbors: List['MultiViewStructon'] = []
        self.memory: Optional[ImprovedResonantMemory] = None
        self.capacity = capacity
        
    def connect(self, all_structons, n_connections=5):
        """建立连接：优先连接同数字的其他视角，再随机连接"""
        others = [s for s in all_structons if s.unique_id != self.unique_id]
        
        # 1. 寻找同胞 (Siblings): 同 Label 但不同 Concern
        siblings = [s for s in others if s.label == self.label]
        
        # 2. 寻找外人 (Others)
        strangers = [s for s in others if s.label != self.label]
        
        self.neighbors = []
        # 必连同胞 (促进内部协作)
        self.neighbors.extend(siblings)
        
        # 补足剩余连接数
        needed = max(0, n_connections - len(self.neighbors))
        if needed > 0 and len(strangers) > 0:
            random_picks = np.random.choice(strangers, size=min(needed, len(strangers)), replace=False)
            self.neighbors.extend(list(random_picks))
            
        # 初始化记忆体
        n_actions = 1 + len(self.neighbors)
        # state_dim 约为 135
        self.memory = ImprovedResonantMemory(
            state_dim=135,
            n_actions=n_actions,
            capacity=self.capacity,
            temperature=0.08
        )
        
    def decide(self, image: np.ndarray, extractor: SpecializedExtractor, visited_ids: Set[str], epsilon: float = 0.1) -> int:
        # [关键] 使用自己的 concern 提取特征
        state = extractor.extract(image, self.concern)
        
        q, _ = self.memory.query(state)
        
        # Taboo Search Logic
        valid_actions = [0] # Claim 总是允许
        
        for i, neighbor in enumerate(self.neighbors):
            if neighbor.unique_id not in visited_ids:
                valid_actions.append(i + 1)
        
        if not valid_actions: valid_actions = [0]

        # Epsilon-Greedy
        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        else:
            masked_q = np.full_like(q, -np.inf)
            masked_q[valid_actions] = q[valid_actions]
            return int(np.argmax(masked_q))
            
    def force_learn(self, image: np.ndarray, extractor: SpecializedExtractor, action: int, target: float):
        """强制教学接口"""
        state = extractor.extract(image, self.concern)
        self.memory.remember(state, action, target)

# =============================================================================
# 4. 多视角路由系统
# =============================================================================
class MultiViewRoutingSystem:
    def __init__(self, capacity=200):
        self.extractor = SpecializedExtractor()
        self.structons: List[MultiViewStructon] = []
        
    def build(self, labels: List[str]):
        # 定义三种关注点
        concerns = ['global', 'top', 'bottom']
        
        self.structons = []
        print(f"构建网络: {len(labels)} 类 x {len(concerns)} 视角 = {len(labels)*len(concerns)} 个 Structons")
        
        for label in labels:
            for concern in concerns:
                s = MultiViewStructon(label, concern, capacity=200)
                self.structons.append(s)
                
        # 建立连接 (Connections)
        # 增加连接数以适应更大的网络
        for s in self.structons:
            s.connect(self.structons, n_connections=6)
            
    def train_episode(self, image: np.ndarray, true_label: str, epsilon: float = 0.1):
        # 随机入口
        current = self.structons[np.random.randint(len(self.structons))]
        
        trajectory = [] 
        visited = set()
        
        max_hops = 12 # 稍微增加跳数上限
        done = False
        final_result = "timeout"
        
        for _ in range(max_hops):
            visited.add(current.unique_id)
            
            # --- [特性] 强制教学 (Teacher Forcing) ---
            # 如果当前节点属于正确类别 (无论它是 top 还是 bottom)，教它认领
            if current.label == true_label:
                current.force_learn(image, self.extractor, action=0, target=1.0)
            
            # 决策
            action = current.decide(image, self.extractor, visited, epsilon)
            trajectory.append((current, action))
            
            if action == 0: # Claim
                if current.label == true_label:
                    final_result = "correct"
                else:
                    final_result = "wrong"
                done = True
                break
            else: # Route
                neighbor_idx = action - 1
                if neighbor_idx < len(current.neighbors):
                    current = current.neighbors[neighbor_idx]
                else:
                    break
        
        # 奖励计算
        if final_result == "correct":
            final_reward = 1.0
        elif final_result == "wrong":
            final_reward = -2.0 # 严厉惩罚
        else: 
            final_reward = -0.5
            
        # 反向传播 (仅更新特征向量对应的状态)
        running_reward = final_reward
        gamma = 0.9 
        step_penalty = -0.02
        
        for structon, action in reversed(trajectory):
            # 获取该 Structon 视角的 State
            s_vec = self.extractor.extract(image, structon.concern)
            structon.memory.remember(s_vec, action, running_reward)
            running_reward = running_reward * gamma + step_penalty

        return final_result

    def predict(self, image: np.ndarray) -> Tuple[str, List[str]]:
        current = self.structons[np.random.randint(len(self.structons))]
        path = []
        visited = set()
        
        for _ in range(15):
            path.append(current.unique_id) # 记录 "1_top", "3_global" 等
            visited.add(current.unique_id)
            
            action = current.decide(image, self.extractor, visited, epsilon=0.0)
            
            if action == 0:
                return current.label, path
            else:
                neighbor_idx = action - 1
                if neighbor_idx < len(current.neighbors):
                    current = current.neighbors[neighbor_idx]
                else:
                    break
                
        return "unknown", path

# =============================================================================
# 5. 主程序
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
    print("Structon Multi-View Experiment (Mixture of Experts)")
    print("Concerns: Global, Top, Bottom")
    print("=" * 60)
    
    # 1. 加载数据
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    
    # 2. 构建系统
    system = MultiViewRoutingSystem()
    system.build([str(i) for i in range(10)])
    
    # 3. 训练
    n_epochs = 5
    batch_size = 2000 
    
    print(f"\n开始训练 ({n_epochs} 轮)...")
    t0 = time.time()
    
    for ep in range(n_epochs):
        indices = np.random.choice(len(train_img), batch_size, replace=False)
        outcomes = {'correct': 0, 'wrong': 0, 'timeout': 0}
        
        # 探索率衰减
        epsilon = max(0.05, 0.5 * (0.6 ** ep))
        
        for i in indices:
            res = system.train_episode(train_img[i], str(train_lbl[i]), epsilon)
            outcomes[res] += 1
            
        acc = outcomes['correct'] / batch_size * 100
        print(f"Epoch {ep+1}: Acc={acc:.1f}% (C: {outcomes['correct']}, W: {outcomes['wrong']}, T: {outcomes['timeout']}) | Eps={epsilon:.2f}")
    
    print(f"训练耗时: {time.time()-t0:.1f}s")
    
    # 4. 测试
    print("\n=== 测试集评估 ===")
    test_size = 500
    test_indices = np.random.choice(len(test_img), test_size, replace=False)
    correct = 0
    
    print(f"示例路径 (注意观察不同视角的协作):")
    for i, idx in enumerate(test_indices):
        pred, path = system.predict(test_img[idx])
        true_lbl = str(test_lbl[idx])
        if pred == true_lbl:
            correct += 1
            
        if i < 15: 
            status = "✓" if pred == true_lbl else "✗"
            path_str = " -> ".join(path)
            print(f"  True: {true_lbl} | Pred: {pred} {status} | Path: {path_str}")
            
    print(f"\n最终准确率: {correct/test_size*100:.1f}%")
    
    # 5. 统计
    print("\n网络统计:")
    total_mem = sum(len(s.memory.keys) for s in system.structons)
    print(f"Total Agents: {len(system.structons)}")
    print(f"Total Memories: {total_mem}")

if __name__ == "__main__":
    main()
