#!/usr/bin/env python3
"""
Structon Vision v10.0 - 动力学满足范式 (Kinetic Satisficing)
1. 局部动作空间：每个节点仅感知 [认领, 邻居1, 邻居2, 邻居3]。
2. 三位一体奖励：认领正确(强吸引), 认领错误(强排斥), 路由动作(持续激励流)。
3. 动能(Momentum)：模拟摩擦力，控制搜索半径，防止死循环。
4. 边走边学：无需反向传播，信号每跳一步即完成局部 LRM 更新。
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple

# =============================================================================
# 1. 基础工具：数据加载与特征提取
# =============================================================================
def load_mnist():
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz', 'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz', 'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    data = {}
    mnist_dir = os.path.expanduser('~/.mnist')
    os.makedirs(mnist_dir, exist_ok=True)
    
    for key, filename in files.items():
        filepath = os.path.join(mnist_dir, filename)
        if not os.path.exists(filepath):
            print(f"下载数据集: {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                f.read(16)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            else:
                f.read(8)
                data[key] = np.frombuffer(f.read(), dtype=np.uint8)
    return data['train_images'], data['train_labels'], data['test_images'], data['test_labels']

class StateExtractor:
    """特征提取：模拟初级视觉皮层的方向差异感知"""
    def __init__(self, grid_size=7, threshold=0.25):
        self.grid_size = grid_size
        self.threshold = threshold
        
    def extract(self, image: np.ndarray) -> np.ndarray:
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        grid = np.zeros((self.grid_size, self.grid_size))
        bh, bw = h // self.grid_size, w // self.grid_size
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                grid[i, j] = np.mean(img[i*bh:(i+1)*bh, j*bw:(j+1)*bw])
        
        # 1. 二进制对比特征
        features = [-1.0 if v < self.threshold else 1.0 for v in grid.flatten()]
        # 2. 宏观结构差异 (上下、左右、对角)
        mid = self.grid_size // 2
        features.append((np.mean(grid[:mid, :]) - np.mean(grid[mid:, :])) * 2) 
        features.append((np.mean(grid[:, :mid]) - np.mean(grid[:, mid:])) * 2)
        
        state = np.array(features, dtype=np.float32)
        norm = np.linalg.norm(state)
        return state / norm if norm > 1e-6 else state

# =============================================================================
# 2. 动力学 LRM (Kinetic LRM)
# =============================================================================
class KineticLRM:
    """存储动作倾向的局部记忆单元"""
    def __init__(self, n_actions=4):
        self.n_actions = n_actions
        self.keys = []
        self.values = []
        self.lr = 0.4
        self.sim_threshold = 0.85

    def query(self, state):
        if not self.keys: return np.zeros(self.n_actions)
        # 向量空间共振
        scores = np.dot(np.array(self.keys), state)
        best_idx = np.argmax(scores)
        if scores[best_idx] > 0.1:
            return self.values[best_idx] * scores[best_idx]
        return np.zeros(self.n_actions)

    def update(self, state, action_idx, reward):
        """局部即时强化"""
        if self.keys:
            scores = np.dot(np.array(self.keys), state)
            best_idx = np.argmax(scores)
            if scores[best_idx] > self.sim_threshold:
                # 更新已有记忆点的动作 Q 值
                self.values[best_idx][action_idx] += self.lr * (reward - self.values[best_idx][action_idx])
                return
        
        # 创建新动作记忆
        val = np.zeros(self.n_actions, dtype=np.float32)
        val[action_idx] = reward
        if len(self.keys) >= 300:
            self.keys.pop(0); self.values.pop(0)
        self.keys.append(state.copy()); self.values.append(val)

# =============================================================================
# 3. Structon 视觉系统
# =============================================================================
class Structon:
    def __init__(self, label):
        self.label = label
        self.connections: List['Structon'] = []
        self.lrm = None

    def init_local_world(self, others):
        # 局部稀疏连接：每个节点只认识 3 个邻居
        self.connections = list(np.random.choice(others, 3, replace=False))
        # 动作空间：0:认领, 1:转给邻居1, 2:转给邻居2, 3:转给邻居3
        self.lrm = KineticLRM(n_actions=4)

class DynamicVisionSystem:
    def __init__(self):
        self.extractor = StateExtractor()
        self.structons = [Structon(str(i)) for i in range(10)]
        for s in self.structons:
            others = [o for o in self.structons if o != s]
            s.init_local_world(others)

    def navigate(self, image, target_label=None, train=False):
        """基于动能耗尽的满足感路由"""
        state = self.extractor.extract(image)
        # 随机初始节点模拟环境扰动
        current = np.random.choice(self.structons)
        momentum = 1.0
        path = []
        
        while momentum > 0.15:
            q_vec = current.lrm.query(state)
            action = np.argmax(q_vec)
            conf = q_vec[action]
            path.append(current.label)
            
            # --- 核心逻辑：认领动作 ---
            if action == 0: 
                if train and target_label:
                    if current.label == target_label:
                        current.lrm.update(state, 0, 2.5) # [1] 认领-对：形成深谷吸引子
                        return current.label, path
                    else:
                        current.lrm.update(state, 0, -1.5) # [2] 认领-错：形成山峰排斥子
                        momentum -= 0.3 # 决策挫败消耗动能
                elif conf > 0.9: # 推理模式：强信心即为满足
                    return current.label, path
                else:
                    momentum -= 0.1 # 弱信心继续探索
            
            # --- 核心逻辑：路由动作 ---
            # [3] 路由下一个：永远给奖励，赋予流动本能
            if train:
                # 即使当前选了认领但被惩罚，训练时也强化一下路由备选方案
                route_act = action if action > 0 else np.random.randint(1, 4)
                current.lrm.update(state, route_act, 0.6) 
            
            # 执行跳转：action 1-3 对应邻居 0-2
            next_conn_idx = (action - 1) if action > 0 else np.random.randint(0, 3)
            current = current.connections[next_conn_idx]
            
            # 动能摩擦损耗
            momentum -= 0.08 
            
        return current.label, path # 动能耗尽，被迫“满足”

# =============================================================================
# 4. 自动化实验
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("Structon v10.0: 动力学满足范式 (Kinetic Satisficing)")
    print("="*60)
    
    sys = DynamicVisionSystem()
    train_img, train_lbl, test_img, test_lbl = load_mnist()
    
    # 边走边学的进化过程
    for ep in range(10):
        t0 = time.time()
        # 采样训练
        sample_idxs = np.random.choice(len(train_img), 800)
        for idx in sample_idxs:
            sys.navigate(train_img[idx], str(train_lbl[idx]), train=True)
            
        # 满足率测试 (Satisficing Rate)
        correct = 0
        test_n = 200
        for i in range(test_n):
            pred, _ = sys.navigate(test_img[i])
            if pred == str(test_lbl[i]):
                correct += 1
        
        print(f"Epoch {ep+1:2d} | 满足准确率: {correct/test_n*100:4.1f}% | 耗时: {time.time()-t0:4.1f}s")

    # 结果路径涌现展示
    print("\n特征流向路径示例 (结果即目标):")
    for i in range(5):
        res, path = sys.navigate(test_img[i])
        status = "✓" if res == str(test_lbl[i]) else "✗"
        print(f"真:{test_lbl[i]} 预:{res} {status} | 路径: {' -> '.join(path)}")
