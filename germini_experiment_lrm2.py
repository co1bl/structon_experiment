"""
Structon HOG-RL (v5)
====================
特性：
1.  **HOG Feature**: 使用方向梯度直方图替代简单的像素统计，大幅提升形状识别能力。
2.  **Commander-Scout**: 保持指挥官(Global)决策机制。
3.  **Resize Trick**: 将 Top/Bottom 视野 resize 到统一尺寸，使局部特征获得更高的分辨率。
"""

import numpy as np
import time
import gzip
import os
import urllib.request
from typing import List, Tuple, Dict, Any, Optional, Set

# try to import skimage, else warn
try:
    from skimage.feature import hog
    from skimage.transform import resize
except ImportError:
    print("Error: 需要安装 scikit-image。请运行: pip install scikit-image")
    exit()

# =============================================================================
# 1. 记忆体 (保持不变)
# =============================================================================
class ImprovedResonantMemory:
    def __init__(self, state_dim, n_actions, capacity=200, temperature=0.1, learning_rate=0.1):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.keys = []
        self.values = []
        self.access_counts = []
    
    def _encode(self, state): return state
    
    def query(self, state):
        if not self.keys: return np.zeros(self.n_actions, dtype=np.float32), 0.0
        # Dot product similarity
        scores = np.array(self.keys) @ state
        max_score = np.max(scores)
        
        # Softmax
        weights = np.exp((scores - max_score) / self.temperature)
        weights /= np.sum(weights) + 1e-8
        
        q_values = weights @ np.array(self.values)
        self.access_counts[int(np.argmax(weights))] += 1
        return q_values, float(max_score)
    
    def remember(self, state, action, target_q):
        if self.keys:
            scores = np.array(self.keys) @ state
            best_idx = int(np.argmax(scores))
            # HOG 特征非常精细，我们可以提高相似度阈值
            if scores[best_idx] > 0.96: 
                self.values[best_idx][action] += self.learning_rate * (target_q - self.values[best_idx][action])
                self.access_counts[best_idx] += 1
                return
        
        if self.keys: new_q = self.query(state)[0].copy()
        else: new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = target_q
        
        if len(self.keys) >= self.capacity:
            idx = int(np.argmin(self.access_counts))
            self.keys.pop(idx); self.values.pop(idx); self.access_counts.pop(idx)
        
        self.keys.append(state); self.values.append(new_q); self.access_counts.append(1)

# =============================================================================
# 2. HOG 特征提取器 (全新替换)
# =============================================================================
class HOGExtractor:
    def __init__(self):
        # 预热一次 HOG 以确定维度
        dummy = np.zeros((32, 32))
        vec = self._compute_hog(dummy)
        self.feature_dim = len(vec)
        print(f"HOG Extractor initialized. Feature dimension: {self.feature_dim}")
        
    def _compute_hog(self, img):
        # 配置 HOG 参数
        # orientations=9: 9个方向
        # pixels_per_cell=(8,8): 32x32 -> 4x4 cells
        # cells_per_block=(2,2): 2x2 blocks normalization
        return hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)

    def extract(self, image, concern='global'):
        # 1. 预处理
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        
        # 2. 裁剪关注区域
        if concern == 'top': 
            crop = img[:h//2, :]
        elif concern == 'bottom': 
            crop = img[h//2:, :]
        else: # global
            crop = img
            
        # 3. [关键] Resize 到固定大小 (32x32)
        # 这让 "局部" 看起来也像是一张完整的图，增加了局部细节的分辨率
        # 同时也保证了输出向量长度一致
        img_resized = resize(crop, (32, 32), anti_aliasing=True)
        
        # 4. 计算 HOG
        state = self._compute_hog(img_resized)
        
        # HOG 已经做了 block norm，但为了安全再做一次整体归一化
        norm = np.linalg.norm(state)
        return state / (norm + 1e-6)

# =============================================================================
# 3. 层级化 Structon (Commander & Scout) - 保持 v4 逻辑
# =============================================================================
class HierarchicalStructon:
    def __init__(self, label: str, concern: str, feature_dim: int, capacity=200):
        self.label = label
        self.concern = concern
        self.unique_id = f"{label}_{concern}"
        self.is_commander = (concern == 'global')
        
        self.neighbors = []
        self.memory = None
        self.capacity = capacity
        self.feature_dim = feature_dim
        
    def connect(self, all_structons, n_connections=6):
        others = [s for s in all_structons if s.unique_id != self.unique_id]
        siblings = [s for s in others if s.label == self.label]
        
        commander = [s for s in siblings if s.concern == 'global']
        rest_siblings = [s for s in siblings if s.concern != 'global']
        strangers = [s for s in others if s.label != self.label]
        
        self.neighbors = []
        self.neighbors.extend(commander)
        self.neighbors.extend(rest_siblings)
        
        needed = max(0, n_connections - len(self.neighbors))
        if needed > 0:
            idx = np.random.choice(len(strangers), min(needed, len(strangers)), replace=False)
            self.neighbors.extend([strangers[i] for i in idx])
            
        self.memory = ImprovedResonantMemory(self.feature_dim, 1+len(self.neighbors), self.capacity)

    def decide(self, image, extractor, visited, epsilon):
        state = extractor.extract(image, self.concern)
        q, _ = self.memory.query(state)
        
        valid_actions = []
        if self.is_commander:
            valid_actions.append(0) # Claim
            
        for i, n in enumerate(self.neighbors):
            if n.unique_id not in visited:
                valid_actions.append(i + 1)
        
        if not valid_actions: 
            return 0 if self.is_commander else np.random.randint(1, len(self.neighbors)+1)

        if np.random.random() < epsilon:
            return np.random.choice(valid_actions)
        else:
            masked_q = np.full_like(q, -np.inf)
            masked_q[valid_actions] = q[valid_actions]
            return int(np.argmax(masked_q))

    def get_commander_action(self):
        for i, n in enumerate(self.neighbors):
            if n.label == self.label and n.concern == 'global':
                return i + 1
        return None

# =============================================================================
# 4. 系统
# =============================================================================
class HierarchicalHOGSystem:
    def __init__(self):
        self.extractor = HOGExtractor()
        self.structons = []
        
    def build(self):
        concerns = ['global', 'top', 'bottom']
        labels = [str(i) for i in range(10)]
        # 传入 extractor.feature_dim
        self.structons = [HierarchicalStructon(l, c, self.extractor.feature_dim) for l in labels for c in concerns]
        for s in self.structons: s.connect(self.structons)
        print(f"Build: {len(self.structons)} Agents using HOG Features (Dim: {self.extractor.feature_dim})")

    def train_episode(self, image, true_label, epsilon):
        current = self.structons[np.random.randint(len(self.structons))]
        trajectory = []
        visited = set()
        max_hops = 15
        final_result = "timeout"
        
        for _ in range(max_hops):
            visited.add(current.unique_id)
            
            # Teacher Forcing
            if current.label == true_label:
                # 提取一次 state，避免重复计算
                s_vec = self.extractor.extract(image, current.concern)
                if current.is_commander:
                    current.memory.remember(s_vec, 0, 1.0)
                else:
                    cmd_action = current.get_commander_action()
                    if cmd_action:
                        current.memory.remember(s_vec, cmd_action, 1.0)

            action = current.decide(image, self.extractor, visited, epsilon)
            trajectory.append((current, action))
            
            if action == 0:
                if current.label == true_label: final_result = "correct"
                else: final_result = "wrong"
                break
            else:
                current = current.neighbors[action-1]
        
        # Reward
        r = 1.0 if final_result == "correct" else (-2.0 if final_result == "wrong" else -0.5)
        running_r = r
        for s, a in reversed(trajectory):
            s.memory.remember(self.extractor.extract(image, s.concern), a, running_r)
            running_r = running_r * 0.95 - 0.02
        return final_result

    def predict(self, image):
        current = self.structons[np.random.randint(len(self.structons))]
        path = []
        visited = set()
        for _ in range(20):
            path.append(current.unique_id)
            visited.add(current.unique_id)
            action = current.decide(image, self.extractor, visited, 0.0)
            if action == 0: return current.label, path
            current = current.neighbors[action-1]
        return "unknown", path

# =============================================================================
# 5. Main
# =============================================================================
def main():
    print("="*60 + "\nStructon HOG-RL (Hierarchical + HOG)\n" + "="*60)
    
    # Load Data
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {'img': 'train-images-idx3-ubyte.gz', 'lbl': 'train-labels-idx1-ubyte.gz', 
             'timg': 't10k-images-idx3-ubyte.gz', 'tlbl': 't10k-labels-idx1-ubyte.gz'}
    data = {}
    os.makedirs(os.path.expanduser('~/.mnist'), exist_ok=True)
    for k, v in files.items():
        fp = os.path.join(os.path.expanduser('~/.mnist'), v)
        if not os.path.exists(fp): urllib.request.urlretrieve(base_url + v, fp)
        with gzip.open(fp, 'rb') as f:
            data[k] = np.frombuffer(f.read()[16 if 'img' in k else 8:], dtype=np.uint8).reshape(-1,28,28) if 'img' in k else np.frombuffer(f.read()[8:], dtype=np.uint8)

    system = HierarchicalHOGSystem()
    system.build()
    
    # Training Loop
    # HOG 计算比之前的简单像素特征慢，所以我们稍微减少每轮的样本数，但因为特征好，收敛应该更快
    print("\nTraining (10 Epochs)...")
    t0 = time.time()
    for ep in range(10):
        # 每次训练 1500 个样本
        idx = np.random.choice(len(data['img']), 1500, replace=False)
        stats = {'correct':0, 'wrong':0, 'timeout':0}
        eps = max(0.05, 0.5 * (0.6 ** ep))
        
        for i in idx:
            res = system.train_episode(data['img'][i], str(data['lbl'][i]), eps)
            stats[res] += 1
        
        acc = stats['correct']/1500*100
        print(f"Epoch {ep+1:02d}: {acc:.1f}% (C:{stats['correct']} W:{stats['wrong']} T:{stats['timeout']}) | Eps:{eps:.2f}")

    print(f"Total Time: {time.time()-t0:.1f}s")

    # Testing
    print("\nTesting...")
    correct = 0
    test_idx = np.random.choice(len(data['timg']), 500, replace=False)
    for i, idx in enumerate(test_idx):
        pred, path = system.predict(data['timg'][idx])
        if pred == str(data['tlbl'][idx]): correct += 1
        if i < 10: 
            mark = "✓" if pred == str(data['tlbl'][idx]) else "✗"
            print(f"True: {data['tlbl'][idx]} | Pred: {pred} {mark} | Path: {' -> '.join(path)}")
            
    print(f"\nFinal Accuracy: {correct/500*100:.1f}%")

if __name__ == "__main__":
    main()
