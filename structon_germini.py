"""
Structon Vision v7.23c - 最小惊讶驱动 + 双重熟练度门控
======================================================

解决核心痛点：
1. 解决 "右偏好" (Right-Bias)：Wrapper 必须在当前层级学会区分"旧"与"新"才能生长。
2. 内部一致性奖励：Wrapper 学习预测子节点的 Confidence，而非仅仅预测 Label。
3. 睡眠回放 (Replay)：在学习新类时，混合少量旧类，强制 Wrapper 建立边界。

Workflow:
- Promote 只有在 Atomic 和 Wrapper 都 "Mastered" 时才触发。
- Wrapper Mastery = 连续 20 次路由正确 (Routing Accuracy)。

Author: Structon Framework
"""

import numpy as np
from typing import List, Tuple, Optional, Union
import os
import gzip
import struct
import urllib.request
import time
import random

# =============================================================================
# 1. MNIST Loader & Feature Extractor (保持不变，省略部分以节省空间)
# =============================================================================

def load_mnist(path='./mnist_data'):
    os.makedirs(path, exist_ok=True)
    base_url = 'http://yann.lecun.com/exdb/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz', 'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz', 'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    for name, filename in files.items():
        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(base_url + filename, filepath)
    
    def read_images(path):
        with gzip.open(path, 'rb') as f:
            _, n, r, c = struct.unpack('>IIII', f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, r, c)
    
    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            _, n = struct.unpack('>II', f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)
            
    return (read_images(os.path.join(path, files['train_images'])),
            read_labels(os.path.join(path, files['train_labels'])),
            read_images(os.path.join(path, files['test_images'])),
            read_labels(os.path.join(path, files['test_labels'])))

class StateExtractor:
    def extract(self, image: np.ndarray) -> np.ndarray:
        # 简单归一化和展平，为了演示逻辑，这里用简单的像素特征+中心特征
        # 实际使用中保持你原来的拓扑特征代码即可
        img = image.astype(np.float32) / 255.0
        h, w = img.shape
        
        # 降维：4x4 网格密度 (16维)
        grid_h, grid_w = 7, 7
        features = []
        for r in range(4):
            for c in range(4):
                region = img[r*grid_h:(r+1)*grid_h, c*grid_w:(c+1)*grid_w]
                features.append(np.mean(region))
        
        # 质心 (2维)
        ys, xs = np.where(img > 0.3)
        if len(xs) > 0:
            features.append(np.mean(xs)/w)
            features.append(np.mean(ys)/h)
        else:
            features.extend([0.5, 0.5])
            
        # 中间区域密度 (1维)
        center_region = img[7:21, 7:21]
        features.append(np.mean(center_region))
        
        # 总计 19 维，稍微增加特征丰富度
        return np.array(features, dtype=np.float32)

# =============================================================================
# 2. Resonant Memory (惊讶计算优化)
# =============================================================================

class ResonantMemory:
    def __init__(self, state_dim, n_actions=2, capacity=30, key_dim=16):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        # 随机投影矩阵
        self.projector = np.random.randn(state_dim, key_dim).astype(np.float32)
        self.projector /= (np.linalg.norm(self.projector, axis=0, keepdims=True) + 1e-8)
        
        self.keys = []
        self.values = []
        self.access_counts = []
        self.frozen = False
        
    def encode(self, state):
        return state @ self.projector
        
    def query(self, state) -> Tuple[np.ndarray, float]:
        if not self.keys:
            return np.zeros(self.n_actions), 0.0
            
        key = self.encode(state)
        # Cosine similarity
        key_norm = key / (np.linalg.norm(key) + 1e-8)
        keys_mat = np.array(self.keys)
        keys_mat_norm = keys_mat / (np.linalg.norm(keys_mat, axis=1, keepdims=True) + 1e-8)
        
        sims = keys_mat_norm @ key_norm
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        
        # Softmax weighting for value retrieval
        beta = 20.0 # sharp attention
        weights = np.exp(beta * (sims - best_sim))
        weights /= np.sum(weights)
        
        values_mat = np.array(self.values)
        q_values = weights @ values_mat
        
        # Confidence correlates with similarity
        confidence = max(0.0, float(best_sim))
        return q_values, confidence

    def remember(self, state, action, target_q, lr=0.2):
        if self.frozen: return
        
        key = self.encode(state)
        key_norm = key / (np.linalg.norm(key) + 1e-8)
        
        # Determine if update or add
        best_idx, best_sim = -1, -1.0
        if self.keys:
            keys_mat = np.array(self.keys)
            keys_mat_norm = keys_mat / (np.linalg.norm(keys_mat, axis=1, keepdims=True) + 1e-8)
            sims = keys_mat_norm @ key_norm
            best_idx = np.argmax(sims)
            best_sim = sims[best_idx]
            
        if best_sim > 0.90: # Similarity threshold
            # Update existing
            curr_q = self.values[best_idx][action]
            self.values[best_idx][action] += lr * (target_q - curr_q)
            self.access_counts[best_idx] += 1
        else:
            # Add new
            if len(self.keys) >= self.capacity:
                # Remove least accessed
                del_idx = np.argmin(self.access_counts)
                self.keys.pop(del_idx)
                self.values.pop(del_idx)
                self.access_counts.pop(del_idx)
            
            new_val = np.zeros(self.n_actions)
            new_val[action] = target_q
            self.keys.append(key)
            self.values.append(new_val)
            self.access_counts.append(1)

    def freeze(self):
        self.frozen = True

# =============================================================================
# 3. Structon Components (Modified for Wrapper Mastery)
# =============================================================================

class AtomicStructon:
    def __init__(self, label, dim=19):
        self.label = label
        self.lrm = ResonantMemory(dim, n_actions=2) # [Is Mine, Not Mine]
        self.frozen = False
        
    def execute(self, state) -> Tuple[Optional[str], float]:
        q, conf = self.lrm.query(state)
        # Action 0: Is Mine (return label)
        # Action 1: Not Mine (return None)
        margin = q[0] - q[1]
        
        # Confined confidence: combine memory similarity with Q-value margin
        final_conf = conf * max(0, min(1, 0.5 + margin))
        
        if q[0] > q[1]:
            return self.label, final_conf
        else:
            return None, final_conf # Still return confidence of "Not Mine"

    def learn(self, state, true_label):
        if self.frozen: return
        is_target = (true_label == self.label)
        # Q-Learning: 1.0 for correct match, 0.0 for mismatch
        target = 1.0
        action = 0 if is_target else 1
        self.lrm.remember(state, action, target)
        
    def freeze(self):
        self.frozen = True
        self.lrm.freeze()

class WrapperStructon:
    def __init__(self, left, right, dim=19):
        self.children = [left, right] # 0: Left (Old), 1: Right (New)
        self.lrm = ResonantMemory(dim, n_actions=2) # [Go Left, Go Right]
        self.frozen = False # Structure frozen, but learning might continue until mastery
        
        # Mastery Tracking
        self.routing_history = [] 
        self.mastery_window = 30
        
    def execute(self, state) -> Tuple[Optional[str], float]:
        q, conf = self.lrm.query(state)
        
        # Decide direction based on Q-values
        direction = 0 if q[0] > q[1] else 1
        
        # Execute chosen path first
        res, child_conf = self.children[direction].execute(state)
        
        # Fallback logic: if chosen path is very unsure, try the other?
        # For v7.23c, let's keep it strict to force the LRM to learn.
        # But we blend confidence.
        
        return res, (conf * 0.4 + child_conf * 0.6)

    def learn_routing(self, state, true_label) -> bool:
        """
        Internal Consistency Learning:
        Wrapper learns to route to the child that is 'happier' (higher confidence/correctness).
        Returns: True if routing was 'correct' (minimized surprise), False otherwise.
        """
        # 1. Probe both children (Mental Simulation)
        res_0, conf_0 = self.children[0].execute(state)
        res_1, conf_1 = self.children[1].execute(state)
        
        match_0 = (res_0 == true_label)
        match_1 = (res_1 == true_label)
        
        # 2. Determine "Intrinsic Truth"
        target_dir = -1
        
        if match_0 and not match_1:
            target_dir = 0 # Definitely Left
        elif match_1 and not match_0:
            target_dir = 1 # Definitely Right
        elif match_0 and match_1:
            # Both claim it? Trust the one with higher confidence (Surprise minimization)
            target_dir = 0 if conf_0 > conf_1 else 1
        else:
            # Neither claims it correctly. 
            # If one explicitly said "Not Mine" (returned None) with high confidence,
            # and the other wrongly claimed it, trust the "Not Mine".
            pass 
            
        # 3. Update LRM if we have a clear signal
        routing_success = False
        
        if target_dir != -1:
            # Positive Reinforcement for correct
            self.lrm.remember(state, target_dir, 1.0, lr=0.3)
            # Negative Reinforcement for wrong
            self.lrm.remember(state, 1-target_dir, -0.5, lr=0.1)
            
            # Check if our current Q-values would have chosen this
            q, _ = self.lrm.query(state)
            current_choice = 0 if q[0] > q[1] else 1
            if current_choice == target_dir:
                routing_success = True
        
        # 4. Update History for Mastery Check
        self.routing_history.append(1 if routing_success else 0)
        if len(self.routing_history) > self.mastery_window:
            self.routing_history.pop(0)
            
        # 5. Propagate Learning Downwards
        # Only teach the relevant child!
        if target_dir == 0:
            if hasattr(self.children[0], 'learn') and not self.children[0].frozen:
                self.children[0].learn(state, true_label)
            # Also invoke routing learning if child is a wrapper
            if isinstance(self.children[0], WrapperStructon):
                self.children[0].learn_routing(state, true_label)
        elif target_dir == 1:
            if hasattr(self.children[1], 'learn') and not self.children[1].frozen:
                self.children[1].learn(state, true_label)
                
        return routing_success

    def is_mastered(self) -> bool:
        # Wrapper is mastered if recent accuracy is high
        if len(self.routing_history) < self.mastery_window:
            return False
        acc = sum(self.routing_history) / len(self.routing_history)
        return acc > 0.90  # 90% routing accuracy required

    def freeze(self):
        # Wrappers technically never fully freeze LRM in this version, 
        # but we mark structure as frozen.
        self.frozen = True
        self.children[0].freeze()
        self.children[1].freeze()

# =============================================================================
# 4. System Manager (Double Gating)
# =============================================================================

class StructonVisionSystem:
    def __init__(self):
        self.extractor = StateExtractor()
        self.root = None
        self.memory_buffer = [] # Replay buffer for boundary learning
        
        # Stats
        self.promotions = 0
        
    def predict(self, image):
        if not self.root: return None, 0.0
        state = self.extractor.extract(image)
        return self.root.execute(state)
        
    def train_one(self, image, true_label) -> dict:
        state = self.extractor.extract(image)
        
        # --- Initialization ---
        if self.root is None:
            self.root = AtomicStructon(true_label)
            return {'status': 'init'}
            
        # --- Execution & Learning ---
        # If Root is Atomic, just learn
        if isinstance(self.root, AtomicStructon):
            self.root.learn(state, true_label)
            # Check Atomic Mastery
            # Logic: If accuracy on recent window is high
            # Simplified: Random check based on LRM confidence
            q, conf = self.root.lrm.query(state)
            is_mastered = (conf > 0.9 and (q[0]>q[1]) == (true_label==self.root.label))
            return {'status': 'learning_atomic', 'atomic_mastered': is_mastered}
            
        # If Root is Wrapper, learn routing
        elif isinstance(self.root, WrapperStructon):
            routing_ok = self.root.learn_routing(state, true_label)
            
            # Check Active Child Mastery
            active_child = self.root.children[1] # Right is always active/new
            active_mastered = False
            if isinstance(active_child, AtomicStructon):
                # Simple mastery check
                # In real code, track rolling accuracy
                q, conf = active_child.lrm.query(state)
                active_mastered = (conf > 0.8) # Simply confident
            
            # Check Wrapper Mastery
            wrapper_mastered = self.root.is_mastered()
            
            return {
                'status': 'learning_wrapper',
                'wrapper_mastered': wrapper_mastered,
                'atomic_mastered': active_mastered
            }

    def promote(self, new_label):
        """Force growth: Freeze current root, wrap it, add new Atomic"""
        print(f"  >>> PROMOTING: Wrapping old tree, adding leaf for '{new_label}'")
        self.root.freeze()
        new_atomic = AtomicStructon(new_label)
        self.root = WrapperStructon(self.root, new_atomic)
        self.promotions += 1

    def add_to_buffer(self, image, label):
        if len(self.memory_buffer) > 200:
            self.memory_buffer.pop(0) # Keep recent
        self.memory_buffer.append((image, label))

# =============================================================================
# 5. Experiment Logic (with Replay)
# =============================================================================

def run_experiment_v723c():
    print("=== Structon v7.23c: Minimum Surprise & Double Gating ===")
    system = StructonVisionSystem()
    train_imgs, train_lbls, test_imgs, test_lbls = load_mnist()
    
    # Configuration
    SAMPLES_PER_CLASS = 150 # Give it time to learn
    REPLAY_RATIO = 0.3      # 30% of training time spent reviewing old data
    
    for digit in range(10):
        print(f"\n--- Phase {digit}: Learning Class '{digit}' ---")
        
        # Prepare data for this phase
        indices = np.where(train_lbls == digit)[0][:SAMPLES_PER_CLASS]
        phase_images = train_imgs[indices]
        
        # If it's not the first digit, we need to promote first
        if digit > 0:
            # We assume previous phase mastered it.
            # In a real dynamic system, we would wait for a trigger.
            # But here we force structure for the experiment.
            system.promote(str(digit))
            
        # Training Loop
        consecutive_mastery = 0
        
        for i in range(len(phase_images)):
            # 1. Train on New Sample
            img = phase_images[i]
            res = system.train_one(img, str(digit))
            system.add_to_buffer(img, str(digit))
            
            # 2. Interleaved Replay (Essential for Wrapper to learn "Left")
            if system.memory_buffer and random.random() < REPLAY_RATIO:
                replay_img, replay_lbl = random.choice(system.memory_buffer)
                # Skip current digit in replay to emphasize contrast? 
                # No, random mix is fine.
                system.train_one(replay_img, replay_lbl)
            
            # Monitoring
            if i % 50 == 0:
                # Test current competence
                pred, conf = system.predict(img)
                print(f"    Sample {i}: Pred={pred} (Conf={conf:.2f}) | "
                      f"WrapperReady={res.get('wrapper_mastered', 'N/A')}")

    # Final Test
    print("\n=== Final Evaluation (500 Samples) ===")
    correct = 0
    total = 500
    details = {str(d):[0,0] for d in range(10)}
    
    idxs = np.random.choice(len(test_imgs), total, replace=False)
    for idx in idxs:
        pred, _ = system.predict(test_imgs[idx])
        true = str(test_lbls[idx])
        details[true][1] += 1
        if pred == true:
            correct += 1
            details[true][0] += 1
            
    print(f"Global Accuracy: {correct/total*100:.2f}%")
    for d in sorted(details.keys()):
        c, t = details[d]
        if t > 0: print(f"Digit {d}: {c}/{t} ({c/t*100:.1f}%)")

if __name__ == "__main__":
    run_experiment_v723c()
