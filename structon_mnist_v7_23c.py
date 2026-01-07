"""
Structon Vision v7.23 - 正确的二分分形架构
==========================================

核心设计：
1. Atomic Structon: 两个动作 [是X, 不是X]，有 LRM
2. Wrapper Structon: 两个动作 [走左, 走右]，有 LRM
3. 结构: Wrapper = frozen子树 + 新Atomic
4. 学习: 只有 root Wrapper 的 LRM + 右边 Atomic 的 LRM
5. 生长: Atomic 满了 → 整个 root 被包进新 Wrapper

Workflow:
- 推理 Top-Down: 从 root 往下，Wrapper 路由，Atomic 判断
- 学习 Bottom-Up 生长: 满了 → 冻结 → 向上包装
- 只有未冻结的节点学习

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
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
        
        # 2. 端点位置 (9D)
        ep_regions = np.zeros(9)
        for y, x in endpoints:
            ry, rx = min(2, y // 10), min(2, x // 10)
            ep_regions[ry * 3 + rx] = 1.0
        features.extend(ep_regions)
        
        # 3. 交叉点位置 (3D)
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
    共振记忆 - 场景记忆
    
    存储: pattern → Q-values
    每条记忆 = 一个场景 + 该场景下各动作的 Q 值
    """
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int = 2,
        capacity: int = 30,
        key_dim: int = 16,
        temperature: float = 0.1,
        similarity_threshold: float = 0.8,
        learning_rate: float = 0.3
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.capacity = capacity
        self.key_dim = key_dim
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.learning_rate = learning_rate
        
        # 固定随机投影
        self.projector = np.random.randn(state_dim, key_dim).astype(np.float32)
        self.projector /= (np.linalg.norm(self.projector, axis=0, keepdims=True) + 1e-8)
        
        # 场景记忆
        self.keys: List[np.ndarray] = []      # 投影后的 pattern
        self.values: List[np.ndarray] = []    # Q-values
        self.access_counts: List[int] = []
        
        # 冻结
        self.frozen = False
        
        # 统计
        self.total_queries = 0
        self.total_writes = 0
    
    def encode(self, state: np.ndarray) -> np.ndarray:
        """投影编码"""
        key = state.astype(np.float32) @ self.projector
        norm = np.linalg.norm(key)
        if norm > 1e-8:
            key /= norm
        return key
    
    def query(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        查询记忆
        
        Returns:
            Q-values: [Q_action0, Q_action1]
            confidence: 置信度
        """
        self.total_queries += 1
        
        if len(self.keys) == 0:
            return np.zeros(self.n_actions, dtype=np.float32), 0.0
        
        key = self.encode(state)
        key_matrix = np.array(self.keys)
        
        # 计算相似度
        scores = key_matrix @ key
        max_score = float(np.max(scores))
        
        # Softmax 加权
        exp_scores = np.exp((scores - max_score) / self.temperature)
        weights = exp_scores / (np.sum(exp_scores) + 1e-8)
        
        # 加权得到 Q-values
        value_matrix = np.array(self.values)
        q_values = weights @ value_matrix
        
        # 更新访问计数
        best_idx = int(np.argmax(scores))
        self.access_counts[best_idx] += 1
        
        # 置信度
        confidence = max(0.0, min(1.0, (max_score - 0.5) * 2))
        
        return q_values.astype(np.float32), confidence
    
    def remember(self, state: np.ndarray, action: int, target_q: float) -> str:
        """
        写入记忆
        
        Args:
            state: 状态
            action: 执行的动作
            target_q: 目标 Q 值
        
        Returns:
            'update' / 'new' / 'frozen'
        """
        if self.frozen:
            return 'frozen'
        
        self.total_writes += 1
        key = self.encode(state)
        
        # 检查是否更新现有记忆
        if len(self.keys) > 0:
            key_matrix = np.array(self.keys)
            scores = key_matrix @ key
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            
            if best_score > self.similarity_threshold:
                # 更新现有记忆
                old_q = self.values[best_idx][action]
                self.values[best_idx][action] = old_q + self.learning_rate * (target_q - old_q)
                self.access_counts[best_idx] += 1
                return 'update'
        
        # 创建新记忆
        if len(self.keys) > 0:
            new_q, _ = self.query(state)
            new_q = new_q.copy()
        else:
            new_q = np.zeros(self.n_actions, dtype=np.float32)
        new_q[action] = target_q
        
        # 容量管理
        if len(self.keys) >= self.capacity:
            min_idx = int(np.argmin(self.access_counts))
            self.keys.pop(min_idx)
            self.values.pop(min_idx)
            self.access_counts.pop(min_idx)
        
        self.keys.append(key.copy())
        self.values.append(new_q)
        self.access_counts.append(1)
        
        return 'new'
    
    def freeze(self):
        self.frozen = True
    
    @property
    def size(self) -> int:
        return len(self.keys)
    
    def is_full(self) -> bool:
        return self.size >= self.capacity


# =============================================================================
# 4. Atomic Structon
# =============================================================================

class AtomicStructon:
    """
    Atomic Structon - 最底层的判断单元
    
    两个动作:
    - action[0]: "是X" → 返回 label
    - action[1]: "不是X" → 返回 None
    
    有 LRM，学习 "这个 pattern 是不是 X"
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        label: str,
        state_dim: int = 25,
        capacity: int = 30,
        key_dim: int = 16
    ):
        AtomicStructon._id_counter += 1
        self.id = f"A{AtomicStructon._id_counter:03d}"
        
        self.label = label
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        # LRM: 2 个动作 [是, 不是]
        self.lrm = ResonantMemory(
            state_dim=state_dim,
            n_actions=2,
            capacity=capacity,
            key_dim=key_dim
        )
        
        self.frozen = False
        
        # 统计
        self.total_executes = 0
        self.total_learns = 0
    
    def execute(self, state: np.ndarray) -> Tuple[Optional[str], float]:
        """
        执行判断
        
        Returns:
            label: 如果"是"，返回 label；如果"不是"，返回 None
            confidence: 置信度
        """
        self.total_executes += 1
        
        q_values, confidence = self.lrm.query(state)
        
        if q_values[0] > q_values[1]:  # action=0: 是
            return self.label, confidence
        else:  # action=1: 不是
            return None, confidence
    
    def learn(self, state: np.ndarray, true_label: str) -> str:
        """
        学习
        
        Args:
            state: 状态
            true_label: 真实标签
        
        Returns:
            状态信息
        """
        if self.frozen:
            return 'frozen'
        
        self.total_learns += 1
        
        is_mine = (true_label == self.label)
        
        if is_mine:
            # 这是我的！强化 "是"
            self.lrm.remember(state, action=0, target_q=1.0)
            return 'positive'
        else:
            # 不是我的！强化 "不是"
            self.lrm.remember(state, action=1, target_q=1.0)
            return 'negative'
    
    def freeze(self):
        self.frozen = True
        self.lrm.freeze()
    
    def is_full(self) -> bool:
        return self.lrm.is_full()
    
    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        icon = "❄️" if self.frozen else "🔥"
        print(f"{prefix}{icon} {self.id} [Atomic, label='{self.label}'] "
              f"mem:{self.lrm.size}/{self.capacity}")


# =============================================================================
# 5. Wrapper Structon
# =============================================================================

class WrapperStructon:
    """
    Wrapper Structon - 路由单元
    
    两个动作:
    - action[0]: 走左边 (frozen 子树)
    - action[1]: 走右边 (新 Atomic)
    
    有 LRM，学习 "这个 pattern 应该走哪边"
    
    关键改变：Wrapper 的 LRM 不随 freeze 停止学习！
    只有 Atomic 的 LRM 会真正停止学习。
    Wrapper 需要持续学习路由。
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        left_child: Union['WrapperStructon', AtomicStructon],
        right_child: AtomicStructon,
        state_dim: int = 25,
        capacity: int = 30,
        key_dim: int = 16
    ):
        WrapperStructon._id_counter += 1
        self.id = f"W{WrapperStructon._id_counter:03d}"
        
        self.children = [left_child, right_child]  # [frozen, active]
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        
        # LRM: 2 个动作 [走左, 走右]
        self.lrm = ResonantMemory(
            state_dim=state_dim,
            n_actions=2,
            capacity=capacity,
            key_dim=key_dim
        )
        
        self.frozen = False
        
        # 路由惊讶追踪（滑动窗口）
        self.routing_history = []  # 1=正确, 0=错误
        self.routing_window = 20   # 窗口大小
        
        # 统计
        self.total_executes = 0
        self.total_learns = 0
    
    def get_routing_surprise(self) -> float:
        """
        获取路由惊讶值
        
        惊讶 = 路由错误的比例
        低惊讶 = 路由学好了
        """
        if len(self.routing_history) < 5:
            return 1.0  # 还没学够，高惊讶
        
        recent = self.routing_history[-self.routing_window:]
        correct_rate = sum(recent) / len(recent)
        return 1.0 - correct_rate  # 惊讶 = 1 - 正确率
    
    def execute(self, state: np.ndarray) -> Tuple[Optional[str], float]:
        """
        执行路由 + 判断
        
        改进：两边都试，返回有结果的那边
        """
        self.total_executes += 1
        
        q_values, confidence = self.lrm.query(state)
        action = 0 if q_values[0] > q_values[1] else 1
        
        # 尝试选择的方向
        result, child_conf = self.children[action].execute(state)
        
        if result is not None:
            return result, (confidence + child_conf) / 2
        
        # 选择的方向说"不是"，尝试另一边
        other_action = 1 - action
        result, child_conf = self.children[other_action].execute(state)
        
        if result is not None:
            return result, child_conf * 0.8  # 稍微降低置信度
        
        # 两边都说"不是"
        return None, 0.0
    
    def learn(self, state: np.ndarray, true_label: str) -> str:
        """
        学习路由决策
        
        关键改变：
        1. Wrapper 的 LRM 始终学习（不管 frozen）
        2. 递归让子节点也学习
        3. 基于两边的执行结果来学习路由
        4. 追踪路由正确率（惊讶值）
        """
        self.total_learns += 1
        
        # 两边都执行，看哪边对
        result_left, conf_left = self.children[0].execute(state)
        result_right, conf_right = self.children[1].execute(state)
        
        left_correct = (result_left == true_label)
        right_correct = (result_right == true_label)
        
        # 当前路由决策
        q_values, _ = self.lrm.query(state)
        current_action = 0 if q_values[0] > q_values[1] else 1
        
        # 判断路由是否正确
        if current_action == 0:
            routing_correct = left_correct
        else:
            routing_correct = right_correct
        
        # 记录路由结果（用于计算惊讶值）
        self.routing_history.append(1 if routing_correct else 0)
        if len(self.routing_history) > self.routing_window * 2:
            self.routing_history = self.routing_history[-self.routing_window:]
        
        # 学习路由 - 关键：所有情况都要学习！
        if left_correct and not right_correct:
            # 只有左边对 → 强化走左
            self.lrm.remember(state, action=0, target_q=1.0)
            self.lrm.remember(state, action=1, target_q=-0.5)
        elif right_correct and not left_correct:
            # 只有右边对 → 强化走右
            self.lrm.remember(state, action=1, target_q=1.0)
            self.lrm.remember(state, action=0, target_q=-0.5)
        elif left_correct and right_correct:
            # 两边都对 → 基于置信度选择，强化更确定的那边
            # 这样可以学到更精确的路由
            if conf_left > conf_right:
                self.lrm.remember(state, action=0, target_q=0.8)
                self.lrm.remember(state, action=1, target_q=0.3)
            else:
                self.lrm.remember(state, action=1, target_q=0.8)
                self.lrm.remember(state, action=0, target_q=0.3)
        # 两边都错：不更新路由（让子节点去学习正确答案）
        
        # 递归：让子节点也学习
        # 左边：如果是 Wrapper，递归学习
        if isinstance(self.children[0], WrapperStructon):
            self.children[0].learn(state, true_label)
        elif isinstance(self.children[0], AtomicStructon) and not self.children[0].frozen:
            self.children[0].learn(state, true_label)
        
        # 右边：Atomic，如果未冻结就学习
        if not self.children[1].frozen:
            self.children[1].learn(state, true_label)
        
        return 'learned'
    
    def freeze(self):
        """
        冻结：只冻结 Atomic 的学习能力
        Wrapper 的路由能力保留（但结构上标记为 frozen）
        """
        self.frozen = True
        # 注意：不冻结 self.lrm！Wrapper 需要持续学习路由
        for child in self.children:
            child.freeze()
    
    def is_full(self) -> bool:
        """检查右边的 Atomic 是否满了"""
        return self.children[1].is_full()
    
    def get_active_atomic(self) -> AtomicStructon:
        """获取当前活跃的 Atomic"""
        return self.children[1]
    
    def depth(self) -> int:
        left_depth = self.children[0].depth() if hasattr(self.children[0], 'depth') else 1
        return 1 + left_depth
    
    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            if hasattr(child, 'count_nodes'):
                count += child.count_nodes()
            else:
                count += 1
        return count
    
    def total_memories(self) -> int:
        count = self.lrm.size
        for child in self.children:
            if hasattr(child, 'total_memories'):
                count += child.total_memories()
            elif hasattr(child, 'lrm'):
                count += child.lrm.size
        return count
    
    def print_tree(self, indent: int = 0):
        prefix = "  " * indent
        icon = "❄️" if self.frozen else "🔥"
        surprise = self.get_routing_surprise()
        print(f"{prefix}{icon} {self.id} [Wrapper] mem:{self.lrm.size}/{self.capacity} "
              f"surprise:{surprise:.2f}")
        print(f"{prefix}  ├─[0] (frozen subtree):")
        self.children[0].print_tree(indent + 2)
        print(f"{prefix}  └─[1] (active):")
        self.children[1].print_tree(indent + 2)


# =============================================================================
# 6. Vision System
# =============================================================================

class StructonVisionSystem:
    """
    Structon 视觉系统
    
    管理整棵树的生长和学习
    
    核心改变：
    - 多巴胺驱动：连续正确 = 熟练 = promote
    - 惊讶驱动：连续错误 = 新类别 = promote
    - Wrapper 惊讶：路由也要学好才能 promote
    - 不区分类别间/类别内：统一的增量结构
    - label 由训练数据决定，结构不知道"类别"
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        capacity: int = 30,
        key_dim: int = 16,
        mastery_threshold: int = 10,  # 连续正确多少次算熟练
        novelty_threshold: int = 5,   # 连续错误多少次算遇到新类别
        wrapper_surprise_threshold: float = 0.2  # Wrapper 惊讶阈值
    ):
        self.extractor = StateExtractor()
        self.state_dim = state_dim
        self.capacity = capacity
        self.key_dim = key_dim
        self.mastery_threshold = mastery_threshold
        self.novelty_threshold = novelty_threshold
        self.wrapper_surprise_threshold = wrapper_surprise_threshold
        
        # 初始：空
        self.root = None
        
        # 多巴胺系统
        self.consecutive_correct = 0  # 连续正确计数
        self.consecutive_wrong = 0    # 连续错误计数
        self.pending_promote = False  # 是否等待 promote
        self.last_wrong_label = None  # 最近错误时的 true_label
        
        # 统计
        self.train_count = 0
        self.correct_count = 0
        self.promote_count = 0
        self.surprise_history = []  # 惊讶值历史
    
    def _create_atomic(self, label: str) -> AtomicStructon:
        """创建新的 Atomic"""
        return AtomicStructon(
            label=label,
            state_dim=self.state_dim,
            capacity=self.capacity,
            key_dim=self.key_dim
        )
    
    def _create_wrapper(
        self,
        left_child: Union[WrapperStructon, AtomicStructon],
        right_child: AtomicStructon
    ) -> WrapperStructon:
        """创建新的 Wrapper"""
        return WrapperStructon(
            left_child=left_child,
            right_child=right_child,
            state_dim=self.state_dim,
            capacity=self.capacity,
            key_dim=self.key_dim
        )
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """预测"""
        if self.root is None:
            return "?", 0.0
        
        state = self.extractor.extract(image)
        result, confidence = self.root.execute(state)
        
        return result if result else "?", confidence
    
    def _promote(self, new_label: str):
        """
        向上生长
        
        1. 冻结当前 root
        2. 创建新的 Atomic
        3. 创建新的 Wrapper 包裹它们
        """
        self.promote_count += 1
        
        # 冻结当前 root
        if self.root is not None:
            self.root.freeze()
        
        # 创建新的 Atomic
        new_atomic = self._create_atomic(new_label)
        
        # 创建新的 Wrapper
        if self.root is None:
            # 第一次：root 就是 Atomic
            self.root = new_atomic
        else:
            # 后续：包成 Wrapper
            self.root = self._create_wrapper(self.root, new_atomic)
        
        # 重置计数
        self.consecutive_correct = 0
        self.consecutive_wrong = 0
        self.pending_promote = False
        self.last_wrong_label = None
        
        print(f"  → Promote! 新 Atomic label='{new_label}', "
              f"总晋升: {self.promote_count}")
    
    def _get_current_atomic_label(self) -> Optional[str]:
        """获取当前活跃 Atomic 的 label"""
        if self.root is None:
            return None
        if isinstance(self.root, AtomicStructon):
            return self.root.label
        elif isinstance(self.root, WrapperStructon):
            return self.root.children[1].label  # 右边是活跃的 Atomic
        return None
    
    def train_one(self, image: np.ndarray, true_label: str) -> Tuple[bool, float]:
        """
        训练一个样本
        
        Returns:
            correct: 是否正确
            surprise: 惊讶值 (0=无惊讶/正确, 1=惊讶/错误)
        """
        self.train_count += 1
        state = self.extractor.extract(image)
        
        # 初始化：第一个样本
        if self.root is None:
            self._promote(true_label)
        
        # 如果等待 promote（上一轮熟练了），检查 Wrapper 惊讶
        if self.pending_promote:
            # 检查 root Wrapper 的惊讶值
            wrapper_ready = True
            if isinstance(self.root, WrapperStructon):
                wrapper_surprise = self.root.get_routing_surprise()
                wrapper_ready = wrapper_surprise <= self.wrapper_surprise_threshold
                if not wrapper_ready:
                    # Wrapper 还没学好，继续学习，不 promote
                    pass  # 保持 pending_promote = True
                else:
                    self._promote(true_label)
            else:
                # root 是 Atomic，直接 promote
                self._promote(true_label)
        
        # 预测
        result, confidence = self.root.execute(state)
        correct = (result == true_label)
        
        # 计算惊讶值（多巴胺信号）
        surprise = 0.0 if correct else 1.0
        self.surprise_history.append(surprise)
        
        if correct:
            self.correct_count += 1
            self.consecutive_correct += 1
            self.consecutive_wrong = 0
            self.last_wrong_label = None
        else:
            self.consecutive_correct = 0
            self.consecutive_wrong += 1
            self.last_wrong_label = true_label
        
        # 学习
        status = self.root.learn(state, true_label)
        
        # 检查是否需要 promote
        
        # 情况1：熟练了（连续正确 >= threshold）
        if self.consecutive_correct >= self.mastery_threshold:
            # 检查 Wrapper 惊讶
            wrapper_ready = True
            if isinstance(self.root, WrapperStructon):
                wrapper_surprise = self.root.get_routing_surprise()
                wrapper_ready = wrapper_surprise <= self.wrapper_surprise_threshold
            
            if wrapper_ready:
                self.pending_promote = True
                self.consecutive_correct = 0
                print(f"  ★ 熟练! 连续正确 {self.mastery_threshold} 次，"
                      f"Wrapper 惊讶低，等待下一个样本触发 promote")
            else:
                # Wrapper 还没学好，继续学习
                self.consecutive_correct = 0  # 重置，但不 promote
                print(f"  ◆ Atomic 熟练但 Wrapper 惊讶高({wrapper_surprise:.2f})，继续学习路由")
        
        # 情况2：遇到新类别（连续错误 >= threshold，且错误的 label 一致）
        elif self.consecutive_wrong >= self.novelty_threshold:
            # 检查当前 Atomic 学的是不是不同的 label
            current_label = self._get_current_atomic_label()
            if current_label != true_label:
                print(f"  ✦ 新类别! 连续错误 {self.novelty_threshold} 次，"
                      f"当前学'{current_label}'，遇到'{true_label}'")
                self._promote(true_label)
        
        return correct, surprise
    
    def get_recent_surprise(self, window: int = 20) -> float:
        """获取最近的平均惊讶值"""
        if len(self.surprise_history) == 0:
            return 1.0
        recent = self.surprise_history[-window:]
        return sum(recent) / len(recent)
    
    def print_stats(self):
        print(f"\n{'='*60}")
        print(f"Structon Vision System v7.23 (多巴胺驱动)")
        print(f"{'='*60}")
        print(f"训练样本: {self.train_count}")
        if self.train_count > 0:
            print(f"训练准确率: {self.correct_count/self.train_count*100:.1f}%")
        print(f"晋升次数: {self.promote_count}")
        print(f"熟练阈值: 连续正确 {self.mastery_threshold} 次")
        print(f"最近惊讶值: {self.get_recent_surprise():.2f}")
        
        if self.root:
            if hasattr(self.root, 'depth'):
                print(f"深度: {self.root.depth()}")
            if hasattr(self.root, 'count_nodes'):
                print(f"节点数: {self.root.count_nodes()}")
            if hasattr(self.root, 'total_memories'):
                print(f"总记忆: {self.root.total_memories()}")
            
            print(f"\n=== 树结构 ===")
            self.root.print_tree()


# =============================================================================
# 7. 实验
# =============================================================================

def run_experiment(
    n_per_class: int = 100,
    n_test: int = 500,
    capacity: int = 30,
    key_dim: int = 16,
    mastery_threshold: int = 10,
    novelty_threshold: int = 5,
    wrapper_surprise_threshold: float = 0.2
):
    """运行实验"""
    print("=" * 70)
    print("Structon Vision v7.23c - Wrapper 惊讶驱动的二分分形架构")
    print("=" * 70)
    print(f"\n参数:")
    print(f"  capacity={capacity}, key_dim={key_dim}")
    print(f"  mastery_threshold={mastery_threshold} (连续正确→熟练)")
    print(f"  novelty_threshold={novelty_threshold} (连续错误→新类别)")
    print(f"  wrapper_surprise_threshold={wrapper_surprise_threshold} (Wrapper 惊讶阈值)")
    print(f"  每类最多训练: {n_per_class}, 测试: {n_test}")
    
    print("\n核心设计:")
    print("  1. Atomic: [是X, 不是X]，有 LRM")
    print("  2. Wrapper: [走左, 走右]，有 LRM + 惊讶追踪")
    print("  3. Promote 条件: Atomic 熟练 AND Wrapper 低惊讶")
    print("  4. 惊讶驱动: 连续错误 → promote（新类别）")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    system = StructonVisionSystem(
        state_dim=25,
        capacity=capacity,
        key_dim=key_dim,
        mastery_threshold=mastery_threshold,
        novelty_threshold=novelty_threshold,
        wrapper_surprise_threshold=wrapper_surprise_threshold
    )
    
    print(f"\n=== 按类别顺序训练（模拟人类学习）===")
    t0 = time.time()
    
    total_samples_used = 0
    
    for digit in range(10):
        indices = np.where(train_labels == digit)[0][:n_per_class]
        
        print(f"\n--- 开始喂数字 {digit} 的样本 ---")
        
        digit_correct = 0
        samples_used = 0
        
        for i, idx in enumerate(indices):
            correct, surprise = system.train_one(
                train_images[idx],
                str(digit)
            )
            samples_used += 1
            total_samples_used += 1
            
            if correct:
                digit_correct += 1
            
            # 如果已经 promote 到下一个（说明这个类别学熟了），可以提前结束
            # 但我们继续喂数据，让结构自己决定
        
        acc = digit_correct / samples_used * 100
        print(f"  数字 {digit}: 用了 {samples_used} 样本, 准确率 {acc:.1f}%")
    
    print(f"\n训练完成: {time.time()-t0:.1f}秒")
    print(f"总共使用样本: {total_samples_used}")
    
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
    parser.add_argument('--per-class', type=int, default=100)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--capacity', type=int, default=30)
    parser.add_argument('--key-dim', type=int, default=16)
    parser.add_argument('--mastery', type=int, default=10, 
                        help='连续正确多少次算熟练')
    parser.add_argument('--novelty', type=int, default=5,
                        help='连续错误多少次算遇到新类别')
    parser.add_argument('--wrapper-surprise', type=float, default=0.2,
                        help='Wrapper 惊讶阈值，低于此值才能 promote')
    args = parser.parse_args()
    
    run_experiment(
        n_per_class=args.per_class,
        n_test=args.test,
        capacity=args.capacity,
        key_dim=args.key_dim,
        mastery_threshold=args.mastery,
        novelty_threshold=args.novelty,
        wrapper_surprise_threshold=args.wrapper_surprise
    )
