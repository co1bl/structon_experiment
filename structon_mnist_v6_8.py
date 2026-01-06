"""
Structon Vision v6.8 - Overlapping Grid + Hole Detection
=========================================================

新增改进：

1. 重叠网格 (Overlapping Grid)
   - 之前：3x3硬性切分，笔画可能被撕裂
   - 现在：网格有30%重叠，确保笔画完整呈现

2. 空洞检测 (Hole Detection via Euler Number)
   - 之前：只有端点和交叉点
   - 现在：增加空洞数量检测
   - 关键区分：
     * 0: 1个洞
     * 8: 2个洞
     * 6, 9: 1个洞
     * 1, 2, 3, 4, 5, 7: 0个洞

欧拉数公式：χ = V - E + F = 连通分量数 - 空洞数 + 1
空洞数 = 连通分量数 - χ + 1

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Types
# =============================================================================

class AtomType(Enum):
    HORIZONTAL = "H"
    VERTICAL = "V"
    DIAGONAL_R = "DR"
    DIAGONAL_L = "DL"
    MIXED = "M"
    EMPTY = "empty"


class Region(Enum):
    TOP_LEFT = "TL"
    TOP = "T"
    TOP_RIGHT = "TR"
    LEFT = "L"
    CENTER = "C"
    RIGHT = "R"
    BOTTOM_LEFT = "BL"
    BOTTOM = "B"
    BOTTOM_RIGHT = "BR"


# =============================================================================
# 2. Skeletonization
# =============================================================================

def skeletonize(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Zhang-Suen骨架化"""
    binary = (image > threshold).astype(np.uint8)
    skeleton = binary.copy()
    h, w = skeleton.shape
    
    def neighbors(y, x):
        return [
            skeleton[y-1, x], skeleton[y-1, x+1], skeleton[y, x+1], skeleton[y+1, x+1],
            skeleton[y+1, x], skeleton[y+1, x-1], skeleton[y, x-1], skeleton[y-1, x-1]
        ]
    
    def transitions(n):
        n = n + [n[0]]
        return sum(n[i] == 0 and n[i+1] == 1 for i in range(8))
    
    changed = True
    iterations = 0
    
    while changed and iterations < 100:
        changed = False
        iterations += 1
        
        # Step 1
        to_remove = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                P = neighbors(y, x)
                B = sum(P)
                A = transitions(P)
                if 2 <= B <= 6 and A == 1 and P[0]*P[2]*P[4] == 0 and P[2]*P[4]*P[6] == 0:
                    to_remove.append((y, x))
        
        for y, x in to_remove:
            skeleton[y, x] = 0
            changed = True
        
        # Step 2
        to_remove = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                P = neighbors(y, x)
                B = sum(P)
                A = transitions(P)
                if 2 <= B <= 6 and A == 1 and P[0]*P[2]*P[6] == 0 and P[0]*P[4]*P[6] == 0:
                    to_remove.append((y, x))
        
        for y, x in to_remove:
            skeleton[y, x] = 0
            changed = True
    
    return skeleton


# =============================================================================
# 3. Topology Analysis with Hole Detection
# =============================================================================

def count_connected_components(binary: np.ndarray, foreground: bool = True) -> int:
    """
    计算连通分量数量 (4-连通)
    
    Args:
        binary: 二值图像
        foreground: True=计算前景(1)的连通分量, False=计算背景(0)的连通分量
    """
    h, w = binary.shape
    target = 1 if foreground else 0
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    
    def bfs(start_y, start_x):
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-连通
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if binary[ny, nx] == target and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    
    for y in range(h):
        for x in range(w):
            if binary[y, x] == target and not visited[y, x]:
                bfs(y, x)
                count += 1
    
    return count


def count_holes(binary: np.ndarray) -> int:
    """
    计算空洞数量（被前景包围的背景区域）
    
    方法：
    1. 找背景连通分量
    2. 减去与边界相连的背景（外部背景）
    3. 剩下的就是空洞
    """
    h, w = binary.shape
    
    # 创建扩展图像（加1像素边框），确保外部背景连通
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = binary
    
    # 从角落flood fill标记外部背景
    visited = np.zeros_like(padded, dtype=bool)
    
    def flood_fill_background(start_y, start_x):
        """标记从边界可达的背景"""
        queue = [(start_y, start_x)]
        visited[start_y, start_x] = True
        while queue:
            y, x = queue.pop(0)
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                    if padded[ny, nx] == 0 and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    
    # 从四个角开始flood fill
    flood_fill_background(0, 0)
    
    # 现在计算未被访问的背景像素区域（这些是空洞）
    # 统计空洞的连通分量数
    n_holes = 0
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] == 0 and not visited[y, x]:
                # 找到一个空洞，flood fill标记它
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


def analyze_topology(skeleton: np.ndarray, original: np.ndarray) -> Tuple[int, int, int]:
    """
    分析拓扑特征
    
    Returns:
        n_endpoints: 端点数
        n_junctions: 交叉点数
        n_holes: 空洞数（使用原图，不是骨架）
    """
    h, w = skeleton.shape
    
    n_endpoints = 0
    n_junctions = 0
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 0:
                continue
            
            # 8邻域
            ring = [
                skeleton[y-1, x], skeleton[y-1, x+1], skeleton[y, x+1], skeleton[y+1, x+1],
                skeleton[y+1, x], skeleton[y+1, x-1], skeleton[y, x-1], skeleton[y-1, x-1]
            ]
            
            n_neighbors = sum(ring)
            crossings = sum(ring[i] != ring[(i+1) % 8] for i in range(8)) // 2
            
            if n_neighbors == 1 or crossings == 1:
                n_endpoints += 1
            if crossings >= 3:
                n_junctions += 1
    
    # 空洞检测（使用原图的二值版本，不是骨架）
    binary = (original > 0.3).astype(np.uint8)
    n_holes = count_holes(binary)
    
    return n_endpoints, n_junctions, n_holes


# =============================================================================
# 4. Overlapping Grid Feature Extraction
# =============================================================================

class OverlappingGrid:
    """
    重叠网格
    
    3x3网格，每个网格有30%重叠
    确保笔画不会被切断
    """
    
    def __init__(self, overlap: float = 0.3):
        self.grid_size = 3
        self.overlap = overlap
        self.regions = [
            Region.TOP_LEFT, Region.TOP, Region.TOP_RIGHT,
            Region.LEFT, Region.CENTER, Region.RIGHT,
            Region.BOTTOM_LEFT, Region.BOTTOM, Region.BOTTOM_RIGHT
        ]
    
    def get_region_bounds(self, h: int, w: int, idx: int) -> Tuple[int, int, int, int]:
        """获取带重叠的区域边界"""
        row = idx // 3
        col = idx % 3
        
        # 基础cell大小
        base_h = h / 3
        base_w = w / 3
        
        # 重叠扩展
        overlap_h = int(base_h * self.overlap)
        overlap_w = int(base_w * self.overlap)
        
        # 计算边界
        y1 = max(0, int(row * base_h) - overlap_h)
        y2 = min(h, int((row + 1) * base_h) + overlap_h)
        x1 = max(0, int(col * base_w) - overlap_w)
        x2 = min(w, int((col + 1) * base_w) + overlap_w)
        
        return y1, y2, x1, x2
    
    def extract_features(self, image: np.ndarray) -> List[Tuple[Region, AtomType, float]]:
        """
        提取每个区域的特征
        
        Returns:
            [(区域, 原子类型, 密度), ...]
        """
        h, w = image.shape
        features = []
        
        for idx, region in enumerate(self.regions):
            y1, y2, x1, x2 = self.get_region_bounds(h, w, idx)
            region_img = image[y1:y2, x1:x2]
            
            density = float(np.mean(region_img > 0.3))
            
            if density < 0.05:
                features.append((region, AtomType.EMPTY, density))
                continue
            
            # 计算主方向
            direction = self._compute_direction(region_img)
            atom_type = self._classify(direction)
            
            features.append((region, atom_type, density))
        
        return features
    
    def _compute_direction(self, region: np.ndarray) -> int:
        """计算区域的主方向"""
        if region.shape[0] < 3 or region.shape[1] < 3:
            return -1
        
        # Sobel梯度
        h, w = region.shape
        gx = np.zeros_like(region)
        gy = np.zeros_like(region)
        
        for i in range(1, h-1):
            for j in range(1, w-1):
                gx[i, j] = region[i, j+1] - region[i, j-1]
                gy[i, j] = region[i+1, j] - region[i-1, j]
        
        magnitudes = np.sqrt(gx**2 + gy**2)
        mask = magnitudes > 0.1
        
        if np.sum(mask) < 3:
            return -1
        
        angles = np.arctan2(gy[mask], gx[mask] + 1e-8)
        
        # 量化到4个方向
        bins = [0, 0, 0, 0]  # V, H, DR, DL
        for a in angles:
            a_abs = abs(a)
            if a_abs < np.pi/8 or a_abs > 7*np.pi/8:
                bins[0] += 1  # V
            elif 3*np.pi/8 < a_abs < 5*np.pi/8:
                bins[1] += 1  # H
            elif np.pi/8 < a_abs < 3*np.pi/8:
                bins[2] += 1  # DR
            else:
                bins[3] += 1  # DL
        
        total = sum(bins)
        if total == 0:
            return -1
        
        max_bin = max(bins)
        if max_bin / total < 0.35:  # 没有明显主方向
            return -1
        
        return bins.index(max_bin)
    
    def _classify(self, direction: int) -> AtomType:
        if direction == -1:
            return AtomType.MIXED
        elif direction == 0:
            return AtomType.VERTICAL
        elif direction == 1:
            return AtomType.HORIZONTAL
        elif direction == 2:
            return AtomType.DIAGONAL_R
        else:
            return AtomType.DIAGONAL_L


# =============================================================================
# 5. Structural Signature
# =============================================================================

@dataclass(frozen=True)
class StructuralSignature:
    # 拓扑特征
    n_endpoints: int
    n_junctions: int
    n_holes: int  # 新增！
    
    # 区域模式
    region_pattern: Tuple[str, ...]
    
    # 密度分布（上/中/下）
    density_top: float
    density_mid: float
    density_bot: float
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 1. 拓扑相似度（权重最高）
        endpoint_sim = 1.0 / (1.0 + abs(self.n_endpoints - other.n_endpoints))
        junction_sim = 1.0 / (1.0 + abs(self.n_junctions - other.n_junctions))
        hole_sim = 1.0 if self.n_holes == other.n_holes else 0.2  # 空洞数必须匹配
        topo_sim = (endpoint_sim + junction_sim + 2 * hole_sim) / 4  # 空洞权重加倍
        
        # 2. 区域模式相似度
        matches = sum(1 for a, b in zip(self.region_pattern, other.region_pattern)
                     if a == b or a == 'empty' or b == 'empty')
        region_sim = matches / 9
        
        # 3. 密度分布相似度（区分6和9）
        density_diff = (abs(self.density_top - other.density_top) +
                       abs(self.density_mid - other.density_mid) +
                       abs(self.density_bot - other.density_bot)) / 3
        density_sim = 1.0 - density_diff
        
        return 0.45 * topo_sim + 0.30 * region_sim + 0.25 * density_sim


# =============================================================================
# 6. Vision System
# =============================================================================

class VisionSystem:
    def __init__(self):
        self.grid = OverlappingGrid(overlap=0.3)
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
    
    def analyze(self, image: np.ndarray) -> StructuralSignature:
        h, w = image.shape
        
        # 1. 骨架化 + 拓扑分析
        skeleton = skeletonize(image)
        n_endpoints, n_junctions, n_holes = analyze_topology(skeleton, image)
        
        # 2. 重叠网格特征
        features = self.grid.extract_features(image)
        region_pattern = tuple(f[1].value for f in features)
        
        # 3. 密度分布（上/中/下）
        third_h = h // 3
        density_top = float(np.mean(image[:third_h] > 0.3))
        density_mid = float(np.mean(image[third_h:2*third_h] > 0.3))
        density_bot = float(np.mean(image[2*third_h:] > 0.3))
        
        return StructuralSignature(
            n_endpoints=n_endpoints,
            n_junctions=n_junctions,
            n_holes=n_holes,
            region_pattern=region_pattern,
            density_top=density_top,
            density_mid=density_mid,
            density_bot=density_bot
        )
    
    def train(self, image: np.ndarray, label: str):
        sig = self.analyze(image)
        self.label_signatures[label].append(sig)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        sig = self.analyze(image)
        
        best_label = None
        best_score = 0.0
        
        for label, sigs in self.label_signatures.items():
            for stored in sigs:
                sim = sig.similarity(stored)
                if sim > best_score:
                    best_score = sim
                    best_label = label
        
        return (best_label or "unknown", best_score)
    
    def debug_image(self, image: np.ndarray) -> dict:
        sig = self.analyze(image)
        return {
            'topology': {
                'endpoints': sig.n_endpoints,
                'junctions': sig.n_junctions,
                'holes': sig.n_holes
            },
            'regions': sig.region_pattern,
            'density': {
                'top': f"{sig.density_top:.2f}",
                'mid': f"{sig.density_mid:.2f}",
                'bot': f"{sig.density_bot:.2f}"
            }
        }
    
    def print_analysis(self):
        print("\n=== 结构签名分析 ===")
        for label in sorted(self.label_signatures.keys()):
            sigs = self.label_signatures[label]
            
            endpoint_dist = defaultdict(int)
            junction_dist = defaultdict(int)
            hole_dist = defaultdict(int)
            
            for sig in sigs:
                endpoint_dist[sig.n_endpoints] += 1
                junction_dist[sig.n_junctions] += 1
                hole_dist[sig.n_holes] += 1
            
            print(f"\n数字 {label}:")
            print(f"  端点: {dict(endpoint_dist)}")
            print(f"  交叉: {dict(junction_dist)}")
            print(f"  空洞: {dict(hole_dist)}")


# =============================================================================
# 7. MNIST & Experiment
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
                except: continue
    
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


def run_experiment(n_train_per_class=10, n_test=500, verbose=True):
    print("=" * 60)
    print("Structon Vision v6.8 - Overlapping Grid + Hole Detection")
    print("=" * 60)
    print("\n新增：")
    print("  1. 重叠网格 (30% overlap)")
    print("  2. 空洞检测 (0有1洞, 8有2洞)")
    print("  3. 密度分布 (上/中/下, 区分6和9)")
    
    print("\nLoading MNIST...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    
    train_indices = []
    for d in range(10):
        train_indices.extend(np.where(train_labels == d)[0][:n_train_per_class])
    
    print(f"训练: {len(train_indices)} 样本, 测试: {n_test} 样本")
    
    system = VisionSystem()
    
    print("\n训练中...")
    t0 = time.time()
    for idx in train_indices:
        system.train(train_images[idx], str(train_labels[idx]))
    print(f"训练完成: {time.time()-t0:.1f}秒")
    
    if verbose:
        system.print_analysis()
    
    print("\n测试中...")
    results = {str(d): {'c': 0, 't': 0} for d in range(10)}
    test_idx = np.random.choice(len(test_images), n_test, replace=False)
    
    t0 = time.time()
    for i, idx in enumerate(test_idx):
        true = str(test_labels[idx])
        pred, _ = system.predict(test_images[idx])
        results[true]['t'] += 1
        if pred == true:
            results[true]['c'] += 1
        
        if verbose and (i+1) % 100 == 0:
            acc = sum(r['c'] for r in results.values()) / (i+1) * 100
            print(f"  {i+1}/{n_test}: {acc:.1f}%")
    
    print(f"测试完成: {time.time()-t0:.1f}秒")
    
    total_c = sum(r['c'] for r in results.values())
    total_t = sum(r['t'] for r in results.values())
    
    print(f"\n总准确率: {total_c/total_t*100:.1f}%")
    print("\n各数字:")
    for d in range(10):
        r = results[str(d)]
        acc = r['c']/r['t']*100 if r['t'] else 0
        print(f"  {d}: {acc:.1f}% ({r['c']}/{r['t']})")
    
    return system


def debug_digits(n=3):
    print("\n=== 调试: 拓扑 + 空洞 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"  样本{i+1}: 拓扑={info['topology']}, 密度={info['density']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-per-class', type=int, default=10)
    parser.add_argument('--test', type=int, default=500)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    
    if args.debug:
        debug_digits(3)
    else:
        run_experiment(args.train_per_class, args.test)
