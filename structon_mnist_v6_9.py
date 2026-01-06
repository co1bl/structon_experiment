"""
Structon Vision v6.9 - Junction Atoms with Location
====================================================

新增改进：

1. Junction原子类型
   - JUNCTION_T: T形交叉 (3个分支)
   - JUNCTION_X: X形交叉 (4个分支)
   - ENDPOINT: 端点 (1个分支)
   
2. 将拓扑特征映射到空间位置
   - 之前：只知道"有几个交叉点"
   - 现在：知道"哪个区域有交叉点"
   
3. 这让我们能区分：
   - 4: 中间有T交叉
   - 8: 中间有X交叉（两个环相接）
   - 7: 上方有端点

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Types - 扩展原子类型
# =============================================================================

class AtomType(Enum):
    # 方向类型
    HORIZONTAL = "H"
    VERTICAL = "V"
    DIAGONAL_R = "DR"
    DIAGONAL_L = "DL"
    MIXED = "M"
    
    # 拓扑类型 - 新增！
    ENDPOINT = "EP"      # 端点
    JUNCTION_T = "JT"    # T形交叉 (3分支)
    JUNCTION_X = "JX"    # X形交叉 (4+分支)
    
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
# 3. Topology Analysis - 返回位置信息
# =============================================================================

def analyze_topology_with_locations(skeleton: np.ndarray) -> Tuple[List[Tuple[int, int]], 
                                                                    List[Tuple[int, int]], 
                                                                    List[Tuple[int, int]]]:
    """
    分析骨架拓扑，返回特征点的位置
    
    Returns:
        endpoints: [(y, x), ...] 端点位置列表
        junctions_t: [(y, x), ...] T形交叉位置列表
        junctions_x: [(y, x), ...] X形交叉位置列表
    """
    h, w = skeleton.shape
    
    endpoints = []
    junctions_t = []
    junctions_x = []
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 0:
                continue
            
            ring = [
                skeleton[y-1, x], skeleton[y-1, x+1], skeleton[y, x+1], skeleton[y+1, x+1],
                skeleton[y+1, x], skeleton[y+1, x-1], skeleton[y, x-1], skeleton[y-1, x-1]
            ]
            
            n_neighbors = sum(ring)
            crossings = sum(ring[i] != ring[(i+1) % 8] for i in range(8)) // 2
            
            if n_neighbors == 1 or crossings == 1:
                endpoints.append((y, x))
            elif crossings == 3:
                junctions_t.append((y, x))
            elif crossings >= 4:
                junctions_x.append((y, x))
    
    return endpoints, junctions_t, junctions_x


def count_holes(binary: np.ndarray) -> int:
    """计算空洞数量"""
    h, w = binary.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = binary
    
    visited = np.zeros_like(padded, dtype=bool)
    
    def flood_fill(start_y, start_x):
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
    
    flood_fill(0, 0)
    
    n_holes = 0
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] == 0 and not visited[y, x]:
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


# =============================================================================
# 4. Grid with Junction Atoms
# =============================================================================

class EnhancedGrid:
    """
    增强网格：结合方向特征和拓扑特征
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
        
        base_h = h / 3
        base_w = w / 3
        
        overlap_h = int(base_h * self.overlap)
        overlap_w = int(base_w * self.overlap)
        
        y1 = max(0, int(row * base_h) - overlap_h)
        y2 = min(h, int((row + 1) * base_h) + overlap_h)
        x1 = max(0, int(col * base_w) - overlap_w)
        x2 = min(w, int((col + 1) * base_w) + overlap_w)
        
        return y1, y2, x1, x2
    
    def get_region_for_point(self, y: int, x: int, h: int, w: int) -> int:
        """确定一个点属于哪个区域（无重叠，用于拓扑点映射）"""
        row = min(2, int(y / (h / 3)))
        col = min(2, int(x / (w / 3)))
        return row * 3 + col
    
    def extract_features(self, image: np.ndarray, skeleton: np.ndarray,
                        endpoints: List[Tuple[int, int]],
                        junctions_t: List[Tuple[int, int]],
                        junctions_x: List[Tuple[int, int]]) -> List[Tuple[Region, AtomType, float]]:
        """
        提取每个区域的特征，结合方向和拓扑
        """
        h, w = image.shape
        features = []
        
        # 统计每个区域的拓扑点
        region_endpoints = defaultdict(int)
        region_junctions_t = defaultdict(int)
        region_junctions_x = defaultdict(int)
        
        for y, x in endpoints:
            idx = self.get_region_for_point(y, x, h, w)
            region_endpoints[idx] += 1
        
        for y, x in junctions_t:
            idx = self.get_region_for_point(y, x, h, w)
            region_junctions_t[idx] += 1
        
        for y, x in junctions_x:
            idx = self.get_region_for_point(y, x, h, w)
            region_junctions_x[idx] += 1
        
        # 对每个区域
        for idx, region in enumerate(self.regions):
            y1, y2, x1, x2 = self.get_region_bounds(h, w, idx)
            region_img = image[y1:y2, x1:x2]
            
            density = float(np.mean(region_img > 0.3))
            
            # 优先使用拓扑类型
            if region_junctions_x[idx] > 0:
                atom_type = AtomType.JUNCTION_X
            elif region_junctions_t[idx] > 0:
                atom_type = AtomType.JUNCTION_T
            elif region_endpoints[idx] > 0 and density < 0.15:
                # 只有在密度低时才标记为端点（避免误标）
                atom_type = AtomType.ENDPOINT
            elif density < 0.05:
                atom_type = AtomType.EMPTY
            else:
                # 使用方向类型
                direction = self._compute_direction(region_img)
                atom_type = self._classify_direction(direction)
            
            features.append((region, atom_type, density))
        
        return features
    
    def _compute_direction(self, region: np.ndarray) -> int:
        """计算区域的主方向"""
        if region.shape[0] < 3 or region.shape[1] < 3:
            return -1
        
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
        
        bins = [0, 0, 0, 0]
        for a in angles:
            a_abs = abs(a)
            if a_abs < np.pi/8 or a_abs > 7*np.pi/8:
                bins[0] += 1
            elif 3*np.pi/8 < a_abs < 5*np.pi/8:
                bins[1] += 1
            elif np.pi/8 < a_abs < 3*np.pi/8:
                bins[2] += 1
            else:
                bins[3] += 1
        
        total = sum(bins)
        if total == 0:
            return -1
        
        max_bin = max(bins)
        if max_bin / total < 0.35:
            return -1
        
        return bins.index(max_bin)
    
    def _classify_direction(self, direction: int) -> AtomType:
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
    # 拓扑计数
    n_endpoints: int
    n_junctions: int
    n_holes: int
    
    # 区域模式（包含拓扑原子）
    region_pattern: Tuple[str, ...]
    
    # 密度分布
    density_top: float
    density_mid: float
    density_bot: float
    
    # 拓扑原子位置特征
    has_center_junction: bool
    has_top_endpoint: bool
    has_bottom_endpoint: bool
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 1. 拓扑计数相似度
        endpoint_sim = 1.0 / (1.0 + abs(self.n_endpoints - other.n_endpoints))
        junction_sim = 1.0 / (1.0 + abs(self.n_junctions - other.n_junctions))
        hole_sim = 1.0 if self.n_holes == other.n_holes else 0.2
        topo_count_sim = (endpoint_sim + junction_sim + 2 * hole_sim) / 4
        
        # 2. 区域模式相似度
        matches = sum(1 for a, b in zip(self.region_pattern, other.region_pattern)
                     if a == b)
        # 对拓扑原子匹配给更高分
        topo_types = {'EP', 'JT', 'JX'}
        topo_matches = sum(1 for a, b in zip(self.region_pattern, other.region_pattern)
                         if a == b and a in topo_types)
        region_sim = (matches + topo_matches) / 12  # 额外奖励拓扑匹配
        
        # 3. 密度分布相似度
        density_diff = (abs(self.density_top - other.density_top) +
                       abs(self.density_mid - other.density_mid) +
                       abs(self.density_bot - other.density_bot)) / 3
        density_sim = 1.0 - density_diff
        
        # 4. 拓扑位置特征
        topo_loc_sim = (
            (1.0 if self.has_center_junction == other.has_center_junction else 0.0) +
            (1.0 if self.has_top_endpoint == other.has_top_endpoint else 0.0) +
            (1.0 if self.has_bottom_endpoint == other.has_bottom_endpoint else 0.0)
        ) / 3
        
        return 0.35 * topo_count_sim + 0.25 * region_sim + 0.20 * density_sim + 0.20 * topo_loc_sim


# =============================================================================
# 6. Vision System
# =============================================================================

class VisionSystem:
    def __init__(self):
        self.grid = EnhancedGrid(overlap=0.3)
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
    
    def analyze(self, image: np.ndarray) -> StructuralSignature:
        h, w = image.shape
        
        # 1. 骨架化
        skeleton = skeletonize(image)
        
        # 2. 拓扑分析（带位置）
        endpoints, junctions_t, junctions_x = analyze_topology_with_locations(skeleton)
        
        # 3. 空洞检测
        binary = (image > 0.3).astype(np.uint8)
        n_holes = count_holes(binary)
        
        # 4. 网格特征（结合拓扑位置）
        features = self.grid.extract_features(image, skeleton, endpoints, junctions_t, junctions_x)
        region_pattern = tuple(f[1].value for f in features)
        
        # 5. 密度分布
        third_h = h // 3
        density_top = float(np.mean(image[:third_h] > 0.3))
        density_mid = float(np.mean(image[third_h:2*third_h] > 0.3))
        density_bot = float(np.mean(image[2*third_h:] > 0.3))
        
        # 6. 拓扑位置特征
        center_region = 4  # CENTER
        top_regions = {0, 1, 2}  # TL, T, TR
        bottom_regions = {6, 7, 8}  # BL, B, BR
        
        has_center_junction = any(
            self.grid.get_region_for_point(y, x, h, w) == center_region
            for y, x in junctions_t + junctions_x
        )
        
        has_top_endpoint = any(
            self.grid.get_region_for_point(y, x, h, w) in top_regions
            for y, x in endpoints
        )
        
        has_bottom_endpoint = any(
            self.grid.get_region_for_point(y, x, h, w) in bottom_regions
            for y, x in endpoints
        )
        
        return StructuralSignature(
            n_endpoints=len(endpoints),
            n_junctions=len(junctions_t) + len(junctions_x),
            n_holes=n_holes,
            region_pattern=region_pattern,
            density_top=density_top,
            density_mid=density_mid,
            density_bot=density_bot,
            has_center_junction=has_center_junction,
            has_top_endpoint=has_top_endpoint,
            has_bottom_endpoint=has_bottom_endpoint
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
            'topo_location': {
                'center_junction': sig.has_center_junction,
                'top_endpoint': sig.has_top_endpoint,
                'bottom_endpoint': sig.has_bottom_endpoint
            },
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
            center_j = 0
            top_ep = 0
            bot_ep = 0
            
            for sig in sigs:
                endpoint_dist[sig.n_endpoints] += 1
                junction_dist[sig.n_junctions] += 1
                hole_dist[sig.n_holes] += 1
                if sig.has_center_junction:
                    center_j += 1
                if sig.has_top_endpoint:
                    top_ep += 1
                if sig.has_bottom_endpoint:
                    bot_ep += 1
            
            print(f"\n数字 {label}:")
            print(f"  端点: {dict(endpoint_dist)}")
            print(f"  交叉: {dict(junction_dist)}")
            print(f"  空洞: {dict(hole_dist)}")
            print(f"  位置: 中心交叉={center_j}, 顶端点={top_ep}, 底端点={bot_ep}")


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
    print("Structon Vision v6.9 - Junction Atoms with Location")
    print("=" * 60)
    print("\n新增：")
    print("  1. Junction原子类型 (JT, JX, EP)")
    print("  2. 拓扑特征映射到空间位置")
    print("  3. 位置特征：中心交叉、顶端点、底端点")
    
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
    print("\n=== 调试: Junction位置 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"  样本{i+1}:")
            print(f"    拓扑: {info['topology']}")
            print(f"    区域: {info['regions']}")
            print(f"    位置: {info['topo_location']}")


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
