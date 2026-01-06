"""
Structon Vision v6.10 - Curve Opening Direction
================================================

假设：准确率停在70%是因为缺少关键特征

新增：曲线开口方向检测
- CURVE_LEFT: 开口朝左 (如 3, 9 的曲线部分)
- CURVE_RIGHT: 开口朝右 (如 2, 6 的曲线部分)
- CURVE_UP: 开口朝上 (如 U)
- CURVE_DOWN: 开口朝下 (如 n)

这个特征应该能区分:
- 2 vs 3: 2顶部曲线开口朝右，3曲线开口朝左
- 5 vs 6: 5底部曲线开口朝左
- 6 vs 9: 已经用位置区分了，但曲线方向也不同

Author: Structon Framework
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import os
import gzip
import struct
import urllib.request
import time


# =============================================================================
# 1. Extended Atom Types
# =============================================================================

class AtomType(Enum):
    # 直线方向
    HORIZONTAL = "H"
    VERTICAL = "V"
    DIAGONAL_R = "DR"   # /
    DIAGONAL_L = "DL"   # \
    
    # 曲线开口方向 - 新增细分！
    CURVE_OPEN_LEFT = "C_L"    # ) 开口朝左
    CURVE_OPEN_RIGHT = "C_R"   # ( 开口朝右  
    CURVE_OPEN_UP = "C_U"      # ∪ 开口朝上
    CURVE_OPEN_DOWN = "C_D"    # ∩ 开口朝下
    
    # 拓扑类型
    ENDPOINT = "EP"
    JUNCTION_T = "JT"
    JUNCTION_X = "JX"
    
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
# 2. Skeletonization & Topology (unchanged)
# =============================================================================

def skeletonize(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
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


def analyze_topology_with_locations(skeleton: np.ndarray) -> Tuple[List[Tuple[int, int]], 
                                                                    List[Tuple[int, int]], 
                                                                    List[Tuple[int, int]]]:
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
# 3. Curve Opening Direction Detection - 新增！
# =============================================================================

def detect_curve_opening(region: np.ndarray, threshold: float = 0.3) -> Optional[str]:
    """
    检测曲线的开口方向
    
    方法：
    1. 找到前景像素的质心
    2. 找到前景像素的边界框中心
    3. 质心相对于边界框中心的偏移 = 开口的反方向
    
    例如：( 形状的质心在右边，开口在左边
    
    Returns:
        'L', 'R', 'U', 'D' 或 None（不是曲线）
    """
    binary = (region > threshold).astype(np.uint8)
    
    # 找前景像素
    ys, xs = np.where(binary > 0)
    
    if len(xs) < 5:
        return None
    
    # 质心
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    # 边界框中心
    bbox_cx = (np.min(xs) + np.max(xs)) / 2
    bbox_cy = (np.min(ys) + np.max(ys)) / 2
    
    # 偏移
    dx = cx - bbox_cx
    dy = cy - bbox_cy
    
    # 需要足够的偏移才算曲线
    min_offset = 0.5
    
    if abs(dx) < min_offset and abs(dy) < min_offset:
        return None  # 不是明显的曲线
    
    # 判断开口方向（质心偏移的反方向）
    if abs(dx) > abs(dy):
        # 水平偏移为主
        return 'L' if dx > 0 else 'R'  # 质心右移=开口左，质心左移=开口右
    else:
        # 垂直偏移为主
        return 'U' if dy > 0 else 'D'  # 质心下移=开口上，质心上移=开口下


# =============================================================================
# 4. Enhanced Grid with Curve Direction
# =============================================================================

class EnhancedGrid:
    def __init__(self, overlap: float = 0.3):
        self.grid_size = 3
        self.overlap = overlap
        self.regions = [
            Region.TOP_LEFT, Region.TOP, Region.TOP_RIGHT,
            Region.LEFT, Region.CENTER, Region.RIGHT,
            Region.BOTTOM_LEFT, Region.BOTTOM, Region.BOTTOM_RIGHT
        ]
    
    def get_region_bounds(self, h: int, w: int, idx: int) -> Tuple[int, int, int, int]:
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
        row = min(2, int(y / (h / 3)))
        col = min(2, int(x / (w / 3)))
        return row * 3 + col
    
    def extract_features(self, image: np.ndarray, skeleton: np.ndarray,
                        endpoints: List[Tuple[int, int]],
                        junctions_t: List[Tuple[int, int]],
                        junctions_x: List[Tuple[int, int]]) -> List[Tuple[Region, AtomType, float]]:
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
                atom_type = AtomType.ENDPOINT
            elif density < 0.05:
                atom_type = AtomType.EMPTY
            else:
                # 检测曲线开口方向
                curve_dir = detect_curve_opening(region_img)
                
                if curve_dir == 'L':
                    atom_type = AtomType.CURVE_OPEN_LEFT
                elif curve_dir == 'R':
                    atom_type = AtomType.CURVE_OPEN_RIGHT
                elif curve_dir == 'U':
                    atom_type = AtomType.CURVE_OPEN_UP
                elif curve_dir == 'D':
                    atom_type = AtomType.CURVE_OPEN_DOWN
                else:
                    # 使用直线方向
                    direction = self._compute_direction(region_img)
                    atom_type = self._classify_direction(direction)
            
            features.append((region, atom_type, density))
        
        return features
    
    def _compute_direction(self, region: np.ndarray) -> int:
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
    n_endpoints: int
    n_junctions: int
    n_holes: int
    
    region_pattern: Tuple[str, ...]
    
    density_top: float
    density_mid: float
    density_bot: float
    
    has_center_junction: bool
    has_top_endpoint: bool
    has_bottom_endpoint: bool
    
    # 新增：曲线开口方向统计
    curve_left_count: int
    curve_right_count: int
    curve_up_count: int
    curve_down_count: int
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 1. 拓扑计数相似度
        endpoint_sim = 1.0 / (1.0 + abs(self.n_endpoints - other.n_endpoints))
        junction_sim = 1.0 / (1.0 + abs(self.n_junctions - other.n_junctions))
        hole_sim = 1.0 if self.n_holes == other.n_holes else 0.2
        topo_count_sim = (endpoint_sim + junction_sim + 2 * hole_sim) / 4
        
        # 2. 区域模式相似度
        matches = sum(1 for a, b in zip(self.region_pattern, other.region_pattern) if a == b)
        region_sim = matches / 9
        
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
        
        # 5. 曲线开口方向相似度 - 新增！
        curve_sim = (
            1.0 / (1.0 + abs(self.curve_left_count - other.curve_left_count)) +
            1.0 / (1.0 + abs(self.curve_right_count - other.curve_right_count)) +
            1.0 / (1.0 + abs(self.curve_up_count - other.curve_up_count)) +
            1.0 / (1.0 + abs(self.curve_down_count - other.curve_down_count))
        ) / 4
        
        return (0.30 * topo_count_sim + 
                0.20 * region_sim + 
                0.15 * density_sim + 
                0.15 * topo_loc_sim +
                0.20 * curve_sim)  # 曲线方向占20%权重


# =============================================================================
# 6. Vision System
# =============================================================================

class VisionSystem:
    def __init__(self):
        self.grid = EnhancedGrid(overlap=0.3)
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
    
    def analyze(self, image: np.ndarray) -> StructuralSignature:
        h, w = image.shape
        
        skeleton = skeletonize(image)
        endpoints, junctions_t, junctions_x = analyze_topology_with_locations(skeleton)
        
        binary = (image > 0.3).astype(np.uint8)
        n_holes = count_holes(binary)
        
        features = self.grid.extract_features(image, skeleton, endpoints, junctions_t, junctions_x)
        region_pattern = tuple(f[1].value for f in features)
        
        # 统计曲线开口方向
        curve_left = sum(1 for f in features if f[1] == AtomType.CURVE_OPEN_LEFT)
        curve_right = sum(1 for f in features if f[1] == AtomType.CURVE_OPEN_RIGHT)
        curve_up = sum(1 for f in features if f[1] == AtomType.CURVE_OPEN_UP)
        curve_down = sum(1 for f in features if f[1] == AtomType.CURVE_OPEN_DOWN)
        
        third_h = h // 3
        density_top = float(np.mean(image[:third_h] > 0.3))
        density_mid = float(np.mean(image[third_h:2*third_h] > 0.3))
        density_bot = float(np.mean(image[2*third_h:] > 0.3))
        
        center_region = 4
        top_regions = {0, 1, 2}
        bottom_regions = {6, 7, 8}
        
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
            has_bottom_endpoint=has_bottom_endpoint,
            curve_left_count=curve_left,
            curve_right_count=curve_right,
            curve_up_count=curve_up,
            curve_down_count=curve_down
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
            'curves': {
                'left': sig.curve_left_count,
                'right': sig.curve_right_count,
                'up': sig.curve_up_count,
                'down': sig.curve_down_count
            },
            'topo_location': {
                'center_junction': sig.has_center_junction,
                'top_endpoint': sig.has_top_endpoint,
                'bottom_endpoint': sig.has_bottom_endpoint
            }
        }
    
    def print_analysis(self):
        print("\n=== 结构签名分析 ===")
        for label in sorted(self.label_signatures.keys()):
            sigs = self.label_signatures[label]
            
            hole_dist = defaultdict(int)
            curve_l = curve_r = curve_u = curve_d = 0
            
            for sig in sigs:
                hole_dist[sig.n_holes] += 1
                curve_l += sig.curve_left_count
                curve_r += sig.curve_right_count
                curve_u += sig.curve_up_count
                curve_d += sig.curve_down_count
            
            n = len(sigs)
            print(f"\n数字 {label}:")
            print(f"  空洞: {dict(hole_dist)}")
            print(f"  曲线方向: L={curve_l/n:.1f}, R={curve_r/n:.1f}, U={curve_u/n:.1f}, D={curve_d/n:.1f}")


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
    print("Structon Vision v6.10 - Curve Opening Direction")
    print("=" * 60)
    print("\n新增：曲线开口方向检测")
    print("  C_L: 开口朝左 (如 3, 9)")
    print("  C_R: 开口朝右 (如 2, 6)")
    print("  C_U: 开口朝上 (如 U)")
    print("  C_D: 开口朝下 (如 n)")
    
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
    print("\n=== 调试: 曲线开口方向 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"  样本{i+1}:")
            print(f"    拓扑: {info['topology']}")
            print(f"    曲线: {info['curves']}")
            print(f"    区域: {info['regions']}")


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
