"""
Structon Vision v6.7 - Fixed Topology Detection
================================================

v6.6的问题：
1. 拓扑检测失败：所有数字都是 endpoints=4, junctions=0
   - 原因：MNIST笔画是粗的（2-3像素），不是细线
   - crossing number算法需要骨架化的图像

2. 太多Junction分类
   - curvature > 0.6 阈值太低
   - 应该根据多个强方向判断，而不是曲率

v6.7的修复：
1. 骨架化处理后再计算拓扑
2. 更好的原子类型分类
3. 简化策略：用更直接的特征

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
    DIAGONAL_R = "DR"   # /
    DIAGONAL_L = "DL"   # \
    MIXED = "M"         # 多方向混合
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
# 2. Skeletonization (Zhang-Suen Thinning)
# =============================================================================

def skeletonize(image: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """
    Zhang-Suen骨架化算法
    把粗笔画变成单像素宽的骨架
    """
    # 二值化
    binary = (image > threshold).astype(np.uint8)
    skeleton = binary.copy()
    
    h, w = skeleton.shape
    
    def neighbors(y, x):
        """获取8邻域，按顺时针顺序: P2,P3,P4,P5,P6,P7,P8,P9"""
        return [
            skeleton[y-1, x],   # P2
            skeleton[y-1, x+1], # P3
            skeleton[y, x+1],   # P4
            skeleton[y+1, x+1], # P5
            skeleton[y+1, x],   # P6
            skeleton[y+1, x-1], # P7
            skeleton[y, x-1],   # P8
            skeleton[y-1, x-1]  # P9
        ]
    
    def transitions(neighbors):
        """计算0->1跳变次数"""
        n = neighbors + [neighbors[0]]  # 环绕
        return sum(n[i] == 0 and n[i+1] == 1 for i in range(8))
    
    changed = True
    iterations = 0
    max_iterations = 100
    
    while changed and iterations < max_iterations:
        changed = False
        iterations += 1
        
        # Step 1
        to_remove = []
        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 0:
                    continue
                
                P = neighbors(y, x)
                B = sum(P)  # 黑色邻居数
                A = transitions(P)  # 0->1跳变
                
                if (2 <= B <= 6 and A == 1 and
                    P[0] * P[2] * P[4] == 0 and
                    P[2] * P[4] * P[6] == 0):
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
                
                if (2 <= B <= 6 and A == 1 and
                    P[0] * P[2] * P[6] == 0 and
                    P[0] * P[4] * P[6] == 0):
                    to_remove.append((y, x))
        
        for y, x in to_remove:
            skeleton[y, x] = 0
            changed = True
    
    return skeleton


# =============================================================================
# 3. Topology Analysis (on skeleton)
# =============================================================================

def analyze_topology(skeleton: np.ndarray) -> Tuple[int, int, int]:
    """
    分析骨架的拓扑特征
    
    Returns:
        n_endpoints: 端点数 (只有1个邻居)
        n_junctions: 交叉点数 (3个或更多分支)
        n_loops: 闭环数 (用欧拉数估计)
    """
    h, w = skeleton.shape
    
    n_endpoints = 0
    n_junctions = 0
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if skeleton[y, x] == 0:
                continue
            
            # 8邻域
            neighbors = [
                skeleton[y-1, x-1], skeleton[y-1, x], skeleton[y-1, x+1],
                skeleton[y, x-1],                     skeleton[y, x+1],
                skeleton[y+1, x-1], skeleton[y+1, x], skeleton[y+1, x+1]
            ]
            
            n_neighbors = sum(neighbors)
            
            # 计算crossing number（更准确的方法）
            ring = [
                skeleton[y-1, x],   skeleton[y-1, x+1],
                skeleton[y, x+1],   skeleton[y+1, x+1],
                skeleton[y+1, x],   skeleton[y+1, x-1],
                skeleton[y, x-1],   skeleton[y-1, x-1]
            ]
            
            crossings = sum(ring[i] != ring[(i+1) % 8] for i in range(8)) // 2
            
            # 端点：1个邻居或1个crossing
            if n_neighbors == 1 or crossings == 1:
                n_endpoints += 1
            
            # 交叉点：3个或更多crossing
            if crossings >= 3:
                n_junctions += 1
    
    # 估计闭环数（欧拉特征）
    # V - E + F = 2 for connected, F-1 = loops
    # 简化：如果有很多像素但端点少，可能有环
    n_pixels = np.sum(skeleton)
    n_loops = 0
    if n_pixels > 20 and n_endpoints <= 2:
        n_loops = 1
    if n_pixels > 40 and n_endpoints == 0:
        n_loops = 2
    
    return n_endpoints, n_junctions, n_loops


# =============================================================================
# 4. Feature Extraction
# =============================================================================

def extract_region_features(image: np.ndarray, y1: int, y2: int, x1: int, x2: int) -> Tuple[float, int]:
    """
    提取区域特征
    
    Returns:
        density: 像素密度
        dominant_direction: 主方向 (0=V, 1=H, 2=DR, 3=DL, -1=empty/mixed)
    """
    region = image[y1:y2, x1:x2]
    
    density = np.mean(region > 0.3)
    
    if density < 0.05:
        return density, -1  # empty
    
    # 计算梯度方向
    if region.shape[0] < 3 or region.shape[1] < 3:
        return density, -1
    
    # Sobel梯度
    gx = np.zeros_like(region)
    gy = np.zeros_like(region)
    
    for i in range(1, region.shape[0]-1):
        for j in range(1, region.shape[1]-1):
            gx[i,j] = region[i, j+1] - region[i, j-1]
            gy[i,j] = region[i+1, j] - region[i-1, j]
    
    # 方向统计
    angles = np.arctan2(gy, gx + 1e-8)
    magnitudes = np.sqrt(gx**2 + gy**2)
    
    # 只看强梯度
    mask = magnitudes > 0.1
    if np.sum(mask) < 3:
        return density, -1
    
    valid_angles = angles[mask]
    
    # 量化到4个方向
    # 0: 垂直 (-π/8 to π/8 or 7π/8 to π or -π to -7π/8)
    # 1: 水平 (3π/8 to 5π/8 or -5π/8 to -3π/8)
    # 2: 斜右上 (π/8 to 3π/8 or -7π/8 to -5π/8)
    # 3: 斜右下 (5π/8 to 7π/8 or -3π/8 to -π/8)
    
    bins = [0, 0, 0, 0]
    for a in valid_angles:
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
        return density, -1
    
    max_bin = max(bins)
    dominant = bins.index(max_bin)
    
    # 如果没有明显主方向，返回mixed
    if max_bin / total < 0.4:
        return density, -1
    
    return density, dominant


def classify_atom(density: float, direction: int) -> AtomType:
    """根据特征分类原子"""
    if density < 0.05:
        return AtomType.EMPTY
    
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
    # 拓扑特征（最稳定）
    n_endpoints: int
    n_junctions: int
    n_loops: int
    
    # 区域占用模式
    region_pattern: Tuple[str, ...]  # 9个区域的类型
    
    # 总像素密度
    total_density: float
    
    def similarity(self, other: 'StructuralSignature') -> float:
        # 拓扑相似度（权重最高）
        endpoint_sim = 1.0 / (1.0 + abs(self.n_endpoints - other.n_endpoints))
        junction_sim = 1.0 / (1.0 + abs(self.n_junctions - other.n_junctions))
        loop_sim = 1.0 if self.n_loops == other.n_loops else 0.3
        topo_sim = (endpoint_sim + junction_sim + loop_sim) / 3
        
        # 区域模式相似度
        matches = sum(1 for a, b in zip(self.region_pattern, other.region_pattern) 
                     if a == b or a == 'empty' or b == 'empty')
        region_sim = matches / 9
        
        # 密度相似度
        density_sim = 1.0 - abs(self.total_density - other.total_density)
        
        return 0.5 * topo_sim + 0.35 * region_sim + 0.15 * density_sim


# =============================================================================
# 6. Vision System
# =============================================================================

class VisionSystem:
    def __init__(self):
        self.regions = [
            Region.TOP_LEFT, Region.TOP, Region.TOP_RIGHT,
            Region.LEFT, Region.CENTER, Region.RIGHT,
            Region.BOTTOM_LEFT, Region.BOTTOM, Region.BOTTOM_RIGHT
        ]
        self.label_signatures: Dict[str, List[StructuralSignature]] = defaultdict(list)
    
    def analyze(self, image: np.ndarray) -> StructuralSignature:
        """分析图像，提取结构签名"""
        h, w = image.shape
        
        # 1. 骨架化 + 拓扑分析
        skeleton = skeletonize(image)
        n_endpoints, n_junctions, n_loops = analyze_topology(skeleton)
        
        # 2. 区域特征
        region_types = []
        cell_h = h // 3
        cell_w = w // 3
        
        for idx in range(9):
            row = idx // 3
            col = idx % 3
            
            y1 = row * cell_h
            y2 = (row + 1) * cell_h if row < 2 else h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w if col < 2 else w
            
            density, direction = extract_region_features(image, y1, y2, x1, x2)
            atom_type = classify_atom(density, direction)
            region_types.append(atom_type.value)
        
        # 3. 总密度
        total_density = float(np.mean(image > 0.3))
        
        return StructuralSignature(
            n_endpoints=n_endpoints,
            n_junctions=n_junctions,
            n_loops=n_loops,
            region_pattern=tuple(region_types),
            total_density=total_density
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
        skeleton = skeletonize(image)
        n_endpoints, n_junctions, n_loops = analyze_topology(skeleton)
        sig = self.analyze(image)
        
        return {
            'topology': {
                'endpoints': n_endpoints,
                'junctions': n_junctions,
                'loops': n_loops
            },
            'regions': sig.region_pattern,
            'density': f"{sig.total_density:.2f}"
        }
    
    def print_analysis(self):
        print("\n=== 结构签名分析 ===")
        for label in sorted(self.label_signatures.keys()):
            sigs = self.label_signatures[label]
            
            endpoint_dist = defaultdict(int)
            junction_dist = defaultdict(int)
            loop_dist = defaultdict(int)
            
            for sig in sigs:
                endpoint_dist[sig.n_endpoints] += 1
                junction_dist[sig.n_junctions] += 1
                loop_dist[sig.n_loops] += 1
            
            print(f"\n数字 {label}:")
            print(f"  端点: {dict(endpoint_dist)}")
            print(f"  交叉: {dict(junction_dist)}")
            print(f"  闭环: {dict(loop_dist)}")


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
    print("Structon Vision v6.7 - Fixed Topology")
    print("=" * 60)
    print("\n修复：")
    print("  1. 骨架化后再计算拓扑")
    print("  2. 更好的原子类型分类")
    print("  3. 闭环检测")
    
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
    print("\n=== 调试: 拓扑特征 ===")
    train_images, train_labels, _, _ = load_mnist()
    system = VisionSystem()
    
    for digit in range(10):
        print(f"\n数字 {digit}:")
        indices = np.where(train_labels == digit)[0][:n]
        for i, idx in enumerate(indices):
            info = system.debug_image(train_images[idx])
            print(f"  样本{i+1}: {info['topology']}")
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
