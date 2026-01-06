"""
Structon Vision v6.1 - MNIST Test
==================================

Test the Structon architecture on MNIST digit recognition.

This script:
1. Downloads MNIST (if not present)
2. Trains Structon on training set
3. Tests on test set
4. Reports accuracy per digit and overall

Usage:
    python structon_mnist_test.py

Requirements:
    pip install numpy

The script will auto-download MNIST data on first run.

Author: Structon Framework
"""

import numpy as np
import os
import gzip
import struct
import urllib.request
from typing import Dict, List, Tuple, Optional, FrozenSet
from dataclasses import dataclass
from collections import defaultdict
import time


# =============================================================================
# MNIST Data Loading
# =============================================================================

MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

MNIST_FILES = {
    'train_images': 'train-images-idx3-ubyte.gz',
    'train_labels': 'train-labels-idx1-ubyte.gz',
    'test_images': 't10k-images-idx3-ubyte.gz',
    'test_labels': 't10k-labels-idx1-ubyte.gz'
}


def download_mnist(data_dir: str = './mnist_data'):
    """
    Download MNIST dataset if not present.
    
    If automatic download fails, manually download from:
        https://storage.googleapis.com/cvdf-datasets/mnist/
    
    And place these files in ./mnist_data/:
        - train-images-idx3-ubyte.gz
        - train-labels-idx1-ubyte.gz  
        - t10k-images-idx3-ubyte.gz
        - t10k-labels-idx1-ubyte.gz
    """
    os.makedirs(data_dir, exist_ok=True)
    
    for name, filename in MNIST_FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            
            downloaded = False
            for mirror in MNIST_MIRRORS:
                try:
                    url = mirror + filename
                    print(f"  Trying {mirror}...")
                    urllib.request.urlretrieve(url, filepath)
                    print(f"  Saved to {filepath}")
                    downloaded = True
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue
            
            if not downloaded:
                print(f"\nERROR: Could not download {filename}")
                print(f"Please manually download from:")
                print(f"  https://storage.googleapis.com/cvdf-datasets/mnist/{filename}")
                print(f"And place in: {data_dir}/")
                raise RuntimeError(f"Failed to download {filename}")
    
    print("MNIST data ready.")


def load_mnist_images(filepath: str) -> np.ndarray:
    """Load MNIST images from gzipped file"""
    with gzip.open(filepath, 'rb') as f:
        magic, num, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images.astype(np.float32) / 255.0


def load_mnist_labels(filepath: str) -> np.ndarray:
    """Load MNIST labels from gzipped file"""
    with gzip.open(filepath, 'rb') as f:
        magic, num = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


def load_mnist(data_dir: str = './mnist_data'):
    """Load full MNIST dataset"""
    download_mnist(data_dir)
    
    train_images = load_mnist_images(os.path.join(data_dir, MNIST_FILES['train_images']))
    train_labels = load_mnist_labels(os.path.join(data_dir, MNIST_FILES['train_labels']))
    test_images = load_mnist_images(os.path.join(data_dir, MNIST_FILES['test_images']))
    test_labels = load_mnist_labels(os.path.join(data_dir, MNIST_FILES['test_labels']))
    
    return train_images, train_labels, test_images, test_labels


# =============================================================================
# Structon Vision v6.1 (Embedded)
# =============================================================================

@dataclass
class Activation:
    """A single activation point from V1"""
    x: float
    y: float
    strength: float
    orientation: int
    feature_response: np.ndarray


@dataclass
class Anchors:
    """Spatial anchors"""
    center: Tuple[float, float]
    head: Tuple[float, float]
    tail: Tuple[float, float]


@dataclass(frozen=True)
class StructuralDescriptor:
    """Composite pattern signature - pure structure"""
    child_orientations: FrozenSet[int]
    relation: str
    connection_type: str
    
    def __hash__(self):
        return hash((self.child_orientations, self.relation, self.connection_type))


@dataclass
class AtomicNode:
    """Atomic pattern - matched by features"""
    id: int
    orientation: int
    feature_signature: np.ndarray
    exemplar_anchors: Optional[Anchors]
    access_count: int = 0
    label: Optional[str] = None


@dataclass
class CompositeNode:
    """Composite pattern - matched by structure"""
    id: int
    descriptor: StructuralDescriptor
    child_orientations: List[int]
    access_count: int = 0
    label: Optional[str] = None


class V1GaborBank:
    """V1 Gabor filters"""
    
    def __init__(self, kernel_size: int = 5):
        self.kernel_size = kernel_size
        self.orientations = [0, 45, 90, 135]  # Simplified for MNIST
        self.scales = [1.0, 1.5]
        self.filters = self._create_gabor_bank()
        self.n_features = len(self.filters)
        self.n_orientations = len(self.orientations)
    
    def _create_gabor_kernel(self, theta, sigma, lambd, gamma=0.5):
        size = self.kernel_size
        half = size // 2
        kernel = np.zeros((size, size), dtype=np.float32)
        for y in range(-half, half + 1):
            for x in range(-half, half + 1):
                x_theta = x * np.cos(theta) + y * np.sin(theta)
                y_theta = -x * np.sin(theta) + y * np.cos(theta)
                gaussian = np.exp(-(x_theta**2 + gamma**2 * y_theta**2) / (2 * sigma**2))
                sinusoid = np.cos(2 * np.pi * x_theta / lambd)
                kernel[y + half, x + half] = gaussian * sinusoid
        kernel -= kernel.mean()
        norm = np.linalg.norm(kernel)
        return kernel / norm if norm > 1e-8 else kernel
    
    def _create_gabor_bank(self):
        filters = []
        for scale in self.scales:
            for theta_deg in self.orientations:
                theta = np.deg2rad(theta_deg)
                kernel = self._create_gabor_kernel(theta, 2.0*scale, 4.0*scale)
                filters.append(kernel)
        return filters
    
    def _convolve(self, image, kernel):
        kh, kw = kernel.shape
        ih, iw = image.shape
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        output = np.zeros((ih, iw), dtype=np.float32)
        for i in range(ih):
            for j in range(iw):
                output[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        return output
    
    def compute_response_maps(self, image: np.ndarray) -> List[np.ndarray]:
        return [np.abs(self._convolve(image, kernel)) for kernel in self.filters]
    
    def extract_activations(self, image: np.ndarray, threshold: float = 0.3, step: int = 2) -> List[Activation]:
        """Extract activations with per-location features"""
        response_maps = self.compute_response_maps(image)
        h, w = image.shape
        
        global_max = max(r.max() for r in response_maps)
        if global_max < 1e-8:
            return []
        
        norm_maps = [r / global_max for r in response_maps]
        activations = []
        
        for y in range(step, h - step, step):
            for x in range(step, w - step, step):
                feature_vec = np.array([norm_maps[i][y, x] for i in range(self.n_features)])
                
                max_response = -1
                dominant_orient = 0
                
                for feat_idx in range(self.n_features):
                    val = norm_maps[feat_idx][y, x]
                    if val > max_response:
                        max_response = val
                        orient_idx = feat_idx % self.n_orientations
                        dominant_orient = self.orientations[orient_idx]
                
                if max_response < threshold:
                    continue
                
                activations.append(Activation(
                    x=float(x),
                    y=float(y),
                    strength=max_response,
                    orientation=dominant_orient,
                    feature_response=feature_vec
                ))
        
        by_pos = {}
        for a in activations:
            pos = (round(a.x / step) * step, round(a.y / step) * step)
            if pos not in by_pos or a.strength > by_pos[pos].strength:
                by_pos[pos] = a
        
        return list(by_pos.values())
    
    def extract_global_features(self, image: np.ndarray) -> np.ndarray:
        """Extract global feature vector (for fallback matching)"""
        response_maps = self.compute_response_maps(image)
        features = [np.mean(r) for r in response_maps]
        return np.array(features, dtype=np.float32)


class StructuralMemory:
    """Structon Memory for MNIST"""
    
    def __init__(self, feature_dim: int, image_size: float = 28.0):
        self.feature_dim = feature_dim
        self.image_size = image_size
        
        self.atomic_nodes: Dict[int, AtomicNode] = {}
        self.composite_nodes: Dict[int, CompositeNode] = {}
        self.next_id: int = 0
        
        self.similarity_threshold = 0.80
        self.port_threshold = image_size * 0.3
        
        self.orientation_to_atoms: Dict[int, List[int]] = defaultdict(list)
        self.structure_index: Dict[StructuralDescriptor, int] = {}
        
        # For MNIST: also track label-based patterns
        self.label_patterns: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.label_centroids: Dict[str, np.ndarray] = {}
    
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-8 else v
    
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a_norm = StructuralMemory._normalize(a)
        b_norm = StructuralMemory._normalize(b)
        return float(np.dot(a_norm, b_norm))
    
    @staticmethod
    def to_canonical_orientation(orient: int) -> int:
        if orient in [0, 180]:
            return 0
        elif orient in [45, 225]:
            return 45
        elif orient in [90, 270]:
            return 90
        elif orient in [135, 315]:
            return 135
        return orient
    
    def compute_orientation_features(self, activations: List[Activation], orientation: int) -> Optional[np.ndarray]:
        canonical = self.to_canonical_orientation(orientation)
        relevant = [a for a in activations if self.to_canonical_orientation(a.orientation) == canonical]
        
        if not relevant:
            return None
        
        total_weight = sum(a.strength for a in relevant)
        if total_weight < 1e-8:
            return None
        
        aggregated = np.zeros(self.feature_dim, dtype=np.float32)
        for a in relevant:
            aggregated += a.feature_response * a.strength
        aggregated /= total_weight
        
        return self._normalize(aggregated)
    
    def compute_anchors(self, activations: List[Activation]) -> Optional[Anchors]:
        if not activations:
            return None
        
        total_weight = sum(a.strength for a in activations)
        if total_weight < 1e-8:
            return None
        
        cx = sum(a.x * a.strength for a in activations) / total_weight
        cy = sum(a.y * a.strength for a in activations) / total_weight
        center = (cx, cy)
        
        max_dist = -1
        head = center
        for a in activations:
            dist_sq = (a.x - cx)**2 + (a.y - cy)**2
            if dist_sq > max_dist:
                max_dist = dist_sq
                head = (a.x, a.y)
        
        max_span = -1
        tail = center
        for a in activations:
            dist_sq = (a.x - head[0])**2 + (a.y - head[1])**2
            if dist_sq > max_span:
                max_span = dist_sq
                tail = (a.x, a.y)
        
        return Anchors(center=center, head=head, tail=tail)
    
    def _classify_connection(self, anchors1: Anchors, anchors2: Anchors) -> Tuple[str, str]:
        if anchors1 is None or anchors2 is None:
            return "generic", "no_anchors"
        
        def dist(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        tight = self.port_threshold * 0.5
        medium = self.port_threshold * 0.75
        loose = self.port_threshold
        
        if dist(anchors1.center, anchors2.center) < tight:
            return "cross", "center-center"
        
        ends1 = [("head", anchors1.head), ("tail", anchors1.tail)]
        ends2 = [("head", anchors2.head), ("tail", anchors2.tail)]
        
        for name1, e1 in ends1:
            d_to_center = dist(e1, anchors2.center)
            d_to_head = dist(e1, anchors2.head)
            d_to_tail = dist(e1, anchors2.tail)
            if d_to_center < loose and d_to_center < min(d_to_head, d_to_tail):
                return "T_shape", f"{name1}-center"
        
        for name2, e2 in ends2:
            d_to_center = dist(anchors1.center, e2)
            d_to_head = dist(anchors1.head, e2)
            d_to_tail = dist(anchors1.tail, e2)
            if d_to_center < loose and d_to_center < min(d_to_head, d_to_tail):
                return "T_shape", f"center-{name2}"
        
        for name1, e1 in ends1:
            for name2, e2 in ends2:
                if dist(e1, e2) < medium:
                    return "L_shape", f"{name1}-{name2}"
        
        return "generic", "distant"
    
    def decompose(self, activations: List[Activation]) -> dict:
        clusters: Dict[int, List[Activation]] = defaultdict(list)
        for a in activations:
            canonical = self.to_canonical_orientation(a.orientation)
            clusters[canonical].append(a)
        
        components = {}
        for orient, acts in clusters.items():
            if len(acts) < 1:
                continue
            
            features = self.compute_orientation_features(activations, orient)
            anchors = self.compute_anchors(acts)
            
            if features is not None:
                components[orient] = {
                    'features': features,
                    'anchors': anchors,
                    'activations': acts
                }
        
        if len(components) <= 1:
            orient = list(components.keys())[0] if components else 0
            return {
                'is_atomic': True,
                'orientation': orient,
                'components': components
            }
        
        orientations = sorted(components.keys())
        
        # Get primary pair
        o1, o2 = orientations[0], orientations[1]
        
        a1 = components[o1].get('anchors')
        a2 = components[o2].get('anchors')
        
        relation, connection_type = self._classify_connection(a1, a2)
        
        descriptor = StructuralDescriptor(
            child_orientations=frozenset(orientations),
            relation=relation,
            connection_type=connection_type
        )
        
        return {
            'is_atomic': False,
            'descriptor': descriptor,
            'components': components,
            'orientations': orientations
        }
    
    def learn(self, activations: List[Activation], global_features: np.ndarray, label: str) -> Tuple[int, str]:
        """Learn a pattern with global features for label matching"""
        # Store global features for label-based matching
        self.label_patterns[label].append(global_features)
        
        # Update centroid
        patterns = self.label_patterns[label]
        self.label_centroids[label] = self._normalize(np.mean(patterns, axis=0))
        
        decomp = self.decompose(activations)
        
        if decomp['is_atomic']:
            orient = decomp['orientation']
            comp = decomp['components'].get(orient, {})
            features = comp.get('features')
            anchors = comp.get('anchors')
            
            if features is None:
                return -1, "no_features"
            
            # Find or create atomic
            best_id = None
            best_sim = -1.0
            for nid in self.orientation_to_atoms.get(orient, []):
                node = self.atomic_nodes[nid]
                sim = self._cosine(features, node.feature_signature)
                if sim > best_sim:
                    best_sim = sim
                    best_id = nid
            
            if best_id is not None and best_sim >= self.similarity_threshold:
                self.atomic_nodes[best_id].access_count += 1
                return best_id, "exists"
            
            # Add new
            node = AtomicNode(
                id=self.next_id,
                orientation=orient,
                feature_signature=self._normalize(features).copy(),
                exemplar_anchors=anchors,
                label=label
            )
            self.atomic_nodes[self.next_id] = node
            self.orientation_to_atoms[orient].append(self.next_id)
            self.next_id += 1
            return node.id, "atomic"
        
        else:
            descriptor = decomp['descriptor']
            
            # Check existing
            if descriptor in self.structure_index:
                nid = self.structure_index[descriptor]
                self.composite_nodes[nid].access_count += 1
                return nid, "exists"
            
            # Add new
            node = CompositeNode(
                id=self.next_id,
                descriptor=descriptor,
                child_orientations=decomp['orientations'],
                label=label
            )
            self.composite_nodes[self.next_id] = node
            self.structure_index[descriptor] = self.next_id
            self.next_id += 1
            return node.id, "composite"
    
    def recognize(self, activations: List[Activation], global_features: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize using label centroids (more suitable for MNIST)"""
        if not self.label_centroids:
            return None, 0.0
        
        # Match against label centroids
        best_label = None
        best_sim = -1.0
        
        norm_feat = self._normalize(global_features)
        for label, centroid in self.label_centroids.items():
            sim = float(np.dot(norm_feat, centroid))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        
        return best_label, best_sim


class MNISTVisionSystem:
    """Vision system for MNIST"""
    
    def __init__(self):
        self.v1 = V1GaborBank(kernel_size=5)
        self.memory = StructuralMemory(
            feature_dim=self.v1.n_features,
            image_size=28.0
        )
    
    def train(self, image: np.ndarray, label: str):
        """Train on a single image"""
        activations = self.v1.extract_activations(image)
        global_features = self.v1.extract_global_features(image)
        self.memory.learn(activations, global_features, label)
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict label for image"""
        activations = self.v1.extract_activations(image)
        global_features = self.v1.extract_global_features(image)
        label, conf = self.memory.recognize(activations, global_features)
        return label or "unknown", conf


# =============================================================================
# Main Experiment
# =============================================================================

def run_mnist_experiment(
    n_train: int = 10000,
    n_test: int = 1000,
    verbose: bool = True
):
    """
    Run MNIST experiment with Structon Vision.
    
    Args:
        n_train: Number of training samples (max 60000)
        n_test: Number of test samples (max 10000)
        verbose: Print progress
    """
    print("=" * 60)
    print("Structon Vision v6.1 - MNIST Experiment")
    print("=" * 60)
    
    # Load data
    print("\nLoading MNIST data...")
    train_images, train_labels, test_images, test_labels = load_mnist()
    print(f"  Train: {len(train_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Limit samples
    n_train = min(n_train, len(train_images))
    n_test = min(n_test, len(test_images))
    
    print(f"\nUsing {n_train} train, {n_test} test samples")
    
    # Create system
    system = MNISTVisionSystem()
    
    # Training
    print("\n" + "=" * 40)
    print("Training...")
    print("=" * 40)
    
    start_time = time.time()
    
    for i in range(n_train):
        image = train_images[i]
        label = str(train_labels[i])
        system.train(image, label)
        
        if verbose and (i + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            print(f"  Trained {i+1}/{n_train} ({rate:.1f} samples/sec)")
    
    train_time = time.time() - start_time
    print(f"\nTraining complete in {train_time:.1f}s")
    print(f"  Atomic nodes: {len(system.memory.atomic_nodes)}")
    print(f"  Composite nodes: {len(system.memory.composite_nodes)}")
    print(f"  Label centroids: {len(system.memory.label_centroids)}")
    
    # Testing
    print("\n" + "=" * 40)
    print("Testing...")
    print("=" * 40)
    
    results = {str(d): {'correct': 0, 'total': 0} for d in range(10)}
    
    start_time = time.time()
    
    for i in range(n_test):
        image = test_images[i]
        true_label = str(test_labels[i])
        
        pred_label, conf = system.predict(image)
        
        results[true_label]['total'] += 1
        if pred_label == true_label:
            results[true_label]['correct'] += 1
        
        if verbose and (i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            current_acc = sum(r['correct'] for r in results.values()) / (i + 1) * 100
            print(f"  Tested {i+1}/{n_test} ({rate:.1f} samples/sec) - Running acc: {current_acc:.1f}%")
    
    test_time = time.time() - start_time
    
    # Results
    print("\n" + "=" * 40)
    print("Results")
    print("=" * 40)
    
    total_correct = sum(r['correct'] for r in results.values())
    total_count = sum(r['total'] for r in results.values())
    
    print(f"\nOverall Accuracy: {total_correct/total_count*100:.1f}%")
    print(f"Random Baseline: 10.0%")
    print(f"\nPer-digit accuracy:")
    
    for digit in range(10):
        r = results[str(digit)]
        if r['total'] > 0:
            acc = r['correct'] / r['total'] * 100
            print(f"  Digit {digit}: {acc:5.1f}% ({r['correct']}/{r['total']})")
    
    print(f"\nTiming:")
    print(f"  Training: {train_time:.1f}s ({n_train/train_time:.1f} samples/sec)")
    print(f"  Testing: {test_time:.1f}s ({n_test/test_time:.1f} samples/sec)")
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print("=" * 60)
    
    return system, results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Structon Vision MNIST Test')
    parser.add_argument('--train', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--test', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--quiet', action='store_true', help='Reduce output')
    
    args = parser.parse_args()
    
    run_mnist_experiment(
        n_train=args.train,
        n_test=args.test,
        verbose=not args.quiet
    )
