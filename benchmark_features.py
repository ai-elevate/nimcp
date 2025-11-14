#!/usr/bin/env python3
"""
Benchmark feature extraction methods
"""
import time
import numpy as np

# Test without optimizations (baseline)
def extract_features_baseline(text: str, dim: int = 512) -> np.ndarray:
    """Original simple feature extraction"""
    features = np.zeros(dim, dtype=np.float32)

    # Character frequencies
    for char in text[:1000]:
        idx = ord(char) % 256
        if idx < dim:
            features[idx] += 1.0

    # Word hashes
    words = text.lower().split()[:128]
    for i, word in enumerate(words):
        if 256 + i < dim:
            features[256 + i] = hash(word) % 1000 / 1000.0

    # Normalize
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm

    return features

# Test with xxhash
try:
    import xxhash
    XXHASH_AVAILABLE = True
except ImportError:
    XXHASH_AVAILABLE = False
    print("xxhash not available - install with: pip install xxhash")

def extract_features_optimized(text: str, dim: int = 512) -> np.ndarray:
    """Optimized hash-based feature extraction"""
    features = np.zeros(dim, dtype=np.float32)

    # Truncate text once
    text_bytes = text[:1000].encode('utf-8', errors='ignore')

    # Character frequencies
    char_counts = np.zeros(256, dtype=np.float32)
    for byte in text_bytes:
        if byte < 256:
            char_counts[byte] += 1.0
    features[:256] = char_counts

    # Fast word hashing
    words = text[:1000].lower().split()[:128]
    if XXHASH_AVAILABLE:
        for i, word in enumerate(words):
            if 256 + i < dim:
                h = xxhash.xxh32(word).intdigest()
                features[256 + i] = (h % 1000) / 1000.0
    else:
        for i, word in enumerate(words):
            if 256 + i < dim:
                features[256 + i] = (hash(word) % 1000) / 1000.0

    # Fast normalization
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features /= norm

    return features

# Test data
test_texts = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "In computer science, artificial intelligence refers to machine learning systems. " * 5,
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)" * 8,
    "Neural networks learn patterns from data through backpropagation and gradient descent. " * 7
]

print("=" * 60)
print("FEATURE EXTRACTION BENCHMARK")
print("=" * 60)
print(f"Test samples: {len(test_texts)}")
print(f"Iterations per method: 1000")
print("=" * 60)

# Benchmark baseline
print("\n📊 Baseline (original hash-based)...")
start = time.time()
for _ in range(1000):
    for text in test_texts:
        _ = extract_features_baseline(text)
baseline_time = time.time() - start
baseline_rate = (1000 * len(test_texts)) / baseline_time
print(f"  Time: {baseline_time:.2f}s")
print(f"  Rate: {baseline_rate:.1f} extractions/sec")

# Benchmark optimized
print("\n⚡ Optimized (xxhash + vectorization)...")
start = time.time()
for _ in range(1000):
    for text in test_texts:
        _ = extract_features_optimized(text)
optimized_time = time.time() - start
optimized_rate = (1000 * len(test_texts)) / optimized_time
print(f"  Time: {optimized_time:.2f}s")
print(f"  Rate: {optimized_rate:.1f} extractions/sec")

# Summary
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x faster")
print(f"Time saved per 10K examples: {(baseline_time - optimized_time) * 10:.1f}s ({(baseline_time - optimized_time) * 10 / 60:.1f} min)")
print("=" * 60)
