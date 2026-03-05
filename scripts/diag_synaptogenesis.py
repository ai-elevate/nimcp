#!/usr/bin/env python3
"""Diagnostic: Does synaptogenesis actually fire during learn_vector?"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '.venv', 'lib', 'python3.12', 'site-packages'))
import nimcp
import numpy as np

# Use 50K neurons for fast iteration (backbone = 1024-32768, same code path)
N = 50_000
print(f"Creating fresh brain with {N} neurons...")
t0 = time.time()
brain = nimcp.Brain("diag", num_inputs=1024, num_outputs=2048, neuron_count=N)
print(f"  Created in {time.time()-t0:.1f}s")

brain.set_fast_training(False)
try:
    brain.enable_biological_plasticity(True)
    print("  Biological plasticity: ON")
except:
    print("  Biological plasticity: FAILED")

# Infer with zero input
print(f"\n=== Zero-input inference ===")
zero_features = [0.0] * 1024
r = brain.decide_full(zero_features)
ov = np.array(r.get("output_vector", []))
if len(ov) > 0:
    print(f"  |output|={np.linalg.norm(ov):.4f} mean={ov.mean():.6f} std={ov.std():.6f} max={ov.max():.4f}")

# Infer with random input
print(f"\n=== Random-input inference ===")
np.random.seed(42)
rand_features = list(np.random.randn(1024).astype(float))
r2 = brain.decide_full(rand_features)
ov2 = np.array(r2.get("output_vector", []))
if len(ov2) > 0:
    print(f"  |output|={np.linalg.norm(ov2):.4f} mean={ov2.mean():.6f} std={ov2.std():.6f} max={ov2.max():.4f}")
    diff = ov2 - ov
    print(f"  diff from zero: |diff|={np.linalg.norm(diff):.6f} max_abs={np.abs(diff).max():.6f}")

# Do 20 learn steps and check output
print(f"\n=== Learning 20 examples ===")
target = np.zeros(2048, dtype=float)
target[0] = 1.0  # teach it "class 0"
for i in range(20):
    features = list(np.random.randn(1024).astype(float))
    loss = brain.learn_vector(features, list(target), label="test", confidence=0.5)
    if i % 5 == 0:
        print(f"  Step {i}: loss={loss:.6f}")

# Check output after learning
print(f"\n=== Post-learning inference ===")
r3 = brain.decide_full(rand_features)
ov3 = np.array(r3.get("output_vector", []))
if len(ov3) > 0:
    print(f"  |output|={np.linalg.norm(ov3):.4f} mean={ov3.mean():.6f} std={ov3.std():.6f} max={ov3.max():.4f}")
    diff2 = ov3 - ov
    print(f"  diff from pre-train: |diff|={np.linalg.norm(diff2):.6f} max_abs={np.abs(diff2).max():.6f}")

print("\nDone")
