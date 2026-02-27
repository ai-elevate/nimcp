#!/usr/bin/env python3
"""Minimal repro for SIGSEGV during brain.learn() — run with ASAN-enabled nimcp.so"""
import sys
import os
import random

# Must preload ASAN library for it to work with Python-loaded .so
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libasan.so.8')

import nimcp

def main():
    print("Creating brain...", flush=True)
    # Use small brain for ASAN (large brain is too slow under instrumentation)
    brain = nimcp.Brain("asan_test", num_inputs=64, num_outputs=8, neuron_count=5000)
    print(f"Brain created: {brain.probe()}", flush=True)

    labels = ["cat", "dog", "bird", "fish", "tree", "rock", "sky", "sun"]

    for i in range(500):
        features = [random.gauss(0, 1) for _ in range(64)]
        label = random.choice(labels)
        try:
            loss = brain.learn(features, label)
            if i % 5 == 0:
                print(f"  learn #{i}: loss={loss:.6f}", flush=True)
        except Exception as e:
            print(f"  learn #{i}: EXCEPTION: {e}", flush=True)
            break

    print("Done!", flush=True)

if __name__ == "__main__":
    main()
