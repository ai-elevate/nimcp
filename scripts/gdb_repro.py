#!/usr/bin/env python3
"""Reproduce SIGSEGV with all elements: HYBRID training + periodic consolidation.
Run under GDB: gdb -batch -ex run -ex bt -ex 'thread apply all bt' --args python3 scripts/gdb_repro.py"""
import sys
import os
import random
import faulthandler

faulthandler.enable()

import nimcp

def main():
    print("Creating brain (1024 in, 64 out, 1.5M neurons)...", flush=True)
    brain = nimcp.Brain("crash_test", num_inputs=1024, num_outputs=64, neuron_count=1500000)
    info = brain.probe()
    print(f"Brain created: neurons={info['num_neurons']}, mem={info['memory_bytes']/(1024**3):.1f}GB", flush=True)

    # Enable multi-network training (LNN + CNN) like real Athena pipeline
    try:
        brain.enable_multi_network()
        print("Multi-network (HYBRID) mode enabled", flush=True)
    except Exception as e:
        print(f"Multi-network failed: {e}", flush=True)

    labels = ["anthropology:culture", "biology:cell", "physics:quantum",
              "ethics:trolley", "history:rome", "economics:market",
              "chemistry:bond", "psychology:cognitive"]

    for i in range(500):
        features = [random.gauss(0, 1) for _ in range(1024)]
        label = random.choice(labels)
        try:
            pred, conf = brain.predict_fast(features)
            loss = brain.learn(features, label, 0.7)
            if i % 10 == 0:
                print(f"  learn #{i}: loss={loss:.6f}", flush=True)
        except Exception as e:
            print(f"  learn #{i}: EXCEPTION: {e}", flush=True)
            break

        # Periodic consolidation like the coordinator does
        if i > 0 and i % 50 == 0:
            print(f"  === Consolidating at step {i} ===", flush=True)
            try:
                brain.consolidate()
                print(f"  === Consolidation done ===", flush=True)
            except Exception as e:
                print(f"  === Consolidation error: {e} ===", flush=True)

    print("Done — survived 500 learn calls!", flush=True)

if __name__ == "__main__":
    main()
