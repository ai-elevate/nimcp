#!/usr/bin/env python3
"""
Test Basic Brain Persistence - Without auto-resize
"""

import os
import sys
import numpy as np
from pathlib import Path

# Set LD_PRELOAD for AddressSanitizer
if 'LD_PRELOAD' not in os.environ:
    asan_lib = '/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so'
    if os.path.exists(asan_lib):
        os.environ['LD_PRELOAD'] = asan_lib
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Add NIMCP to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src/bindings/python"))

try:
    import nimcp
    print("✓ NIMCP Python bindings loaded")
except ImportError as e:
    print(f"✗ Failed to load NIMCP: {e}")
    sys.exit(1)

def test_save_load():
    """Test basic save and load without resize"""
    print("\n" + "="*60)
    print("TEST: Basic Save/Load (No Resize)")
    print("="*60)

    checkpoint_path = "/tmp/test_brain_simple.checkpoint"

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Create brain
    print("\n1. Creating brain...")
    brain1 = nimcp.Brain('test_brain', 0, 0, 10, 5)

    # Get neuron count
    try:
        initial_size = brain1.get_neuron_count()
        print(f"   ✓ Brain created ({initial_size} neurons)")
    except Exception as e:
        print(f"   ✗ Failed to get neuron count: {e}")
        return False

    # Train
    print("\n2. Training (100 examples)...")
    for i in range(100):
        features = np.random.rand(10).tolist()
        label = str(i % 5)
        brain1.learn(features, label, 0.8)
    print("   ✓ Training complete")

    # Save
    print("\n3. Saving brain...")
    try:
        brain1.save(checkpoint_path)
        print(f"   ✓ Saved to {checkpoint_path}")
    except Exception as e:
        print(f"   ✗ Save failed: {e}")
        return False

    # Verify file
    if not os.path.exists(checkpoint_path):
        print("   ✗ Checkpoint file not created!")
        return False

    file_size = os.path.getsize(checkpoint_path)
    print(f"   ✓ Checkpoint: {file_size:,} bytes")

    # Destroy
    print("\n4. Destroying original brain...")
    del brain1
    print("   ✓ Destroyed")

    # Load
    print("\n5. Loading from checkpoint...")
    try:
        brain2 = nimcp.Brain.load(checkpoint_path)
        loaded_size = brain2.get_neuron_count()
        print(f"   ✓ Loaded ({loaded_size} neurons)")
    except Exception as e:
        print(f"   ✗ Load failed: {e}")
        return False

    # Verify
    if loaded_size != initial_size:
        print(f"   ✗ Size mismatch: {initial_size} → {loaded_size}")
        return False

    print(f"   ✓ Size verified: {loaded_size} neurons")

    # Continue training
    print("\n6. Continuing training (50 examples)...")
    for i in range(50):
        features = np.random.rand(10).tolist()
        label = str(i % 5)
        brain2.learn(features, label, 0.8)
    print("   ✓ Training resumed")

    # Cleanup
    del brain2
    os.remove(checkpoint_path)

    print("\n✅ TEST PASSED: Save/load works!")
    return True


if __name__ == "__main__":
    print("\n🧪 NIMCP BASIC PERSISTENCE TEST")
    print("="*60)

    try:
        result = test_save_load()
        sys.exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
