#!/usr/bin/env python3
"""
Test Brain Persistence - Verify save/load functionality
Tests that brain can save its state and resume training from checkpoints
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

def test_basic_persistence():
    """Test basic save and load functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic Save/Load")
    print("="*60)

    checkpoint_path = "/tmp/test_brain_basic.checkpoint"

    # Clean up any existing checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Create brain
    print("\n1. Creating brain...")
    brain1 = nimcp.Brain(
        'test_brain',
        0,  # TINY
        0,  # CLASSIFICATION
        10, # inputs
        5   # outputs
    )
    initial_size = brain1.get_neuron_count()
    print(f"   ✓ Brain created ({initial_size} neurons)")

    # Train a bit
    print("\n2. Training brain (100 examples)...")
    for i in range(100):
        features = np.random.rand(10).tolist()
        label = str(i % 5)
        brain1.learn(features, label, 0.8)
    print("   ✓ Training complete")

    # Save
    print("\n3. Saving brain to checkpoint...")
    brain1.save(checkpoint_path)
    print(f"   ✓ Saved to {checkpoint_path}")

    # Verify file exists
    if not os.path.exists(checkpoint_path):
        print("   ✗ ERROR: Checkpoint file not created!")
        return False

    file_size = os.path.getsize(checkpoint_path)
    print(f"   ✓ Checkpoint file size: {file_size:,} bytes")

    # Destroy original brain
    print("\n4. Destroying original brain...")
    del brain1
    print("   ✓ Original brain destroyed")

    # Load from checkpoint
    print("\n5. Loading brain from checkpoint...")
    brain2 = nimcp.Brain.load(checkpoint_path)
    loaded_size = brain2.get_neuron_count()
    print(f"   ✓ Brain loaded ({loaded_size} neurons)")

    # Verify neuron count matches
    if loaded_size != initial_size:
        print(f"   ✗ ERROR: Neuron count mismatch! {initial_size} → {loaded_size}")
        return False

    print(f"   ✓ Neuron count verified: {loaded_size} neurons")

    # Continue training
    print("\n6. Continuing training (50 more examples)...")
    for i in range(50):
        features = np.random.rand(10).tolist()
        label = str(i % 5)
        brain2.learn(features, label, 0.8)
    print("   ✓ Training resumed successfully")

    # Cleanup
    del brain2
    os.remove(checkpoint_path)

    print("\n✅ TEST 1 PASSED: Basic save/load works!")
    return True


def test_persistence_with_resize():
    """Test persistence after brain resize"""
    print("\n" + "="*60)
    print("TEST 2: Save/Load with Brain Resize")
    print("="*60)

    checkpoint_path = "/tmp/test_brain_resize.checkpoint"

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Create small brain
    print("\n1. Creating TINY brain...")
    brain1 = nimcp.Brain('test_resize', 0, 0, 10, 5)
    initial_size = brain1.get_neuron_count()
    print(f"   ✓ Brain created ({initial_size} neurons)")

    # Train and trigger resize
    print("\n2. Training to trigger resize...")
    for i in range(300):
        features = np.random.rand(10).tolist()
        label = str(i % 5)
        brain1.learn(features, label, 0.8)

        # Try auto-resize every 100 examples
        if (i + 1) % 100 == 0:
            resized = brain1.auto_resize()
            if resized:
                new_size = brain1.get_neuron_count()
                print(f"   🔧 Auto-resized: {initial_size} → {new_size} neurons")
                break

    resized_size = brain1.get_neuron_count()

    # Save after resize
    print("\n3. Saving resized brain...")
    brain1.save(checkpoint_path)
    print(f"   ✓ Saved ({resized_size} neurons)")

    # Destroy
    del brain1

    # Load and verify
    print("\n4. Loading resized brain...")
    brain2 = nimcp.Brain.load(checkpoint_path)
    loaded_size = brain2.get_neuron_count()
    print(f"   ✓ Brain loaded ({loaded_size} neurons)")

    if loaded_size != resized_size:
        print(f"   ✗ ERROR: Size mismatch! {resized_size} → {loaded_size}")
        return False

    print(f"   ✓ Size verified: {loaded_size} neurons")

    # Continue training
    print("\n5. Continuing training after resize...")
    for i in range(50):
        features = np.random.rand(10).tolist()
        label = str(i % 5)
        brain2.learn(features, label, 0.8)
    print("   ✓ Training resumed successfully")

    # Cleanup
    del brain2
    os.remove(checkpoint_path)

    print("\n✅ TEST 2 PASSED: Save/load with resize works!")
    return True


def test_multiple_saves():
    """Test multiple save/load cycles"""
    print("\n" + "="*60)
    print("TEST 3: Multiple Save/Load Cycles")
    print("="*60)

    checkpoint_path = "/tmp/test_brain_multi.checkpoint"

    # Clean up
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Create brain
    print("\n1. Creating brain...")
    brain = nimcp.Brain('test_multi', 0, 0, 10, 5)
    initial_size = brain.get_neuron_count()
    print(f"   ✓ Brain created ({initial_size} neurons)")

    # Multiple save/load cycles
    for cycle in range(3):
        print(f"\n2.{cycle+1}. Cycle {cycle+1}:")

        # Train
        print(f"   Training (50 examples)...")
        for i in range(50):
            features = np.random.rand(10).tolist()
            label = str(i % 5)
            brain.learn(features, label, 0.8)

        # Save
        print(f"   Saving...")
        brain.save(checkpoint_path)
        current_size = brain.get_neuron_count()
        print(f"   ✓ Saved ({current_size} neurons)")

        # Destroy and reload
        del brain
        print(f"   Loading...")
        brain = nimcp.Brain.load(checkpoint_path)
        loaded_size = brain.get_neuron_count()

        if loaded_size != current_size:
            print(f"   ✗ ERROR: Size mismatch in cycle {cycle+1}!")
            return False

        print(f"   ✓ Loaded ({loaded_size} neurons)")

    # Cleanup
    del brain
    os.remove(checkpoint_path)

    print("\n✅ TEST 3 PASSED: Multiple save/load cycles work!")
    return True


def main():
    print("\n" + "="*60)
    print("🧪 NIMCP BRAIN PERSISTENCE TESTS")
    print("="*60)
    print("Testing brain checkpoint save/load functionality...")

    tests = [
        ("Basic Save/Load", test_basic_persistence),
        ("Save/Load with Resize", test_persistence_with_resize),
        ("Multiple Cycles", test_multiple_saves),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print("="*60)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
