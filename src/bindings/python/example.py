#!/usr/bin/env python3
"""
NIMCP Python Bindings - Complete Example (v2.7.0)
==================================================

WHAT: Demonstrates all NIMCP Python API features
WHY:  Reference implementation showing proper usage patterns
HOW:  Create brain, train, predict, use v2.7 enhancements

Features Demonstrated:
- Brain creation with different sizes/tasks
- Single example learning and prediction
- Batch processing (v2.7.0)
- Checkpointing (v2.7.0)
- SIMD operations (v2.7.0)
- Error handling
- Save/load functionality

Usage:
    python example.py
"""

import nimcp
import os
import tempfile

def main():
    print("=" * 70)
    print("NIMCP Python Bindings Example (v2.7.0)")
    print("=" * 70)
    print(f"NIMCP Version: {nimcp.version()}\n")

    # =========================================================================
    # 1. Basic Brain Operations
    # =========================================================================
    print("1. Creating and training a brain...")

    # Create a small brain for classification with 3 inputs, 5 outputs
    brain = nimcp.Brain(
        name="iris_classifier",
        size=nimcp.BRAIN_SMALL,
        task=nimcp.TASK_CLASSIFICATION,
        num_inputs=3,
        num_outputs=5
    )

    # Train on some examples
    print("   Training on examples...")
    examples = [
        ([0.1, 0.2, 0.3], "setosa", 0.9),
        ([0.4, 0.5, 0.6], "versicolor", 0.8),
        ([0.7, 0.8, 0.9], "virginica", 0.95),
        ([0.15, 0.25, 0.35], "setosa", 0.85),
        ([0.45, 0.55, 0.65], "versicolor", 0.9),
    ]

    for features, label, confidence in examples:
        loss = brain.learn(features, label, confidence)
        print(f"   Learned {label}: loss = {loss:.4f}")

    # Make predictions
    print("\n   Making predictions...")
    test_features = [
        [0.12, 0.22, 0.32],  # Similar to setosa
        [0.72, 0.82, 0.92],  # Similar to virginica
    ]

    for features in test_features:
        label, conf = brain.predict(features)
        print(f"   Features {features} -> {label} (confidence: {conf:.3f})")

    # =========================================================================
    # 2. Batch Processing (v2.7.0 Enhancement)
    # =========================================================================
    print("\n2. Batch prediction (v2.7.0)...")

    batch_features = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [0.13, 0.23, 0.33],
    ]

    labels, confidences = brain.predict_batch(batch_features)
    print("   Batch prediction results:")
    for i, (label, conf) in enumerate(zip(labels, confidences)):
        print(f"   [{i}] {label} (confidence: {conf:.3f})")

    # =========================================================================
    # 3. Checkpointing (v2.7.0 Enhancement)
    # =========================================================================
    print("\n3. Checkpointing (v2.7.0)...")

    # Create temporary directory for checkpoints
    checkpoint_dir = tempfile.mkdtemp(prefix="nimcp_checkpoints_")
    print(f"   Checkpoint directory: {checkpoint_dir}")

    # Enable checkpointing
    brain.enable_checkpointing(
        checkpoint_dir=checkpoint_dir,
        interval_minutes=0,  # Manual checkpoints only
        max_checkpoints=5
    )
    print("   Checkpointing enabled")

    # Create manual checkpoint
    brain.checkpoint("after_initial_training")
    print("   Created checkpoint: after_initial_training")

    # Train more
    for i in range(3):
        loss = brain.learn([0.2 + i*0.1, 0.3 + i*0.1, 0.4 + i*0.1], "setosa", 0.9)

    # Another checkpoint
    brain.checkpoint("after_additional_training")
    print("   Created checkpoint: after_additional_training")

    # =========================================================================
    # 4. SIMD Operations (v2.7.0 Enhancement)
    # =========================================================================
    print("\n4. SIMD operations (v2.7.0)...")

    vec_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    vec_b = [2.0, 3.0, 4.0, 5.0, 6.0]

    dot_product = nimcp.simd_dot_product(vec_a, vec_b)
    print(f"   Dot product of {vec_a}")
    print(f"              and {vec_b}")
    print(f"              = {dot_product:.2f}")

    # =========================================================================
    # 5. Save and Load
    # =========================================================================
    print("\n5. Save and load brain...")

    # Save brain
    save_path = os.path.join(tempfile.gettempdir(), "brain_save.nimcp")
    brain.save(save_path)
    print(f"   Brain saved to: {save_path}")

    # Load brain
    loaded_brain = nimcp.Brain.load(save_path)
    print("   Brain loaded successfully")

    # Verify loaded brain works
    label, conf = loaded_brain.predict([0.1, 0.2, 0.3])
    print(f"   Loaded brain prediction: {label} (confidence: {conf:.3f})")

    # Clean up
    os.remove(save_path)

    # =========================================================================
    # 6. Error Handling
    # =========================================================================
    print("\n6. Error handling...")

    try:
        # Try to predict with wrong number of features
        bad_brain = nimcp.Brain("test", nimcp.BRAIN_TINY, nimcp.TASK_CLASSIFICATION, 5, 2)
        bad_brain.predict([1.0, 2.0])  # Only 2 features, needs 5
    except RuntimeError as e:
        print(f"   Caught expected error: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ Brain creation and configuration")
    print("  ✓ Single-example learning and prediction")
    print("  ✓ Batch prediction (v2.7.0)")
    print("  ✓ Checkpointing system (v2.7.0)")
    print("  ✓ SIMD operations (v2.7.0)")
    print("  ✓ Save/load functionality")
    print("  ✓ Error handling")
    print("\nFor more information, see:")
    print("  - API documentation: docs/python_api.md")
    print("  - NIMCP repository: https://github.com/youruser/nimcp")


if __name__ == "__main__":
    main()
