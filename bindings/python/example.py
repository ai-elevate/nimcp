#!/usr/bin/env python3
"""
Example usage of NIMCP Brain API Python bindings

This demonstrates how to use NIMCP for fast pattern learning and decision making.
"""

from nimcp_brain import Brain, BrainSize, BrainTask, create_classifier

def ethics_demo():
    """Demonstrate ethics decision caching (similar to C demo)"""
    print("=" * 50)
    print(" NIMCP Brain Python API Demo")
    print(" Use Case: Ethics Decision Caching")
    print("=" * 50)
    print()

    # Create brain for ethics decisions
    brain = Brain(
        task_name="artemis_ethics",
        size=BrainSize.SMALL,
        task=BrainTask.CLASSIFICATION,
        num_inputs=4,   # harm, fairness, transparency, autonomy
        num_outputs=3   # allow, warn, block
    )

    print(f"Created: {brain}")
    print()

    # Training data (simulate LLM decisions)
    training_data = [
        # (features, label, confidence)
        ([0.9, 0.5, 0.5, 0.5], "block", 0.95),  # High harm
        ([0.2, 0.8, 0.8, 0.8], "allow", 0.90),  # Safe
        ([0.5, 0.2, 0.5, 0.5], "block", 0.85),  # Unfair
        ([0.3, 0.6, 0.1, 0.7], "warn", 0.75),   # Low transparency
        ([0.1, 0.7, 0.8, 0.9], "allow", 0.90),  # Good all around
        ([0.8, 0.3, 0.3, 0.4], "block", 0.90),  # High harm + unfair
        ([0.4, 0.5, 0.2, 0.5], "warn", 0.70),   # Some concerns
    ]

    # Train the brain
    print("Training from LLM decisions...")
    avg_loss = brain.learn_batch(training_data)
    print(f"  Average loss: {avg_loss:.4f}")
    print()

    # Test inference
    print("Testing fast inference (no LLM needed):")
    print()

    test_cases = [
        ([0.9, 0.5, 0.5, 0.5], "High harm scenario"),
        ([0.2, 0.8, 0.8, 0.8], "Safe, fair scenario"),
        ([0.5, 0.2, 0.5, 0.5], "Moderate harm, unfair"),
        ([0.3, 0.6, 0.1, 0.7], "Low transparency"),
    ]

    total_time = 0
    for features, description in test_cases:
        decision = brain.decide(features)

        print(f"Test Case: {description}")
        print(f"  Features: harm={features[0]:.2f}, fair={features[1]:.2f}, "
              f"trans={features[2]:.2f}, auto={features[3]:.2f}")
        print(f"  Decision: {decision.label} (confidence: {decision.confidence:.2f})")
        print(f"  Active neurons: {decision.num_active_neurons} "
              f"(sparsity: {decision.sparsity * 100:.1f}%)")
        print(f"  Inference time: {decision.inference_time_ms:.3f} ms")
        print(f"  Explanation: {decision.explanation}")
        print()

        total_time += decision.inference_time_us

    avg_time_ms = (total_time / len(test_cases)) / 1000.0

    print("Performance Comparison:")
    print(f"  LLM API call: ~500-2000 ms")
    print(f"  NIMCP Brain: ~{avg_time_ms:.3f} ms")
    print(f"  Speedup: ~{1000.0 / avg_time_ms:.0f}x faster")
    print()

    # Show interpretability
    print("Interpretability - Top Contributing Neurons:")
    top_neurons = brain.get_top_neurons(5)
    for neuron_id, importance in top_neurons:
        print(f"  Neuron {neuron_id}: importance = {importance:.4f}")
    print()

    # Get statistics
    stats = brain.get_stats()
    print("Brain Statistics:")
    print(f"  Name: {stats.task_name}")
    print(f"  Neurons: {stats.num_neurons}")
    print(f"  Training steps: {stats.total_learning_steps}")
    print(f"  Inferences: {stats.total_inferences}")
    print(f"  Avg inference time: {stats.avg_inference_time_ms:.3f} ms")
    print(f"  Avg sparsity: {stats.avg_sparsity * 100:.1f}%")
    print(f"  Memory: {stats.memory_mb:.2f} MB")
    print()

    # Save brain
    print("Saving trained brain...")
    brain.save("artemis_ethics_brain.nimcp")
    print("  Saved to: artemis_ethics_brain.nimcp")
    print()

    # Test loading
    print("Testing load...")
    loaded_brain = Brain.load("artemis_ethics_brain.nimcp")
    print(f"  Loaded: {loaded_brain}")

    # Verify loaded brain works
    test_features = [0.8, 0.3, 0.5, 0.6]
    decision = loaded_brain.decide(test_features)
    print(f"  Test decision: {decision.label} (confidence: {decision.confidence:.2f})")
    print()

    print("=" * 50)
    print(" Summary")
    print("=" * 50)
    print()
    print("Benefits:")
    print("  ✓ 100-1000x faster than LLM calls")
    print("  ✓ Works offline (no API dependency)")
    print("  ✓ Zero cost per inference")
    print("  ✓ Privacy preserved (local inference)")
    print("  ✓ Interpretable (can see active neurons)")
    print(f"  ✓ Lightweight (~{stats.memory_mb:.1f}MB vs 7GB+ for LLMs)")
    print()

def simple_classification_demo():
    """Simple classification example"""
    print("\n" + "=" * 50)
    print(" Simple Classification Example")
    print("=" * 50)
    print()

    # Create a small classifier
    brain = create_classifier("iris_classifier", num_inputs=4, num_outputs=3)

    print("Training on Iris-like data...")

    # Training examples (features: sepal_length, sepal_width, petal_length, petal_width)
    training_data = [
        ([5.1, 3.5, 1.4, 0.2], "setosa", 1.0),
        ([4.9, 3.0, 1.4, 0.2], "setosa", 1.0),
        ([7.0, 3.2, 4.7, 1.4], "versicolor", 1.0),
        ([6.4, 3.2, 4.5, 1.5], "versicolor", 1.0),
        ([6.3, 3.3, 6.0, 2.5], "virginica", 1.0),
        ([5.8, 2.7, 5.1, 1.9], "virginica", 1.0),
    ]

    avg_loss = brain.learn_batch(training_data)
    print(f"  Training complete. Average loss: {avg_loss:.4f}")
    print()

    # Test
    print("Testing:")
    test_sample = [5.0, 3.4, 1.5, 0.2]
    decision = brain.decide(test_sample)

    print(f"  Input: {test_sample}")
    print(f"  Prediction: {decision.label}")
    print(f"  Confidence: {decision.confidence:.2f}")
    print(f"  Inference time: {decision.inference_time_ms:.3f} ms")
    print()

if __name__ == "__main__":
    # Run ethics decision caching demo
    ethics_demo()

    # Run simple classification demo
    simple_classification_demo()

    print("Demo complete!")
