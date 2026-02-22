#!/usr/bin/env python3
"""
Diagnostic: Verify that brain weight matrices are updated during training.

Tests:
1. Predict before training → get baseline predictions
2. Train on labeled examples
3. Predict again → predictions should change
4. Train more → predictions should converge toward labels
"""

import sys
import os

# Add build directory to path
build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
sys.path.insert(0, build_dir)

# Suppress C library debug output
fd = os.open(os.devnull, os.O_WRONLY)
old_stderr = os.dup(2)
os.dup2(fd, 2)

import nimcp

# Restore stderr for Python output
os.dup2(old_stderr, 2)
os.close(fd)
os.close(old_stderr)

def main():
    print("=" * 60)
    print("NIMCP Weight Update Verification")
    print("=" * 60)

    # Create a TINY brain with known dimensions
    # size=0 is BRAIN_TINY (100 neurons, <1MB)
    brain = nimcp.Brain("verify_test", size=0, num_inputs=4, num_outputs=3)
    print(f"\nBrain created: TINY, 4 inputs, 3 outputs")

    # Define simple training data (iris-like)
    class_a = [5.1, 3.5, 1.4, 0.2]  # setosa-like
    class_b = [7.0, 3.2, 4.7, 1.4]  # versicolor-like
    class_c = [6.3, 3.3, 6.0, 2.5]  # virginica-like

    # === Test 1: Baseline predictions (untrained) ===
    print("\n--- Test 1: Baseline predictions (untrained brain) ---")
    pred_a_before = brain.predict(class_a)
    pred_b_before = brain.predict(class_b)
    pred_c_before = brain.predict(class_c)
    print(f"  class_a → {pred_a_before}")
    print(f"  class_b → {pred_b_before}")
    print(f"  class_c → {pred_c_before}")

    # === Test 2: Train on a few examples ===
    print("\n--- Test 2: Training 10 examples per class ---")
    for i in range(10):
        brain.learn(class_a, "setosa", 1.0)
        brain.learn(class_b, "versicolor", 1.0)
        brain.learn(class_c, "virginica", 1.0)
    print("  Trained 30 examples (10 per class)")

    # === Test 3: Predict after training ===
    print("\n--- Test 3: Predictions after 30 examples ---")
    pred_a_after = brain.predict(class_a)
    pred_b_after = brain.predict(class_b)
    pred_c_after = brain.predict(class_c)
    print(f"  class_a → {pred_a_after}")
    print(f"  class_b → {pred_b_after}")
    print(f"  class_c → {pred_c_after}")

    # Check if predictions changed
    changed_a = pred_a_before != pred_a_after
    changed_b = pred_b_before != pred_b_after
    changed_c = pred_c_before != pred_c_after
    print(f"\n  Prediction changed for class_a: {changed_a}")
    print(f"  Prediction changed for class_b: {changed_b}")
    print(f"  Prediction changed for class_c: {changed_c}")

    # === Test 4: Train more and check convergence ===
    print("\n--- Test 4: Training 200 more examples per class ---")
    for i in range(200):
        brain.learn(class_a, "setosa", 1.0)
        brain.learn(class_b, "versicolor", 1.0)
        brain.learn(class_c, "virginica", 1.0)
    print("  Trained 600 additional examples (200 per class)")

    pred_a_final = brain.predict(class_a)
    pred_b_final = brain.predict(class_b)
    pred_c_final = brain.predict(class_c)
    print(f"  class_a → {pred_a_final}")
    print(f"  class_b → {pred_b_final}")
    print(f"  class_c → {pred_c_final}")

    # === Test 5: Check if correct labels are predicted ===
    print("\n--- Test 5: Label accuracy check ---")
    correct = 0
    total = 3
    if pred_a_final[0] == "setosa":
        correct += 1
        print(f"  class_a: CORRECT (predicted 'setosa')")
    else:
        print(f"  class_a: WRONG (predicted '{pred_a_final[0]}', expected 'setosa')")

    if pred_b_final[0] == "versicolor":
        correct += 1
        print(f"  class_b: CORRECT (predicted 'versicolor')")
    else:
        print(f"  class_b: WRONG (predicted '{pred_b_final[0]}', expected 'versicolor')")

    if pred_c_final[0] == "virginica":
        correct += 1
        print(f"  class_c: CORRECT (predicted 'virginica')")
    else:
        print(f"  class_c: WRONG (predicted '{pred_c_final[0]}', expected 'virginica')")

    print(f"\n  Accuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    # === Test 6: Confidence trend ===
    print("\n--- Test 6: Confidence trend over training ---")
    brain2 = nimcp.Brain("verify_trend", size=0, num_inputs=4, num_outputs=3)
    test_input = [5.1, 3.5, 1.4, 0.2]
    confidences = []
    for epoch in range(0, 501, 50):
        pred = brain2.predict(test_input)
        confidences.append((epoch, pred[0], pred[1] if len(pred) > 1 else 0.0))
        if epoch < 500:
            for _ in range(50):
                brain2.learn(test_input, "setosa", 1.0)

    for epoch, label, conf in confidences:
        bar = "#" * int(conf * 50) if isinstance(conf, float) else ""
        print(f"  Epoch {epoch:4d}: label='{label}', conf={conf:.4f} {bar}")

    # === Summary ===
    print("\n" + "=" * 60)
    any_changed = changed_a or changed_b or changed_c
    if any_changed:
        print("RESULT: WEIGHTS ARE BEING UPDATED")
        print("  Predictions changed after training → learning is working.")
    else:
        print("RESULT: WARNING — WEIGHTS MAY NOT BE UPDATING")
        print("  Predictions did NOT change after training!")
    print("=" * 60)

    nimcp.shutdown()


if __name__ == "__main__":
    main()
