#!/usr/bin/env python3
"""
Test Trained Model
Load a trained NIMCP model and test it on validation/test data.

Features:
- Load saved models
- Evaluate on test sets
- Per-class accuracy
- Confusion matrix
- Confidence analysis
"""

import argparse
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import nimcp
except ImportError:
    print("Error: nimcp module not found")
    sys.exit(1)


class ModelTester:
    """Test trained NIMCP models"""

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.brain = self.load_model()

    def load_model(self) -> nimcp.Brain:
        """Load trained model"""
        print(f"Loading model: {self.model_path}")

        brain = nimcp.Brain.load(str(self.model_path))

        # Print model stats
        stats = brain.probe()
        print(f"  Neurons: {stats['num_neurons']:,}")
        print(f"  Synapses: {stats['num_synapses']:,}")
        print(f"  Learning steps: {stats['total_learning_steps']:,}")
        print(f"  Memory: {stats['memory_bytes'] / (1024*1024):.2f} MB")

        return brain

    def load_test_data(self, dataset_path: str, split: str = 'test') -> List[Dict]:
        """Load test dataset"""
        data_file = Path(dataset_path) / f'{split}.jsonl'

        if not data_file.exists():
            raise FileNotFoundError(f"Test data not found: {data_file}")

        samples = []
        with open(data_file, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        print(f"Loaded {len(samples)} test samples from {data_file}")
        return samples

    def test(self, samples: List[Dict]) -> Dict:
        """
        Test model on samples.

        Returns:
            Dict with test results
        """
        print("\nTesting model...")

        correct = 0
        total = 0
        total_confidence = 0.0

        # Per-class metrics
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        # Confusion matrix
        confusion = defaultdict(lambda: defaultdict(int))

        # Confidence analysis
        confidences = []

        for sample in samples:
            features = sample['features']
            true_label = sample['label']

            # Predict
            pred_label, confidence = self.brain.decide(features)

            total += 1
            total_confidence += confidence
            confidences.append(confidence)

            # Update metrics
            class_total[true_label] += 1

            if pred_label == true_label:
                correct += 1
                class_correct[true_label] += 1

            # Confusion matrix
            confusion[true_label][pred_label] += 1

        # Calculate overall accuracy
        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = total_confidence / total if total > 0 else 0.0

        # Per-class accuracy
        class_accuracy = {}
        for label in class_total:
            class_accuracy[label] = class_correct[label] / class_total[label]

        return {
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'class_accuracy': class_accuracy,
            'class_total': dict(class_total),
            'confusion_matrix': {k: dict(v) for k, v in confusion.items()},
            'confidences': confidences
        }

    def print_results(self, results: Dict):
        """Print test results"""
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)

        print(f"\nOverall:")
        print(f"  Total samples: {results['total']}")
        print(f"  Correct: {results['correct']}")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Avg confidence: {results['avg_confidence']:.4f}")

        # Per-class results
        print(f"\nPer-class accuracy:")
        for label, acc in sorted(results['class_accuracy'].items()):
            count = results['class_total'][label]
            print(f"  {label:<20s}: {acc:.4f} ({count} samples)")

        # Confidence statistics
        confs = results['confidences']
        if confs:
            print(f"\nConfidence statistics:")
            print(f"  Min: {min(confs):.4f}")
            print(f"  Max: {max(confs):.4f}")
            print(f"  Avg: {sum(confs) / len(confs):.4f}")

        # Confusion matrix (if not too large)
        if len(results['confusion_matrix']) <= 10:
            print(f"\nConfusion Matrix:")
            print("(rows=true, cols=predicted)")

            labels = sorted(results['confusion_matrix'].keys())
            header = "True\\Pred".ljust(15)
            for label in labels:
                header += f"{label[:8]:>10s}"
            print(header)
            print("-" * len(header))

            for true_label in labels:
                row = f"{true_label[:13]:13s}  "
                for pred_label in labels:
                    count = results['confusion_matrix'][true_label].get(pred_label, 0)
                    row += f"{count:>10d}"
                print(row)

    def save_results(self, results: Dict, output_file: str):
        """Save results to JSON"""
        # Remove confidences list (too large)
        results_to_save = {k: v for k, v in results.items() if k != 'confidences'}

        # Add confidence summary
        confs = results['confidences']
        if confs:
            results_to_save['confidence_stats'] = {
                'min': min(confs),
                'max': max(confs),
                'mean': sum(confs) / len(confs)
            }

        with open(output_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)

        print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Test trained NIMCP model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test model on test set
  python test_trained_model.py \\
    --model ../models/run1/foundation_model_final.bin \\
    --dataset ../datasets/processed/imdb_sentiment

  # Test on validation set
  python test_trained_model.py \\
    --model ../models/run1/foundation_model_final.bin \\
    --dataset ../datasets/processed/imdb_sentiment \\
    --split val

  # Save results to file
  python test_trained_model.py \\
    --model ../models/run1/foundation_model_final.bin \\
    --dataset ../datasets/processed/imdb_sentiment \\
    --output test_results.json
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.bin file)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to processed dataset directory'
    )

    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split to test on (default: test)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file for results (JSON)'
    )

    args = parser.parse_args()

    # Create tester
    tester = ModelTester(args.model)

    # Load test data
    samples = tester.load_test_data(args.dataset, args.split)

    # Run test
    results = tester.test(samples)

    # Print results
    tester.print_results(results)

    # Save if requested
    if args.output:
        tester.save_results(results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
