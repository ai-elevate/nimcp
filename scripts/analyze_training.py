#!/usr/bin/env python3
"""
Training Results Analyzer
Analyzes training metrics and generates reports.

Features:
- Load and analyze training metrics
- Calculate statistics (mean, std, min, max)
- Compare multiple training runs
- Generate text reports
- Export summary data
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict


class TrainingAnalyzer:
    """Analyzes training results"""

    def __init__(self, metrics_file: str):
        self.metrics_file = Path(metrics_file)
        self.metrics = self.load_metrics()
        self.model_dir = self.metrics_file.parent

    def load_metrics(self) -> Dict:
        """Load training metrics from JSON"""
        with open(self.metrics_file, 'r') as f:
            return json.load(f)

    def get_metric_stats(self, metric_name: str) -> Dict:
        """Calculate statistics for a metric"""
        if metric_name not in self.metrics:
            return None

        values = [entry['value'] for entry in self.metrics[metric_name]]

        if not values:
            return None

        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'final': values[-1] if values else None,
            'best': max(values) if 'accuracy' in metric_name else min(values)
        }

    def print_metric_summary(self, metric_name: str):
        """Print summary for a metric"""
        stats = self.get_metric_stats(metric_name)

        if not stats:
            print(f"  {metric_name}: No data")
            return

        print(f"\n{metric_name}:")
        print(f"  Count:   {stats['count']}")
        print(f"  Mean:    {stats['mean']:.6f}")
        print(f"  Min:     {stats['min']:.6f}")
        print(f"  Max:     {stats['max']:.6f}")
        print(f"  Final:   {stats['final']:.6f}")
        print(f"  Best:    {stats['best']:.6f}")

    def analyze_convergence(self, metric_name: str = 'val_accuracy',
                          window: int = 5) -> Dict:
        """
        Analyze convergence behavior.

        Returns:
            Dict with convergence info
        """
        if metric_name not in self.metrics:
            return None

        values = [entry['value'] for entry in self.metrics[metric_name]]

        if len(values) < window:
            return {'converged': False, 'reason': 'Insufficient data'}

        # Check if last N values are similar (low variance)
        last_values = values[-window:]
        mean_last = sum(last_values) / len(last_values)
        variance = sum((x - mean_last) ** 2 for x in last_values) / len(last_values)
        std_dev = variance ** 0.5

        # Consider converged if std dev is very small
        converged = std_dev < 0.01

        # Find best epoch
        best_idx = max(range(len(values)), key=lambda i: values[i])
        best_value = values[best_idx]

        # Check if improving
        recent_trend = values[-1] - values[-window] if len(values) >= window else 0

        return {
            'converged': converged,
            'std_dev_last_n': std_dev,
            'window': window,
            'best_epoch': best_idx,
            'best_value': best_value,
            'final_value': values[-1],
            'recent_trend': recent_trend,
            'improving': recent_trend > 0
        }

    def print_full_report(self):
        """Print comprehensive training report"""
        print("="*60)
        print("TRAINING ANALYSIS REPORT")
        print("="*60)
        print(f"Metrics file: {self.metrics_file}")
        print(f"Model directory: {self.model_dir}")

        # Load model stats if available
        stats_file = self.model_dir / "foundation_model_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                model_stats = json.load(f)

            print(f"\nModel Statistics:")
            print(f"  Task: {model_stats.get('task_name', 'N/A')}")
            print(f"  Neurons: {model_stats.get('num_neurons', 0):,}")
            print(f"  Synapses: {model_stats.get('num_synapses', 0):,}")
            print(f"  Learning steps: {model_stats.get('total_learning_steps', 0):,}")
            print(f"  Memory: {model_stats.get('memory_bytes', 0) / (1024*1024):.2f} MB")

        # Metric summaries
        print(f"\n{'-'*60}")
        print("METRIC SUMMARIES")
        print(f"{'-'*60}")

        metric_names = [
            'train_accuracy',
            'val_accuracy',
            'val_confidence',
            'learning_rate',
            'consolidation_accuracy'
        ]

        for metric_name in metric_names:
            if metric_name in self.metrics:
                self.print_metric_summary(metric_name)

        # Convergence analysis
        print(f"\n{'-'*60}")
        print("CONVERGENCE ANALYSIS")
        print(f"{'-'*60}")

        conv = self.analyze_convergence('val_accuracy')
        if conv:
            print(f"\nValidation Accuracy:")
            print(f"  Converged: {conv['converged']}")
            print(f"  Best value: {conv['best_value']:.6f} (epoch {conv['best_epoch']})")
            print(f"  Final value: {conv['final_value']:.6f}")
            print(f"  Recent trend: {'+' if conv['recent_trend'] > 0 else ''}{conv['recent_trend']:.6f}")
            print(f"  Stability (std dev): {conv['std_dev_last_n']:.6f}")

        # Domain-specific analysis
        print(f"\n{'-'*60}")
        print("TRAINING PHASES")
        print(f"{'-'*60}")

        total_epochs = len(self.metrics.get('train_accuracy', []))
        consolidation_epochs = len(self.metrics.get('consolidation_accuracy', []))

        print(f"\nTotal training epochs: {total_epochs}")
        print(f"Consolidation epochs: {consolidation_epochs}")

        # Recommendations
        print(f"\n{'-'*60}")
        print("RECOMMENDATIONS")
        print(f"{'-'*60}")

        if conv and not conv['converged']:
            print("\n- Model has not fully converged. Consider:")
            print("  * Increase epochs_per_domain")
            print("  * Adjust learning rate")

        if conv and conv['final_value'] < 0.7:
            print("\n- Low accuracy detected. Consider:")
            print("  * Increase brain size")
            print("  * Increase feature dimensions")
            print("  * Check data quality")
            print("  * Add more training data")

        if conv and conv['converged'] and conv['final_value'] > 0.85:
            print("\n- Good convergence and accuracy!")
            print("  * Model is ready for deployment")
            print("  * Consider fine-tuning on specific tasks")

    def export_summary(self, output_file: str = None):
        """Export summary statistics to JSON"""
        if output_file is None:
            output_file = self.model_dir / "training_summary.json"
        else:
            output_file = Path(output_file)

        summary = {
            'metrics_file': str(self.metrics_file),
            'model_dir': str(self.model_dir),
            'metric_stats': {}
        }

        # Add stats for each metric
        for metric_name in self.metrics.keys():
            stats = self.get_metric_stats(metric_name)
            if stats:
                summary['metric_stats'][metric_name] = stats

        # Add convergence analysis
        conv = self.analyze_convergence('val_accuracy')
        if conv:
            summary['convergence'] = conv

        # Save
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary exported to: {output_file}")


def compare_runs(run_dirs: List[str]):
    """Compare multiple training runs"""
    print("="*60)
    print("TRAINING RUNS COMPARISON")
    print("="*60)

    runs = []

    for run_dir in run_dirs:
        metrics_file = Path(run_dir) / "training_metrics.json"
        if not metrics_file.exists():
            print(f"Warning: Metrics not found in {run_dir}")
            continue

        analyzer = TrainingAnalyzer(str(metrics_file))
        runs.append({
            'dir': run_dir,
            'analyzer': analyzer
        })

    if not runs:
        print("No valid runs found")
        return

    # Compare key metrics
    print(f"\n{'Run':<30} {'Val Acc':<12} {'Best Epoch':<12} {'Final LR':<12}")
    print("-"*70)

    for run in runs:
        analyzer = run['analyzer']

        val_acc_stats = analyzer.get_metric_stats('val_accuracy')
        lr_stats = analyzer.get_metric_stats('learning_rate')
        conv = analyzer.analyze_convergence('val_accuracy')

        val_acc = val_acc_stats['best'] if val_acc_stats else 0.0
        best_epoch = conv['best_epoch'] if conv else -1
        final_lr = lr_stats['final'] if lr_stats else 0.0

        run_name = Path(run['dir']).name

        print(f"{run_name:<30} {val_acc:<12.6f} {best_epoch:<12} {final_lr:<12.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze NIMCP training results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single training run
  python analyze_training.py --metrics ../models/run1/training_metrics.json

  # Compare multiple runs
  python analyze_training.py --compare ../models/run1 ../models/run2

  # Export summary
  python analyze_training.py --metrics ../models/run1/training_metrics.json --export
        """
    )

    parser.add_argument(
        '--metrics',
        type=str,
        help='Path to training_metrics.json file'
    )

    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple training runs (provide run directories)'
    )

    parser.add_argument(
        '--export',
        action='store_true',
        help='Export summary to JSON'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output file for export (default: training_summary.json)'
    )

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare)
        return 0

    if not args.metrics:
        print("Error: Provide --metrics or --compare")
        parser.print_help()
        return 1

    # Analyze single run
    analyzer = TrainingAnalyzer(args.metrics)
    analyzer.print_full_report()

    if args.export:
        analyzer.export_summary(args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
