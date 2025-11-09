#!/usr/bin/env python3
"""
Example: Using NIMCP Progressive Training Framework

This script demonstrates how to use the progressive training framework
with custom configurations and data.
"""

import os
import sys

# Add progressive training module to path
sys.path.insert(0, os.path.dirname(__file__))

from progressive_training import ProgressiveTrainer, StageConfig


def example_1_default_training():
    """
    Example 1: Train with default configuration

    Simplest usage - just create trainer and run.
    """
    print("\n" + "=" * 70)
    print("Example 1: Default Progressive Training")
    print("=" * 70)

    trainer = ProgressiveTrainer(
        name="default_example",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics"
    )

    # Train through all stages
    trainer.train_all_stages()

    print("\nCheckpoints saved to: ./example_checkpoints/")
    print("Metrics saved to: ./example_metrics/")


def example_2_custom_stages():
    """
    Example 2: Custom training stages

    Define your own stages with specific domains and parameters.
    """
    print("\n" + "=" * 70)
    print("Example 2: Custom Training Stages")
    print("=" * 70)

    # Create custom stages for a science-focused curriculum
    custom_stages = [
        StageConfig(
            name="foundation",
            description="Basic scientific concepts",
            age_range="Elementary",
            duration_samples=3000,
            learning_rate=0.01,
            consolidation_frequency=300,
            domains=["science", "mathematics"],
            complexity_level=0.2,
            success_threshold=0.65
        ),
        StageConfig(
            name="intermediate",
            description="Advanced scientific reasoning",
            age_range="Middle School",
            duration_samples=5000,
            learning_rate=0.005,
            consolidation_frequency=500,
            domains=["science", "mathematics", "technical"],
            complexity_level=0.5,
            success_threshold=0.75
        ),
        StageConfig(
            name="advanced",
            description="Specialized scientific knowledge",
            age_range="High School",
            duration_samples=8000,
            learning_rate=0.002,
            consolidation_frequency=800,
            domains=["science", "technical", "philosophy"],
            complexity_level=0.8,
            success_threshold=0.82
        )
    ]

    trainer = ProgressiveTrainer(
        name="science_focused",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics"
    )

    # Replace default stages with custom ones
    trainer.stages = custom_stages

    # Train through custom curriculum
    trainer.train_all_stages()


def example_3_single_stage():
    """
    Example 3: Train a single stage

    Useful for debugging or focused training.
    """
    print("\n" + "=" * 70)
    print("Example 3: Single Stage Training")
    print("=" * 70)

    trainer = ProgressiveTrainer(
        name="single_stage_example",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics"
    )

    # Train just the first stage (infant)
    stage_idx = 0
    progress = trainer.train_stage(stage_idx)

    print(f"\nStage completed!")
    print(f"  Samples trained: {progress.samples_trained:,}")
    print(f"  Final accuracy: {progress.current_accuracy:.2%}")
    print(f"  Duration: {progress.end_time - progress.start_time:.1f}s")


def example_4_save_and_load_config():
    """
    Example 4: Save and load custom configuration

    Create a reusable configuration file.
    """
    print("\n" + "=" * 70)
    print("Example 4: Save and Load Configuration")
    print("=" * 70)

    # Create trainer with custom stages
    trainer = ProgressiveTrainer(
        name="config_example",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics"
    )

    # Modify default configuration
    trainer.stages[0].learning_rate = 0.02  # Increase infant LR
    trainer.stages[1].duration_samples = 8000  # More child samples

    # Save configuration
    config_path = "./example_config.json"
    trainer.save_config(config_path)
    print(f"Configuration saved to: {config_path}")

    # Create new trainer from saved config
    trainer2 = ProgressiveTrainer(
        name="loaded_example",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics",
        config_path=config_path
    )

    print(f"\nLoaded configuration with {len(trainer2.stages)} stages:")
    for i, stage in enumerate(trainer2.stages):
        print(f"  Stage {i+1}: {stage.name} ({stage.duration_samples:,} samples, LR={stage.learning_rate})")


def example_5_analyze_metrics():
    """
    Example 5: Analyze training metrics

    Load and analyze metrics from a completed training run.
    """
    print("\n" + "=" * 70)
    print("Example 5: Analyze Training Metrics")
    print("=" * 70)

    import json
    from pathlib import Path

    # First, run a quick training
    print("\nRunning training to generate metrics...")
    trainer = ProgressiveTrainer(
        name="metrics_example",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics"
    )

    # Train just first stage for quick demo
    trainer.train_stage(0)

    # Load and analyze metrics
    metrics_file = Path("./example_metrics/metrics_example_metrics.jsonl")
    if metrics_file.exists():
        print(f"\nAnalyzing metrics from: {metrics_file}")

        metrics = []
        with open(metrics_file, 'r') as f:
            for line in f:
                metrics.append(json.loads(line))

        print(f"Total metric records: {len(metrics)}")

        if metrics:
            # Calculate statistics
            accuracies = [m['accuracy'] for m in metrics]
            losses = [m['loss'] for m in metrics]

            print(f"\nAccuracy Statistics:")
            print(f"  Min: {min(accuracies):.2%}")
            print(f"  Max: {max(accuracies):.2%}")
            print(f"  Final: {accuracies[-1]:.2%}")

            print(f"\nLoss Statistics:")
            print(f"  Initial: {losses[0]:.4f}")
            print(f"  Final: {losses[-1]:.4f}")
            print(f"  Improvement: {(losses[0] - losses[-1]) / losses[0]:.1%}")

        # Load summary
        summary_file = Path("./example_metrics/metrics_example_summary.json")
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            print(f"\nTraining Summary:")
            print(f"  Total duration: {summary['total_duration']:.1f}s")
            print(f"  Total samples: {summary['total_samples']:,}")
            print(f"  Stages completed: {len(summary['stages'])}")


def example_6_progressive_complexity():
    """
    Example 6: Demonstrate progressive complexity increase

    Show how complexity affects training difficulty.
    """
    print("\n" + "=" * 70)
    print("Example 6: Progressive Complexity")
    print("=" * 70)

    stages = []
    for i in range(5):
        complexity = 0.1 + i * 0.2  # 0.1, 0.3, 0.5, 0.7, 0.9

        stage = StageConfig(
            name=f"level_{i+1}",
            description=f"Complexity level {complexity:.1f}",
            age_range=f"Level {i+1}",
            duration_samples=2000,  # Short for demo
            learning_rate=0.01 - i * 0.001,  # Decreasing LR
            consolidation_frequency=200,
            domains=["general"],
            complexity_level=complexity,
            success_threshold=0.60 + i * 0.05  # Increasing threshold
        )
        stages.append(stage)

    trainer = ProgressiveTrainer(
        name="complexity_demo",
        checkpoint_dir="./example_checkpoints",
        metrics_dir="./example_metrics"
    )

    trainer.stages = stages

    print(f"\nTraining through {len(stages)} complexity levels:")
    for stage in stages:
        print(f"  {stage.name}: complexity={stage.complexity_level:.1f}, "
              f"threshold={stage.success_threshold:.1%}")

    # Train through all levels
    trainer.train_all_stages()


def main():
    """Run all examples"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Progressive Training Examples",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--example', type=int, default=0,
                       help='Which example to run (1-6), or 0 for all')

    args = parser.parse_args()

    examples = {
        1: ("Default Training", example_1_default_training),
        2: ("Custom Stages", example_2_custom_stages),
        3: ("Single Stage", example_3_single_stage),
        4: ("Save/Load Config", example_4_save_and_load_config),
        5: ("Analyze Metrics", example_5_analyze_metrics),
        6: ("Progressive Complexity", example_6_progressive_complexity)
    }

    if args.example == 0:
        # Run all examples
        print("\nRunning all examples...\n")
        for num, (name, func) in examples.items():
            print(f"\n{'='*70}")
            print(f"Running Example {num}: {name}")
            print(f"{'='*70}")
            try:
                func()
            except Exception as e:
                print(f"Error in example {num}: {e}")
                import traceback
                traceback.print_exc()

    elif args.example in examples:
        # Run specific example
        name, func = examples[args.example]
        print(f"\nRunning Example {args.example}: {name}")
        func()

    else:
        print(f"Invalid example number: {args.example}")
        print(f"Valid options: 0 (all), or 1-{len(examples)}")


if __name__ == '__main__':
    main()
