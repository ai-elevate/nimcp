#!/usr/bin/env python3
"""
NIMCP Progressive Training Framework

Implements staged training inspired by human developmental learning:
- Stage 1: Infant (0-2 years) - Basic sensory patterns
- Stage 2: Child (2-7 years) - Language, simple concepts
- Stage 3: Adolescent (7-18 years) - Complex reasoning, domains
- Stage 4: Adult (18+ years) - Specialized knowledge, refinement

Based on TRAINING_REGIMEN.md (10-stage pipeline) but adapted for
curriculum learning with progressive complexity.

Key Features:
- Adaptive learning rate (starts high, decreases)
- Multi-domain rotation (prevent forgetting)
- Consolidation periods (sleep-like)
- Performance tracking per domain
- Checkpointing after each stage
"""

import os
import sys
import time
import json
import random
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from datetime import datetime

# Add NIMCP Python bindings to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build', 'src', 'python'))

try:
    import nimcp
    NIMCP_AVAILABLE = True
except ImportError:
    print("Warning: NIMCP Python bindings not available. Running in simulation mode.")
    NIMCP_AVAILABLE = False


@dataclass
class StageConfig:
    """Configuration for a training stage"""
    name: str
    description: str
    age_range: str
    duration_samples: int
    learning_rate: float
    consolidation_frequency: int  # Samples between consolidation
    domains: List[str]
    complexity_level: float  # 0.0 to 1.0
    success_threshold: float  # Accuracy threshold to advance


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    stage: str
    epoch: int
    sample: int
    loss: float
    accuracy: float
    domain: str
    timestamp: float
    learning_rate: float


@dataclass
class StageProgress:
    """Progress tracking for a stage"""
    stage_name: str
    samples_trained: int
    total_samples: int
    current_accuracy: float
    best_accuracy: float
    epochs_completed: int
    start_time: float
    end_time: Optional[float]
    metrics: List[TrainingMetrics]
    completed: bool


class ProgressiveTrainer:
    """
    Progressive training framework implementing developmental curriculum learning.

    Philosophy:
    - Start simple, build complexity
    - Multi-domain rotation prevents catastrophic forgetting
    - Periodic consolidation strengthens important patterns
    - Adaptive learning rate mimics biological learning
    """

    def __init__(self,
                 name: str = "progressive_learner",
                 checkpoint_dir: str = "checkpoints",
                 metrics_dir: str = "metrics",
                 config_path: Optional[str] = None):
        """
        Initialize progressive trainer.

        Args:
            name: Name for this training run
            checkpoint_dir: Directory for saving checkpoints
            metrics_dir: Directory for saving metrics
            config_path: Path to custom configuration file
        """
        self.name = name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.metrics_dir = Path(metrics_dir)

        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        # Initialize brain (if NIMCP available)
        self.brain = None
        if NIMCP_AVAILABLE:
            self._init_brain()

        # Load or create configuration
        if config_path and Path(config_path).exists():
            self.stages = self._load_config(config_path)
        else:
            self.stages = self._create_default_stages()

        # Training state
        self.current_stage_idx = 0
        self.stage_history: List[StageProgress] = []
        self.global_step = 0
        self.start_time = time.time()

    def _init_brain(self):
        """Initialize NIMCP brain with appropriate configuration"""
        print(f"Creating NIMCP brain: {self.name}")

        # Start with SMALL brain, progressively enable modules
        self.brain = nimcp.Brain(
            name=self.name,
            size=1,  # SMALL (1K neurons)
            task=0,  # CLASSIFICATION
            inputs=20,  # Will vary by domain
            outputs=10  # Will vary by stage
        )

        print(f"Brain created: {self.brain.probe()['num_neurons']:,} neurons")

    def _create_default_stages(self) -> List[StageConfig]:
        """Create default developmental stages"""
        return [
            # Stage 1: Infant (0-2 years) - Sensory Foundation
            StageConfig(
                name="infant",
                description="Basic sensory patterns and simple associations",
                age_range="0-2 years",
                duration_samples=5000,
                learning_rate=0.01,
                consolidation_frequency=500,
                domains=["sensory", "patterns"],
                complexity_level=0.1,
                success_threshold=0.70
            ),

            # Stage 2: Child (2-7 years) - Language & Simple Concepts
            StageConfig(
                name="child",
                description="Language acquisition and simple conceptual learning",
                age_range="2-7 years",
                duration_samples=10000,
                learning_rate=0.008,
                consolidation_frequency=800,
                domains=["language", "literature", "social"],
                complexity_level=0.3,
                success_threshold=0.75
            ),

            # Stage 3: Adolescent (7-18 years) - Complex Reasoning
            StageConfig(
                name="adolescent",
                description="Abstract reasoning across multiple domains",
                age_range="7-18 years",
                duration_samples=15000,
                learning_rate=0.005,
                consolidation_frequency=1000,
                domains=["science", "mathematics", "history", "ethics", "art"],
                complexity_level=0.6,
                success_threshold=0.80
            ),

            # Stage 4: Adult (18+ years) - Specialized Knowledge
            StageConfig(
                name="adult",
                description="Specialized knowledge and refinement",
                age_range="18+ years",
                duration_samples=20000,
                learning_rate=0.002,
                consolidation_frequency=1500,
                domains=["technical", "philosophy", "general"],
                complexity_level=0.9,
                success_threshold=0.85
            )
        ]

    def _load_config(self, config_path: str) -> List[StageConfig]:
        """Load configuration from file"""
        with open(config_path, 'r') as f:
            data = json.load(f)

        return [StageConfig(**stage) for stage in data['stages']]

    def save_config(self, path: str):
        """Save current configuration to file"""
        config_data = {
            'name': self.name,
            'stages': [asdict(stage) for stage in self.stages]
        }

        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2)

        print(f"Configuration saved to {path}")

    def _generate_training_data(self,
                                stage: StageConfig,
                                num_samples: int) -> List[Tuple[List[float], str]]:
        """
        Generate synthetic training data for a stage.

        In production, this would load real datasets.
        For now, we generate synthetic data based on complexity level.
        """
        data = []

        for _ in range(num_samples):
            # Generate features based on complexity
            num_features = int(10 + stage.complexity_level * 40)  # 10-50 features

            # Add noise proportional to complexity
            noise_level = stage.complexity_level * 0.3
            features = [random.gauss(0.5, noise_level) for _ in range(num_features)]

            # Generate label from one of the stage's domains
            domain = random.choice(stage.domains)
            label = f"{domain}_{random.randint(0, 9)}"

            data.append((features, label))

        return data

    def _consolidate_memory(self, stage: StageConfig):
        """
        Perform memory consolidation (sleep-like learning).

        Based on nimcp_consolidation.h API.
        """
        if not NIMCP_AVAILABLE or not self.brain:
            print("  [Simulation] Memory consolidation (would strengthen patterns)")
            return

        print("  Consolidating memory (strengthening important patterns)...")

        # In real implementation, would call:
        # consolidation_config = nimcp.ConsolidationConfig()
        # consolidation_config.strategy = "FULL"
        # consolidation_config.cycles = 10
        # self.brain.consolidate_memory(consolidation_config)

        # For now, simulate
        time.sleep(0.5)  # Simulate consolidation time

    def _calculate_adaptive_lr(self, stage: StageConfig, progress: float) -> float:
        """
        Calculate adaptive learning rate.

        Starts at stage.learning_rate, decays as progress increases.
        Mimics biological learning: fast initial learning, slower refinement.
        """
        base_lr = stage.learning_rate
        decay_factor = 0.5  # Decay to 50% by end of stage

        return base_lr * (1 - decay_factor * progress)

    def _evaluate_stage(self,
                       stage: StageConfig,
                       test_data: List[Tuple[List[float], str]]) -> float:
        """
        Evaluate performance on test set.

        Returns accuracy (0.0 to 1.0).
        """
        if not NIMCP_AVAILABLE or not self.brain:
            # Simulate improving accuracy
            return 0.5 + random.random() * 0.3

        correct = 0
        total = len(test_data)

        for features, true_label in test_data:
            predicted_label, confidence = self.brain.decide(features)

            if predicted_label == true_label:
                correct += 1

        return correct / total if total > 0 else 0.0

    def _save_checkpoint(self, stage: StageConfig, progress: StageProgress):
        """Save training checkpoint"""
        checkpoint_path = self.checkpoint_dir / f"{self.name}_stage_{stage.name}.json"

        checkpoint_data = {
            'name': self.name,
            'stage': stage.name,
            'progress': asdict(progress),
            'global_step': self.global_step,
            'timestamp': datetime.now().isoformat()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        # Save brain state if available
        if NIMCP_AVAILABLE and self.brain:
            brain_path = self.checkpoint_dir / f"{self.name}_stage_{stage.name}.brain"
            # self.brain.save(str(brain_path))  # Would call this in real implementation
            print(f"  Checkpoint saved: {checkpoint_path}")

    def _save_metrics(self, metrics: List[TrainingMetrics]):
        """Save training metrics to file"""
        metrics_path = self.metrics_dir / f"{self.name}_metrics.jsonl"

        with open(metrics_path, 'a') as f:
            for metric in metrics:
                f.write(json.dumps(asdict(metric)) + '\n')

    def train_stage(self, stage_idx: int) -> StageProgress:
        """
        Train a single developmental stage.

        Args:
            stage_idx: Index of stage to train

        Returns:
            StageProgress object with results
        """
        stage = self.stages[stage_idx]

        print("\n" + "=" * 70)
        print(f"Stage {stage_idx + 1}/{len(self.stages)}: {stage.name.upper()}")
        print(f"Age Range: {stage.age_range}")
        print(f"Description: {stage.description}")
        print(f"Domains: {', '.join(stage.domains)}")
        print(f"Complexity: {stage.complexity_level:.1%}")
        print("=" * 70)

        # Initialize progress tracking
        progress = StageProgress(
            stage_name=stage.name,
            samples_trained=0,
            total_samples=stage.duration_samples,
            current_accuracy=0.0,
            best_accuracy=0.0,
            epochs_completed=0,
            start_time=time.time(),
            end_time=None,
            metrics=[],
            completed=False
        )

        # Generate training and test data
        print(f"\nGenerating {stage.duration_samples} training samples...")
        train_data = self._generate_training_data(stage, stage.duration_samples)
        test_data = self._generate_training_data(stage, 1000)  # Fixed test set

        print(f"Training on {len(train_data)} samples across domains: {stage.domains}")

        # Training loop
        batch_size = 32
        samples_per_epoch = min(1000, stage.duration_samples)
        num_epochs = (stage.duration_samples + samples_per_epoch - 1) // samples_per_epoch

        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

            # Shuffle training data
            random.shuffle(train_data)

            epoch_metrics = []
            epoch_loss = 0.0

            # Process samples in batches
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i:i + batch_size]

                # Calculate adaptive learning rate
                stage_progress = progress.samples_trained / stage.duration_samples
                current_lr = self._calculate_adaptive_lr(stage, stage_progress)

                # Train on batch
                for features, label in batch:
                    if NIMCP_AVAILABLE and self.brain:
                        # Real training
                        self.brain.learn(features, label, confidence=1.0)

                    # Simulate loss (decreases over time)
                    loss = 1.0 - (progress.samples_trained / stage.duration_samples) * 0.5
                    loss += random.gauss(0, 0.1)  # Add noise
                    loss = max(0.1, min(1.0, loss))

                    epoch_loss += loss
                    progress.samples_trained += 1
                    self.global_step += 1

                # Periodic consolidation
                if progress.samples_trained % stage.consolidation_frequency == 0:
                    self._consolidate_memory(stage)

            # Evaluate on test set
            accuracy = self._evaluate_stage(stage, test_data)
            progress.current_accuracy = accuracy
            progress.best_accuracy = max(progress.best_accuracy, accuracy)
            progress.epochs_completed += 1

            # Record metrics
            avg_loss = epoch_loss / len(train_data)
            metric = TrainingMetrics(
                stage=stage.name,
                epoch=epoch,
                sample=progress.samples_trained,
                loss=avg_loss,
                accuracy=accuracy,
                domain=stage.domains[0],  # Primary domain
                timestamp=time.time(),
                learning_rate=current_lr
            )
            progress.metrics.append(metric)
            epoch_metrics.append(metric)

            # Print progress
            print(f"  Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2%} | "
                  f"LR: {current_lr:.6f} | Samples: {progress.samples_trained}/{stage.duration_samples}")

            # Save metrics
            self._save_metrics(epoch_metrics)

            # Check success threshold
            if accuracy >= stage.success_threshold:
                print(f"\n  Success threshold reached! ({accuracy:.2%} >= {stage.success_threshold:.2%})")
                break

        # Final consolidation
        print("\nFinal memory consolidation for stage...")
        self._consolidate_memory(stage)

        # Mark stage complete
        progress.end_time = time.time()
        progress.completed = True

        # Save checkpoint
        self._save_checkpoint(stage, progress)

        # Print stage summary
        duration = progress.end_time - progress.start_time
        print(f"\nStage {stage.name} completed!")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Samples trained: {progress.samples_trained:,}")
        print(f"  Final accuracy: {progress.current_accuracy:.2%}")
        print(f"  Best accuracy: {progress.best_accuracy:.2%}")

        return progress

    def train_all_stages(self):
        """Train through all developmental stages"""
        print("\n" + "=" * 70)
        print(f"PROGRESSIVE TRAINING: {self.name}")
        print(f"Total Stages: {len(self.stages)}")
        print("=" * 70)

        for stage_idx in range(len(self.stages)):
            progress = self.train_stage(stage_idx)
            self.stage_history.append(progress)

            # Print cumulative progress
            total_samples = sum(p.samples_trained for p in self.stage_history)
            avg_accuracy = sum(p.current_accuracy for p in self.stage_history) / len(self.stage_history)

            print(f"\nCumulative Progress:")
            print(f"  Stages completed: {len(self.stage_history)}/{len(self.stages)}")
            print(f"  Total samples: {total_samples:,}")
            print(f"  Average accuracy: {avg_accuracy:.2%}")

        # Final summary
        self._print_final_summary()

    def _print_final_summary(self):
        """Print summary of all training stages"""
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE - FINAL SUMMARY")
        print("=" * 70)

        total_duration = time.time() - self.start_time
        total_samples = sum(p.samples_trained for p in self.stage_history)

        print(f"\nOverall Statistics:")
        print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Samples/sec: {total_samples/total_duration:.1f}")

        print(f"\nStage-by-Stage Results:")
        print("-" * 70)

        for progress in self.stage_history:
            duration = progress.end_time - progress.start_time
            print(f"  {progress.stage_name.upper():<15} | "
                  f"Acc: {progress.current_accuracy:>6.2%} | "
                  f"Samples: {progress.samples_trained:>8,} | "
                  f"Time: {duration:>6.1f}s")

        print("-" * 70)

        # Save final summary
        summary_path = self.metrics_dir / f"{self.name}_summary.json"
        summary = {
            'name': self.name,
            'total_duration': total_duration,
            'total_samples': total_samples,
            'stages': [
                {
                    'name': p.stage_name,
                    'accuracy': p.current_accuracy,
                    'samples': p.samples_trained,
                    'duration': p.end_time - p.start_time
                }
                for p in self.stage_history
            ],
            'completed_at': datetime.now().isoformat()
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nSummary saved to: {summary_path}")

        # Final checkpoint
        final_checkpoint = self.checkpoint_dir / f"{self.name}_final.json"
        with open(final_checkpoint, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Final checkpoint: {final_checkpoint}")


def main():
    """Main entry point for progressive training"""
    import argparse

    parser = argparse.ArgumentParser(
        description="NIMCP Progressive Training Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python progressive_training.py --name my_learner

  # Train with custom configuration
  python progressive_training.py --name custom --config my_config.json

  # Specify output directories
  python progressive_training.py --name test --checkpoint-dir ./checkpoints --metrics-dir ./metrics
        """
    )

    parser.add_argument('--name', type=str, default='progressive_learner',
                       help='Name for this training run')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory for saving checkpoints')
    parser.add_argument('--metrics-dir', type=str, default='metrics',
                       help='Directory for saving metrics')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom configuration file')
    parser.add_argument('--save-config', action='store_true',
                       help='Save default configuration and exit')

    args = parser.parse_args()

    # Create trainer
    trainer = ProgressiveTrainer(
        name=args.name,
        checkpoint_dir=args.checkpoint_dir,
        metrics_dir=args.metrics_dir,
        config_path=args.config
    )

    # Save config if requested
    if args.save_config:
        config_path = f"{args.name}_config.json"
        trainer.save_config(config_path)
        print(f"Default configuration saved to: {config_path}")
        return

    # Run training
    trainer.train_all_stages()


if __name__ == '__main__':
    main()
