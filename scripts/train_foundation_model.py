#!/usr/bin/env python3
"""
NIMCP Foundation Model Training
Progressive multi-domain training with curriculum learning and consolidation.

Features:
- Multi-domain sequential training
- Curriculum learning (easy to hard)
- Consolidation between domains
- Adaptive learning rate
- Early stopping
- Checkpoint saving
- Comprehensive logging and metrics

Training Strategy:
1. Load multiple preprocessed datasets (different domains)
2. Train on each domain sequentially
3. Consolidation phase between domains (replay previous domains)
4. Adaptive learning based on performance
5. Save checkpoints regularly
6. Log all metrics for analysis

Refactored: 2025-01-16 to use current Python bindings API v2.7.0
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random


# Add NIMCP to path - check multiple possible locations
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent / "build/lib/python",  # From scripts/
    script_dir / "build/lib/python",          # From repo root
    Path("/home/bbrelin/nimcp/build/lib/python"),  # Absolute path
]

for path in possible_paths:
    if path.exists():
        sys.path.insert(0, str(path))
        break

# Try to import nimcp
try:
    import nimcp
    print(f"NIMCP Python bindings loaded (version: {nimcp.version()})")
except ImportError:
    print("Error: nimcp module not found. Please build and install the NIMCP Python module.")
    print("Run: cd build && make nimcp_python && export PYTHONPATH=$PWD/lib/python:$PYTHONPATH")
    sys.exit(1)


def get_brain_stats(brain) -> Dict[str, Any]:
    """
    Get brain statistics using available Python bindings API.

    Replaces brain.probe() which doesn't exist in v2.7.0 bindings.
    Uses: get_neuron_count(), get_utilization_metrics()
    """
    try:
        neuron_count = brain.get_neuron_count()
        utilization, saturation = brain.get_utilization_metrics()
        return {
            'num_neurons': neuron_count,
            'num_synapses': 0,  # Not available in current bindings
            'utilization': utilization,
            'saturation': saturation,
            'total_learning_steps': 0,  # Not tracked in bindings
            'memory_bytes': 0,  # Not available in current bindings
        }
    except Exception as e:
        return {
            'num_neurons': 0,
            'num_synapses': 0,
            'utilization': 0.0,
            'saturation': 0.0,
            'total_learning_steps': 0,
            'memory_bytes': 0,
            'error': str(e)
        }


class TrainingMetrics:
    """Track and compute training metrics"""

    def __init__(self):
        self.history = defaultdict(list)

    def add(self, metric: str, value: float, epoch: int = None):
        """Add a metric value"""
        self.history[metric].append({
            'value': value,
            'epoch': epoch,
            'timestamp': time.time()
        })

    def get_latest(self, metric: str) -> Optional[float]:
        """Get latest value for a metric"""
        if metric in self.history and self.history[metric]:
            return self.history[metric][-1]['value']
        return None

    def get_average(self, metric: str, last_n: int = None) -> Optional[float]:
        """Get average of metric over last N entries"""
        if metric not in self.history:
            return None

        values = [entry['value'] for entry in self.history[metric]]
        if last_n:
            values = values[-last_n:]

        return sum(values) / len(values) if values else None

    def save(self, filepath: Path):
        """Save metrics to JSON"""
        with open(filepath, 'w') as f:
            json.dump(dict(self.history), f, indent=2)


class ProgressiveTrainer:
    """Progressive multi-domain trainer"""

    def __init__(self, config: Dict):
        self.config = config
        self.brain = None
        self.metrics = TrainingMetrics()

        # Training parameters
        self.initial_lr = config.get('learning_rate', 0.01)
        self.current_lr = self.initial_lr
        self.lr_decay = config.get('lr_decay', 0.95)
        self.min_lr = config.get('min_lr', 0.0001)

        self.batch_size = config.get('batch_size', 32)
        self.epochs_per_domain = config.get('epochs_per_domain', 5)
        self.consolidation_epochs = config.get('consolidation_epochs', 2)
        self.early_stop_patience = config.get('early_stop_patience', 3)

        # Paths
        self.datasets_dir = Path(config.get('datasets_dir', '../datasets/processed'))
        self.output_dir = Path(config.get('output_dir', '../models'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint settings
        self.checkpoint_interval = config.get('checkpoint_interval', 1)  # epochs

        # Domain rotation
        self.domain_order = config.get('domain_order', [])
        self.current_domain_idx = 0

        # Domain replay buffer for consolidation
        self.domain_samples = {}  # domain -> samples

    def create_brain(self) -> nimcp.Brain:
        """Create NIMCP brain for training"""
        print("\nCreating brain...")

        brain_config = self.config.get('brain', {})

        name = brain_config.get('name', 'foundation_model')
        size = brain_config.get('size', nimcp.BRAIN_MEDIUM)
        task = brain_config.get('task', nimcp.TASK_CLASSIFICATION)
        num_inputs = brain_config.get('num_inputs', 512)
        num_outputs = brain_config.get('num_outputs', 100)

        # Python bindings v2.7.0 API: Brain(name, size, task, num_inputs, num_outputs)
        brain = nimcp.Brain(name, size, task, num_inputs, num_outputs)

        print(f"  Name: {name}")
        print(f"  Size: {size} (BRAIN_TINY=0, BRAIN_SMALL=1, BRAIN_MEDIUM=2, BRAIN_LARGE=3)")
        print(f"  Task: {task} (TASK_CLASSIFICATION=0)")
        print(f"  Inputs: {num_inputs}")
        print(f"  Outputs: {num_outputs}")

        return brain

    def load_dataset(self, dataset_name: str, split: str = 'train') -> List[Dict]:
        """Load preprocessed dataset"""
        dataset_dir = self.datasets_dir / dataset_name
        filepath = dataset_dir / f'{split}.jsonl'

        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")

        samples = []
        with open(filepath, 'r') as f:
            for line in f:
                samples.append(json.loads(line))

        return samples

    def train_epoch(self, samples: List[Dict], epoch: int,
                   domain: str = None) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with metrics
        """
        random.shuffle(samples)

        total_loss = 0.0
        correct = 0
        total = 0

        # Process in batches
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i:i + self.batch_size]

            for sample in batch:
                features = sample['features']
                label = sample['label']

                # Confidence based on learning rate (higher LR = more confident updates)
                confidence = min(1.0, self.current_lr * 10)

                # Train
                self.brain.learn(features, label, confidence=confidence)

                # Validate prediction
                pred_label, pred_conf = self.brain.predict(features)

                total += 1
                if pred_label == label:
                    correct += 1

        # Calculate metrics
        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            'accuracy': accuracy,
            'learning_rate': self.current_lr,
            'samples': total
        }

        return metrics

    def validate(self, samples: List[Dict]) -> Dict[str, float]:
        """
        Validate on a dataset.

        Returns:
            Dict with validation metrics
        """
        correct = 0
        total = 0
        total_confidence = 0.0

        for sample in samples:
            features = sample['features']
            label = sample['label']

            pred_label, pred_conf = self.brain.predict(features)

            total += 1
            total_confidence += pred_conf

            if pred_label == label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        avg_confidence = total_confidence / total if total > 0 else 0.0

        return {
            'val_accuracy': accuracy,
            'val_confidence': avg_confidence,
            'val_samples': total
        }

    def train_domain(self, domain_name: str, domain_config: Dict) -> Dict:
        """
        Train on a single domain.

        Returns:
            Training results
        """
        print(f"\n{'='*60}")
        print(f"Training Domain: {domain_name}")
        print(f"{'='*60}")

        # Load datasets
        print("Loading data...")
        train_samples = self.load_dataset(domain_name, 'train')
        val_samples = self.load_dataset(domain_name, 'val')

        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")

        # Store samples for later consolidation
        self.domain_samples[domain_name] = {
            'train': train_samples[:1000],  # Keep subset for replay
            'val': val_samples
        }

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(self.epochs_per_domain):
            epoch_start = time.time()

            print(f"\nEpoch {epoch + 1}/{self.epochs_per_domain}")

            # Train
            train_metrics = self.train_epoch(train_samples, epoch, domain_name)

            # Validate
            val_metrics = self.validate(val_samples)

            # Update learning rate
            if epoch > 0:
                self.current_lr = max(self.min_lr, self.current_lr * self.lr_decay)

            epoch_time = time.time() - epoch_start

            # Log metrics
            print(f"  Train Acc: {train_metrics['accuracy']:.4f}")
            print(f"  Val Acc: {val_metrics['val_accuracy']:.4f}")
            print(f"  Val Conf: {val_metrics['val_confidence']:.4f}")
            print(f"  LR: {self.current_lr:.6f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Save metrics
            self.metrics.add('train_accuracy', train_metrics['accuracy'], epoch)
            self.metrics.add('val_accuracy', val_metrics['val_accuracy'], epoch)
            self.metrics.add('val_confidence', val_metrics['val_confidence'], epoch)
            self.metrics.add('learning_rate', self.current_lr, epoch)

            # Early stopping
            if val_metrics['val_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['val_accuracy']
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                print(f"\nEarly stopping triggered (patience={self.early_stop_patience})")
                break

            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(domain_name, epoch)

        return {
            'domain': domain_name,
            'best_val_accuracy': best_val_acc,
            'final_lr': self.current_lr
        }

    def consolidation_phase(self, epoch: int):
        """
        Consolidation phase: replay samples from all previous domains.
        Prevents catastrophic forgetting.
        """
        if not self.domain_samples:
            return

        print(f"\n{'='*60}")
        print(f"Consolidation Phase")
        print(f"{'='*60}")

        # Collect samples from all domains
        all_samples = []
        for domain, samples_dict in self.domain_samples.items():
            all_samples.extend(samples_dict['train'])

        print(f"Consolidation samples: {len(all_samples)}")

        # Train on mixed samples
        for cons_epoch in range(self.consolidation_epochs):
            print(f"\nConsolidation Epoch {cons_epoch + 1}/{self.consolidation_epochs}")

            metrics = self.train_epoch(all_samples, epoch, domain='consolidation')

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  LR: {self.current_lr:.6f}")

            self.metrics.add('consolidation_accuracy', metrics['accuracy'], epoch)

    def save_checkpoint(self, domain: str, epoch: int):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_name = f"checkpoint_{domain}_epoch{epoch}_{timestamp}.bin"
        checkpoint_path = self.output_dir / checkpoint_name

        print(f"  Saving checkpoint: {checkpoint_name}")
        self.brain.save(str(checkpoint_path))

    def save_final_model(self):
        """Save final trained model"""
        model_path = self.output_dir / "foundation_model_final.bin"
        print(f"\nSaving final model: {model_path}")
        self.brain.save(str(model_path))

        # Save brain statistics
        stats = get_brain_stats(self.brain)
        stats_path = self.output_dir / "foundation_model_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("NIMCP Foundation Model Training")
        print("="*60)

        # Create brain
        self.brain = self.create_brain()

        # Get domains to train on
        domains = self.config.get('domains', [])

        if not domains:
            print("Error: No domains specified in config")
            return False

        print(f"\nTraining on {len(domains)} domains:")
        for i, domain in enumerate(domains):
            print(f"  {i+1}. {domain['name']}")

        # Train on each domain
        domain_results = []

        for i, domain_config in enumerate(domains):
            domain_name = domain_config['name']

            # Train domain
            result = self.train_domain(domain_name, domain_config)
            domain_results.append(result)

            # Consolidation between domains (except after last domain)
            if i < len(domains) - 1:
                self.consolidation_phase(i)

            # Save checkpoint after each domain
            self.save_checkpoint(domain_name, self.epochs_per_domain)

        # Final consolidation
        print("\nFinal consolidation phase...")
        self.consolidation_phase(len(domains))

        # Save final model
        self.save_final_model()

        # Save metrics
        metrics_path = self.output_dir / "training_metrics.json"
        self.metrics.save(metrics_path)
        print(f"Metrics saved: {metrics_path}")

        # Summary
        print(f"\n{'='*60}")
        print("TRAINING SUMMARY")
        print(f"{'='*60}")

        for result in domain_results:
            print(f"  {result['domain']}: {result['best_val_accuracy']:.4f} accuracy")

        # Overall statistics
        final_stats = get_brain_stats(self.brain)
        print(f"\nFinal Brain Statistics:")
        print(f"  Neurons: {final_stats['num_neurons']:,}")
        print(f"  Utilization: {final_stats['utilization']:.2%}")
        print(f"  Saturation: {final_stats['saturation']:.2%}")
        if 'error' in final_stats:
            print(f"  Warning: {final_stats['error']}")

        return True


def load_config(config_file: str) -> Dict:
    """Load training configuration from JSON"""
    with open(config_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train NIMCP foundation model on multiple domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train from config file
  python train_foundation_model.py --config training_config.json

  # Train with custom output directory
  python train_foundation_model.py --config training_config.json --output ../models/run1

  # Resume from checkpoint
  python train_foundation_model.py --config training_config.json --resume checkpoint.bin
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='JSON configuration file'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for models and logs (default: ../models)'
    )

    parser.add_argument(
        '--datasets-dir',
        type=str,
        help='Directory with processed datasets (default: ../datasets/processed)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line args
    if args.output:
        config['output_dir'] = args.output
    if args.datasets_dir:
        config['datasets_dir'] = args.datasets_dir

    # Print configuration
    print("Configuration:")
    print(json.dumps(config, indent=2))

    if args.dry_run:
        print("\nDry run mode - exiting")
        return 0

    # Create trainer
    trainer = ProgressiveTrainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        trainer.brain = nimcp.Brain.load(args.resume)
        print("Checkpoint loaded successfully")

    # Run training
    success = trainer.train()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
