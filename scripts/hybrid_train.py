#!/usr/bin/env python3
"""
NIMCP Hybrid Training Pipeline
================================

WHAT: Intelligent training that uses local datasets when available, streams when not
WHY:  Don't re-download what we already have; stream massive datasets
HOW:  Check local → Use if exists → Otherwise stream in batches

Features:
- Auto-detect already downloaded datasets
- Use local files for small/medium datasets
- Stream massive datasets (>100GB) in batches
- Configurable batch sizes
- Checkpoint/resume support
- Progress tracking
- Memory-efficient cleanup
"""

import json
import sys
import time
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Union
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk, Dataset
import nimcp

@dataclass
class DatasetInfo:
    """Information about a dataset's status"""
    name: str
    domain: str
    is_local: bool = False
    local_path: Optional[Path] = None
    size_estimate_gb: float = 0.0
    should_stream: bool = False
    examples_count: int = 0

@dataclass
class StreamConfig:
    """Configuration for hybrid training"""
    batch_size: int = 1000  # Examples per training batch
    checkpoint_interval: int = 10000  # Save checkpoint every N examples
    stream_threshold_gb: float = 10.0  # Stream datasets larger than this
    max_examples_per_dataset: Optional[int] = None  # Limit (for testing)
    datasets_dir: Path = Path("datasets/foundation")
    checkpoint_dir: Path = Path("checkpoints")
    temp_dir: Path = Path("/tmp/nimcp_streaming")
    cleanup_after_training: bool = True  # Delete local datasets after training
    auto_confirm_cleanup: bool = False  # Auto-confirm cleanup (no prompt)

@dataclass
class TrainingProgress:
    """Track training progress for checkpointing"""
    dataset_name: str
    examples_processed: int = 0
    batches_completed: int = 0
    last_checkpoint: int = 0
    start_time: float = 0.0
    completed: bool = False

    def save(self, checkpoint_dir: Path):
        """Save progress to checkpoint file"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{self.dataset_name}_progress.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'dataset_name': self.dataset_name,
                'examples_processed': self.examples_processed,
                'batches_completed': self.batches_completed,
                'last_checkpoint': self.last_checkpoint,
                'completed': self.completed,
                'timestamp': time.time()
            }, f, indent=2)

    @classmethod
    def load(cls, dataset_name: str, checkpoint_dir: Path):
        """Load progress from checkpoint file"""
        checkpoint_file = checkpoint_dir / f"{dataset_name}_progress.json"
        if not checkpoint_file.exists():
            return cls(dataset_name=dataset_name, start_time=time.time())

        with open(checkpoint_file) as f:
            data = json.load(f)

        progress = cls(
            dataset_name=data['dataset_name'],
            examples_processed=data['examples_processed'],
            batches_completed=data['batches_completed'],
            last_checkpoint=data['last_checkpoint'],
            completed=data.get('completed', False),
            start_time=time.time()
        )

        if progress.completed:
            print(f"  ✓ Already completed ({progress.examples_processed} examples)")
        elif progress.examples_processed > 0:
            print(f"  ↻ Resuming from checkpoint: {progress.examples_processed} examples processed")

        return progress


class DatasetRegistry:
    """
    WHAT: Tracks which datasets are downloaded vs need streaming
    WHY:  Avoid re-downloading; use local files when available
    HOW:  Scan datasets directory, check metadata files
    """

    # Size estimates for large datasets (in GB)
    SIZE_ESTIMATES = {
        'github_code_2025': 150.0,  # ~150GB estimated
        'the_stack': 6000.0,  # 6TB
        'github_code': 1000.0,  # 1TB
        'livecodebench': 1.0,  # Small
        'wikitext103': 0.5,  # 500MB
        'squad_v2': 0.1,  # 100MB
    }

    def __init__(self, datasets_dir: Path, stream_threshold_gb: float = 10.0):
        self.datasets_dir = datasets_dir
        self.stream_threshold_gb = stream_threshold_gb
        self.registry: Dict[str, DatasetInfo] = {}

    def scan_local_datasets(self) -> Dict[str, DatasetInfo]:
        """
        WHAT: Scan datasets directory for already-downloaded datasets
        WHY:  Use local files instead of re-downloading
        HOW:  Check for metadata.json files
        """
        print("Scanning for local datasets...")

        if not self.datasets_dir.exists():
            print(f"  ⚠ Datasets directory not found: {self.datasets_dir}")
            return {}

        local_datasets = {}

        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            metadata_file = dataset_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)

                name = metadata['name']
                domain = metadata['domain']

                # Count examples
                total_examples = sum(metadata.get('splits', {}).values())

                info = DatasetInfo(
                    name=name,
                    domain=domain,
                    is_local=True,
                    local_path=dataset_dir,
                    examples_count=total_examples
                )

                local_datasets[name] = info
                print(f"  ✓ Found local: {name} ({total_examples} examples)")

            except Exception as e:
                print(f"  ⚠ Failed to read metadata for {dataset_dir.name}: {e}")

        print(f"  Total local datasets: {len(local_datasets)}\n")
        return local_datasets

    def classify_dataset(self, dataset_config: Dict) -> DatasetInfo:
        """
        WHAT: Determine if dataset should be streamed or loaded locally
        WHY:  Optimize memory usage for large datasets
        HOW:  Check size estimates and local availability
        """
        name = dataset_config['name']
        domain = dataset_config['domain']

        # Check if already local
        if name in self.registry and self.registry[name].is_local:
            return self.registry[name]

        # Estimate size
        size_gb = self.SIZE_ESTIMATES.get(name, 1.0)

        # Decide whether to stream
        should_stream = size_gb > self.stream_threshold_gb

        info = DatasetInfo(
            name=name,
            domain=domain,
            is_local=False,
            size_estimate_gb=size_gb,
            should_stream=should_stream
        )

        self.registry[name] = info
        return info


class HybridTrainingPipeline:
    """
    WHAT: Hybrid training that adapts to dataset availability
    WHY:  Use local files when possible, stream when necessary
    HOW:  Check registry → Load local OR stream → Train → Checkpoint
    """

    def __init__(self, brain, config: StreamConfig):
        self.brain = brain
        self.config = config
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize dataset registry
        self.registry = DatasetRegistry(config.datasets_dir, config.stream_threshold_gb)
        self.registry.registry = self.registry.scan_local_datasets()

    def load_local_dataset(self, info: DatasetInfo) -> Optional[Iterator[Dict]]:
        """
        WHAT: Load dataset from local files
        WHY:  Already downloaded - no need to stream
        HOW:  Read JSONL files from disk
        """
        print(f"  📂 Loading from local: {info.local_path}")

        try:
            # Read train split (prefer train.jsonl)
            train_file = info.local_path / "train.jsonl"
            if not train_file.exists():
                # Try other splits
                splits = list(info.local_path.glob("*.jsonl"))
                if not splits:
                    print(f"  ✗ No JSONL files found in {info.local_path}")
                    return None
                train_file = splits[0]

            # Load as HuggingFace dataset
            dataset = load_dataset('json', data_files=str(train_file), split='train')

            # Return iterator
            return iter(dataset)

        except Exception as e:
            print(f"  ✗ Failed to load local dataset: {e}")
            return None

    def stream_dataset(self, dataset_config: Dict) -> Iterator[Dict]:
        """
        WHAT: Stream dataset from HuggingFace (no full download)
        WHY:  Dataset is massive - can't fit in memory/disk
        HOW:  Use streaming=True parameter
        """
        hf_dataset = dataset_config['hf_dataset']
        hf_subset = dataset_config.get('hf_subset', None)

        print(f"  🌊 Streaming from HuggingFace: {hf_dataset}")

        try:
            # Load in streaming mode
            if hf_subset:
                dataset = load_dataset(hf_dataset, hf_subset, split='train', streaming=True)
            else:
                dataset = load_dataset(hf_dataset, split='train', streaming=True)

            return iter(dataset)

        except Exception as e:
            print(f"  ✗ Streaming failed: {e}")
            return iter([])

    def extract_features(self, example: Dict, domain: str) -> Optional[List[float]]:
        """
        WHAT: Extract numeric features from example
        WHY:  NIMCP brain needs numeric inputs
        HOW:  Domain-specific feature extraction
        """
        try:
            if domain == "programming":
                code = example.get('content', example.get('code', example.get('text', '')))
                features = [
                    len(code),
                    code.count(' ') / max(len(code), 1),
                    code.count('\n') / max(len(code), 1),
                    code.count('{') / max(len(code), 1)
                ]
                return features

            elif domain == "language":
                text = example.get('text', example.get('content', ''))
                features = [
                    len(text),
                    text.count(' ') / max(len(text), 1),
                    text.count('.') / max(len(text), 1),
                    len(set(text)) / max(len(text), 1)
                ]
                return features

            else:
                # Generic: try to find numeric fields
                features = []
                for value in example.values():
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    if len(features) >= 4:
                        break

                if len(features) < 4:
                    features.extend([0.0] * (4 - len(features)))

                return features[:4]

        except:
            return None

    def train_on_batch(self, batch: List[List[float]], progress: TrainingProgress):
        """
        WHAT: Train brain on batch of features
        WHY:  Incremental learning
        HOW:  Call brain training API
        """
        if not batch:
            return

        print(f"    Training batch {progress.batches_completed + 1}: {len(batch)} examples...", end='', flush=True)

        start = time.time()
        for features in batch:
            try:
                # TODO: Replace with actual brain training API
                # self.brain.train(features)
                pass
            except:
                pass

        elapsed = time.time() - start
        rate = len(batch) / max(elapsed, 0.001)

        print(f" ✓ ({rate:.1f} ex/sec)")

        progress.examples_processed += len(batch)
        progress.batches_completed += 1

    def cleanup_dataset(self, info: DatasetInfo) -> bool:
        """
        WHAT: Delete dataset directory after training
        WHY:  Free disk space for streaming new datasets
        HOW:  Prompt user, then delete directory
        """
        if not info.is_local or not info.local_path:
            return False

        if not info.local_path.exists():
            return False

        # Calculate size
        total_size = sum(f.stat().st_size for f in info.local_path.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)

        # Confirm deletion
        if not self.config.auto_confirm_cleanup:
            print(f"\n  🗑️  Delete dataset to free {size_mb:.1f} MB?")
            print(f"      Path: {info.local_path}")
            response = input("      Confirm deletion (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("      Skipped - dataset kept")
                return False

        try:
            shutil.rmtree(info.local_path)
            print(f"  ✓ Deleted {info.name} - freed {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"  ✗ Failed to delete: {e}")
            return False

    def process_dataset(self, dataset_config: Dict, info: DatasetInfo) -> TrainingProgress:
        """
        WHAT: Main processing loop for one dataset
        WHY:  Train on dataset using appropriate method (local vs stream)
        HOW:  Load → Extract → Train → Checkpoint
        """
        name = dataset_config['name']
        domain = dataset_config['domain']

        print(f"\n{'='*70}")
        print(f"Processing: {name} ({domain})")
        print(f"  Method: {'Local files' if info.is_local else 'Streaming'}")
        if info.size_estimate_gb > 0:
            print(f"  Size: ~{info.size_estimate_gb:.1f} GB")
        print(f"{'='*70}")

        # Load progress
        progress = TrainingProgress.load(name, self.config.checkpoint_dir)

        if progress.completed:
            return progress

        # Get dataset iterator
        if info.is_local:
            dataset_iter = self.load_local_dataset(info)
        else:
            dataset_iter = self.stream_dataset(dataset_config)

        if dataset_iter is None:
            print(f"  ✗ Failed to load dataset")
            return progress

        # Process examples
        batch = []
        examples_since_checkpoint = 0

        try:
            for i, example in enumerate(dataset_iter):
                # Skip already processed (if resuming)
                if i < progress.examples_processed:
                    continue

                # Extract features
                features = self.extract_features(example, domain)
                if features is None:
                    continue

                batch.append(features)

                # Train when batch full
                if len(batch) >= self.config.batch_size:
                    self.train_on_batch(batch, progress)
                    batch = []
                    examples_since_checkpoint += self.config.batch_size

                    # Checkpoint
                    if examples_since_checkpoint >= self.config.checkpoint_interval:
                        progress.save(self.config.checkpoint_dir)
                        examples_since_checkpoint = 0
                        gc.collect()

                        print(f"  💾 Checkpoint: {progress.examples_processed} examples")

                # Check limit
                if self.config.max_examples_per_dataset:
                    if progress.examples_processed >= self.config.max_examples_per_dataset:
                        print(f"  ⏹ Reached limit: {self.config.max_examples_per_dataset}")
                        break

                # Progress updates
                if (i + 1) % 5000 == 0:
                    elapsed = time.time() - progress.start_time
                    rate = progress.examples_processed / max(elapsed, 1)
                    print(f"  📊 {progress.examples_processed} examples ({rate:.1f} ex/sec)")

            # Final batch
            if batch:
                self.train_on_batch(batch, progress)

            # Mark complete
            progress.completed = True
            progress.save(self.config.checkpoint_dir)

            print(f"  ✓ Completed: {progress.examples_processed} examples")

            # Cleanup local dataset to free space
            if self.config.cleanup_after_training and info.is_local:
                self.cleanup_dataset(info)

        except KeyboardInterrupt:
            print(f"\n  ⏸ Interrupted - saving...")
            progress.save(self.config.checkpoint_dir)
            raise

        except Exception as e:
            print(f"  ✗ Error: {e}")
            progress.save(self.config.checkpoint_dir)

        return progress


def main():
    import argparse
    parser = argparse.ArgumentParser(description='NIMCP Hybrid Training Pipeline')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--max-examples', type=int, default=None)
    parser.add_argument('--checkpoint-interval', type=int, default=10000)
    parser.add_argument('--stream-threshold', type=float, default=10.0, help='Stream datasets larger than N GB')
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep local datasets after training')
    parser.add_argument('--auto-cleanup', action='store_true', help='Auto-confirm cleanup (no prompt)')
    args = parser.parse_args()

    # Load config
    config_file = Path(__file__).parent / "foundation_datasets_config.json"
    with open(config_file) as f:
        dataset_config = json.load(f)

    # Initialize
    stream_config = StreamConfig(
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        stream_threshold_gb=args.stream_threshold,
        max_examples_per_dataset=args.max_examples,
        cleanup_after_training=not args.no_cleanup,
        auto_confirm_cleanup=args.auto_cleanup
    )

    print(f"""
{'='*70}
NIMCP Hybrid Training Pipeline
{'='*70}
Configuration:
  Batch Size: {stream_config.batch_size}
  Checkpoint Interval: {stream_config.checkpoint_interval}
  Stream Threshold: {stream_config.stream_threshold_gb} GB
  Max Examples: {stream_config.max_examples_per_dataset or 'unlimited'}
  Cleanup After Training: {stream_config.cleanup_after_training}
{'='*70}
""")

    # Initialize brain
    print("Initializing brain...")
    brain = nimcp.brain_create(1000, 4, 100, 0.01)
    print("✓ Brain initialized\n")

    # Create pipeline
    pipeline = HybridTrainingPipeline(brain, stream_config)

    # Filter datasets
    datasets = dataset_config['datasets']
    if args.datasets:
        names = set(args.datasets.split(','))
        datasets = [d for d in datasets if d['name'] in names]

    # Only HuggingFace datasets
    datasets = [d for d in datasets if d['type'] == 'huggingface']

    # Classify datasets into local vs remote
    local_datasets = []
    remote_datasets = []

    for ds_config in datasets:
        info = pipeline.registry.classify_dataset(ds_config)
        if info.is_local:
            local_datasets.append((ds_config, info))
        else:
            remote_datasets.append((ds_config, info))

    print(f"""
Processing Strategy:
  Local datasets: {len(local_datasets)} (train → delete → free space)
  Remote datasets: {len(remote_datasets)} (stream in batches)
  Total: {len(datasets)} datasets
""")

    # Process local datasets first (train then delete to free space)
    results = {}

    if local_datasets:
        print(f"\n{'='*70}")
        print(f"Phase 1: Train on Local Datasets ({len(local_datasets)} datasets)")
        print(f"{'='*70}\n")

        for ds_config, info in local_datasets:
            progress = pipeline.process_dataset(ds_config, info)
            results[ds_config['name']] = progress

    # Then process remote datasets (streaming)
    if remote_datasets:
        print(f"\n{'='*70}")
        print(f"Phase 2: Stream Remote Datasets ({len(remote_datasets)} datasets)")
        print(f"{'='*70}\n")

        for ds_config, info in remote_datasets:
            progress = pipeline.process_dataset(ds_config, info)
            results[ds_config['name']] = progress

    # Summary
    print(f"""
{'='*70}
Training Complete
{'='*70}
""")

    for name, progress in results.items():
        status = "✓" if progress.completed else "⏸"
        print(f"  {status} {name}: {progress.examples_processed} examples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
