#!/usr/bin/env python3
"""
NIMCP Parallel Training Pipeline
==================================

WHAT: Parallel batch training on datasets with automatic cleanup
WHY:  Maximize throughput by training multiple batches simultaneously
HOW:  Multi-process training → Delete trained datasets → Download new ones

Features:
- Parallel batch processing using multiprocessing
- Auto-detect and train on existing datasets
- Delete datasets after training to free space
- Download new datasets in batches
- Thread-safe progress tracking
- Checkpoint/resume support
"""

import json
import sys
import time
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Iterator, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Manager, Lock
import threading

from datasets import load_dataset, load_from_disk, Dataset

# Try to import nimcp, but make it optional for now
try:
    import nimcp
    NIMCP_AVAILABLE = True
except ImportError:
    NIMCP_AVAILABLE = False
    print("Warning: nimcp module not available, using stub training")

@dataclass
class ParallelConfig:
    """Configuration for parallel training"""
    batch_size: int = 1000  # Examples per training batch
    num_workers: int = 4  # Number of parallel workers
    checkpoint_interval: int = 10000  # Save checkpoint every N examples
    stream_threshold_gb: float = 10.0  # Stream datasets larger than this
    max_examples_per_dataset: Optional[int] = None  # Limit (for testing)
    datasets_dir: Path = Path("datasets/foundation")
    checkpoint_dir: Path = Path("checkpoints")
    temp_dir: Path = Path("/tmp/nimcp_streaming")
    cleanup_after_training: bool = True
    auto_confirm_cleanup: bool = False
    download_batch_size: int = 3  # Download N datasets at a time

@dataclass
class TrainingProgress:
    """Track training progress"""
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

        return cls(
            dataset_name=data['dataset_name'],
            examples_processed=data['examples_processed'],
            batches_completed=data['batches_completed'],
            last_checkpoint=data['last_checkpoint'],
            completed=data.get('completed', False),
            start_time=time.time()
        )


def train_batch_worker(batch_data: List[tuple], worker_id: int) -> int:
    """
    WHAT: Worker function to train on a batch of examples
    WHY:  Enable parallel processing of batches
    HOW:  Each worker trains on a subset of data

    Returns: Number of examples successfully trained
    """
    # Create a brain instance for this worker
    if NIMCP_AVAILABLE:
        try:
            # Initialize NIMCP for this worker process
            nimcp.init()

            # Create brain with appropriate size
            brain = nimcp.Brain(
                size=nimcp.BRAIN_SMALL,
                task=nimcp.TASK_CLASSIFICATION
            )

            trained_count = 0

            for features, label in batch_data:
                try:
                    # Train brain on this example
                    # Note: features should be a list of floats, label should be an int
                    brain.train([float(f) for f in features], int(label))
                    trained_count += 1
                except Exception as e:
                    pass  # Skip failed examples

            # Cleanup
            nimcp.shutdown()

            return trained_count

        except Exception as e:
            print(f"Worker {worker_id} error: {e}")
            return 0
    else:
        # Stub training if NIMCP not available
        trained_count = 0
        for features, label in batch_data:
            try:
                trained_count += 1
                time.sleep(0.0001)  # Simulate training time
            except Exception as e:
                pass
        return trained_count


class ParallelTrainingPipeline:
    """
    WHAT: Parallel training pipeline with batch processing
    WHY:  Maximize throughput using multiple CPU cores
    HOW:  ProcessPoolExecutor for parallel batch training
    """

    def __init__(self, config: ParallelConfig):
        self.config = config
        self.config.datasets_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)

    def scan_local_datasets(self) -> List[Dict]:
        """
        WHAT: Scan datasets directory for existing datasets
        WHY:  Find what we already have before downloading
        HOW:  Check for metadata.json files
        """
        local_datasets = []

        if not self.config.datasets_dir.exists():
            return local_datasets

        for dataset_dir in self.config.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            metadata_file = dataset_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    # Calculate size
                    total_size = sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file())
                    size_gb = total_size / (1024**3)

                    local_datasets.append({
                        'name': dataset_dir.name,
                        'domain': metadata.get('domain', 'unknown'),
                        'path': dataset_dir,
                        'size_gb': size_gb,
                        'metadata': metadata
                    })

                    print(f"  Found: {dataset_dir.name} ({size_gb:.2f} GB)")

                except Exception as e:
                    print(f"  Warning: Failed to read {metadata_file}: {e}")

        return local_datasets

    def load_local_dataset_batches(self, dataset_path: Path, batch_size: int) -> Iterator[List[tuple]]:
        """
        WHAT: Load local dataset in batches
        WHY:  Memory-efficient batch processing
        HOW:  Read JSONL files in chunks
        """
        train_file = dataset_path / "train.jsonl"
        if not train_file.exists():
            return

        batch = []

        with open(train_file) as f:
            for line in f:
                try:
                    example = json.loads(line)

                    # Extract simple features
                    text = example.get('text', str(example))
                    features = [
                        len(text),
                        text.count(' ') / max(len(text), 1),
                        text.count('\n') / max(len(text), 1),
                        len(set(text)) / max(len(text), 1)
                    ]

                    label = 0  # Dummy label
                    batch.append((features, label))

                    if len(batch) >= batch_size:
                        yield batch
                        batch = []

                except Exception as e:
                    continue

        if batch:
            yield batch

    def train_on_dataset_parallel(self, dataset_info: Dict, progress: TrainingProgress) -> TrainingProgress:
        """
        WHAT: Train on dataset using parallel batch processing
        WHY:  Maximize CPU utilization
        HOW:  Split batches across worker processes
        """
        print(f"\n{'='*70}")
        print(f"Training: {dataset_info['name']} ({dataset_info['domain']})")
        print(f"Size: {dataset_info['size_gb']:.2f} GB")
        print(f"Workers: {self.config.num_workers}")
        print(f"{'='*70}")

        total_trained = 0
        batches_processed = 0

        try:
            # Create worker pool
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []

                # Load and submit batches
                for batch_idx, batch_data in enumerate(self.load_local_dataset_batches(
                    dataset_info['path'],
                    self.config.batch_size
                )):
                    # Submit batch to worker pool
                    future = executor.submit(train_batch_worker, batch_data, batch_idx % self.config.num_workers)
                    futures.append((future, len(batch_data)))

                    # Process completed batches
                    if len(futures) >= self.config.num_workers * 2:
                        for future, batch_len in list(futures):
                            if future.done():
                                try:
                                    trained_count = future.result()
                                    total_trained += trained_count
                                    batches_processed += 1
                                    futures.remove((future, batch_len))

                                    # Progress update
                                    if batches_processed % 10 == 0:
                                        elapsed = time.time() - progress.start_time
                                        rate = total_trained / max(elapsed, 1)
                                        print(f"  Progress: {total_trained} examples, {batches_processed} batches ({rate:.1f} ex/sec)")

                                    # Checkpoint
                                    if total_trained - progress.last_checkpoint >= self.config.checkpoint_interval:
                                        progress.examples_processed = total_trained
                                        progress.batches_completed = batches_processed
                                        progress.last_checkpoint = total_trained
                                        progress.save(self.config.checkpoint_dir)
                                        print(f"  💾 Checkpoint saved at {total_trained} examples")

                                except Exception as e:
                                    print(f"  Warning: Batch training error: {e}")

                    # Check limit
                    if self.config.max_examples_per_dataset and total_trained >= self.config.max_examples_per_dataset:
                        print(f"  ⏹ Reached limit of {self.config.max_examples_per_dataset} examples")
                        break

                # Process remaining batches
                for future, batch_len in futures:
                    try:
                        trained_count = future.result(timeout=60)
                        total_trained += trained_count
                        batches_processed += 1
                    except Exception as e:
                        print(f"  Warning: Final batch error: {e}")

            # Update progress
            progress.examples_processed = total_trained
            progress.batches_completed = batches_processed
            progress.completed = True
            progress.save(self.config.checkpoint_dir)

            elapsed = time.time() - progress.start_time
            rate = total_trained / max(elapsed, 1)
            print(f"  ✓ Completed: {total_trained} examples in {elapsed:.1f}s ({rate:.1f} ex/sec)")

        except KeyboardInterrupt:
            print(f"\n  ⏸ Interrupted - saving checkpoint...")
            progress.examples_processed = total_trained
            progress.batches_completed = batches_processed
            progress.save(self.config.checkpoint_dir)
            raise

        except Exception as e:
            print(f"  ✗ Error: {e}")
            progress.examples_processed = total_trained
            progress.batches_completed = batches_processed
            progress.save(self.config.checkpoint_dir)

        return progress

    def cleanup_dataset(self, dataset_info: Dict) -> bool:
        """
        WHAT: Delete dataset after training
        WHY:  Free disk space for new datasets
        HOW:  Prompt user, then delete directory
        """
        dataset_path = dataset_info['path']
        size_mb = dataset_info['size_gb'] * 1024

        if not dataset_path.exists():
            return False

        # Confirm deletion
        if not self.config.auto_confirm_cleanup:
            print(f"\n  🗑️  Delete dataset to free {size_mb:.1f} MB?")
            print(f"      Path: {dataset_path}")
            response = input("      Confirm deletion (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("      Skipped - dataset kept")
                return False

        try:
            shutil.rmtree(dataset_path)
            print(f"  ✓ Deleted {dataset_info['name']} - freed {size_mb:.1f} MB")
            return True
        except Exception as e:
            print(f"  ✗ Failed to delete: {e}")
            return False

    def stream_and_train_dataset(self, ds_config: Dict) -> TrainingProgress:
        """
        WHAT: Stream and train on dataset without downloading
        WHY:  Avoid downloading massive datasets (like github_code_2025)
        HOW:  Use streaming=True and process batches directly
        """
        name = ds_config['name']
        domain = ds_config['domain']
        hf_dataset = ds_config['hf_dataset']
        hf_subset = ds_config.get('hf_subset', None)

        print(f"\n{'='*70}")
        print(f"Streaming: {name} ({domain})")
        print(f"{'='*70}")

        progress = TrainingProgress.load(name, self.config.checkpoint_dir)

        if progress.completed:
            print(f"  ✓ Already completed")
            return progress

        try:
            # Load dataset in streaming mode (no download!)
            print(f"  🔄 Loading streaming dataset...")
            if hf_subset:
                dataset = load_dataset(hf_dataset, hf_subset, split='train', streaming=True)
            else:
                dataset = load_dataset(hf_dataset, split='train', streaming=True)

            print(f"  ✓ Streaming enabled - processing batches")

            batch = []
            total_trained = 0
            batches_processed = 0

            # Create worker pool
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = []

                for i, example in enumerate(dataset):
                    # Extract features
                    try:
                        text = example.get('content', example.get('code', example.get('text', str(example))))
                        features = [
                            len(text),
                            text.count(' ') / max(len(text), 1),
                            text.count('\n') / max(len(text), 1),
                            len(set(text)) / max(len(text), 1)
                        ]
                        label = 0
                        batch.append((features, label))
                    except Exception as e:
                        continue

                    # When batch is full, submit to workers
                    if len(batch) >= self.config.batch_size:
                        future = executor.submit(train_batch_worker, batch, batches_processed % self.config.num_workers)
                        futures.append((future, len(batch)))
                        batch = []
                        batches_processed += 1

                        # Process completed futures
                        if len(futures) >= self.config.num_workers * 2:
                            for future, batch_len in list(futures):
                                if future.done():
                                    try:
                                        trained_count = future.result()
                                        total_trained += trained_count
                                        futures.remove((future, batch_len))
                                    except Exception as e:
                                        print(f"  Warning: Batch error: {e}")

                        # Progress update
                        if batches_processed % 10 == 0:
                            elapsed = time.time() - progress.start_time
                            rate = total_trained / max(elapsed, 1)
                            print(f"  Progress: {total_trained} examples, {batches_processed} batches ({rate:.1f} ex/sec)")

                        # Checkpoint
                        if total_trained - progress.last_checkpoint >= self.config.checkpoint_interval:
                            progress.examples_processed = total_trained
                            progress.batches_completed = batches_processed
                            progress.last_checkpoint = total_trained
                            progress.save(self.config.checkpoint_dir)
                            print(f"  💾 Checkpoint saved at {total_trained} examples")

                    # Check limit
                    if self.config.max_examples_per_dataset and total_trained >= self.config.max_examples_per_dataset:
                        print(f"  ⏹ Reached limit of {self.config.max_examples_per_dataset} examples")
                        break

                # Process remaining batch
                if batch:
                    future = executor.submit(train_batch_worker, batch, batches_processed % self.config.num_workers)
                    futures.append((future, len(batch)))

                # Wait for all futures to complete
                for future, batch_len in futures:
                    try:
                        trained_count = future.result(timeout=60)
                        total_trained += trained_count
                    except Exception as e:
                        print(f"  Warning: Final batch error: {e}")

            # Update progress
            progress.examples_processed = total_trained
            progress.batches_completed = batches_processed
            progress.completed = True
            progress.save(self.config.checkpoint_dir)

            elapsed = time.time() - progress.start_time
            rate = total_trained / max(elapsed, 1)
            print(f"  ✓ Completed: {total_trained} examples in {elapsed:.1f}s ({rate:.1f} ex/sec)")

            return progress

        except KeyboardInterrupt:
            print(f"\n  ⏸ Interrupted - saving checkpoint...")
            progress.examples_processed = total_trained
            progress.batches_completed = batches_processed
            progress.save(self.config.checkpoint_dir)
            raise
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return progress

    def download_datasets_batch(self, dataset_configs: List[Dict]):
        """
        WHAT: Download small datasets or stream large ones
        WHY:  Avoid downloading massive datasets (>1GB)
        HOW:  Check size estimate, use streaming for large datasets
        """
        if not dataset_configs:
            return

        print(f"\n{'='*70}")
        print(f"Processing {len(dataset_configs)} datasets")
        print(f"{'='*70}\n")

        # Datasets known to be large (will stream instead of download)
        LARGE_DATASETS = {
            'github_code_2025', 'the_stack', 'github_code',
            'wikitext103'  # Also relatively large
        }

        for ds_config in dataset_configs:
            name = ds_config['name']

            # Use streaming for known large datasets
            if name in LARGE_DATASETS:
                print(f"  🔄 {name}: Large dataset - using streaming mode")
                self.stream_and_train_dataset(ds_config)
                # No cleanup needed - never downloaded
            else:
                # Download small datasets normally
                self._download_single(ds_config)

    def _download_single(self, ds_config):
        """Download a single small dataset"""
        name = ds_config['name']
        hf_dataset = ds_config['hf_dataset']
        hf_subset = ds_config.get('hf_subset', None)

        print(f"  📥 Downloading: {name}")

        try:
            if hf_subset:
                dataset = load_dataset(hf_dataset, hf_subset, split='train', streaming=False)
            else:
                dataset = load_dataset(hf_dataset, split='train', streaming=False)

            # Save to disk
            output_dir = self.config.datasets_dir / name
            output_dir.mkdir(parents=True, exist_ok=True)

            dataset.to_json(output_dir / "train.jsonl")

            # Save metadata
            with open(output_dir / "metadata.json", 'w') as f:
                json.dump({
                    'name': name,
                    'domain': ds_config['domain'],
                    'hf_dataset': hf_dataset,
                    'hf_subset': hf_subset,
                    'downloaded': time.time()
                }, f, indent=2)

            print(f"  ✓ Downloaded: {name}")
            return True

        except Exception as e:
            print(f"  ✗ Failed: {name} - {e}")
            return False


def main():
    """Main parallel training pipeline"""

    import argparse
    parser = argparse.ArgumentParser(description='NIMCP Parallel Training Pipeline')
    parser.add_argument('--batch-size', type=int, default=1000, help='Examples per training batch')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--max-examples', type=int, default=None, help='Max examples per dataset')
    parser.add_argument('--checkpoint-interval', type=int, default=10000, help='Checkpoint every N examples')
    parser.add_argument('--no-cleanup', action='store_true', help='Keep datasets after training')
    parser.add_argument('--auto-cleanup', action='store_true', help='Auto-confirm cleanup')
    parser.add_argument('--download-batch-size', type=int, default=3, help='Download N datasets at once')
    args = parser.parse_args()

    # Initialize config
    config = ParallelConfig(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        checkpoint_interval=args.checkpoint_interval,
        max_examples_per_dataset=args.max_examples,
        cleanup_after_training=not args.no_cleanup,
        auto_confirm_cleanup=args.auto_cleanup,
        download_batch_size=args.download_batch_size
    )

    print(f"""
{'='*70}
NIMCP Parallel Training Pipeline
{'='*70}
Configuration:
  Batch Size: {config.batch_size} examples
  Parallel Workers: {config.num_workers}
  Checkpoint Interval: {config.checkpoint_interval} examples
  Max Examples per Dataset: {config.max_examples_per_dataset or 'unlimited'}
  Cleanup After Training: {config.cleanup_after_training}
  Download Batch Size: {config.download_batch_size}
  NIMCP Available: {NIMCP_AVAILABLE}
{'='*70}
""")

    # Create pipeline
    pipeline = ParallelTrainingPipeline(config)

    # Phase 1: Scan and train on existing datasets
    print("\n=== Phase 1: Training on Existing Datasets ===\n")
    local_datasets = pipeline.scan_local_datasets()

    if local_datasets:
        print(f"\nFound {len(local_datasets)} local datasets")

        results = {}
        for dataset_info in local_datasets:
            name = dataset_info['name']

            # Load progress
            progress = TrainingProgress.load(name, config.checkpoint_dir)

            if progress.completed:
                print(f"\n{name}: Already trained, skipping")
            else:
                # Train
                progress = pipeline.train_on_dataset_parallel(dataset_info, progress)
                results[name] = progress

            # Cleanup after training
            if config.cleanup_after_training:
                pipeline.cleanup_dataset(dataset_info)
    else:
        print("No local datasets found")

    # Phase 2: Download and train on new datasets
    print("\n\n=== Phase 2: Downloading and Training on New Datasets ===\n")

    # Load dataset config
    config_file = Path(__file__).parent / "foundation_datasets_config.json"
    with open(config_file) as f:
        ds_config = json.load(f)

    # Get HuggingFace datasets not already processed
    processed_names = {ds['name'] for ds in local_datasets}
    remaining_datasets = [
        ds for ds in ds_config['datasets']
        if ds['type'] == 'huggingface' and ds['name'] not in processed_names
    ]

    print(f"Remaining datasets to download: {len(remaining_datasets)}")

    # Download and train in batches
    for i in range(0, len(remaining_datasets), config.download_batch_size):
        batch = remaining_datasets[i:i + config.download_batch_size]

        print(f"\n--- Batch {i//config.download_batch_size + 1} ---")

        # Download batch
        pipeline.download_datasets_batch(batch)

        # Scan newly downloaded
        new_local = pipeline.scan_local_datasets()

        # Train on newly downloaded
        for dataset_info in new_local:
            if dataset_info['name'] in [ds['name'] for ds in batch]:
                progress = TrainingProgress.load(dataset_info['name'], config.checkpoint_dir)

                if not progress.completed:
                    progress = pipeline.train_on_dataset_parallel(dataset_info, progress)

                # Cleanup
                if config.cleanup_after_training:
                    pipeline.cleanup_dataset(dataset_info)

    print(f"""
{'='*70}
Training Complete!
{'='*70}
All datasets processed
Checkpoints saved to: {config.checkpoint_dir}
{'='*70}
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
