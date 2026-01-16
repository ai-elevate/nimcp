#!/usr/bin/env python3
"""
NIMCP Training State Persistence
================================

WHAT: Persistent state tracking for AGI training sessions
WHY:  Resume training after interruptions (disconnects, crashes, restarts)
HOW:  JSON state files + SQLite for efficient querying

Features:
- Dataset download tracking (which datasets are cached)
- Training progress tracking (examples processed per dataset)
- Checkpoint management (brain state snapshots)
- Session recovery (resume from exact point of interruption)
- Metrics history (preserve all training metrics)

State Files:
- training_state.json: Current training session state
- download_cache.json: Dataset download status
- checkpoints/: Brain state snapshots
- metrics/: Training metrics history
"""

import json
import sqlite3
import hashlib
import time
import os
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import Enum
import threading
import atexit


class DownloadStatus(Enum):
    """Dataset download status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CACHED = "cached"


class TrainingStatus(Enum):
    """Training session status"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DatasetProgress:
    """Progress tracking for a single dataset"""
    dataset_name: str
    dataset_identifier: str
    source: str
    download_status: str = DownloadStatus.NOT_STARTED.value
    download_path: Optional[str] = None
    download_started_at: Optional[str] = None
    download_completed_at: Optional[str] = None
    download_error: Optional[str] = None

    # Training progress
    examples_seen: int = 0
    examples_total: int = 0  # 0 = unknown (streaming)
    batches_processed: int = 0
    last_batch_index: int = 0
    training_started_at: Optional[str] = None
    training_completed_at: Optional[str] = None

    # Metrics
    total_loss: float = 0.0
    total_accuracy: float = 0.0
    samples_for_metrics: int = 0


@dataclass
class CheckpointInfo:
    """Information about a saved checkpoint"""
    checkpoint_id: str
    filepath: str
    created_at: str
    stage: str
    dataset_name: str
    examples_processed: int
    total_loss: float
    brain_size: int
    is_valid: bool = True


@dataclass
class TrainingState:
    """Complete training session state"""
    # Session info
    session_id: str
    created_at: str
    last_updated_at: str
    status: str = TrainingStatus.NOT_STARTED.value

    # Current position
    current_stage: str = "infant"
    current_dataset_index: int = 0
    current_batch_index: int = 0

    # Overall progress
    total_examples_processed: int = 0
    total_datasets_completed: int = 0
    total_training_time_seconds: float = 0.0

    # Per-dataset progress
    dataset_progress: Dict[str, Dict] = field(default_factory=dict)

    # Checkpoints
    checkpoints: List[Dict] = field(default_factory=list)
    last_checkpoint_id: Optional[str] = None

    # Configuration
    config: Dict = field(default_factory=dict)

    # Error recovery
    last_error: Optional[str] = None
    error_count: int = 0
    recovery_attempts: int = 0


class TrainingStateManager:
    """
    Manages persistent training state with automatic saving.

    Usage:
        state_mgr = TrainingStateManager("./training_state")
        state_mgr.start_session(config)

        # During training:
        state_mgr.update_dataset_progress("GSM8K", examples=100, loss=0.5)
        state_mgr.save_checkpoint(brain, "epoch_1")

        # On restart:
        state_mgr = TrainingStateManager("./training_state")
        if state_mgr.can_resume():
            state, checkpoint = state_mgr.get_resume_point()
    """

    def __init__(self, state_dir: str = "./training_state"):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # State files
        self.state_file = self.state_dir / "training_state.json"
        self.download_cache_file = self.state_dir / "download_cache.json"
        self.checkpoints_dir = self.state_dir / "checkpoints"
        self.metrics_dir = self.state_dir / "metrics"

        self.checkpoints_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True)

        # Current state
        self.state: Optional[TrainingState] = None
        self.download_cache: Dict[str, Dict] = {}

        # Auto-save
        self._save_lock = threading.Lock()
        self._dirty = False
        self._last_save_time = 0
        self._auto_save_interval = 30  # seconds

        # Load existing state
        self._load_state()
        self._load_download_cache()

        # Register cleanup
        atexit.register(self._cleanup)

    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
        return f"session_{timestamp}_{random_suffix}"

    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:4]
        return f"ckpt_{timestamp}_{random_suffix}"

    def _load_state(self):
        """Load existing training state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                self.state = TrainingState(**data)
                print(f"Loaded training state: {self.state.session_id}")
                print(f"  Status: {self.state.status}")
                print(f"  Examples processed: {self.state.total_examples_processed:,}")
                print(f"  Datasets completed: {self.state.total_datasets_completed}")
            except Exception as e:
                print(f"Warning: Could not load training state: {e}")
                self.state = None

    def _load_download_cache(self):
        """Load download cache from disk"""
        if self.download_cache_file.exists():
            try:
                with open(self.download_cache_file, 'r') as f:
                    self.download_cache = json.load(f)
                cached_count = sum(1 for v in self.download_cache.values()
                                   if v.get('status') == DownloadStatus.CACHED.value)
                print(f"Loaded download cache: {cached_count} datasets cached")
            except Exception as e:
                print(f"Warning: Could not load download cache: {e}")
                self.download_cache = {}

    def _save_state(self, force: bool = False):
        """Save current state to disk"""
        if not self.state:
            return

        current_time = time.time()
        if not force and (current_time - self._last_save_time) < self._auto_save_interval:
            self._dirty = True
            return

        with self._save_lock:
            try:
                self.state.last_updated_at = datetime.now().isoformat()

                # Atomic write
                temp_file = self.state_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(asdict(self.state), f, indent=2)
                temp_file.rename(self.state_file)

                self._last_save_time = current_time
                self._dirty = False
            except Exception as e:
                print(f"Error saving state: {e}")

    def _save_download_cache(self):
        """Save download cache to disk"""
        with self._save_lock:
            try:
                temp_file = self.download_cache_file.with_suffix('.tmp')
                with open(temp_file, 'w') as f:
                    json.dump(self.download_cache, f, indent=2)
                temp_file.rename(self.download_cache_file)
            except Exception as e:
                print(f"Error saving download cache: {e}")

    def _cleanup(self):
        """Cleanup on exit - save any pending state"""
        if self._dirty:
            self._save_state(force=True)

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(self, config: Dict) -> str:
        """Start a new training session"""
        session_id = self._generate_session_id()
        now = datetime.now().isoformat()

        self.state = TrainingState(
            session_id=session_id,
            created_at=now,
            last_updated_at=now,
            status=TrainingStatus.IN_PROGRESS.value,
            config=config
        )

        self._save_state(force=True)
        print(f"Started new training session: {session_id}")
        return session_id

    def resume_session(self) -> bool:
        """Resume existing session if possible"""
        if not self.state:
            return False

        if self.state.status in [TrainingStatus.COMPLETED.value, TrainingStatus.FAILED.value]:
            print(f"Previous session {self.state.session_id} is {self.state.status}")
            return False

        self.state.status = TrainingStatus.IN_PROGRESS.value
        self.state.recovery_attempts += 1
        self._save_state(force=True)

        print(f"Resuming session: {self.state.session_id}")
        print(f"  Recovery attempt: {self.state.recovery_attempts}")
        return True

    def can_resume(self) -> bool:
        """Check if there's a resumable session"""
        if not self.state:
            return False
        return self.state.status in [
            TrainingStatus.IN_PROGRESS.value,
            TrainingStatus.PAUSED.value
        ]

    def pause_session(self):
        """Pause the current session"""
        if self.state:
            self.state.status = TrainingStatus.PAUSED.value
            self._save_state(force=True)
            print(f"Session paused: {self.state.session_id}")

    def complete_session(self):
        """Mark session as completed"""
        if self.state:
            self.state.status = TrainingStatus.COMPLETED.value
            self._save_state(force=True)
            print(f"Session completed: {self.state.session_id}")

    def fail_session(self, error: str):
        """Mark session as failed"""
        if self.state:
            self.state.status = TrainingStatus.FAILED.value
            self.state.last_error = error
            self.state.error_count += 1
            self._save_state(force=True)
            print(f"Session failed: {error}")

    # =========================================================================
    # Progress Tracking
    # =========================================================================

    def update_position(self, stage: str, dataset_index: int, batch_index: int):
        """Update current training position"""
        if self.state:
            self.state.current_stage = stage
            self.state.current_dataset_index = dataset_index
            self.state.current_batch_index = batch_index
            self._save_state()

    def get_resume_point(self) -> tuple:
        """Get the point to resume training from"""
        if not self.state:
            return None, None

        return (
            self.state.current_stage,
            self.state.current_dataset_index,
            self.state.current_batch_index,
            self.state.last_checkpoint_id
        )

    def update_dataset_progress(
        self,
        dataset_name: str,
        examples: int = 0,
        batches: int = 0,
        loss: float = 0.0,
        accuracy: float = 0.0
    ):
        """Update progress for a specific dataset"""
        if not self.state:
            return

        if dataset_name not in self.state.dataset_progress:
            self.state.dataset_progress[dataset_name] = {
                'examples_seen': 0,
                'batches_processed': 0,
                'total_loss': 0.0,
                'total_accuracy': 0.0,
                'samples_for_metrics': 0
            }

        progress = self.state.dataset_progress[dataset_name]
        progress['examples_seen'] += examples
        progress['batches_processed'] += batches
        progress['total_loss'] += loss * examples
        progress['total_accuracy'] += accuracy * examples
        progress['samples_for_metrics'] += examples

        self.state.total_examples_processed += examples
        self._save_state()

    def mark_dataset_completed(self, dataset_name: str):
        """Mark a dataset as completed"""
        if self.state:
            if dataset_name in self.state.dataset_progress:
                self.state.dataset_progress[dataset_name]['completed'] = True
                self.state.dataset_progress[dataset_name]['completed_at'] = datetime.now().isoformat()
            self.state.total_datasets_completed += 1
            self._save_state()

    def get_dataset_progress(self, dataset_name: str) -> Optional[Dict]:
        """Get progress for a specific dataset"""
        if self.state and dataset_name in self.state.dataset_progress:
            return self.state.dataset_progress[dataset_name]
        return None

    def add_training_time(self, seconds: float):
        """Add elapsed training time"""
        if self.state:
            self.state.total_training_time_seconds += seconds
            self._save_state()

    # =========================================================================
    # Download Cache Management
    # =========================================================================

    def is_dataset_cached(self, dataset_identifier: str) -> bool:
        """Check if a dataset is already downloaded"""
        if dataset_identifier in self.download_cache:
            cache_entry = self.download_cache[dataset_identifier]
            if cache_entry.get('status') == DownloadStatus.CACHED.value:
                path = cache_entry.get('path')
                if path and Path(path).exists():
                    return True
        return False

    def get_cached_path(self, dataset_identifier: str) -> Optional[str]:
        """Get path to cached dataset"""
        if self.is_dataset_cached(dataset_identifier):
            return self.download_cache[dataset_identifier].get('path')
        return None

    def mark_download_started(self, dataset_identifier: str, dataset_name: str):
        """Mark dataset download as started"""
        self.download_cache[dataset_identifier] = {
            'name': dataset_name,
            'status': DownloadStatus.IN_PROGRESS.value,
            'started_at': datetime.now().isoformat(),
            'path': None
        }
        self._save_download_cache()

    def mark_download_completed(self, dataset_identifier: str, path: str):
        """Mark dataset download as completed"""
        if dataset_identifier in self.download_cache:
            self.download_cache[dataset_identifier]['status'] = DownloadStatus.CACHED.value
            self.download_cache[dataset_identifier]['path'] = path
            self.download_cache[dataset_identifier]['completed_at'] = datetime.now().isoformat()
        else:
            self.download_cache[dataset_identifier] = {
                'status': DownloadStatus.CACHED.value,
                'path': path,
                'completed_at': datetime.now().isoformat()
            }
        self._save_download_cache()

    def mark_download_failed(self, dataset_identifier: str, error: str):
        """Mark dataset download as failed"""
        if dataset_identifier in self.download_cache:
            self.download_cache[dataset_identifier]['status'] = DownloadStatus.FAILED.value
            self.download_cache[dataset_identifier]['error'] = error
        self._save_download_cache()

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def save_checkpoint(
        self,
        brain,
        name: str,
        stage: str,
        dataset_name: str,
        examples_processed: int,
        loss: float = 0.0
    ) -> Optional[str]:
        """
        Save a brain checkpoint.

        Args:
            brain: NIMCP brain object
            name: Checkpoint name
            stage: Current developmental stage
            dataset_name: Current dataset being trained
            examples_processed: Total examples processed so far
            loss: Current loss value

        Returns:
            Checkpoint ID or None if failed
        """
        checkpoint_id = self._generate_checkpoint_id()
        checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}_{name}.bin"

        try:
            # Save brain state
            brain.save(str(checkpoint_file))

            # Get brain size
            brain_size = checkpoint_file.stat().st_size

            # Create checkpoint info
            checkpoint_info = CheckpointInfo(
                checkpoint_id=checkpoint_id,
                filepath=str(checkpoint_file),
                created_at=datetime.now().isoformat(),
                stage=stage,
                dataset_name=dataset_name,
                examples_processed=examples_processed,
                total_loss=loss,
                brain_size=brain_size
            )

            # Add to state
            if self.state:
                self.state.checkpoints.append(asdict(checkpoint_info))
                self.state.last_checkpoint_id = checkpoint_id
                self._save_state(force=True)

            print(f"Saved checkpoint: {checkpoint_id} ({brain_size / 1024 / 1024:.1f} MB)")
            return checkpoint_id

        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_id: str = None):
        """
        Load a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint to load, or None for latest

        Returns:
            Path to checkpoint file, or None if not found
        """
        if not self.state or not self.state.checkpoints:
            return None

        if checkpoint_id is None:
            checkpoint_id = self.state.last_checkpoint_id

        if checkpoint_id is None:
            return None

        # Find checkpoint
        for ckpt in self.state.checkpoints:
            if ckpt['checkpoint_id'] == checkpoint_id:
                filepath = ckpt['filepath']
                if Path(filepath).exists():
                    return filepath
                else:
                    print(f"Checkpoint file missing: {filepath}")
                    return None

        return None

    def get_latest_checkpoint(self) -> Optional[Dict]:
        """Get info about the latest checkpoint"""
        if self.state and self.state.checkpoints:
            return self.state.checkpoints[-1]
        return None

    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints"""
        if self.state:
            return self.state.checkpoints
        return []

    def cleanup_old_checkpoints(self, keep_latest: int = 5):
        """Remove old checkpoints, keeping the latest N"""
        if not self.state or len(self.state.checkpoints) <= keep_latest:
            return

        to_remove = self.state.checkpoints[:-keep_latest]
        self.state.checkpoints = self.state.checkpoints[-keep_latest:]

        for ckpt in to_remove:
            filepath = Path(ckpt['filepath'])
            if filepath.exists():
                filepath.unlink()
                print(f"Removed old checkpoint: {ckpt['checkpoint_id']}")

        self._save_state(force=True)

    # =========================================================================
    # Metrics Persistence
    # =========================================================================

    def save_metrics(self, metrics: Dict, name: str = None):
        """Save training metrics to file"""
        if name is None:
            name = datetime.now().strftime("%Y%m%d_%H%M%S")

        metrics_file = self.metrics_dir / f"metrics_{name}.json"

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved metrics: {metrics_file}")

    def load_metrics(self, name: str) -> Optional[Dict]:
        """Load metrics from file"""
        metrics_file = self.metrics_dir / f"metrics_{name}.json"

        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return None

    # =========================================================================
    # Summary and Reporting
    # =========================================================================

    def get_summary(self) -> Dict:
        """Get summary of current training state"""
        if not self.state:
            return {"status": "no_session"}

        return {
            "session_id": self.state.session_id,
            "status": self.state.status,
            "current_stage": self.state.current_stage,
            "total_examples": self.state.total_examples_processed,
            "total_datasets_completed": self.state.total_datasets_completed,
            "training_time_hours": self.state.total_training_time_seconds / 3600,
            "checkpoints_saved": len(self.state.checkpoints),
            "last_checkpoint": self.state.last_checkpoint_id,
            "datasets_in_progress": len(self.state.dataset_progress),
            "error_count": self.state.error_count,
            "recovery_attempts": self.state.recovery_attempts
        }

    def print_status(self):
        """Print current training status"""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("TRAINING STATE SUMMARY")
        print("=" * 60)

        if summary["status"] == "no_session":
            print("No active training session")
            return

        print(f"Session ID: {summary['session_id']}")
        print(f"Status: {summary['status']}")
        print(f"Current Stage: {summary['current_stage']}")
        print(f"Total Examples: {summary['total_examples']:,}")
        print(f"Datasets Completed: {summary['total_datasets_completed']}")
        print(f"Training Time: {summary['training_time_hours']:.2f} hours")
        print(f"Checkpoints: {summary['checkpoints_saved']}")
        print(f"Errors: {summary['error_count']}")
        print("=" * 60 + "\n")


# =============================================================================
# Convenience Functions
# =============================================================================

_global_state_manager: Optional[TrainingStateManager] = None

def get_state_manager(state_dir: str = "./training_state") -> TrainingStateManager:
    """Get or create the global state manager"""
    global _global_state_manager
    if _global_state_manager is None:
        _global_state_manager = TrainingStateManager(state_dir)
    return _global_state_manager


def quick_checkpoint(brain, name: str, **kwargs):
    """Quick checkpoint save using global state manager"""
    mgr = get_state_manager()
    return mgr.save_checkpoint(brain, name, **kwargs)


def can_resume_training() -> bool:
    """Check if training can be resumed"""
    mgr = get_state_manager()
    return mgr.can_resume()


# =============================================================================
# Main - Testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIMCP Training State Manager")
    parser.add_argument("--status", action="store_true", help="Show training status")
    parser.add_argument("--checkpoints", action="store_true", help="List checkpoints")
    parser.add_argument("--cleanup", type=int, help="Cleanup old checkpoints, keep N")
    parser.add_argument("--state-dir", type=str, default="./training_state",
                       help="State directory path")

    args = parser.parse_args()

    mgr = TrainingStateManager(args.state_dir)

    if args.status:
        mgr.print_status()

    if args.checkpoints:
        checkpoints = mgr.list_checkpoints()
        print(f"\nCheckpoints ({len(checkpoints)}):")
        for ckpt in checkpoints:
            print(f"  - {ckpt['checkpoint_id']}: {ckpt['stage']} @ {ckpt['examples_processed']:,} examples")

    if args.cleanup:
        mgr.cleanup_old_checkpoints(keep_latest=args.cleanup)
        print(f"Kept latest {args.cleanup} checkpoints")
