"""Thread-safe multi-brain registry.

Adapts to the actual nimcp Python API:
  - brain.predict(features) -> (label_str, confidence)
  - brain.learn(features, label_str, confidence) -> loss_float
  - brain.get_neuron_count() -> int
  - brain.get_utilization_metrics() -> (utilization, saturation)
  - brain.broadcast_probe() -> bool
  - brain.snapshot_cow() -> capsule
  - brain.restore_cow(capsule) -> bool
  - brain.save(path) / brain.load(path)
  - No brain.probe() returning dict; we build metrics ourselves.
"""
import json
import os
import shutil
import threading
import time
from collections import deque
from typing import Optional

import nimcp

import nimcp_logger
from config import MAX_BRAIN_COUNT, PROBE_HISTORY_SIZE, BRAIN_STORAGE_DIR

_log = nimcp_logger.get("brain_manager")

SIZE_LABELS = {0: "Tiny", 1: "Small", 2: "Medium", 3: "Large"}
TASK_LABELS = {0: "Classification", 1: "Regression", 2: "Pattern Matching", 3: "Sequence", 4: "Association"}


class BrainStats:
    """Tracks metrics that the C API doesn't expose as a probe dict."""
    __slots__ = ("total_inferences", "total_learning_steps", "total_correct",
                 "last_loss", "losses", "inference_times")

    def __init__(self):
        self.total_inferences = 0
        self.total_learning_steps = 0
        self.total_correct = 0
        self.last_loss = 0.0
        self.losses: deque = deque(maxlen=200)
        self.inference_times: deque = deque(maxlen=200)

    def to_dict(self) -> dict:
        return {
            "total_inferences": self.total_inferences,
            "total_learning_steps": self.total_learning_steps,
            "total_correct": self.total_correct,
            "last_loss": self.last_loss,
            "losses": list(self.losses),
        }

    def load_dict(self, d: dict):
        self.total_inferences = d.get("total_inferences", 0)
        self.total_learning_steps = d.get("total_learning_steps", 0)
        self.total_correct = d.get("total_correct", 0)
        self.last_loss = d.get("last_loss", 0.0)
        for v in d.get("losses", []):
            self.losses.append(v)


class BrainMetadata:
    __slots__ = ("name", "created_at", "dataset", "parent_id", "num_inputs", "num_outputs", "size", "task")

    def __init__(self, name: str, num_inputs: int = 0, num_outputs: int = 0,
                 size: int = 1, task: int = 0,
                 dataset: Optional[str] = None, parent_id: Optional[int] = None,
                 created_at: Optional[str] = None):
        self.name = name
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.dataset = dataset
        self.parent_id = parent_id
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.size = size
        self.task = task

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "dataset": self.dataset,
            "parent_id": self.parent_id,
            "num_inputs": self.num_inputs,
            "num_outputs": self.num_outputs,
            "size": self.size,
            "task": self.task,
        }


class BrainManager:
    """Thread-safe singleton managing brain instances."""

    def __init__(self):
        self._lock = threading.Lock()
        self._brains: dict[int, nimcp.Brain] = {}
        self._metadata: dict[int, BrainMetadata] = {}
        self._stats: dict[int, BrainStats] = {}
        self._probe_history: dict[int, deque] = {}
        self._next_id = 0
        self._snapshots: dict[int, dict] = {}  # brain_id -> {name: path}
        self._cow_snapshots: dict[int, dict] = {}  # brain_id -> {name: capsule}

    # --- Persistence ---

    def _brain_dir(self, bid: int) -> str:
        return os.path.join(BRAIN_STORAGE_DIR, str(bid))

    def _save_brain_to_disk(self, bid: int):
        """Save brain weights + metadata to disk. Must be called with _lock held."""
        brain = self._brains.get(bid)
        if brain is None:
            return
        bdir = self._brain_dir(bid)
        os.makedirs(bdir, exist_ok=True)
        try:
            brain.save(os.path.join(bdir, "brain.bin"))
            meta_dict = self._metadata[bid].to_dict()
            meta_dict["stats"] = self._stats[bid].to_dict()
            with open(os.path.join(bdir, "meta.json"), "w") as f:
                json.dump(meta_dict, f, indent=2)
            _log.debug("Saved brain %d to disk", bid)
        except Exception as exc:
            _log.error("Failed to save brain %d to disk: %s", bid, exc)

    def save_brain_async(self, bid: int):
        """Public wrapper for saving a brain to disk (acquires lock)."""
        with self._lock:
            self._save_brain_to_disk(bid)

    def _delete_brain_from_disk(self, bid: int):
        bdir = self._brain_dir(bid)
        if os.path.isdir(bdir):
            shutil.rmtree(bdir, ignore_errors=True)
            _log.debug("Deleted brain %d from disk", bid)

    def load_all_from_disk(self):
        """Scan storage dir and restore all saved brains."""
        if not os.path.isdir(BRAIN_STORAGE_DIR):
            _log.info("No brain storage dir found at %s", BRAIN_STORAGE_DIR)
            return
        loaded = 0
        max_id = -1
        for entry in sorted(os.listdir(BRAIN_STORAGE_DIR)):
            bdir = os.path.join(BRAIN_STORAGE_DIR, entry)
            meta_path = os.path.join(bdir, "meta.json")
            bin_path = os.path.join(bdir, "brain.bin")
            if not os.path.isfile(meta_path) or not os.path.isfile(bin_path):
                continue
            try:
                bid = int(entry)
            except ValueError:
                continue
            try:
                with open(meta_path) as f:
                    meta_dict = json.load(f)
                brain = nimcp.Brain(
                    name=meta_dict["name"],
                    size=meta_dict.get("size", 1),
                    task=meta_dict.get("task", 0),
                    num_inputs=meta_dict.get("num_inputs", 4),
                    num_outputs=meta_dict.get("num_outputs", 3),
                )
                brain.load(bin_path)
                meta = BrainMetadata(
                    name=meta_dict["name"],
                    num_inputs=meta_dict.get("num_inputs", 4),
                    num_outputs=meta_dict.get("num_outputs", 3),
                    size=meta_dict.get("size", 1),
                    task=meta_dict.get("task", 0),
                    dataset=meta_dict.get("dataset"),
                    parent_id=meta_dict.get("parent_id"),
                    created_at=meta_dict.get("created_at"),
                )
                stats = BrainStats()
                stats_dict = meta_dict.get("stats")
                if stats_dict:
                    stats.load_dict(stats_dict)
                with self._lock:
                    self._brains[bid] = brain
                    self._metadata[bid] = meta
                    self._stats[bid] = stats
                    self._probe_history[bid] = deque(maxlen=PROBE_HISTORY_SIZE)
                    self._snapshots[bid] = {}
                    self._cow_snapshots[bid] = {}
                    if bid > max_id:
                        max_id = bid
                loaded += 1
                _log.info("Restored brain %d (%s) from disk", bid, meta_dict["name"])
            except Exception as exc:
                _log.error("Failed to restore brain %d: %s", bid, exc)
        with self._lock:
            if max_id >= self._next_id:
                self._next_id = max_id + 1
        _log.info("Loaded %d brains from disk", loaded)

    # --- CRUD ---

    def create_brain(self, name: str, size: int, task: int,
                     num_inputs: int, num_outputs: int,
                     dataset: Optional[str] = None) -> int:
        with self._lock:
            if len(self._brains) >= MAX_BRAIN_COUNT:
                raise ValueError(f"Maximum brain count ({MAX_BRAIN_COUNT}) reached")
            brain = nimcp.Brain(
                name=name, size=size, task=task,
                num_inputs=num_inputs, num_outputs=num_outputs
            )
            bid = self._next_id
            self._next_id += 1
            self._brains[bid] = brain
            self._metadata[bid] = BrainMetadata(
                name=name, num_inputs=num_inputs, num_outputs=num_outputs,
                size=size, task=task, dataset=dataset,
            )
            self._stats[bid] = BrainStats()
            self._probe_history[bid] = deque(maxlen=PROBE_HISTORY_SIZE)
            self._snapshots[bid] = {}
            self._cow_snapshots[bid] = {}
            self._save_brain_to_disk(bid)
            _log.info("Created brain %d (%s) inputs=%d outputs=%d", bid, name, num_inputs, num_outputs)
            return bid

    def destroy_brain(self, bid: int) -> bool:
        with self._lock:
            if bid not in self._brains:
                return False
            del self._brains[bid]
            del self._metadata[bid]
            del self._stats[bid]
            del self._probe_history[bid]
            self._snapshots.pop(bid, None)
            self._cow_snapshots.pop(bid, None)
            self._delete_brain_from_disk(bid)
            _log.info("Destroyed brain %d", bid)
            return True

    def get_brain(self, bid: int) -> Optional[nimcp.Brain]:
        with self._lock:
            return self._brains.get(bid)

    def get_metadata(self, bid: int) -> Optional[BrainMetadata]:
        with self._lock:
            return self._metadata.get(bid)

    def has_brain(self, bid: int) -> bool:
        with self._lock:
            return bid in self._brains

    # --- Rename ---

    def rename_brain(self, bid: int, new_name: str) -> bool:
        with self._lock:
            meta = self._metadata.get(bid)
            if meta is None:
                return False
            meta.name = new_name
            self._save_brain_to_disk(bid)
            _log.info("Renamed brain %d to '%s'", bid, new_name)
            return True

    # --- Detail view ---

    def get_brain_detail(self, bid: int) -> Optional[dict]:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return None
            meta = self._metadata[bid]
            stats = self._stats[bid]
            probe = self._build_probe(bid)
            accuracy = (stats.total_correct / stats.total_inferences) if stats.total_inferences > 0 else 0.0
            return {
                "id": bid,
                "name": meta.name,
                "created_at": meta.created_at,
                "size": meta.size,
                "size_label": SIZE_LABELS.get(meta.size, "Unknown"),
                "task": meta.task,
                "task_label": TASK_LABELS.get(meta.task, "Unknown"),
                "num_inputs": meta.num_inputs,
                "num_outputs": meta.num_outputs,
                "dataset": meta.dataset,
                "parent_id": meta.parent_id,
                "probe": probe,
                "total_inferences": stats.total_inferences,
                "total_learning_steps": stats.total_learning_steps,
                "accuracy": accuracy,
                "last_loss": stats.last_loss,
                "loss_history": list(stats.losses),
            }

    # --- Probe ---

    def _build_probe(self, bid: int) -> Optional[dict]:
        """Build a probe-like dict from available API calls + tracked stats."""
        brain = self._brains.get(bid)
        if brain is None:
            return None
        meta = self._metadata[bid]
        stats = self._stats[bid]

        try:
            neuron_count = brain.get_neuron_count()
        except Exception as exc:
            _log.debug("get_neuron_count failed for brain %d: %s", bid, exc)
            neuron_count = 0

        try:
            utilization, saturation = brain.get_utilization_metrics()
        except Exception as exc:
            _log.debug("get_utilization_metrics failed for brain %d: %s", bid, exc)
            utilization, saturation = 0.0, 0.0

        accuracy = (stats.total_correct / stats.total_inferences) if stats.total_inferences > 0 else 0.0
        avg_inference_us = 0.0
        if stats.inference_times:
            avg_inference_us = sum(stats.inference_times) / len(stats.inference_times)

        return {
            "task_name": meta.name,
            "size": meta.size,
            "task": meta.task,
            "num_neurons": neuron_count,
            "num_synapses": 0,
            "num_active_synapses": 0,
            "total_inferences": stats.total_inferences,
            "total_learning_steps": stats.total_learning_steps,
            "avg_sparsity": saturation,
            "avg_inference_time_us": avg_inference_us,
            "current_learning_rate": 0.001,
            "accuracy": accuracy,
            "memory_bytes": 0,
            "num_inputs": meta.num_inputs,
            "num_outputs": meta.num_outputs,
            "utilization": utilization,
            "last_loss": stats.last_loss,
            "is_cow_clone": meta.parent_id is not None,
            "cow_ref_count": 0,
            "cow_shared_bytes": 0,
            "cow_private_bytes": 0,
        }

    def list_brains(self) -> list[dict]:
        with self._lock:
            result = []
            for bid, meta in self._metadata.items():
                info = {
                    "id": bid,
                    "name": meta.name,
                    "created_at": meta.created_at,
                    "dataset": meta.dataset,
                    "parent_id": meta.parent_id,
                    "probe": self._build_probe(bid),
                }
                result.append(info)
            return result

    def probe_brain(self, bid: int) -> Optional[dict]:
        with self._lock:
            probe = self._build_probe(bid)
            if probe is None:
                return None
            self._probe_history[bid].append({
                "timestamp": time.time(),
                **probe,
            })
            return probe

    def get_probe_history(self, bid: int) -> list[dict]:
        with self._lock:
            hist = self._probe_history.get(bid)
            return list(hist) if hist else []

    # --- Inference / Learning ---

    def predict(self, bid: int, features: list[float]) -> Optional[tuple[str, float]]:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return None
            stats = self._stats[bid]
            t0 = time.monotonic()
            result = brain.predict(features)
            elapsed_us = (time.monotonic() - t0) * 1_000_000
            stats.total_inferences += 1
            stats.inference_times.append(elapsed_us)
            return result  # (label_str, confidence)

    def decide(self, bid: int, features: list[float]) -> Optional[dict]:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return None
            stats = self._stats[bid]
            t0 = time.monotonic()
            result = brain.decide(features)
            elapsed_us = (time.monotonic() - t0) * 1_000_000
            stats.total_inferences += 1
            stats.inference_times.append(elapsed_us)
            return result  # dict with label, confidence, explanation, sparsity, etc.

    def learn(self, bid: int, features: list[float], label: str, confidence: float = 1.0) -> Optional[float]:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return None
            stats = self._stats[bid]
            loss = brain.learn(features, label, confidence)
            stats.total_learning_steps += 1
            stats.last_loss = float(loss) if loss is not None else stats.last_loss
            stats.losses.append(stats.last_loss)
            return stats.last_loss

    def track_correct(self, bid: int, correct: bool):
        """Track prediction correctness for accuracy calculation."""
        with self._lock:
            if correct and bid in self._stats:
                self._stats[bid].total_correct += 1

    def update_training_loss(self, bid: int, loss: float):
        """Update loss during training so dashboard charts show it live."""
        with self._lock:
            stats = self._stats.get(bid)
            if stats is None:
                return
            stats.last_loss = loss
            stats.losses.append(loss)

    # --- Save/Load (snapshots) ---

    def save_brain(self, bid: int, path: str) -> bool:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return False
            try:
                brain.save(path)
                return True
            except Exception as exc:
                _log.error("Failed to save brain %d to %s: %s", bid, path, exc)
                return False

    def load_brain(self, bid: int, path: str) -> bool:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return False
            try:
                brain.load(path)
                return True
            except Exception as exc:
                _log.error("Failed to load brain %d from %s: %s", bid, path, exc)
                return False

    def snapshot_cow(self, bid: int) -> bool:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return False
            try:
                brain.snapshot_cow()
                return True
            except Exception as exc:
                _log.error("COW snapshot failed for brain %d: %s", bid, exc)
                return False

    def restore_cow(self, bid: int) -> bool:
        with self._lock:
            brain = self._brains.get(bid)
            if brain is None:
                return False
            try:
                brain.restore_cow()
                return True
            except Exception as exc:
                _log.error("COW restore failed for brain %d: %s", bid, exc)
                return False

    # --- Snapshot accessors (thread-safe) ---

    def register_snapshot(self, bid: int, name: str, path: str) -> None:
        with self._lock:
            self._snapshots.setdefault(bid, {})[name] = path

    def list_snapshot_names(self, bid: int) -> list[dict]:
        with self._lock:
            return [{"name": n, "path": p} for n, p in self._snapshots.get(bid, {}).items()]

    def get_snapshot_path(self, bid: int, name: str) -> Optional[str]:
        with self._lock:
            return self._snapshots.get(bid, {}).get(name)

    def remove_snapshot(self, bid: int, name: str) -> bool:
        with self._lock:
            snaps = self._snapshots.get(bid, {})
            if name not in snaps:
                return False
            del snaps[name]
            return True

    def destroy_all(self):
        with self._lock:
            _log.info("Destroying all %d brains", len(self._brains))
            self._brains.clear()
            self._metadata.clear()
            self._stats.clear()
            self._probe_history.clear()
            self._snapshots.clear()
            self._cow_snapshots.clear()


manager = BrainManager()
