#!/usr/bin/env python3
"""
School — Parallel Instructor Coordinator for Athena Training
==============================================================

WHAT: Coordinate 23 parallel InstructorAgent threads teaching one shared brain
WHY:  Like a school: each teacher specializes in a domain, all teach the same
      student simultaneously. The "Principal" (School) manages recess
      (consolidation), dashboards, checkpoints, and graduation.
HOW:  Threading + queue-based communication. Each instructor is a daemon thread.
      The coordinator loop ticks every 1s, drains reports, calls recess,
      prints dashboards, and saves checkpoints.

Architecture:
  School (main thread — coordinator/"Principal")
    ├── stop_event, recess_event        (threading.Event)
    ├── school_queue                     (instructor → school reports)
    ├── cross_domain_queues              (per-domain inter-instructor exemplars)
    ├── SchoolProgressBoard              (thread-safe metrics)
    └── InstructorAgent × 23             (20 text + 3 multimodal)
"""

import collections
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cognitive_orchestrator import CognitiveOrchestrator
from instructor_agent import InstructorAgent, InstructorConfig


# ---------------------------------------------------------------------------
# Thread-Safe Brain Wrapper
# ---------------------------------------------------------------------------

class ThreadSafeBrain:
    """Wraps a nimcp Brain with a reentrant lock so concurrent instructor
    threads serialize their brain.learn() / brain.decide() calls.
    The underlying C library is NOT thread-safe for concurrent mutations."""

    def __init__(self, brain):
        self._brain = brain
        self._lock = threading.RLock()

    def learn(self, *args, **kwargs):
        with self._lock:
            return self._brain.learn(*args, **kwargs)

    def predict_fast(self, *args, **kwargs):
        with self._lock:
            return self._brain.predict_fast(*args, **kwargs)

    def predict_in_domain(self, *args, **kwargs):
        with self._lock:
            return self._brain.predict_in_domain(*args, **kwargs)

    def predict(self, *args, **kwargs):
        with self._lock:
            return self._brain.predict(*args, **kwargs)

    def decide(self, *args, **kwargs):
        with self._lock:
            return self._brain.decide(*args, **kwargs)

    def ti_compute_unified_lr(self, *args, **kwargs):
        with self._lock:
            return self._brain.ti_compute_unified_lr(*args, **kwargs)

    def ti_compute_modulation_state(self, *args, **kwargs):
        with self._lock:
            return self._brain.ti_compute_modulation_state(*args, **kwargs)

    def ti_post_batch_update(self, *args, **kwargs):
        with self._lock:
            return self._brain.ti_post_batch_update(*args, **kwargs)

    def ti_compute_decision_cycle(self, *args, **kwargs):
        with self._lock:
            return self._brain.ti_compute_decision_cycle(*args, **kwargs)

    def get_last_gradient_norm(self, *args, **kwargs):
        with self._lock:
            return self._brain.get_last_gradient_norm(*args, **kwargs)

    def consolidate(self, *args, **kwargs):
        with self._lock:
            return self._brain.consolidate(*args, **kwargs)

    def save(self, *args, **kwargs):
        with self._lock:
            return self._brain.save(*args, **kwargs)

    def load(self, *args, **kwargs):
        with self._lock:
            return self._brain.load(*args, **kwargs)

    def probe(self, *args, **kwargs):
        with self._lock:
            return self._brain.probe(*args, **kwargs)

    def cache_communities(self, *args, **kwargs):
        with self._lock:
            return self._brain.cache_communities(*args, **kwargs)

    def invalidate_community_cache(self, *args, **kwargs):
        with self._lock:
            return self._brain.invalidate_community_cache(*args, **kwargs)

    def audio_cortex_process(self, *args, **kwargs):
        with self._lock:
            return self._brain.audio_cortex_process(*args, **kwargs)

    def visual_cortex_process(self, *args, **kwargs):
        with self._lock:
            return self._brain.visual_cortex_process(*args, **kwargs)

    def speech_cortex_process(self, *args, **kwargs):
        with self._lock:
            return self._brain.speech_cortex_process(*args, **kwargs)

    # Read-only accessors — still need RLock because the underlying C
    # brain may be mid-mutation from another thread.
    def get_neuron_count(self, *args, **kwargs):
        with self._lock:
            return self._brain.get_neuron_count(*args, **kwargs)

    def get_accuracy(self, *args, **kwargs):
        with self._lock:
            return self._brain.get_accuracy(*args, **kwargs)

    def get_synapse_count(self, *args, **kwargs):
        with self._lock:
            return self._brain.get_synapse_count(*args, **kwargs)

    def get_last_loss(self, *args, **kwargs):
        with self._lock:
            return self._brain.get_last_loss(*args, **kwargs)

    def get_uncertainty(self, *args, **kwargs):
        with self._lock:
            return self._brain.get_uncertainty(*args, **kwargs)

    def self_assess(self, *args, **kwargs):
        with self._lock:
            return self._brain.self_assess(*args, **kwargs)

    def curiosity_detect_gaps(self, *args, **kwargs):
        with self._lock:
            return self._brain.curiosity_detect_gaps(*args, **kwargs)

    def acquire_exclusive(self):
        """Acquire exclusive brain access for consolidation."""
        self._lock.acquire()

    def release_exclusive(self):
        """Release exclusive brain access."""
        self._lock.release()

    def get_raw_brain(self):
        """Get the underlying brain for direct access while lock is held."""
        return self._brain

    def __getattr__(self, name):
        """Proxy all other attributes to the underlying brain with lock.

        NOTE: Only attributes not explicitly wrapped above go through this
        fallback.  All proxied access acquires the RLock to prevent
        concurrent C-level mutations.  If you add a new brain method that
        mutates state, add an explicit wrapper above rather than relying on
        this passthrough.

        L1 note (TOCTOU): The getattr → callable check → cache sequence has
        a narrow window where another thread could see a stale cache entry.
        Under CPython's GIL, object.__setattr__ is atomic for simple
        attribute assignment, so the worst case is a redundant wrapper
        creation (benign). No additional synchronization needed.
        """
        # M3: Acquire lock before initial getattr to prevent reading a
        # non-callable attribute while another thread is mid-mutation.
        with self._lock:
            attr = getattr(self._brain, name)
        if callable(attr):
            def _locked_method(*args, **kwargs):
                with self._lock:
                    return attr(*args, **kwargs)
            # Cache the wrapper on the instance
            object.__setattr__(self, name, _locked_method)
            return _locked_method
        else:
            with self._lock:
                return getattr(self._brain, name)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SchoolConfig:
    recess_interval_s: float = 300.0       # Consolidation every 5 min
    report_interval_s: float = 30.0        # Dashboard every 30s
    checkpoint_interval_s: float = 600.0   # Checkpoint every 10 min
    max_training_time_s: float = 86400.0   # 24h max
    graduation_mastery: float = 0.85       # Graduate when all domains reach this
    max_examples_per_dataset: int = 50_000
    startup_stagger_s: float = 2.0         # Delay between instructor starts
    num_inputs: int = 1024   # Match ATHENA_NUM_INPUTS (was 128)
    num_outputs: int = 256   # Match ATHENA_NUM_OUTPUTS (kept for API compat, not used internally)
    max_concurrent_instructors: int = 1    # Sequential: C library has memory corruption with concurrent access even through RLock
    min_domain_accuracy: float = 0.0       # Per-domain min accuracy before "done" (0=disabled)
    max_retry_passes: int = 5              # Max re-teach passes if below accuracy threshold


# ---------------------------------------------------------------------------
# Thread-Safe Progress Board
# ---------------------------------------------------------------------------

class SchoolProgressBoard:
    """Thread-safe aggregate metrics across all instructors."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reports: Dict[str, dict] = {}
        self._recess_count = 0
        self._checkpoint_count = 0

    def update_instructor(self, domain: str, report: dict):
        with self._lock:
            self._reports[domain] = report

    def increment_recess(self):
        with self._lock:
            self._recess_count += 1

    def increment_checkpoint(self):
        with self._lock:
            self._checkpoint_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            domains = {}
            total = 0
            for domain, r in self._reports.items():
                domains[domain] = {
                    "examples": r.get("total_examples", 0),
                    "accuracy": r.get("accuracy", 0.0),
                    "mastery": r.get("mastery", 0.0),
                    "rate": r.get("examples_per_sec", 0.0),
                    "modality": r.get("modality", "text"),
                }
                total += r.get("total_examples", 0)
            return {
                "domains": domains,
                "total_examples": total,
                "num_instructors": len(self._reports),
                "recess_count": self._recess_count,
                "checkpoint_count": self._checkpoint_count,
            }


# ---------------------------------------------------------------------------
# Metacognitive Training Strategy
# ---------------------------------------------------------------------------

class TrainingMetacognition:
    """Metacognitive controller for training strategy adaptation.

    WHAT: Assesses training quality per domain and adapts resource allocation
    WHY:  Fixed instructor scheduling wastes resources on mastered domains
          and under-invests in struggling domains
    HOW:  Per-domain EMA tracking of accuracy, loss trend, and stall detection
    """

    def __init__(self, logger, mastery_threshold: float = 0.85):
        self.logger = logger
        # M2 fix: Use configurable mastery threshold instead of hardcoded 0.85
        self._mastery_threshold = mastery_threshold
        self._domain_stats: Dict[str, dict] = {}  # domain -> {ema_accuracy, ema_loss, stall_count, ...}
        self._assessment_count = 0
        self._reallocation_count = 0
        # L4: Cache last assess() result so format_dashboard doesn't duplicate logic
        self._last_assessment: Dict[str, dict] = {}

    def update(self, domain: str, accuracy: float, loss: float, examples: int):
        """Update per-domain statistics with latest instructor report."""
        if domain not in self._domain_stats:
            self._domain_stats[domain] = {
                'ema_accuracy': accuracy,
                'ema_loss': loss,
                'prev_accuracy': accuracy,
                'stall_count': 0,
                'total_examples': 0,
                'assessments': 0,
                'priority': 1.0,  # 1.0 = normal, >1 = needs more attention
                'mastered': False,
            }

        stats = self._domain_stats[domain]
        alpha = 0.1  # EMA smoothing
        stats['ema_accuracy'] = alpha * accuracy + (1 - alpha) * stats['ema_accuracy']
        stats['ema_loss'] = alpha * loss + (1 - alpha) * stats['ema_loss']
        stats['total_examples'] = examples
        stats['assessments'] += 1

        # Stall detection: accuracy not improving over multiple assessments
        if abs(stats['ema_accuracy'] - stats['prev_accuracy']) < 0.005:
            stats['stall_count'] += 1
        else:
            stats['stall_count'] = 0
        stats['prev_accuracy'] = stats['ema_accuracy']

        # Mastery detection (M2 fix: uses configurable threshold)
        if stats['ema_accuracy'] > self._mastery_threshold and stats['assessments'] > 10:
            stats['mastered'] = True

        # De-mastery: if accuracy drops significantly below threshold, revoke mastery
        if stats['mastered'] and stats['ema_accuracy'] < (self._mastery_threshold - 0.10):
            stats['mastered'] = False

    def assess(self) -> Dict[str, dict]:
        """Run metacognitive assessment across all domains.

        Returns dict of domain -> {priority, status, recommendation}
        """
        self._assessment_count += 1
        results = {}

        for domain, stats in self._domain_stats.items():
            if stats['mastered']:
                priority = 0.3  # Reduce attention to mastered domains
                status = 'mastered'
                recommendation = 'reduce_slots'
            elif stats['stall_count'] > 5:
                priority = 2.0  # Double attention to stalled domains
                status = 'stalled'
                recommendation = 'increase_slots_and_diagnose'
            elif stats['ema_accuracy'] < 0.3 and stats['assessments'] > 5:
                priority = 1.8  # High attention to struggling domains
                status = 'struggling'
                recommendation = 'increase_slots'
            elif stats['ema_accuracy'] > 0.7:
                priority = 0.8  # Slightly reduce near-mastery domains
                status = 'progressing_well'
                recommendation = 'maintain'
            else:
                priority = 1.0
                status = 'learning'
                recommendation = 'maintain'

            stats['priority'] = priority
            results[domain] = {
                'priority': priority,
                'status': status,
                'recommendation': recommendation,
                'ema_accuracy': stats['ema_accuracy'],
                'ema_loss': stats['ema_loss'],
                'stall_count': stats['stall_count'],
                'total_examples': stats['total_examples'],
            }

        # L4: Cache for format_dashboard reuse
        self._last_assessment = results
        return results

    def get_priority_ranking(self) -> List[tuple]:
        """Return domains sorted by priority (highest first)."""
        return sorted(
            [(d, s['priority']) for d, s in self._domain_stats.items()],
            key=lambda x: x[1],
            reverse=True
        )

    def get_domain_priority(self, domain: str) -> float:
        """Return the priority for a given domain (public accessor)."""
        return self._domain_stats.get(domain, {}).get('priority', 1.0)

    def get_stalled_domains(self) -> List[str]:
        """Return list of domains that appear stalled."""
        return [d for d, s in self._domain_stats.items() if s['stall_count'] > 5]

    def get_mastered_domains(self) -> List[str]:
        """Return list of domains that appear mastered."""
        return [d for d, s in self._domain_stats.items() if s['mastered']]

    def has_stats(self):
        """Return True if any domain stats are available."""
        return bool(self._domain_stats)

    def format_dashboard(self) -> str:
        """Format metacognitive assessment as dashboard string.

        L4: Uses cached status from the last assess() call instead of
        duplicating the status derivation logic.
        """
        lines = ["\n[Metacognition] Domain Assessment:"]
        lines.append(f"  {'Domain':20s} {'Status':15s} {'Priority':>8s} {'Accuracy':>10s} {'Stall':>6s}")
        lines.append("  " + "-" * 65)
        for domain, stats in sorted(self._domain_stats.items()):
            # L4: Prefer cached status from assess() over re-deriving
            cached = self._last_assessment.get(domain)
            if cached:
                status = cached['status']
            elif stats['mastered']:
                status = 'mastered'
            elif stats['stall_count'] > 5:
                status = 'stalled'
            elif stats['ema_accuracy'] < 0.3 and stats['assessments'] > 5:
                status = 'struggling'
            elif stats['ema_accuracy'] > 0.7:
                status = 'progressing_well'
            else:
                status = 'learning'
            lines.append(
                f"  {domain:20s} {status:15s} {stats['priority']:8.2f} "
                f"{stats['ema_accuracy']:10.4f} {stats['stall_count']:6d}"
            )
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# School Coordinator
# ---------------------------------------------------------------------------

class School:
    """
    Coordinator that manages 23 parallel instructor agents.
    """

    def __init__(self, brain, config: SchoolConfig, logger):
        self.brain = ThreadSafeBrain(brain)
        self.config = config
        self.logger = logger

        self.stop_event = threading.Event()
        self.recess_event = threading.Event()
        self.school_queue: queue.Queue = queue.Queue(maxsize=1000)
        # H1: Per-domain cross-domain queues so consumers only receive exemplars
        # targeted at their domain (prevents mismatched cross-domain transfer).
        # The shared fallback queue is kept for backward compat with _publish_exemplar.
        self.cross_domain_queues: Dict[str, queue.Queue] = {}
        self.cross_domain_queue: queue.Queue = queue.Queue(maxsize=500)  # fallback
        self.board = SchoolProgressBoard()
        self.metacognition = TrainingMetacognition(
            self.logger, mastery_threshold=config.graduation_mastery)
        self.cognitive = CognitiveOrchestrator(self.brain)

        self._last_metacog = 0.0

        # Per-domain accuracy history for stall detection via CognitiveOrchestrator
        self._domain_accuracy_history: Dict[str, List[float]] = {}

        # Phase 4: Domain similarity matrix & cross-domain transfer
        self._domain_centroids: Dict[str, np.ndarray] = {}
        self._domain_similarity: Dict[str, Dict[str, float]] = {}
        self._last_similarity_update = time.time()
        self._similarity_update_interval = 120.0  # Update every 2 min
        self._last_transfer_check = time.time()
        self._transfer_check_interval = 180.0  # Check transfer opportunities every 3 min

        self.instructors: List[InstructorAgent] = []
        self._start_time = 0.0
        self._last_recess = 0.0
        self._last_report = 0.0
        self._last_checkpoint = 0.0
        self._last_config_check = time.time()

        # H1 fix: _domain_output_map removed — domain isolation uses string label prefixing only.

        # Hot-reload tracking
        self._config_path: Optional[Path] = None
        self._config_mtime: float = 0.0
        self._known_dataset_names: set = set()
        self._config_check_interval_s: float = 30.0

        # Logging
        self._school_log_path: Optional[Path] = None
        self._school_log_file = None

    def setup(self, config_path: Path):
        """Load dataset config and create instructor agents."""
        self._config_path = Path(config_path)
        self._config_mtime = self._config_path.stat().st_mtime

        try:
            with open(config_path) as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.log(f"[School] ERROR: Failed to load config {config_path}: {e}")
            config_data = {}

        datasets = config_data.get("datasets", [])

        # Track known dataset names for hot-reload diffing
        self._known_dataset_names = {ds.get("name", "") for ds in datasets}

        # Group datasets by domain
        domain_datasets: Dict[str, List[dict]] = {}
        for ds in datasets:
            domain = ds.get("domain", "general")
            domain_datasets.setdefault(domain, []).append(ds)

        # Determine modality per domain
        multimodal_domains = {"audio", "visual", "speech"}

        # Create instructor agents (no per-agent stagger — batching controls concurrency)
        log_dir = Path(self.logger.log_file).parent / "instructors" if hasattr(self.logger, 'log_file') else None

        for domain, ds_list in sorted(domain_datasets.items()):
            modality = domain if domain in multimodal_domains else "text"
            ic = InstructorConfig(
                domain=domain,
                modality=modality,
                max_examples_per_dataset=self.config.max_examples_per_dataset,
                startup_delay_s=0.0,
                min_domain_accuracy=self.config.min_domain_accuracy,
                max_retry_passes=self.config.max_retry_passes,
            )
            # H1: Ensure per-domain queue exists
            if domain not in self.cross_domain_queues:
                self.cross_domain_queues[domain] = queue.Queue(maxsize=500)
            agent = InstructorAgent(
                brain=self.brain,
                config=ic,
                datasets=ds_list,
                school_queue=self.school_queue,
                cross_domain_queue=self.cross_domain_queue,
                stop_event=self.stop_event,
                recess_event=self.recess_event,
                num_inputs=self.config.num_inputs,
                log_dir=log_dir,
                cross_domain_queues=self.cross_domain_queues,
            )
            self.instructors.append(agent)

        # H1 fix: Domain isolation is handled entirely by string label prefixing
        # (e.g. "biology:answer_a"), NOT by vector-based output neuron masking.
        # The old _allocate_domain_outputs() code was dead — the C backend does
        # not support output_range kwargs, and all targets are strings.

        self.logger.log(f"[School] Created {len(self.instructors)} instructors "
                        f"across {len(domain_datasets)} domains")
        for agent in self.instructors:
            self.logger.log(f"  {agent.config.domain:20s} [{agent.config.modality}] "
                            f"— {len(agent.datasets)} datasets, "
                            f"delay={agent.config.startup_delay_s:.1f}s")

    def start(self):
        """Start all instructors and run coordinator loop."""
        self._start_time = time.time()
        self._last_recess = self._start_time
        self._last_report = self._start_time
        self._last_checkpoint = self._start_time
        self._last_metacog = self._start_time

        # Open school-level log
        self._open_school_log()

        max_conc = self.config.max_concurrent_instructors
        self.logger.log(f"\n[School] Starting {len(self.instructors)} instructor threads "
                        f"(max {max_conc} concurrent)...")

        # Start instructors in batches to avoid heap corruption from too many
        # concurrent threads hitting the brain simultaneously
        self._pending_instructors = collections.deque(self.instructors)
        self._active_instructors = []
        self._start_next_batch(max_conc)

        # Run coordinator loop
        try:
            self._coordinator_loop()
        except KeyboardInterrupt:
            self.logger.log("[School] Interrupted — shutting down...")
        finally:
            self._shutdown()

    def _start_next_batch(self, count: int):
        """Start up to `count` instructors from the pending deque."""
        started = 0
        while self._pending_instructors and started < count:
            agent = self._pending_instructors.popleft()
            agent.start()
            self._active_instructors.append(agent)
            self.logger.log(f"  [School] Started instructor: {agent.config.domain}")
            started += 1
            # Brief stagger between starts within a batch to reduce contention
            if started < count and self._pending_instructors:
                time.sleep(self.config.startup_stagger_s)

    def _rotate_batches(self):
        """Retire finished instructors and start replacements (priority-aware)."""
        max_conc = self.config.max_concurrent_instructors
        # Remove finished threads from active list
        still_active = []
        for agent in self._active_instructors:
            if agent.is_finished or not agent.is_alive():
                self.logger.log(f"  [School] Instructor finished: {agent.config.domain}")
            else:
                still_active.append(agent)
        self._active_instructors = still_active

        # Sort pending by metacognitive priority (highest first)
        if self._pending_instructors and self._domain_stats_available():
            self._pending_instructors = collections.deque(sorted(
                self._pending_instructors,
                key=lambda a: self.metacognition.get_domain_priority(a.config.domain),
                reverse=True
            ))

        # Start new instructors to fill vacant slots
        slots = max_conc - len(self._active_instructors)
        if slots > 0 and self._pending_instructors:
            self.logger.log(f"  [School] {slots} slot(s) free, "
                            f"{len(self._pending_instructors)} pending — starting next batch")
            self._start_next_batch(slots)

    def _domain_stats_available(self):
        """Check if metacognition has accumulated any domain stats."""
        return self.metacognition.has_stats()

    def _check_config_reload(self):
        """Check if foundation_datasets_config.json has changed on disk.
        If new datasets were added, create InstructorAgents and queue them."""
        if not self._config_path or not self._config_path.exists():
            return

        try:
            current_mtime = self._config_path.stat().st_mtime
        except OSError:
            return

        if current_mtime <= self._config_mtime:
            return  # No change

        self._config_mtime = current_mtime
        self.logger.log("[School] Config file changed — checking for new datasets...")

        try:
            with open(self._config_path) as f:
                config_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            self.logger.log(f"[School] Config reload failed: {e}")
            return

        datasets = config_data.get("datasets", [])
        new_datasets = [ds for ds in datasets
                        if ds.get("name", "") not in self._known_dataset_names]

        if not new_datasets:
            self.logger.log("[School] Config changed but no new datasets found")
            return

        # Group new datasets by domain
        multimodal_domains = {"audio", "visual", "speech"}
        domain_new: Dict[str, List[dict]] = {}
        for ds in new_datasets:
            domain = ds.get("domain", "general")
            domain_new.setdefault(domain, []).append(ds)
            self._known_dataset_names.add(ds.get("name", ""))

        log_dir = (Path(self.logger.log_file).parent / "instructors"
                   if hasattr(self.logger, 'log_file') else None)

        added = 0
        for domain, ds_list in sorted(domain_new.items()):
            # Check if an existing instructor already covers this domain
            existing = [a for a in self.instructors
                        if a.config.domain == domain and not a.is_finished]
            if existing:
                # Domain already has an active instructor — skip
                # (the existing instructor won't see the new dataset since it
                #  was created with the old list, but it will finish eventually
                #  and the new datasets will need a fresh instructor)
                # L6 note: When the new instructor starts, its reports will
                # overwrite the old instructor's entry on the dashboard. This
                # is expected — the dashboard shows the most recent report per
                # domain, and the new instructor is the active one.
                self.logger.log(f"  [Hot-reload] {domain}: active instructor exists, "
                                f"queuing {len(ds_list)} new dataset(s) as separate instructor")

            modality = domain if domain in multimodal_domains else "text"
            ic = InstructorConfig(
                domain=domain,
                modality=modality,
                max_examples_per_dataset=self.config.max_examples_per_dataset,
                startup_delay_s=0.0,
                min_domain_accuracy=self.config.min_domain_accuracy,
                max_retry_passes=self.config.max_retry_passes,
            )
            # H1: Ensure per-domain queue exists for hot-reloaded domains
            if domain not in self.cross_domain_queues:
                self.cross_domain_queues[domain] = queue.Queue(maxsize=500)
            agent = InstructorAgent(
                brain=self.brain,
                config=ic,
                datasets=ds_list,
                school_queue=self.school_queue,
                cross_domain_queue=self.cross_domain_queue,
                stop_event=self.stop_event,
                recess_event=self.recess_event,
                num_inputs=self.config.num_inputs,
                log_dir=log_dir,
                cross_domain_queues=self.cross_domain_queues,
            )
            self.instructors.append(agent)
            self._pending_instructors.append(agent)
            added += 1

            self.logger.log(f"  [Hot-reload] Added instructor: {domain} "
                            f"[{modality}] — {len(ds_list)} dataset(s)")
            for ds in ds_list:
                self.logger.log(f"    + {ds.get('name', '?')}: "
                                f"{ds.get('description', '')[:80]}")

        self.logger.log(f"[School] Hot-reload complete: {added} new instructor(s), "
                        f"{len(self._pending_instructors)} pending")

    # ------------------------------------------------------------------
    # Phase 4: Domain Similarity & Cross-Domain Transfer
    # ------------------------------------------------------------------

    def _update_domain_centroids(self):
        """Collect centroids from all active instructors and compute similarity."""
        for agent in self.instructors:
            centroid = agent.get_domain_centroid()
            if centroid is not None:
                self._domain_centroids[agent.config.domain] = centroid

        # Compute pairwise cosine similarity
        domains = list(self._domain_centroids.keys())
        if len(domains) < 2:
            return

        for i, d1 in enumerate(domains):
            if d1 not in self._domain_similarity:
                self._domain_similarity[d1] = {}
            c1 = self._domain_centroids[d1]
            norm1 = np.linalg.norm(c1)
            if norm1 < 1e-8:
                continue
            for j, d2 in enumerate(domains):
                if i >= j:
                    continue
                c2 = self._domain_centroids[d2]
                norm2 = np.linalg.norm(c2)
                if norm2 < 1e-8:
                    continue
                sim = float(np.dot(c1, c2) / (norm1 * norm2))
                if d2 not in self._domain_similarity:
                    self._domain_similarity[d2] = {}
                self._domain_similarity[d1][d2] = sim
                self._domain_similarity[d2][d1] = sim

    def _attempt_cross_domain_transfer(self):
        """Transfer knowledge from high-mastery to low-mastery similar domains."""
        if not self._domain_similarity:
            return

        # Find mastery levels
        domain_mastery = {}
        for agent in self.instructors:
            domain_mastery[agent.config.domain] = (
                agent.total_correct / max(agent.total_examples, 1))

        transfers = 0
        for d_high, acc_high in domain_mastery.items():
            if acc_high < 0.75:
                continue  # Not mastered enough to transfer from
            for d_low, acc_low in domain_mastery.items():
                if d_low == d_high or acc_low > 0.40:
                    continue  # Already doing okay
                sim = self._domain_similarity.get(d_high, {}).get(d_low, 0.0)
                if sim < 0.5:
                    continue  # Not similar enough

                # H1: Transfer: put high-mastery exemplars onto per-domain
                # queue for the low-mastery domain (at reduced LR)
                for agent in self.instructors:
                    if agent.config.domain == d_high and agent.adversarial_bank:
                        # M3: Snapshot the bank to avoid data race with instructor thread
                        bank = list(agent.adversarial_bank)
                        exemplars = random.sample(bank, min(5, len(bank)))
                        # Ensure per-domain queue exists for target
                        if d_low not in self.cross_domain_queues:
                            self.cross_domain_queues[d_low] = queue.Queue(maxsize=500)
                        target_q = self.cross_domain_queues[d_low]
                        for feat, orig_lbl in exemplars:
                            try:
                                target_q.put_nowait({
                                    "target_domain": d_low,
                                    "domain": d_low,
                                    "features": feat,
                                    "label": orig_lbl,
                                    "modality": agent.config.modality,
                                })
                            except Exception:
                                break
                        transfers += 1
                        break

        if transfers > 0:
            self.logger.log(f"[School] Cross-domain transfer: {transfers} "
                            f"domain pair(s) eligible")

    def _log_domain_similarity(self):
        """Log domain similarity matrix."""
        if not self._domain_similarity:
            return
        lines = ["\n[School] Domain Similarity Matrix (top pairs):"]
        pairs = []
        seen = set()
        for d1, sims in self._domain_similarity.items():
            for d2, sim in sims.items():
                key = tuple(sorted([d1, d2]))
                if key not in seen:
                    seen.add(key)
                    pairs.append((d1, d2, sim))
        pairs.sort(key=lambda x: x[2], reverse=True)
        for d1, d2, sim in pairs[:10]:
            lines.append(f"  {d1:15s} <-> {d2:15s} : {sim:.3f}")
        self.logger.log('\n'.join(lines))

    # ------------------------------------------------------------------
    # Phase 3: Enhanced Metacognition Actions
    # ------------------------------------------------------------------

    def _metacognition_act(self, assessment: Dict[str, dict]):
        """Take real actions based on metacognitive assessment."""
        needs_recess = False
        for domain, info in assessment.items():
            stall = info.get('stall_count', 0)

            # Stall > 20: Flag consolidation + LR reset (called once after loop)
            if stall > 20:
                self.logger.log(f"[Metacognition] {domain}: stall={stall}, "
                                f"flagging consolidation + LR restart")
                needs_recess = True

            # Mastery > 0.9: reduce frequency (already handled by priority)
            if info.get('ema_accuracy', 0) > 0.9:
                # Already reducing via priority — just log
                pass

        # H1: Call recess at most once per metacognition cycle
        # H2: Only update timer when recess actually ran
        if needs_recess:
            if self._call_recess():
                self._last_recess = time.time()

    def _coordinator_loop(self):
        """Main coordinator loop — ticks every 1s."""
        self._last_config_check = time.time()

        while not self.stop_event.is_set():
            now = time.time()
            elapsed = now - self._start_time

            # Check max training time
            if elapsed >= self.config.max_training_time_s:
                self.logger.log("[School] Max training time reached — graduating")
                break

            # Rotate batches — retire finished instructors, start new ones
            self._rotate_batches()

            # Hot-reload config check (before "all finished" so new datasets
            # get queued before we decide to exit)
            if now - self._last_config_check >= self._config_check_interval_s:
                self._check_config_reload()
                self._last_config_check = now

            # Check if all instructors finished (after hot-reload so new
            # instructors are counted)
            if (all(a.is_finished for a in self.instructors)
                    and not self._pending_instructors):
                self.logger.log("[School] All instructors finished")
                break

            # Drain instructor reports
            self._drain_reports()

            # Re-capture time after drain (may have taken non-trivial time)
            now = time.time()

            # Phase 4: Update domain similarity matrix periodically
            if now - self._last_similarity_update >= self._similarity_update_interval:
                self._update_domain_centroids()
                self._log_domain_similarity()
                self._last_similarity_update = now

            # Phase 4: Cross-domain transfer check
            if now - self._last_transfer_check >= self._transfer_check_interval:
                self._attempt_cross_domain_transfer()
                self._last_transfer_check = now

            # Metacognitive assessment every 60s
            if now - self._last_metacog >= 60.0:
                assessment = self.metacognition.assess()
                if assessment:
                    self.logger.log(self.metacognition.format_dashboard())
                    # Phase 3: Take real actions based on assessment
                    self._metacognition_act(assessment)
                    stalled = self.metacognition.get_stalled_domains()
                    if stalled:
                        self.logger.log(f"[Metacognition] STALLED domains: {stalled}")
                        self.logger.log(f"[Metacognition] Recommendation: increase instructor slots for stalled domains")
                    mastered = self.metacognition.get_mastered_domains()
                    if mastered:
                        self.logger.log(f"[Metacognition] MASTERED domains: {mastered}")

                # Log brain modulation state
                try:
                    state = self.brain.ti_compute_modulation_state()
                    if state:
                        self.logger.log(
                            f"[BrainState] LR×{state.get('final_lr_factor', 1.0):.3f} "
                            f"arousal={state.get('arousal_cognitive_gain', 0):.2f} "
                            f"circadian={state.get('circadian_efficiency', 0):.2f} "
                            f"inflam={state.get('inflammation_learning_factor', 1):.2f} "
                            f"portia={state.get('portia_learning_gate', 1):.2f} "
                            f"stress={state.get('stress_level', 0):.2f} "
                            f"pause={state.get('should_pause', False)}"
                        )
                        if state.get("should_pause", False):
                            self.logger.log("[BrainState] WARNING: Brain signals training should PAUSE")
                            # H2: Only update timer when recess actually ran
                            if self._call_recess():
                                self._last_recess = now
                except Exception:
                    pass

                self._last_metacog = now

            # Recess (consolidation)
            # H2: Only update timer when recess actually ran
            if now - self._last_recess >= self.config.recess_interval_s:
                if self._call_recess():
                    self._last_recess = now

            # Dashboard
            if now - self._last_report >= self.config.report_interval_s:
                self._print_dashboard()
                self._last_report = now

            # Checkpoint
            if now - self._last_checkpoint >= self.config.checkpoint_interval_s:
                self._save_checkpoint()
                self._last_checkpoint = now

            # Graduation check
            if self._check_graduation():
                self.logger.log("[School] All domains graduated!")
                break

            time.sleep(1.0)

    def _drain_reports(self, allow_recess=True):
        """Process all pending instructor reports."""
        needs_recess = False
        while True:
            try:
                report = self.school_queue.get_nowait()
            except queue.Empty:
                break

            domain = report.get("domain", "unknown")
            self.board.update_instructor(domain, report)

            # Feed into metacognitive tracker
            self.metacognition.update(
                domain,
                report.get("accuracy", 0.0),
                report.get("loss", 1.0),
                report.get("total_examples", 0),
            )

            # Accumulate per-domain accuracy for stall detection (L7: deque)
            accuracy = report.get("accuracy", 0.0)
            if domain not in self._domain_accuracy_history:
                self._domain_accuracy_history[domain] = collections.deque(maxlen=100)
            self._domain_accuracy_history[domain].append(accuracy)

            # Stall detection: check if learning is stalled for this domain
            recent_accs = list(self._domain_accuracy_history[domain])
            if self.cognitive.is_learning_stalled(recent_accs):
                self.logger.log(
                    f"[School] Learning stalled in {domain}, "
                    f"flagging consolidation")
                needs_recess = True

            # Log to school-level JSONL
            if self._school_log_file:
                self._school_log_file.write(json.dumps(report, default=str) + "\n")
                self._school_log_file.flush()

        # H1: Call recess at most once per drain cycle
        # H2: Only update timer when recess actually ran
        if needs_recess and allow_recess:
            if self._call_recess():
                self._last_recess = time.time()

    def _call_recess(self) -> bool:
        """Full-system consolidation with multi-domain interleaved replay.

        Returns True if recess actually ran, False if skipped (too recent).
        """
        if time.time() - self._last_recess < 30.0:
            return False
        self.logger.log("\n[School] === RECESS — Memory Consolidation ===")
        recess_start = time.time()

        # Signal all instructors to pause
        self.recess_event.set()
        time.sleep(0.5)  # Let threads notice

        # C2: Acquire brain lock to ensure exclusive access during recess.
        # No instructor can be mid-learn while we replay and consolidate.
        self.brain.acquire_exclusive()
        try:
            raw_brain = self.brain.get_raw_brain()

            # Phase 4: Multi-domain interleaved replay before consolidation
            # Gather exemplars from ALL active instructors (round-robin)
            all_exemplars = []
            for agent in self.instructors:
                if agent.adversarial_bank:
                    # M3: Snapshot to avoid data race with instructor thread
                    # NOTE: CPython GIL makes list() atomic for simple lists
                    bank = list(agent.adversarial_bank)
                    samples = random.sample(bank, min(10, len(bank)))
                    for feat, lbl in samples:
                        all_exemplars.append((feat, lbl, agent.config.domain))
            if all_exemplars:
                random.shuffle(all_exemplars)
                replayed = 0
                # L5: Use brain's modulation system for replay LR (fallback to 0.3)
                replay_lr = 0.3
                try:
                    replay_lr = raw_brain.ti_compute_unified_lr(0.3)
                except Exception:
                    pass
                for feat, lbl, dom in all_exemplars[:50]:
                    # L2 fix: Check stop_event during replay loop
                    if self.stop_event.is_set():
                        break
                    try:
                        # H2 fix: Removed output_range kwarg — the C backend
                        # does not support it. Domain isolation uses string labels.
                        raw_brain.learn(feat, lbl, replay_lr)
                        replayed += 1
                    except Exception as e:
                        self.logger.log(f"[School] WARNING: Recess replay failed for {dom}: {e}")
                        continue
                self.logger.log(
                    f"[School] Interleaved replay: {replayed} examples "
                    f"from {len(set(d for _, _, d in all_exemplars))} domains")

            # Consolidation
            try:
                raw_brain.consolidate()
                self.logger.log("[School] brain.consolidate() complete")
            except Exception as e:
                self.logger.log(f"[School] consolidation error: {e}")
        finally:
            self.brain.release_exclusive()

        self.board.increment_recess()
        recess_duration = time.time() - recess_start

        # Log recess metrics
        snap = self.board.snapshot()
        self.logger.log(f"[School] Recess #{snap['recess_count']}: "
                        f"{recess_duration:.1f}s, "
                        f"total examples={snap['total_examples']:,}")
        self.logger.metric({
            "event": "recess",
            "recess_number": snap["recess_count"],
            "duration_s": round(recess_duration, 2),
            "total_examples": snap["total_examples"],
        })

        # Resume
        self.recess_event.clear()
        return True

    def _print_dashboard(self):
        """Print aggregate dashboard to log."""
        snap = self.board.snapshot()
        elapsed = time.time() - self._start_time

        lines = [
            f"\n[School] === Dashboard ({elapsed/60:.1f} min) ===",
            f"  Total examples: {snap['total_examples']:,}  |  "
            f"Instructors: {snap['num_instructors']}  |  "
            f"Recesses: {snap['recess_count']}",
            f"  {'Domain':<20s} {'Mod':>5s} {'Examples':>10s} {'Acc':>7s} "
            f"{'Mastery':>8s} {'Rate':>8s}",
            f"  {'-'*60}",
        ]
        for domain, d in sorted(snap["domains"].items()):
            lines.append(
                f"  {domain:<20s} {d['modality']:>5s} {d['examples']:>10,d} "
                f"{d['accuracy']:>7.3f} {d['mastery']:>8.3f} "
                f"{d['rate']:>7.1f}/s"
            )

        dashboard_text = "\n".join(lines)
        self.logger.log(dashboard_text)

        # JSONL metric
        self.logger.metric({
            "event": "dashboard",
            "elapsed_s": round(elapsed, 1),
            **snap,
        })

    def _save_checkpoint(self):
        """Save brain checkpoint."""
        ckpt_dir = Path(__file__).parent.parent / "checkpoints" / "athena"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = ckpt_dir / f"athena_school_{ts}.bin"
        try:
            self.brain.save(str(path))
            self.board.increment_checkpoint()
            self.logger.log(f"[School] Checkpoint saved: {path.name}")
        except Exception as e:
            self.logger.log(f"[School] Checkpoint failed: {e}")

    def _check_graduation(self) -> bool:
        """Check if all domains have reached graduation mastery.

        M4: Check ALL instructors (including finished-but-not-mastered) to
        avoid premature graduation when an instructor finished its dataset
        without reaching mastery.
        M2: Skip errored instructors — they will never reach mastery and
        should not block graduation of the remaining domains.
        """
        if not self.instructors:
            return False
        threshold = self.config.graduation_mastery
        has_non_errored = False
        for agent in self.instructors:
            # M4: Use has_error property instead of direct _error access
            # across thread boundary
            if agent.has_error:
                continue
            has_non_errored = True
            if agent.get_mastery() < threshold:
                return False
        return has_non_errored

    def _shutdown(self):
        """Graceful shutdown of all instructors."""
        self.logger.log("[School] Shutting down...")
        self.stop_event.set()
        self.recess_event.clear()

        # Wait for all threads (with timeout)
        for agent in self.instructors:
            agent.join(timeout=10.0)
            if agent.is_alive():
                self.logger.log(f"  WARNING: {agent.config.domain} thread still alive")
                # L4 fix: Use public close_log() (renamed from _close_log)
                agent.close_log()

        # Final dashboard
        self._drain_reports(allow_recess=False)
        self._print_dashboard()

        # M1: Save final checkpoint before closing log
        self._save_checkpoint()

        # Close log
        if self._school_log_file:
            self._school_log_file.close()
            self._school_log_file = None

        self.logger.log("[School] Shutdown complete")

    def _open_school_log(self):
        """Open school-level JSONL log."""
        if hasattr(self.logger, 'log_file'):
            log_dir = Path(self.logger.log_file).parent
        else:
            log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = log_dir / f"athena_training_{ts}.jsonl"
        self._school_log_path = path
        self._school_log_file = open(path, "a")

    def get_report_card(self) -> dict:
        """Get final report card for all instructors."""
        snap = self.board.snapshot()
        domain_reports = []
        for agent in self.instructors:
            r = agent.get_report()
            domain_reports.append(r)

        # Build readable report
        lines = ["Domain Report Card:", "=" * 65]
        lines.append(f"{'Domain':<20s} {'Mod':>5s} {'Examples':>10s} {'Acc':>7s} "
                      f"{'Mastery':>8s} {'Status':>10s}")
        lines.append("-" * 65)

        graduated = 0
        for r in sorted(domain_reports, key=lambda x: -x.get("mastery", 0)):
            status = "GRADUATED" if r["mastery"] >= self.config.graduation_mastery else "learning"
            if r.get("error"):
                status = "ERROR"
            elif r["mastery"] >= self.config.graduation_mastery:
                graduated += 1
            lines.append(
                f"{r['domain']:<20s} {r['modality']:>5s} {r['total_examples']:>10,d} "
                f"{r['accuracy']:>7.3f} {r['mastery']:>8.3f} {status:>10s}"
            )
        lines.append("-" * 65)
        lines.append(f"Graduated: {graduated}/{len(domain_reports)} domains")

        return {
            "num_instructors": len(self.instructors),
            "total_examples": snap["total_examples"],
            "graduated": graduated,
            "total_domains": len(domain_reports),
            "domain_reports": domain_reports,
            "domain_report": "\n".join(lines),
            "recess_count": snap["recess_count"],
            "checkpoint_count": snap["checkpoint_count"],
        }
