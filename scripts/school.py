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
    ├── cross_domain_queue               (inter-instructor exemplars)
    ├── SchoolProgressBoard              (thread-safe metrics)
    └── InstructorAgent × 23             (20 text + 3 multimodal)
"""

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

    def decide(self, *args, **kwargs):
        with self._lock:
            return self._brain.decide(*args, **kwargs)

    def __getattr__(self, name):
        """Proxy all other attributes to the underlying brain."""
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
    cross_domain_interval: int = 10_000    # Cross-domain exemplar exchange interval
    max_examples_per_dataset: int = 50_000
    startup_stagger_s: float = 2.0         # Delay between instructor starts
    num_inputs: int = 128
    num_outputs: int = 32
    max_concurrent_instructors: int = 6    # Max instructor threads starting per batch


# ---------------------------------------------------------------------------
# Thread-Safe Progress Board
# ---------------------------------------------------------------------------

class SchoolProgressBoard:
    """Thread-safe aggregate metrics across all instructors."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reports: Dict[str, dict] = {}
        self._total_examples = 0
        self._recess_count = 0
        self._checkpoint_count = 0

    def update_instructor(self, domain: str, report: dict):
        with self._lock:
            self._reports[domain] = report

    def add_examples(self, count: int):
        with self._lock:
            self._total_examples += count

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
        self.cross_domain_queue: queue.Queue = queue.Queue(maxsize=500)
        self.board = SchoolProgressBoard()

        self.instructors: List[InstructorAgent] = []
        self._start_time = 0.0
        self._last_recess = 0.0
        self._last_report = 0.0
        self._last_checkpoint = 0.0
        self._last_config_check = 0.0

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

        with open(config_path) as f:
            config_data = json.load(f)

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
            )
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
            )
            self.instructors.append(agent)

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

        # Open school-level log
        self._open_school_log()

        max_conc = self.config.max_concurrent_instructors
        self.logger.log(f"\n[School] Starting {len(self.instructors)} instructor threads "
                        f"(max {max_conc} concurrent)...")

        # Start instructors in batches to avoid heap corruption from too many
        # concurrent threads hitting the brain simultaneously
        self._pending_instructors = list(self.instructors)
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
        """Start up to `count` instructors from the pending list."""
        started = 0
        while self._pending_instructors and started < count:
            agent = self._pending_instructors.pop(0)
            agent.start()
            self._active_instructors.append(agent)
            self.logger.log(f"  [School] Started instructor: {agent.config.domain}")
            started += 1
            # Brief stagger between starts within a batch to reduce contention
            if started < count and self._pending_instructors:
                time.sleep(self.config.startup_stagger_s)

    def _rotate_batches(self):
        """Retire finished instructors and start replacements."""
        max_conc = self.config.max_concurrent_instructors
        # Remove finished threads from active list
        still_active = []
        for agent in self._active_instructors:
            if agent.is_finished or not agent.is_alive():
                self.logger.log(f"  [School] Instructor finished: {agent.config.domain}")
            else:
                still_active.append(agent)
        self._active_instructors = still_active

        # Start new instructors to fill vacant slots
        slots = max_conc - len(self._active_instructors)
        if slots > 0 and self._pending_instructors:
            self.logger.log(f"  [School] {slots} slot(s) free, "
                            f"{len(self._pending_instructors)} pending — starting next batch")
            self._start_next_batch(slots)

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
                self.logger.log(f"  [Hot-reload] {domain}: active instructor exists, "
                                f"queuing {len(ds_list)} new dataset(s) as separate instructor")

            modality = domain if domain in multimodal_domains else "text"
            ic = InstructorConfig(
                domain=domain,
                modality=modality,
                max_examples_per_dataset=self.config.max_examples_per_dataset,
                startup_delay_s=0.0,
            )
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

            # Recess (consolidation)
            if now - self._last_recess >= self.config.recess_interval_s:
                self._call_recess()
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

    def _drain_reports(self):
        """Process all pending instructor reports."""
        while True:
            try:
                report = self.school_queue.get_nowait()
            except queue.Empty:
                break

            domain = report.get("domain", "unknown")
            self.board.update_instructor(domain, report)

            # Log to school-level JSONL
            if self._school_log_file:
                self._school_log_file.write(json.dumps(report) + "\n")
                self._school_log_file.flush()

    def _call_recess(self):
        """Full-system consolidation pause."""
        self.logger.log("\n[School] === RECESS — Memory Consolidation ===")
        recess_start = time.time()

        # Signal all instructors to pause
        self.recess_event.set()
        time.sleep(0.5)  # Let threads notice

        # Consolidation
        try:
            self.brain.consolidate()
            self.logger.log("[School] brain.consolidate() complete")
        except Exception as e:
            self.logger.log(f"[School] consolidation error: {e}")

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
        ckpt_dir = Path("checkpoints") / "athena"
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
        """Check if all domains have reached graduation mastery."""
        if not self.instructors:
            return False
        threshold = self.config.graduation_mastery
        for agent in self.instructors:
            if not agent.is_finished and agent.get_mastery() < threshold:
                return False
        return True

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

        # Final dashboard
        self._drain_reports()
        self._print_dashboard()

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
