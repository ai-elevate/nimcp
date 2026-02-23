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
        self.brain = brain
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

        # Logging
        self._school_log_path: Optional[Path] = None
        self._school_log_file = None

    def setup(self, config_path: Path):
        """Load dataset config and create instructor agents."""
        with open(config_path) as f:
            config_data = json.load(f)

        datasets = config_data.get("datasets", [])

        # Group datasets by domain
        domain_datasets: Dict[str, List[dict]] = {}
        for ds in datasets:
            domain = ds.get("domain", "general")
            domain_datasets.setdefault(domain, []).append(ds)

        # Determine modality per domain
        multimodal_domains = {"audio", "visual", "speech"}

        # Create instructor agents
        stagger = 0.0
        log_dir = Path(self.logger.log_file).parent / "instructors" if hasattr(self.logger, 'log_file') else None

        for domain, ds_list in sorted(domain_datasets.items()):
            modality = domain if domain in multimodal_domains else "text"
            ic = InstructorConfig(
                domain=domain,
                modality=modality,
                max_examples_per_dataset=self.config.max_examples_per_dataset,
                startup_delay_s=stagger,
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
            stagger += self.config.startup_stagger_s

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

        self.logger.log(f"\n[School] Starting {len(self.instructors)} instructor threads...")

        # Start all instructor threads
        for agent in self.instructors:
            agent.start()

        # Run coordinator loop
        try:
            self._coordinator_loop()
        except KeyboardInterrupt:
            self.logger.log("[School] Interrupted — shutting down...")
        finally:
            self._shutdown()

    def _coordinator_loop(self):
        """Main coordinator loop — ticks every 1s."""
        while not self.stop_event.is_set():
            now = time.time()
            elapsed = now - self._start_time

            # Check max training time
            if elapsed >= self.config.max_training_time_s:
                self.logger.log("[School] Max training time reached — graduating")
                break

            # Check if all instructors finished
            if all(a.is_finished for a in self.instructors):
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
