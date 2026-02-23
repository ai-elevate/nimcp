#!/usr/bin/env python3
"""
Socratic Trainer — Layer 1 of Athena's Active Learning System
==============================================================

WHAT: Predict-before-learn training with adaptive confidence and spaced repetition
WHY:  Passive learn-then-predict wastes the brain's existing knowledge; Socratic
      training makes the brain attempt answers first, then corrects with scaled
      confidence — mimicking how humans learn through testing + feedback
HOW:  1. Brain predicts (attempt)
      2. Compare with ground truth → compute adaptive confidence
      3. Brain learns with scaled confidence
      4. Mistakes enter Leitner spaced-repetition replay buffer
      5. 20% of each batch interleaved with replay reviews
"""

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SocraticConfig:
    """Configuration for Socratic training."""
    # Confidence scaling thresholds
    confidence_threshold: float = 0.7  # Brain "confident" if above this
    correct_confident_lr: float = 0.2  # Already knows — minimal reinforcement
    correct_uncertain_lr: float = 0.5  # Weak knowledge — reinforce
    wrong_confident_lr: float = 1.0    # Confidently wrong — max correction
    wrong_uncertain_lr: float = 0.8    # Uncertain + wrong — strong correction

    # Replay buffer (Leitner spaced repetition)
    replay_buffer_max: int = 5000
    replay_fraction: float = 0.2       # 20% of batch is replay
    leitner_intervals: Tuple[int, ...] = (10, 50, 200, 1000)
    leitner_max_box: int = 4           # Graduated at box 4 (removed)

    # Counter-example re-encoding on MMLU-style mistakes
    counter_example_enabled: bool = True


# ---------------------------------------------------------------------------
# Domain Mastery Tracker
# ---------------------------------------------------------------------------

class DomainMastery:
    """Per-domain rolling accuracy tracker."""

    def __init__(self, window: int = 100):
        self._window = window
        self._results: Dict[str, deque] = {}

    def record(self, domain: str, correct: bool):
        if domain not in self._results:
            self._results[domain] = deque(maxlen=self._window)
        self._results[domain].append(1 if correct else 0)

    def mastery(self, domain: str) -> float:
        if domain not in self._results or len(self._results[domain]) == 0:
            return 0.0
        return sum(self._results[domain]) / len(self._results[domain])

    def all_masteries(self) -> Dict[str, float]:
        return {d: self.mastery(d) for d in self._results}

    def total_samples(self, domain: str) -> int:
        if domain not in self._results:
            return 0
        return len(self._results[domain])


# ---------------------------------------------------------------------------
# Leitner Replay Item
# ---------------------------------------------------------------------------

@dataclass
class ReviewItem:
    """A mistake queued for spaced-repetition review."""
    features: List[float]
    label: str
    domain: str
    box: int = 0              # Leitner box (0 = most frequent review)
    next_review_step: int = 0 # Global step at which this should be reviewed
    times_reviewed: int = 0


# ---------------------------------------------------------------------------
# Socratic Trainer
# ---------------------------------------------------------------------------

class SocraticTrainer:
    """
    Predict-before-learn training with adaptive confidence and Leitner replay.

    Usage:
        trainer = SocraticTrainer(brain, SocraticConfig())
        for epoch in range(N):
            examples = [(features, label), ...]
            result = trainer.train_batch_socratic(examples, domain='mmlu')
            print(result['batch_accuracy'])
    """

    def __init__(self, brain, config: SocraticConfig = None):
        self.brain = brain
        self.config = config or SocraticConfig()
        self.mastery = DomainMastery()
        self.replay_buffer: List[ReviewItem] = []
        self.global_step = 0
        self._total_correct = 0
        self._total_seen = 0

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def train_example(self, features: List[float], label: str,
                      domain: str = "general",
                      metadata: Optional[dict] = None) -> dict:
        """
        Socratic training on a single example:
        1. Attempt (predict)
        2. Compute adaptive confidence
        3. Teach (learn with scaled confidence)
        4. Queue mistakes for replay
        """
        self.global_step += 1

        # Step 1: Attempt — let the brain try first
        pred_label, pred_conf = self.brain.predict(features)
        correct = (pred_label == label)

        # Step 2: Adaptive confidence scaling
        confidence = self._compute_confidence(correct, pred_conf)

        # Step 3: Teach with scaled confidence
        loss = self.brain.learn(features, label, confidence)

        # Step 4: Counter-example on mistakes (re-encode correct answer only)
        if not correct and self.config.counter_example_enabled:
            self.brain.learn(features, label, confidence * 0.5)

        # Step 5: Update mastery + replay buffer
        self.mastery.record(domain, correct)
        self._total_seen += 1
        if correct:
            self._total_correct += 1

        if not correct:
            self._add_to_replay(features, label, domain)

        return {
            "correct": correct,
            "predicted": pred_label,
            "pred_confidence": pred_conf,
            "learn_confidence": confidence,
            "loss": loss,
            "domain_mastery": self.mastery.mastery(domain),
        }

    def train_batch_socratic(self, examples: List[Tuple[List[float], str]],
                             domain: str = "general") -> dict:
        """
        Train on a batch with 20% replay interleaving.

        Args:
            examples: List of (features, label) tuples
            domain: Training domain name

        Returns:
            Dict with batch_accuracy, avg_loss, buffer_size, etc.
        """
        batch_start = time.time()

        # Interleave replay items (20% of batch size)
        replay_count = int(len(examples) * self.config.replay_fraction)
        replay_items = self._get_replay_items(replay_count)

        # Build combined batch: original examples + replay
        combined = [(f, l, domain, False) for f, l in examples]
        for item in replay_items:
            combined.append((item.features, item.label, item.domain, True))

        random.shuffle(combined)

        batch_correct = 0
        batch_total = 0
        total_loss = 0.0
        replay_correct = 0
        replay_total = 0

        for feats, label, dom, is_replay in combined:
            result = self.train_example(feats, label, dom)
            batch_total += 1
            if result["correct"]:
                batch_correct += 1
            if result["loss"] is not None:
                total_loss += float(result["loss"])

            if is_replay:
                replay_total += 1
                if result["correct"]:
                    replay_correct += 1

        # Update replay items based on review results
        for item in replay_items:
            # Re-check: did the brain get this right now?
            pred, _ = self.brain.predict(item.features)
            item.times_reviewed += 1
            if pred == item.label:
                # Correct review → promote to next Leitner box
                item.box = min(item.box + 1, self.config.leitner_max_box)
            else:
                # Wrong review → demote to box 0
                item.box = 0

            # Graduated items get removed
            if item.box >= self.config.leitner_max_box:
                self.replay_buffer.remove(item)
            else:
                # Schedule next review
                interval = self.config.leitner_intervals[
                    min(item.box, len(self.config.leitner_intervals) - 1)
                ]
                item.next_review_step = self.global_step + interval

        batch_acc = batch_correct / max(batch_total, 1)
        avg_loss = total_loss / max(batch_total, 1)
        elapsed = time.time() - batch_start

        return {
            "batch_accuracy": batch_acc,
            "avg_loss": avg_loss,
            "batch_size": batch_total,
            "original_size": len(examples),
            "replay_size": replay_total,
            "replay_accuracy": replay_correct / max(replay_total, 1),
            "buffer_size": len(self.replay_buffer),
            "domain_mastery": self.mastery.mastery(domain),
            "global_step": self.global_step,
            "elapsed_s": elapsed,
            "examples_per_sec": batch_total / max(elapsed, 0.001),
        }

    # ------------------------------------------------------------------
    # Confidence scaling
    # ------------------------------------------------------------------

    def _compute_confidence(self, correct: bool, pred_confidence: float) -> float:
        """Adaptive confidence scaling based on correctness and brain confidence."""
        c = self.config
        confident = pred_confidence > c.confidence_threshold

        if correct and confident:
            return c.correct_confident_lr      # Already knows — minimal
        elif correct and not confident:
            return c.correct_uncertain_lr      # Weak knowledge — reinforce
        elif not correct and confident:
            return c.wrong_confident_lr        # Confidently wrong — max correction
        else:
            return c.wrong_uncertain_lr        # Uncertain + wrong — strong correction

    # ------------------------------------------------------------------
    # Leitner replay buffer
    # ------------------------------------------------------------------

    def _add_to_replay(self, features: List[float], label: str, domain: str):
        """Add a mistake to the replay buffer."""
        if len(self.replay_buffer) >= self.config.replay_buffer_max:
            # Evict highest-box (least needy) item
            if self.replay_buffer:
                self.replay_buffer.sort(key=lambda r: -r.box)
                self.replay_buffer.pop(0)

        item = ReviewItem(
            features=features,
            label=label,
            domain=domain,
            box=0,
            next_review_step=self.global_step + self.config.leitner_intervals[0],
        )
        self.replay_buffer.append(item)

    def _get_replay_items(self, count: int) -> List[ReviewItem]:
        """Get items due for review from the replay buffer."""
        due = [r for r in self.replay_buffer
               if r.next_review_step <= self.global_step]

        if len(due) < count:
            # Supplement with lowest-box items not yet due
            not_due = [r for r in self.replay_buffer if r not in due]
            not_due.sort(key=lambda r: r.box)
            due.extend(not_due[:count - len(due)])

        random.shuffle(due)
        return due[:count]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_domain_report(self) -> str:
        """Human-readable domain mastery report."""
        masteries = self.mastery.all_masteries()
        if not masteries:
            return "No domain data yet."

        lines = ["Domain Mastery Report:"]
        for domain, acc in sorted(masteries.items(), key=lambda x: -x[1]):
            n = self.mastery.total_samples(domain)
            bar = "#" * int(acc * 20)
            lines.append(f"  {domain:20s}: {acc:.1%} ({n:4d} samples) [{bar:20s}]")

        overall = self._total_correct / max(self._total_seen, 1)
        lines.append(f"\n  Overall: {overall:.1%} ({self._total_correct}/{self._total_seen})")
        lines.append(f"  Replay buffer: {len(self.replay_buffer)} items")
        lines.append(f"  Global step: {self.global_step}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Machine-readable stats summary."""
        return {
            "global_step": self.global_step,
            "total_correct": self._total_correct,
            "total_seen": self._total_seen,
            "overall_accuracy": self._total_correct / max(self._total_seen, 1),
            "replay_buffer_size": len(self.replay_buffer),
            "domain_masteries": self.mastery.all_masteries(),
        }
