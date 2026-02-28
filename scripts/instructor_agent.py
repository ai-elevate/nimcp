#!/usr/bin/env python3
"""
InstructorAgent — Parallel Domain-Specific Teaching Thread
===========================================================

WHAT: A threaded instructor that teaches one domain using 7 teaching methods
WHY:  Parallel domain-specialized instruction — like a school with 23 teachers
HOW:  Each InstructorAgent owns a SocraticTrainer, CognitiveOrchestrator,
      SafetyGate, StreamingDatasetProcessor, and DataSkeptic. It streams
      HuggingFace datasets and trains the shared brain using method rotation.

Teaching Methods:
  1. SOCRATIC    — Predict-before-learn, adaptive confidence
  2. CURRICULUM  — Difficulty-ordered progression
  3. CONTRASTIVE — Learn what things are NOT (negative examples)
  4. DEBATE      — Two perspectives argue, brain resolves
  5. META        — Learn HOW to learn (adapt method weights)
  6. ADVERSARIAL — Find weaknesses, generate hard examples
  7. ANALOGICAL  — Cross-domain transfer via feature blending

Modalities: text, audio, visual, speech
"""
from __future__ import annotations

import hashlib
import heapq
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

from socratic_trainer import SocraticTrainer, SocraticConfig
from cognitive_orchestrator import CognitiveOrchestrator
from safety_gate import SafetyGate, SafetyConfig

# ---------------------------------------------------------------------------
# Lazy-loaded sentence-transformer embedding model (Phase 1)
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL = None
_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_AVAILABLE = None  # None = not checked yet

# Progressive feature unfreezing (Enhancement #8) — per-instructor step tracking
# Note: Thresholds (1000/3000) are per-instructor, not global. With N concurrent
# instructors, each independently ramps from semantic to blended features.

def _get_feature_blend_ratio_for_step(step: int) -> float:
    """Progressive unfreezing: start pure semantic, blend in n-grams over time.

    Args:
        step: Per-instructor feature step count.
    """
    # Phase 1 (steps 0-1000): 100% semantic (0% n-gram)
    if step < 1000:
        return 0.0
    # Phase 2 (steps 1000-3000): Linear ramp from 0% to 50% n-gram
    elif step < 3000:
        return 0.5 * (step - 1000) / 2000.0
    # Phase 3 (steps 3000+): Steady 50/50
    else:
        return 0.5

def _get_embedding_model():
    """Lazy-load sentence-transformer model (thread-safe singleton)."""
    global _EMBEDDING_MODEL, _EMBEDDING_AVAILABLE
    if _EMBEDDING_AVAILABLE is False:
        return None
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    with _EMBEDDING_LOCK:
        if _EMBEDDING_MODEL is not None:
            return _EMBEDDING_MODEL
        if _EMBEDDING_AVAILABLE is False:
            return None
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            _EMBEDDING_AVAILABLE = True
            return _EMBEDDING_MODEL
        except Exception:
            _EMBEDDING_AVAILABLE = False
            return None

try:
    from streaming_train import StreamingDatasetProcessor, StreamConfig
    STREAMING_AVAILABLE = True
except ImportError:
    STREAMING_AVAILABLE = False

try:
    from data_skeptic import DataSkeptic, DataGrade
    SKEPTIC_AVAILABLE = True
except ImportError:
    SKEPTIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Teaching Methods
# ---------------------------------------------------------------------------

class TeachingMethod(Enum):
    SOCRATIC    = "socratic"
    CURRICULUM  = "curriculum"
    CONTRASTIVE = "contrastive"
    DEBATE      = "debate"
    META        = "meta"
    ADVERSARIAL = "adversarial"
    ANALOGICAL  = "analogical"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class InstructorConfig:
    domain: str
    modality: str = "text"  # "text", "audio", "visual", "speech"
    examples_per_method: int = 50
    report_interval: int = 10
    max_examples_per_dataset: int = 50_000
    difficulty_ramp_rate: float = 0.01
    adversarial_fraction: float = 0.15
    debate_noise_level: float = 0.1
    analogical_blend_ratio: float = 0.5
    startup_delay_s: float = 0.0
    min_domain_accuracy: float = 0.0    # Min accuracy before domain is "done" (0=disabled)
    max_retry_passes: int = 5           # Max times to re-teach datasets if below threshold
    # NOTE: output_range is kept for backward compatibility but is not used.
    # Domain isolation is handled by string label prefixing, not output neuron masking.
    output_range: Optional[Tuple[int, int]] = None


# ---------------------------------------------------------------------------
# Method Stats Tracker
# ---------------------------------------------------------------------------

class MethodStats:
    """Track effectiveness per teaching method."""

    def __init__(self):
        self._stats: Dict[str, Dict] = {}
        for m in TeachingMethod:
            self._stats[m.value] = {
                "attempts": 0, "correct": 0, "total_loss": 0.0,
            }
        # Meta-learning weights (all start equal)
        self._weights = {m.value: 1.0 for m in TeachingMethod}

    def record(self, method: str, correct: bool, loss: float = 0.0):
        s = self._stats.get(method)
        if not s:
            return
        s["attempts"] += 1
        if correct:
            s["correct"] += 1
        s["total_loss"] += loss

    def accuracy(self, method: str) -> float:
        s = self._stats.get(method, {})
        a = s.get("attempts", 0)
        return s.get("correct", 0) / max(a, 1)

    def select_method(self, epsilon: float = 0.15) -> TeachingMethod:
        """Epsilon-greedy method selection weighted by effectiveness."""
        if random.random() < epsilon:
            return random.choice(list(TeachingMethod))
        # Weighted selection by accuracy
        methods = list(TeachingMethod)
        weights = []
        for m in methods:
            acc = self.accuracy(m.value)
            w = self._weights[m.value] * (0.3 + 0.7 * acc)
            weights.append(max(w, 0.01))
        total = sum(weights)
        r = random.random() * total
        cum = 0.0
        for m, w in zip(methods, weights):
            cum += w
            if r <= cum:
                return m
        return methods[-1]

    def update_weights(self):
        """Meta-learning: adjust method weights based on recent performance delta.

        Phase 3: Methods that improve accuracy get boosted, stalled methods
        get reduced (but never zeroed — exploration floor of 0.1).
        """
        for m in TeachingMethod:
            acc = self.accuracy(m.value)
            # Blend recent accuracy with weight history
            new_w = 0.7 * self._weights[m.value] + 0.3 * (0.3 + 0.7 * acc)
            self._weights[m.value] = max(new_w, 0.1)  # exploration floor

    def summary(self) -> Dict:
        return {
            m: {"acc": round(self.accuracy(m), 4), "n": self._stats[m]["attempts"]}
            for m in self._stats
        }


# ---------------------------------------------------------------------------
# InstructorAgent Thread
# ---------------------------------------------------------------------------

class InstructorAgent(threading.Thread):
    """
    A domain instructor that runs as a thread, teaching the shared brain
    from streamed datasets using rotating teaching methods.
    """

    def __init__(self, brain, config: InstructorConfig, datasets: List[Dict],
                 school_queue, cross_domain_queue,
                 stop_event: threading.Event, recess_event: threading.Event,
                 num_inputs: int, log_dir: Optional[Path] = None,
                 cross_domain_queues: Optional[Dict] = None):
        super().__init__(name=f"Instructor-{config.domain}", daemon=True)
        self.brain = brain
        self.config = config
        self.datasets = datasets
        self.school_queue = school_queue
        self.cross_domain_queue = cross_domain_queue
        # H1: Per-domain cross-domain queues for targeted transfer
        self.cross_domain_queues = cross_domain_queues
        self.stop_event = stop_event
        self.recess_event = recess_event
        self.num_inputs = num_inputs

        # Per-instructor components
        self.socratic = SocraticTrainer(brain, SocraticConfig())
        self.cognitive = CognitiveOrchestrator(brain)
        self.safety = SafetyGate(brain, SafetyConfig())
        self.skeptic = DataSkeptic(brain) if SKEPTIC_AVAILABLE else None
        self.method_stats = MethodStats()

        # Streaming processor
        if STREAMING_AVAILABLE:
            stream_cfg = StreamConfig(
                batch_size=500,
                max_examples_per_dataset=config.max_examples_per_dataset,
            )
            self.processor = StreamingDatasetProcessor(
                brain, stream_cfg, num_inputs=num_inputs,
                hf_token=os.environ.get("HF_TOKEN"),
            )
        else:
            self.processor = None

        # Counters
        self.total_examples = 0
        self.total_correct = 0
        # M2: Separate counters for remedial teaching (excluded from primary accuracy)
        self.remedial_examples = 0
        self.remedial_correct = 0
        self.difficulty = 0.0
        self.adversarial_bank: List[Tuple[list, str]] = []
        self._adv_bank_idx = 0  # M5: rotating index for bounded adversarial bank
        self._start_time = 0.0
        self._finished = False
        self._error: Optional[str] = None

        # Rolling metrics for decision cycle (Layers 1/2/3)
        self._loss_history: deque = deque(maxlen=50)
        self._grad_history: deque = deque(maxlen=50)
        self._last_loss = 0.0
        self._last_grad_norm = 0.0
        self._last_decision: Optional[dict] = None

        # Phase 2: Cosine annealing LR scheduler (per-instructor)
        self._lr_scheduler = _CosineAnnealingLR(
            base_lr=1.0, min_lr=0.05, cycle_steps=5000, warmup_steps=500)

        # Per-instructor feature step counter (H5: replaces global counter)
        self._feature_step_count = 0

        # H3: Rolling window for recent accuracy (replaces cumulative for metacognition)
        self._recent_results: deque = deque(maxlen=500)

        # H6: LR adjustment cooldown to prevent ratcheting
        self._last_lr_adjust_step = 0
        # M4: Track last LR direction to suppress oscillation between detectors
        self._last_lr_direction = None  # "up" or "down"
        self._last_lr_direction_step = 0

        # Phase 3: Spaced repetition + difficulty tracking
        self._spaced_replay: list = []  # heap: (next_review_step, interval, tiebreak, features, label)
        self._spaced_replay_counter = 0  # L5: monotonic tiebreaker for heap
        self._example_difficulty: dict = {}  # hash -> {'fails': int, 'attempts': int, 'conf_history': deque(maxlen=10)}
        self._peak_accuracy = 0.0

        # Phase 3: Self-assessment holdout
        self._holdout_buffer: List[Tuple[list, str]] = []  # (features, label)
        self._holdout_max = 50
        self._holdout_candidates_seen = 0  # M7: separate counter for reservoir sampling
        self._last_self_assessment_step = 0
        self._self_assessment_interval = 500
        self._held_out_accuracy: float = 0.0

        # Phase 4: Domain centroid tracking (running EMA)
        self._domain_centroid: Optional[np.ndarray] = None
        self._centroid_count = 0

        # H3: Track domain-scoped prediction fallback count
        self._predict_fallback_count = 0
        self._predict_fallback_warned = False

        # Logging
        self._log_dir = log_dir
        self._log_file = None
        self._log_lock = threading.Lock()

    def _open_log(self):
        self._log_write_count = 0  # L4: counter for periodic flush
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = self._log_dir / f"{self.config.domain}_{ts}.jsonl"
            self._log_file = open(path, "a")

    def _log_example(self, data: dict):
        if self._log_file:
            data["ts"] = time.time()
            data["domain"] = self.config.domain
            with self._log_lock:
                self._log_file.write(json.dumps(data) + "\n")
                self._log_write_count += 1
                # L4: Periodic flush every 100 writes for crash safety
                if self._log_write_count % 100 == 0:
                    self._log_file.flush()

    def close_log(self):
        """Close the instructor's log file.

        L4 fix: Renamed from _close_log to public — called from School._shutdown()
        across class boundary.
        """
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def run(self):
        """Main instructor loop — repeats until accuracy threshold or max passes."""
        self._start_time = time.time()
        self._open_log()
        try:
            # Staggered startup
            if self.config.startup_delay_s > 0:
                time.sleep(self.config.startup_delay_s)

            threshold = self.config.min_domain_accuracy
            max_passes = self.config.max_retry_passes if threshold > 0 else 1

            for pass_num in range(max_passes):
                if self.stop_event.is_set():
                    break

                # Teach each dataset
                for ds_config in self.datasets:
                    if self.stop_event.is_set():
                        break
                    self._teach_dataset(ds_config)

                # Check accuracy after this pass
                current_acc = self.total_correct / max(self.total_examples, 1)
                if threshold > 0 and current_acc >= threshold:
                    self._log_example({
                        "action": "GRADUATED",
                        "pass": pass_num + 1,
                        "accuracy": round(current_acc, 4),
                        "threshold": threshold,
                    })
                    break

                if threshold > 0 and pass_num < max_passes - 1:
                    self._log_example({
                        "action": "RETRY_PASS",
                        "pass": pass_num + 1,
                        "accuracy": round(current_acc, 4),
                        "threshold": threshold,
                        "reason": f"accuracy {current_acc:.1%} < threshold {threshold:.1%}",
                    })

            # Remedial: re-teach hard items from replay buffer
            self._remedial_teaching()

        except Exception as e:
            self._error = str(e)
        finally:
            self._finished = True
            self.close_log()
            # Send final report
            self._send_report(final=True)

    def _teach_dataset(self, ds_config: dict):
        """Stream and teach one dataset."""
        name = ds_config.get("name", "unknown")
        domain = self.config.domain
        modality = self.config.modality
        source_name = ds_config.get("hf_dataset", name)

        if modality == "text" and self.processor:
            self._teach_text_dataset(ds_config, name, domain, source_name)
        elif modality in ("audio", "visual", "speech"):
            self._teach_multimodal_dataset(ds_config, name, domain, modality, source_name)
        else:
            self._teach_text_fallback(ds_config, name, domain, source_name)

    def _teach_text_dataset(self, ds_config: dict, name: str, domain: str,
                            source_name: str):
        """Teach a text dataset using StreamingDatasetProcessor."""
        API_TYPES = {"wikipedia", "arxiv", "stackexchange", "pubmed",
                     "gutenberg", "conceptnet", "news_rss"}
        try:
            ds_type = ds_config.get("type", "")
            if ds_type == "local":
                dataset = self.processor.load_local_dataset(ds_config)
            elif ds_type in API_TYPES:
                dataset = self.processor.load_api_stream(ds_config)
            else:
                dataset = self.processor.load_streaming_dataset(ds_config)
            if dataset is None:
                return
        except Exception as e:
            self._log_example({"action": "DATASET_ERROR", "dataset": name,
                               "error": f"[{self.config.domain}] Dataset load error: {e}"})
            return

        count = 0
        for example in dataset:
            if self.stop_event.is_set():
                break
            self._wait_for_recess()

            result = self.processor.extract_features_and_label(example, domain)
            if result is None:
                continue

            _, label = result
            # Domain-prefix label to prevent cross-domain collision
            label = f"{domain}:{label}"

            # C1: Use hybrid features instead of pure n-gram from processor
            text = self._extract_text(example)
            if not text or not text.strip():
                continue
            self._feature_step_count += 1
            features = _text_to_features(text, self.num_inputs, self._feature_step_count)

            # Grade data quality
            grade = self._grade_example(text, domain, source_name)

            # Select and execute teaching method
            method = self.method_stats.select_method()
            correct, loss = self._execute_method(method, features, str(label),
                                                  domain, grade)

            # Record
            self.method_stats.record(method.value, correct, loss)
            self.total_examples += 1
            if correct:
                self.total_correct += 1
            count += 1

            # Log
            self._log_example({
                "action": "LEARN", "method": method.value,
                "correct": correct, "loss": round(loss, 5),
                "confidence_mod": round(grade.confidence_modifier, 3) if grade else 1.0,
                "ethics_label": grade.ethics_label if grade else "neutral",
                "dataset": name, "features_dim": len(features),
            })

            # Periodic report + meta-learning update
            if count % self.config.report_interval == 0:
                self.method_stats.update_weights()
                self._send_report()

            # Difficulty ramp
            self.difficulty = min(1.0, self.difficulty + self.config.difficulty_ramp_rate / 1000)

            if count >= self.config.max_examples_per_dataset:
                break

        # Publish a cross-domain exemplar
        self._publish_exemplar(domain)

    def _teach_multimodal_dataset(self, ds_config: dict, name: str, domain: str,
                                   modality: str, source_name: str):
        """Teach an audio/visual/speech dataset using cortex bindings."""
        try:
            from datasets import load_dataset
            hf_dataset = ds_config.get("hf_dataset", "")
            hf_subset = ds_config.get("hf_subset")
            kwargs = {"split": "train", "streaming": True}
            if os.environ.get("HF_TOKEN"):
                kwargs["token"] = os.environ["HF_TOKEN"]
            if hf_subset:
                dataset = load_dataset(hf_dataset, hf_subset, **kwargs)
            else:
                dataset = load_dataset(hf_dataset, **kwargs)
        except Exception as e:
            self._log_example({"action": "DATASET_ERROR", "dataset": name,
                               "error": f"[{self.config.domain}] Dataset load error: {e}"})
            return

        count = 0
        for example in dataset:
            if self.stop_event.is_set():
                break
            self._wait_for_recess()

            features, label = self._extract_multimodal(example, modality, domain)
            if features is None:
                continue
            # Domain-prefix label to prevent cross-domain collision
            label = f"{domain}:{label}"

            grade = self._grade_example("", domain, source_name)
            method = self.method_stats.select_method()
            correct, loss = self._execute_method(method, features, str(label),
                                                  domain, grade)

            self.method_stats.record(method.value, correct, loss)
            self.total_examples += 1
            if correct:
                self.total_correct += 1
            count += 1

            self._log_example({
                "action": "LEARN", "method": method.value, "modality": modality,
                "correct": correct, "loss": round(loss, 5), "dataset": name,
                "features_dim": len(features),
            })

            if count % self.config.report_interval == 0:
                self.method_stats.update_weights()
                self._send_report()

            # Difficulty ramp
            self.difficulty = min(1.0, self.difficulty + self.config.difficulty_ramp_rate / 1000)

            if count >= self.config.max_examples_per_dataset:
                break

        # L3: Publish cross-domain exemplar for multimodal datasets
        self._publish_exemplar(domain)

    def _teach_text_fallback(self, ds_config: dict, name: str, domain: str,
                              source_name: str):
        """Fallback text teaching without StreamingDatasetProcessor."""
        try:
            from datasets import load_dataset
            hf_dataset = ds_config.get("hf_dataset", "")
            hf_subset = ds_config.get("hf_subset")
            kwargs = {"split": "train", "streaming": True}
            if os.environ.get("HF_TOKEN"):
                kwargs["token"] = os.environ["HF_TOKEN"]
            if hf_subset:
                dataset = load_dataset(hf_dataset, hf_subset, **kwargs)
            else:
                dataset = load_dataset(hf_dataset, **kwargs)
        except Exception as e:
            self._log_example({"action": "DATASET_ERROR", "dataset": name,
                               "error": f"[{self.config.domain}] Dataset load error: {e}"})
            return

        count = 0
        for example in dataset:
            if self.stop_event.is_set():
                break
            self._wait_for_recess()

            text = self._extract_text(example)
            if not text or not text.strip():
                continue

            self._feature_step_count += 1
            features = _text_to_features(text, self.num_inputs, self._feature_step_count)
            # Domain-prefix label to prevent cross-domain collision
            label_val = f"{domain}:{example.get('answer', example.get('label', 0))}"

            grade = self._grade_example(text, domain, source_name)
            method = self.method_stats.select_method()
            correct, loss = self._execute_method(method, features, str(label_val),
                                                  domain, grade)

            self.method_stats.record(method.value, correct, loss)
            self.total_examples += 1
            if correct:
                self.total_correct += 1
            count += 1

            if count % self.config.report_interval == 0:
                self.method_stats.update_weights()
                self._send_report()

            # Difficulty ramp
            self.difficulty = min(1.0, self.difficulty + self.config.difficulty_ramp_rate / 1000)

            if count >= self.config.max_examples_per_dataset:
                break

        # L3: Publish cross-domain exemplar for fallback text datasets
        self._publish_exemplar(domain)

    # --- Brain-State LR Modulation ---

    def _modulate_lr(self, base_lr: float) -> float:
        """Modulate learning rate through cosine schedule + decision cycle.

        Pipeline:
          1. Cosine annealing with warm restarts (per-instructor)
          2. Decision cycle lr_factor (Layers 1/2/3) multiplied on top
          3. Fallback to unified 8-factor pipeline if no decision
        """
        # Step 1: Cosine annealing schedule
        schedule_factor = self._lr_scheduler.get_lr()

        # Step 2: Decision cycle lr_factor if available
        if self._last_decision is not None:
            dc_factor = self._last_decision.get("lr_factor", 1.0)
            result = base_lr * schedule_factor * dc_factor
            # H4: Final clamp to prevent unbounded compounding
            return max(0.01, min(result, 3.0))

        # Step 3: Fallback to unified pipeline
        # M3: Normalize fallback path — extract factor from adaptive result so
        # magnitude is comparable to DC path (base_lr * schedule * factor)
        try:
            adaptive = float(self.cognitive.compute_adaptive_lr(base_lr))
            # adaptive already incorporates base_lr; extract the factor
            fallback_factor = adaptive / max(base_lr, 1e-6)
            result = base_lr * schedule_factor * fallback_factor
        except Exception:
            result = base_lr * schedule_factor
        # H4: Final clamp to prevent unbounded compounding
        return max(0.01, min(result, 3.0))

    def _update_metrics_and_decide(self, loss: float):
        """Track rolling metrics and run decision cycle at report intervals.

        Called after each learn(). Captures gradient norm from C, accumulates
        loss/gradient history, and runs the full decision cycle (observe ->
        diagnose -> simulate -> decide) every report_interval examples.
        """
        # Capture gradient norm from C backprop
        grad_norm = self.cognitive.get_last_gradient_norm()
        if grad_norm < 0:
            grad_norm = 0.0

        self._loss_history.append(loss)
        self._grad_history.append(grad_norm)
        self._last_loss = loss
        self._last_grad_norm = grad_norm

        # Run decision cycle every report_interval
        if self.total_examples > 0 and self.total_examples % self.config.report_interval == 0:
            self._run_decision_cycle()

    def _run_decision_cycle(self):
        """Run the full Layer 1/2/3 decision cycle."""
        losses = list(self._loss_history)
        grads = list(self._grad_history)
        if len(losses) < 2 or len(grads) < 2:
            return

        loss_current = losses[-1]
        loss_previous = losses[-2]
        grad_norm = grads[-1]
        grad_norm_previous = grads[-2]

        # Compute volatility and variance from rolling window
        mean_loss = sum(losses) / len(losses)
        loss_volatility = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5

        mean_grad = sum(grads) / len(grads)
        gradient_std = (sum((g - mean_grad) ** 2 for g in grads) / len(grads)) ** 0.5

        # Get current LR from the unified pipeline (use actual base_lr, not hardcoded)
        current_lr = self.cognitive.compute_adaptive_lr(self._lr_scheduler.base_lr)

        decision = self.cognitive.compute_decision_cycle(
            loss_current, loss_previous,
            grad_norm, grad_norm_previous,
            loss_volatility, gradient_std,
            current_lr, 32.0)

        if decision is not None:
            self._last_decision = decision

    # --- Teaching Methods ---

    def _predict_domain(self, features, domain: str):
        """Domain-scoped prediction — only considers labels from this domain.

        C2 fix: All return paths sanitize confidence against NaN/Inf.
        """
        prefix = f"{domain}:"
        try:
            pred, conf = self.brain.predict_in_domain(features, prefix)
            if math.isnan(conf) or math.isinf(conf):
                conf = 0.0
            return pred, conf
        except (AttributeError, TypeError):
            # H3: Fallback silently degrades domain scoping — log warning once
            self._predict_fallback_count += 1
            if not self._predict_fallback_warned:
                self._predict_fallback_warned = True
                logger.warning(
                    "[%s] predict_in_domain unavailable, falling back to "
                    "predict_fast (domain scoping disabled)", domain)
            pred, conf = self.brain.predict_fast(features)
            if math.isnan(conf) or math.isinf(conf):
                conf = 0.0
            return pred, conf
        except Exception:
            # M8: Catch any other prediction errors with debug log
            logger.debug("[%s] _predict_domain unexpected error", domain, exc_info=True)
            raise

    def _scope_target_to_domain(self, target):
        """Pass-through for domain-scoped targets.

        H1/H4 fix: Domain isolation is handled entirely by string label
        prefixing (e.g. "biology:answer_a"), NOT by vector-based output
        neuron masking. The C backend does not support output_range kwargs.
        The old vector-masking code was dead (all targets are strings).
        """
        return target

    def _ensemble_predict(self, features, domain: str, k=3, noise_sigma=0.01):
        """Run K forward passes with input perturbation, return majority-vote prediction.

        Used during self-assessment and evaluation for more stable accuracy measurement.
        Not used during training (too slow).
        """
        votes = {}   # label -> count
        confidences = []  # confidence per pass
        features_arr = np.array(features, dtype=np.float32)

        for i in range(k):
            if i == 0:
                # First pass: original features (no noise)
                query = features
            else:
                # Subsequent passes: add Gaussian noise
                noisy = features_arr + np.random.normal(
                    0, noise_sigma, size=features_arr.shape
                ).astype(np.float32)
                # Re-normalize to maintain unit norm
                norm = np.linalg.norm(noisy)
                if norm > 1e-6:
                    noisy /= norm
                query = noisy.tolist()

            try:
                pred, conf = self._predict_domain(query, domain)
                votes[pred] = votes.get(pred, 0) + 1
                confidences.append(conf)
            except Exception:
                continue

        if not votes:
            return None, 0.0, 0.0

        # Majority vote
        best_label = max(votes, key=votes.get)
        agreement = votes[best_label] / k
        avg_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        return best_label, avg_confidence * agreement, agreement

    def _execute_method(self, method: TeachingMethod, features: list,
                        label: str, domain: str,
                        grade: Optional[DataGrade]) -> Tuple[bool, float]:
        """Execute a teaching method with dynamic learning hooks."""
        conf_mod = grade.confidence_modifier if grade else 1.0

        # C4: Capture pre-prediction BEFORE learning so confidence reflects
        # the brain's state prior to this example's training signal.
        try:
            pre_pred_label, pre_conf = self._predict_domain(features, domain)
            # C2 fix: Guard against NaN/Inf confidence from predict
            if math.isnan(pre_conf) or math.isinf(pre_conf):
                pre_conf = 0.0
            pre_correct = (pre_pred_label == label)
        except Exception:
            pre_pred_label = None
            pre_conf = 0.5
            pre_correct = False

        # Phase 3: Collect holdout samples for self-assessment
        # H1: Holdout examples are NOT trained on — keeps self-assessment unbiased
        # M3: Scheduler must NOT advance on holdout examples (moved step() below)
        if random.random() < 0.02:  # 2% holdout rate
            self._maybe_collect_holdout(features, label)
            return pre_correct, 0.0  # holdout: use actual prediction correctness

        # H3: Step scheduler once per example (not inside get_lr)
        # M3: Only step after holdout check so holdout examples don't advance schedule
        self._lr_scheduler.step()

        # Enhancement #6: Difficulty-scaled LR modifier (confidence trajectory)
        diff_scale = self._get_difficulty_lr_scale(features, label)
        conf_mod *= diff_scale

        # Enhancement #3: Focal loss — scale LR by prediction uncertainty
        # M1: Reuse pre-prediction instead of making a redundant predict_fast call
        focal_factor = self._compute_focal_factor(features, label,
                                                  pre_confidence=pre_conf,
                                                  pre_correct=pre_correct)
        conf_mod *= focal_factor

        # Phase 3: Process any due spaced-replay examples
        self._spaced_replay_tick()

        # Phase 4: Update domain centroid
        self._update_domain_centroid(features)

        if method == TeachingMethod.SOCRATIC:
            result = self._method_socratic(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        elif method == TeachingMethod.CURRICULUM:
            result = self._method_curriculum(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        elif method == TeachingMethod.CONTRASTIVE:
            result = self._method_contrastive(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        elif method == TeachingMethod.DEBATE:
            result = self._method_debate(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        elif method == TeachingMethod.META:
            result = self._method_meta(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        elif method == TeachingMethod.ADVERSARIAL:
            result = self._method_adversarial(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        elif method == TeachingMethod.ANALOGICAL:
            result = self._method_analogical(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)
        else:
            result = self._method_socratic(features, label, domain, conf_mod, pre_pred=pre_pred_label, pre_conf=pre_conf)

        # Enhancement #6: Record difficulty with confidence trajectory
        correct, loss = result
        fh = self._feature_hash(features)
        if fh not in self._example_difficulty:
            self._example_difficulty[fh] = {
                'fails': 0, 'attempts': 0, 'conf_history': deque(maxlen=10)}
        info = self._example_difficulty[fh]
        info['attempts'] += 1
        if not correct:
            info['fails'] += 1
        # C4: Use pre-prediction confidence captured before learning
        info['conf_history'].append(pre_conf)

        # H4: Evict oldest entries to prevent unbounded growth
        # M5: Skip entries that are actively in the replay heap
        if len(self._example_difficulty) > 50000:
            replay_hashes = set()
            for entry in self._spaced_replay:
                replay_hashes.add(self._feature_hash(entry[3]))  # index 3 = features
            keys_to_remove = [
                k for k in list(self._example_difficulty.keys())
                if k not in replay_hashes
            ][:10000]
            for k in keys_to_remove:
                del self._example_difficulty[k]

        # H3: Track recent results for rolling accuracy
        self._recent_results.append(correct)

        return result

    def _method_socratic(self, features, label, domain, conf_mod,
                         pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Predict-before-learn with adaptive confidence + spaced replay."""
        if pre_pred is not None:
            pred, conf = pre_pred, pre_conf
        else:
            pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Phase 3: Skip easy examples at high mastery
        if self._should_skip_easy(features, label, conf) and correct:
            self.socratic.mastery.record(domain, correct)
            return correct, 0.0

        # Adaptive confidence scaling
        if correct and conf > 0.7:
            lr = 0.2 * conf_mod
        elif correct:
            lr = 0.5 * conf_mod
        elif conf > 0.7:
            lr = 1.0 * conf_mod  # Confidently wrong — max correction
        else:
            lr = 0.8 * conf_mod

        self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(lr))
        # M11: Loss is a heuristic proxy, not the actual network loss.
        # It serves as a directional signal for decision cycle and logging, but does
        # not reflect true cross-entropy or MSE from the C backprop path.
        # C1 fix: Use confidence directly as loss when wrong — higher confidence
        # when wrong means higher loss (cross-entropy-like behavior).
        loss = 0.0 if correct else max(0.1, conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)

        # Phase 3: Schedule failed examples for spaced repetition
        if not correct:
            self._spaced_replay_push(features, label)

        return correct, loss

    def _method_curriculum(self, features, label, domain, conf_mod,
                           pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Difficulty-ordered: skip easy examples as difficulty ramps."""
        if pre_pred is not None:
            pred, conf = pre_pred, pre_conf
        else:
            pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Skip if brain already knows AND difficulty is still low
        if correct and conf > 0.9 and self.difficulty < 0.5:
            return correct, 0.0

        lr = (0.5 + 0.5 * self.difficulty) * conf_mod
        self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(lr))
        # C1 fix: confident-wrong = high loss
        loss = 0.0 if correct else max(0.1, conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_contrastive(self, features, label, domain, conf_mod,
                            pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Learn what things are NOT — teach with negative examples.

        M6: Actually perform contrastive learning by teaching a negative label
        at reduced LR so the brain learns to discriminate.
        """
        if pre_pred is not None:
            pred, conf = pre_pred, pre_conf
        else:
            pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # M1: Teach positive example with boosted LR instead of polluting label
        # space with "NOT_" negative labels that persist in the brain's memory
        scoped_label = self._scope_target_to_domain(label)
        boost = 1.2 if not correct else 1.0  # Extra boost for incorrect predictions
        self.brain.learn(features, scoped_label, self._modulate_lr(0.7 * conf_mod * boost))

        # M6: Update metrics after the positive learn (not after a negative learn)
        # C1 fix: confident-wrong = high loss
        loss = 0.0 if correct else max(0.1, conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_debate(self, features, label, domain, conf_mod,
                       pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Two perspectives (original + noisy) argue, brain resolves."""
        # Perspective 1: original features
        if pre_pred is not None:
            pred1, conf1 = pre_pred, pre_conf
        else:
            pred1, conf1 = self._predict_domain(features, domain)

        # Perspective 2: perturbed features (different viewpoint)
        noise_level = self.config.debate_noise_level
        noisy = [f + random.gauss(0, noise_level) for f in features]
        # L2: Re-normalize after noise addition to maintain unit norm
        noisy_norm = math.sqrt(sum(v * v for v in noisy))
        if noisy_norm > 1e-6:
            noisy = [v / noisy_norm for v in noisy]
        # H1: Guard against prediction failure on noisy features
        try:
            pred2, conf2 = self._predict_domain(noisy, domain)
        except Exception:
            pred2, conf2 = pred1, conf1  # Fall back to perspective 1

        # Brain resolves: if both agree, lower confidence. If disagree, higher.
        if pred1 == pred2:
            lr = 0.5 * conf_mod  # consensus → moderate
        else:
            lr = 0.9 * conf_mod  # disagreement → stronger teaching

        correct = (pred1 == label)
        self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(lr))

        # C1 fix: confident-wrong = high loss (use max confidence of the two perspectives)
        loss = 0.0 if correct else max(0.1, max(conf1, conf2))
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_meta(self, features, label, domain, conf_mod,
                     pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Meta-learning: adapt method weights based on performance."""
        if pre_pred is not None:
            pred, conf = pre_pred, pre_conf
        else:
            pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Use mastery to set learning rate
        mastery = self.socratic.mastery.mastery(domain)
        lr = (0.3 + 0.7 * (1.0 - mastery)) * conf_mod
        self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(lr))

        # C1 fix: confident-wrong = high loss
        loss = 0.0 if correct else max(0.1, conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_adversarial(self, features, label, domain, conf_mod,
                            pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Adversarial training: find minimal perturbation that flips prediction, then train on it.

        WARNING: ~40x more expensive than socratic method due to binary search
        over multiple random directions (5 directions x 8 binary search steps =
        40 forward passes). Use adversarial_fraction config to limit frequency.
        """
        features_arr = np.array(features, dtype=np.float32)

        # Step 1: Get current prediction
        if pre_pred is not None:
            pred, conf = pre_pred, pre_conf
        else:
            pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Phase 3: Skip easy examples at high mastery
        if self._should_skip_easy(features, label, conf) and correct:
            self.socratic.mastery.record(domain, correct)
            return correct, 0.0

        # Store hard examples in adversarial_bank AND spaced replay
        if not correct or conf < 0.5:
            # M5: Rotate oldest entry when bank is full instead of refusing
            if len(self.adversarial_bank) < 500:
                self.adversarial_bank.append((features, label))
            else:
                self.adversarial_bank[self._adv_bank_idx % 500] = (features, label)
            self._adv_bank_idx += 1
            self._spaced_replay_push(features, label)

        # Step 2: Binary search for decision boundary
        # Try multiple random directions to find adversarial examples
        best_adversarial = None
        best_epsilon = float('inf')
        num_directions = 5

        for _ in range(num_directions):
            # Random unit direction
            direction = np.random.randn(len(features_arr)).astype(np.float32)
            direction /= (np.linalg.norm(direction) + 1e-8)

            # Binary search for minimal epsilon that flips prediction
            lo, hi = 0.0, 0.3
            flip_found = False

            for _ in range(8):  # 8 binary search steps
                mid = (lo + hi) / 2.0
                perturbed = features_arr + mid * direction
                norm = np.linalg.norm(perturbed)
                if norm > 1e-6:
                    perturbed /= norm

                try:
                    pert_pred, _ = self._predict_domain(
                        perturbed.tolist(), domain
                    )
                except Exception:
                    break

                if pert_pred != pred:
                    hi = mid
                    flip_found = True
                else:
                    lo = mid

            if flip_found and hi < best_epsilon:
                best_epsilon = hi
                best_adversarial = features_arr + hi * direction
                norm = np.linalg.norm(best_adversarial)
                if norm > 1e-6:
                    best_adversarial /= norm

        # Step 3: Train on original example
        lr = (0.9 if not correct else 0.4) * conf_mod
        scoped_label = self._scope_target_to_domain(label)
        self.brain.learn(features, scoped_label, self._modulate_lr(lr))

        # Step 4: Train on adversarial example (if found) with correct label
        if best_adversarial is not None:
            try:
                # Boost LR for boundary examples — most informative
                self.brain.learn(
                    best_adversarial.tolist(), scoped_label,
                    self._modulate_lr(lr * 1.5)
                )
                # Push to spaced replay — adversarial examples are high-value
                self._spaced_replay_push(best_adversarial.tolist(), label)
            except Exception as e:
                logger.debug(f"Learn failed: {e}")
        else:
            # No adversarial found — fallback: add random noise
            noise = np.random.normal(
                0, 0.05, size=features_arr.shape
            ).astype(np.float32)
            noisy_features = features_arr + noise
            norm = np.linalg.norm(noisy_features)
            if norm > 1e-6:
                noisy_features /= norm
            try:
                self.brain.learn(
                    noisy_features.tolist(), scoped_label, self._modulate_lr(lr)
                )
            except Exception as e:
                logger.debug(f"Learn failed: {e}")

        # C1 fix: confident-wrong = high loss
        loss = 0.0 if correct else max(0.1, conf)
        self._update_metrics_and_decide(loss)

        # Periodically re-teach hard examples from adversarial bank
        # L6: Call _update_metrics_and_decide for bank re-teaching too
        # H1 fix: Re-predict to get fresh confidence, remove from bank if now correct
        if (self.total_examples % 200 == 0 and self.adversarial_bank
                and random.random() < self.config.adversarial_fraction):
            bank_idx = random.randrange(len(self.adversarial_bank))
            hard_feat, hard_label = self.adversarial_bank[bank_idx]
            try:
                bank_pred, bank_conf = self._predict_domain(hard_feat, domain)
                bank_correct = (bank_pred == hard_label)
            except Exception:
                bank_correct = False
                bank_conf = 0.5
            if bank_correct and bank_conf > 0.8:
                # Brain now gets this right with high confidence — remove from bank
                self.adversarial_bank.pop(bank_idx)
            else:
                bank_loss = 0.0 if bank_correct else max(0.1, bank_conf)
                self.brain.learn(
                    hard_feat, self._scope_target_to_domain(hard_label),
                    self._modulate_lr(0.8 * conf_mod)
                )
                self._update_metrics_and_decide(bank_loss)
                # H2 fix: Include bank replay result in rolling metrics
                self._recent_results.append(bank_correct)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_analogical(self, features, label, domain, conf_mod,
                           pre_pred=None, pre_conf=None) -> Tuple[bool, float]:
        """Cross-domain transfer via blending with cross-domain exemplar."""
        if pre_pred is not None:
            pred, conf = pre_pred, pre_conf
        else:
            pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(0.7 * conf_mod))

        # H1: Try to blend with cross-domain exemplar from per-domain queue
        # (with fallback to shared queue for backward compat)
        # L5 note: exemplar["domain"] = source domain (who published it),
        # while per-domain queue key = target domain (who should consume it).
        domain_q = (self.cross_domain_queues.get(self.config.domain)
                     if self.cross_domain_queues else None)
        source_q = domain_q or self.cross_domain_queue
        try:
            exemplar = source_q.get_nowait()
            if exemplar.get("modality", "text") == self.config.modality:
                ex_feats = exemplar.get("features", [])
                if len(ex_feats) == len(features):
                    ratio = self.config.analogical_blend_ratio
                    blended = [
                        f * (1 - ratio) + e * ratio
                        for f, e in zip(features, ex_feats)
                    ]
                    self.brain.learn(blended, self._scope_target_to_domain(label), self._modulate_lr(0.4 * conf_mod))
                else:
                    # M1 fix: Dimension mismatch — re-queue with bounce_count,
                    # discard if bounced too many times to prevent infinite cycling
                    bounce = exemplar.get("bounce_count", 0) + 1
                    if bounce <= 3:
                        exemplar["bounce_count"] = bounce
                        source_q.put_nowait(exemplar)
                    # else: silently discard — bounced too many times
            else:
                # M1 fix: Modality mismatch — re-queue with bounce_count
                bounce = exemplar.get("bounce_count", 0) + 1
                if bounce <= 3:
                    exemplar["bounce_count"] = bounce
                    source_q.put_nowait(exemplar)
                # else: silently discard — bounced too many times
        except Exception as e:
            logger.debug(f"Learn failed: {e}")

        # C1 fix: confident-wrong = high loss
        loss = 0.0 if correct else max(0.1, conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    # --- Multimodal Feature Extraction ---

    def _extract_multimodal(self, example: dict, modality: str,
                            domain: str) -> Tuple[Optional[list], str]:
        """Extract features from audio/visual/speech examples."""
        label = str(example.get("label", example.get("classID", 0)))

        if modality == "audio":
            return self._extract_audio(example, label)
        elif modality == "visual":
            return self._extract_visual(example, label)
        elif modality == "speech":
            return self._extract_speech(example, label)
        return None, label

    def _extract_audio(self, example: dict, label: str) -> Tuple[Optional[list], str]:
        """Extract audio features via audio cortex or fallback."""
        audio = example.get("audio")
        if audio and isinstance(audio, dict):
            samples = audio.get("array", [])
            if isinstance(samples, (list, np.ndarray)) and len(samples) > 0:
                try:
                    features = self.brain.audio_cortex_process(samples)
                    return _pad_or_truncate(features, self.num_inputs), label
                except (AttributeError, Exception):
                    pass
            # Fallback: basic feature extraction from samples
            samples_list = list(samples) if hasattr(samples, '__iter__') else []
            if samples_list:
                return _audio_fallback_features(samples_list, self.num_inputs), label
        return None, label

    def _extract_visual(self, example: dict, label: str) -> Tuple[Optional[list], str]:
        """Extract visual features via visual cortex or fallback."""
        img = example.get("image") or example.get("img")
        if img is not None:
            try:
                # PIL Image → pixel array
                if hasattr(img, 'size'):
                    w, h = img.size
                    channels = len(img.getbands()) if hasattr(img, 'getbands') else 3
                    pixels = list(img.getdata())
                    flat = []
                    for p in pixels:
                        if isinstance(p, (tuple, list)):
                            flat.extend(p)
                        else:
                            flat.append(p)
                    # Normalize to [0, 1]
                    flat = [float(v) / 255.0 for v in flat]
                    try:
                        features = self.brain.visual_cortex_process(flat, w, h, channels)
                        return _pad_or_truncate(features, self.num_inputs), label
                    except (AttributeError, Exception):
                        pass
                    return _visual_fallback_features(flat, w, h, self.num_inputs), label
            except Exception:
                pass
        return None, label

    def _extract_speech(self, example: dict, label: str) -> Tuple[Optional[list], str]:
        """Extract speech features via speech cortex or fallback."""
        audio = example.get("audio")
        if audio and isinstance(audio, dict):
            samples = audio.get("array", [])
            if isinstance(samples, (list, np.ndarray)) and len(samples) > 0:
                try:
                    features = self.brain.speech_cortex_process(samples)
                    return _pad_or_truncate(features, self.num_inputs), label
                except (AttributeError, Exception):
                    pass
            samples_list = list(samples) if hasattr(samples, '__iter__') else []
            if samples_list:
                return _audio_fallback_features(samples_list, self.num_inputs), label
        # Fallback: treat text transcription as text
        text = example.get("text", example.get("transcription", ""))
        if text:
            self._feature_step_count += 1
            return _text_to_features(str(text), self.num_inputs, self._feature_step_count), label
        return None, label

    # --- Helpers ---

    def _extract_text(self, example: dict) -> str:
        """Extract raw text from a HuggingFace example."""
        for key in ("text", "question", "content", "input", "ctx", "sentence"):
            if key in example and example[key]:
                return str(example[key])
        # Concatenate all string values
        parts = [str(v) for v in example.values() if isinstance(v, str) and v.strip()]
        return " ".join(parts)

    def _grade_example(self, text: str, domain: str,
                       source_name: str) -> Optional[DataGrade]:
        """Grade example quality if DataSkeptic is available."""
        if not self.skeptic or not text:
            return None
        return self.skeptic.grade(text, domain, source_name)

    def _wait_for_recess(self):
        """Block while recess is active."""
        while self.recess_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.1)

    def _remedial_teaching(self):
        """Re-teach hard items from adversarial bank.

        M2: Remedial results are tracked in separate counters (remedial_examples,
        remedial_correct) so they don't inflate the primary accuracy metric.
        """
        if not self.adversarial_bank:
            return
        domain = self.config.domain
        random.shuffle(self.adversarial_bank)
        for features, label in self.adversarial_bank[:200]:
            if self.stop_event.is_set():
                break
            # H2: Guard prediction against failure
            try:
                pred, conf = self._predict_domain(features, domain)
            except Exception:
                continue
            correct = (pred == label)
            focal = self._compute_focal_factor(features, label,
                                               pre_confidence=conf, pre_correct=correct)
            self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(0.8 * focal))
            # H2 fix: Include remedial results in decision cycle metrics
            remedial_loss = 0.0 if correct else max(0.1, conf)
            self._update_metrics_and_decide(remedial_loss)
            # M2: Track remedial separately to avoid inflating primary accuracy
            self.remedial_examples += 1
            if correct:
                self.remedial_correct += 1
            # M4: Include remedial results in rolling accuracy (separate from cumulative)
            self._recent_results.append(correct)
            self.socratic.mastery.record(domain, correct)

    def _publish_exemplar(self, domain: str):
        """Publish a representative exemplar for cross-domain teaching.

        Design note (H3): This intentionally publishes to the shared fallback
        queue (not per-domain queues). Instructor-published exemplars go to the
        shared "grab bag" pool where any domain can pick them up for analogical
        blending. School-directed transfers go to per-domain queues instead.
        """
        if self.adversarial_bank:
            feats, lbl = random.choice(self.adversarial_bank)
            try:
                self.cross_domain_queue.put_nowait({
                    "domain": domain, "features": feats, "label": lbl,
                    "modality": self.config.modality,
                })
            except Exception:
                pass

    def _send_report(self, final: bool = False):
        """Send progress report to school coordinator."""
        elapsed = time.time() - self._start_time
        rate = self.total_examples / max(elapsed, 0.001)
        mastery = self.socratic.mastery.mastery(self.config.domain)
        accuracy = self.total_correct / max(self.total_examples, 1)

        # Close the training loop: feed accuracy back to BG + medulla
        try:
            self.cognitive.post_batch_update(accuracy, 0.7, self.config.domain)
        except Exception:
            pass

        # Phase 3: Run self-assessment and metacognition checks
        self._maybe_self_assess()
        self._metacognition_check()

        # Include decision cycle state if available
        decision_info = None
        if self._last_decision is not None:
            d = self._last_decision
            decision_info = {
                "consensus_action": d.get("consensus_action", -1),
                "lr_factor": round(d.get("lr_factor", 1.0), 4),
                "batch_factor": round(d.get("batch_factor", 1.0), 4),
                "urgency": round(d.get("urgency", 0.0), 3),
                "converged": d.get("converged", False),
                "diagnosis": d.get("primary_diagnosis", ""),
                "recommend_pause": d.get("recommend_pause", False),
            }

        report = {
            "type": "final_report" if final else "progress",
            "domain": self.config.domain,
            "modality": self.config.modality,
            "total_examples": self.total_examples,
            "accuracy": round(accuracy, 4),
            "mastery": round(mastery, 4),
            "examples_per_sec": round(rate, 1),
            "elapsed_s": round(elapsed, 1),
            "method_stats": self.method_stats.summary(),
            "decision_cycle": decision_info,
            # M2: Remedial stats tracked separately
            "remedial_examples": self.remedial_examples,
            "remedial_correct": self.remedial_correct,
            "error": self._error,
            "ts": time.time(),
        }
        try:
            self.school_queue.put_nowait(report)
        except Exception:
            # M3 fix: Log dropped reports at DEBUG level
            logger.debug("[%s] Report dropped — school_queue full",
                         self.config.domain)

    @property
    def is_finished(self) -> bool:
        return self._finished

    @property
    def has_error(self) -> bool:
        """M4: Thread-safe accessor for error state (avoids direct _error access)."""
        return self._error is not None

    def get_mastery(self) -> float:
        return self.socratic.mastery.mastery(self.config.domain)

    def get_report(self) -> Dict:
        elapsed = time.time() - self._start_time
        return {
            "domain": self.config.domain,
            "modality": self.config.modality,
            "total_examples": self.total_examples,
            "accuracy": round(self.total_correct / max(self.total_examples, 1), 4),
            "mastery": round(self.get_mastery(), 4),
            "method_stats": self.method_stats.summary(),
            # M2: Remedial stats tracked separately
            "remedial_examples": self.remedial_examples,
            "remedial_correct": self.remedial_correct,
            "elapsed_s": round(elapsed, 1),
            "finished": self._finished,
            "error": self._error,
        }

    # ------------------------------------------------------------------
    # Phase 3: Spaced Repetition
    # ------------------------------------------------------------------

    def _spaced_replay_push(self, features: list, label: str):
        """Schedule a failed example for spaced review.

        Note: This method only handles scheduling. Difficulty tracking (fail
        count, attempt count, confidence history) is managed by _execute_method
        to avoid double-counting failures (C2 fix).
        """
        # Reset interval on failure
        interval = 1
        next_step = self.total_examples + interval
        # L5: Use tiebreaker counter to prevent heap comparison falling through to feature list
        self._spaced_replay_counter += 1
        heapq.heappush(self._spaced_replay, (next_step, interval, self._spaced_replay_counter, features, label))

        # H2: Cap replay heap to prevent unbounded growth
        if len(self._spaced_replay) > 2000:
            # Remove the lowest-priority items (highest next_review_step)
            # by rebuilding as a list, sorting, and keeping the 2000 most urgent
            items = sorted(self._spaced_replay)[:2000]
            self._spaced_replay = items
            heapq.heapify(self._spaced_replay)

    def _spaced_replay_tick(self):
        """Process any due spaced-replay examples."""
        reviewed = 0
        while (self._spaced_replay and
               self._spaced_replay[0][0] <= self.total_examples and
               reviewed < 5):  # max 5 replays per tick
            next_step, interval, _tiebreak, features, label = heapq.heappop(self._spaced_replay)
            # H2: Guard prediction against failure
            try:
                pred, conf = self._predict_domain(features, self.config.domain)
            except Exception:
                continue
            correct = (pred == label)
            # Always re-learn (even if correct, reinforce)
            focal = self._compute_focal_factor(features, label,
                                               pre_confidence=conf, pre_correct=correct)
            self.brain.learn(features, self._scope_target_to_domain(label), self._modulate_lr(0.6 * focal))
            # H2 fix: Include spaced replay in decision cycle metrics
            replay_loss = 0.0 if correct else max(0.1, conf)
            self._update_metrics_and_decide(replay_loss)
            self.socratic.mastery.record(self.config.domain, correct)
            # M4: Include spaced replay results in rolling accuracy
            self._recent_results.append(correct)
            if correct:
                # Extend interval (exponential backoff)
                new_interval = min(interval * 2, 256)
            else:
                # Reset interval
                new_interval = 1
            new_step = self.total_examples + new_interval
            # L5: Use tiebreaker counter to prevent heap comparison falling through to feature list
            self._spaced_replay_counter += 1
            heapq.heappush(self._spaced_replay, (new_step, new_interval, self._spaced_replay_counter, features, label))
            reviewed += 1

    def _feature_hash(self, features: list) -> str:
        """Fast hash of a feature vector for difficulty tracking."""
        if len(features) > 32:
            sample = features[:16] + features[-16:]
        else:
            sample = features
        return hashlib.md5(str(sample).encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Phase 3: Difficulty-Gated Curriculum
    # ------------------------------------------------------------------

    def _should_skip_easy(self, features: list, label: str,
                          confidence: float) -> bool:
        """Skip examples the brain already knows well, considering confidence trajectory."""
        accuracy = self.total_correct / max(self.total_examples, 1)
        if accuracy < 0.7:
            return False  # Don't skip anything at low mastery

        fh = self._feature_hash(features)
        info = self._example_difficulty.get(fh)

        if info is None:
            return False  # Never seen, don't skip

        if confidence > 0.95 and info['fails'] == 0:
            return True  # High confidence, never failed -- skip

        # Check confidence trajectory: if confidence is rising, prioritize
        if len(info['conf_history']) >= 3:
            recent = list(info['conf_history'])
            trend = recent[-1] - recent[0]
            if trend > 0.1 and confidence > 0.8:
                return True  # Confidence rising and already high -- skip

        return False

    def _get_difficulty_lr_scale(self, features: list, label: str = "") -> float:
        """Scale LR based on example difficulty and confidence trajectory."""
        fh = self._feature_hash(features)
        info = self._example_difficulty.get(fh)

        if info is None:
            return 1.0  # Unseen example -- normal LR

        fail_rate = info['fails'] / max(info['attempts'], 1)
        accuracy = self.total_correct / max(self.total_examples, 1)

        # Check if confidence is improving on this example
        if len(info['conf_history']) >= 2:
            recent = list(info['conf_history'])
            trend = recent[-1] - recent[0]

            if trend > 0.05:
                # Confidence improving -- this example is being learned, boost slightly
                return 1.0 + 0.3 * fail_rate
            elif trend < -0.05:
                # Confidence decreasing -- possibly too hard, reduce LR to prevent oscillation
                return max(0.3, 0.7 - 0.2 * fail_rate)

        # Default: higher LR for harder examples
        if accuracy > 0.8:
            return 0.5 + fail_rate  # Focus on hard at high mastery
        else:
            return 1.0 - 0.3 * fail_rate  # Easier examples first at low mastery

    # ------------------------------------------------------------------
    # Phase 3: Self-Assessment
    # ------------------------------------------------------------------

    def _maybe_collect_holdout(self, features: list, label: str):
        """Reservoir-sample into holdout buffer for self-assessment."""
        # M7: Use separate counter for holdout candidates (not total_examples)
        self._holdout_candidates_seen += 1
        if len(self._holdout_buffer) < self._holdout_max:
            self._holdout_buffer.append((features, label))
        else:
            # Reservoir sampling with correct population count
            idx = random.randint(0, self._holdout_candidates_seen - 1)
            if idx < self._holdout_max:
                self._holdout_buffer[idx] = (features, label)

    def _rolling_accuracy(self) -> float:
        """Compute rolling accuracy from recent results (H3).

        H3 fix: Return 0.5 (chance level) when no results yet, rather than
        cumulative accuracy which is biased by early examples.
        """
        if not self._recent_results:
            return 0.5  # chance-level prior, not cumulative
        return sum(self._recent_results) / len(self._recent_results)

    def _maybe_self_assess(self):
        """Run self-assessment cycle if due."""
        if (self.total_examples - self._last_self_assessment_step
                < self._self_assessment_interval):
            return
        if len(self._holdout_buffer) < 20:
            return
        self._last_self_assessment_step = self.total_examples
        domain = self.config.domain

        # Pure prediction on holdout (no learning) — ensemble for stability
        correct = 0
        total_agreement = 0.0
        for features, label in self._holdout_buffer:
            pred, conf, agreement = self._ensemble_predict(
                features, domain, k=3, noise_sigma=0.01
            )
            if pred == label:
                correct += 1
            total_agreement += agreement
        self._held_out_accuracy = correct / len(self._holdout_buffer)
        avg_agreement = total_agreement / len(self._holdout_buffer)

        # H4: Use rolling window accuracy instead of cumulative (less inertial)
        train_acc = self._rolling_accuracy()
        gap = train_acc - self._held_out_accuracy

        self._log_example({
            "action": "SELF_ASSESS",
            "held_out_accuracy": round(self._held_out_accuracy, 4),
            "training_accuracy": round(train_acc, 4),
            "gap": round(gap, 4),
            "holdout_size": len(self._holdout_buffer),
            "ensemble_agreement": round(avg_agreement, 4),
            "verdict": "overfitting" if gap > 0.1 else "good_generalization",
        })

        # If overfitting: reduce LR via scheduler reset to slow down
        # H6: Only adjust if cooldown has elapsed
        # M2: Symmetric suppression — suppress downward adjustment if last was "up"
        _suppress_down = (self._last_lr_direction == "up"
                          and (self.total_examples - self._last_lr_direction_step) < 500)
        if (gap > 0.15
                and (self.total_examples - self._last_lr_adjust_step) >= 200
                and not _suppress_down):
            self._lr_scheduler.base_lr *= 0.8
            self._lr_scheduler.base_lr = max(self._lr_scheduler.base_lr, 0.1)
            self._last_lr_adjust_step = self.total_examples
            # M4: Record direction to suppress opposite adjustment for 500 steps
            self._last_lr_direction = "down"
            self._last_lr_direction_step = self.total_examples

    # ------------------------------------------------------------------
    # Phase 3: Metacognition-Driven Actions
    # ------------------------------------------------------------------

    def _metacognition_check(self):
        """Check for stall/regression and take corrective action."""
        # H3: Use rolling window accuracy instead of cumulative
        train_acc = self._rolling_accuracy()

        # Track peak accuracy
        if train_acc > self._peak_accuracy:
            self._peak_accuracy = train_acc

        # Detect regression (>5% drop from peak)
        # H6: Only adjust LR if cooldown has elapsed (200 steps)
        # M4: Suppress upward adjustment if last direction was "down" within 500 steps
        _suppress_up = (self._last_lr_direction == "down"
                        and (self.total_examples - self._last_lr_direction_step) < 500)
        if (self._peak_accuracy > 0.3 and train_acc < self._peak_accuracy - 0.05
                and (self.total_examples - self._last_lr_adjust_step) >= 200
                and not _suppress_up):
            # Temporary LR boost to escape
            self._lr_scheduler.base_lr = min(self._lr_scheduler.base_lr * 1.3, 1.5)
            self._last_lr_adjust_step = self.total_examples
            # M4: Record direction to suppress opposite adjustment for 500 steps
            self._last_lr_direction = "up"
            self._last_lr_direction_step = self.total_examples
            self._log_example({
                "action": "METACOG_REGRESSION",
                "peak": round(self._peak_accuracy, 4),
                "current": round(train_acc, 4),
                "new_base_lr": round(self._lr_scheduler.base_lr, 4),
            })

    # ------------------------------------------------------------------
    # Phase 4: Domain Centroid Tracking
    # ------------------------------------------------------------------

    def _update_domain_centroid(self, features: list):
        """Update running EMA centroid for this domain."""
        feat_arr = np.array(features, dtype=np.float32)
        if self._domain_centroid is None:
            self._domain_centroid = feat_arr.copy()
            self._centroid_count = 1
        else:
            alpha = min(0.01, 2.0 / (self._centroid_count + 1))
            self._domain_centroid = (1 - alpha) * self._domain_centroid + alpha * feat_arr
            self._centroid_count += 1

    def get_domain_centroid(self) -> Optional[np.ndarray]:
        """Return the running domain centroid (for cross-domain similarity)."""
        return self._domain_centroid

    # ------------------------------------------------------------------
    # Enhancement #3: Focal Loss (Confidence-Calibrated Training Signal)
    # ------------------------------------------------------------------

    def _compute_focal_factor(self, features, label,
                              pre_confidence=None, pre_correct=None):
        """Compute focal loss scaling factor for this example.

        Scales learning rate by how much the brain needs to learn:
        - Already correct with high confidence -> very low LR (save compute)
        - Incorrect with high confidence -> high LR (strong correction needed)

        Args:
            pre_confidence: Pre-computed confidence from caller (M1: avoids
                redundant prediction). If None, performs its own prediction.
            pre_correct: Pre-computed correctness from caller. If None,
                performs its own prediction.
        """
        try:
            if pre_confidence is None or pre_correct is None:
                # C1: predict_fast returns a tuple (label_string, confidence_float)
                pred_label, pre_confidence = self.brain.predict_fast(features)
                # C2 fix: Guard NaN/Inf confidence
                if math.isnan(pre_confidence) or math.isinf(pre_confidence):
                    pre_confidence = 0.0
                pre_correct = (pred_label == label)

            if pre_correct:
                # Already correct -- scale down LR proportional to confidence
                # High confidence correct -> very low LR (don't waste compute)
                return max(0.05, 1.0 - pre_confidence)
            else:
                # Incorrect -- scale up LR proportional to confidence
                # High confidence wrong -> high LR (need strong correction)
                return min(2.0, 0.5 + pre_confidence)
        except Exception:
            return 1.0  # Fallback: no scaling


# ---------------------------------------------------------------------------
# Cosine Annealing LR Scheduler (per-instructor)
# ---------------------------------------------------------------------------

class _CosineAnnealingLR:
    """Cosine annealing with warm restarts — returns a multiplicative factor."""

    def __init__(self, base_lr: float = 1.0, min_lr: float = 0.05,
                 cycle_steps: int = 5000, warmup_steps: int = 500):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def step(self):
        """Advance scheduler by one step. Call once per example."""
        self.step_count += 1

    def get_lr(self) -> float:
        """Compute current LR factor without advancing step."""
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (
                self.step_count / max(self.warmup_steps, 1))
        # Cosine annealing with warm restarts
        cycle_pos = (self.step_count - self.warmup_steps) % self.cycle_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / self.cycle_steps))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def reset(self):
        self.step_count = 0


# ---------------------------------------------------------------------------
# Feature Utilities
# ---------------------------------------------------------------------------

def _text_to_features(text: str, num_inputs: int, feature_step: int = 3000) -> list:
    """Hybrid text feature encoding: semantic embeddings + character n-grams.

    Enhancement #8: Progressive feature unfreezing — starts with 100% semantic
    features, gradually blends in n-grams over training steps.

    Phase 1 (steps 0-1000):    100% semantic (pure embedding signal)
    Phase 2 (steps 1000-3000): Linear ramp from 0% to 50% n-gram
    Phase 3 (steps 3000+):     Steady 50/50 (original behavior)

    Falls back to pure n-grams if sentence-transformers unavailable.

    Args:
        feature_step: Per-instructor feature step count (default 3000 = steady state).
    """
    model = _get_embedding_model()
    if model is not None:
        try:
            ngram_ratio = _get_feature_blend_ratio_for_step(feature_step)
            semantic_ratio = 1.0 - ngram_ratio

            # Compute dimensions for each component
            sem_dims = max(1, int(num_inputs * semantic_ratio))
            ngram_dims = num_inputs - sem_dims

            # Semantic features
            embedding = model.encode(text[:1000], convert_to_numpy=True)
            sem_features = np.zeros(sem_dims, dtype=np.float32)
            copy_len = min(len(embedding), sem_dims)
            sem_features[:copy_len] = embedding[:copy_len]

            if ngram_dims > 0:
                ngram_features = np.array(
                    _text_to_ngram_features(text, ngram_dims), dtype=np.float32)
                combined = np.concatenate([sem_features, ngram_features])
            else:
                # Pad semantic to full width
                combined = np.zeros(num_inputs, dtype=np.float32)
                combined[:len(sem_features)] = sem_features

            norm = np.linalg.norm(combined)
            if norm > 1e-6:
                combined /= norm
            return combined.tolist()
        except Exception:
            pass  # Fall through to pure n-grams
    return _text_to_ngram_features(text, num_inputs)


def _text_to_ngram_features(text: str, num_inputs: int) -> list:
    """Text -> character n-gram feature encoding (fallback).

    Original 4-channel n-gram encoder kept as fallback when
    sentence-transformers is unavailable.
    """
    try:
        from benchmark_datasets import text_to_features
        return text_to_features(text, num_inputs)
    except ImportError:
        pass
    # Inline fallback — collision-free character n-gram encoding
    features = [0.0] * num_inputs
    if not text:
        return features
    text_lower = text[:4000].lower().strip()
    if not text_lower:
        return features

    ch1_size = int(num_inputs * 0.30)
    ch2_size = int(num_inputs * 0.30)
    ch3_size = int(num_inputs * 0.25)
    ch1_start = 0
    ch2_start = ch1_size
    ch3_start = ch2_start + ch2_size
    ch4_start = ch3_start + ch3_size

    # Channel 1: character unigram frequencies (explicit bins)
    n_chars = len(text_lower)
    for ch in text_lower:
        code = ord(ch)
        if 97 <= code <= 122:
            idx = code - 97
        elif 48 <= code <= 57:
            idx = 26 + (code - 48)
        elif ch in ' \t\n':
            idx = 36
        elif ch in '.,;:!?':
            idx = 37 + min(ord(ch) % 6, 5)
        elif ch in '"\'`()[]{}':
            idx = 43
        elif ch in '-_/\\|':
            idx = 44
        elif ch in '@#$%^&*~':
            idx = 45
        elif ch in '+=<>':
            idx = 46
        elif 192 <= code <= 687:
            idx = 47 + ((code - 192) % min(max(ch1_size - 48, 1), 16))
        else:
            idx = min(ch1_size - 1, 47)
        if idx < ch1_size:
            features[ch1_start + idx] += 1.0
    if n_chars > 0:
        for i in range(ch1_start, ch1_start + ch1_size):
            features[i] /= n_chars

    # Channel 2: character bigram frequencies (explicit bins)
    for i in range(len(text_lower) - 1):
        c1 = ord(text_lower[i]) - 97
        c2 = ord(text_lower[i + 1]) - 97
        if 0 <= c1 < 26 and 0 <= c2 < 26:
            bigram_idx = (c1 * 26 + c2) % ch2_size
            features[ch2_start + bigram_idx] += 1.0
    bigram_total = max(sum(features[ch2_start:ch2_start + ch2_size]), 1.0)
    for i in range(ch2_start, ch2_start + ch2_size):
        features[i] /= bigram_total

    # Channel 3: character trigrams + word structure
    trigram_bins = ch3_size // 2
    word_bins = ch3_size - trigram_bins
    for i in range(len(text_lower) - 2):
        c1 = ord(text_lower[i]) - 97
        c2 = ord(text_lower[i + 1]) - 97
        c3 = ord(text_lower[i + 2]) - 97
        if 0 <= c1 < 26 and 0 <= c2 < 26 and 0 <= c3 < 26:
            trigram_idx = (c1 * 676 + c2 * 26 + c3) % max(trigram_bins, 1)
            features[ch3_start + trigram_idx] += 1.0
    tri_total = max(sum(features[ch3_start:ch3_start + trigram_bins]), 1.0)
    for i in range(ch3_start, ch3_start + trigram_bins):
        features[i] /= tri_total

    words = text_lower.split()
    n_words = len(words)
    word_base = ch3_start + trigram_bins
    if n_words > 0 and word_bins > 0:
        wl_bins = min(16, word_bins // 4)
        for w in words:
            wl = min(len(w), wl_bins) - 1
            if 0 <= wl < wl_bins:
                features[word_base + wl] += 1.0
        for i in range(wl_bins):
            features[word_base + i] /= n_words
        init_base = word_base + wl_bins
        init_bins = min(26, max(word_bins - wl_bins, 0))
        if init_bins > 0:
            for w in words:
                if w and 97 <= ord(w[0]) <= 122:
                    li = ord(w[0]) - 97
                    if li < init_bins:
                        features[init_base + li] += 1.0
            for i in range(init_bins):
                features[init_base + i] /= max(n_words, 1)
        end_base = init_base + max(init_bins, 0)
        end_bins = min(26, max(word_bins - wl_bins - max(init_bins, 0), 0))
        if end_bins > 0:
            for w in words:
                if w and 97 <= ord(w[-1]) <= 122:
                    li = ord(w[-1]) - 97
                    if li < end_bins:
                        features[end_base + li] += 1.0
            for i in range(end_bins):
                features[end_base + i] /= max(n_words, 1)
        vc_base = end_base + max(end_bins, 0)
        remaining = word_bins - wl_bins - max(init_bins, 0) - max(end_bins, 0)
        if remaining > 0:
            vowels = set('aeiou')
            for pos in range(min(5, remaining)):
                vowel_count = sum(1 for w in words if len(w) > pos and w[pos] in vowels)
                features[vc_base + pos] = vowel_count / max(n_words, 1)

    # Channel 4: structural / meta features
    meta_base = ch4_start
    if meta_base + 0 < num_inputs:
        features[meta_base + 0] = min(n_chars / 1000.0, 1.0)
    if meta_base + 1 < num_inputs:
        features[meta_base + 1] = min(n_words / 200.0, 1.0) if n_words else 0.0
    if meta_base + 2 < num_inputs:
        features[meta_base + 2] = min(sum(len(w) for w in words) / max(n_words, 1) / 15.0, 1.0) if words else 0.0
    sentences = max(text_lower.count('.') + text_lower.count('!') + text_lower.count('?'), 1)
    if meta_base + 3 < num_inputs:
        features[meta_base + 3] = min(sentences / 20.0, 1.0)
    if meta_base + 4 < num_inputs:
        features[meta_base + 4] = min(n_words / max(sentences, 1) / 30.0, 1.0)
    unique_words = len(set(words)) if words else 0
    if meta_base + 5 < num_inputs:
        features[meta_base + 5] = unique_words / max(n_words, 1)
    if meta_base + 6 < num_inputs:
        features[meta_base + 6] = min(unique_words / 500.0, 1.0)
    if meta_base + 7 < num_inputs:
        features[meta_base + 7] = sum(1 for c in text_lower if c.isalpha()) / max(n_chars, 1)
    if meta_base + 8 < num_inputs:
        features[meta_base + 8] = sum(1 for c in text_lower if c.isdigit()) / max(n_chars, 1)
    if meta_base + 9 < num_inputs:
        features[meta_base + 9] = sum(1 for c in text_lower if c in '.,;:!?') / max(n_chars, 1)
    if meta_base + 10 < num_inputs:
        features[meta_base + 10] = sum(1 for c in text_lower if c == ' ') / max(n_chars, 1)
    if meta_base + 11 < num_inputs:
        features[meta_base + 11] = 1.0 if '?' in text_lower else 0.0
    if meta_base + 12 < num_inputs:
        features[meta_base + 12] = 1.0 if '!' in text_lower else 0.0
    if meta_base + 13 < num_inputs:
        features[meta_base + 13] = min(text_lower.count('"') / 10.0, 1.0)
    original = text[:4000]
    if meta_base + 14 < num_inputs:
        features[meta_base + 14] = sum(1 for c in original if c.isupper()) / max(len(original), 1)
    numbers = re.findall(r'\d+', text_lower)
    if meta_base + 15 < num_inputs:
        features[meta_base + 15] = min(len(numbers) / 10.0, 1.0)

    math_terms = {'equation', 'theorem', 'proof', 'integral', 'derivative',
                  'algebra', 'calculus', 'matrix', 'vector', 'polynomial'}
    science_terms = {'experiment', 'hypothesis', 'molecule', 'electron', 'quantum',
                     'cell', 'dna', 'protein', 'energy', 'gravity'}
    medical_terms = {'patient', 'diagnosis', 'treatment', 'symptom', 'disease',
                     'clinical', 'surgery', 'therapy', 'medication', 'dose'}
    legal_terms = {'court', 'plaintiff', 'defendant', 'statute', 'jurisdiction',
                   'verdict', 'testimony', 'lawyer', 'judge', 'appeal'}
    tech_terms = {'algorithm', 'database', 'function', 'variable', 'compiler',
                  'server', 'api', 'protocol', 'encryption', 'software'}
    finance_terms = {'market', 'stock', 'investment', 'portfolio', 'dividend',
                     'revenue', 'profit', 'inflation', 'interest', 'bond'}
    philosophy_terms = {'ethics', 'moral', 'metaphysics', 'epistemology', 'ontology',
                        'consciousness', 'existence', 'virtue', 'logic', 'reasoning'}
    literature_terms = {'novel', 'poem', 'author', 'narrative', 'character',
                        'plot', 'metaphor', 'fiction', 'genre', 'literary'}
    word_set = set(words)
    domain_lists = [math_terms, science_terms, medical_terms, legal_terms,
                    tech_terms, finance_terms, philosophy_terms, literature_terms]
    for d_idx, terms in enumerate(domain_lists):
        if meta_base + 16 + d_idx < num_inputs:
            overlap = len(word_set & terms)
            features[meta_base + 16 + d_idx] = min(overlap / 5.0, 1.0)

    remaining_start = meta_base + 24
    vowel_clusters = len(re.findall(r'[aeiou]+', text_lower))
    if remaining_start < num_inputs:
        features[remaining_start] = min(vowel_clusters / max(n_words, 1) / 3.0, 1.0)
    if remaining_start + 1 < num_inputs:
        features[remaining_start + 1] = min(vowel_clusters / max(n_words, 1) / 5.0, 1.0)
    if remaining_start + 2 < num_inputs:
        features[remaining_start + 2] = (
            1.0 - min(sum(len(w) for w in words) / max(n_words, 1) / 10.0, 1.0)
            if words else 0.5
        )

    norm = math.sqrt(sum(v * v for v in features))
    if norm > 0:
        features = [v / norm for v in features]
    return features


def _pad_or_truncate(features: list, target_len: int) -> list:
    """Pad with zeros or truncate to target length."""
    if len(features) >= target_len:
        return features[:target_len]
    return features + [0.0] * (target_len - len(features))


def _audio_fallback_features(samples: list, num_inputs: int) -> list:
    """Basic audio feature extraction when cortex bindings unavailable."""
    features = [0.0] * num_inputs
    n = len(samples)
    if n == 0:
        return features

    # Basic statistics
    mean = sum(samples) / n
    variance = sum((s - mean) ** 2 for s in samples) / max(n, 1)
    rms = (sum(s * s for s in samples) / n) ** 0.5

    features[0] = mean
    features[1] = variance
    features[2] = rms

    # Zero crossing rate
    zc = sum(1 for i in range(1, n) if (samples[i] >= 0) != (samples[i-1] >= 0))
    features[3] = zc / max(n, 1)

    # Downsample to fill remaining features
    step = max(1, n // (num_inputs - 4))
    for i in range(4, num_inputs):
        idx = (i - 4) * step
        if idx < n:
            features[i] = samples[idx]

    return features


def _visual_fallback_features(pixels: list, w: int, h: int,
                               num_inputs: int) -> list:
    """Basic visual feature extraction when cortex bindings unavailable."""
    features = [0.0] * num_inputs
    n = len(pixels)
    if n == 0:
        return features

    # Global statistics
    mean = sum(pixels) / max(n, 1)
    features[0] = mean
    features[1] = (sum((p - mean) ** 2 for p in pixels) / max(n, 1)) ** 0.5

    # Spatial subsampling
    step = max(1, n // (num_inputs - 2))
    for i in range(2, num_inputs):
        idx = (i - 2) * step
        if idx < n:
            features[i] = pixels[idx]

    return features
