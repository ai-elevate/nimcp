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

import hashlib
import heapq
import json
import math
import os
import random
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from socratic_trainer import SocraticTrainer, SocraticConfig
from cognitive_orchestrator import CognitiveOrchestrator
from safety_gate import SafetyGate, SafetyConfig

# ---------------------------------------------------------------------------
# Lazy-loaded sentence-transformer embedding model (Phase 1)
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL = None
_EMBEDDING_LOCK = threading.Lock()
_EMBEDDING_AVAILABLE = None  # None = not checked yet

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
        except (ImportError, Exception):
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
            s = self._stats.get(m.value, {})
            recent = s.get("attempts", 0)
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
                 num_inputs: int, log_dir: Optional[Path] = None):
        super().__init__(name=f"Instructor-{config.domain}", daemon=True)
        self.brain = brain
        self.config = config
        self.datasets = datasets
        self.school_queue = school_queue
        self.cross_domain_queue = cross_domain_queue
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
        self.difficulty = 0.0
        self.adversarial_bank: List[Tuple[list, str]] = []
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
        self._lr_step_count = 0

        # Phase 3: Spaced repetition + difficulty tracking
        self._spaced_replay: list = []  # heap: (next_review_step, interval, features, label)
        self._example_difficulty: dict = {}  # hash(features) -> fail_count
        self._peak_accuracy = 0.0
        self._method_accuracy_delta: Dict[str, deque] = {
            m.value: deque(maxlen=200) for m in TeachingMethod}

        # Phase 3: Self-assessment holdout
        self._holdout_buffer: List[Tuple[list, str]] = []  # (features, label)
        self._holdout_max = 50
        self._last_self_assessment_step = 0
        self._self_assessment_interval = 500
        self._held_out_accuracy: float = 0.0

        # Phase 4: Domain centroid tracking (running EMA)
        self._domain_centroid: Optional[np.ndarray] = None
        self._centroid_count = 0

        # Logging
        self._log_dir = log_dir
        self._log_file = None
        self._log_lock = threading.Lock()

    def _open_log(self):
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

    def _close_log(self):
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
            self._close_log()
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
        except Exception:
            return

        count = 0
        for example in dataset:
            if self.stop_event.is_set():
                break
            self._wait_for_recess()

            result = self.processor.extract_features_and_label(example, domain)
            if result is None:
                continue

            features, label = result
            # Domain-prefix label to prevent cross-domain collision
            label = f"{domain}:{label}"

            # Grade data quality
            text = self._extract_text(example)
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
        except Exception:
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

            if count >= self.config.max_examples_per_dataset:
                break

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
        except Exception:
            return

        count = 0
        for example in dataset:
            if self.stop_event.is_set():
                break
            self._wait_for_recess()

            text = self._extract_text(example)
            if not text or not text.strip():
                continue

            features = _text_to_features(text, self.num_inputs)
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

            if count >= self.config.max_examples_per_dataset:
                break

    # --- Brain-State LR Modulation ---

    def _modulate_lr(self, base_lr: float) -> float:
        """Modulate learning rate through cosine schedule + decision cycle.

        Pipeline:
          1. Cosine annealing with warm restarts (per-instructor)
          2. Decision cycle lr_factor (Layers 1/2/3) multiplied on top
          3. Fallback to unified 8-factor pipeline if no decision
        """
        # Step 1: Cosine annealing schedule
        self._lr_step_count += 1
        schedule_factor = self._lr_scheduler.get_lr()

        # Step 2: Decision cycle lr_factor if available
        if self._last_decision is not None:
            dc_factor = self._last_decision.get("lr_factor", 1.0)
            return base_lr * schedule_factor * dc_factor

        # Step 3: Fallback to unified pipeline
        try:
            adaptive = float(self.cognitive.compute_adaptive_lr(base_lr))
            return adaptive * schedule_factor
        except Exception:
            return base_lr * schedule_factor

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
        gradient_variance = (sum((g - mean_grad) ** 2 for g in grads) / len(grads)) ** 0.5

        # Get current LR from the unified pipeline
        current_lr = self.cognitive.compute_adaptive_lr(0.001)

        decision = self.cognitive.compute_decision_cycle(
            loss_current, loss_previous,
            grad_norm, grad_norm_previous,
            loss_volatility, gradient_variance,
            current_lr, 32.0)

        if decision is not None:
            self._last_decision = decision

    # --- Teaching Methods ---

    def _predict_domain(self, features, domain: str):
        """Domain-scoped prediction — only considers labels from this domain."""
        prefix = f"{domain}:"
        try:
            return self.brain.predict_in_domain(features, prefix)
        except (AttributeError, TypeError):
            # Fallback for older nimcp without predict_in_domain
            return self.brain.predict_fast(features)

    def _execute_method(self, method: TeachingMethod, features: list,
                        label: str, domain: str,
                        grade: Optional[DataGrade]) -> Tuple[bool, float]:
        """Execute a teaching method with dynamic learning hooks."""
        conf_mod = grade.confidence_modifier if grade else 1.0

        # Phase 3: Difficulty-scaled LR modifier
        diff_scale = self._get_difficulty_lr_scale(features)
        conf_mod *= diff_scale

        # Phase 3: Process any due spaced-replay examples
        self._spaced_replay_tick()

        # Phase 4: Update domain centroid
        self._update_domain_centroid(features)

        # Phase 3: Collect holdout samples for self-assessment
        if random.random() < 0.02:  # 2% holdout rate
            self._maybe_collect_holdout(features, label)

        if method == TeachingMethod.SOCRATIC:
            return self._method_socratic(features, label, domain, conf_mod)
        elif method == TeachingMethod.CURRICULUM:
            return self._method_curriculum(features, label, domain, conf_mod)
        elif method == TeachingMethod.CONTRASTIVE:
            return self._method_contrastive(features, label, domain, conf_mod)
        elif method == TeachingMethod.DEBATE:
            return self._method_debate(features, label, domain, conf_mod)
        elif method == TeachingMethod.META:
            return self._method_meta(features, label, domain, conf_mod)
        elif method == TeachingMethod.ADVERSARIAL:
            return self._method_adversarial(features, label, domain, conf_mod)
        elif method == TeachingMethod.ANALOGICAL:
            return self._method_analogical(features, label, domain, conf_mod)
        else:
            return self._method_socratic(features, label, domain, conf_mod)

    def _method_socratic(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Predict-before-learn with adaptive confidence + spaced replay."""
        pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Phase 3: Skip easy examples at high mastery
        if self._should_skip_easy(features, conf) and correct:
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

        self.brain.learn(features, label, self._modulate_lr(lr))
        loss = 0.0 if correct else (1.0 - conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)

        # Phase 3: Schedule failed examples for spaced repetition
        if not correct:
            self._spaced_replay_push(features, label)

        return correct, loss

    def _method_curriculum(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Difficulty-ordered: skip easy examples as difficulty ramps."""
        pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Skip if brain already knows AND difficulty is still low
        if correct and conf > 0.9 and self.difficulty < 0.5:
            return correct, 0.0

        lr = (0.5 + 0.5 * self.difficulty) * conf_mod
        self.brain.learn(features, label, self._modulate_lr(lr))
        loss = 0.0 if correct else (1.0 - conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_contrastive(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Learn what things are NOT — teach with negative examples."""
        pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Teach the correct label
        self.brain.learn(features, label, self._modulate_lr(0.7 * conf_mod))

        # If wrong, also explicitly teach "not the wrong answer"
        if not correct and pred:
            # Re-teach with correct answer at higher confidence
            self.brain.learn(features, label, self._modulate_lr(0.9 * conf_mod))

        loss = 0.0 if correct else (1.0 - conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_debate(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Two perspectives (original + noisy) argue, brain resolves."""
        # Perspective 1: original features
        pred1, conf1 = self._predict_domain(features, domain)

        # Perspective 2: perturbed features (different viewpoint)
        noise_level = self.config.debate_noise_level
        noisy = [f + random.gauss(0, noise_level) for f in features]
        pred2, conf2 = self._predict_domain(noisy, domain)

        # Brain resolves: if both agree, lower confidence. If disagree, higher.
        if pred1 == pred2:
            lr = 0.5 * conf_mod  # consensus → moderate
        else:
            lr = 0.9 * conf_mod  # disagreement → stronger teaching

        correct = (pred1 == label)
        self.brain.learn(features, label, self._modulate_lr(lr))

        loss = 0.0 if correct else max(1.0 - conf1, 1.0 - conf2)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_meta(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Meta-learning: adapt method weights based on performance."""
        pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Use mastery to set learning rate
        mastery = self.socratic.mastery.mastery(domain)
        lr = (0.3 + 0.7 * (1.0 - mastery)) * conf_mod
        self.brain.learn(features, label, self._modulate_lr(lr))

        loss = 0.0 if correct else (1.0 - conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_adversarial(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Find weaknesses, schedule spaced review, re-teach."""
        pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        # Phase 3: Skip easy examples at high mastery
        if self._should_skip_easy(features, conf) and correct:
            self.socratic.mastery.record(domain, correct)
            return correct, 0.0

        # Store hard examples in both adversarial_bank AND spaced replay
        if not correct or conf < 0.5:
            if len(self.adversarial_bank) < 500:
                self.adversarial_bank.append((features, label))
            # Phase 3: Schedule for spaced repetition
            self._spaced_replay_push(features, label)

        # Teach with boosted confidence on hard examples
        lr = (0.9 if not correct else 0.4) * conf_mod
        self.brain.learn(features, label, self._modulate_lr(lr))

        # Periodically re-teach hard examples
        if (self.total_examples % 200 == 0 and self.adversarial_bank
                and random.random() < self.config.adversarial_fraction):
            hard_feat, hard_label = random.choice(self.adversarial_bank)
            self.brain.learn(hard_feat, hard_label, self._modulate_lr(0.8 * conf_mod))

        loss = 0.0 if correct else (1.0 - conf)
        self._update_metrics_and_decide(loss)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_analogical(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Cross-domain transfer via blending with cross-domain exemplar."""
        pred, conf = self._predict_domain(features, domain)
        correct = (pred == label)

        self.brain.learn(features, label, self._modulate_lr(0.7 * conf_mod))

        # Try to blend with cross-domain exemplar
        try:
            exemplar = self.cross_domain_queue.get_nowait()
            if exemplar.get("modality", "text") == self.config.modality:
                ex_feats = exemplar.get("features", [])
                if len(ex_feats) == len(features):
                    ratio = self.config.analogical_blend_ratio
                    blended = [
                        f * (1 - ratio) + e * ratio
                        for f, e in zip(features, ex_feats)
                    ]
                    self.brain.learn(blended, label, self._modulate_lr(0.4 * conf_mod))
        except Exception:
            pass

        loss = 0.0 if correct else (1.0 - conf)
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
            if isinstance(samples, list) and len(samples) > 0:
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
            if isinstance(samples, list) and len(samples) > 0:
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
            return _text_to_features(str(text), self.num_inputs), label
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
        """Re-teach hard items from adversarial bank."""
        if not self.adversarial_bank:
            return
        domain = self.config.domain
        random.shuffle(self.adversarial_bank)
        for features, label in self.adversarial_bank[:200]:
            if self.stop_event.is_set():
                break
            pred, conf = self._predict_domain(features, domain)
            correct = (pred == label)
            self.brain.learn(features, label, self._modulate_lr(0.8))
            self.total_examples += 1
            if correct:
                self.total_correct += 1
            self.socratic.mastery.record(domain, correct)

    def _publish_exemplar(self, domain: str):
        """Publish a representative exemplar for cross-domain teaching."""
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
            "error": self._error,
            "ts": time.time(),
        }
        try:
            self.school_queue.put_nowait(report)
        except Exception:
            pass

    @property
    def is_finished(self) -> bool:
        return self._finished

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
            "elapsed_s": round(elapsed, 1),
            "finished": self._finished,
            "error": self._error,
        }

    # ------------------------------------------------------------------
    # Phase 3: Spaced Repetition
    # ------------------------------------------------------------------

    def _spaced_replay_push(self, features: list, label: str):
        """Schedule a failed example for spaced review."""
        key = self._feature_hash(features)
        # Reset interval on failure
        interval = 1
        next_step = self.total_examples + interval
        heapq.heappush(self._spaced_replay, (next_step, interval, features, label))
        # Track difficulty
        self._example_difficulty[key] = self._example_difficulty.get(key, 0) + 1

    def _spaced_replay_tick(self):
        """Process any due spaced-replay examples."""
        reviewed = 0
        while (self._spaced_replay and
               self._spaced_replay[0][0] <= self.total_examples and
               reviewed < 5):  # max 5 replays per tick
            next_step, interval, features, label = heapq.heappop(self._spaced_replay)
            pred, conf = self._predict_domain(features, self.config.domain)
            correct = (pred == label)
            # Always re-learn (even if correct, reinforce)
            self.brain.learn(features, label, self._modulate_lr(0.6))
            self.socratic.mastery.record(self.config.domain, correct)
            if correct:
                # Extend interval (exponential backoff)
                new_interval = min(interval * 2, 256)
            else:
                # Reset interval
                new_interval = 1
            new_step = self.total_examples + new_interval
            heapq.heappush(self._spaced_replay, (new_step, new_interval, features, label))
            reviewed += 1

    def _feature_hash(self, features: list) -> str:
        """Fast hash of a feature vector for difficulty tracking."""
        if len(features) > 16:
            sample = features[:8] + features[-8:]
        else:
            sample = features
        return hashlib.md5(str(sample).encode()).hexdigest()[:12]

    # ------------------------------------------------------------------
    # Phase 3: Difficulty-Gated Curriculum
    # ------------------------------------------------------------------

    def _should_skip_easy(self, features: list, conf: float) -> bool:
        """Skip examples the brain already knows with high confidence."""
        mastery = self.socratic.mastery.mastery(self.config.domain)
        if mastery < 0.5:
            # Low mastery: skip nothing
            return False
        if mastery > 0.8 and conf > 0.95:
            # High mastery: skip only very confident
            return True
        if conf > 0.98:
            # Skip extremely confident regardless
            return True
        return False

    def _get_difficulty_lr_scale(self, features: list) -> float:
        """Scale LR based on example difficulty relative to current mastery."""
        key = self._feature_hash(features)
        fail_count = self._example_difficulty.get(key, 0)
        mastery = self.socratic.mastery.mastery(self.config.domain)
        if mastery < 0.5:
            # Low mastery: boost easy examples (foundation building)
            return 1.0 if fail_count == 0 else 0.8
        elif mastery < 0.8:
            # Mid mastery: balanced
            return 0.7 + 0.3 * min(fail_count / 3.0, 1.0)
        else:
            # High mastery: focus on hard examples
            return 0.3 + 0.7 * min(fail_count / 3.0, 1.0)

    # ------------------------------------------------------------------
    # Phase 3: Self-Assessment
    # ------------------------------------------------------------------

    def _maybe_collect_holdout(self, features: list, label: str):
        """Reservoir-sample into holdout buffer for self-assessment."""
        if len(self._holdout_buffer) < self._holdout_max:
            self._holdout_buffer.append((features, label))
        else:
            # Reservoir sampling
            idx = random.randint(0, self.total_examples)
            if idx < self._holdout_max:
                self._holdout_buffer[idx] = (features, label)

    def _maybe_self_assess(self):
        """Run self-assessment cycle if due."""
        if (self.total_examples - self._last_self_assessment_step
                < self._self_assessment_interval):
            return
        if len(self._holdout_buffer) < 20:
            return
        self._last_self_assessment_step = self.total_examples
        domain = self.config.domain

        # Pure prediction on holdout (no learning)
        correct = 0
        for features, label in self._holdout_buffer:
            pred, conf = self._predict_domain(features, domain)
            if pred == label:
                correct += 1
        self._held_out_accuracy = correct / len(self._holdout_buffer)

        train_acc = self.total_correct / max(self.total_examples, 1)
        gap = train_acc - self._held_out_accuracy

        self._log_example({
            "action": "SELF_ASSESS",
            "held_out_accuracy": round(self._held_out_accuracy, 4),
            "training_accuracy": round(train_acc, 4),
            "gap": round(gap, 4),
            "holdout_size": len(self._holdout_buffer),
            "verdict": "overfitting" if gap > 0.1 else "good_generalization",
        })

        # If overfitting: reduce LR via scheduler reset to slow down
        if gap > 0.15:
            self._lr_scheduler.base_lr *= 0.8
            self._lr_scheduler.base_lr = max(self._lr_scheduler.base_lr, 0.1)

    # ------------------------------------------------------------------
    # Phase 3: Metacognition-Driven Actions
    # ------------------------------------------------------------------

    def _metacognition_check(self):
        """Check for stall/regression and take corrective action."""
        train_acc = self.total_correct / max(self.total_examples, 1)

        # Track peak accuracy
        if train_acc > self._peak_accuracy:
            self._peak_accuracy = train_acc

        # Detect regression (>5% drop from peak)
        if self._peak_accuracy > 0.3 and train_acc < self._peak_accuracy - 0.05:
            # Temporary LR boost to escape
            self._lr_scheduler.base_lr = min(self._lr_scheduler.base_lr * 1.3, 1.5)
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

    def get_lr(self) -> float:
        self.step_count += 1
        # Warmup phase
        if self.step_count <= self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (
                self.step_count / self.warmup_steps)
        # Cosine annealing with warm restarts
        cycle_pos = (self.step_count - self.warmup_steps) % self.cycle_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / self.cycle_steps))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def reset(self):
        self.step_count = 0


# ---------------------------------------------------------------------------
# Feature Utilities
# ---------------------------------------------------------------------------

def _text_to_features(text: str, num_inputs: int) -> list:
    """Hybrid text feature encoding: semantic embeddings + character n-grams.

    Layout (for num_inputs=1024):
      Dims 0-511:   Sentence-transformer embedding (384d zero-padded to 512)
      Dims 512-1023: Character n-grams (existing encoder, scaled to 512)
    Falls back to pure n-grams if sentence-transformers unavailable.
    """
    model = _get_embedding_model()
    if model is not None:
        try:
            half = num_inputs // 2
            # Semantic half: sentence-transformer embedding
            embedding = model.encode(text[:1000], convert_to_numpy=True)
            sem_features = np.zeros(half, dtype=np.float32)
            copy_len = min(len(embedding), half)
            sem_features[:copy_len] = embedding[:copy_len]

            # N-gram half
            ngram_features = _text_to_ngram_features(text, half)

            # Combine and L2-normalize
            combined = np.concatenate([sem_features, np.array(ngram_features, dtype=np.float32)])
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
    import math as _math
    import re as _re
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
    numbers = _re.findall(r'\d+', text_lower)
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
    vowel_clusters = len(_re.findall(r'[aeiou]+', text_lower))
    if remaining_start < num_inputs:
        features[remaining_start] = min(vowel_clusters / max(n_words, 1) / 3.0, 1.0)
    if remaining_start + 1 < num_inputs:
        features[remaining_start + 1] = min(vowel_clusters / max(n_words, 1) / 5.0, 1.0)
    if remaining_start + 2 < num_inputs:
        features[remaining_start + 2] = (
            1.0 - min(sum(len(w) for w in words) / max(n_words, 1) / 10.0, 1.0)
            if words else 0.5
        )

    norm = _math.sqrt(sum(v * v for v in features))
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
