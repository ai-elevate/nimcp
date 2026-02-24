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

import json
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

from socratic_trainer import SocraticTrainer, SocraticConfig
from cognitive_orchestrator import CognitiveOrchestrator
from safety_gate import SafetyGate, SafetyConfig

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
    report_interval: int = 100
    max_examples_per_dataset: int = 50_000
    difficulty_ramp_rate: float = 0.01
    adversarial_fraction: float = 0.15
    debate_noise_level: float = 0.1
    analogical_blend_ratio: float = 0.5
    startup_delay_s: float = 0.0


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
        """Meta-learning: adjust method weights based on performance."""
        for m in TeachingMethod:
            acc = self.accuracy(m.value)
            self._weights[m.value] = 0.8 * self._weights[m.value] + 0.2 * (0.5 + acc)

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
        """Main instructor loop."""
        self._start_time = time.time()
        self._open_log()
        try:
            # Staggered startup
            if self.config.startup_delay_s > 0:
                time.sleep(self.config.startup_delay_s)

            # Teach each dataset
            for ds_config in self.datasets:
                if self.stop_event.is_set():
                    break
                self._teach_dataset(ds_config)

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
        try:
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

    # --- Teaching Methods ---

    def _execute_method(self, method: TeachingMethod, features: list,
                        label: str, domain: str,
                        grade: Optional[DataGrade]) -> Tuple[bool, float]:
        """Execute a teaching method. Returns (correct, loss)."""
        conf_mod = grade.confidence_modifier if grade else 1.0

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
        """Predict-before-learn with adaptive confidence."""
        pred, conf = self.brain.predict_fast(features)
        correct = (pred == label)

        # Adaptive confidence scaling
        if correct and conf > 0.7:
            lr = 0.2 * conf_mod
        elif correct:
            lr = 0.5 * conf_mod
        elif conf > 0.7:
            lr = 1.0 * conf_mod  # Confidently wrong — max correction
        else:
            lr = 0.8 * conf_mod

        self.brain.learn(features, label, lr)
        loss = 0.0 if correct else (1.0 - conf)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_curriculum(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Difficulty-ordered: skip easy examples as difficulty ramps."""
        pred, conf = self.brain.predict_fast(features)
        correct = (pred == label)

        # Skip if brain already knows AND difficulty is still low
        if correct and conf > 0.9 and self.difficulty < 0.5:
            return correct, 0.0

        lr = (0.5 + 0.5 * self.difficulty) * conf_mod
        self.brain.learn(features, label, lr)
        loss = 0.0 if correct else (1.0 - conf)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_contrastive(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Learn what things are NOT — teach with negative examples."""
        pred, conf = self.brain.predict_fast(features)
        correct = (pred == label)

        # Teach the correct label
        self.brain.learn(features, label, 0.7 * conf_mod)

        # If wrong, also explicitly teach "not the wrong answer"
        if not correct and pred:
            # Re-teach with correct answer at higher confidence
            self.brain.learn(features, label, 0.9 * conf_mod)

        loss = 0.0 if correct else (1.0 - conf)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_debate(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Two perspectives (original + noisy) argue, brain resolves."""
        # Perspective 1: original features
        pred1, conf1 = self.brain.predict_fast(features)

        # Perspective 2: perturbed features (different viewpoint)
        noise_level = self.config.debate_noise_level
        noisy = [f + random.gauss(0, noise_level) for f in features]
        pred2, conf2 = self.brain.predict_fast(noisy)

        # Brain resolves: if both agree, lower confidence. If disagree, higher.
        if pred1 == pred2:
            lr = 0.5 * conf_mod  # consensus → moderate
        else:
            lr = 0.9 * conf_mod  # disagreement → stronger teaching

        correct = (pred1 == label)
        self.brain.learn(features, label, lr)

        loss = 0.0 if correct else max(1.0 - conf1, 1.0 - conf2)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_meta(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Meta-learning: adapt method weights based on performance."""
        pred, conf = self.brain.predict_fast(features)
        correct = (pred == label)

        # Use mastery to set learning rate
        mastery = self.socratic.mastery.mastery(domain)
        lr = (0.3 + 0.7 * (1.0 - mastery)) * conf_mod
        self.brain.learn(features, label, lr)

        loss = 0.0 if correct else (1.0 - conf)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_adversarial(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Find weaknesses, store hard examples, re-teach."""
        pred, conf = self.brain.predict_fast(features)
        correct = (pred == label)

        # Store hard examples
        if not correct or conf < 0.5:
            if len(self.adversarial_bank) < 500:
                self.adversarial_bank.append((features, label))

        # Teach with boosted confidence on hard examples
        lr = (0.9 if not correct else 0.4) * conf_mod
        self.brain.learn(features, label, lr)

        # Periodically re-teach hard examples
        if (self.total_examples % 200 == 0 and self.adversarial_bank
                and random.random() < self.config.adversarial_fraction):
            hard_feat, hard_label = random.choice(self.adversarial_bank)
            self.brain.learn(hard_feat, hard_label, 0.8 * conf_mod)

        loss = 0.0 if correct else (1.0 - conf)
        self.socratic.mastery.record(domain, correct)
        return correct, loss

    def _method_analogical(self, features, label, domain, conf_mod) -> Tuple[bool, float]:
        """Cross-domain transfer via blending with cross-domain exemplar."""
        pred, conf = self.brain.predict_fast(features)
        correct = (pred == label)

        self.brain.learn(features, label, 0.7 * conf_mod)

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
                    self.brain.learn(blended, label, 0.4 * conf_mod)
        except Exception:
            pass

        loss = 0.0 if correct else (1.0 - conf)
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
            self.brain.learn(features, label, 0.8)
            self.total_examples += 1
            self.socratic.mastery.record(domain, True)

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


# ---------------------------------------------------------------------------
# Feature Utilities
# ---------------------------------------------------------------------------

def _text_to_features(text: str, num_inputs: int) -> list:
    """Text → feature encoding using shared benchmark_datasets encoder."""
    try:
        from benchmark_datasets import text_to_features
        return text_to_features(text[:2000], num_inputs)
    except ImportError:
        pass
    # Inline fallback
    import hashlib as _h
    text_lower = text[:2000].lower().strip()
    features = [0.0] * num_inputs
    if not text_lower:
        return features
    q = num_inputs // 4
    for ch in text_lower:
        features[ord(ch) % q] += 1.0
    for i in range(len(text_lower) - 1):
        bg = text_lower[i:i + 2]
        h = int(_h.md5(bg.encode()).hexdigest(), 16)
        features[q + (h % q)] += 1.0
    words = text_lower.split()
    for wi, word in enumerate(words):
        h = int(_h.md5(word.encode()).hexdigest(), 16)
        features[2 * q + (h % q)] += 1.0
    mx = max(features) if features else 1.0
    if mx > 0:
        features = [v / mx for v in features]
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
