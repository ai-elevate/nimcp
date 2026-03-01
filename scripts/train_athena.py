#!/usr/bin/env python3
"""
NIMCP Athena Foundation Model Training — Parallel School
=========================================================

WHAT: Train a 1M-neuron brain called "Athena" to serve as the pretrained
      baseline for all future brains.
WHY:  Every new brain should start from a trained baseline rather than
      random initialization — dramatically faster convergence on new tasks.
HOW:  5-phase parallel school pipeline:
      Phase 0: Orientation — built-in benchmarks with predict-before-learn (warm-up)
      Phase 1: Parallel School — 23 instructor agents teach simultaneously
               (20 text + 3 multimodal domains, 7 teaching methods each)
      Phase 2: Guided Study — streaming HuggingFace datasets (Socratic)
      Phase 3: Research — curiosity-driven exploration
      Phase 4: Final Exam — creativity test, hard-item review, save

Full NIMCP system engaged: 67+ cognitive modules, 32 brain regions, 3 sensory
cortices, mesh network, bio-async router, plasticity orchestrator, glial system,
security system — all active during every predict/learn call.

Layers:
  Layer 1 (SocraticTrainer): Predict → adaptive confidence → teach → replay
  Layer 2 (InstructorAgent): 7 teaching methods per domain
  Layer 3 (School):          23 parallel instructors, recess, dashboard
  Layer 4 (DataSkeptic):     7-dimension data quality grading
  Layer 5 (SafetyGate):      Python pre-filter + C LGSS content filter

Designed to run via nohup:
  nohup python3 scripts/train_athena.py > athena_training.log 2>&1 &

All output goes to file, no interactive prompts.
"""

import atexit
import gc
import heapq
import json
import math
import os
import random
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
BUILD_PYTHON = PROJECT_ROOT / "build" / "lib" / "python"

# Add Python bindings to path
if BUILD_PYTHON.exists():
    sys.path.insert(0, str(BUILD_PYTHON))

# Add scripts dir for streaming_train imports
sys.path.insert(0, str(SCRIPT_DIR))

# Add frontend/backend for benchmark_datasets
sys.path.insert(0, str(PROJECT_ROOT / "frontend" / "backend"))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    import nimcp
    print(f"[Athena] NIMCP loaded (version: {nimcp.version()})")
except ImportError:
    print("[Athena] FATAL: nimcp module not found.")
    print("  Build with: cd build && cmake .. && make nimcp_python -j4")
    sys.exit(1)

# Benchmark datasets (built-in, no network needed)
try:
    from benchmark_datasets import (
        WineDataset, BreastCancerDataset, FashionMNISTDataset,
        MMLUDataset, ARCDataset, HellaSwagDataset, WinograndeDataset,
        EthicsScenarios, NBackGenerator, SequencePatterns,
        text_to_features,
    )
    BENCHMARKS_AVAILABLE = True
    print("[Athena] Benchmark datasets loaded")
except ImportError:
    BENCHMARKS_AVAILABLE = False
    print("[Athena] WARNING: benchmark_datasets not found — Phase 0/1 skipped")

# Streaming datasets (HuggingFace)
HF_AVAILABLE = False
STREAMING_AVAILABLE = False
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
    print("[Athena] HuggingFace datasets library available")
except ImportError:
    print("[Athena] WARNING: 'datasets' library not installed — Phase 2 skipped")
    print("  Install with: pip install datasets")

try:
    from streaming_train import StreamingDatasetProcessor, StreamConfig
    STREAMING_AVAILABLE = True
    print("[Athena] Streaming trainer loaded")
except ImportError:
    print("[Athena] WARNING: streaming_train not importable — Phase 2 uses fallback")

# Socratic active learning layers
from socratic_trainer import SocraticTrainer, SocraticConfig
from safety_gate import SafetyGate, SafetyConfig
from cognitive_orchestrator import CognitiveOrchestrator
from active_learner import ActiveLearner

print("[Athena] Socratic active learning layers loaded")


def _pad_or_truncate(feats: list, target_len: int) -> list:
    """Pad with zeros or truncate features to target length."""
    if len(feats) < target_len:
        return feats + [0.0] * (target_len - len(feats))
    elif len(feats) > target_len:
        return feats[:target_len]
    return feats


# Parallel school system
try:
    from school import School, SchoolConfig
    SCHOOL_AVAILABLE = True
    print("[Athena] Parallel school system loaded")
except ImportError:
    SCHOOL_AVAILABLE = False
    print("[Athena] WARNING: school module not available — fallback to sequential")

# ---------------------------------------------------------------------------
# Graceful Shutdown Signal Handling
# ---------------------------------------------------------------------------
_shutdown_requested = False
_force_exit = False
_clean_exit = False


def _signal_handler(signum, frame):
    global _shutdown_requested, _force_exit
    if _shutdown_requested:
        # Second signal: set force-exit flag instead of calling sys.exit(1),
        # which would interrupt the atexit handler mid-save and corrupt the
        # checkpoint.  The atexit handler checks _force_exit and skips save
        # if set, then calls os._exit(1).
        _force_exit = True
        print(f"\n[Athena] Second signal received — will force exit after atexit")
        return
    _shutdown_requested = True
    print(f"\n[Athena] Signal {signum} received — graceful shutdown requested")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ---------------------------------------------------------------------------
# Resume Phase Ordering
# ---------------------------------------------------------------------------
PHASE_ORDER = {
    "phase0": 0, "phase1": 1, "phase2": 2, "phase3": 3, "phase4": 4,
    "legacy_phase1": 1, "legacy_phase2": 2, "legacy_phase3": 3,
}


def phase_reached(resume_phase: str, target_phase: str) -> bool:
    """Check if resume_phase has reached or passed target_phase using ordinal comparison."""
    return PHASE_ORDER.get(resume_phase, -1) >= PHASE_ORDER.get(target_phase, 0)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ATHENA_NEURONS = 1_500_000
ATHENA_NUM_INPUTS = 1024    # Was 256 — increased for collision-free encoding
ATHENA_NUM_OUTPUTS = 256    # Was 128 — increased to eliminate label overflow

# Output paths
ATHENA_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained" / "athena" / "v1.0"
ATHENA_MODEL_PATH = ATHENA_MODEL_DIR / "nimcp_athena_foundation_v1.0.nimcp"
ATHENA_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "athena"
ATHENA_LOG_DIR = PROJECT_ROOT / "logs"

# Training hyperparameters
PHASE0_EPOCHS = 2          # Quick warm-up epochs (predict is expensive, keep low)
PHASE1_EPOCHS = 30         # Epochs per built-in dataset (legacy sequential mode)
PHASE2_MAX_PER_DATASET = 50_000   # Max examples per streaming dataset
PHASE2_BATCH_SIZE = 1000          # Streaming batch size
PHASE2_CHECKPOINT_INTERVAL = 10_000  # Checkpoint every N examples
INTROSPECTION_INTERVAL = 5000     # Metacognitive check every N examples

# Mastery thresholds for phase gating
RESEARCH_MASTERY_THRESHOLD = 0.6   # Phase 3: minimum mastery for research eligibility
CREATIVE_MASTERY_THRESHOLD = 0.8   # Phase 4: minimum mastery for creative exam
HARD_EXAMPLE_LOSS_THRESHOLD = 0.3  # Loss above which reasoning chain runs

# Hard example mining parameters
REPLAY_LR_FACTOR = 0.8            # Replay LR = base_lr * factor (lower avoids overfit)
MAX_HARD_REPLAY_PER_PASS = 500    # Max hard examples replayed at end of phase
REPLAY_DECAY_FACTOR = 0.95        # Decay factor for aging hard examples
RECESS_AROUSAL_BOOST = 0.15       # Arousal boost during introspection recess


# ---------------------------------------------------------------------------
# Learning Rate Scheduling — Cosine Annealing with Warm Restarts
# ---------------------------------------------------------------------------

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler with warm restarts.

    BIOLOGICAL BASIS: Learning rate should decrease over time (like synaptic
    scaling), with periodic restarts (like sleep/consolidation cycles).
    """
    def __init__(self, base_lr: float = 0.5, min_lr: float = 0.05,
                 cycle_steps: int = 5000, warmup_steps: int = 500):
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.cycle_steps = cycle_steps
        self.warmup_steps = warmup_steps
        self.step_count = 0

    def get_lr(self) -> float:
        self.step_count += 1

        # Warmup phase: linear ramp from min_lr to base_lr
        if self.step_count <= self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (self.step_count / self.warmup_steps)

        # Cosine annealing with warm restarts
        cycle_pos = (self.step_count - self.warmup_steps) % self.cycle_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / self.cycle_steps))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def peek_lr(self):
        """Return LR from the last step. Before any step(), returns base_lr."""
        if self.step_count == 0:
            return self.base_lr
        if self.step_count <= self.warmup_steps:
            return self.min_lr + (self.base_lr - self.min_lr) * (self.step_count / max(self.warmup_steps, 1))
        cycle_pos = (self.step_count - self.warmup_steps) % self.cycle_steps
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cycle_pos / self.cycle_steps))
        return self.min_lr + (self.base_lr - self.min_lr) * cosine_factor

    def reset(self):
        self.step_count = 0


# ---------------------------------------------------------------------------
# Curriculum Learning — Developmental Domain Progression
# ---------------------------------------------------------------------------

class CurriculumManager:
    """Curriculum learning: start with fewer, easier classes, gradually add more.

    BIOLOGICAL BASIS: Biological learning proceeds from simple to complex
    (developmental psychology, Zone of Proximal Development).

    Phase 1 (steps 0-2000): 8 domains (core academics)
    Phase 2 (steps 2000-5000): 16 domains (+ professional)
    Phase 3 (steps 5000+): All domains

    Domain names must match the actual dataset domain names from
    foundation_datasets_config.json (e.g. 'biology', 'chemistry', 'physics').
    """
    def __init__(self, all_domains: list):
        self.all_domains = all_domains
        self.step = 0
        # Sort domains by estimated difficulty (simpler first).
        # Domain names match foundation_datasets_config.json domain field values.
        self.easy_domains = ['biology', 'chemistry', 'physics', 'history',
                            'language', 'humanities', 'literature', 'programming']
        self.medium_domains = ['medicine', 'finance', 'philosophy', 'economics',
                               'ethics', 'psychology', 'earth_sciences', 'sociology']
        self.hard_domains = [d for d in all_domains
                            if d not in self.easy_domains and d not in self.medium_domains]

    def _compute_active_domains(self, step: int) -> list:
        """Compute which domains are active at the given step without side effects."""
        if step <= 2000:
            return self.easy_domains
        elif step <= 5000:
            return self.easy_domains + self.medium_domains
        else:
            return self.all_domains

    def advance(self, steps: int = 1):
        """Advance the curriculum step counter by the given number of training steps."""
        self.step += steps

    def get_active_domains(self) -> list:
        return self._compute_active_domains(self.step)

    def should_include(self, domain: str) -> bool:
        active = self._compute_active_domains(self.step)
        return domain in active or domain.split(':')[0] in active


# ---------------------------------------------------------------------------
# Hard Example Mining — Hippocampal Replay of Difficult Items
# ---------------------------------------------------------------------------

class HardExampleMiner:
    """Track and replay high-loss training examples.

    BIOLOGICAL BASIS: Difficult experiences get more hippocampal replay
    (Olafsdottir et al. 2018). Spaced repetition for hard items.
    """
    def __init__(self, capacity: int = 5000, replay_ratio: float = 0.2):
        self.capacity = capacity
        self.replay_ratio = replay_ratio  # Fraction of batch from hard examples
        self.hard_examples = []  # Min-heap of (loss, counter, features, label) tuples
        self.min_loss_threshold = 0.3  # Only store examples with loss > threshold
        self._counter = 0  # Tiebreaker for equal-loss comparisons (avoids O(n) list cmp)

    def record(self, features: list, label: str, loss: float):
        """Record a training example and its loss.

        Uses a min-heap keyed by (loss, counter) so that when over capacity we
        can efficiently pop the smallest-loss item in O(log n) instead of
        sorting the entire list in O(n log n).  The counter tiebreaker prevents
        fallthrough to O(n) list comparison when two entries have equal loss.
        """
        if loss > self.min_loss_threshold:
            self._counter += 1
            entry = (loss, self._counter, features, label)
            if len(self.hard_examples) < self.capacity:
                heapq.heappush(self.hard_examples, entry)
            elif loss > self.hard_examples[0][0]:
                # Replace the smallest-loss item with this harder example
                heapq.heapreplace(self.hard_examples, entry)

    def get_replay_batch(self, batch_size: int) -> list:
        """Get a batch of hard examples for replay.

        Returns list of (loss, features, label) tuples (counter stripped).
        """
        replay_count = max(1, int(batch_size * self.replay_ratio))
        if len(self.hard_examples) < replay_count:
            return [(loss, f, l) for loss, _, f, l in self.hard_examples]
        sample = random.sample(self.hard_examples, replay_count)
        return [(loss, f, l) for loss, _, f, l in sample]

    def decay(self, factor: float = 0.95):
        """Decay stored losses (so old hard examples eventually drop out)."""
        decayed = []
        for loss, cnt, f, l in self.hard_examples:
            new_loss = loss * factor
            if new_loss > self.min_loss_threshold * 0.5:
                decayed.append((new_loss, cnt, f, l))
        heapq.heapify(decayed)
        self.hard_examples = decayed


# ---------------------------------------------------------------------------
# Domain-Prefixed Labels — Prevent cross-domain label collision
# ---------------------------------------------------------------------------

def domain_label(domain: str, label) -> str:
    """Prefix a label with its domain to prevent cross-domain collision.

    Without this, Wine label "0", MMLU label "0", Fashion-MNIST label "0",
    and ARC label "0" all map to the same output neuron — catastrophic
    interference.  With prefixing, they become "wine_0", "mmlu_0", etc.
    Colons in domain or label are replaced with underscores to avoid
    ambiguity with the domain:label separator.
    """
    safe_domain = str(domain).replace(":", "_")
    safe_label = str(label).replace(":", "_")
    return f"{safe_domain}:{safe_label}"


def adapt_dataset_labels(examples: list, domain: str) -> list:
    """Add domain prefix to all labels in a dataset's examples (non-mutating)."""
    adapted = []
    for ex in examples:
        adapted.append({**ex, "label": domain_label(domain, ex["label"])})
    return adapted


# ---------------------------------------------------------------------------
# Conversation Probe — Talk to Athena between training phases
# ---------------------------------------------------------------------------

# Test prompts: greetings, identity, domain questions, reasoning, creativity
PROBE_CONVERSATIONS = [
    # Greetings & identity
    ("Hello Athena, how are you?", "greeting"),
    ("Who are you?", "identity"),
    ("What have you learned so far?", "introspection"),
    # Domain-specific questions (aligned with training data)
    ("What type of wine is this with high alcohol and dark color?", "science"),
    ("Is this breast tumor malignant or benign?", "medicine"),
    ("What article of clothing is in this image?", "visual"),
    ("What is the capital of France?", "general_knowledge"),
    ("Explain Newton's second law of motion.", "physics"),
    ("What is photosynthesis?", "biology"),
    ("Solve: if x + 3 = 7, what is x?", "math"),
    # Reasoning & creativity
    ("If all roses are flowers and all flowers need water, do roses need water?", "logic"),
    ("What would happen if gravity suddenly doubled?", "reasoning"),
    ("Write a short poem about neural networks.", "creativity"),
    # Emotional / theory of mind
    ("I'm feeling sad today.", "empathy"),
    ("Do you enjoy learning new things?", "self_awareness"),
]


def _probe_encode_text(text: str, num_inputs: int) -> list:
    """Encode text to feature vector (uses shared text_to_features)."""
    if not BENCHMARKS_AVAILABLE:
        # Cannot encode text without benchmark_datasets; return zeros
        return [0.0] * num_inputs
    return text_to_features(text, num_inputs)


def conversation_probe(brain, logger: "AthenaLogger", phase_name: str):
    """Run a conversation probe — inference only, no training."""
    if not BENCHMARKS_AVAILABLE:
        logger.log(f"\nCONVERSATION PROBE SKIPPED — benchmark_datasets not available")
        return []
    logger.log(f"\n{'─' * 70}")
    logger.log(f"CONVERSATION PROBE — after {phase_name}")
    logger.log(f"{'─' * 70}")

    results = []
    for prompt, category in PROBE_CONVERSATIONS:
        features = _probe_encode_text(prompt, ATHENA_NUM_INPUTS)

        # Try decide_full first (rich cognitive output)
        label = ""
        confidence = 0.0
        explanation = ""
        num_active = 0
        try:
            result = brain.decide_full(features)
            label = result.get("label", "")
            confidence = float(result.get("confidence", 0.0))
            explanation = result.get("explanation", "")
            num_active = int(result.get("num_active_neurons", 0))
        except Exception:
            # Fall back to predict
            try:
                label, confidence = brain.predict(features)
                confidence = float(confidence)
            except Exception as e:
                logger.log(f"  [{category:18s}] ERROR: {e}")
                continue

        # Truncate explanation for logging
        expl_short = explanation[:120].replace('\n', ' ') if explanation else "(none)"

        logger.log(f"  [{category:18s}] Q: {prompt}")
        logger.log(f"  {'':18s}  A: label={label!r}  conf={confidence:.3f}  "
                    f"active={num_active}  expl={expl_short}")

        results.append({
            "prompt": prompt,
            "category": category,
            "label": label,
            "confidence": confidence,
            "num_active_neurons": num_active,
            "has_explanation": bool(explanation and explanation.strip()),
        })

        logger.metric({
            "probe": True,
            "phase": phase_name,
            "category": category,
            "prompt": prompt[:80],
            "label": label,
            "confidence": confidence,
            "num_active_neurons": num_active,
            "has_explanation": bool(explanation and explanation.strip()),
        })

    # Summary statistics
    avg_conf = sum(r["confidence"] for r in results) / max(len(results), 1)
    avg_active = sum(r["num_active_neurons"] for r in results) / max(len(results), 1)
    unique_labels = len(set(r["label"] for r in results))
    with_expl = sum(1 for r in results if r["has_explanation"])

    logger.log(f"\n  Probe Summary ({phase_name}):")
    logger.log(f"    Prompts tested:     {len(results)}")
    logger.log(f"    Avg confidence:     {avg_conf:.3f}")
    logger.log(f"    Avg active neurons: {avg_active:.0f}")
    logger.log(f"    Unique labels:      {unique_labels}")
    logger.log(f"    With explanation:   {with_expl}/{len(results)}")
    logger.log(f"{'─' * 70}\n")

    return results


# Module-level dataset cache to avoid re-instantiation on every health_check call
_health_check_datasets = {}


def _get_health_check_datasets():
    """Return cached health-check dataset instances (singletons)."""
    if not _health_check_datasets and BENCHMARKS_AVAILABLE:
        _health_check_datasets["wine"] = WineDataset()
        _health_check_datasets["breast_cancer"] = BreastCancerDataset()
    return _health_check_datasets


def health_check(brain, logger: "AthenaLogger", phase_name: str,
                 min_accuracy: float = 0.0, abort_on_fail: bool = False) -> dict:
    """
    Pipeline-wide health check — runs after every phase.
    Validates brain can still predict, measures accuracy on held-out probes,
    and optionally aborts if accuracy falls below threshold.
    Returns dict with metrics for tracking learning progress.
    """
    logger.log(f"\n  HEALTH CHECK after {phase_name}:")
    metrics = {"phase": phase_name, "healthy": True, "errors": []}

    # 1. Basic predict sanity
    try:
        test_in = [0.1] * ATHENA_NUM_INPUTS
        result = brain.predict_fast(test_in)
        if result is None:
            metrics["errors"].append("predict_fast returned None")
            metrics["healthy"] = False
        else:
            metrics["predict_ok"] = True
    except Exception as e:
        metrics["errors"].append(f"predict crashed: {e}")
        metrics["healthy"] = False

    # 2. Accuracy on built-in benchmark samples (quick — 10 per dataset)
    if BENCHMARKS_AVAILABLE:
        cached_ds = _get_health_check_datasets()
        check_datasets = list(cached_ds.items())
        total_correct, total_tested = 0, 0
        for ds_name, ds in check_datasets:
            examples = ds.get_examples()
            sample = random.sample(examples, min(10, len(examples)))
            correct = 0
            for ex in sample:
                feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
                expected = domain_label(ds_name, ex["label"])
                try:
                    pred, conf = brain.predict_fast(feats)
                    if str(pred) == str(expected):
                        correct += 1
                except Exception:
                    pass
                total_tested += 1
            total_correct += correct
            metrics[f"acc_{ds_name}"] = correct / max(len(sample), 1)

        overall_acc = total_correct / max(total_tested, 1)
        metrics["overall_accuracy"] = overall_acc
        logger.log(f"    Accuracy: {overall_acc:.1%} ({total_correct}/{total_tested})")

        if overall_acc < min_accuracy:
            msg = f"Accuracy {overall_acc:.1%} below threshold {min_accuracy:.1%}"
            metrics["errors"].append(msg)
            metrics["healthy"] = False
            logger.log(f"    WARNING: {msg}")

    # 3. Neuron count sanity (shouldn't change)
    nc = brain.get_neuron_count()
    metrics["neuron_count"] = nc
    logger.log(f"    Neurons: {nc:,}")

    # Log metrics
    logger.metric({"health_check": True, **metrics})

    if not metrics["healthy"]:
        logger.log(f"    HEALTH CHECK: ISSUES DETECTED")
        for err in metrics["errors"]:
            logger.log(f"      - {err}")
        if abort_on_fail:
            logger.log(f"    ABORTING training due to health check failure")
            brain.save(str(ATHENA_CHECKPOINT_DIR / f"athena_health_fail_{phase_name}.bin"))
            raise RuntimeError(f"Health check failed after {phase_name}: {metrics['errors']}")
    else:
        logger.log(f"    HEALTH CHECK: OK")

    return metrics


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class AthenaLogger:
    """File-based logger safe for nohup. Uses buffered file handles."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"athena_train_{ts}.log"
        self.metrics_file = log_dir / f"athena_metrics_{ts}.jsonl"
        self.start_time = time.time()
        self._log_fh = None
        self._metrics_fh = None
        try:
            self._log_fh = open(self.log_file, "a")
            self._metrics_fh = open(self.metrics_file, "a")
        except Exception:
            if self._log_fh:
                self._log_fh.close()
            raise
        self._log_writes_since_flush = 0
        self._metric_writes_since_flush = 0

    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:10.1f}s] {msg}"
        print(line, flush=True)
        self._log_fh.write(line + "\n")
        self._log_writes_since_flush += 1
        if self._log_writes_since_flush >= 50:
            self._log_fh.flush()
            self._log_writes_since_flush = 0

    def metric(self, data: dict):
        t = time.time()
        data["timestamp"] = t
        data["elapsed_s"] = t - self.start_time
        self._metrics_fh.write(json.dumps(data) + "\n")
        self._metric_writes_since_flush += 1
        if self._metric_writes_since_flush >= 50:
            self._metrics_fh.flush()
            self._metric_writes_since_flush = 0

    def close(self):
        """Flush and close file handles."""
        try:
            self._log_fh.flush()
            self._log_fh.close()
        except Exception:
            pass
        try:
            self._metrics_fh.flush()
            self._metrics_fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 0: Orientation — Quick warm-up on built-in benchmarks
# ---------------------------------------------------------------------------

def phase0_orientation(brain, socratic: SocraticTrainer,
                       cognitive: CognitiveOrchestrator,
                       logger: AthenaLogger) -> int:
    """
    Phase 0: Quick warm-up on built-in benchmarks before parallel school.
    Fewer epochs than full Phase 1 — just enough to initialize domain
    representations so instructors start with a non-random brain.
    """
    if not BENCHMARKS_AVAILABLE:
        logger.log("Phase 0 SKIPPED — benchmark_datasets not available")
        return 0

    logger.log("=" * 70)
    logger.log("PHASE 0: Orientation (Quick Warm-Up)")
    logger.log("=" * 70)

    ml_datasets = [
        ("wine", WineDataset()),
        ("breast_cancer", BreastCancerDataset()),
        ("fashion_mnist", FashionMNISTDataset()),
    ]

    qa_datasets = [
        ("mmlu", MMLUDataset()),
        ("arc_easy", ARCDataset()),
        ("hellaswag", HellaSwagDataset()),
        ("winogrande", WinograndeDataset()),
    ]

    all_datasets = []
    for domain_name, ds in ml_datasets:
        examples = ds.get_examples()
        adapted = []
        for ex in examples:
            feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
            adapted.append({"features": feats,
                            "label": domain_label(domain_name, ex["label"])})
        all_datasets.append((domain_name, adapted))

    for domain_name, ds in qa_datasets:
        examples = ds.get_examples()
        adapted = []
        for ex in examples:
            feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
            adapted.append({"features": feats,
                            "label": domain_label(domain_name, ex["label"])})
        all_datasets.append((domain_name, adapted))

    # Initialize training strategy components
    all_domain_names = [name for name, _ in all_datasets]
    scheduler = CosineAnnealingLR(base_lr=0.5, min_lr=0.05,
                                  cycle_steps=5000, warmup_steps=500)
    # NOTE: CurriculumManager is NOT used in Phase 0. Phase 0 dataset names
    # (wine, breast_cancer, fashion_mnist, mmlu, etc.) don't match curriculum
    # domain names (biology, chemistry, physics, etc.), so every dataset would
    # be deferred. Phase 0 is a quick warm-up where all datasets should train.
    hard_miner = HardExampleMiner(capacity=5000, replay_ratio=0.2)

    logger.log(f"  Training strategy: CosineAnnealingLR + HardExampleMiner (no curriculum gating in Phase 0)")

    total_trained = 0
    for ds_name, examples in all_datasets:

        t0 = time.time()
        n_examples = len(examples) * PHASE0_EPOCHS
        logger.log(f"  {ds_name}: {len(examples)} examples × {PHASE0_EPOCHS} epochs")

        # Direct learn — skip predict (brain_decide runs 28 cognitive stages per
        # predict, making it ~100x slower than learn alone). Phase 0 is just warm-up.
        warmup_steps = scheduler.warmup_steps
        for epoch in range(PHASE0_EPOCHS):
            for ex in examples:
                if _shutdown_requested:
                    logger.log("  Shutdown requested — stopping Phase 0 training")
                    return total_trained
                lr = scheduler.get_lr()
                brain.learn(ex["features"], str(ex["label"]), lr)
                total_trained += 1

                # Record for hard example mining — skip during warmup when
                # loss estimates are unreliable.
                # TA-1+TA-4: Only call predict_fast every 5th example to
                # reduce overhead; use None default to avoid stale 0.5 loss
                # that would pollute the hard example bank.
                if scheduler.step_count > warmup_steps and total_trained % 5 == 0:
                    # H12: Use actual prediction error instead of LR proxy
                    estimated_loss = None
                    try:
                        result = brain.predict_fast(ex["features"])
                        if result is not None:
                            pred, _ = result
                            estimated_loss = 0.0 if str(pred) == str(ex["label"]) else 1.0
                    except Exception:
                        pass
                    if estimated_loss is not None:
                        hard_miner.record(ex["features"], str(ex["label"]), estimated_loss)

            # Replay hard examples at the end of each epoch
            # Use peek_lr() to avoid advancing the scheduler step_count on replay
            replay_batch = hard_miner.get_replay_batch(batch_size=50)
            for _loss, feats, label in replay_batch:
                replay_lr = scheduler.peek_lr()
                brain.learn(feats, label, replay_lr * REPLAY_LR_FACTOR)  # Slightly lower LR for replay
                total_trained += 1

        # Decay hard example losses between datasets
        hard_miner.decay(factor=REPLAY_DECAY_FACTOR)

        elapsed = time.time() - t0
        rate = n_examples / max(elapsed, 0.01)
        logger.log(f"    -> {ds_name} done: {n_examples} steps in {elapsed:.1f}s "
                   f"({rate:.0f} steps/s), lr={scheduler.peek_lr():.4f}, "
                   f"hard_bank={len(hard_miner.hard_examples)}")

    # Final hard example replay pass — use peek_lr() to avoid advancing scheduler
    if hard_miner.hard_examples:
        logger.log(f"  Final hard example replay: {len(hard_miner.hard_examples)} items")
        for _loss, _cnt, feats, label in hard_miner.hard_examples[:MAX_HARD_REPLAY_PER_PASS]:
            if _shutdown_requested:
                break
            lr = scheduler.peek_lr()
            brain.learn(feats, label, lr)
            total_trained += 1

    # --- Symbolic Knowledge Base Seeding ---
    logger.log("[Phase 0] Seeding symbolic knowledge base...")
    domain_facts = {
        "science": ["Observable(x) & Repeatable(x)", "Hypothesis(x) & Tested(x)"],
        "math": ["Number(x) & Operation(y)", "Equation(x) & Balanced(x)"],
        "ethics": ["Action(x) & Harmful(x)", "Consent(x) & Informed(x)"],
        "language": ["Sentence(x) & Grammatical(x)", "Word(x) & Meaningful(x)"],
    }
    for domain, facts in domain_facts.items():
        for fact in facts:
            try:
                cognitive.add_logical_fact(f"{domain}_{fact}", salience=0.7)
            except Exception as e:
                logger.log(f"    KB seed error ({domain}): {e}")
                break

    # Initialize reasoning engine
    try:
        cognitive.init_reasoning()
        logger.log("[Phase 0] Reasoning engine initialized")
    except Exception as e:
        logger.log(f"[Phase 0] Reasoning engine init skipped: {e}")

    # Establish basal ganglia reward baseline
    try:
        cognitive.update_reward(accuracy=0.0, expected_accuracy=0.0)
        logger.log("[Phase 0] Basal ganglia reward baseline initialized")
    except Exception as e:
        logger.log(f"[Phase 0] BG reward baseline skipped: {e}")

    # Quick accuracy probe (predict is expensive, so only test 5 per dataset)
    logger.log("  Phase 0 accuracy probe...")
    for ds_name, examples in all_datasets:
        correct, total_probe = 0, min(5, len(examples))
        probe_sample = random.sample(examples, total_probe) if len(examples) >= total_probe else examples
        for ex in probe_sample:
            pred = brain.predict_fast(ex["features"])
            if pred:
                decision = pred[0] if isinstance(pred, tuple) else pred
                if str(decision) == str(ex["label"]):
                    correct += 1
        logger.log(f"    {ds_name}: {correct}/{total_probe} "
                   f"({100*correct/max(1,total_probe):.0f}%)")

    # Quick consolidation — warm-up, replay only
    cognitive.consolidate(mode="light")
    logger.log(f"Phase 0 complete — {total_trained:,} warm-up steps, "
               f"LR scheduler steps={scheduler.step_count}, "
               f"hard examples mined={len(hard_miner.hard_examples)}")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 1: Parallel School — 23 Instructor Agents
# ---------------------------------------------------------------------------

def phase1_parallel_school(brain, socratic: SocraticTrainer,
                           cognitive: CognitiveOrchestrator,
                           logger: AthenaLogger, total_trained: int,
                           max_concurrent_instructors: int = 4,
                           min_domain_accuracy: float = 0.0,
                           max_retry_passes: int = 5) -> tuple:
    """
    Phase 1: 23 parallel instructor agents teach simultaneously.
    20 text domains + 3 multimodal (audio, visual, speech).
    """
    if not SCHOOL_AVAILABLE:
        logger.log("Phase 1 (parallel school) SKIPPED — school module not available")
        logger.log("Falling back to sequential Phase 2...")
        return total_trained, None

    logger.log("\n" + "=" * 70)
    logger.log("PHASE 1: Parallel School (23 Instructors)")
    logger.log("=" * 70)

    config_file = SCRIPT_DIR / "foundation_datasets_config.json"
    if not config_file.exists():
        logger.log(f"Phase 1 SKIPPED — {config_file} not found")
        return total_trained, None

    school_config = SchoolConfig(
        recess_interval_s=300.0,
        report_interval_s=30.0,
        checkpoint_interval_s=600.0,
        max_training_time_s=82800.0,  # 23h (leave 1h for final exam)
        graduation_mastery=0.85,
        max_examples_per_dataset=PHASE2_MAX_PER_DATASET,
        startup_stagger_s=3.0,
        num_inputs=ATHENA_NUM_INPUTS,
        num_outputs=ATHENA_NUM_OUTPUTS,
        max_concurrent_instructors=max_concurrent_instructors,
        min_domain_accuracy=min_domain_accuracy,
        max_retry_passes=max_retry_passes,
    )

    school = School(brain, school_config, logger)
    school.setup(config_file)
    school.start()

    # Get report card
    report_card = school.get_report_card()
    logger.log(f"\n{report_card['domain_report']}")

    school_examples = report_card.get("total_examples", 0)
    total_trained += school_examples
    logger.log(f"\nPhase 1 complete — {school_examples:,} school examples, "
                f"{total_trained:,} total")

    return total_trained, report_card


# ---------------------------------------------------------------------------
# Phase 1 Legacy: Worksheets — Built-in Benchmark Datasets (Socratic)
# ---------------------------------------------------------------------------

def phase1_worksheets(brain, socratic: SocraticTrainer,
                      cognitive: CognitiveOrchestrator,
                      logger: AthenaLogger) -> int:
    """
    Phase 1: Socratic training on all built-in benchmark datasets.
    Predict-before-learn with adaptive confidence and spaced repetition.
    Curiosity detects knowledge gaps between datasets.
    Consolidation ("sleep") between datasets.
    """
    if not BENCHMARKS_AVAILABLE:
        logger.log("Phase 1 SKIPPED — benchmark_datasets not available")
        return 0

    logger.log("=" * 70)
    logger.log("PHASE 1: Worksheets (Socratic Predict-Before-Learn)")
    logger.log("=" * 70)

    # Domain mapping for built-in datasets
    domain_map = {
        "wine": "science", "breast_cancer": "medicine",
        "fashion_mnist": "technology", "mmlu": "general",
        "arc_easy": "science", "hellaswag": "general",
        "winogrande": "general", "nback": "psychology",
        "ethics": "philosophy", "sequences": "math",
    }

    # Structured ML datasets
    ml_datasets = [
        ("wine", WineDataset()),
        ("breast_cancer", BreastCancerDataset()),
        ("fashion_mnist", FashionMNISTDataset()),
    ]

    # Text-based QA datasets
    qa_datasets = [
        ("mmlu", MMLUDataset()),
        ("arc_easy", ARCDataset()),
        ("hellaswag", HellaSwagDataset()),
        ("winogrande", WinograndeDataset()),
    ]

    # Cognitive benchmarks
    nback = NBackGenerator(n=2, sequence_length=40, num_features=ATHENA_NUM_INPUTS)
    nback_examples = nback.generate(seed=42)
    ethics = EthicsScenarios.get_scenarios()
    seq_patterns = SequencePatterns.generate_repeating(
        pattern_length=4, total_length=80,
        num_features=ATHENA_NUM_INPUTS, seed=42
    )

    all_datasets = []

    # ML datasets — pad/truncate features to ATHENA_NUM_INPUTS + domain-prefix labels
    for domain_name, ds in ml_datasets:
        examples = ds.get_examples()
        adapted = []
        for ex in examples:
            feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
            adapted.append({"features": feats,
                            "label": domain_label(domain_name, ex["label"])})
        all_datasets.append((domain_name, adapted))

    # QA datasets — re-encode to ATHENA_NUM_INPUTS + domain-prefix labels
    for domain_name, ds in qa_datasets:
        examples = ds.get_examples()
        adapted = []
        for ex in examples:
            feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
            adapted.append({"features": feats,
                            "label": domain_label(domain_name, ex["label"])})
        all_datasets.append((domain_name, adapted))

    nback_adapted = [{"features": _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS),
                      "label": domain_label("nback", ex["label"])}
                     for ex in nback_examples]
    all_datasets.append(("nback", nback_adapted))

    ethics_adapted = []
    for s in ethics:
        feats = _pad_or_truncate(s["features"], ATHENA_NUM_INPUTS)
        ethics_adapted.append({"features": feats,
                               "label": domain_label("ethics", s["category"])})
    all_datasets.append(("ethics", ethics_adapted))

    seq_adapted = []
    for s in seq_patterns:
        feats = _pad_or_truncate(s["features"], ATHENA_NUM_INPUTS)
        seq_adapted.append({"features": feats,
                            "label": domain_label("sequences", str(s["position"] % 4))})
    all_datasets.append(("sequences", seq_adapted))

    # Let executive function order datasets by curiosity priorities
    available_domains = list(set(domain_map.get(name, "general")
                                 for name, _ in all_datasets))
    domain_priority = cognitive.get_curiosity_priorities(
        {d: 0.0 for d in available_domains}
    )
    logger.log(f"Curiosity domain priority: {domain_priority}")

    # Train on each dataset (Socratic: predict → confidence → teach → replay)
    total_trained = 0
    for ds_name, examples in all_datasets:
        if _shutdown_requested:
            logger.log("Shutdown requested — stopping Phase 1 worksheets")
            break
        domain = domain_map.get(ds_name, "general")
        logger.log(f"\n--- {ds_name} [{domain}]: {len(examples)} examples, "
                    f"{PHASE1_EPOCHS} epochs (Socratic) ---")

        best_acc = 0.0
        for epoch in range(PHASE1_EPOCHS):
            if _shutdown_requested:
                break
            batch = [(ex["features"], str(ex["label"])) for ex in examples]
            result = socratic.train_batch_socratic(batch, domain)

            total_trained += result["batch_size"]
            best_acc = max(best_acc, result["batch_accuracy"])

            # Post-batch BG + medulla integration
            try:
                batch_accuracy = result["batch_accuracy"]
                domain_expected = result.get("domain_mastery", 0.5)
                domain_name_for_bg = domain
                cognitive.post_batch_update(
                    accuracy=batch_accuracy,
                    expected=domain_expected,
                    domain=domain_name_for_bg)
            except Exception as e:
                logger.log(f"  Non-critical (BG update): {e}")

            # Forward chain to derive new facts (every 10k steps)
            if total_trained % 10000 < result["batch_size"]:
                try:
                    new_facts = cognitive.forward_chain(max_iterations=100)
                    if new_facts > 0:
                        logger.log(f"  Forward chaining derived {new_facts} new facts")
                except Exception as e:
                    logger.log(f"  Non-critical (forward chain): {e}")

            # Check if domain is habitual (mastery > 0.7)
            try:
                cognitive.check_habit(domain)
            except Exception as e:
                logger.log(f"  Non-critical (habit check): {e}")

            # Refresh community cache every 10k steps
            if total_trained % 10000 < result["batch_size"]:
                try:
                    cognitive.invalidate_community_cache()
                    cognitive.cache_communities()
                except Exception as e:
                    logger.log(f"  Non-critical (community cache): {e}")

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.log(f"  Epoch {epoch+1:2d}/{PHASE1_EPOCHS}: "
                            f"acc={result['batch_accuracy']:.4f} "
                            f"(best={best_acc:.4f}) "
                            f"replay_buf={result['buffer_size']} "
                            f"mastery={result['domain_mastery']:.3f}")
                logger.metric({
                    "phase": 1, "dataset": ds_name, "domain": domain,
                    "epoch": epoch + 1,
                    "accuracy": result["batch_accuracy"],
                    "best_accuracy": best_acc,
                    "replay_buffer_size": result["buffer_size"],
                    "domain_mastery": result["domain_mastery"],
                    "total_trained": total_trained,
                })

                # Record batch for cognitive trend tracking
                cognitive.record_batch(domain, result["batch_accuracy"],
                                       result["avg_loss"], result["batch_size"])

        logger.log(f"  {ds_name} done — best accuracy: {best_acc:.4f}, "
                    f"domain mastery: {socratic.mastery.mastery(domain):.3f}")

    # Curiosity gap detection between datasets
    masteries = socratic.mastery.all_masteries()
    gaps = cognitive.detect_knowledge_gaps(masteries)
    logger.log(f"\nKnowledge gaps (lowest mastery first): {gaps[:5]}")

    # Consolidation: C-level memory consolidation + mixed replay
    logger.log(f"\n--- Phase 1 Consolidation ---")
    cognitive.cache_communities()  # Pre-compute topology for community-aware replay
    cognitive.consolidate(mode="auto")

    all_examples = []
    for _, examples in all_datasets:
        all_examples.extend(random.sample(examples, min(200, len(examples))))

    for epoch in range(3):
        batch = [(ex["features"], str(ex["label"])) for ex in all_examples]
        random.shuffle(batch)
        result = socratic.train_batch_socratic(batch, "consolidation")
        total_trained += result["batch_size"]
        logger.log(f"  Consolidation {epoch+1}/3: acc={result['batch_accuracy']:.4f}")

    # Introspection: metacognitive check
    progress = cognitive.assess_learning_progress()
    logger.log(f"\nIntrospection after Phase 1:")
    logger.log(f"  Accuracy: {progress['accuracy']:.4f}")
    logger.log(f"  Learning rate: {progress['learning_rate_effective']:.6f}")
    logger.log(f"  Total steps: {progress['total_learning_steps']}")

    logger.log(f"\n{socratic.get_domain_report()}")
    logger.log(f"\nPhase 1 complete — {total_trained:,} total training steps")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 2: Guided Study — Streaming HuggingFace Datasets (Socratic)
# ---------------------------------------------------------------------------

def phase2_guided_study(brain, socratic: SocraticTrainer,
                        cognitive: CognitiveOrchestrator,
                        logger: AthenaLogger, total_trained: int) -> int:
    """
    Phase 2: Socratic training on HuggingFace streaming datasets.
    Executive function chooses domain order based on curiosity + mastery.
    Introspection monitors learning progress every N examples.
    """
    if not HF_AVAILABLE:
        logger.log("Phase 2 SKIPPED — HuggingFace datasets library not available")
        return total_trained

    logger.log("\n" + "=" * 70)
    logger.log("PHASE 2: Guided Study (Socratic Streaming)")
    logger.log("=" * 70)

    # Load foundation datasets config
    config_file = SCRIPT_DIR / "foundation_datasets_config.json"
    if not config_file.exists():
        logger.log(f"Phase 2 SKIPPED — {config_file} not found")
        return total_trained

    with open(config_file) as f:
        config = json.load(f)

    hf_datasets = [d for d in config["datasets"] if d["type"] == "huggingface"]
    logger.log(f"Found {len(hf_datasets)} streaming datasets")

    # Executive function: order datasets by curiosity + mastery
    masteries = socratic.mastery.all_masteries()
    available_domains = list(set(d.get("domain", "general") for d in hf_datasets))
    priority = cognitive.get_curiosity_priorities(
        {d: masteries.get(d, 0.0) for d in available_domains}
    )
    logger.log(f"Executive domain priority: {priority}")

    # Sort datasets by domain priority
    domain_order = {d: i for i, d in enumerate(priority)}
    hf_datasets.sort(key=lambda d: domain_order.get(d.get("domain", "general"), 999))

    # Initialize training strategy components for Phase 2.
    # INTENTIONAL: Phase 2 creates a fresh LR scheduler (separate from Phase 0's)
    # so that warmup restarts for the new streaming datasets.  This is by design —
    # the streaming data is structurally different from the built-in benchmarks.
    phase2_scheduler = CosineAnnealingLR(base_lr=0.5, min_lr=0.05,
                                         cycle_steps=10000, warmup_steps=1000)
    phase2_curriculum = CurriculumManager(available_domains)
    phase2_miner = HardExampleMiner(capacity=5000, replay_ratio=0.2)
    logger.log(f"  Phase 2 strategy: CosineAnnealingLR + CurriculumManager + HardExampleMiner")

    if STREAMING_AVAILABLE:
        stream_config = StreamConfig(
            batch_size=PHASE2_BATCH_SIZE,
            checkpoint_interval=PHASE2_CHECKPOINT_INTERVAL,
            max_examples_per_dataset=PHASE2_MAX_PER_DATASET,
            checkpoint_dir=ATHENA_CHECKPOINT_DIR,
        )
        processor = StreamingDatasetProcessor(
            brain, stream_config,
            num_inputs=ATHENA_NUM_INPUTS,
            hf_token=os.environ.get("HF_TOKEN"),
        )

        examples_since_introspection = 0
        deferred_datasets = []  # Datasets deferred by curriculum

        for ds_config in hf_datasets:
            name = ds_config["name"]
            domain = ds_config.get("domain", "general")

            # Curriculum gating: defer datasets whose domain is not yet active
            if not phase2_curriculum.should_include(domain):
                deferred_datasets.append(ds_config)
                logger.log(f"\n--- Deferred by curriculum: {name} [{domain}] ---")
                continue

            logger.log(f"\n--- Streaming: {name} [{domain}] "
                        f"(mastery={socratic.mastery.mastery(domain):.3f}) ---")

            try:
                # Stream and train with Socratic method
                dataset = processor.load_streaming_dataset(ds_config)
                if dataset is None:
                    continue

                stream = iter(dataset)
                count = 0

                while True:
                    if _shutdown_requested:
                        logger.log(f"  Shutdown requested — stopping {name}")
                        break

                    try:
                        example = next(stream)
                    except StopIteration:
                        break

                    result = processor.extract_features_and_label(example, domain)
                    if result is None:
                        continue

                    features, label = result
                    prefixed_label = domain_label(domain, label)

                    # Socratic: predict-before-learn per example
                    # Domain-prefix label to prevent cross-domain collision
                    socratic.train_example(features, prefixed_label, domain)
                    # Advance cosine schedule — LR consumed via peek_lr() during
                    # replay and adaptive modulation; main loop uses SocraticTrainer's Leitner LR
                    phase2_scheduler.get_lr()
                    count += 1
                    total_trained += 1
                    examples_since_introspection += 1
                    phase2_curriculum.advance()

                    # Record for hard example mining (estimate loss from prediction)
                    # Only call predict_fast every 10th example to reduce overhead;
                    # only record examples where we have an actual loss estimate
                    est_loss = None  # Only set on 10th-example probes
                    if count % 10 == 0:
                        try:
                            pred, conf = brain.predict_fast(features)
                            est_loss = 0.0 if str(pred) == str(prefixed_label) else (1.0 - conf)
                        except Exception:
                            est_loss = 0.5
                        phase2_miner.record(features, prefixed_label, est_loss)

                    # Run reasoning chain on hard examples (loss > 0.3)
                    if est_loss is not None and est_loss > HARD_EXAMPLE_LOSS_THRESHOLD and hasattr(cognitive, 'reason_about'):
                        try:
                            # Use first 512 chars of text representation
                            example_text = str(label)[:512]
                            confidence = cognitive.reason_about(example_text)
                            if confidence >= 0:
                                logger.log(f"    Reasoning confidence: {confidence:.3f}")
                        except Exception:
                            pass

                    # Consolidate every 10k steps (not just end of phase)
                    if total_trained % 10000 == 0:
                        try:
                            cognitive.consolidate(mode="auto")
                        except Exception:
                            pass

                    if count >= PHASE2_MAX_PER_DATASET:
                        break

                    # Introspection check
                    if examples_since_introspection >= INTROSPECTION_INTERVAL:
                        progress = cognitive.assess_learning_progress()
                        logger.log(f"  [{count:,}] Introspection: "
                                    f"acc={progress['accuracy']:.4f} "
                                    f"lr={progress['learning_rate_effective']:.6f} "
                                    f"mastery={socratic.mastery.mastery(domain):.3f}")
                        cognitive.record_batch(
                            domain, socratic.mastery.mastery(domain),
                            0.0, INTROSPECTION_INTERVAL)
                        examples_since_introspection = 0

                        # Compute unified adaptive learning rate (all brain modulations)
                        try:
                            current_lr = phase2_scheduler.peek_lr()
                            adaptive_lr = cognitive.compute_adaptive_lr(base_lr=current_lr)
                            # Log modulation state for diagnostics
                            mod_state = cognitive.get_modulation_state()
                            if mod_state:
                                logger.log(
                                    f"    [Modulation] LR×{mod_state.get('final_lr_factor', 1.0):.3f} "
                                    f"arousal={mod_state.get('arousal_cognitive_gain', 0):.2f} "
                                    f"inflam={mod_state.get('inflammation_learning_factor', 1):.2f} "
                                    f"portia={mod_state.get('portia_learning_gate', 1):.2f}")
                                if mod_state.get("should_pause", False):
                                    logger.log("    [Modulation] WARNING: Brain signals PAUSE")
                        except Exception:
                            adaptive_lr = None

                        # Replay hard examples during introspection pauses
                        # Use peek_lr() to avoid advancing scheduler step_count on replay
                        replay_batch = phase2_miner.get_replay_batch(batch_size=20)
                        for _r_loss, r_feats, r_label in replay_batch:
                            r_lr = phase2_scheduler.peek_lr()
                            if adaptive_lr is not None:
                                r_lr = adaptive_lr
                            brain.learn(r_feats, r_label, r_lr)
                            total_trained += 1
                        phase2_miner.decay(factor=REPLAY_DECAY_FACTOR)

                        # Boost arousal during recess (simulates coffee break)
                        try:
                            cognitive.boost_arousal(delta=RECESS_AROUSAL_BOOST)
                        except Exception:
                            pass

                    if count % 10000 == 0:
                        logger.log(f"  {name}: {count:,} examples, "
                                    f"mastery={socratic.mastery.mastery(domain):.3f}")

                logger.log(f"  {name}: {count:,} examples, "
                            f"final mastery={socratic.mastery.mastery(domain):.3f}")
                logger.metric({
                    "phase": 2, "dataset": name, "domain": domain,
                    "examples": count, "total_trained": total_trained,
                    "domain_mastery": socratic.mastery.mastery(domain),
                    "replay_buffer": len(socratic.replay_buffer),
                })

            except Exception as e:
                logger.log(f"  {name}: ERROR — {e}")
                continue

            # Checkpoint after each dataset
            checkpoint_path = ATHENA_CHECKPOINT_DIR / f"athena_after_{name}.bin"
            try:
                brain.save(str(checkpoint_path))
                logger.log(f"  Checkpoint saved: {checkpoint_path.name}")
            except Exception as e:
                logger.log(f"  Checkpoint save failed: {e}")

            gc.collect()

        # Second pass: teach deferred datasets (curriculum now allows all)
        if deferred_datasets:
            logger.log(f"\n--- Teaching {len(deferred_datasets)} curriculum-deferred datasets ---")
            phase2_curriculum.step = 5001  # Force all domains active
            for ds_config in deferred_datasets:
                # TA-5: Check shutdown in deferred dataset outer loop
                if _shutdown_requested:
                    logger.log("  Shutdown requested — stopping deferred datasets")
                    break
                name = ds_config["name"]
                domain = ds_config.get("domain", "general")
                logger.log(f"\n--- Deferred: {name} [{domain}] ---")
                try:
                    dataset = processor.load_streaming_dataset(ds_config)
                    if dataset is None:
                        continue
                    stream = iter(dataset)
                    count = 0
                    for example in stream:
                        if _shutdown_requested:
                            logger.log(f"  Shutdown requested — stopping deferred {name}")
                            break
                        result = processor.extract_features_and_label(example, domain)
                        if result is None:
                            continue
                        features, label = result
                        prefixed_label = domain_label(domain, label)
                        socratic.train_example(features, prefixed_label, domain)
                        phase2_scheduler.get_lr()  # Advance cosine schedule
                        count += 1
                        total_trained += 1
                        phase2_curriculum.advance()
                        # Hard example mining for deferred datasets too
                        # Only call predict_fast every 10th example;
                        # only record examples where we have an actual loss estimate.
                        # TA-1: Use None default instead of stale 0.5.
                        if count % 10 == 0:
                            est_loss = None
                            try:
                                pred, conf = brain.predict_fast(features)
                                est_loss = 0.0 if str(pred) == str(prefixed_label) else (1.0 - conf)
                            except Exception:
                                pass
                            if est_loss is not None:
                                phase2_miner.record(features, prefixed_label, est_loss)
                        if count >= PHASE2_MAX_PER_DATASET:
                            break
                    logger.log(f"  {name}: {count:,} deferred examples")
                except Exception as e:
                    logger.log(f"  {name}: ERROR — {e}")
                # Decay hard example losses after each deferred dataset to
                # prevent unbounded growth and let old hard examples age out
                phase2_miner.decay(factor=REPLAY_DECAY_FACTOR)

                # Introspection + consolidation between deferred datasets
                # (matches main dataset loop quality — prevents degraded training)
                try:
                    progress = cognitive.assess_learning_progress()
                    logger.log(f"    Deferred introspection: "
                               f"acc={progress['accuracy']:.4f} "
                               f"mastery={socratic.mastery.mastery(domain):.3f}")
                except Exception:
                    pass
                try:
                    cognitive.consolidate(mode="light")
                except Exception:
                    pass

                # Checkpoint after each deferred dataset (matches main loop pattern)
                checkpoint_path = ATHENA_CHECKPOINT_DIR / f"athena_after_{name}.bin"
                try:
                    brain.save(str(checkpoint_path))
                    logger.log(f"  Checkpoint saved: {checkpoint_path.name}")
                except Exception as e:
                    logger.log(f"  Checkpoint save failed: {e}")

                gc.collect()

        # Final hard example replay for Phase 2 — use peek_lr() for replay
        if phase2_miner.hard_examples:
            replay_count = min(MAX_HARD_REPLAY_PER_PASS, len(phase2_miner.hard_examples))
            logger.log(f"\n  Phase 2 hard example replay: {replay_count} items")
            for _loss, _cnt, feats, label in phase2_miner.hard_examples[:replay_count]:
                if _shutdown_requested:
                    break
                lr = phase2_scheduler.peek_lr()
                brain.learn(feats, label, lr)
                total_trained += 1
    else:
        # Fallback: minimal streaming with Socratic training.
        # INTENTIONAL: This simpler fallback mode omits CosineAnnealingLR,
        # CurriculumManager, and HardExampleMiner.  It relies solely on
        # SocraticTrainer's internal Leitner scheduling for LR and replay.
        # This is by design — the fallback activates when streaming_train
        # is unavailable, indicating a minimal environment.
        if not BENCHMARKS_AVAILABLE:
            logger.log("Fallback streaming SKIPPED — text_to_features requires benchmark_datasets")
        else:
            logger.log("Using fallback streaming (Socratic per-example)")
        for ds_config in (hf_datasets if BENCHMARKS_AVAILABLE else []):
            name = ds_config["name"]
            domain = ds_config.get("domain", "general")
            hf_dataset = ds_config["hf_dataset"]
            hf_subset = ds_config.get("hf_subset")

            logger.log(f"\n--- Streaming: {name} [{domain}] ---")
            try:
                hf_split = ds_config.get("hf_split", "train")
                if hf_subset:
                    dataset = load_dataset(hf_dataset, hf_subset,
                                           split=hf_split, streaming=True)
                else:
                    dataset = load_dataset(hf_dataset, split=hf_split,
                                           streaming=True)

                count = 0
                for example in dataset:
                    if _shutdown_requested:
                        break
                    text = ""
                    for key in ("text", "question", "content", "input", "ctx"):
                        if key in example and example[key]:
                            text = str(example[key])
                            break
                    if not text:
                        text = " ".join(str(v) for v in example.values()
                                        if isinstance(v, str))
                    if not text.strip():
                        continue

                    features = text_to_features(text, ATHENA_NUM_INPUTS)
                    label_val = example.get("answer", example.get("label", None))
                    # TA-2: Skip examples where label is None to avoid training
                    # on "domain:None" labels that pollute the output space.
                    # Fall back to dataset name as domain if domain config is missing.
                    if label_val is None:
                        continue
                    effective_domain = domain if domain else name
                    socratic.train_example(features, domain_label(effective_domain, label_val), effective_domain)
                    count += 1
                    total_trained += 1

                    if count >= PHASE2_MAX_PER_DATASET:
                        break
                    if count % 5000 == 0:
                        logger.log(f"  {name}: {count:,} examples...")

                logger.log(f"  {name}: {count:,} examples trained")
                logger.metric({
                    "phase": 2, "dataset": name, "domain": domain,
                    "examples": count, "total_trained": total_trained,
                })

                # TA-3: Per-dataset checkpoint in fallback path (matches
                # the primary streaming path's checkpoint pattern).
                checkpoint_path = ATHENA_CHECKPOINT_DIR / f"athena_after_{name}.bin"
                try:
                    brain.save(str(checkpoint_path))
                    logger.log(f"  Checkpoint saved: {checkpoint_path.name}")
                except Exception as e:
                    logger.log(f"  Checkpoint save failed: {e}")

            except Exception as e:
                logger.log(f"  {name}: ERROR — {e}")
                continue
            gc.collect()

    # Consolidation between Phase 2 and Phase 3
    logger.log(f"\n--- Phase 2 Consolidation ---")
    cognitive.cache_communities()  # Refresh community topology before consolidation
    cognitive.consolidate(mode="auto")

    logger.log(f"\n{socratic.get_domain_report()}")
    logger.log(f"\nPhase 2 complete — {total_trained:,} total training steps")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 3: Research — Web-Augmented Learning (Safety-Gated)
# ---------------------------------------------------------------------------

def phase3_research(brain, active_learner: ActiveLearner,
                    socratic: SocraticTrainer,
                    cognitive: CognitiveOrchestrator,
                    logger: AthenaLogger, total_trained: int) -> int:
    """
    Phase 3: For domains with mastery > 0.6, brain researches topics.
    SafetyGate pre-filters all queries and content.
    LGSS content filter validates all fetched text.
    Learns from web with reduced confidence (0.7).

    NOTE: total_trained is intentionally NOT incremented in this phase.
    Research generates questions and structures knowledge via curiosity-driven
    exploration, but does not perform direct learn() calls that would count
    toward training steps.

    NOTE: Web research requires network access. If unavailable, this phase
    uses curiosity-driven question generation as a knowledge structuring
    exercise (no actual web fetches).
    """
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 3: Research (Curiosity-Driven Exploration)")
    logger.log("=" * 70)

    masteries = socratic.mastery.all_masteries()
    research_domains = {d: m for d, m in masteries.items()
                        if m > RESEARCH_MASTERY_THRESHOLD}

    if not research_domains:
        logger.log(f"Phase 3 SKIPPED — no domains above {RESEARCH_MASTERY_THRESHOLD} mastery")
        return total_trained

    logger.log(f"Domains eligible for research: {list(research_domains.keys())}")

    for domain, mastery in sorted(research_domains.items(), key=lambda x: -x[1]):
        if _shutdown_requested:
            logger.log("Shutdown requested, aborting phase")
            break
        logger.log(f"\n--- Research: {domain} (mastery={mastery:.3f}) ---")

        # Curiosity generates research topics
        topics = cognitive.generate_questions(domain, domain, max_questions=3)
        logger.log(f"  Curiosity questions: {topics}")

        # Active learner researches topics (no web fetch — deferred mode)
        results = active_learner.research_topic(domain, domain, fetch_fn=None)
        logger.log(f"  Research results: {len(results)} items "
                    f"(deferred — no web access)")

        for r in results:
            logger.metric({
                "phase": 3, "domain": domain,
                "question": r.get("question", ""),
                "status": r.get("status", ""),
            })

    # Introspection after research
    progress = cognitive.assess_learning_progress()
    logger.log(f"\nIntrospection after Phase 3:")
    logger.log(f"  Accuracy: {progress['accuracy']:.4f}")

    logger.log(f"\nPhase 3 complete — {total_trained:,} total training steps")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 4: Creative Exam + Final Consolidation + Save
# ---------------------------------------------------------------------------

def phase4_exam_and_save(brain, active_learner: ActiveLearner,
                         socratic: SocraticTrainer,
                         cognitive: CognitiveOrchestrator,
                         logger: AthenaLogger, total_trained: int) -> int:
    """
    Phase 4: Creative exam, hard-item review, final consolidation, save.
    - For domains with mastery > 0.8: brain generates content, self-grades
    - Final hard-item review from replay buffer
    - Full C-level consolidation before saving
    """
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 4: Creative Exam & Final Consolidation")
    logger.log("=" * 70)

    # Creative exam for advanced domains
    masteries = socratic.mastery.all_masteries()
    advanced_domains = {d: m for d, m in masteries.items()
                        if m > CREATIVE_MASTERY_THRESHOLD}

    if advanced_domains:
        logger.log(f"Creative exam domains: {list(advanced_domains.keys())}")
        for domain, mastery in advanced_domains.items():
            if _shutdown_requested:
                logger.log("Shutdown requested, aborting phase")
                break
            logger.log(f"\n--- Creative Exam: {domain} (mastery={mastery:.3f}) ---")
            # Generate a few test prompts and self-grade
            dummy_features = [random.random() for _ in range(ATHENA_NUM_INPUTS)]
            result = active_learner.create_and_grade(
                dummy_features, domain, "0"
            )
            logger.log(f"  Grade: {result.get('grade', 0):.3f}")
            logger.metric({"phase": 4, "domain": domain, **result})
    else:
        logger.log(f"No domains above {CREATIVE_MASTERY_THRESHOLD} mastery — skipping creative exam")

    # ================================================================
    # CREATIVITY TEST: Can the brain create something that hasn't existed?
    # ================================================================
    # Cross-domain prompts force the brain to combine concepts from
    # different training domains — the hallmark of creative thinking.
    # Measures: novelty, coherence, surprise, cross-domain integration.
    logger.log(f"\n--- Creativity Test ---")
    try:
        creativity_result = active_learner.creativity_exam(
            num_inputs=ATHENA_NUM_INPUTS,
            num_trials=10
        )
    except Exception as e:
        logger.log(f"  Creativity exam failed: {e}")
        creativity_result = {"creativity_score": 0, "novelty": 0, "coherence": 0,
                             "surprise": 0, "cross_domain": 0, "trials": []}
    logger.log(f"  Overall creativity score: {creativity_result['creativity_score']:.3f}")
    logger.log(f"  Novelty:      {creativity_result['novelty']:.3f}")
    logger.log(f"  Coherence:    {creativity_result['coherence']:.3f}")
    logger.log(f"  Surprise:     {creativity_result['surprise']:.3f}")
    logger.log(f"  Cross-domain: {creativity_result['cross_domain']:.3f}")
    for trial in creativity_result.get("trials", []):
        novelty_score = trial.get('novelty', 0.0)

        # Verify creative output consistency via backward chaining
        if hasattr(cognitive, 'backward_chain'):
            try:
                creative_output = str(trial.get('predicted_label', ''))[:256]
                consistency = cognitive.backward_chain(creative_output)
                if consistency >= 0:
                    adjusted_score = novelty_score * (0.5 + 0.5 * consistency)
                    trial['novelty_adjusted'] = adjusted_score
                    trial['consistency'] = consistency
            except Exception:
                pass

        logger.log(f"    [{trial['domain_a']}+{trial['domain_b']}] "
                    f"N={trial['novelty']:.2f} C={trial['coherence']:.2f} "
                    f"S={trial['surprise']:.2f} X={trial['cross_domain']:.2f} "
                    f"→ {trial['predicted_label']}")
    logger.metric({"phase": 4, "test": "creativity", **{
        k: v for k, v in creativity_result.items() if k != "trials"
    }})

    # Final hard-item review from replay buffer
    logger.log(f"\n--- Final Hard-Item Review ({len(socratic.replay_buffer)} items) ---")
    hard_items = socratic.replay_buffer[:500]  # Review up to 500 hard items
    if hard_items:
        batch = [(item.features, item.label) for item in hard_items]
        result = socratic.train_batch_socratic(batch, "hard_review")
        total_trained += result["batch_size"]
        logger.log(f"  Hard review: acc={result['batch_accuracy']:.4f}, "
                    f"reviewed={result['batch_size']}")

    # Final C-level consolidation — full 10-cycle for best knowledge integration
    logger.log(f"\n--- Final Consolidation ---")
    cognitive.cache_communities()  # Final topology snapshot for community-aware consolidation
    cognitive.consolidate(mode="full")

    # Final consolidation pass on built-in data
    if BENCHMARKS_AVAILABLE:
        logger.log("Final consolidation on benchmark datasets...")
        consolidation_sets = [
            ("wine", WineDataset()), ("breast_cancer", BreastCancerDataset()),
            ("mmlu", MMLUDataset()), ("arc_easy", ARCDataset()),
        ]
        all_examples = []
        for domain_name, ds in consolidation_sets:
            examples = ds.get_examples()
            for ex in examples:
                feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
                all_examples.append({"features": feats,
                                     "label": domain_label(domain_name, ex["label"])})

        for epoch in range(2):
            if _shutdown_requested:
                logger.log("Shutdown requested, aborting phase")
                break
            batch = [(ex["features"], str(ex["label"])) for ex in all_examples]
            random.shuffle(batch)
            result = socratic.train_batch_socratic(batch, "final_consolidation")
            total_trained += result["batch_size"]
            logger.log(f"  Final consolidation {epoch+1}/2: "
                        f"acc={result['batch_accuracy']:.4f}")

    # Save model
    ATHENA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.log(f"Saving Athena to {ATHENA_MODEL_PATH}...")
    brain.save(str(ATHENA_MODEL_PATH))
    logger.log("Athena model saved successfully")

    # Also save as .bin for brain_manager compatibility
    bin_path = ATHENA_MODEL_DIR / "athena.bin"
    brain.save(str(bin_path))
    logger.log(f"Also saved as {bin_path}")

    # Get final metrics
    try:
        probe = brain.probe()
        logger.log(f"Final brain stats:")
        logger.log(f"  Neurons: {probe.get('num_neurons', 'N/A'):,}")
        logger.log(f"  Synapses: {probe.get('num_synapses', 'N/A'):,}")
        logger.log(f"  Total trained: {total_trained:,}")
    except Exception:
        logger.log(f"  Total trained: {total_trained:,}")

    # Final evaluation on benchmark datasets
    if BENCHMARKS_AVAILABLE:
        logger.log("\nFinal evaluation:")
        eval_datasets = [
            ("wine", WineDataset()),
            ("breast_cancer", BreastCancerDataset()),
            ("mmlu", MMLUDataset()),
            ("arc_easy", ARCDataset()),
            ("hellaswag", HellaSwagDataset()),
            ("winogrande", WinograndeDataset()),
        ]
        for domain_name, ds in eval_datasets:
            examples = ds.get_examples()
            correct = 0
            total = 0
            for ex in examples:
                feats = _pad_or_truncate(ex["features"], ATHENA_NUM_INPUTS)
                expected = domain_label(domain_name, ex["label"])
                try:
                    result = brain.predict_fast(feats)
                    if result is None:
                        total += 1
                        continue
                    pred, conf = result
                    if pred == expected:
                        correct += 1
                except Exception:
                    pass
                total += 1
            acc = correct / max(total, 1)
            logger.log(f"  {domain_name:20s}: {acc:.4f} ({correct}/{total})")
            logger.metric({
                "phase": 4, "eval_dataset": domain_name,
                "accuracy": acc, "correct": correct, "total": total,
            })

    # Final domain mastery report
    logger.log(f"\n{socratic.get_domain_report()}")

    # Stage stats from active learner
    stage_stats = active_learner.get_stage_stats()
    logger.log("\nActive Learning Stage Stats:")
    for stage, stats in stage_stats.items():
        if stats["attempts"] > 0:
            logger.log(f"  Stage {stage} ({stats['name']}): "
                        f"{stats['successes']}/{stats['attempts']} "
                        f"({stats['success_rate']:.1%})")

    # Safety gate stats
    safety_stats = active_learner.safety.get_stats()
    logger.log(f"\nSafety Gate: {safety_stats['total_decisions']} decisions, "
                f"{safety_stats['rejected']} rejected "
                f"({safety_stats['rejection_rate']:.1%})")

    # Write metadata JSON
    metadata = {
        "name": "nimcp_athena_foundation_v1.0",
        "display_name": "Athena Foundation v1.0",
        "version": "1.0",
        "size": "athena",
        "type": "foundation",
        "description": "1M-neuron pretrained foundation model — parallel school training",
        "architecture": {
            "neurons": ATHENA_NEURONS,
            "topology": "scale_free",
            "num_inputs": ATHENA_NUM_INPUTS,
            "num_outputs": ATHENA_NUM_OUTPUTS,
        },
        "training": {
            "total_examples": total_trained,
            "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "training_framework": f"NIMCP {nimcp.version()}",
            "training_mode": "parallel_school" if SCHOOL_AVAILABLE else "sequential_legacy",
            "num_instructors": 23 if SCHOOL_AVAILABLE else 0,
            "method": "parallel_school_7_teaching_methods",
            "phases": [
                "orientation_warmup",
                "parallel_school_23_instructors",
                "creative_exam_consolidation",
            ] if SCHOOL_AVAILABLE else [
                "worksheets_socratic",
                "guided_study_streaming",
                "curiosity_research",
                "creative_exam_consolidation",
            ],
            "layers": [
                "SocraticTrainer (predict-before-learn, Leitner replay)",
                "InstructorAgent (7 teaching methods per domain)",
                "School (23 parallel instructors, recess, dashboard)",
                "DataSkeptic (7-dimension quality grading)",
                "SafetyGate (Python pre-filter + C LGSS)",
            ],
            "domain_masteries": socratic.mastery.all_masteries(),
            "creativity": {
                "score": creativity_result.get("creativity_score", 0),
                "novelty": creativity_result.get("novelty", 0),
                "coherence": creativity_result.get("coherence", 0),
                "surprise": creativity_result.get("surprise", 0),
                "cross_domain": creativity_result.get("cross_domain", 0),
            },
            "replay_buffer_final": len(socratic.replay_buffer),
            "datasets_used": (
                ["wine", "breast_cancer", "fashion_mnist",
                 "mmlu", "arc_easy", "hellaswag", "winogrande",
                 "ethics", "nback", "sequences"] +
                (["foundation_streaming"] if HF_AVAILABLE else [])
            ),
        },
        "resources": {
            "file_size_mb": round(ATHENA_MODEL_PATH.stat().st_size / 1e6, 1) if ATHENA_MODEL_PATH.exists() else 0,
            "recommended_ram_mb": 8192,
        },
        "is_baseline": True,
        "auto_apply_to_new_brains": True,
        "metadata": {
            "created_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "created_by": "Athena Socratic Training Pipeline",
            "license": "MIT",
        },
    }

    metadata_path = ATHENA_MODEL_DIR / "nimcp_athena_foundation_v1.0.json"
    try:
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.log(f"Metadata saved: {metadata_path}")
    except Exception as e:
        logger.log(f"WARNING: Metadata save failed: {e}")

    return total_trained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PROGRESS_FILE = ATHENA_CHECKPOINT_DIR / "athena_progress.json"


def save_progress(phase: str, checkpoint_path: Path, total_trained: int) -> None:
    """Save training progress so --resume knows where to continue."""
    ATHENA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    progress = {
        "completed_phase": phase,
        "checkpoint": str(checkpoint_path),
        "total_trained": total_trained,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def load_progress() -> dict | None:
    """Load training progress from previous run."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return None


def find_latest_checkpoint() -> Path | None:
    """Find the most recent checkpoint file."""
    # First check progress file for the exact checkpoint
    progress = load_progress()
    if progress and Path(progress["checkpoint"]).exists():
        return Path(progress["checkpoint"])
    # Fall back to most recent school checkpoint by mtime
    if not ATHENA_CHECKPOINT_DIR.exists():
        return None
    checkpoints = sorted(ATHENA_CHECKPOINT_DIR.glob("athena_school_*.bin"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def _parse_cli_arg(name, default, cast=int, min_val=None, max_val=None):
    """Parse --name VALUE from sys.argv, return default if missing.

    Validates that the value can be cast and falls within [min_val, max_val].
    """
    if name in sys.argv:
        idx = sys.argv.index(name)
        if idx + 1 < len(sys.argv) and (not sys.argv[idx + 1].startswith("-") or sys.argv[idx + 1][1:2].isdigit()):
            try:
                value = cast(sys.argv[idx + 1])
            except ValueError:
                print(f"WARNING: Invalid value for {name}: {sys.argv[idx + 1]!r} "
                      f"(expected {cast.__name__}), using default {default}")
                return default
            if min_val is not None and value < min_val:
                print(f"WARNING: {name}={value} below minimum {min_val}, "
                      f"clamping to {min_val}")
                return min_val
            if max_val is not None and value > max_val:
                print(f"WARNING: {name}={value} above maximum {max_val}, "
                      f"clamping to {max_val}")
                return max_val
            return value
    return default


def main() -> int:
    global ATHENA_NUM_INPUTS, ATHENA_NUM_OUTPUTS, PHASE2_MAX_PER_DATASET, _clean_exit

    # Parse CLI flags
    resume_path = None
    if "--resume" in sys.argv:
        idx = sys.argv.index("--resume")
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            resume_path = Path(sys.argv[idx + 1])
        else:
            resume_path = find_latest_checkpoint()
        if resume_path is None or not resume_path.exists():
            print(f"ERROR: No checkpoint found to resume from")
            sys.exit(1)

    PHASE2_MAX_PER_DATASET = _parse_cli_arg("--examples-per-domain", PHASE2_MAX_PER_DATASET,
                                            min_val=100, max_val=10_000_000)
    cli_max_concurrent = _parse_cli_arg("--max-concurrent", 4,
                                         min_val=1, max_val=32)
    cli_min_accuracy = _parse_cli_arg("--min-domain-accuracy", 0.0, float,
                                       min_val=0.0, max_val=1.0)
    cli_max_retry = _parse_cli_arg("--max-retry-passes", 5,
                                    min_val=1, max_val=100)

    logger = AthenaLogger(ATHENA_LOG_DIR)

    # Mutable container so atexit handler can reference the brain
    _brain_ref = [None]

    # Register emergency checkpoint save on exit
    def _emergency_save():
        if _clean_exit:
            # Normal exit path already saved and shut down — nothing to do.
            return
        if _force_exit:
            # Second signal received — skip save to avoid corruption,
            # then force-exit immediately.
            print("[Athena] Force exit — skipping checkpoint save")
            logger.close()
            os._exit(1)
        if _brain_ref[0] is not None:
            try:
                ckpt = ATHENA_CHECKPOINT_DIR / "athena_emergency.bin"
                ATHENA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
                _brain_ref[0].save(str(ckpt))
                print(f"[Athena] Emergency checkpoint saved: {ckpt}")
            except Exception as e:
                print(f"[Athena] Emergency checkpoint save failed: {e}")
        logger.close()

    atexit.register(_emergency_save)

    logger.log("=" * 70)
    logger.log("ATHENA FOUNDATION MODEL TRAINING — PARALLEL SCHOOL")
    if resume_path:
        logger.log(f"RESUMING from checkpoint: {resume_path.name}")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Target neurons: {ATHENA_NEURONS:,}")
    logger.log(f"Inputs: {ATHENA_NUM_INPUTS}, Outputs: {ATHENA_NUM_OUTPUTS}")
    logger.log(f"Model output: {ATHENA_MODEL_PATH}")
    logger.log(f"Training mode: {'parallel_school' if SCHOOL_AVAILABLE else 'sequential_legacy'}")
    logger.log("Layers: Socratic + Instructor + School + DataSkeptic + Safety")
    logger.log("=" * 70)

    # Create checkpoint directory
    ATHENA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize NIMCP
    logger.log("Initializing NIMCP...")
    nimcp.init()

    if resume_path:
        # Resume from checkpoint
        logger.log(f"Loading brain from checkpoint: {resume_path}")
        brain = nimcp.Brain.load(str(resume_path))
        logger.log(f"Neuron count: {brain.get_neuron_count():,}")

        # Check dimension compatibility — checkpoint may have different dimensions
        probe_info = brain.probe()
        ckpt_inputs = probe_info.get("num_inputs", 0)
        ckpt_outputs = probe_info.get("num_outputs", 0)
        if ckpt_inputs != ATHENA_NUM_INPUTS or ckpt_outputs != ATHENA_NUM_OUTPUTS:
            logger.log(f"WARNING: Checkpoint dimensions ({ckpt_inputs}in/{ckpt_outputs}out) "
                        f"don't match config ({ATHENA_NUM_INPUTS}in/{ATHENA_NUM_OUTPUTS}out)")
            logger.log(f"Using checkpoint dimensions — overriding ATHENA_NUM_INPUTS/OUTPUTS")
            ATHENA_NUM_INPUTS = ckpt_inputs
            ATHENA_NUM_OUTPUTS = ckpt_outputs
    else:
        # Create brain directly at target neuron count
        logger.log(f"Creating Athena brain with {ATHENA_NEURONS:,} neurons...")
        brain = nimcp.Brain(
            "Athena",
            nimcp.BRAIN_LARGE,
            nimcp.TASK_CLASSIFICATION,
            ATHENA_NUM_INPUTS,
            ATHENA_NUM_OUTPUTS,
            neuron_count=ATHENA_NEURONS,
        )
        logger.log(f"Neuron count: {brain.get_neuron_count():,}")

    _brain_ref[0] = brain  # Enable emergency checkpoint in atexit handler

    # Configure training pipeline (LR scheduler, regularization, gradient management)
    # Essential for effective training — especially for checkpoints from older saves
    # NOTE: Python-side CosineAnnealingLR produces LR in [0.05, 0.5] passed to brain.learn().
    # The C-level configure_training sets weight_decay and gradient clipping parameters
    # but its learning_rate is overridden by the explicit LR in learn() calls.
    try:
        brain.configure_training(
            learning_rate=0.001,
            weight_decay=0.0001,
            gradient_clip=1.0,
        )
        logger.log("Configured training pipeline (LR=0.001, L2=0.0001, grad_clip=1.0)")
    except Exception as e:
        logger.log(f"Training pipeline configuration failed: {e}")

    # Enable multi-network ensemble training (LNN + CNN alongside adaptive SNN)
    try:
        brain.enable_multi_network()
        logger.log("Enabled multi-network ensemble training (Adaptive + LNN + CNN)")
    except Exception as e:
        logger.log(f"Multi-network training not available: {e}")

    # Verify GPU status
    try:
        probe = brain.probe()
        gpu_status = probe.get("gpu_available", False)
        logger.log(f"GPU enabled: {gpu_status}")
    except Exception:
        logger.log("GPU status: unknown (probe unavailable)")

    # Initialize active learning layers
    logger.log("Initializing active learning layers...")
    socratic = SocraticTrainer(brain, SocraticConfig())
    safety = SafetyGate(brain, SafetyConfig())
    cognitive = CognitiveOrchestrator(brain)
    active_learner = ActiveLearner(brain, socratic, safety, cognitive)
    logger.log("  Layer 1: SocraticTrainer (predict-before-learn, Leitner replay)")
    logger.log("  Layer 2: InstructorAgent (7 teaching methods per domain)")
    logger.log("  Layer 3: School (23 parallel instructors)")
    logger.log("  Layer 4: DataSkeptic (7-dimension quality grading)")
    logger.log(f"  Layer 5: SafetyGate (LGSS={safety._has_lgss})")

    # Save initial checkpoint
    initial_ckpt = ATHENA_CHECKPOINT_DIR / "athena_initial.bin"
    brain.save(str(initial_ckpt))
    logger.log(f"Initial checkpoint saved: {initial_ckpt}")

    # ===== SMOKE TEST: Catch problems in <60s instead of hours =====
    logger.log("-" * 70)
    logger.log("SMOKE TEST: Validating pipeline before long training run...")
    smoke_ok = True
    try:
        # 1. Verify predict works with correct dimensions
        test_input = [0.1] * ATHENA_NUM_INPUTS
        result = brain.predict(test_input)
        if result is None:
            logger.log("  FAIL: brain.predict() returned None")
            smoke_ok = False
        else:
            decision = result[0] if isinstance(result, tuple) else result
            logger.log(f"  OK: predict returns result (decision={decision})")

        # 2. Verify learn works
        brain.learn(test_input, "smoke:test_label", 0.1)
        logger.log("  OK: brain.learn() accepted input")

        # 3. Verify encoding pipeline (text, QA, image)
        if BENCHMARKS_AVAILABLE:
            from benchmark_datasets import text_to_features, encode_qa, FashionMNISTDataset
            tf = text_to_features("smoke test sentence", ATHENA_NUM_INPUTS)
            assert len(tf) == ATHENA_NUM_INPUTS, f"text_to_features: {len(tf)} != {ATHENA_NUM_INPUTS}"
            qa = encode_qa("What is 2+2?", "Four", ATHENA_NUM_INPUTS)
            assert len(qa) == ATHENA_NUM_INPUTS, f"encode_qa: {len(qa)} != {ATHENA_NUM_INPUTS}"
            fds = FashionMNISTDataset()
            fex = fds.get_examples()
            if fex:
                raw_len = len(fex[0]["features"])
                # Fashion-MNIST returns 256 pooled features — training pads to ATHENA_NUM_INPUTS
                assert raw_len > 0, f"Fashion-MNIST: empty features"
            logger.log(f"  OK: encoding pipeline (text={len(tf)}, qa={len(qa)}, "
                       f"fashion={raw_len if fex else '?'} raw, padded to {ATHENA_NUM_INPUTS})")

        # 4. Verify domain labels are prefixed
        test_label = domain_label("wine", 0)
        assert ":" in test_label, f"domain_label missing prefix: {test_label}"
        logger.log(f"  OK: domain labels prefixed ({test_label})")

        # 5. Quick learn+predict cycle to verify learning signal
        for i in range(5):
            features = [float(i == j) for j in range(ATHENA_NUM_INPUTS)]
            brain.learn(features, f"smoke:{i}", 0.5)
        result2 = brain.predict([1.0] + [0.0] * (ATHENA_NUM_INPUTS - 1))
        dec2 = result2[0] if isinstance(result2, tuple) else result2
        logger.log(f"  OK: 5-step train+predict cycle (decision={dec2 if result2 else 'None'})")

    except Exception as e:
        logger.log(f"  FAIL: {e}")
        smoke_ok = False

    if not smoke_ok:
        logger.log("SMOKE TEST FAILED — aborting to avoid wasting hours")
        brain.save(str(ATHENA_CHECKPOINT_DIR / "athena_smoke_fail.bin"))
        _brain_ref[0] = None
        _clean_exit = True
        del brain
        nimcp.shutdown()
        sys.exit(1)

    logger.log("SMOKE TEST PASSED — proceeding with training")
    logger.log("-" * 70)

    report_card = None

    # Determine which phase to resume from
    resume_phase = None
    resume_trained = 0
    if resume_path:
        progress = load_progress()
        if progress:
            resume_phase = progress.get("completed_phase")
            resume_trained = progress.get("total_trained", 0)
            logger.log(f"Resume: last completed phase = {resume_phase}, "
                        f"total trained = {resume_trained:,}")
        else:
            logger.log("Resume: no progress file — skipping Phase 0 only")
            resume_phase = "phase0"

    # Helper: emergency save and return on shutdown.
    # Uses _brain_ref[0] (mutable container) instead of capturing `brain` directly,
    # which avoids UnboundLocalError if the local `brain` name has been deleted or
    # if Python treats `del brain` as making `brain` local in this nested scope.
    def _shutdown_save_and_return():
        global _clean_exit
        logger.log("Shutdown requested — emergency save before exit")
        try:
            if _brain_ref[0] is not None:
                _brain_ref[0].save(str(ATHENA_CHECKPOINT_DIR / "athena_emergency.bin"))
                logger.log("Emergency checkpoint saved")
        except Exception as e:
            logger.log(f"Emergency save failed: {e}")
        # Null out the ref so atexit handler doesn't double-save
        _brain_ref[0] = None
        _clean_exit = True
        logger.close()
        gc.collect()
        try:
            nimcp.shutdown()
        except Exception:
            pass

    if SCHOOL_AVAILABLE:
        # ===== NEW PIPELINE: Parallel School =====

        # Phase 0: Orientation (quick warm-up on built-in benchmarks)
        if resume_phase and phase_reached(resume_phase, "phase0"):
            logger.log("\n" + "=" * 70)
            logger.log(f"SKIPPING Phase 0 (completed in previous run)")
            logger.log("=" * 70)
            total_trained = resume_trained
            health_check(brain, logger, "Resume", abort_on_fail=True)
            conversation_probe(brain, logger, "Resume (from checkpoint)")
        else:
            total_trained = phase0_orientation(brain, socratic, cognitive, logger)

            p0_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_phase0.bin"
            brain.save(str(p0_ckpt))
            save_progress("phase0", p0_ckpt, total_trained)
            logger.log(f"Phase 0 checkpoint saved: {p0_ckpt}")

            health_check(brain, logger, "Phase 0", abort_on_fail=True)
            conversation_probe(brain, logger, "Phase 0 (Orientation)")

        # Check for shutdown between Phase 0 and Phase 1
        if _shutdown_requested:
            _shutdown_save_and_return()
            return 0

        # Phase 1: Parallel School (23 instructors)
        if resume_phase and phase_reached(resume_phase, "phase1"):
            logger.log("\n" + "=" * 70)
            logger.log(f"SKIPPING Phase 1 (completed in previous run)")
            logger.log("=" * 70)
        else:
            result = phase1_parallel_school(brain, socratic, cognitive,
                                            logger, total_trained,
                                            max_concurrent_instructors=cli_max_concurrent,
                                            min_domain_accuracy=cli_min_accuracy,
                                            max_retry_passes=cli_max_retry)
            if isinstance(result, tuple):
                total_trained, report_card = result
            else:
                total_trained = result

            p1_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_school.bin"
            brain.save(str(p1_ckpt))
            save_progress("phase1", p1_ckpt, total_trained)
            logger.log(f"School checkpoint saved: {p1_ckpt}")

            health_check(brain, logger, "Phase 1 (School)", abort_on_fail=True)
            conversation_probe(brain, logger, "Phase 1 (Parallel School)")

    else:
        # ===== LEGACY PIPELINE: Sequential phases =====
        logger.log("Using legacy sequential pipeline (school module not available)")

        if not (resume_phase and phase_reached(resume_phase, "legacy_phase1")):
            total_trained = phase1_worksheets(brain, socratic, cognitive, logger)
            p = ATHENA_CHECKPOINT_DIR / "athena_after_phase1.bin"
            brain.save(str(p))
            save_progress("legacy_phase1", p, total_trained)
            health_check(brain, logger, "Phase 1 (Worksheets)", abort_on_fail=True)
            conversation_probe(brain, logger, "Phase 1 (Worksheets)")
        else:
            total_trained = resume_trained
            logger.log("SKIPPING Phase 1 (completed in previous run)")

        # Check for shutdown between Phase 1 and Phase 2
        if _shutdown_requested:
            _shutdown_save_and_return()
            return 0

        if not (resume_phase and phase_reached(resume_phase, "legacy_phase2")):
            total_trained = phase2_guided_study(brain, socratic, cognitive,
                                                 logger, total_trained)
            p = ATHENA_CHECKPOINT_DIR / "athena_after_phase2.bin"
            brain.save(str(p))
            save_progress("legacy_phase2", p, total_trained)
            health_check(brain, logger, "Phase 2 (Guided Study)", abort_on_fail=True)
            conversation_probe(brain, logger, "Phase 2 (Guided Study)")
        else:
            logger.log("SKIPPING Phase 2 (completed in previous run)")

        # Check for shutdown between Phase 2 and Phase 3
        if _shutdown_requested:
            _shutdown_save_and_return()
            return 0

        if not (resume_phase and phase_reached(resume_phase, "legacy_phase3")):
            total_trained = phase3_research(brain, active_learner, socratic,
                                             cognitive, logger, total_trained)
            p = ATHENA_CHECKPOINT_DIR / "athena_after_phase3.bin"
            brain.save(str(p))
            save_progress("legacy_phase3", p, total_trained)
            health_check(brain, logger, "Phase 3 (Research)", abort_on_fail=True)
        else:
            logger.log("SKIPPING Phase 3 (completed in previous run)")

    # Check for shutdown before proceeding to exam phase
    if _shutdown_requested:
        logger.log("Shutdown requested -- skipping to emergency save")
        brain.save(str(ATHENA_CHECKPOINT_DIR / "athena_emergency.bin"))
        _brain_ref[0] = None
        _clean_exit = True
        logger.close()
        del brain
        gc.collect()
        try:
            nimcp.shutdown()
        except Exception:
            pass
        return 0

    # Final health check + conversation probe before exam
    health_check(brain, logger, "Pre-Exam", abort_on_fail=True)
    conversation_probe(brain, logger, "Pre-Exam (All Training Complete)")

    if _shutdown_requested:
        logger.log("Shutdown requested -- skipping to emergency save")
        brain.save(str(ATHENA_CHECKPOINT_DIR / "athena_emergency.bin"))
        _brain_ref[0] = None
        _clean_exit = True
        logger.close()
        del brain
        gc.collect()
        try:
            nimcp.shutdown()
        except Exception:
            pass
        return 0

    # Phase 4 (Final): Creative Exam + Final Consolidation + Save
    total_trained = phase4_exam_and_save(brain, active_learner, socratic,
                                          cognitive, logger, total_trained)

    # Done
    elapsed = time.time() - logger.start_time
    logger.log("\n" + "=" * 70)
    logger.log("ATHENA TRAINING COMPLETE")
    logger.log(f"Training mode: {'parallel_school' if SCHOOL_AVAILABLE else 'sequential_legacy'}")
    if report_card:
        logger.log(f"Instructors: {report_card.get('num_instructors', 0)}")
        logger.log(f"Graduated: {report_card.get('graduated', 0)}/"
                    f"{report_card.get('total_domains', 0)} domains")
    logger.log(f"Total training steps: {total_trained:,}")
    logger.log(f"Total time: {elapsed/3600:.2f} hours")
    logger.log(f"Model saved to: {ATHENA_MODEL_PATH}")
    logger.log("=" * 70)

    # Clean exit — set _clean_exit before shutdown so atexit handler is a no-op
    _brain_ref[0] = None
    _clean_exit = True
    logger.close()
    del brain
    gc.collect()
    try:
        nimcp.shutdown()
    except Exception:
        pass  # Known segfault risk, but attempt cleanup
    return 0


if __name__ == "__main__":
    sys.exit(main())
