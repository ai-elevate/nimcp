#!/usr/bin/env python3
"""
NIMCP Athena Foundation Model Training — Socratic Active Learning
==================================================================

WHAT: Train a 1M-neuron brain called "Athena" to serve as the pretrained
      baseline for all future brains.
WHY:  Every new brain should start from a trained baseline rather than
      random initialization — dramatically faster convergence on new tasks.
HOW:  4-phase Socratic active learning pipeline:
      Phase 1: Worksheets — built-in benchmarks with predict-before-learn
      Phase 2: Guided Study — HuggingFace streaming with domain mastery tracking
      Phase 3: Research — web-augmented learning (safety-gated) for advanced domains
      Phase 4: Creative Exam — content creation, self-grading, final consolidation

Layers:
  Layer 1 (SocraticTrainer): Predict → adaptive confidence → teach → replay
  Layer 2 (ActiveLearner):   Worksheets → explain → research → create
  Layer 3 (CognitiveOrchestrator): Curiosity, introspection, consolidation
  Layer 4 (SafetyGate):      Python pre-filter + C LGSS content filter

Designed to run via nohup:
  nohup python3 scripts/train_athena.py > athena_training.log 2>&1 &

All output goes to file, no interactive prompts.
"""

import gc
import json
import os
import random
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
for p in [BUILD_PYTHON, PROJECT_ROOT / "build/lib/python"]:
    if p.exists():
        sys.path.insert(0, str(p))
        break

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
    print("[Athena] WARNING: benchmark_datasets not found — Phase 1 skipped")

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

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ATHENA_NEURONS = 1_000_000
ATHENA_NUM_INPUTS = 128
ATHENA_NUM_OUTPUTS = 32

# Output paths
ATHENA_MODEL_DIR = PROJECT_ROOT / "models" / "pretrained" / "athena" / "v1.0"
ATHENA_MODEL_PATH = ATHENA_MODEL_DIR / "nimcp_athena_foundation_v1.0.nimcp"
ATHENA_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "athena"
ATHENA_LOG_DIR = PROJECT_ROOT / "logs"

# Training hyperparameters
PHASE1_EPOCHS = 30        # Epochs per built-in dataset
PHASE2_MAX_PER_DATASET = 50_000   # Max examples per streaming dataset
PHASE2_BATCH_SIZE = 1000          # Streaming batch size
PHASE2_CHECKPOINT_INTERVAL = 10_000  # Checkpoint every N examples
INTROSPECTION_INTERVAL = 5000     # Metacognitive check every N examples

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class AthenaLogger:
    """File-based logger safe for nohup."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = log_dir / f"athena_train_{ts}.log"
        self.metrics_file = log_dir / f"athena_metrics_{ts}.jsonl"
        self.start_time = time.time()

    def log(self, msg: str):
        elapsed = time.time() - self.start_time
        line = f"[{elapsed:10.1f}s] {msg}"
        print(line, flush=True)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")

    def metric(self, data: dict):
        data["timestamp"] = time.time()
        data["elapsed_s"] = time.time() - self.start_time
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(data) + "\n")


# ---------------------------------------------------------------------------
# Phase 1: Worksheets — Built-in Benchmark Datasets (Socratic)
# ---------------------------------------------------------------------------

def phase1_worksheets(brain, socratic: SocraticTrainer,
                      cognitive: CognitiveOrchestrator,
                      logger: AthenaLogger):
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
        "Wine": "science", "Breast Cancer": "medicine",
        "Fashion-MNIST": "technology", "MMLU": "general",
        "ARC-Easy": "science", "HellaSwag": "general",
        "Winogrande": "general", "N-back": "psychology",
        "Ethics": "philosophy", "Sequences": "math",
    }

    # Structured ML datasets
    ml_datasets = [
        ("Wine", WineDataset()),
        ("Breast Cancer", BreastCancerDataset()),
        ("Fashion-MNIST", FashionMNISTDataset()),
    ]

    # Text-based QA datasets
    qa_datasets = [
        ("MMLU", MMLUDataset()),
        ("ARC-Easy", ARCDataset()),
        ("HellaSwag", HellaSwagDataset()),
        ("Winogrande", WinograndeDataset()),
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

    # ML datasets — pad/truncate features to ATHENA_NUM_INPUTS
    for name, ds in ml_datasets:
        examples = ds.get_examples()
        adapted = []
        for ex in examples:
            feats = ex["features"]
            if len(feats) < ATHENA_NUM_INPUTS:
                feats = feats + [0.0] * (ATHENA_NUM_INPUTS - len(feats))
            elif len(feats) > ATHENA_NUM_INPUTS:
                feats = feats[:ATHENA_NUM_INPUTS]
            adapted.append({"features": feats, "label": ex["label"]})
        all_datasets.append((name, adapted))

    # QA datasets — already 128 features
    for name, ds in qa_datasets:
        all_datasets.append((name, ds.get_examples()))

    all_datasets.append(("N-back", nback_examples))

    ethics_adapted = [{"features": s["features"] + [0.0] * (ATHENA_NUM_INPUTS - len(s["features"])),
                       "label": s["category"]}
                      for s in ethics]
    all_datasets.append(("Ethics", ethics_adapted))

    seq_adapted = [{"features": s["features"] + [0.0] * (ATHENA_NUM_INPUTS - len(s["features"])),
                    "label": str(s["position"] % 4)}
                   for s in seq_patterns]
    all_datasets.append(("Sequences", seq_adapted))

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
        domain = domain_map.get(ds_name, "general")
        logger.log(f"\n--- {ds_name} [{domain}]: {len(examples)} examples, "
                    f"{PHASE1_EPOCHS} epochs (Socratic) ---")

        best_acc = 0.0
        for epoch in range(PHASE1_EPOCHS):
            batch = [(ex["features"], str(ex["label"])) for ex in examples]
            result = socratic.train_batch_socratic(batch, domain)

            total_trained += result["batch_size"]
            best_acc = max(best_acc, result["batch_accuracy"])

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
    cognitive.consolidate()

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
                        logger: AthenaLogger, total_trained: int):
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

        for ds_config in hf_datasets:
            name = ds_config["name"]
            domain = ds_config.get("domain", "general")
            logger.log(f"\n--- Streaming: {name} [{domain}] "
                        f"(mastery={socratic.mastery.mastery(domain):.3f}) ---")

            try:
                # Stream and train with Socratic method
                dataset = processor.load_streaming_dataset(ds_config)
                if dataset is None:
                    continue

                stream = iter(dataset)
                count = 0
                batch = []

                while True:
                    try:
                        example = next(stream)
                    except StopIteration:
                        break

                    result = processor.extract_features_and_label(example, domain)
                    if result is None:
                        continue

                    features, label = result
                    # Socratic: predict-before-learn per example
                    socratic.train_example(features, str(label), domain)
                    count += 1
                    total_trained += 1
                    examples_since_introspection += 1

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
    else:
        # Fallback: minimal streaming with Socratic training
        logger.log("Using fallback streaming (Socratic per-example)")
        for ds_config in hf_datasets:
            name = ds_config["name"]
            domain = ds_config.get("domain", "general")
            hf_dataset = ds_config["hf_dataset"]
            hf_subset = ds_config.get("hf_subset")

            logger.log(f"\n--- Streaming: {name} [{domain}] ---")
            try:
                if hf_subset:
                    dataset = load_dataset(hf_dataset, hf_subset,
                                           split="train", streaming=True)
                else:
                    dataset = load_dataset(hf_dataset, split="train",
                                           streaming=True)

                count = 0
                for example in dataset:
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
                    label_val = example.get("answer", example.get("label", 0))
                    socratic.train_example(features, str(label_val), domain)
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
            except Exception as e:
                logger.log(f"  {name}: ERROR — {e}")
                continue
            gc.collect()

    # Consolidation between Phase 2 and Phase 3
    logger.log(f"\n--- Phase 2 Consolidation ---")
    cognitive.consolidate()

    logger.log(f"\n{socratic.get_domain_report()}")
    logger.log(f"\nPhase 2 complete — {total_trained:,} total training steps")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 3: Research — Web-Augmented Learning (Safety-Gated)
# ---------------------------------------------------------------------------

def phase3_research(brain, active_learner: ActiveLearner,
                    socratic: SocraticTrainer,
                    cognitive: CognitiveOrchestrator,
                    logger: AthenaLogger, total_trained: int):
    """
    Phase 3: For domains with mastery > 0.6, brain researches topics.
    SafetyGate pre-filters all queries and content.
    LGSS content filter validates all fetched text.
    Learns from web with reduced confidence (0.7).

    NOTE: Web research requires network access. If unavailable, this phase
    uses curiosity-driven question generation as a knowledge structuring
    exercise (no actual web fetches).
    """
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 3: Research (Curiosity-Driven Exploration)")
    logger.log("=" * 70)

    masteries = socratic.mastery.all_masteries()
    research_domains = {d: m for d, m in masteries.items() if m > 0.6}

    if not research_domains:
        logger.log("Phase 3 SKIPPED — no domains above 0.6 mastery")
        return total_trained

    logger.log(f"Domains eligible for research: {list(research_domains.keys())}")

    for domain, mastery in sorted(research_domains.items(), key=lambda x: -x[1]):
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
                         logger: AthenaLogger, total_trained: int):
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
    advanced_domains = {d: m for d, m in masteries.items() if m > 0.8}

    if advanced_domains:
        logger.log(f"Creative exam domains: {list(advanced_domains.keys())}")
        for domain, mastery in advanced_domains.items():
            logger.log(f"\n--- Creative Exam: {domain} (mastery={mastery:.3f}) ---")
            # Generate a few test prompts and self-grade
            dummy_features = [random.random() for _ in range(ATHENA_NUM_INPUTS)]
            result = active_learner.create_and_grade(
                dummy_features, domain, "0"
            )
            logger.log(f"  Grade: {result.get('grade', 0):.3f}")
            logger.metric({"phase": 4, "domain": domain, **result})
    else:
        logger.log("No domains above 0.8 mastery — skipping creative exam")

    # ================================================================
    # CREATIVITY TEST: Can the brain create something that hasn't existed?
    # ================================================================
    # Cross-domain prompts force the brain to combine concepts from
    # different training domains — the hallmark of creative thinking.
    # Measures: novelty, coherence, surprise, cross-domain integration.
    logger.log(f"\n--- Creativity Test ---")
    creativity_result = active_learner.creativity_exam(
        num_inputs=ATHENA_NUM_INPUTS,
        num_trials=10
    )
    logger.log(f"  Overall creativity score: {creativity_result['creativity_score']:.3f}")
    logger.log(f"  Novelty:      {creativity_result['novelty']:.3f}")
    logger.log(f"  Coherence:    {creativity_result['coherence']:.3f}")
    logger.log(f"  Surprise:     {creativity_result['surprise']:.3f}")
    logger.log(f"  Cross-domain: {creativity_result['cross_domain']:.3f}")
    for trial in creativity_result.get("trials", []):
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

    # Final C-level consolidation
    logger.log(f"\n--- Final Consolidation ---")
    cognitive.consolidate()

    # Final consolidation pass on built-in data
    if BENCHMARKS_AVAILABLE:
        logger.log("Final consolidation on benchmark datasets...")
        consolidation_sets = [
            WineDataset(), BreastCancerDataset(),
            MMLUDataset(), ARCDataset(),
        ]
        all_examples = []
        for ds in consolidation_sets:
            examples = ds.get_examples()
            for ex in examples:
                feats = ex["features"]
                if len(feats) < ATHENA_NUM_INPUTS:
                    feats = feats + [0.0] * (ATHENA_NUM_INPUTS - len(feats))
                elif len(feats) > ATHENA_NUM_INPUTS:
                    feats = feats[:ATHENA_NUM_INPUTS]
                all_examples.append({"features": feats, "label": ex["label"]})

        for epoch in range(2):
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
            ("Wine", WineDataset()),
            ("Breast Cancer", BreastCancerDataset()),
            ("MMLU", MMLUDataset()),
            ("ARC-Easy", ARCDataset()),
            ("HellaSwag", HellaSwagDataset()),
            ("Winogrande", WinograndeDataset()),
        ]
        for name, ds in eval_datasets:
            examples = ds.get_examples()
            correct = 0
            total = 0
            for ex in examples:
                feats = ex["features"]
                if len(feats) < ATHENA_NUM_INPUTS:
                    feats = feats + [0.0] * (ATHENA_NUM_INPUTS - len(feats))
                elif len(feats) > ATHENA_NUM_INPUTS:
                    feats = feats[:ATHENA_NUM_INPUTS]
                pred, conf = brain.predict(feats)
                if pred == str(ex["label"]):
                    correct += 1
                total += 1
            acc = correct / max(total, 1)
            logger.log(f"  {name:20s}: {acc:.4f} ({correct}/{total})")
            logger.metric({
                "phase": 4, "eval_dataset": name,
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
        "description": "1M-neuron pretrained foundation model — Socratic active learning",
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
            "method": "socratic_active_learning",
            "phases": [
                "worksheets_socratic",
                "guided_study_streaming",
                "curiosity_research",
                "creative_exam_consolidation",
            ],
            "layers": [
                "SocraticTrainer (predict-before-learn, Leitner replay)",
                "ActiveLearner (4-stage progressive)",
                "CognitiveOrchestrator (curiosity, introspection, consolidation)",
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
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.log(f"Metadata saved: {metadata_path}")

    return total_trained


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger = AthenaLogger(ATHENA_LOG_DIR)
    logger.log("=" * 70)
    logger.log("ATHENA FOUNDATION MODEL TRAINING — SOCRATIC ACTIVE LEARNING")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Target neurons: {ATHENA_NEURONS:,}")
    logger.log(f"Inputs: {ATHENA_NUM_INPUTS}, Outputs: {ATHENA_NUM_OUTPUTS}")
    logger.log(f"Model output: {ATHENA_MODEL_PATH}")
    logger.log("Layers: Socratic + Active + Cognitive + Safety")
    logger.log("=" * 70)

    # Create checkpoint directory
    ATHENA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize NIMCP
    logger.log("Initializing NIMCP...")
    nimcp.init()

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

    # Initialize 4-layer active learning system
    logger.log("Initializing Socratic active learning layers...")
    socratic = SocraticTrainer(brain, SocraticConfig())
    safety = SafetyGate(brain, SafetyConfig())
    cognitive = CognitiveOrchestrator(brain)
    active_learner = ActiveLearner(brain, socratic, safety, cognitive)
    logger.log("  Layer 1: SocraticTrainer (predict-before-learn, Leitner replay)")
    logger.log("  Layer 2: ActiveLearner (4-stage progressive)")
    logger.log("  Layer 3: CognitiveOrchestrator (curiosity, introspection, consolidation)")
    logger.log(f"  Layer 4: SafetyGate (LGSS={safety._has_lgss})")

    # Save initial checkpoint
    initial_ckpt = ATHENA_CHECKPOINT_DIR / "athena_initial.bin"
    brain.save(str(initial_ckpt))
    logger.log(f"Initial checkpoint saved: {initial_ckpt}")

    # Phase 1: Worksheets (Socratic predict-before-learn)
    total_trained = phase1_worksheets(brain, socratic, cognitive, logger)

    p1_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_phase1.bin"
    brain.save(str(p1_ckpt))
    logger.log(f"Phase 1 checkpoint saved: {p1_ckpt}")

    # Phase 2: Guided Study (Socratic streaming)
    total_trained = phase2_guided_study(brain, socratic, cognitive,
                                         logger, total_trained)

    p2_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_phase2.bin"
    brain.save(str(p2_ckpt))
    logger.log(f"Phase 2 checkpoint saved: {p2_ckpt}")

    # Phase 3: Research (curiosity-driven exploration)
    total_trained = phase3_research(brain, active_learner, socratic,
                                     cognitive, logger, total_trained)

    p3_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_phase3.bin"
    brain.save(str(p3_ckpt))
    logger.log(f"Phase 3 checkpoint saved: {p3_ckpt}")

    # Phase 4: Creative Exam + Final Consolidation + Save
    total_trained = phase4_exam_and_save(brain, active_learner, socratic,
                                          cognitive, logger, total_trained)

    # Done
    elapsed = time.time() - logger.start_time
    logger.log("\n" + "=" * 70)
    logger.log("ATHENA SOCRATIC TRAINING COMPLETE")
    logger.log(f"Total training steps: {total_trained:,}")
    logger.log(f"Total time: {elapsed/3600:.2f} hours")
    logger.log(f"Model saved to: {ATHENA_MODEL_PATH}")
    logger.log("=" * 70)

    # Clean exit — avoid nimcp.shutdown() segfault
    del brain
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
