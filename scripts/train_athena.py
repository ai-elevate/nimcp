#!/usr/bin/env python3
"""
NIMCP Athena Foundation Model Training
========================================

WHAT: Train a 1M-neuron brain called "Athena" to serve as the pretrained
      baseline for all future brains.
WHY:  Every new brain should start from a trained baseline rather than
      random initialization — dramatically faster convergence on new tasks.
HOW:  Phase 1: Built-in benchmark datasets (fast convergence)
      Phase 2: HuggingFace streaming datasets (deep knowledge)
      Phase 3: Final consolidation + save as pretrained model

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
PHASE1_CONSOLIDATION = 3  # Consolidation epochs after each dataset
PHASE2_MAX_PER_DATASET = 50_000   # Max examples per streaming dataset
PHASE2_BATCH_SIZE = 1000          # Streaming batch size
PHASE2_CHECKPOINT_INTERVAL = 10_000  # Checkpoint every N examples

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
# Phase 1: Built-in Benchmark Datasets
# ---------------------------------------------------------------------------

def phase1_builtin_datasets(brain, logger: AthenaLogger):
    """Train on all built-in benchmark datasets with multiple epochs."""
    if not BENCHMARKS_AVAILABLE:
        logger.log("Phase 1 SKIPPED — benchmark_datasets not available")
        return

    logger.log("=" * 70)
    logger.log("PHASE 1: Built-in Benchmark Datasets")
    logger.log("=" * 70)

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

    # ML datasets — need to pad/truncate features to ATHENA_NUM_INPUTS
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
        examples = ds.get_examples()
        all_datasets.append((name, examples))

    # N-back
    all_datasets.append(("N-back", nback_examples))

    # Ethics
    ethics_adapted = [{"features": s["features"] + [0.0] * (ATHENA_NUM_INPUTS - len(s["features"])),
                       "label": s["category"]}
                      for s in ethics]
    all_datasets.append(("Ethics", ethics_adapted))

    # Sequence patterns
    seq_adapted = [{"features": s["features"] + [0.0] * (ATHENA_NUM_INPUTS - len(s["features"])),
                    "label": str(s["position"] % 4)}
                   for s in seq_patterns]
    all_datasets.append(("Sequences", seq_adapted))

    # Train on each dataset
    total_trained = 0
    for ds_name, examples in all_datasets:
        logger.log(f"\n--- {ds_name}: {len(examples)} examples, {PHASE1_EPOCHS} epochs ---")

        best_acc = 0.0
        for epoch in range(PHASE1_EPOCHS):
            random.shuffle(examples)
            epoch_correct = 0
            epoch_total = 0

            for ex in examples:
                brain.learn(ex["features"], str(ex["label"]))
                pred_label, pred_conf = brain.predict(ex["features"])
                if pred_label == str(ex["label"]):
                    epoch_correct += 1
                epoch_total += 1
                total_trained += 1

            acc = epoch_correct / max(epoch_total, 1)
            best_acc = max(best_acc, acc)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.log(f"  Epoch {epoch+1:2d}/{PHASE1_EPOCHS}: acc={acc:.4f} (best={best_acc:.4f})")
                logger.metric({
                    "phase": 1, "dataset": ds_name, "epoch": epoch + 1,
                    "accuracy": acc, "best_accuracy": best_acc,
                    "total_trained": total_trained,
                })

        logger.log(f"  {ds_name} done — best accuracy: {best_acc:.4f}")

    # Consolidation: replay mixed samples from all datasets
    logger.log(f"\n--- Phase 1 Consolidation: {PHASE1_CONSOLIDATION} epochs ---")
    all_examples = []
    for _, examples in all_datasets:
        # Take a subset from each for consolidation
        all_examples.extend(random.sample(examples, min(200, len(examples))))

    for epoch in range(PHASE1_CONSOLIDATION):
        random.shuffle(all_examples)
        correct = 0
        for ex in all_examples:
            brain.learn(ex["features"], str(ex["label"]))
            pred, _ = brain.predict(ex["features"])
            if pred == str(ex["label"]):
                correct += 1
            total_trained += 1

        acc = correct / len(all_examples)
        logger.log(f"  Consolidation {epoch+1}/{PHASE1_CONSOLIDATION}: acc={acc:.4f}")

    logger.log(f"Phase 1 complete — {total_trained:,} total training steps")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 2: Streaming HuggingFace Datasets
# ---------------------------------------------------------------------------

def phase2_streaming_datasets(brain, logger: AthenaLogger, total_trained: int):
    """Stream from HuggingFace datasets using existing infrastructure."""
    if not HF_AVAILABLE:
        logger.log("Phase 2 SKIPPED — HuggingFace datasets library not available")
        return total_trained

    logger.log("\n" + "=" * 70)
    logger.log("PHASE 2: HuggingFace Streaming Datasets")
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

    # Use the existing StreamingDatasetProcessor if available
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

        for ds_config in hf_datasets:
            name = ds_config["name"]
            logger.log(f"\n--- Streaming: {name} ---")
            try:
                progress = processor.process_dataset_streaming(ds_config)
                total_trained += progress.examples_processed
                logger.log(f"  {name}: {progress.examples_processed:,} examples trained")
                logger.metric({
                    "phase": 2, "dataset": name,
                    "examples": progress.examples_processed,
                    "total_trained": total_trained,
                })
            except Exception as e:
                logger.log(f"  {name}: ERROR — {e}")
                continue

            # Save checkpoint after each dataset
            checkpoint_path = ATHENA_CHECKPOINT_DIR / f"athena_after_{name}.bin"
            try:
                brain.save(str(checkpoint_path))
                logger.log(f"  Checkpoint saved: {checkpoint_path.name}")
            except Exception as e:
                logger.log(f"  Checkpoint save failed: {e}")

            gc.collect()
    else:
        # Fallback: minimal streaming without the StreamingDatasetProcessor
        logger.log("Using fallback streaming (StreamingDatasetProcessor not available)")
        for ds_config in hf_datasets:
            name = ds_config["name"]
            hf_dataset = ds_config["hf_dataset"]
            hf_subset = ds_config.get("hf_subset")

            logger.log(f"\n--- Streaming: {name} ---")
            try:
                if hf_subset:
                    dataset = load_dataset(hf_dataset, hf_subset, split="train", streaming=True)
                else:
                    dataset = load_dataset(hf_dataset, split="train", streaming=True)

                count = 0
                for example in dataset:
                    # Simple text extraction
                    text = ""
                    for key in ("text", "question", "content", "input", "ctx"):
                        if key in example and example[key]:
                            text = str(example[key])
                            break
                    if not text:
                        text = " ".join(str(v) for v in example.values() if isinstance(v, str))

                    if not text.strip():
                        continue

                    features = text_to_features(text, ATHENA_NUM_INPUTS)
                    label_val = example.get("answer", example.get("label", 0))
                    brain.learn(features, str(label_val))
                    count += 1
                    total_trained += 1

                    if count >= PHASE2_MAX_PER_DATASET:
                        break

                    if count % 5000 == 0:
                        logger.log(f"  {name}: {count:,} examples...")

                logger.log(f"  {name}: {count:,} examples trained")
                logger.metric({
                    "phase": 2, "dataset": name,
                    "examples": count, "total_trained": total_trained,
                })

            except Exception as e:
                logger.log(f"  {name}: ERROR — {e}")
                continue

            gc.collect()

    logger.log(f"\nPhase 2 complete — {total_trained:,} total training steps")
    return total_trained


# ---------------------------------------------------------------------------
# Phase 3: Final Consolidation + Save
# ---------------------------------------------------------------------------

def phase3_consolidation_and_save(brain, logger: AthenaLogger, total_trained: int):
    """Final consolidation pass on built-in data, then save as pretrained model."""
    logger.log("\n" + "=" * 70)
    logger.log("PHASE 3: Final Consolidation & Save")
    logger.log("=" * 70)

    # Quick consolidation on built-in datasets
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
            random.shuffle(all_examples)
            correct = 0
            for ex in all_examples:
                brain.learn(ex["features"], str(ex["label"]))
                pred, _ = brain.predict(ex["features"])
                if pred == str(ex["label"]):
                    correct += 1
                total_trained += 1
            acc = correct / len(all_examples)
            logger.log(f"  Final consolidation {epoch+1}/2: acc={acc:.4f}")

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

    # Evaluate on benchmark datasets
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
                "phase": 3, "eval_dataset": name,
                "accuracy": acc, "correct": correct, "total": total,
            })

    # Write metadata JSON
    metadata = {
        "name": "nimcp_athena_foundation_v1.0",
        "display_name": "Athena Foundation v1.0",
        "version": "1.0",
        "size": "athena",
        "type": "foundation",
        "description": "1M-neuron pretrained foundation model — baseline for all new brains",
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
            "phases": ["builtin_benchmarks", "huggingface_streaming", "consolidation"],
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
            "created_by": "Athena Training Pipeline",
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
    logger.log("ATHENA FOUNDATION MODEL TRAINING")
    logger.log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.log(f"Target neurons: {ATHENA_NEURONS:,}")
    logger.log(f"Inputs: {ATHENA_NUM_INPUTS}, Outputs: {ATHENA_NUM_OUTPUTS}")
    logger.log(f"Model output: {ATHENA_MODEL_PATH}")
    logger.log("=" * 70)

    # Create checkpoint directory
    ATHENA_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize NIMCP
    logger.log("Initializing NIMCP...")
    nimcp.init()

    # Create brain directly at target neuron count (no resize needed)
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

    # Save initial checkpoint (pre-training)
    initial_ckpt = ATHENA_CHECKPOINT_DIR / "athena_initial.bin"
    brain.save(str(initial_ckpt))
    logger.log(f"Initial checkpoint saved: {initial_ckpt}")

    # Phase 1: Built-in benchmarks
    total_trained = phase1_builtin_datasets(brain, logger) or 0

    # Save Phase 1 checkpoint
    p1_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_phase1.bin"
    brain.save(str(p1_ckpt))
    logger.log(f"Phase 1 checkpoint saved: {p1_ckpt}")

    # Phase 2: Streaming datasets
    total_trained = phase2_streaming_datasets(brain, logger, total_trained)

    # Save Phase 2 checkpoint
    p2_ckpt = ATHENA_CHECKPOINT_DIR / "athena_after_phase2.bin"
    brain.save(str(p2_ckpt))
    logger.log(f"Phase 2 checkpoint saved: {p2_ckpt}")

    # Phase 3: Final consolidation + save
    total_trained = phase3_consolidation_and_save(brain, logger, total_trained)

    # Done
    elapsed = time.time() - logger.start_time
    logger.log("\n" + "=" * 70)
    logger.log("ATHENA TRAINING COMPLETE")
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
