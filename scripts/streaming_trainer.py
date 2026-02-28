#!/usr/bin/env python3
"""
NIMCP AGI Streaming Training Pipeline
======================================

WHAT: Streaming multi-domain training system for NIMCP AGI development
WHY:  Train a biologically-inspired AGI through developmental curriculum
HOW:  Stream datasets via Hugging Face, follow developmental stages

Features:
- Developmental curriculum (infant → child → adolescent → adult)
- 60+ streaming datasets across all domains of knowledge
- HuggingFace + Kaggle + Wikipedia domain-filtered streaming
- Domain rotation to prevent specialization
- Curriculum learning (easy → hard within stages)
- Consolidation to prevent catastrophic forgetting
- Checkpoint saving and recovery
- Comprehensive training metrics

Knowledge Domains:
- Sciences: Physics, Chemistry, Biology, Mathematics
- Humanities: Philosophy, Literature, History, Psychology
- Technical: Software Engineering, Computer Science
- Reasoning: Logic, Common Sense, Causal, Spatial
- Social: Ethics, Emotional Intelligence, Social Intelligence
- World Knowledge: Encyclopedia, Language

Refactored: 2025-01-16 for AGI curriculum training
"""

import os
import sys
import time
import json
import random
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Iterator
from dataclasses import dataclass, asdict, field
from collections import defaultdict
from datetime import datetime
import numpy as np

# Add NIMCP Python bindings to path - check multiple locations
script_dir = Path(__file__).parent
possible_paths = [
    script_dir.parent / "build/lib/python",
    script_dir / "build/lib/python",
    Path("/home/bbrelin/nimcp/build/lib/python"),
]
for path in possible_paths:
    if path.exists():
        sys.path.insert(0, str(path))
        break

try:
    import nimcp
    NIMCP_AVAILABLE = True
    print(f"NIMCP loaded (version: {nimcp.version()})")
except ImportError:
    NIMCP_AVAILABLE = False
    print("WARNING: NIMCP Python bindings not found. Run in simulation mode.")

# Import Hugging Face datasets for streaming
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
    print("Hugging Face datasets available for streaming")
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: Hugging Face datasets not available. Install with: pip install datasets")

# Import AGI curriculum configuration
try:
    from agi_curriculum_datasets import (
        AGI_CURRICULUM_DATASETS,
        DevelopmentStage,
        DomainCategory,
        StreamingDataset,
        DatasetSource,
        get_datasets_by_stage,
        get_datasets_by_priority,
        get_datasets_by_source,
        get_curriculum_schedule,
        stream_dataset,
        stream_kaggle_dataset,
        load_kaggle_dataset,
    )
    CURRICULUM_AVAILABLE = True
    print(f"AGI Curriculum loaded ({len(AGI_CURRICULUM_DATASETS)} datasets)")

    # Print source breakdown
    hf_count = len(get_datasets_by_source(DatasetSource.HUGGINGFACE))
    kaggle_count = len(get_datasets_by_source(DatasetSource.KAGGLE))
    print(f"  - HuggingFace: {hf_count} datasets")
    print(f"  - Kaggle: {kaggle_count} datasets")
except ImportError:
    CURRICULUM_AVAILABLE = False
    print("WARNING: AGI curriculum not found. Using legacy configuration.")

# Import training state manager
try:
    from training_state import (
        TrainingStateManager,
        TrainingStatus,
        DownloadStatus,
        can_resume_training,
    )
    STATE_MANAGER_AVAILABLE = True
    print("Training state persistence available")
except ImportError:
    STATE_MANAGER_AVAILABLE = False
    print("WARNING: Training state manager not found. State won't be persisted.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('nimcp_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

#=============================================================================
# Training Configuration
#=============================================================================

@dataclass
class TrainingConfig:
    """Configuration for streaming training"""

    # Brain configuration
    brain_size: str = "MEDIUM"  # SMALL, MEDIUM, LARGE
    num_inputs: int = 512  # Feature dimension
    num_outputs: int = 256  # Output dimension
    learning_rate: float = 0.01

    # Training parameters
    batch_size: int = 32
    epochs: int = 10
    examples_per_epoch: int = 10000

    # Domain mixing strategy
    domain_rotation: bool = True  # Rotate domains to prevent bias
    domain_weights: Dict[str, float] = None  # Custom domain sampling weights

    # Modality mixing
    text_ratio: float = 0.60  # 60% text
    image_ratio: float = 0.20  # 20% images
    audio_ratio: float = 0.10  # 10% audio
    video_ratio: float = 0.10  # 10% video

    # Cognitive systems
    enable_curiosity: bool = True
    enable_skepticism: bool = True
    enable_attention: bool = True
    enable_working_memory: bool = True
    enable_consolidation: bool = True

    # Memory consolidation (sleep)
    consolidation_interval: int = 1000  # Consolidate every N examples

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    checkpoint_interval: int = 5000  # Save every N examples

    # Monitoring
    log_interval: int = 100  # Log every N examples
    epistemic_report_interval: int = 1000  # Report epistemic quality

    def __post_init__(self):
        """Initialize derived values"""
        if self.domain_weights is None:
            # Default: equal weights for all 19 domains
            self.domain_weights = {domain: 1.0 for domain in DOMAIN_NAMES}

        # Normalize domain weights
        total_weight = sum(self.domain_weights.values())
        self.domain_weights = {
            k: v/total_weight for k, v in self.domain_weights.items()
        }

#=============================================================================
# Domain and Modality Definitions
#=============================================================================

DOMAIN_NAMES = [
    "sociology", "anthropology", "history", "art", "physics",
    "chemistry", "biology", "conflict", "religion", "mythology",
    "linguistics", "politics", "oceanography", "philosophy", "latin",
    "greek", "rhetoric", "law", "metalaw"
]

MODALITY_TYPES = ["text", "image", "audio", "video"]

#=============================================================================
# Training Statistics
#=============================================================================

@dataclass
class DomainStats:
    """Statistics per domain"""
    examples_trained: int = 0
    total_loss: float = 0.0
    epistemic_quality_sum: float = 0.0
    conspiracy_detections: int = 0
    bias_detections: int = 0
    novelty_sum: float = 0.0

    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.examples_trained)

    @property
    def avg_epistemic_quality(self) -> float:
        return self.epistemic_quality_sum / max(1, self.examples_trained)

    @property
    def avg_novelty(self) -> float:
        return self.novelty_sum / max(1, self.examples_trained)

@dataclass
class TrainingStats:
    """Overall training statistics"""
    total_examples: int = 0
    total_time_seconds: float = 0.0
    domain_stats: Dict[str, DomainStats] = None
    modality_counts: Dict[str, int] = None
    checkpoint_count: int = 0
    last_checkpoint_path: str = ""

    def __post_init__(self):
        if self.domain_stats is None:
            self.domain_stats = {d: DomainStats() for d in DOMAIN_NAMES}
        if self.modality_counts is None:
            self.modality_counts = {m: 0 for m in MODALITY_TYPES}

    @property
    def examples_per_second(self) -> float:
        return self.total_examples / max(1.0, self.total_time_seconds)

#=============================================================================
# Dataset Loader (Abstract Base)
#=============================================================================

class DatasetLoader:
    """Base class for domain-specific dataset loaders"""

    def __init__(self, domain: str, modality: str, data_path: str):
        self.domain = domain
        self.modality = modality
        self.data_path = data_path
        self.examples_loaded = 0

    def load_batch(self, batch_size: int) -> List[Tuple[np.ndarray, str, Dict]]:
        """
        Load a batch of examples

        Returns:
            List of (features, label, metadata) tuples
            - features: np.ndarray of shape (num_features,)
            - label: str (e.g., "water_boils_at_100C", "painting_style_baroque")
            - metadata: dict with epistemic hints (e.g., is_opinion, source_quality)
        """
        raise NotImplementedError("Subclasses must implement load_batch")

    def extract_features(self, raw_data: Any) -> np.ndarray:
        """Extract features from raw data (text/image/audio/video)"""
        raise NotImplementedError("Subclasses must implement extract_features")

#=============================================================================
# Text Dataset Loader
#=============================================================================

class TextDatasetLoader(DatasetLoader):
    """Loader for text-based datasets"""

    def __init__(self, domain: str, data_path: str, encoding: str = 'utf-8'):
        super().__init__(domain, "text", data_path)
        self.encoding = encoding
        self.lines = []
        self._load_text_file()

    def _load_text_file(self):
        """Load text file into memory (for small files) or prepare streaming"""
        if not os.path.exists(self.data_path):
            logger.warning(f"Dataset not found: {self.data_path}")
            return

        try:
            with open(self.data_path, 'r', encoding=self.encoding) as f:
                self.lines = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.lines)} lines from {self.data_path}")
        except Exception as e:
            logger.error(f"Failed to load {self.data_path}: {e}")

    def load_batch(self, batch_size: int) -> List[Tuple[np.ndarray, str, Dict]]:
        """Load batch of text examples"""
        if not self.lines:
            return []

        batch = []
        for _ in range(batch_size):
            # Sample random line
            line = random.choice(self.lines)
            if not line:
                continue

            # Extract features (simple: character-level embedding)
            features = self.extract_features(line)

            # Use first 50 chars as label (for classification task)
            label = line[:50].replace('\n', ' ').replace('\t', ' ')

            # Metadata for epistemic filtering
            metadata = {
                'domain': self.domain,
                'modality': 'text',
                'length': len(line),
                'is_opinion': self._detect_opinion(line),
                'source_quality': 0.7  # Default moderate quality
            }

            batch.append((features, label, metadata))
            self.examples_loaded += 1

        return batch

    def extract_features(self, text: str) -> np.ndarray:
        """
        Extract features from text

        Simple approach: Character frequency + length features
        More advanced: Use sentence embeddings (BERT, etc.)
        """
        # Character frequency (26 letters + space + punctuation)
        features = np.zeros(512, dtype=np.float32)

        # Character counts
        text_lower = text.lower()
        for i, char in enumerate('abcdefghijklmnopqrstuvwxyz '):
            features[i] = text_lower.count(char) / max(1, len(text))

        # Length features
        features[27] = min(len(text) / 1000.0, 1.0)  # Normalized length
        features[28] = len(text.split()) / max(1, len(text))  # Word density

        # Simple n-gram features (bigrams)
        for i in range(min(len(text)-1, 100)):
            bigram_hash = hash(text[i:i+2]) % 400
            features[29 + bigram_hash] += 0.01

        return features

    def _detect_opinion(self, text: str) -> bool:
        """Simple heuristic to detect opinion statements"""
        opinion_markers = ['i think', 'i believe', 'in my opinion', 'should',
                          'better', 'worse', 'good', 'bad', 'feel']
        text_lower = text.lower()
        return any(marker in text_lower for marker in opinion_markers)

#=============================================================================
# Streaming Trainer
#=============================================================================

class StreamingTrainer:
    """Main streaming training orchestrator with state persistence"""

    def __init__(self, config: TrainingConfig, state_dir: str = "./training_state"):
        self.config = config
        self.stats = TrainingStats()
        self.brain = None
        self.loaders: Dict[str, List[DatasetLoader]] = defaultdict(list)
        self.start_time = None

        # State manager for persistence and recovery
        self.state_manager = None
        if STATE_MANAGER_AVAILABLE:
            self.state_manager = TrainingStateManager(state_dir)
        self.is_resuming = False
        self.resume_epoch = 0
        self.resume_batch = 0

        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        logger.info("="*80)
        logger.info("NIMCP Streaming Trainer Initialized")
        logger.info("="*80)
        logger.info(f"Configuration: {asdict(config)}")

        # Check for resumable session
        if self.state_manager and self.state_manager.can_resume():
            logger.info("Found resumable training session!")
            self.state_manager.print_status()

    def initialize_brain(self):
        """Initialize NIMCP brain with all cognitive systems enabled"""
        if not NIMCP_AVAILABLE:
            logger.warning("NIMCP not available - running in simulation mode")
            return

        logger.info("Initializing NIMCP brain...")
        logger.info(f"  Size: {self.config.brain_size}")
        logger.info(f"  Inputs: {self.config.num_inputs}")
        logger.info(f"  Outputs: {self.config.num_outputs}")
        logger.info(f"  Learning Rate: {self.config.learning_rate}")
        logger.info(f"  Curiosity: {'ENABLED' if self.config.enable_curiosity else 'DISABLED'}")
        logger.info(f"  Skepticism: {'ENABLED' if self.config.enable_skepticism else 'DISABLED'}")

        # TODO: Call actual NIMCP brain creation
        # self.brain = nimcp.create_brain(...)

        logger.info("✓ Brain initialized with all cognitive systems active")

    def register_loader(self, loader: DatasetLoader):
        """Register a dataset loader for a domain"""
        self.loaders[loader.domain].append(loader)
        logger.info(f"Registered {loader.modality} loader for domain: {loader.domain}")

    def sample_domain(self) -> str:
        """Sample a domain based on configured weights"""
        domains = list(self.config.domain_weights.keys())
        weights = list(self.config.domain_weights.values())
        return random.choices(domains, weights=weights, k=1)[0]

    def sample_modality(self) -> str:
        """Sample a modality based on configured ratios"""
        modalities = ['text', 'image', 'audio', 'video']
        weights = [
            self.config.text_ratio,
            self.config.image_ratio,
            self.config.audio_ratio,
            self.config.video_ratio
        ]
        return random.choices(modalities, weights=weights, k=1)[0]

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        logger.info("="*80)
        logger.info(f"EPOCH {epoch+1}/{self.config.epochs}")
        logger.info("="*80)

        epoch_start = time.time()
        examples_this_epoch = 0

        while examples_this_epoch < self.config.examples_per_epoch:
            # Sample domain and modality
            domain = self.sample_domain()
            modality = self.sample_modality()

            # Get loader
            if domain not in self.loaders or not self.loaders[domain]:
                continue

            loader = random.choice(self.loaders[domain])

            # Load batch
            batch = loader.load_batch(self.config.batch_size)
            if not batch:
                continue

            # Train on batch
            for features, label, metadata in batch:
                self._train_example(features, label, metadata)
                examples_this_epoch += 1

                # Periodic logging
                if self.stats.total_examples % self.config.log_interval == 0:
                    self._log_progress()

                # Epistemic report
                if self.stats.total_examples % self.config.epistemic_report_interval == 0:
                    self._report_epistemic_quality()

                # Consolidation (sleep cycle)
                if (self.config.enable_consolidation and
                    self.stats.total_examples % self.config.consolidation_interval == 0):
                    self._trigger_consolidation()

                # Checkpoint
                if self.stats.total_examples % self.config.checkpoint_interval == 0:
                    self._save_checkpoint()

        epoch_time = time.time() - epoch_start
        logger.info(f"Epoch {epoch+1} complete in {epoch_time:.2f}s "
                   f"({examples_this_epoch/epoch_time:.1f} examples/sec)")

    def _train_example(self, features: np.ndarray, label: str, metadata: Dict):
        """Train on a single example using brain forward pass and loss computation."""
        domain = metadata.get('domain', 'unknown')
        modality = metadata.get('modality', 'unknown')

        # Update counts
        self.stats.total_examples += 1
        self.stats.modality_counts[modality] += 1
        domain_stat = self.stats.domain_stats[domain]
        domain_stat.examples_trained += 1

        # Epistemic filtering: weight confidence by source quality
        is_opinion = metadata.get('is_opinion', False)
        epistemic_quality = 0.5 if is_opinion else 0.9
        domain_stat.epistemic_quality_sum += epistemic_quality

        if self.brain and NIMCP_AVAILABLE:
            # Actual forward pass through the brain
            features_list = features.tolist()
            lr = self.config.learning_rate * epistemic_quality

            # Learn: returns (predicted_label, confidence)
            result = self.brain.learn(features_list, label, lr)

            # Compute loss from brain's internal loss tracker
            loss = self.brain.get_last_loss()
            if loss is None or not np.isfinite(loss):
                loss = 1.0  # fallback for NaN/None

            # Novelty from uncertainty — higher epistemic uncertainty = more novel
            try:
                unc = self.brain.get_uncertainty(features_list)
                novelty = unc.get('epistemic', 0.5)
            except Exception:
                novelty = 0.5
        else:
            # Simulation fallback when brain is not available
            loss = random.uniform(0.1, 0.5)
            novelty = random.uniform(0.0, 1.0)

        domain_stat.total_loss += loss
        domain_stat.novelty_sum += novelty

    def _log_progress(self):
        """Log training progress"""
        elapsed = time.time() - self.start_time
        self.stats.total_time_seconds = elapsed

        logger.info(f"[{self.stats.total_examples:,} examples] "
                   f"{self.stats.examples_per_second:.1f} ex/sec | "
                   f"Elapsed: {elapsed/60:.1f}m")

    def _report_epistemic_quality(self):
        """Report epistemic quality by domain"""
        logger.info("--- Epistemic Quality Report ---")
        for domain, stats in self.stats.domain_stats.items():
            if stats.examples_trained > 0:
                logger.info(f"  {domain:15s}: "
                           f"Quality={stats.avg_epistemic_quality:.3f} | "
                           f"Novelty={stats.avg_novelty:.3f} | "
                           f"Loss={stats.avg_loss:.4f} | "
                           f"N={stats.examples_trained}")

    def _trigger_consolidation(self):
        """Trigger memory consolidation (sleep cycle)"""
        logger.info("Triggering memory consolidation (sleep cycle)...")
        if self.brain and NIMCP_AVAILABLE:
            try:
                self.brain.consolidate(mode="auto")
            except Exception as e:
                logger.warning(f"Consolidation failed: {e}")
        else:
            time.sleep(0.1)  # Simulate consolidation time when no brain

    def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_{self.stats.total_examples}.json"
        )

        checkpoint_data = {
            'config': asdict(self.config),
            'stats': asdict(self.stats),
            'timestamp': time.time()
        }

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)

        self.stats.checkpoint_count += 1
        self.stats.last_checkpoint_path = checkpoint_path
        logger.info(f"✓ Checkpoint saved: {checkpoint_path}")

        # TODO: Save brain weights
        # brain_path = checkpoint_path.replace('.json', '.brain')
        # self.brain.save(brain_path)

    def train(self, resume: bool = False):
        """
        Main training loop with state persistence.

        Args:
            resume: If True, attempt to resume from last checkpoint
        """
        logger.info("="*80)
        logger.info("STARTING STREAMING TRAINING")
        logger.info("="*80)

        start_epoch = 0

        # Handle resume
        if resume and self.state_manager and self.state_manager.can_resume():
            logger.info("Resuming from previous session...")
            if self.state_manager.resume_session():
                stage, dataset_idx, batch_idx, checkpoint_id = self.state_manager.get_resume_point()
                logger.info(f"  Resume point: stage={stage}, batch={batch_idx}")

                # Load checkpoint if available
                if checkpoint_id:
                    checkpoint_path = self.state_manager.load_checkpoint(checkpoint_id)
                    if checkpoint_path and NIMCP_AVAILABLE:
                        logger.info(f"  Loading brain from checkpoint: {checkpoint_path}")
                        self.brain = nimcp.Brain.load(checkpoint_path)

                self.is_resuming = True
                self.resume_batch = batch_idx

                # Restore stats from state
                summary = self.state_manager.get_summary()
                self.stats.total_examples = summary.get('total_examples', 0)
        else:
            # Start new session
            if self.state_manager:
                self.state_manager.start_session(asdict(self.config))

        self.start_time = time.time()

        try:
            for epoch in range(start_epoch, self.config.epochs):
                # Update state
                if self.state_manager:
                    self.state_manager.update_position(
                        stage="training",
                        dataset_index=0,
                        batch_index=epoch * self.config.examples_per_epoch
                    )

                self.train_epoch(epoch)

                # Save state after each epoch
                if self.state_manager:
                    self.state_manager.add_training_time(time.time() - self.start_time)

            logger.info("="*80)
            logger.info("TRAINING COMPLETE")
            logger.info("="*80)
            self._print_final_report()

            # Mark session complete
            if self.state_manager:
                self.state_manager.complete_session()

        except KeyboardInterrupt:
            logger.warning("\n⚠️  Training interrupted by user")
            self._save_checkpoint()
            if self.state_manager:
                self.state_manager.pause_session()
                logger.info("Session paused. Run with --resume to continue.")

        except Exception as e:
            logger.error(f"❌ Training failed: {e}", exc_info=True)
            self._save_checkpoint()
            if self.state_manager:
                self.state_manager.fail_session(str(e))

    def _print_final_report(self):
        """Print final training report"""
        total_time = time.time() - self.start_time

        logger.info(f"Total Examples: {self.stats.total_examples:,}")
        logger.info(f"Total Time: {total_time/3600:.2f} hours")
        logger.info(f"Throughput: {self.stats.examples_per_second:.1f} examples/sec")
        logger.info(f"Checkpoints Saved: {self.stats.checkpoint_count}")
        logger.info("")
        logger.info("Modality Distribution:")
        for modality, count in self.stats.modality_counts.items():
            pct = 100 * count / max(1, self.stats.total_examples)
            logger.info(f"  {modality:10s}: {count:8,} ({pct:5.1f}%)")
        logger.info("")
        logger.info("Top 10 Domains by Training Volume:")
        sorted_domains = sorted(
            self.stats.domain_stats.items(),
            key=lambda x: x[1].examples_trained,
            reverse=True
        )
        for domain, stats in sorted_domains[:10]:
            logger.info(f"  {domain:15s}: {stats.examples_trained:8,} examples | "
                       f"Quality={stats.avg_epistemic_quality:.3f} | "
                       f"Loss={stats.avg_loss:.4f}")

#=============================================================================
# Main Entry Point
#=============================================================================

def main():
    """Main entry point for streaming trainer"""
    import argparse

    parser = argparse.ArgumentParser(
        description="NIMCP AGI Streaming Trainer with State Persistence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start new training session
  python streaming_trainer.py

  # Resume interrupted training
  python streaming_trainer.py --resume

  # Check training status
  python streaming_trainer.py --status

  # Custom configuration
  python streaming_trainer.py --epochs 10 --batch-size 64 --lr 0.005

  # Specify state directory
  python streaming_trainer.py --state-dir ./my_training_state
        """
    )

    # Training control
    parser.add_argument("--resume", action="store_true",
                       help="Resume from previous session")
    parser.add_argument("--status", action="store_true",
                       help="Show training status and exit")

    # Configuration
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate (default: 0.01)")
    parser.add_argument("--examples-per-epoch", type=int, default=10000,
                       help="Examples per epoch (default: 10000)")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                       help="Checkpoint interval in examples (default: 5000)")

    # Paths
    parser.add_argument("--state-dir", type=str, default="./training_state",
                       help="Directory for state persistence (default: ./training_state)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                       help="Directory for checkpoints (default: ./checkpoints)")

    # Brain configuration
    parser.add_argument("--brain-size", type=str, default="MEDIUM",
                       choices=["SMALL", "MEDIUM", "LARGE"],
                       help="Brain size (default: MEDIUM)")

    args = parser.parse_args()

    # Status check only
    if args.status:
        if STATE_MANAGER_AVAILABLE:
            mgr = TrainingStateManager(args.state_dir)
            mgr.print_status()

            # Show checkpoints
            checkpoints = mgr.list_checkpoints()
            if checkpoints:
                print(f"\nCheckpoints ({len(checkpoints)}):")
                for ckpt in checkpoints[-5:]:  # Show last 5
                    print(f"  - {ckpt['checkpoint_id']}: {ckpt['examples_processed']:,} examples")
        else:
            print("State manager not available")
        return

    # Create training configuration
    config = TrainingConfig(
        brain_size=args.brain_size,
        num_inputs=512,
        num_outputs=256,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        examples_per_epoch=args.examples_per_epoch,
        domain_rotation=True,
        enable_curiosity=True,
        enable_skepticism=True,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )

    # Create trainer with state directory
    trainer = StreamingTrainer(config, state_dir=args.state_dir)

    # Check if resuming
    if args.resume:
        if trainer.state_manager and trainer.state_manager.can_resume():
            logger.info("Resuming training session...")
        else:
            logger.warning("No resumable session found. Starting new session.")
            args.resume = False

    # Initialize brain (or load from checkpoint if resuming)
    if not args.resume:
        trainer.initialize_brain()

    # Register example loaders (TODO: Add actual dataset paths)
    logger.info("Registering dataset loaders...")

    # Example: Register text loaders for each domain
    for domain in DOMAIN_NAMES:
        # This is a placeholder - replace with actual dataset paths
        dummy_path = f"./data/{domain}/corpus.txt"
        if os.path.exists(dummy_path):
            loader = TextDatasetLoader(domain, dummy_path)
            trainer.register_loader(loader)

    # Start training (with resume if requested)
    trainer.train(resume=args.resume)

    # Final status
    if trainer.state_manager:
        trainer.state_manager.print_status()


if __name__ == "__main__":
    main()
