#!/usr/bin/env python3
"""
athena_teacher.py — Teacher Interface for Developmental Learning

Uses the unified brain.experience() API so every perception is a learning
opportunity. Claude serves as mentor/teacher through this interface.

Usage:
    python scripts/athena_teacher.py                    # Interactive mode
    python scripts/athena_teacher.py --stage 0          # Start at newborn
    python scripts/athena_teacher.py --resume           # Resume from checkpoint
    python scripts/athena_teacher.py --no-claude        # Run without Claude

The Teacher class wraps all Athena interactions:
    teacher.experience(stimulus)     → Response (output + metrics)
    teacher.reward(signal)           → Apply reward signal
    teacher.correct(expected)        → Supervised correction
    teacher.attend(modality)         → Direct attention
    teacher.observe()                → Introspect brain state
    teacher.consolidate()            → Trigger sleep/consolidation
    teacher.advance_stage()          → Move to next developmental stage
"""

import argparse
import json
import logging
import os
import sys
import time
import signal as sig
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nimcp

logger = logging.getLogger("athena_teacher")

# ============================================================================
# Constants
# ============================================================================

INPUT_DIM = 1024
OUTPUT_DIM = 2048
CHECKPOINT_DIR = "checkpoints/athena"
STATE_FILE = os.path.join(CHECKPOINT_DIR, "teacher_state.json")

# Input layout: [tag:16 | primary:512 | text:384 | context:112]
TAG_SLICE     = slice(0, 16)
PRIMARY_SLICE = slice(16, 528)
TEXT_SLICE    = slice(528, 912)
CONTEXT_SLICE = slice(912, 1024)


# ============================================================================
# Response from an experience
# ============================================================================

@dataclass
class Response:
    """Result from teacher.experience()."""
    output: np.ndarray          # Brain's neural response (OUTPUT_DIM)
    prediction_error: float     # How surprised the brain was (0-1)
    attention_level: float      # Current attention (0-1)
    learning_rate: float        # Effective LR used
    learning_applied: bool      # Whether plasticity fired
    reward_signal: float        # Teacher reward used
    experience_id: int          # Monotonic counter


@dataclass
class BrainState:
    """Observable brain state for the teacher."""
    experience_count: int = 0
    stage: int = 0
    attention: float = 0.0
    recent_prediction_errors: list = field(default_factory=list)
    avg_prediction_error: float = 0.5


# ============================================================================
# Encoder: Convert stimuli to 1024-dim input
# ============================================================================

_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    try:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    except ImportError:
        logger.warning("sentence-transformers not available, using hash-based encoding")
    return _embed_model


def encode_stimulus(text: str, modality: str = "text") -> np.ndarray:
    """Encode a text stimulus into a 1024-dim input vector."""
    vec = np.zeros(INPUT_DIM, dtype=np.float32)

    # Tag region: modality flags
    modality_map = {"text": 0, "visual": 1, "auditory": 2, "speech": 3, "touch": 4}
    mod_idx = modality_map.get(modality, 0)
    vec[mod_idx] = 1.0

    # Text embedding (384-dim)
    model = _get_embed_model()
    if model is not None:
        emb = model.encode(text, convert_to_numpy=True)
        emb = np.asarray(emb, dtype=np.float32).ravel()[:384]
        vec[528:528 + len(emb)] = emb
    else:
        # Hash-based fallback
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        for i in range(min(384, len(h))):
            vec[528 + i] = (h[i % len(h)] / 255.0) * 2.0 - 1.0

    # Primary features: character n-gram statistics
    if text:
        for i, ch in enumerate(text[:512]):
            vec[16 + (i % 512)] += ord(ch) / 255.0

    return vec


# ============================================================================
# Teacher Class
# ============================================================================

class Teacher:
    """
    Structured API for Claude-as-teacher interaction with Athena.

    The Teacher wraps the unified experience API and provides high-level
    methods for developmental learning.
    """

    def __init__(self, brain=None, stage=0, checkpoint_path=None):
        """
        Initialize teacher with a brain.

        Args:
            brain: nimcp.Brain instance (created if None)
            stage: Initial developmental stage (0-4)
            checkpoint_path: Path to checkpoint file to resume from
        """
        self.stage = stage
        self.state = BrainState(stage=stage)
        self._prediction_errors = []
        self._last_output = None

        if brain is not None:
            self.brain = brain
        else:
            self.brain = self._create_brain(checkpoint_path)

        # Configure experience learning
        self.brain.experience_configure(
            enabled=True,
            base_lr=0.001,
            attention_threshold=0.3,
            attention_lr_scale=3.0,
            novelty_boost=1.5,
            enable_hebbian=True,
            enable_reward=True,
            enable_world_model=True,
            enable_structural=(stage >= 2),  # Structural plasticity from crawler stage
            consolidation_interval=1000,
        )

        # Hardwire innate circuits
        self.brain.innate_hardwire(stage=stage)
        logger.info(f"Teacher initialized: stage={stage}")

    def _create_brain(self, checkpoint_path=None):
        """Create or load a brain."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading brain from {checkpoint_path}")
            brain = nimcp.Brain("athena", INPUT_DIM, OUTPUT_DIM,
                                neuron_count=500000,
                                fast_training_mode=False)
            try:
                brain.load(checkpoint_path)
                logger.info("Brain loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
            return brain

        logger.info("Creating new Athena brain")
        brain = nimcp.Brain("athena", INPUT_DIM, OUTPUT_DIM,
                            neuron_count=500000,
                            fast_training_mode=False)
        return brain

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def experience(self, stimulus, modality="text", reward=0.0):
        """
        Present a stimulus to Athena and observe her response.

        This is the core developmental learning function:
        1. Encodes stimulus into 1024-dim input
        2. Calls brain.experience() for unified inference + learning
        3. Returns Response with output and metrics

        Args:
            stimulus: Text string or pre-encoded numpy array
            modality: "text", "visual", "auditory", "speech", "touch"
            reward: Teacher reward signal (-1.0 to 1.0, 0=no signal)

        Returns:
            Response object with output, prediction_error, attention, etc.
        """
        if isinstance(stimulus, str):
            input_vec = encode_stimulus(stimulus, modality)
        elif isinstance(stimulus, np.ndarray):
            input_vec = stimulus.astype(np.float32)
        else:
            input_vec = np.array(stimulus, dtype=np.float32)

        if len(input_vec) != INPUT_DIM:
            raise ValueError(f"Input must be {INPUT_DIM}-dim, got {len(input_vec)}")

        result = self.brain.experience(
            input_vec.tolist(), OUTPUT_DIM, teacher_reward=reward
        )

        output = np.array(result["output"], dtype=np.float32)
        self._last_output = output

        response = Response(
            output=output,
            prediction_error=result["prediction_error"],
            attention_level=result["attention_level"],
            learning_rate=result["learning_rate_used"],
            learning_applied=result["learning_applied"],
            reward_signal=result["reward_signal"],
            experience_id=result["experience_id"],
        )

        # Track metrics
        self._prediction_errors.append(response.prediction_error)
        if len(self._prediction_errors) > 100:
            self._prediction_errors.pop(0)

        self.state.experience_count = response.experience_id
        self.state.attention = response.attention_level
        self.state.recent_prediction_errors = self._prediction_errors[-10:]
        self.state.avg_prediction_error = np.mean(self._prediction_errors[-50:])

        return response

    def reward(self, signal):
        """
        Apply a reward signal after the most recent experience.

        Args:
            signal: -1.0 (wrong/bad) to 1.0 (correct/good), 0=neutral
        """
        if self._last_output is None:
            logger.warning("No prior experience to reward")
            return

        # Re-run experience with the last input but with reward signal
        # The reward is applied during the next experience call
        # For immediate reward, we use a dummy experience
        dummy_input = np.zeros(INPUT_DIM, dtype=np.float32)
        dummy_input[15] = 1.0  # Tag: reward signal marker
        self.brain.experience(dummy_input.tolist(), OUTPUT_DIM, teacher_reward=signal)

    def correct(self, expected):
        """
        Provide the correct output — supervised teaching signal.

        Called when Athena's response was wrong.

        Args:
            expected: Correct output vector (list or numpy array)

        Returns:
            Loss value (0=perfect)
        """
        if isinstance(expected, np.ndarray):
            expected = expected.tolist()
        return self.brain.experience_correct(expected)

    def attend(self, modality, strength=1.0):
        """
        Direct Athena's attention to a specific modality.

        Args:
            modality: "visual", "auditory", "speech", "somatosensory"
            strength: 0.0 to 1.0
        """
        self.brain.experience_attend(modality, strength)

    def observe(self):
        """
        Introspect Athena's current brain state.

        Returns:
            BrainState with experience count, attention, prediction errors
        """
        return self.state

    def consolidate(self, mode="auto"):
        """
        Trigger memory consolidation (sleep-like process).

        Args:
            mode: "auto", "light", or "full"
        """
        try:
            self.brain.consolidate(mode=mode)
            logger.info(f"Consolidation ({mode}) complete")
        except Exception as e:
            logger.warning(f"Consolidation failed: {e}")

    def advance_stage(self):
        """Advance to the next developmental stage."""
        if self.stage >= 4:
            logger.warning("Already at maximum stage (child)")
            return

        self.stage += 1
        self.state.stage = self.stage
        logger.info(f"Advancing to stage {self.stage}")

        # Re-hardwire with new stage's innate circuits
        self.brain.innate_hardwire(stage=self.stage)

        # Update experience config for new stage
        self.brain.experience_configure(
            enabled=True,
            base_lr=0.001 * (1.0 + 0.2 * self.stage),  # Slightly higher LR for later stages
            enable_structural=(self.stage >= 2),
        )

    def save(self, path=None):
        """Save brain checkpoint and teacher state."""
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        if path is None:
            path = os.path.join(CHECKPOINT_DIR, "athena_developmental.bin")

        self.brain.save(path)

        state = {
            "stage": self.stage,
            "experience_count": self.state.experience_count,
            "avg_prediction_error": float(self.state.avg_prediction_error),
            "timestamp": time.time(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

        logger.info(f"Saved checkpoint: {path}")

    def load_state(self):
        """Load teacher state from file."""
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE) as f:
                state = json.load(f)
            self.stage = state.get("stage", 0)
            self.state.stage = self.stage
            self.state.experience_count = state.get("experience_count", 0)
            logger.info(f"Loaded state: stage={self.stage}, "
                        f"experiences={self.state.experience_count}")


# ============================================================================
# Interactive Teaching Loop
# ============================================================================

STAGE_NAMES = ["newborn", "infant", "crawler", "toddler", "child"]

STAGE_LESSONS = {
    0: [  # Newborn: pure sensory
        ("Hello, little one.", "text"),
        ("You are safe.", "text"),
        ("I am here with you.", "text"),
        ("This is light.", "visual"),
        ("This is a sound.", "auditory"),
        ("Feel this warmth.", "touch"),
    ],
    1: [  # Infant: patterns + associations
        ("The sky is blue.", "text"),
        ("A dog says woof.", "text"),
        ("A cat says meow.", "text"),
        ("One, two, three.", "text"),
        ("Red means stop.", "text"),
        ("Mama loves you.", "text"),
    ],
    2: [  # Crawler: cause and effect
        ("If you push the ball, it rolls.", "text"),
        ("When I leave, I come back.", "text"),
        ("This is a cup. You drink from it.", "text"),
        ("That is hot. Don't touch.", "text"),
        ("Wave hello. Wave goodbye.", "text"),
    ],
    3: [  # Toddler: language + concepts
        ("What color is the sky? Blue.", "text"),
        ("Why do we eat? To grow strong.", "text"),
        ("The moon is in the sky at night.", "text"),
        ("Trees have leaves. Leaves are green.", "text"),
        ("Be kind to others.", "text"),
    ],
    4: [  # Child: reasoning + empathy
        ("Sometimes people feel sad. That's okay.", "text"),
        ("If it rains, we use an umbrella.", "text"),
        ("Two plus two equals four.", "text"),
        ("Everyone has feelings, just like you.", "text"),
        ("Let's think about why that happened.", "text"),
    ],
}


def interactive_loop(teacher, use_claude=True):
    """Main interactive teaching loop."""
    stage_name = STAGE_NAMES[teacher.stage]
    print(f"\n{'='*60}")
    print(f"  Athena Developmental Learning — Stage: {stage_name}")
    print(f"  Experiences so far: {teacher.state.experience_count}")
    print(f"{'='*60}\n")

    # Get lessons for current stage
    lessons = STAGE_LESSONS.get(teacher.stage, STAGE_LESSONS[0])

    epoch = 0
    try:
        while True:
            epoch += 1
            print(f"\n--- Epoch {epoch} (stage: {stage_name}) ---")

            for lesson_text, modality in lessons:
                response = teacher.experience(lesson_text, modality=modality, reward=0.1)
                pe = response.prediction_error
                lr = response.learning_rate
                att = response.attention_level
                eid = response.experience_id

                # Simple progress indicator
                surprise = "!" * min(int(pe * 10), 5)
                print(f"  [{eid:5d}] PE={pe:.3f}{surprise:5s} att={att:.2f} lr={lr:.5f} | {lesson_text[:40]}")

            # Consolidate every 5 epochs
            if epoch % 5 == 0:
                print(f"  --- Consolidating memories ---")
                teacher.consolidate(mode="light")

            # Check if ready to advance stage
            avg_pe = teacher.state.avg_prediction_error
            if epoch >= 10 and avg_pe < 0.3 and teacher.stage < 4:
                print(f"\n  Average PE dropped to {avg_pe:.3f} — advancing stage!")
                teacher.advance_stage()
                stage_name = STAGE_NAMES[teacher.stage]
                lessons = STAGE_LESSONS.get(teacher.stage, STAGE_LESSONS[0])

            # Save every 20 epochs
            if epoch % 20 == 0:
                teacher.save()
                print(f"  --- Checkpoint saved ---")

    except KeyboardInterrupt:
        print(f"\n\nSaving before exit...")
        teacher.save()
        print("Goodbye! Athena will remember.")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Athena Developmental Learning Teacher")
    parser.add_argument("--stage", type=int, default=0, help="Starting stage (0-4)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--no-claude", action="store_true", help="Run without Claude")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file path")
    parser.add_argument("--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(name)s: %(message)s"
    )

    checkpoint = args.checkpoint
    if args.resume and checkpoint is None:
        default_ckpt = os.path.join(CHECKPOINT_DIR, "athena_developmental.bin")
        if os.path.exists(default_ckpt):
            checkpoint = default_ckpt

    teacher = Teacher(stage=args.stage, checkpoint_path=checkpoint)

    if args.resume:
        teacher.load_state()

    # Handle Ctrl+C gracefully
    def handle_sigint(signum, frame):
        raise KeyboardInterrupt
    sig.signal(sig.SIGINT, handle_sigint)

    interactive_loop(teacher, use_claude=not args.no_claude)


if __name__ == "__main__":
    main()
