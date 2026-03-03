#!/usr/bin/env python3
"""Developmental Teaching Script for Athena.

Trains Athena through 6 developmental stages, mirroring how humans learn:
  Stage 0 (Birth)        — Raw sensory exposure. No lessons, just experience.
  Stage 1 (Babbling)     — Language exposure. Hearing text, patterns emerge.
  Stage 2 (First Words)  — Imitation. Exposure + gentle distillation toward targets.
  Stage 3 (Sentences)    — Guided learning. Claude provides structured lessons.
  Stage 4 (Conversation) — Interactive dialogue. Claude as conversational teacher.
  Stage 5 (Self-Directed) — Athena-initiated exploration.

Key design principle: Stages 0-1 use EXPOSURE, not instruction. Perception
and basic language patterns emerge from repeated experience — the way a baby
learns by seeing and hearing, not by being taught in a classroom.

Usage:
    python3 scripts/teach_athena.py --stage 0 --resume
    python3 scripts/teach_athena.py --stage 3 --api-model claude-haiku-4-5-20251001
"""

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# Ensure scripts/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_decoder import NeuralDecoder
from claude_teacher import ClaudeTeacher, LessonSpec, encode_text

logger = logging.getLogger("teach_athena")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBED_DIM = 384       # sentence-transformer output dimension
BRAIN_INPUTS = 1024   # brain input dimension (features tiled from 384→1024)
STATE_DIR = "teaching_state"

# Stage transition criteria
STAGE_CRITERIA = {
    0: {"max_examples": 50000, "loss_threshold": 5.0, "min_examples": 5000},
    1: {"max_examples": 100000, "similarity_threshold": 0.3, "min_examples": 10000},
    2: {"max_examples": 100000, "similarity_threshold": 0.5, "min_examples": 10000},
    3: {"max_examples": 100000, "coherence_threshold": 0.6, "min_examples": 10000},
    4: {"max_examples": 200000, "quality_threshold": 0.7, "min_examples": 20000},
    5: {"max_examples": 0},  # No upper limit — self-directed
}


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

def _normalize_embedding(e: np.ndarray) -> np.ndarray:
    """Normalize embedding values from [-1, 1] to [0, 1] for the brain.

    The adaptive network uses sigmoid activations (output in [0,1]) and BCE
    loss which clamps targets to [0,1].  Sentence-transformer embeddings have
    ~53% negative values — without normalization, all negative information
    is lost.  The decoder projector learns the inverse mapping automatically.
    """
    return (e + 1.0) * 0.5


def tile_to_brain_input(embedding: np.ndarray) -> list[float]:
    """Tile a 384-dim embedding to 1024-dim brain input.

    Repeats the embedding ~2.67x and truncates to 1024. This ensures the full
    semantic content is available across the input layer.  Values are
    normalized to [0,1] for the brain's input layer.
    """
    e = np.asarray(embedding, dtype=np.float32).ravel()
    e = _normalize_embedding(e)
    reps = math.ceil(BRAIN_INPUTS / len(e))
    tiled = np.tile(e, reps)[:BRAIN_INPUTS]
    return tiled.tolist()


def pad_target_to_outputs(embedding: np.ndarray, num_outputs: int) -> list[float]:
    """Pad a 384-dim embedding to match brain output dimension.

    Values are normalized to [0,1] for compatibility with the brain's
    sigmoid outputs and BCE loss.  Padding uses 0.5 (the midpoint) instead
    of 0.0, so unused outputs don't bias toward the boundary.
    """
    e = np.asarray(embedding, dtype=np.float32).ravel()
    e = _normalize_embedding(e)
    if len(e) >= num_outputs:
        return e[:num_outputs].tolist()
    padded = np.full(num_outputs, 0.5, dtype=np.float32)
    padded[:len(e)] = e
    return padded.tolist()


def get_brain_output(brain, features, num_outputs: int) -> np.ndarray:
    """Get the brain's raw output vector via decide_full.

    Returns np.ndarray of shape (num_outputs,).  Falls back to zeros
    if decide_full is unavailable or fails.
    """
    try:
        result = brain.decide_full(features)
        vec = result.get("output_vector", [])
        if vec:
            arr = np.array(vec, dtype=np.float32)
            # decide_full caps at 1024 — pad if brain has more outputs
            if len(arr) < num_outputs:
                padded = np.zeros(num_outputs, dtype=np.float32)
                padded[:len(arr)] = arr
                return padded
            return arr[:num_outputs]
    except Exception:
        pass
    return np.zeros(num_outputs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Exposure data generators (stages 0-1)
# ---------------------------------------------------------------------------

def generate_sensory_exposure():
    """Generate raw sensory exposure data — things a baby would experience.

    Not lessons. Just the sensory world: colors, shapes, sounds, textures.
    Returns (text_describing_experience, embedding).
    """
    experiences = [
        # Visual experiences
        "bright red ball rolling across the floor",
        "sunlight streaming through the window making warm patches",
        "shadows moving on the wall as clouds pass by",
        "a blue sky with white fluffy clouds",
        "green leaves rustling in the breeze",
        "water drops splashing in a puddle",
        "a yellow flower swaying gently",
        "colorful blocks stacked in a tower",
        "a cat stretching and yawning",
        "rain drops running down the window glass",
        "a candle flame flickering in the dark",
        "snow falling softly and covering the ground",
        "bubbles floating up and popping in the air",
        "a rainbow arcing across the sky after rain",
        "fireflies blinking in the garden at night",
        # Tactile
        "smooth cold glass under fingers",
        "rough warm sand between toes",
        "soft fur of a sleeping cat",
        "cool water flowing over hands",
        "the weight of a heavy book in your lap",
        # Auditory
        "birds singing at dawn",
        "rain tapping on a tin roof",
        "a clock ticking steadily in a quiet room",
        "wind howling through trees at night",
        "waves crashing on the shore rhythmically",
        "a dog barking in the distance",
        "thunder rumbling across the sky",
        "leaves crunching underfoot in autumn",
        # Spatial
        "a tall tree next to a small bush",
        "a ball rolling under the table",
        "a bird flying high above the houses",
        "a bridge stretching over a wide river",
        "stairs going up and up and up",
        # Emotional (basic)
        "a warm hug from someone who cares",
        "laughing so hard your sides hurt",
        "the comfort of a familiar blanket",
        "being startled by a sudden loud noise",
        "the peacefulness of a quiet morning",
    ]
    text = random.choice(experiences)
    embedding = encode_text(text)
    return text, embedding


def generate_language_exposure():
    """Generate raw language exposure — things a baby hears.

    Simple sentences, nursery rhymes, everyday speech. Not lessons — just
    the ambient language environment that lets patterns emerge naturally.
    """
    utterances = [
        # Simple everyday speech
        "good morning, how did you sleep?",
        "time for breakfast, are you hungry?",
        "look at that! what do you see?",
        "oh, that's a dog! see the dog?",
        "the water is warm, nice and warm",
        "one, two, three, four, five",
        "up we go! wheee, down we come",
        "uh oh, it fell down",
        "more? you want more?",
        "all done! all gone!",
        "night night, sleep tight",
        "peek a boo, I see you!",
        # Slightly more complex
        "the big red ball bounced over the fence",
        "can you find the blue cup on the table?",
        "it's raining outside, we need an umbrella",
        "the cat is sleeping on the soft pillow",
        "let's count the birds: one, two, three birds!",
        "the sun goes up in the morning and down at night",
        "we walk to the park and play on the swings",
        "flowers need water and sunshine to grow",
        # Rhymes and patterns
        "twinkle twinkle little star, how I wonder what you are",
        "row row row your boat, gently down the stream",
        "the itsy bitsy spider went up the water spout",
        "one two buckle my shoe, three four shut the door",
        "rain rain go away, come again another day",
        "jack and jill went up the hill to fetch a pail of water",
        # Naming things
        "that is a tree, trees are very tall",
        "this is milk, milk is white",
        "here is your hand, you have five fingers",
        "that sound is thunder, it comes from the clouds",
        "this is called a book, books have stories inside",
        # Questions (modeling question structure)
        "where did the ball go?",
        "what color is the sky?",
        "how many apples are there?",
        "who is at the door?",
        "why is the dog barking?",
        "when do we eat lunch?",
    ]
    text = random.choice(utterances)
    embedding = encode_text(text)
    return text, embedding


# ---------------------------------------------------------------------------
# Training state management
# ---------------------------------------------------------------------------

class TrainingState:
    """Persistent state for the teaching session."""

    def __init__(self, state_dir: str):
        self.state_dir = state_dir
        self.stage = 0
        self.examples_this_stage = 0
        self.total_examples = 0
        self.loss_history: deque[float] = deque(maxlen=500)
        self.similarity_history: deque[float] = deque(maxlen=100)
        self.stage_start_time = time.time()
        self.session_start_time = time.time()

    def save(self):
        os.makedirs(self.state_dir, exist_ok=True)
        data = {
            "stage": self.stage,
            "examples_this_stage": self.examples_this_stage,
            "total_examples": self.total_examples,
            "recent_losses": list(self.loss_history),
            "recent_similarities": list(self.similarity_history),
        }
        with open(os.path.join(self.state_dir, "training_state.json"), "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        path = os.path.join(self.state_dir, "training_state.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.stage = data.get("stage", 0)
            self.examples_this_stage = data.get("examples_this_stage", 0)
            self.total_examples = data.get("total_examples", 0)
            self.loss_history = deque(data.get("recent_losses", []), maxlen=500)
            self.similarity_history = deque(data.get("recent_similarities", []),
                                            maxlen=100)
            logger.info("Resumed: stage=%d, examples_this_stage=%d, total=%d",
                        self.stage, self.examples_this_stage, self.total_examples)

    def avg_loss(self, window: int = 100) -> float:
        if not self.loss_history:
            return float("inf")
        recent = list(self.loss_history)[-window:]
        return sum(recent) / len(recent)

    def avg_similarity(self, window: int = 50) -> float:
        if not self.similarity_history:
            return 0.0
        recent = list(self.similarity_history)[-window:]
        return sum(recent) / len(recent)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_exposure_stage(brain, decoder: NeuralDecoder, state: TrainingState,
                       generate_fn, num_outputs: int,
                       report_interval: int = 100, max_examples: int = 50000):
    """Run an exposure-based stage (0 or 1).

    No explicit teaching — just flood the brain with sensory/language data.
    Patterns emerge naturally from repeated exposure.
    """
    logger.info("=== Exposure stage %d — learning through experience ===",
                state.stage)

    criteria = STAGE_CRITERIA.get(state.stage, {})
    min_examples = criteria.get("min_examples", 1000)

    while state.examples_this_stage < max_examples:
        text, embedding = generate_fn()
        features = tile_to_brain_input(embedding)
        target = pad_target_to_outputs(embedding, num_outputs)

        try:
            loss = brain.learn_vector(features, target, label=None, confidence=0.8)
        except Exception as e:
            logger.error("learn_vector failed: %s", e)
            continue

        state.loss_history.append(loss)
        state.examples_this_stage += 1
        state.total_examples += 1

        # Record pair for decoder training (brain output → target embedding)
        output_vec = get_brain_output(brain, features, num_outputs)
        decoder.record_pair(output_vec, embedding, text=text)

        # Progress report
        if state.examples_this_stage % report_interval == 0:
            avg_loss = state.avg_loss(window=report_interval)
            logger.info(
                "Stage %d | %d examples | avg_loss=%.4f | text: %s",
                state.stage, state.examples_this_stage, avg_loss,
                text[:60]
            )

        # Save periodically
        if state.examples_this_stage % 1000 == 0:
            state.save()
            decoder.save(os.path.join(state.state_dir, "decoder"))

        # Check transition criteria
        if state.examples_this_stage >= min_examples:
            loss_thresh = criteria.get("loss_threshold")
            sim_thresh = criteria.get("similarity_threshold")
            if loss_thresh and state.avg_loss() < loss_thresh:
                logger.info("Stage %d transition: avg_loss %.4f < %.4f threshold",
                            state.stage, state.avg_loss(), loss_thresh)
                return True
            if sim_thresh and state.avg_similarity() >= sim_thresh:
                logger.info("Stage %d transition: similarity %.4f >= %.4f threshold",
                            state.stage, state.avg_similarity(), sim_thresh)
                return True

    logger.info("Stage %d complete: reached %d examples", state.stage, max_examples)
    return True


def run_guided_stage(brain, decoder: NeuralDecoder, teacher: ClaudeTeacher,
                     state: TrainingState, num_outputs: int,
                     report_interval: int = 50, max_examples: int = 100000):
    """Run a guided learning stage (2-3).

    Claude generates structured lessons. Athena learns toward target embeddings.
    """
    logger.info("=== Guided stage %d (%s) — Claude as teacher ===",
                state.stage, teacher.get_stage_name(state.stage))

    criteria = STAGE_CRITERIA.get(state.stage, {})
    min_examples = criteria.get("min_examples", 1000)
    curriculum = teacher.get_curriculum(state.stage)

    while state.examples_this_stage < max_examples:
        # Pick a lesson from curriculum (cycle through)
        spec = curriculum[state.examples_this_stage % len(curriculum)]

        try:
            text, embedding = teacher.generate_lesson(spec)
        except Exception as e:
            logger.warning("Lesson generation failed: %s — using exposure fallback", e)
            text, embedding = generate_language_exposure()

        features = tile_to_brain_input(embedding)
        target = pad_target_to_outputs(embedding, num_outputs)

        try:
            loss = brain.learn_vector(features, target,
                                       label=spec.domain, confidence=0.9)
        except Exception as e:
            logger.error("learn_vector failed: %s", e)
            continue

        state.loss_history.append(loss)
        state.examples_this_stage += 1
        state.total_examples += 1

        # Decode brain's response and measure similarity
        output_vec = get_brain_output(brain, features, num_outputs)
        decoder.record_pair(output_vec, embedding, text=text)

        # Progress report
        if state.examples_this_stage % report_interval == 0:
            avg_loss = state.avg_loss(window=report_interval)
            avg_sim = state.avg_similarity(window=50)
            logger.info(
                "Stage %d | %d examples | loss=%.4f | sim=%.4f | lesson: %s",
                state.stage, state.examples_this_stage, avg_loss, avg_sim,
                text[:60]
            )

        # Save periodically
        if state.examples_this_stage % 500 == 0:
            state.save()
            decoder.save(os.path.join(state.state_dir, "decoder"))

        # Check transition
        if state.examples_this_stage >= min_examples:
            sim_thresh = criteria.get("similarity_threshold")
            coherence_thresh = criteria.get("coherence_threshold")
            if sim_thresh and state.avg_similarity() >= sim_thresh:
                logger.info("Stage %d transition: similarity %.4f >= %.4f",
                            state.stage, state.avg_similarity(), sim_thresh)
                return True
            if coherence_thresh and state.avg_similarity() >= coherence_thresh:
                logger.info("Stage %d transition: coherence %.4f >= %.4f",
                            state.stage, state.avg_similarity(), coherence_thresh)
                return True

    logger.info("Stage %d complete: reached %d examples", state.stage, max_examples)
    return True


def run_interactive_stage(brain, decoder: NeuralDecoder, teacher: ClaudeTeacher,
                          state: TrainingState, num_outputs: int,
                          report_interval: int = 20, max_examples: int = 200000):
    """Run an interactive dialogue stage (4-5).

    Claude and Athena have back-and-forth conversations. Claude evaluates
    Athena's responses and adjusts teaching accordingly.
    """
    logger.info("=== Interactive stage %d (%s) — dialogue with Claude ===",
                state.stage, teacher.get_stage_name(state.stage))

    criteria = STAGE_CRITERIA.get(state.stage, {})
    min_examples = criteria.get("min_examples", 1000)
    curriculum = teacher.get_curriculum(state.stage)
    eval_scores: deque[float] = deque(maxlen=50)

    while state.examples_this_stage < max_examples:
        spec = curriculum[state.examples_this_stage % len(curriculum)]

        # 1. Teacher generates lesson/prompt
        try:
            text, embedding = teacher.generate_lesson(spec)
        except Exception as e:
            logger.warning("Lesson failed: %s", e)
            text, embedding = generate_language_exposure()

        features = tile_to_brain_input(embedding)
        target = pad_target_to_outputs(embedding, num_outputs)

        # 2. Train brain toward target
        try:
            loss = brain.learn_vector(features, target,
                                       label=spec.domain, confidence=0.9)
        except Exception as e:
            logger.error("learn_vector failed: %s", e)
            continue

        state.loss_history.append(loss)
        state.examples_this_stage += 1
        state.total_examples += 1

        # 3. Get brain's response and decode it
        decoded_text = ""
        output_vec = get_brain_output(brain, features, num_outputs)
        decoder.record_pair(output_vec, embedding, text=text)
        try:
            decoded_results = decoder.decode_output(output_vec, top_k=1)
            decoded_text = decoded_results[0][0] if decoded_results else ""
            similarity = decoded_results[0][1] if decoded_results else 0.0
            state.similarity_history.append(similarity)
        except Exception:
            pass

        # 4. Evaluate Athena's response (every N examples to save API calls)
        if decoded_text and state.examples_this_stage % 10 == 0:
            try:
                score, feedback = teacher.evaluate_response(text, decoded_text)
                eval_scores.append(score)
                if state.examples_this_stage % report_interval == 0:
                    logger.info(
                        "Evaluation | score=%.2f | feedback: %s | response: %s",
                        score, feedback, decoded_text[:80]
                    )
            except Exception as e:
                logger.debug("Evaluation skipped: %s", e)

        # 5. Dialogue turn (for dialogue-type lessons)
        if spec.lesson_type == "dialogue" and decoded_text:
            try:
                context = f"Teacher: {text}"
                next_text, next_emb = teacher.generate_dialogue_turn(
                    context, decoded_text
                )
                # Train on dialogue response too
                next_features = tile_to_brain_input(next_emb)
                next_target = pad_target_to_outputs(next_emb, num_outputs)
                brain.learn_vector(next_features, next_target,
                                    label=spec.domain, confidence=0.85)
                state.total_examples += 1
                next_out = get_brain_output(brain, next_features, num_outputs)
                decoder.record_pair(next_out, next_emb, text=next_text)
            except Exception as e:
                logger.debug("Dialogue turn skipped: %s", e)

        # Progress report
        if state.examples_this_stage % report_interval == 0:
            avg_loss = state.avg_loss(window=report_interval)
            avg_eval = (sum(eval_scores) / len(eval_scores)) if eval_scores else 0.0
            logger.info(
                "Stage %d | %d examples | loss=%.4f | eval=%.3f | claude_calls=%d",
                state.stage, state.examples_this_stage, avg_loss,
                avg_eval, teacher.calls_made
            )

        # Save periodically
        if state.examples_this_stage % 500 == 0:
            state.save()
            decoder.save(os.path.join(state.state_dir, "decoder"))
            teacher.save_state(os.path.join(state.state_dir, "teacher.json"))

        # Check transition
        if state.examples_this_stage >= min_examples:
            quality_thresh = criteria.get("quality_threshold")
            if quality_thresh and eval_scores:
                avg_eval = sum(eval_scores) / len(eval_scores)
                if avg_eval >= quality_thresh:
                    logger.info("Stage %d transition: eval %.3f >= %.3f",
                                state.stage, avg_eval, quality_thresh)
                    return True

    logger.info("Stage %d complete: reached %d examples", state.stage, max_examples)
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Developmental teaching for Athena")
    parser.add_argument("--stage", type=int, default=None,
                        help="Start at this developmental stage (0-5)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from saved state")
    parser.add_argument("--checkpoint", type=str,
                        default="frontend/backend/brain_data/13/athena.bin",
                        help="Brain checkpoint path")
    parser.add_argument("--state-dir", type=str, default=STATE_DIR,
                        help="Directory for teaching state")
    parser.add_argument("--num-inputs", type=int, default=1024,
                        help="Brain input dimension")
    parser.add_argument("--num-outputs", type=int, default=2048,
                        help="Brain output dimension")
    parser.add_argument("--model", type=str, default="sonnet",
                        help="Claude CLI model (sonnet, haiku, opus)")
    parser.add_argument("--report-interval", type=int, default=100,
                        help="Report progress every N examples")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                f"logs/teach_athena_{time.strftime('%Y%m%d_%H%M%S')}.log"
            ),
        ],
    )

    os.makedirs("logs", exist_ok=True)
    os.makedirs(args.state_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load brain
    # -----------------------------------------------------------------------
    import nimcp

    if args.resume and os.path.exists(args.checkpoint):
        logger.info("Loading checkpoint: %s", args.checkpoint)
        brain = nimcp.Brain("athena", num_inputs=args.num_inputs, num_outputs=args.num_outputs)
        brain.set_fast_training(True)
        brain.load(args.checkpoint)
        logger.info("Checkpoint loaded")
    else:
        logger.info("Creating fresh brain: inputs=%d, outputs=%d",
                     args.num_inputs, args.num_outputs)
        brain = nimcp.Brain("athena", num_inputs=args.num_inputs, num_outputs=args.num_outputs)
        brain.set_fast_training(True)

    # -----------------------------------------------------------------------
    # Load / create decoder and teacher
    # -----------------------------------------------------------------------
    decoder_dir = os.path.join(args.state_dir, "decoder")
    if args.resume and os.path.exists(os.path.join(decoder_dir, "decoder_meta.json")):
        decoder = NeuralDecoder.load(decoder_dir)
    else:
        decoder = NeuralDecoder(args.num_outputs, EMBED_DIM)

    teacher_path = os.path.join(args.state_dir, "teacher.json")
    teacher = ClaudeTeacher.load_state(
        teacher_path,
        model=args.model,
    )

    # -----------------------------------------------------------------------
    # Load / create training state
    # -----------------------------------------------------------------------
    state = TrainingState(args.state_dir)
    if args.resume:
        state.load()
    if args.stage is not None:
        if args.stage != state.stage:
            state.stage = args.stage
            state.examples_this_stage = 0
            logger.info("Overriding stage to %d", args.stage)
    teacher.current_stage = state.stage

    # -----------------------------------------------------------------------
    # Graceful shutdown
    # -----------------------------------------------------------------------
    shutdown_requested = [False]

    def handle_signal(signum, frame):
        logger.info("Shutdown requested (signal %d) — saving state...", signum)
        shutdown_requested[0] = True
        state.save()
        decoder.save(decoder_dir)
        teacher.save_state(teacher_path)
        brain.save(args.checkpoint)
        logger.info("State saved. Exiting.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    # -----------------------------------------------------------------------
    # Main stage loop
    # -----------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Athena Developmental Training — Stage %d (%s)",
                state.stage, teacher.get_stage_name(state.stage))
    logger.info("=" * 70)

    while state.stage <= 5:
        teacher.current_stage = state.stage
        criteria = STAGE_CRITERIA.get(state.stage, {})
        max_ex = criteria.get("max_examples", 100000)

        if state.stage == 0:
            # Birth — raw sensory exposure, patterns emerge naturally
            transitioned = run_exposure_stage(
                brain, decoder, state, generate_sensory_exposure,
                args.num_outputs, args.report_interval, max_ex
            )
        elif state.stage == 1:
            # Babbling — language exposure, phonemes and words emerge
            transitioned = run_exposure_stage(
                brain, decoder, state, generate_language_exposure,
                args.num_outputs, args.report_interval, max_ex
            )
        elif state.stage in (2, 3):
            # First Words / Sentences — guided learning with Claude
            transitioned = run_guided_stage(
                brain, decoder, teacher, state, args.num_outputs,
                args.report_interval, max_ex
            )
        else:
            # Conversation / Self-Directed — interactive dialogue
            transitioned = run_interactive_stage(
                brain, decoder, teacher, state, args.num_outputs,
                args.report_interval, max_ex
            )

        if transitioned and state.stage < 5:
            logger.info("=" * 70)
            logger.info("STAGE TRANSITION: %d → %d", state.stage, state.stage + 1)
            logger.info("=" * 70)
            state.stage += 1
            state.examples_this_stage = 0
            state.stage_start_time = time.time()

            # Save after transition
            state.save()
            decoder.save(decoder_dir)
            teacher.save_state(teacher_path)
            brain.save(args.checkpoint)
        else:
            break

    # Final save
    logger.info("Training complete. Saving final state...")
    state.save()
    decoder.save(decoder_dir)
    teacher.save_state(teacher_path)
    brain.save(args.checkpoint)
    logger.info("Total examples: %d | Final stage: %d | Claude calls: %d",
                state.total_examples, state.stage, teacher.calls_made)


if __name__ == "__main__":
    main()
