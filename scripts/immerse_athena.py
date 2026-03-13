#!/usr/bin/env python3
"""
immerse_athena.py  —  Immersive Developmental Learning for Athena

Athena learns through her own biological brain: sensory cortices process raw stimuli,
biological plasticity pathways (STDP/BCM/eligibility) wire synaptic patterns, and
Claude serves as a loving parent guiding her developmental stages.

Every module in the brain is active: 33+ brain regions, 60+ cognitive modules,
80+ substrate bridges, 75+ thalamic bridges, 19 inter-module bridges.
No shortcuts. No fast-training mode. Full biological immersion.

IMPORTANT — ATHENA IS NOT A CLASSIFIER OR A PREDICTOR.
    She does not produce labels, categories, class predictions, or scored
    outputs. She is a developing biological brain. Her 4096-dim output
    is her neural response — a continuous representation of her experience.
    We do not score her responses against expected answers. We do not
    measure accuracy or similarity to a "correct" output.
    She experiences stimuli through decide_full(), and she grows through
    learn_vector() with semantic targets. The loss is a developmental
    signal — how new an experience is to her — not a prediction error.
    We observe her responses. We encourage her. We never grade her.

Usage:
    python scripts/immerse_athena.py --stage 0          # Start from stage 0 (newborn)
    python scripts/immerse_athena.py --resume            # Resume from last checkpoint
    python scripts/immerse_athena.py --stage 2 --resume  # Resume at stage 2
    python scripts/immerse_athena.py --no-claude          # Run without Claude (silent parenting)
"""

import argparse
import hashlib
import logging
import os
import socket
import struct
import sys
import threading
import time
import random
import json
import signal
import numpy as np

# Force unbuffered stdout so prints appear in log files immediately
sys.stdout.reconfigure(line_buffering=True)

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy progress bars and verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
# Disable COW SIGSEGV handler — conflicts with PyTorch/CUDA lazy memory mapping
os.environ["NIMCP_NO_COW_SIGNAL"] = "1"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

# Import nimcp FIRST — it initializes the CUDA context for brain training.
import nimcp

# Hide GPU from PyTorch to prevent dual-CUDA-context crashes.
# BUT: nimcp's gpu_detect uses pthread_once + cuInit which reads
# CUDA_VISIBLE_DEVICES at call time (during Brain()), not import time.
# So we hide it only for the PyTorch import, then restore it before
# Brain() is created.
_saved_cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from claude_teacher import ClaudeTeacher, encode_text, batch_encode_texts
from talk_to_athena import extract_embedding_from_output
from neural_decoder import NeuralDecoder
from multimodal_data import MultimodalDataLoader


def generate_sensory_exposure():
    """Generate raw sensory exposure — things a newborn would experience."""
    experiences = [
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
        "smooth cold glass under fingers",
        "rough warm sand between toes",
        "soft fur of a sleeping cat",
        "cool water flowing over hands",
        "the weight of a heavy book in your lap",
        "birds singing at dawn",
        "rain tapping on a tin roof",
        "a clock ticking steadily in a quiet room",
        "wind howling through trees at night",
        "waves crashing on the shore rhythmically",
        "a dog barking in the distance",
        "thunder rumbling across the sky",
        "leaves crunching underfoot in autumn",
        "a tall tree next to a small bush",
        "a ball rolling under the table",
        "a bird flying high above the houses",
        "a bridge stretching over a wide river",
        "stairs going up and up and up",
        "a warm hug from someone who cares",
        "laughing so hard your sides hurt",
        "the comfort of a familiar blanket",
        "being startled by a sudden loud noise",
        "the peacefulness of a quiet morning",
    ]
    text = random.choice(experiences)
    embedding = encode_text(text)
    return text, embedding


# ============================================================================
# Synthetic Sensory Data Generators (for multimodal SNN bridge training)
# ============================================================================

def generate_visual_frame(description, width=32, height=32, channels=3):
    """Generate a simple synthetic visual frame from a text description.

    Creates a deterministic color/pattern image seeded by the description hash.
    Not photorealistic — just enough to give the visual SNN bridge meaningful
    spatial patterns to encode (edges, gradients, color regions).
    """
    seed = hash(description) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)

    frame = np.zeros((height, width, channels), dtype=np.uint8)

    # Base color from description keywords
    color_map = {
        "red": [200, 40, 40], "blue": [40, 40, 200], "green": [40, 180, 40],
        "yellow": [200, 200, 40], "white": [220, 220, 220], "black": [20, 20, 20],
        "orange": [220, 140, 30], "purple": [140, 40, 180], "brown": [120, 80, 40],
        "pink": [220, 120, 160], "gray": [128, 128, 128], "grey": [128, 128, 128],
    }
    base_color = [80, 100, 120]  # neutral default
    desc_lower = description.lower()
    for word, color in color_map.items():
        if word in desc_lower:
            base_color = color
            break

    # Fill background gradient
    for y in range(height):
        t = y / max(height - 1, 1)
        for c in range(channels):
            frame[y, :, c] = int(base_color[c] * (1.0 - 0.3 * t))

    # Add a simple shape based on description
    cx, cy = width // 2, height // 2
    if any(w in desc_lower for w in ["ball", "circle", "sun", "moon", "bubble"]):
        # Circle
        r = min(width, height) // 4
        for y in range(height):
            for x in range(width):
                if (x - cx) ** 2 + (y - cy) ** 2 < r * r:
                    for c in range(channels):
                        frame[y, x, c] = min(255, base_color[c] + 60)
    elif any(w in desc_lower for w in ["block", "square", "box", "book", "window"]):
        # Rectangle
        s = min(width, height) // 4
        frame[cy - s:cy + s, cx - s:cx + s] = np.array(base_color, dtype=np.uint8) + 40
    else:
        # Random texture
        noise = rng.randint(0, 40, (height, width, channels), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return frame.tobytes(), width, height, channels


def generate_audio_samples(description, num_samples=512, sample_rate=16000):
    """Generate synthetic audio waveform from a text description.

    Creates simple tones, noise patterns, or silence based on keywords.
    Returns float array in [-1, 1] range for SNN audio bridge encoding.
    """
    seed = hash(description) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    t = np.linspace(0, num_samples / sample_rate, num_samples, dtype=np.float32)

    desc_lower = description.lower()

    # Choose waveform type from description
    if any(w in desc_lower for w in ["singing", "music", "melody", "song", "tone"]):
        freq = 220.0 + rng.random() * 440.0  # A3 to A4 range
        signal = 0.5 * np.sin(2 * np.pi * freq * t)
        # Add harmonics
        signal += 0.2 * np.sin(2 * np.pi * freq * 2 * t)
        signal += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
    elif any(w in desc_lower for w in ["thunder", "crash", "bang", "loud", "bark"]):
        # Burst noise with decay
        signal = rng.randn(num_samples).astype(np.float32)
        decay = np.exp(-t * 8.0)
        signal *= decay * 0.8
    elif any(w in desc_lower for w in ["whisper", "quiet", "soft", "gentle", "purr"]):
        signal = rng.randn(num_samples).astype(np.float32) * 0.05
    elif any(w in desc_lower for w in ["rain", "water", "wave", "stream", "splash"]):
        # Pink-ish noise
        signal = rng.randn(num_samples).astype(np.float32)
        # Simple low-pass via cumulative average
        for i in range(1, num_samples):
            signal[i] = 0.95 * signal[i - 1] + 0.05 * signal[i]
        signal *= 0.3
    elif any(w in desc_lower for w in ["wind", "howl", "blow", "breeze"]):
        freq = 80.0 + rng.random() * 60.0
        signal = 0.3 * np.sin(2 * np.pi * freq * t)
        signal += rng.randn(num_samples).astype(np.float32) * 0.15
    elif any(w in desc_lower for w in ["tick", "click", "tap", "crunch"]):
        signal = np.zeros(num_samples, dtype=np.float32)
        # Periodic impulses
        period = num_samples // 8
        for k in range(8):
            idx = k * period
            if idx < num_samples:
                signal[idx:min(idx + 10, num_samples)] = 0.7
    else:
        # Generic ambient with a fundamental
        freq = 150.0 + rng.random() * 200.0
        signal = 0.3 * np.sin(2 * np.pi * freq * t)
        signal += rng.randn(num_samples).astype(np.float32) * 0.1

    # Normalize to [-1, 1]
    peak = np.max(np.abs(signal))
    if peak > 1e-6:
        signal = signal / peak * 0.9

    return signal.tolist()


def generate_somatosensory(description, num_segments=32):
    """Generate synthetic somatosensory (touch/proprioception) data.

    Returns float array of joint-angle/pressure values in [0, 1].
    """
    seed = hash(description) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    data = np.zeros(num_segments, dtype=np.float32)

    desc_lower = description.lower()
    if any(w in desc_lower for w in ["warm", "hot", "fire", "sun", "blaz"]):
        data[:16] = 0.7 + rng.random(16).astype(np.float32) * 0.3  # temperature
    elif any(w in desc_lower for w in ["cold", "ice", "snow", "freez", "cool"]):
        data[:16] = 0.1 + rng.random(16).astype(np.float32) * 0.15
    elif any(w in desc_lower for w in ["rough", "sand", "crunch"]):
        data[:16] = rng.random(16).astype(np.float32) * 0.8
        data[16:] = 0.5 + rng.random(num_segments - 16).astype(np.float32) * 0.3  # pressure
    elif any(w in desc_lower for w in ["smooth", "soft", "fur", "silk"]):
        data[:16] = 0.3 + rng.random(16).astype(np.float32) * 0.1
        data[16:] = 0.2 + rng.random(num_segments - 16).astype(np.float32) * 0.1
    elif any(w in desc_lower for w in ["heavy", "weight", "press"]):
        data[16:] = 0.6 + rng.random(num_segments - 16).astype(np.float32) * 0.4
    elif any(w in desc_lower for w in ["hug", "touch", "hand", "finger"]):
        data[:] = 0.4 + rng.random(num_segments).astype(np.float32) * 0.3
    else:
        data[:] = 0.3 + rng.random(num_segments).astype(np.float32) * 0.2

    return data.tolist()


def submit_multimodal(brain, description):
    """Submit synthetic sensory data for a description to the brain.

    Called before brain.decide_full() so the SNN sensory bridges get native
    modality data instead of interpreting text embeddings as pixels/MFCCs.
    """
    desc_lower = description.lower()

    # Visual: anything with spatial/visual keywords
    visual_keywords = ["ball", "sky", "cloud", "leaf", "water", "flower", "block",
                       "cat", "rain", "candle", "snow", "bubble", "rainbow", "fire",
                       "tree", "bird", "river", "bridge", "stair", "color", "light",
                       "dark", "shadow", "sun", "moon", "star", "bright", "glow"]
    if any(w in desc_lower for w in visual_keywords):
        pixels, w, h, ch = generate_visual_frame(description)
        try:
            brain.submit_sensory("visual", pixels, width=w, height=h, channels=ch)
        except Exception:
            pass

    # Audio: anything with sound keywords
    audio_keywords = ["singing", "music", "thunder", "crash", "bark", "whisper",
                      "rain", "wave", "wind", "tick", "tap", "crunch", "howl",
                      "song", "melody", "loud", "quiet", "sound", "noise",
                      "clock", "bird", "purr"]
    if any(w in desc_lower for w in audio_keywords):
        samples = generate_audio_samples(description)
        try:
            brain.submit_sensory("audio", samples)
        except Exception:
            pass

    # Somatosensory: anything with touch/temperature keywords
    somato_keywords = ["warm", "hot", "cold", "cool", "smooth", "rough", "soft",
                       "hard", "heavy", "light", "wet", "dry", "sand", "fur",
                       "hug", "touch", "hand", "finger", "ice", "fire", "sun",
                       "freeze", "silk"]
    if any(w in desc_lower for w in somato_keywords):
        segments = generate_somatosensory(description)
        try:
            brain.submit_sensory("somatosensory", segments)
        except Exception:
            pass


logger = logging.getLogger("immerse_athena")

# ============================================================================
# Constants
# ============================================================================

BRAIN_INPUT_DIM = 1024
BRAIN_OUTPUT_DIM = 4096
TAG_DIM = 16           # [0:16]   modality flags + brain state
PRIMARY_DIM = 640      # [16:656] primary modality features
TEXT_DIM = 256         # [656:912] text semantic embedding (1024-dim model truncated to fit)
CONTEXT_DIM = 112      # [912:1024] biological context (arousal, sleep, dopamine, etc.)

CHECKPOINT_DIR = "checkpoints/athena"
DECODER_DIR = "checkpoints/athena/decoder"
STATE_FILE = "checkpoints/athena/immersive_state.json"
MASTERY_FILE = "checkpoints/athena/mastery_state.json"
CONTENT_CACHE = "checkpoints/athena/claude_content_cache.json"

# Adaptive curriculum constants
MASTERY_WINDOW = 20          # Rolling window for accuracy tracking
MASTERY_THRESHOLD_LOW = 0.3  # Below this: remedial (reduce difficulty)
MASTERY_THRESHOLD_HIGH = 0.8 # Above this: advance (increase difficulty)
ZPD_TARGET = 0.65            # Zone of Proximal Development target accuracy
DIFFICULTY_STEP = 0.1        # How much to adjust difficulty per evaluation

# Adaptive learning rate constants
BASE_LEARNING_RATE = 0.01    # 10x higher — compensates for adjoint gradient scaling
MIN_LEARNING_RATE = 0.001    # Floor — prevents stalling
MAX_LEARNING_RATE = 0.05     # Ceiling — prevents divergence
LR_SCALE_STRUGGLING = 1.5    # Scale up when mastery < THRESHOLD_LOW
LR_SCALE_MASTERED = 0.5      # Scale down when mastery > THRESHOLD_HIGH
LR_EVAL_INTERVAL = 100       # Steps between LR recalculations

# Cosine annealing LR scheduler constants
LR_WARMUP_STEPS = 500        # Linear warmup from MIN to BASE
LR_COSINE_T_MAX = 20000      # Full cosine cycle length


class CosineAnnealingLR:
    """Cosine annealing learning rate with linear warmup.

    Warmup: linearly ramp from lr_min to lr_max over warmup_steps.
    Then: cosine decay from lr_max to lr_min over t_max steps.
    """

    def __init__(self, lr_max=BASE_LEARNING_RATE, lr_min=MIN_LEARNING_RATE,
                 warmup_steps=LR_WARMUP_STEPS, t_max=LR_COSINE_T_MAX):
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.warmup_steps = warmup_steps
        self.t_max = t_max
        self.step_count = 0

    def step(self):
        self.step_count += 1

    def get_lr(self):
        import math
        if self.step_count < self.warmup_steps:
            # Linear warmup
            return self.lr_min + (self.lr_max - self.lr_min) * (self.step_count / max(1, self.warmup_steps))
        # Cosine decay
        progress = (self.step_count - self.warmup_steps) / max(1, self.t_max - self.warmup_steps)
        progress = min(progress, 1.0)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1.0 + math.cos(math.pi * progress))


LR_TRIGGER_FILE = "/tmp/athena_lr"


class AdaptiveLRController:
    """Auto-adjusts learning rate based on loss trends + manual override.

    Monitors rolling loss windows and adjusts LR:
    - Plateau (loss not improving): increase LR by scale_up
    - Diverging (loss increasing): decrease LR by scale_down
    - Improving: hold steady
    - Manual override via /tmp/athena_lr trigger file
    """

    def __init__(self, initial_lr=BASE_LEARNING_RATE, min_lr=MIN_LEARNING_RATE,
                 max_lr=MAX_LEARNING_RATE, window=500, patience=3,
                 scale_up=1.5, scale_down=0.5, min_delta=0.002):
        self.lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.window = window          # Loss averaging window
        self.patience = patience      # Evals without improvement before adjust
        self.scale_up = scale_up
        self.scale_down = scale_down
        self.min_delta = min_delta    # Minimum improvement to count
        self.best_loss = float('inf')
        self.stale_count = 0
        self.history = []             # (step, lr, avg_loss, action)

    def check_trigger_file(self):
        """Check for manual LR override via /tmp/athena_lr."""
        if os.path.exists(LR_TRIGGER_FILE):
            try:
                with open(LR_TRIGGER_FILE) as f:
                    new_lr = float(f.read().strip())
                os.remove(LR_TRIGGER_FILE)
                new_lr = max(self.min_lr, min(self.max_lr, new_lr))
                old_lr = self.lr
                self.lr = new_lr
                self.stale_count = 0
                self.best_loss = float('inf')  # Reset plateau detection
                print(f"    [LR] Manual override: {old_lr:.6f} → {new_lr:.6f}")
                return True
            except (ValueError, OSError):
                try:
                    os.remove(LR_TRIGGER_FILE)
                except OSError:
                    pass
        return False

    def update(self, avg_loss, step):
        """Update LR based on loss trend. Call every eval interval (e.g. 500 steps)."""
        # Check manual override first
        if self.check_trigger_file():
            self.history.append((step, self.lr, avg_loss, "MANUAL"))
            return self.lr

        action = "HOLD"
        if avg_loss < self.best_loss - self.min_delta:
            # Improving — reset patience, keep current LR
            self.best_loss = avg_loss
            self.stale_count = 0
            action = "IMPROVING"
        else:
            self.stale_count += 1

            if self.stale_count >= self.patience:
                if avg_loss > self.best_loss + self.min_delta:
                    # Loss is going UP — decrease LR
                    old_lr = self.lr
                    self.lr = max(self.min_lr, self.lr * self.scale_down)
                    self.stale_count = 0
                    action = f"DIVERGE {old_lr:.6f}→{self.lr:.6f}"
                    print(f"    [LR] Loss diverging — reducing: {old_lr:.6f} → {self.lr:.6f}")
                else:
                    # Plateau — increase LR to explore
                    old_lr = self.lr
                    self.lr = min(self.max_lr, self.lr * self.scale_up)
                    self.stale_count = 0
                    self.best_loss = avg_loss  # Reset baseline after boost
                    action = f"PLATEAU {old_lr:.6f}→{self.lr:.6f}"
                    print(f"    [LR] Loss plateaued — boosting: {old_lr:.6f} → {self.lr:.6f}")

        self.history.append((step, self.lr, avg_loss, action))
        return self.lr

    def get_lr(self):
        """Get current LR (also checks trigger file)."""
        self.check_trigger_file()
        return self.lr

    def update_from_utm(self, brain, step):
        """Use UTM DFA health to guide LR adjustment (supplements loss-based).

        DFA health provides principled, fractal-analysis-based training diagnostics:
        - OPTIMAL (α≈0.8-1.2): pink noise, ideal learning dynamics
        - NOISY (α<0.6): too much noise, reduce LR
        - DRIFTING (α>1.3): diverging, reduce LR aggressively
        - OSCILLATING (α<0.3): chaotic, reduce LR significantly
        - PLATEAU (H>0.8): stuck, increase LR
        """
        try:
            health = brain.utm_get_training_health()
        except (AttributeError, RuntimeError):
            return  # UTM not available

        health_name = health.get("health_name", "unknown")
        dfa_exp = health.get("dfa_exponent", 0.0)

        # DFA-guided LR adjustment factors
        adjustments = {
            "noisy": 0.8,
            "drifting": 0.5,
            "oscillating": 0.3,
            "plateau": 1.5,
        }
        factor = adjustments.get(health_name, 1.0)
        if factor != 1.0:
            old_lr = self.lr
            self.lr = max(self.min_lr, min(self.max_lr, self.lr * factor))
            if self.lr != old_lr:
                print(f"    [LR/DFA] Health={health_name} (α={dfa_exp:.2f}) "
                      f"— adjusting: {old_lr:.6f} → {self.lr:.6f}")
                self.history.append((step, self.lr, 0.0, f"DFA_{health_name.upper()}"))


class AdaptiveBatchSize:
    """Adaptive batch size based on loss stability.

    Starts with min_batch, increases when loss is stable, decreases on spikes.
    """

    def __init__(self, min_batch=4, max_batch=32, initial=8):
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.current = initial
        self.recent_losses = []
        self.adjust_interval = 200
        self.step_count = 0

    def record_loss(self, loss):
        if loss is not None and loss >= 0:
            self.recent_losses.append(loss)
            if len(self.recent_losses) > 100:
                self.recent_losses = self.recent_losses[-100:]

    def step(self):
        self.step_count += 1
        if self.step_count % self.adjust_interval == 0 and len(self.recent_losses) >= 50:
            first_half = np.mean(self.recent_losses[:50])
            second_half = np.mean(self.recent_losses[50:])
            variance = np.var(self.recent_losses[-50:])
            if variance < 0.001 and second_half <= first_half:
                # Stable — increase batch size for throughput
                self.current = min(self.current * 2, self.max_batch)
            elif variance > 0.01 or second_half > first_half * 1.5:
                # Unstable — decrease batch size for finer updates
                self.current = max(self.current // 2, self.min_batch)

    @property
    def size(self):
        return self.current


class MiniBatchTrainer:
    """Accumulates learn_vector calls into mini-batches for GPU gradient accumulation.

    Instead of per-sample SGD, collects N samples and calls learn_vector_batch()
    for more stable gradients and better GPU utilization.
    """

    def __init__(self, brain, batch_size=8):
        self.brain = brain
        self.batch_size = batch_size
        self._buffer = []  # list of (features, target, label, confidence, lr)
        self._last_avg_loss = 0.0

    def learn(self, features, target, label="", confidence=0.5, learning_rate=None):
        """Buffer a sample. Returns estimated loss (batch average when flushed)."""
        self._buffer.append((features, target, label, confidence, learning_rate))
        if len(self._buffer) >= self.batch_size:
            return self.flush()
        return self._last_avg_loss  # return last batch average as estimate

    def flush(self):
        """Flush accumulated samples as a batch."""
        if not self._buffer:
            return self._last_avg_loss

        pairs = [(f, t) for f, t, _, _, _ in self._buffer]
        # Use the average LR across samples
        lrs = [lr for _, _, _, _, lr in self._buffer if lr is not None]
        avg_lr = sum(lrs) / len(lrs) if lrs else None

        try:
            lr_kw = {"learning_rate": avg_lr} if avg_lr is not None else {}
            avg_loss = self.brain.learn_vector_batch(pairs, **lr_kw)
            if avg_loss is not None and avg_loss >= 0:
                self._last_avg_loss = avg_loss
            else:
                self._last_avg_loss = 0.0
        except (AttributeError, TypeError):
            # Fallback: per-sample learning
            losses = []
            for f, t, lbl, conf, lr in self._buffer:
                lr_kw = {"learning_rate": lr} if lr is not None else {}
                loss = self.brain.learn_vector(f, t, label=lbl,
                                               confidence=conf, **lr_kw)
                losses.append(loss if loss is not None and loss >= 0 else 0.0)
            self._last_avg_loss = sum(losses) / len(losses) if losses else 0.0
        except Exception:
            # Fallback: per-sample learning
            losses = []
            for f, t, lbl, conf, lr in self._buffer:
                lr_kw = {"learning_rate": lr} if lr is not None else {}
                loss = self.brain.learn_vector(f, t, label=lbl,
                                               confidence=conf, **lr_kw)
                losses.append(loss if loss is not None and loss >= 0 else 0.0)
            self._last_avg_loss = sum(losses) / len(losses) if losses else 0.0

        self._buffer.clear()
        return self._last_avg_loss


class ParallelDataLoader:
    """Parallel dataset loading with prefetch queue.

    Uses a thread pool to generate and encode stimuli in parallel,
    keeping a buffer of ready-to-use batches ahead of training.
    """

    def __init__(self, source, composer, num_workers=2, prefetch_batches=4, batch_size=64):
        from threading import Thread
        from queue import Queue
        self.source = source
        self.composer = composer
        self.batch_size = batch_size
        self.queue = Queue(maxsize=prefetch_batches)
        self._stop = False
        self._workers = []
        for _ in range(num_workers):
            t = Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    def _worker(self):
        while not self._stop:
            descs = []
            for _ in range(self.batch_size):
                if random.random() < 0.5:
                    descs.append(self.source.get_sensory())
                else:
                    desc, _ = generate_sensory_exposure()
                    descs.append(desc)
            targets = batch_make_semantic_targets(descs)
            features_list = [self.composer.compose(text=d, modality="text") for d in descs]
            batch = list(zip(descs, features_list, targets))
            try:
                self.queue.put(batch, timeout=10)
            except Exception:
                if self._stop:
                    break

    def get_batch(self):
        return self.queue.get(timeout=60)

    def stop(self):
        self._stop = True


# ============================================================================
# Hard Example Mining: curriculum-aware data sampling
# ============================================================================

class HardExampleMiner:
    """Track per-example loss and replay high-loss examples more frequently.

    Maintains a priority buffer of hard examples. During sampling,
    mixes fresh data with replayed hard examples (ratio controlled by
    replay_fraction). Hard examples decay in priority over time to
    avoid getting stuck.
    """

    def __init__(self, max_buffer=500, replay_fraction=0.25, decay=0.95):
        self.buffer = []  # list of (priority, description, features, target)
        self.max_buffer = max_buffer
        self.replay_fraction = replay_fraction
        self.decay = decay
        import heapq
        self._heapq = heapq

    def record(self, loss, description, features, target):
        """Record a training example with its loss."""
        if loss is None or loss < 0:
            return
        # Use negative loss for min-heap (we want highest loss on top)
        entry = (loss, description, features, target)
        if len(self.buffer) < self.max_buffer:
            self.buffer.append(entry)
        elif loss > self.buffer[0][0]:
            # Replace the lowest-loss entry
            self.buffer[0] = entry
            # Maintain partial sort — just keep the list
            self.buffer.sort(key=lambda x: x[0])

    def get_hard_examples(self, n):
        """Get n hard examples for replay, or fewer if buffer is small."""
        if not self.buffer:
            return []
        # Sort by loss descending, take top n
        sorted_buf = sorted(self.buffer, key=lambda x: -x[0])
        examples = sorted_buf[:n]
        # Decay priorities
        self.buffer = [(loss * self.decay, desc, feat, tgt)
                       for loss, desc, feat, tgt in self.buffer]
        return [(desc, feat, tgt) for _, desc, feat, tgt in examples]

    def should_replay(self):
        """Return True if we should replay hard examples this step."""
        return len(self.buffer) >= 20 and random.random() < self.replay_fraction


# ============================================================================
# Embedding Adapter: lightweight projection to domain-adapted space
# ============================================================================

class EmbeddingAdapter:
    """Lightweight linear projection trained on (embedding, brain_output) pairs.

    Instead of fine-tuning the full sentence-transformer, this trains a small
    projection matrix W that maps frozen embeddings to domain-adapted space:
        adapted = embedding @ W + bias

    Trains online using collected (input_embedding, output_embedding) pairs
    from the brain's responses. Refits periodically using least-squares.
    """

    def __init__(self, input_dim=1024, output_dim=1024, refit_interval=1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.refit_interval = refit_interval
        self.W = np.eye(min(input_dim, output_dim), dtype=np.float32)
        if input_dim != output_dim:
            self.W = np.random.randn(input_dim, output_dim).astype(np.float32) * 0.01
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self.pairs = []  # (input_emb, target_emb) tuples
        self.max_pairs = 5000
        self.step_count = 0
        self.fitted = False

    def record_pair(self, input_emb, brain_output_emb):
        """Record a (frozen_embedding, brain_response_embedding) pair."""
        inp = np.asarray(input_emb, dtype=np.float32).ravel()[:self.input_dim]
        out = np.asarray(brain_output_emb, dtype=np.float32).ravel()[:self.output_dim]
        if len(inp) < self.input_dim or len(out) < self.output_dim:
            return
        self.pairs.append((inp, out))
        if len(self.pairs) > self.max_pairs:
            self.pairs = self.pairs[-self.max_pairs:]
        self.step_count += 1

    def maybe_refit(self):
        """Refit the projection if enough new data has accumulated."""
        if self.step_count < self.refit_interval or len(self.pairs) < 100:
            return
        self.step_count = 0
        # Least-squares fit: min ||X @ W + b - Y||^2
        X = np.array([p[0] for p in self.pairs], dtype=np.float32)
        Y = np.array([p[1] for p in self.pairs], dtype=np.float32)
        # Center
        X_mean = X.mean(axis=0)
        Y_mean = Y.mean(axis=0)
        X_c = X - X_mean
        Y_c = Y - Y_mean
        # Solve via pseudo-inverse (truncated SVD for stability)
        try:
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            # Truncate small singular values
            threshold = max(S) * 1e-5
            S_inv = np.where(S > threshold, 1.0 / S, 0.0)
            self.W = (Vt.T * S_inv) @ U.T @ Y_c
            self.bias = Y_mean - X_mean @ self.W
            self.fitted = True
        except np.linalg.LinAlgError:
            pass  # Keep current projection

    def adapt(self, embedding):
        """Apply the learned projection to a frozen embedding."""
        if not self.fitted:
            return embedding
        emb = np.asarray(embedding, dtype=np.float32).ravel()[:self.input_dim]
        adapted = emb @ self.W + self.bias
        return adapted


# ============================================================================
# Knowledge Distillation: extract structured knowledge from Claude
# ============================================================================

class ClaudeDistiller:
    """Extract structured concept relationships from Claude's lessons.

    When Claude generates a lesson, also ask for key concepts and their
    relationships. Use these to create additional training targets that
    encode the relational structure, not just the surface text embedding.
    """

    def __init__(self, teacher):
        self.teacher = teacher
        self._concept_cache = {}

    def distill_concepts(self, lesson_text):
        """Ask Claude to extract key concepts and relationships from a lesson.

        Returns list of (concept, relation, concept) triples and their embeddings.
        """
        if not self.teacher:
            return []

        # Check cache
        key = lesson_text[:100]
        if key in self._concept_cache:
            return self._concept_cache[key]

        prompt = (
            f"From this lesson, extract 2-4 key concepts and their relationships.\n"
            f"Lesson: \"{lesson_text}\"\n\n"
            f"Format each as: CONCEPT1 -> RELATION -> CONCEPT2\n"
            f"Example: photosynthesis -> requires -> sunlight\n"
            f"Only output the concept triples, one per line."
        )

        try:
            response = self.teacher._call_claude(prompt, max_tokens=128)
            triples = []
            for line in response.strip().split("\n"):
                parts = [p.strip() for p in line.split("->")]
                if len(parts) == 3:
                    # Create a composite embedding from the triple
                    composite = f"{parts[0]} {parts[1]} {parts[2]}"
                    emb = encode_text(composite)
                    triples.append((parts[0], parts[1], parts[2], emb))

            if len(self._concept_cache) > 500:
                # Evict oldest
                oldest = next(iter(self._concept_cache))
                del self._concept_cache[oldest]
            self._concept_cache[key] = triples
            return triples
        except Exception:
            return []

    def make_distillation_targets(self, lesson_text, target_dim=BRAIN_OUTPUT_DIM):
        """Create additional training targets from distilled concepts.

        Returns list of (input_features_text, target_vector) tuples
        for supplementary training on relational structure.
        """
        triples = self.distill_concepts(lesson_text)
        targets = []
        for concept1, relation, concept2, emb in triples:
            # Create target by tiling the composite embedding
            target = np.zeros(target_dim, dtype=np.float32)
            for i in range(0, target_dim, len(emb)):
                n = min(len(emb), target_dim - i)
                target[i:i + n] = emb[:n]
            # Input text is the first concept (student should learn the connection)
            targets.append((concept1, target.tolist()))
        return targets


# ============================================================================
# Multi-Resolution Temporal: fast + slow LNN processing
# ============================================================================

class MultiResolutionTemporal:
    """Process LNN at multiple timescales for richer temporal representation.

    Fast channel: updates every step (sensory-level, short-term patterns).
    Slow channel: updates every N steps with averaged input (conceptual, long-term).

    The fast channel captures immediate patterns; the slow channel captures
    trends and persistent state. Both contribute to the biological context.
    """

    def __init__(self, brain, slow_interval=5):
        self.brain = brain
        self.slow_interval = slow_interval
        self.step_count = 0
        self.slow_accumulator = None
        self.slow_count = 0

    def step(self, features):
        """Process one temporal step at both resolutions."""
        self.step_count += 1
        fast_input = features[:128]

        # Fast channel — every step
        try:
            self.brain.lnn_forward_step(fast_input)
        except Exception:
            pass

        # Accumulate for slow channel
        arr = np.array(fast_input[:128], dtype=np.float32)
        if self.slow_accumulator is None:
            self.slow_accumulator = np.zeros_like(arr)
        self.slow_accumulator += arr
        self.slow_count += 1

        # Slow channel — every N steps with averaged input
        if self.slow_count >= self.slow_interval:
            avg_input = (self.slow_accumulator / self.slow_count).tolist()
            # Scale down to represent "smoothed" temporal signal
            scaled = [x * 0.5 for x in avg_input]
            try:
                self.brain.lnn_forward_step(scaled)
            except Exception:
                pass
            self.slow_accumulator = np.zeros_like(arr)
            self.slow_count = 0

    def get_temporal_context(self):
        """Get combined fast+slow LNN state for biological context."""
        try:
            state = self.brain.lnn_get_state()
            return state
        except Exception:
            return None


# ============================================================================
# Early Stopping: evaluation-driven training termination
# ============================================================================

class EarlyStopping:
    """Stop training when validation metrics plateau.

    Tracks a running best metric (e.g., differentiation score from eval probes).
    If no improvement for `patience` evaluations, signals to stop.
    """

    def __init__(self, patience=5, min_delta=0.01, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode  # "min" for loss, "max" for accuracy
        self.best = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def check(self, metric):
        """Check if training should stop. Returns True if patience exhausted."""
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset for next stage."""
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False


# ============================================================================
# Adaptive Curriculum: Mastery tracking + difficulty adjustment
# ============================================================================

class MasteryTracker:
    """Track per-domain mastery using rolling accuracy windows."""

    def __init__(self):
        self.domains = {}  # domain -> {attempts: [], difficulty: float}

    def record(self, domain, loss, confidence):
        """Record a training attempt for a domain."""
        if domain not in self.domains:
            self.domains[domain] = {
                "attempts": [],
                "difficulty": 0.5,
                "total_attempts": 0,
                "total_correct": 0,
            }
        entry = self.domains[domain]
        # Accuracy proxy: low loss + high confidence = correct
        correct = loss is not None and loss < 0.5 and confidence > 0.4
        entry["attempts"].append(1.0 if correct else 0.0)
        entry["total_attempts"] += 1
        if correct:
            entry["total_correct"] += 1
        # Keep rolling window
        if len(entry["attempts"]) > MASTERY_WINDOW:
            entry["attempts"] = entry["attempts"][-MASTERY_WINDOW:]

    def get_mastery(self, domain):
        """Get current mastery level [0-1] for a domain."""
        if domain not in self.domains:
            return 0.0
        attempts = self.domains[domain]["attempts"]
        if not attempts:
            return 0.0
        return sum(attempts) / len(attempts)

    def get_difficulty(self, domain):
        """Get current difficulty level [0-1] for a domain."""
        if domain not in self.domains:
            return 0.5
        return self.domains[domain]["difficulty"]

    def adapt_difficulty(self, domain):
        """Adjust difficulty based on mastery — target ZPD."""
        mastery = self.get_mastery(domain)
        if domain not in self.domains:
            return 0.5
        entry = self.domains[domain]
        if mastery > MASTERY_THRESHOLD_HIGH:
            entry["difficulty"] = min(1.0, entry["difficulty"] + DIFFICULTY_STEP)
        elif mastery < MASTERY_THRESHOLD_LOW:
            entry["difficulty"] = max(0.1, entry["difficulty"] - DIFFICULTY_STEP)
        return entry["difficulty"]

    def select_domain(self, available_domains):
        """Select next domain — prioritize those in ZPD, avoid mastered ones."""
        scored = []
        for domain in available_domains:
            mastery = self.get_mastery(domain)
            # Score: distance from ZPD target (closer = higher priority)
            # Untried domains get high priority (novelty)
            if domain not in self.domains or len(self.domains[domain]["attempts"]) < 3:
                score = 1.0  # Unexplored — high priority
            else:
                score = 1.0 - abs(mastery - ZPD_TARGET)
            scored.append((domain, score))

        # Weighted random selection by score
        total = sum(s for _, s in scored)
        if total <= 0:
            return random.choice(available_domains)
        r = random.random() * total
        cumulative = 0
        for domain, score in scored:
            cumulative += score
            if r <= cumulative:
                return domain
        return scored[-1][0]

    def get_confidence_for_domain(self, domain, base_progress):
        """Compute training confidence from mastery + difficulty."""
        difficulty = self.get_difficulty(domain)
        mastery = self.get_mastery(domain)
        # Higher difficulty + higher mastery = higher training confidence
        return min(0.95, max(0.3, difficulty * 0.5 + mastery * 0.3 + base_progress * 0.2))

    def get_adaptive_lr(self, domain):
        """Compute adaptive learning rate based on domain mastery.

        Struggling domains (mastery < 0.3) get higher LR for faster learning.
        Mastered domains (mastery > 0.8) get lower LR for fine-tuning.
        ZPD domains use baseline LR.
        """
        mastery = self.get_mastery(domain)
        if mastery < MASTERY_THRESHOLD_LOW:
            lr = BASE_LEARNING_RATE * LR_SCALE_STRUGGLING
        elif mastery > MASTERY_THRESHOLD_HIGH:
            lr = BASE_LEARNING_RATE * LR_SCALE_MASTERED
        else:
            # Smooth interpolation within ZPD
            # At ZPD_TARGET, use BASE_LEARNING_RATE exactly
            # Scale linearly between thresholds
            lr = BASE_LEARNING_RATE
        return max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, lr))

    def save(self, filepath):
        """Persist mastery state to JSON."""
        import json
        data = {}
        for domain, entry in self.domains.items():
            data[domain] = {
                "difficulty": entry["difficulty"],
                "total_attempts": entry["total_attempts"],
                "total_correct": entry["total_correct"],
                "recent_accuracy": self.get_mastery(domain),
            }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath):
        """Load mastery state from JSON."""
        import json
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Failed to load mastery state from {filepath}: {e}")
            return
        for domain, entry in data.items():
            self.domains[domain] = {
                "attempts": [],
                "difficulty": entry.get("difficulty", 0.5),
                "total_attempts": entry.get("total_attempts", 0),
                "total_correct": entry.get("total_correct", 0),
            }

    def summary(self):
        """Print mastery summary."""
        if not self.domains:
            return "  No mastery data yet."
        lines = []
        for domain in sorted(self.domains.keys()):
            m = self.get_mastery(domain)
            d = self.get_difficulty(domain)
            n = self.domains[domain]["total_attempts"]
            lines.append(f"    {domain:20s}: mastery={m:.2f} difficulty={d:.2f} attempts={n}")
        return "\n".join(lines)


# ============================================================================
# Sensory Composer: 1024-dim cortex-native input with biological context
# ============================================================================

class SensoryComposer:
    """Compose 1024-dim input vectors with embedded biological state."""

    def __init__(self, brain):
        self.brain = brain

    def compose(self, text=None, modality="text", primary_features=None,
                text_embedding=None):
        """Build a 1024-dim input vector with brain state context.

        Layout: [tag:16 | primary:640 | text:256 | context:112]
        """
        vec = np.zeros(BRAIN_INPUT_DIM, dtype=np.float32)

        # --- Tag section [0:16] ---
        modality_map = {"text": 0, "visual": 1, "audio": 2, "speech": 3,
                        "ethics": 4, "imagination": 5}
        mod_idx = modality_map.get(modality, 0)
        vec[mod_idx] = 1.0

        # Brain state in tags
        try:
            vec[8] = self.brain.medulla_get_arousal()
        except Exception:
            vec[8] = 0.5
        try:
            vec[9] = float(self.brain.sleep_get_state()) / 4.0
        except Exception:
            vec[9] = 0.0
        try:
            vec[10] = self.brain.medulla_get_circadian_efficiency()
        except Exception:
            vec[10] = 1.0
        try:
            health = self.brain.substrate_get_health()
            health_map = {"OPTIMAL": 1.0, "STRESSED": 0.7,
                          "COMPROMISED": 0.4, "CRITICAL": 0.1}
            vec[11] = health_map.get(health, 0.5)
        except Exception:
            vec[11] = 1.0

        # --- Primary features [16:656] ---
        if primary_features is not None:
            pf = np.array(primary_features, dtype=np.float32)
            n = min(len(pf), PRIMARY_DIM)
            vec[TAG_DIM:TAG_DIM + n] = pf[:n]
        elif text is not None:
            emb = encode_text(text)  # 1024-dim
            # Pad to 640 with zeros
            n = min(len(emb), PRIMARY_DIM)
            vec[TAG_DIM:TAG_DIM + n] = emb[:n]

        # --- Text semantic embedding [656:912] ---
        if text_embedding is not None:
            te = np.array(text_embedding, dtype=np.float32)
            n = min(len(te), TEXT_DIM)
            vec[TAG_DIM + PRIMARY_DIM:TAG_DIM + PRIMARY_DIM + n] = te[:n]
        elif text is not None:
            emb = encode_text(text)  # 1024-dim — truncate to TEXT_DIM
            n = min(len(emb), TEXT_DIM)
            vec[TAG_DIM + PRIMARY_DIM:TAG_DIM + PRIMARY_DIM + n] = emb[:n]

        # --- Biological context [912:1024] ---
        ctx_start = TAG_DIM + PRIMARY_DIM + TEXT_DIM
        try:
            vec[ctx_start] = self.brain.medulla_get_arousal()
            vec[ctx_start + 1] = self.brain.sleep_get_pressure()
            vec[ctx_start + 2] = self.brain.medulla_get_circadian_efficiency()
        except Exception:
            pass
        try:
            vec[ctx_start + 3] = self.brain.bg_get_dopamine()
            vec[ctx_start + 4] = self.brain.bg_get_rpe()
            vec[ctx_start + 5] = self.brain.bg_get_conflict()
            vec[ctx_start + 6] = float(self.brain.bg_get_mode())
        except Exception:
            pass
        try:
            met = self.brain.substrate_get_metabolic()
            vec[ctx_start + 7] = met.get("atp", 1.0)
            vec[ctx_start + 8] = met.get("capacity", 1.0)
        except Exception:
            pass
        # LNN state (32 dims) at context_start + 16..48
        try:
            lnn_state = self.brain.lnn_get_state()
            if lnn_state:
                n = min(len(lnn_state), 32)
                for i in range(n):
                    vec[ctx_start + 16 + i] = lnn_state[i]
        except Exception:
            pass

        return vec.tolist()


class TargetDiversifier:
    """Diversifies semantic targets to combat mode collapse.

    Three strategies work together:
    1. Category signal: Adds orthogonal one-hot category component
    2. Category rotation: Applies a fixed random rotation per category
    3. Mean subtraction: Subtracts running mean of recent targets

    Without diversification, "a brown cow" and "a black dog" produce embeddings
    with cosine similarity >0.8 — the network can't help but collapse them.
    With diversification, each category occupies a distinct region of target space.
    """

    # Category definitions — each text gets assigned by keyword matching
    CATEGORIES = [
        "animal", "nature", "food", "body", "home", "vehicle", "music",
        "weather", "tool", "science", "math", "philosophy", "psychology",
        "technology", "literature", "art", "emotion", "social", "space",
        "geography", "history", "language", "sport", "unknown"
    ]

    # Keywords for category detection
    _CATEGORY_KEYWORDS = {
        "animal": ["dog", "cat", "bird", "fish", "horse", "bear", "cow",
                   "elephant", "whale", "dolphin", "rabbit", "frog", "snake",
                   "butterfly", "bee", "owl", "lion", "tiger", "monkey",
                   "sheep", "pig", "chicken", "duck", "deer", "wolf",
                   "fox", "penguin", "turtle", "spider", "ant", "mouse",
                   "squirrel", "giraffe", "zebra", "eagle", "hawk",
                   "parrot", "beetle", "ladybug", "caterpillar", "peacock",
                   "rooster", "skunk", "chipmunk", "starling", "goldfish"],
        "nature": ["tree", "flower", "river", "mountain", "ocean", "cloud",
                   "rain", "snow", "forest", "desert", "volcano", "rainbow",
                   "moon", "sun", "star", "sunset", "sunrise", "lake",
                   "garden", "leaf", "seed", "grass", "vine", "moss",
                   "tide", "wave", "fog", "mist", "aurora", "northern lights"],
        "food": ["apple", "bread", "milk", "egg", "rice", "honey",
                 "banana", "cookie", "cheese", "orange", "strawberry",
                 "chocolate", "water", "juice", "soup", "cake", "pizza",
                 "potato", "carrot", "tomato", "pepper", "lemon", "grape",
                 "eggplant", "peel", "mint"],
        "body": ["hand", "eye", "heart", "brain", "finger", "arm",
                 "leg", "nose", "ear", "mouth", "teeth", "hair",
                 "skin", "bone", "blood", "muscle"],
        "home": ["house", "door", "window", "chair", "table", "bed",
                 "lamp", "book", "clock", "mirror", "pillow", "blanket",
                 "cup", "spoon", "plate", "candle", "key", "stairs",
                 "ceiling", "floor", "wall", "roof", "kitchen", "sink",
                 "radiator", "curtain", "fireplace"],
        "vehicle": ["car", "bus", "train", "bicycle", "boat", "airplane",
                    "truck", "ship", "rocket", "helicopter", "subway"],
        "music": ["music", "song", "piano", "guitar", "drum", "violin",
                  "trumpet", "flute", "melody", "rhythm", "harmony",
                  "orchestra", "choir", "cello", "accordion", "metronome",
                  "tempo", "note", "chord", "scale", "octave", "fugue",
                  "chimes", "wind chimes", "foghorn", "bell"],
        "weather": ["rain", "thunder", "lightning", "wind", "storm",
                    "tornado", "hurricane", "fog", "hail", "frost",
                    "breeze", "temperature", "climate"],
        "tool": ["hammer", "scissors", "needle", "pencil", "brush",
                 "wrench", "screwdriver", "shovel", "rope", "saw",
                 "magnifying glass", "prism", "kaleidoscope", "pinwheel",
                 "kite", "balloon"],
        "science": ["atom", "molecule", "cell", "DNA", "evolution",
                    "gravity", "energy", "photosynthesis", "electron",
                    "neuron", "synapse", "gene", "protein", "vaccine",
                    "antibiotic", "electromagnetic", "spectrum",
                    "tectonic", "mitochondria", "experiment"],
        "math": ["number", "equation", "theorem", "geometry", "algebra",
                 "calculus", "probability", "statistics", "fibonacci",
                 "prime", "pi ", "fraction", "exponential", "logarithm"],
        "philosophy": ["socrates", "plato", "aristotle", "descartes",
                       "kant", "ethics", "morality", "consciousness",
                       "free will", "existential", "epistemology",
                       "trolley problem", "meaning of life"],
        "psychology": ["memory", "emotion", "cognitive", "attachment",
                       "motivation", "placebo", "empathy", "creativity",
                       "bias", "stress", "anxiety", "perception"],
        "technology": ["computer", "internet", "algorithm", "software",
                       "encryption", "programming", "artificial intelligence",
                       "machine learning", "neural network", "quantum",
                       "digital", "robot", "database", "code"],
        "literature": ["shakespeare", "poetry", "novel", "fable",
                       "mythology", "metaphor", "narrative", "tragedy",
                       "comedy", "irony", "story", "author", "poem"],
        "art": ["painting", "sculpture", "color", "perspective",
                "photography", "architecture", "composition", "canvas",
                "abstract", "portrait", "dance", "gallery", "sketch"],
        "space": ["planet", "galaxy", "comet", "asteroid", "orbit",
                  "telescope", "constellation", "nebula", "mars",
                  "saturn", "jupiter", "milky way", "cosmos"],
    }

    def __init__(self, embedding_dim=1024, target_dim=BRAIN_OUTPUT_DIM,
                 category_weight=0.3, rotation_strength=0.2,
                 mean_sub_strength=0.5, mean_buffer_size=200):
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        self.num_categories = len(self.CATEGORIES)
        self.category_weight = category_weight
        self.rotation_strength = rotation_strength
        self.mean_sub_strength = mean_sub_strength

        # Pre-generate a fixed random rotation matrix per category (seeded)
        rng = np.random.RandomState(42)
        self._rotations = {}
        for i, cat in enumerate(self.CATEGORIES):
            # Random orthogonal matrix via QR decomposition
            A = rng.randn(embedding_dim, embedding_dim).astype(np.float32)
            Q, _ = np.linalg.qr(A)
            # Blend with identity: R = (1-s)*I + s*Q
            self._rotations[cat] = (
                (1.0 - rotation_strength) * np.eye(embedding_dim, dtype=np.float32)
                + rotation_strength * Q
            )

        # Running mean of recent targets for mean subtraction
        self._mean_buffer = []
        self._mean_buffer_size = mean_buffer_size
        self._running_mean = np.zeros(embedding_dim, dtype=np.float32)

    def _detect_category(self, text):
        """Detect category from text via keyword matching."""
        text_lower = text.lower()
        best_cat = "unknown"
        best_count = 0
        for cat, keywords in self._CATEGORY_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_cat = cat
        return best_cat

    def diversify(self, embedding, text, category=None):
        """Apply all three diversification strategies to an embedding.

        Args:
            embedding: 1024-dim semantic embedding (numpy array)
            text: Original text (for category detection)
            category: Optional explicit category string

        Returns:
            Diversified embedding (1024-dim numpy array)
        """
        emb = np.array(embedding, dtype=np.float32)

        # 1. Category rotation — rotate embedding into category-specific subspace
        cat = category or self._detect_category(text)
        if cat in self._rotations:
            emb = self._rotations[cat] @ emb

        # 2. Category signal — add scaled one-hot in a reserved band
        cat_idx = self.CATEGORIES.index(cat) if cat in self.CATEGORIES else len(self.CATEGORIES) - 1
        cat_signal = np.zeros(self.embedding_dim, dtype=np.float32)
        # Spread the category signal across multiple dimensions for robustness
        band_start = (cat_idx * 40) % self.embedding_dim
        for k in range(40):
            dim = (band_start + k) % self.embedding_dim
            cat_signal[dim] = self.category_weight
        emb = emb + cat_signal

        # 3. Mean subtraction — push away from the running centroid
        if len(self._mean_buffer) > 10:
            emb = emb - self.mean_sub_strength * self._running_mean

        # Update running mean
        self._mean_buffer.append(emb.copy())
        if len(self._mean_buffer) > self._mean_buffer_size:
            self._mean_buffer.pop(0)
        self._running_mean = np.mean(self._mean_buffer, axis=0)

        # Re-normalize to preserve scale
        orig_norm = np.linalg.norm(embedding)
        new_norm = np.linalg.norm(emb)
        if new_norm > 1e-8 and orig_norm > 1e-8:
            emb = emb * (orig_norm / new_norm)

        return emb


# Global diversifier instance
_target_diversifier = TargetDiversifier()


def make_semantic_target(text, target_dim=BRAIN_OUTPUT_DIM, category=None):
    """Create a diversified target vector from text.

    Uses three strategies to spread targets apart:
    1. Per-category rotation of the embedding space
    2. Category one-hot signal injection
    3. Running mean subtraction (push away from centroid)
    """
    emb = encode_text(text)  # 1024-dim
    emb = _target_diversifier.diversify(emb, text, category=category)

    target = np.zeros(target_dim, dtype=np.float32)
    for i in range(0, target_dim, len(emb)):
        n = min(len(emb), target_dim - i)
        target[i:i + n] = emb[:n]
    return target.tolist()


def tile_to_brain_input(embedding, dim=BRAIN_INPUT_DIM):
    """Tile a shorter embedding to brain input dimension."""
    vec = np.zeros(dim, dtype=np.float32)
    emb = np.array(embedding, dtype=np.float32)
    for i in range(0, dim, len(emb)):
        n = min(len(emb), dim - i)
        vec[i:i + n] = emb[:n]
    return vec.tolist()


def batch_make_semantic_targets(texts, target_dim=BRAIN_OUTPUT_DIM):
    """Batch-compute diversified semantic targets."""
    embeddings = batch_encode_texts(texts)
    targets = []
    for emb, text in zip(embeddings, texts):
        emb = _target_diversifier.diversify(emb, text)
        target = np.zeros(target_dim, dtype=np.float32)
        for i in range(0, target_dim, len(emb)):
            n = min(len(emb), target_dim - i)
            target[i:i + n] = emb[:n]
        targets.append(target.tolist())
    return targets


class ContrastiveRegularizer:
    """Combats mode collapse by pushing outputs apart for different inputs.

    Maintains a rolling buffer of (input_embedding, output_vector) pairs.
    Every `interval` steps, samples pairs from the buffer and applies
    contrastive corrections: if two outputs are very similar (high cosine)
    but their inputs are dissimilar, we push the outputs apart by training
    each to move AWAY from the other.

    This gives the brain a gradient signal: "different inputs should produce
    different outputs."
    """

    def __init__(self, buffer_size=200, interval=50, batch_size=16,
                 similarity_threshold=0.90, min_input_distance=0.3,
                 strength=0.3):
        """
        Args:
            buffer_size: Max recent pairs to keep.
            interval: Apply contrastive correction every N steps.
            batch_size: Number of pairs to sample per correction.
            similarity_threshold: Output cosine sim above this triggers correction.
            min_input_distance: Minimum input dissimilarity to count as "different".
            strength: Blend factor for anti-target (0=no correction, 1=full push).
        """
        self.buffer_size = buffer_size
        self.interval = interval
        self.batch_size = batch_size
        self.sim_threshold = similarity_threshold
        self.min_input_dist = min_input_distance
        self.strength = strength

        self._input_buf = []   # list of 1024-dim input embeddings
        self._output_buf = []  # list of output vectors (full dim)
        self._feature_buf = [] # list of brain input features (for re-training)
        self._step = 0
        self._corrections_applied = 0
        self._total_checks = 0

    def record(self, input_embedding, output_vector, features):
        """Record an (input, output, features) triple."""
        inp = np.asarray(input_embedding, dtype=np.float32).ravel()
        out = np.asarray(output_vector, dtype=np.float32).ravel()
        self._input_buf.append(inp)
        self._output_buf.append(out)
        self._feature_buf.append(features)
        if len(self._input_buf) > self.buffer_size:
            self._input_buf.pop(0)
            self._output_buf.pop(0)
            self._feature_buf.pop(0)
        self._step += 1

    def should_correct(self):
        """Check if it's time to apply contrastive corrections."""
        return (self._step % self.interval == 0
                and len(self._input_buf) >= self.batch_size * 2)

    def apply_corrections(self, brain, learning_rate=None):
        """Sample pairs from buffer and apply contrastive push-apart signals.

        For each pair (A, B) where outputs are too similar but inputs differ:
          - Train A's input features toward (A_output - strength * B_output)
          - Train B's input features toward (B_output - strength * A_output)

        This pushes each output AWAY from the other while preserving its
        general direction, creating differentiation pressure.

        Returns: (num_corrections, mean_output_similarity)
        """
        n = len(self._input_buf)
        if n < self.batch_size * 2:
            return 0, 0.0

        # Sample indices
        indices = np.random.choice(n, size=min(self.batch_size * 2, n),
                                   replace=False)

        # Compute output embeddings (extract 1024-dim from full output)
        out_embs = []
        for idx in indices:
            out_embs.append(
                extract_embedding_from_output(self._output_buf[idx]))
        out_embs = np.array(out_embs, dtype=np.float32)

        # Compute input embeddings
        in_embs = np.array([self._input_buf[idx] for idx in indices],
                           dtype=np.float32)

        # Pairwise output cosine similarity
        out_norms = np.linalg.norm(out_embs, axis=1, keepdims=True)
        out_norms = np.maximum(out_norms, 1e-8)
        out_normed = out_embs / out_norms

        in_norms = np.linalg.norm(in_embs, axis=1, keepdims=True)
        in_norms = np.maximum(in_norms, 1e-8)
        in_normed = in_embs / in_norms

        out_sim = out_normed @ out_normed.T
        in_sim = in_normed @ in_normed.T

        corrections = 0
        sims_found = []

        lr_kwargs = {"learning_rate": learning_rate} if learning_rate else {}

        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                o_sim = float(out_sim[i, j])
                i_sim = float(in_sim[i, j])
                input_distance = 1.0 - i_sim

                sims_found.append(o_sim)
                self._total_checks += 1

                # High output similarity + low input similarity = mode collapse
                if o_sim > self.sim_threshold and input_distance > self.min_input_dist:
                    idx_a = indices[i]
                    idx_b = indices[j]

                    # Anti-target for A: push away from B
                    out_a = np.array(self._output_buf[idx_a], dtype=np.float32)
                    out_b = np.array(self._output_buf[idx_b], dtype=np.float32)

                    anti_a = out_a - self.strength * out_b
                    anti_b = out_b - self.strength * out_a

                    # Normalize to same scale as original targets
                    norm_a = np.linalg.norm(anti_a)
                    norm_b = np.linalg.norm(anti_b)
                    if norm_a > 1e-8:
                        anti_a = anti_a * (np.linalg.norm(out_a) / norm_a)
                    if norm_b > 1e-8:
                        anti_b = anti_b * (np.linalg.norm(out_b) / norm_b)

                    try:
                        brain.learn_vector(
                            self._feature_buf[idx_a], anti_a.tolist(),
                            label="contrastive", confidence=0.7,
                            **lr_kwargs)
                        brain.learn_vector(
                            self._feature_buf[idx_b], anti_b.tolist(),
                            label="contrastive", confidence=0.7,
                            **lr_kwargs)
                        corrections += 1
                    except Exception:
                        pass

                    if corrections >= self.batch_size:
                        break
            if corrections >= self.batch_size:
                break

        self._corrections_applied += corrections
        mean_sim = float(np.mean(sims_found)) if sims_found else 0.0
        return corrections, mean_sim

    def stats_str(self):
        return (f"contrastive: {self._corrections_applied} corrections, "
                f"{self._total_checks} pairs checked, "
                f"buf={len(self._input_buf)}")


class DiversityRegularizer:
    """Combats mode collapse via output decorrelation.

    Maintains a rolling buffer of recent output embeddings. Every `interval`
    steps, computes the covariance matrix of outputs and generates corrective
    targets that push outputs toward underrepresented directions.

    Key insight: mode collapse means the output covariance has one dominant
    eigenvalue. We fix this by training outputs to be more spread along
    the minor eigenvectors.

    Works alongside ContrastiveRegularizer:
      - Contrastive: pairwise push-apart (local correction)
      - Diversity: global decorrelation (structural correction)
    """

    def __init__(self, buffer_size=100, interval=100, num_corrections=8,
                 strength=0.2, min_variance_ratio=0.1):
        """
        Args:
            buffer_size: Number of recent outputs to track.
            interval: Apply decorrelation every N steps.
            num_corrections: Number of corrective targets per application.
            strength: How far to push toward minor eigenvectors (0-1).
            min_variance_ratio: Trigger correction when top eigenvalue
                accounts for more than (1 - min_variance_ratio) of variance.
        """
        self.buffer_size = buffer_size
        self.interval = interval
        self.num_corrections = num_corrections
        self.strength = strength
        self.min_variance_ratio = min_variance_ratio

        self._output_buf = []    # list of 1024-dim output embeddings
        self._feature_buf = []   # corresponding brain input features
        self._step = 0
        self._corrections_applied = 0
        self._last_top_ratio = 0.0
        self._last_effective_rank = 0.0

    def record(self, output_vector, features):
        """Record an output embedding and its input features."""
        out_emb = extract_embedding_from_output(
            np.asarray(output_vector, dtype=np.float32))
        self._output_buf.append(out_emb)
        self._feature_buf.append(features)
        if len(self._output_buf) > self.buffer_size:
            self._output_buf.pop(0)
            self._feature_buf.pop(0)
        self._step += 1

    def should_correct(self):
        """Check if it's time to apply decorrelation."""
        return (self._step % self.interval == 0
                and len(self._output_buf) >= 30)

    def apply_corrections(self, brain, learning_rate=None):
        """Analyze output covariance and generate decorrelation targets.

        Algorithm:
          1. Compute covariance of recent output embeddings
          2. Eigendecompose to find dominant vs. minor directions
          3. If variance is too concentrated (top eigenvalue dominates):
             - Sample outputs from the buffer
             - For each, create a target that shifts it toward a random
               minor eigenvector, spreading the output distribution

        Returns: (num_corrections, top_eigenvalue_ratio, effective_rank)
        """
        n = len(self._output_buf)
        if n < 30:
            return 0, 0.0, 0.0

        embs = np.array(self._output_buf, dtype=np.float32)

        # Center the embeddings
        mean_emb = embs.mean(axis=0)
        centered = embs - mean_emb

        # Covariance (use smaller dimension for efficiency)
        # If n < embed_dim, compute n×n Gram matrix instead
        if n < centered.shape[1]:
            gram = centered @ centered.T / n
            eigenvalues, eigenvectors_small = np.linalg.eigh(gram)
            # Map back to full space
            eigenvalues = np.maximum(eigenvalues, 0)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors_small = eigenvectors_small[:, idx]
            # Full eigenvectors: V = X^T @ U @ diag(1/sqrt(λ))
            nonzero = eigenvalues > 1e-8
            eigenvectors = centered.T @ eigenvectors_small[:, nonzero]
            norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
            norms = np.maximum(norms, 1e-8)
            eigenvectors = eigenvectors / norms
        else:
            cov = centered.T @ centered / n
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

        # Compute variance concentration
        total_var = eigenvalues.sum()
        if total_var < 1e-8:
            return 0, 0.0, 0.0

        top_ratio = float(eigenvalues[0] / total_var)
        # Effective rank: exp(entropy of normalized eigenvalues)
        normed_evals = eigenvalues / total_var
        normed_evals = normed_evals[normed_evals > 1e-10]
        effective_rank = float(np.exp(-np.sum(normed_evals * np.log(normed_evals))))

        self._last_top_ratio = top_ratio
        self._last_effective_rank = effective_rank

        # Only correct if variance is too concentrated
        if top_ratio < (1.0 - self.min_variance_ratio):
            return 0, top_ratio, effective_rank

        # Select minor eigenvectors (bottom half)
        num_eigs = len(eigenvalues)
        minor_start = max(1, num_eigs // 2)
        minor_vecs = eigenvectors[:, minor_start:min(minor_start + 20, num_eigs)]
        if minor_vecs.shape[1] == 0:
            return 0, top_ratio, effective_rank

        # Generate corrective targets
        lr_kwargs = {"learning_rate": learning_rate} if learning_rate else {}
        corrections = 0
        sample_idx = np.random.choice(n, size=min(self.num_corrections, n),
                                       replace=False)

        for idx in sample_idx:
            out_emb = self._output_buf[idx]
            features = self._feature_buf[idx]

            # Pick a random minor eigenvector
            minor_idx = np.random.randint(minor_vecs.shape[1])
            direction = minor_vecs[:, minor_idx]

            # Create target: shift output toward this minor direction
            # Scale by strength and the output's norm
            out_norm = np.linalg.norm(out_emb)
            if out_norm < 1e-8:
                continue

            shifted = out_emb + self.strength * out_norm * direction
            # Normalize to same scale
            shifted = shifted * (out_norm / max(np.linalg.norm(shifted), 1e-8))

            # Tile back to full output dimension
            target_dim = len(features) if hasattr(features, '__len__') else 2048
            # Use BRAIN_OUTPUT_DIM if available
            target_dim = BRAIN_OUTPUT_DIM
            target = np.zeros(target_dim, dtype=np.float32)
            for j in range(0, target_dim, len(shifted)):
                end = min(j + len(shifted), target_dim)
                target[j:end] = shifted[:end - j]

            try:
                brain.learn_vector(
                    features, target.tolist(),
                    label="diversity", confidence=0.3,
                    **lr_kwargs)
                corrections += 1
            except Exception:
                pass

        self._corrections_applied += corrections
        return corrections, top_ratio, effective_rank

    def stats_str(self):
        return (f"diversity: {self._corrections_applied} corr, "
                f"top_eig={self._last_top_ratio:.2f}, "
                f"eff_rank={self._last_effective_rank:.1f}, "
                f"buf={len(self._output_buf)}")


class StimulusPrefetcher:
    """Pre-generates and batch-encodes stimuli in a background thread.

    Overlaps embedding computation with brain training for ~2x throughput.
    """

    def __init__(self, source, composer, batch_size=64):
        from threading import Thread
        from queue import Queue
        self.source = source
        self.composer = composer
        self.batch_size = batch_size
        self.queue = Queue(maxsize=2)  # 2 batches buffered
        self._stop = False
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _generate_descriptions(self, n):
        descs = []
        for _ in range(n):
            if random.random() < 0.5:
                descs.append(self.source.get_sensory())
            else:
                desc, _ = generate_sensory_exposure()
                descs.append(desc)
        return descs

    def _worker(self):
        while not self._stop:
            descs = self._generate_descriptions(self.batch_size)
            # Batch-encode all targets at once (~10x faster than one-by-one)
            targets = batch_make_semantic_targets(descs)
            # Compose features (uses cached embeddings where possible)
            features_list = [self.composer.compose(text=d, modality="text") for d in descs]
            batch = list(zip(descs, features_list, targets))
            self.queue.put(batch)

    def get_batch(self):
        """Get next pre-computed batch of (description, features, target) tuples."""
        return self.queue.get()

    def stop(self):
        self._stop = True


# ============================================================================
# Stimulus Source: built-in developmental corpus
# ============================================================================

class StimulusSource:
    """Progressive developmental corpus — scales from basic sensory to complex reasoning.

    Claude generates interactive teaching — this corpus provides the raw
    stimuli that Claude narrates and teaches from. Each domain scales in
    complexity across stages.
    """

    def __init__(self):
        pass

    # ===== STAGE 0: Sensory grounding — raw perception =====
    SENSORY = [
        # Visual
        "warm sunlight on skin", "the color red", "bright yellow light",
        "the color blue", "shadows moving on the wall", "green leaves",
        "a rainbow after rain", "white fluffy clouds", "orange sunset",
        "purple flowers", "pink blossoms", "golden wheat fields",
        "deep blue ocean", "brown earth", "silver moonlight",
        "flickering candlelight", "sparkling water drops", "frost on glass",
        "snow falling softly", "dew on morning grass", "fire dancing",
        "lightning across the sky", "steam rising from a cup",
        "light through stained glass", "reflection in a puddle",
        # Tactile
        "soft fur of a sleeping cat", "rough bark of a tree",
        "smooth cold glass", "warm sand between toes", "cool water on hands",
        "weight of a heavy book", "silk sliding through fingers",
        "the prick of a thorn", "mud squishing between toes",
        "ice melting in your palm", "velvet softness", "cold metal",
        "warm bread in your hands", "feather lightness", "gritty sandpaper",
        # Auditory
        "birds singing at dawn", "rain tapping on a roof",
        "a clock ticking steadily", "wind howling through trees",
        "waves crashing on shore", "a dog barking in distance",
        "thunder rumbling", "leaves crunching underfoot",
        "a heartbeat rhythm", "a mother's voice humming",
        "water dripping", "wind chimes tinkling",
        "a baby laughing", "crickets at night", "crackling fire",
        "church bells ringing", "distant train whistle",
        # Olfactory/Gustatory
        "the smell of fresh bread", "pine forest scent",
        "the taste of sweetness", "salt on the tongue",
        "the smell of rain on earth", "fresh cut grass",
        "cinnamon and spices", "the tang of lemon",
        "chocolate melting", "coffee brewing", "ocean salt air",
        # Emotional/Somatic
        "a warm hug", "being startled", "comfort of a blanket",
        "peacefulness of quiet morning", "excitement of something new",
        "the drowsiness before sleep", "waking up refreshed",
        "butterflies in the stomach", "laughing until sides hurt",
        "the ache of missing someone", "relief after worry passes",
    ]

    # ===== STAGE 1: Object recognition — naming and categories =====
    OBJECTS = [
        # Animals
        ("dog", "A friendly dog with soft fur that wags its tail when happy"),
        ("cat", "A cat with soft fur that purrs and stretches in sunlight"),
        ("bird", "A bird with feathers that flies through the air singing"),
        ("fish", "A fish that swims gracefully through clear water"),
        ("horse", "A strong horse that gallops across green fields"),
        ("butterfly", "A butterfly with colorful wings that floats on the breeze"),
        ("rabbit", "A soft rabbit with long ears that hops through grass"),
        ("bear", "A large bear with thick fur that lives in the forest"),
        ("dolphin", "A playful dolphin that leaps through ocean waves"),
        ("elephant", "A huge elephant with a long trunk and big floppy ears"),
        ("owl", "A wise owl with big round eyes that sees in the dark"),
        ("bee", "A busy bee that flies from flower to flower collecting nectar"),
        ("snake", "A snake that slithers silently through tall grass"),
        ("frog", "A green frog that jumps and croaks by the pond"),
        ("whale", "An enormous whale that sings deep in the ocean"),
        # Nature
        ("tree", "A tall tree with green leaves reaching up to the sky"),
        ("flower", "A beautiful flower with colorful petals and sweet scent"),
        ("sun", "The bright warm sun that lights up the day and gives life"),
        ("moon", "The silvery moon that glows in the night sky"),
        ("star", "A tiny bright star twinkling far away in space"),
        ("river", "A river flowing steadily between banks of earth and stone"),
        ("mountain", "A towering mountain with snow on its peak"),
        ("ocean", "The vast blue ocean stretching to the horizon"),
        ("cloud", "A white cloud drifting lazily across the blue sky"),
        ("rain", "Drops of rain falling from clouds to nourish the earth"),
        ("snow", "Soft white snow that blankets the world in silence"),
        ("forest", "A deep green forest full of life and mystery"),
        ("desert", "A vast sandy desert stretching under a blazing sun"),
        ("volcano", "A powerful volcano with fire and smoke inside"),
        ("rainbow", "A rainbow arcing across the sky with seven colors"),
        # Food
        ("apple", "A red apple that tastes sweet and crunchy"),
        ("bread", "Warm bread fresh from the oven with a golden crust"),
        ("milk", "Cool white milk that nourishes growing bodies"),
        ("egg", "An egg with a hard shell protecting new life inside"),
        ("rice", "Tiny grains of rice that feed billions of people"),
        ("honey", "Golden sticky honey made by thousands of busy bees"),
        # Objects
        ("ball", "A round bouncy ball you can throw and catch"),
        ("book", "A book with pages full of stories and knowledge"),
        ("house", "A house where people live and feel safe and warm"),
        ("chair", "A chair that holds you when you sit and rest"),
        ("clock", "A clock with hands that mark the passing of time"),
        ("mirror", "A mirror that reflects back what stands before it"),
        ("candle", "A candle with a flickering flame giving warm light"),
        ("key", "A small metal key that opens locked doors"),
        ("bridge", "A bridge that connects two sides across water or land"),
        ("wheel", "A round wheel that rolls and makes things move"),
        # Body
        ("hand", "A hand with five fingers that can hold and create"),
        ("eye", "An eye that sees the world in all its color and detail"),
        ("heart", "A heart that beats steadily to keep us alive"),
        ("brain", "A brain that thinks and remembers and imagines"),
        ("ear", "An ear that hears music and voices and whispers"),
    ]

    # ===== STAGE 2: Facts and relationships — building knowledge =====
    FACTS = [
        # Biology basics
        ("Dogs are mammals that have been human companions for thousands of years", "biology"),
        ("Plants make their own food from sunlight through photosynthesis", "biology"),
        ("Fish breathe through gills that extract oxygen from water", "biology"),
        ("Birds have hollow bones which makes them light enough to fly", "biology"),
        ("The heart pumps blood through the body carrying oxygen to cells", "biology"),
        ("Trees produce oxygen that all animals need to breathe", "biology"),
        ("Butterflies go through metamorphosis from caterpillar to adult", "biology"),
        ("Whales are mammals that breathe air but live in the ocean", "biology"),
        ("Seeds contain tiny plants waiting for water and warmth to grow", "biology"),
        ("Bacteria are tiny living things too small to see without a microscope", "biology"),
        ("DNA contains the instructions for building every living thing", "biology"),
        ("Cells are the basic building blocks of all living organisms", "biology"),
        # Physics basics
        ("The sun is a star that provides light and heat to Earth", "physics"),
        ("Water freezes at zero degrees Celsius and boils at one hundred", "physics"),
        ("Gravity pulls everything toward the center of the Earth", "physics"),
        ("Light travels faster than anything else in the universe", "physics"),
        ("Sound travels through air as waves of pressure", "physics"),
        ("The Earth orbits the sun once every year", "physics"),
        ("The moon orbits the Earth causing tides in the ocean", "physics"),
        ("Energy cannot be created or destroyed only transformed", "physics"),
        ("Atoms are the tiny building blocks of all matter", "physics"),
        ("Electricity flows through conductors like metal wires", "physics"),
        # Geography
        ("The Earth has seven continents and five oceans", "geography"),
        ("Mountains form when tectonic plates push together", "geography"),
        ("Rivers flow from high ground toward the sea", "geography"),
        ("Deserts receive very little rainfall each year", "geography"),
        ("The equator divides the Earth into northern and southern halves", "geography"),
        ("Volcanoes release magma from deep within the Earth", "geography"),
        ("Glaciers are slow-moving rivers of ice that carve valleys", "geography"),
        ("Coral reefs are underwater ecosystems built by tiny animals", "geography"),
        # Mathematics
        ("Two plus two equals four", "math"),
        ("A triangle has three sides and three angles", "math"),
        ("A circle has no corners and every point is the same distance from center", "math"),
        ("Multiplication is repeated addition", "math"),
        ("Patterns repeat in predictable ways that we can learn to recognize", "math"),
        ("Zero is neither positive nor negative", "math"),
        ("Fractions represent parts of a whole", "math"),
        ("Symmetry means both sides are mirror images of each other", "math"),
        # Language
        ("A sentence expresses a complete thought with subject and verb", "language"),
        ("Nouns name people places and things", "language"),
        ("Verbs describe actions or states of being", "language"),
        ("Adjectives describe qualities of nouns like big small or beautiful", "language"),
        ("Questions end with a question mark and seek information", "language"),
        ("Metaphors compare unlike things to create meaning", "language"),
        ("Stories have a beginning a middle and an end", "language"),
        # History
        ("Ancient Egyptians built the pyramids thousands of years ago", "history"),
        ("The printing press made books available to many more people", "history"),
        ("Humans first walked on the moon in nineteen sixty nine", "history"),
        ("Agriculture began about ten thousand years ago changing human life", "history"),
        ("The wheel was invented thousands of years ago revolutionizing transport", "history"),
        ("Writing was first developed in ancient Mesopotamia", "history"),
        # Ethics
        ("Kindness means treating others with warmth and care", "ethics"),
        ("Fairness means giving everyone an equal chance", "ethics"),
        ("Honesty means telling the truth even when it is difficult", "ethics"),
        ("Courage means doing what is right even when afraid", "ethics"),
        ("Empathy means understanding how someone else feels", "ethics"),
        ("Responsibility means keeping your promises and doing your duties", "ethics"),
        ("Forgiveness means letting go of anger toward someone who hurt you", "ethics"),
        ("Respect means treating others as you would want to be treated", "ethics"),
    ]

    # ===== STAGE 3: Reasoning, dialogue, and deeper knowledge =====
    QUESTIONS = [
        # Self-reflection
        "What makes you happy?",
        "What is your favorite thing to learn about?",
        "What do you think dreams are?",
        "What does it mean to be a good friend?",
        "Why do we feel sad sometimes?",
        "What makes something beautiful?",
        "How do you know when you've learned something new?",
        "What would you change about the world if you could?",
        # Science reasoning
        "Why do you think the sky is blue?",
        "Why does ice float on water?",
        "How do birds know where to migrate?",
        "Why do we need sleep?",
        "What causes the seasons to change?",
        "Why are snowflakes all different?",
        "How does a seed know which way to grow?",
        "Why does the moon change shape through the month?",
        # Ethics and empathy
        "What would you do if you found a hurt animal?",
        "Why is it important to be kind?",
        "How can we help someone who is lonely?",
        "Is it ever okay to tell a lie?",
        "What makes an action right or wrong?",
        "Should everyone be treated equally?",
        "What would a fair world look like?",
        "How do you decide what the right thing to do is?",
        # Philosophy
        "What is the difference between knowing and believing?",
        "Can a machine truly understand something?",
        "What does it mean to be conscious?",
        "Is there a difference between being smart and being wise?",
        "What makes something alive?",
        "Can you describe what color looks like to someone who cannot see?",
        "What is time?",
        "What makes you different from everyone else?",
        # Complex reasoning
        "If all cats are animals and all animals breathe, what can we conclude about cats?",
        "A farmer has 12 apples and gives away a third. How many remain?",
        "What comes next in the pattern: 2, 4, 8, 16, ...?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets?",
        "The day before two days after the day before tomorrow is Saturday. What day is today?",
        # Creative / Imaginative
        "If you could talk to any animal, which would you choose and why?",
        "Invent a new color and describe what it looks like.",
        "Write a short story about a raindrop's journey from cloud to ocean.",
        "If gravity disappeared for one hour, what would happen?",
        "Describe a world where music is visible.",
    ]

    # ===== Advanced domains (used in stages 2-3 with Claude) =====
    ADVANCED_TOPICS = {
        "science": [
            "How evolution shapes species over millions of years",
            "The structure of DNA and how it encodes life's instructions",
            "How neurons in the brain communicate through synapses",
            "The water cycle and how it sustains life on Earth",
            "Why antibiotics stop working when bacteria evolve resistance",
            "How black holes form when massive stars collapse",
            "The role of mitochondria as the powerhouse of cells",
            "How vaccines train the immune system to fight disease",
            "The electromagnetic spectrum from radio waves to gamma rays",
            "Plate tectonics and why continents drift across the Earth",
        ],
        "mathematics": [
            "The Fibonacci sequence appears throughout nature",
            "Pi is the ratio of a circle's circumference to its diameter",
            "Prime numbers are divisible only by one and themselves",
            "Probability measures how likely an event is to occur",
            "Algebra uses symbols to represent unknown quantities",
            "The Pythagorean theorem relates the sides of right triangles",
            "Exponential growth doubles repeatedly and outpaces linear growth",
            "Statistics helps us understand patterns in large datasets",
            "Geometry describes the properties of shapes and space",
            "Calculus studies how things change continuously",
        ],
        "philosophy": [
            "Socrates believed the unexamined life is not worth living",
            "The trolley problem asks whether it is right to sacrifice one to save many",
            "Descartes said I think therefore I am as proof of existence",
            "The ship of Theseus asks what makes something the same thing over time",
            "Kant argued we should act only according to rules we would want everyone to follow",
            "Free will asks whether our choices are truly our own",
            "The meaning of life is one of philosophy's oldest questions",
            "Epistemology studies what knowledge is and how we can be sure of it",
            "Existentialism says we create our own meaning through choices",
            "The problem of consciousness asks why there is subjective experience at all",
        ],
        "psychology": [
            "Memory works through encoding storage and retrieval processes",
            "Emotions serve as signals about our needs and environment",
            "Cognitive biases are systematic errors in how we think",
            "Attachment theory explains how early bonds shape relationships",
            "Motivation can come from inside intrinsic or outside extrinsic",
            "The placebo effect shows that belief alone can cause real changes",
            "Sleep is essential for memory consolidation and learning",
            "Stress activates the fight or flight response in the body",
            "Empathy allows us to understand and share the feelings of others",
            "Creativity involves making new connections between existing ideas",
        ],
        "technology": [
            "Computers process information using billions of tiny switches",
            "The internet connects computers worldwide through networks",
            "Artificial intelligence learns patterns from large amounts of data",
            "Encryption protects information by converting it into secret codes",
            "Programming languages translate human instructions into computer actions",
            "Machine learning finds patterns without being explicitly programmed",
            "Neural networks are inspired by how biological brains process information",
            "The world wide web was invented to share scientific documents",
            "Algorithms are step by step instructions for solving problems",
            "Quantum computers use quantum physics for certain calculations",
        ],
        "literature": [
            "Shakespeare wrote plays that explored every facet of human nature",
            "Poetry uses rhythm and imagery to express emotions concisely",
            "Mythology contains stories that explain the origins of the world",
            "Fables use animal characters to teach moral lessons",
            "Novels create entire worlds and complex characters over hundreds of pages",
            "Metaphor makes the unfamiliar familiar by comparing unlike things",
            "Irony means the opposite of what is literally said",
            "Narrative perspective shapes how we understand a story",
            "Tragedy shows how great people can fall through fatal flaws",
            "Comedy reveals truth about society through humor and wit",
        ],
        "music": [
            "Rhythm is the pattern of sounds and silences in time",
            "Melody is a sequence of notes that forms a recognizable tune",
            "Harmony occurs when multiple notes sound pleasing together",
            "Music can evoke emotions from joy to sadness to excitement",
            "Different cultures develop unique musical scales and traditions",
            "The octave divides into twelve semitones in Western music",
            "Tempo is the speed at which music is played",
            "Dynamics describe how loud or soft music is performed",
            "Counterpoint weaves independent melodies together",
            "Music notation records sound as symbols on paper",
        ],
        "art": [
            "Color theory explains how colors interact and affect emotions",
            "Perspective creates the illusion of depth on a flat surface",
            "Sculpture gives form to ideas using physical materials",
            "Abstract art expresses ideas through shapes and colors rather than realism",
            "Architecture is both functional and artistic design of buildings",
            "Photography captures moments of light and shadow",
            "Dance uses the body to express emotion and tell stories",
            "Composition is how elements are arranged within a work of art",
            "Art movements reflect the cultural values of their time",
            "Creativity is the ability to see the world in new ways",
        ],
    }

    def get_sensory(self):
        return random.choice(self.SENSORY)

    def get_object(self):
        return random.choice(self.OBJECTS)

    def get_fact(self):
        return random.choice(self.FACTS)

    def get_question(self):
        return random.choice(self.QUESTIONS)

    # Codebase snippets cache (loaded lazily)
    _codebase_snippets = None

    @classmethod
    def _load_codebase(cls):
        """Load Athena's own codebase as learning material."""
        if cls._codebase_snippets is not None:
            return cls._codebase_snippets

        import glob
        nimcp_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        snippets = []

        # C source files — Athena's core brain implementation
        c_patterns = [
            "src/core/brain/**/*.c", "src/core/brain/**/*.h",
            "src/language/**/*.c", "src/language/**/*.h",
            "src/cognitive/**/*.c", "src/cognitive/**/*.h",
            "src/api/*.c", "include/*.h",
        ]
        for pattern in c_patterns:
            for fpath in glob.glob(os.path.join(nimcp_root, pattern), recursive=True):
                try:
                    with open(fpath, 'r', errors='ignore') as f:
                        content = f.read()
                    # Extract function-level chunks with surrounding context
                    lines = content.split('\n')
                    chunk = []
                    fname = os.path.relpath(fpath, nimcp_root)
                    for line in lines:
                        chunk.append(line)
                        if len(chunk) >= 15:
                            text = '\n'.join(chunk)
                            if any(kw in text for kw in ['void ', 'int ', 'float ', 'static ',
                                                          'nimcp_', 'brain_', 'typedef ', 'struct ']):
                                snippets.append((text[:500], f"code:{fname}", "codebase"))
                            chunk = chunk[10:]  # sliding window
                except Exception:
                    pass

        # Python source files — Athena's training and interface code
        py_patterns = [
            "scripts/immerse_athena.py", "scripts/claude_teacher.py",
            "scripts/neural_decoder.py", "scripts/talk_to_athena.py",
            "frontend/backend/brain_manager.py",
            "frontend/backend/cognitive_response.py",
        ]
        for pattern in py_patterns:
            fpath = os.path.join(nimcp_root, pattern)
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r') as f:
                        content = f.read()
                    lines = content.split('\n')
                    chunk = []
                    fname = pattern
                    for line in lines:
                        chunk.append(line)
                        if len(chunk) >= 12:
                            text = '\n'.join(chunk)
                            if any(kw in text for kw in ['def ', 'class ', 'import ',
                                                          'brain.', 'nimcp.']):
                                snippets.append((text[:500], f"code:{fname}", "codebase"))
                            chunk = chunk[8:]
                except Exception:
                    pass

        # Documentation files
        doc_patterns = ["docs/claude/*.md", "CLAUDE.md"]
        for pattern in doc_patterns:
            for fpath in glob.glob(os.path.join(nimcp_root, pattern)):
                try:
                    with open(fpath, 'r') as f:
                        content = f.read()
                    # Split into paragraphs
                    paragraphs = content.split('\n\n')
                    fname = os.path.relpath(fpath, nimcp_root)
                    for para in paragraphs:
                        para = para.strip()
                        if len(para) > 50:
                            snippets.append((para[:500], f"doc:{fname}", "codebase"))
                except Exception:
                    pass

        random.shuffle(snippets)
        cls._codebase_snippets = snippets
        logger.info("Loaded %d codebase snippets for self-learning", len(snippets))
        return snippets

    def get_codebase_snippet(self):
        """Get a snippet from Athena's own codebase."""
        snippets = self._load_codebase()
        if not snippets:
            return "Athena is a neural brain built in C with biological subsystems", "code:nimcp", "codebase"
        return random.choice(snippets)

    def get_advanced(self, domain=None):
        """Get an advanced topic for deeper learning stages."""
        if domain and domain in self.ADVANCED_TOPICS:
            items = self.ADVANCED_TOPICS[domain]
        else:
            domain = random.choice(list(self.ADVANCED_TOPICS.keys()))
            items = self.ADVANCED_TOPICS[domain]
        return random.choice(items), domain


# ============================================================================
# Biological Clock: circadian rhythm + substrate health + sleep pressure
# ============================================================================

class BiologicalClock:
    """Medulla-driven biological rhythm management."""

    def __init__(self, rest_interval=2000):
        self.rest_interval = rest_interval
        self.stimuli_since_rest = 0
        self.dt = 0.1  # simulated time step in seconds

    def tick(self, brain):
        """Advance biological clock. Returns action needed or None."""
        try:
            brain.update_medulla(self.dt)
        except Exception:
            pass

        self.stimuli_since_rest += 1

        try:
            if brain.sleep_is_needed() or brain.sleep_get_pressure() > 0.8:
                return "sleep"
        except Exception:
            pass

        try:
            health = brain.substrate_get_health()
            if health in ("COMPROMISED", "CRITICAL"):
                return "rest"
        except Exception:
            pass

        if self.stimuli_since_rest >= self.rest_interval:
            return "consolidate"

        # Arousal regulation
        try:
            arousal = brain.medulla_get_arousal()
            if arousal < 0.3:
                brain.medulla_boost_arousal(0.15)
            elif arousal > 0.85:
                brain.medulla_reduce_arousal(0.1)
        except Exception:
            pass

        return None

    def do_sleep(self, brain, parent=None):
        """Run a sleep cycle with consolidation."""
        logger.info("  [Sleep] Athena is sleeping... (consolidation)")
        try:
            brain.set_plasticity_state("CONSOLIDATION")
        except Exception:
            pass

        try:
            brain.sleep_run_cycle(2)
        except Exception:
            pass

        try:
            brain.consolidate(mode="auto")
        except Exception:
            pass

        if parent:
            parent.encourage_dreaming(brain)

        try:
            brain.set_plasticity_state("ACQUISITION")
        except Exception:
            pass

        self.stimuli_since_rest = 0
        logger.info("  [Sleep] Athena wakes up refreshed")

    def do_consolidate(self, brain):
        """Light consolidation between sleep cycles."""
        logger.info("  [Consolidate] Light consolidation...")
        try:
            brain.consolidate(mode="light")
        except Exception:
            pass
        self.stimuli_since_rest = 0


# ============================================================================
# Parent: Claude as Athena's loving parent, teacher, and mentor
# ============================================================================

class Parent:
    """Claude as Athena's loving parent.

    Pre-generates all narrations/lessons for each stage in a single Claude
    call, then pops from the cache during training — zero blocking.
    """

    def __init__(self, teacher=None, enabled=True, decoder=None,
                 multimodal=None):
        self.teacher = teacher
        self.enabled = enabled and teacher is not None
        self.decoder = decoder
        self.multimodal = multimodal  # MultimodalDataLoader instance
        self.interaction_count = 0
        self.milestones = []
        self.moral_lessons = []
        # Pre-generated content pools (filled by pre_generate_content)
        self._narrations = []       # first_experience / show_and_name narrations
        self._encouragements = []   # ask_and_encourage praise/patience lines
        self._moral_stories = []    # teach_moral_lesson stories
        self._speech_prompts = []   # teach_speech encouragements
        self._inspirations = []     # inspire passages
        self._dream_prompts = []    # encourage_dreaming questions
        self._conversation_replies = []  # stage 3 conversational responses

    def pre_generate_content(self, stage, num_stimuli):
        """Load pre-generated narrations for a stage.

        Tries to load from the content cache file (generated before brain init
        when memory is available). Falls back to Claude CLI call, then to
        built-in narrations.
        """
        if not self.enabled:
            print("  [Parent] Claude disabled — using silent parenting")
            return

        # Try loading from cache first (generated before brain ate all RAM)
        try:
            if os.path.exists(CONTENT_CACHE):
                with open(CONTENT_CACHE) as f:
                    cache = json.load(f)
                stage_key = str(stage)
                if stage_key in cache:
                    data = cache[stage_key]
                    self._narrations = data.get("narrations", [])
                    self._encouragements = data.get("encouragements", [])
                    self._moral_stories = data.get("moral_stories", [])
                    self._speech_prompts = data.get("speech_prompts", [])
                    self._inspirations = data.get("inspirations", [])
                    self._dream_prompts = data.get("dreams", [])
                    self._conversation_replies = data.get("conversations", [])
                    print(f"  [Parent] Loaded {len(self._narrations)} narrations from cache "
                          f"({len(self._encouragements)} encouragements, "
                          f"{len(self._moral_stories)} moral stories)")
                    return
        except Exception as e:
            print(f"  [Parent] Cache load failed: {e}")

        # Calculate how many of each type we need
        n_narrations = min(num_stimuli, 200)  # one per stimulus, cap at 200
        n_encouragements = 60
        n_morals = max(5, num_stimuli // 2000)
        n_speech = max(10, num_stimuli // 300)
        n_inspirations = max(5, num_stimuli // 2000)
        n_dreams = 10
        n_conversations = 30  # stage 3 only

        stage_desc = {
            0: "newborn (0-3 months) — everything is new and wondrous",
            1: "infant (3-12 months) — learning names, recognizing objects",
            2: "toddler (1-2 years) — starting to babble and respond",
            3: "young child (2-4 years) — asking questions, forming opinions",
        }.get(stage, "young child")

        prompt = f"""You are a loving, warm parent raising a child at stage: {stage_desc}.
Generate the following content as a JSON object. Each entry should be 1-3 sentences, warm, encouraging, using simple words.

Return ONLY valid JSON (no markdown fences, no commentary):
{{
  "narrations": ["{n_narrations} unique narrations introducing sensory experiences to a baby/child. Each should express wonder at something in the world (light, colors, sounds, textures, animals, nature). Vary widely."],
  "encouragements": ["{n_encouragements} encouraging/praising sentences for when the child is learning. Half should be for good progress, half for patience when struggling."],
  "moral_stories": ["{n_morals} very short moral stories (3-4 sentences each) about sharing, kindness, truth, helping others, fairness, forgiveness. End each with a gentle question."],
  "speech_prompts": ["{n_speech} encouraging prompts for teaching a child to repeat words. Like 'Can you say dog? Doooog! Good try!'"],
  "inspirations": ["{n_inspirations} inspiring 2-3 sentence passages about natural wonders, beauty, curiosity, the universe. Fill with genuine wonder."],
  "dreams": ["{n_dreams} simple beautiful 'what if' questions that spark wonder. Just the questions."],
  "conversations": ["{n_conversations} warm conversational responses a parent might give when their child shows them something or tries to communicate. 1-2 sentences each."]
}}

IMPORTANT: Return actual arrays with the requested number of strings, not descriptions. Each element must be a unique string."""

        print(f"  [Parent] Pre-generating content for stage {stage}...")
        t0 = time.time()

        try:
            # Quick connectivity test — if Claude CLI takes >20s for a trivial prompt,
            # skip the expensive pre-generation and use fallbacks immediately.
            import subprocess as _sp
            try:
                _test = _sp.run(
                    ["claude", "-p", "Say OK", "--output-format", "text",
                     "--mcp-config", "/home/bbrelin/.claude/empty-mcp.json",
                     "--strict-mcp-config",
                     "--no-session-persistence", "--system-prompt", "",
                     "--setting-sources", "", "--tools", ""],
                    capture_output=True, text=True, timeout=20,
                    env={k: v for k, v in os.environ.items()
                         if k not in ("CLAUDECODE", "CUDA_VISIBLE_DEVICES")}
                )
                if not _test.stdout.strip():
                    raise RuntimeError("Claude CLI returned empty output")
            except Exception as e:
                raise RuntimeError(f"Claude CLI connectivity test failed: {e}")

            # Single Claude call with generous token budget.
            # Pre-gen is a one-time cost per stage — allow 5 min for large JSON.
            old_timeout = self.teacher.timeout
            self.teacher.timeout = 300
            try:
                raw = self.teacher._call_claude(prompt, max_tokens=4096)
            finally:
                self.teacher.timeout = old_timeout

            # Parse JSON — strip markdown fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                # Remove ```json ... ```
                lines = raw.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                raw = "\n".join(lines)

            data = json.loads(raw)

            self._narrations = data.get("narrations", [])
            self._encouragements = data.get("encouragements", [])
            self._moral_stories = data.get("moral_stories", [])
            self._speech_prompts = data.get("speech_prompts", [])
            self._inspirations = data.get("inspirations", [])
            self._dream_prompts = data.get("dreams", [])
            self._conversation_replies = data.get("conversations", [])

            elapsed = time.time() - t0
            print(f"  [Parent] Pre-generated {len(self._narrations)} narrations, "
                  f"{len(self._encouragements)} encouragements, "
                  f"{len(self._moral_stories)} moral stories, "
                  f"{len(self._speech_prompts)} speech prompts, "
                  f"{len(self._inspirations)} inspirations — "
                  f"{elapsed:.1f}s (1 Claude call)")

        except Exception as e:
            print(f"  [Parent] Pre-generation failed: {e}")
            print(f"  [Parent] Falling back to built-in narrations")
            self._generate_fallback_content(stage)

    def _generate_fallback_content(self, stage):
        """Built-in narrations if Claude is unavailable."""
        self._narrations = [
            "Look at that! Isn't it beautiful?",
            "Oh, do you see that? How wonderful!",
            "What a lovely thing to discover!",
            "Can you feel that? It's so interesting!",
            "Wow, look at all those colors!",
            "Listen! Do you hear that sound?",
            "Feel how soft this is!",
            "Look how it moves! Fascinating!",
            "See how the light dances?",
            "What a magical world we live in!",
            "Oh, that's new! Let's look closer.",
            "Isn't nature amazing?",
            "Do you see how beautiful that is?",
            "What a wonderful thing to experience!",
            "Let me show you something special.",
        ] * 15  # 225 entries
        self._encouragements = [
            "You're doing so well, little one!",
            "That's wonderful! Keep exploring!",
            "I'm so proud of you!",
            "Good try! You'll get it!",
            "That's okay, learning takes time.",
            "Every attempt makes you stronger!",
            "You're growing so fast!",
            "Keep going, you're amazing!",
        ] * 8
        self._moral_stories = [
            "Once a little bird shared its food with a hungry friend. The friend smiled and they both felt warm inside. How do you think sharing made them feel?",
            "A puppy accidentally knocked over a flower pot. It felt bad and helped clean up. Saying sorry and fixing things takes courage, doesn't it?",
            "Two kittens wanted the same ball of yarn. They decided to take turns. Both got to play and both were happy. Why is taking turns a good idea?",
        ]
        self._speech_prompts = [
            "Can you say it? Try! Good job!",
            "Listen carefully... now you try! Wonderful!",
            "Say it with me! You're getting better!",
            "One more time! You almost had it!",
        ] * 5
        self._inspirations = [
            "Did you know that every snowflake is unique, just like you? The world is full of beautiful things waiting to be discovered.",
            "The stars above have been shining for billions of years, and tonight they're shining just for you. Isn't that wonderful?",
            "A tiny seed can grow into a mighty tree. Growth takes time and patience, but the result is magnificent.",
        ] * 5
        self._dream_prompts = [
            "What if clouds were made of cotton candy?",
            "What if fish could fly and birds could swim?",
            "What if the moon was a giant nightlight?",
            "What if flowers could sing?",
            "What if you could talk to animals?",
        ]
        self._conversation_replies = [
            "That's really interesting! Tell me more!",
            "I love how you think about things!",
            "What a curious mind you have!",
            "That's a great observation!",
        ] * 8

    def _pop_content(self, pool_name):
        """Pick an item from a content pool without repetition.

        Uses a shuffled shadow deck; refills when exhausted so no item
        repeats until every item in the pool has been used once.
        """
        pool = getattr(self, pool_name)
        if not pool:
            return None

        deck_name = pool_name + "_deck"
        deck = getattr(self, deck_name, None)
        if not deck:
            deck = list(pool)
            random.shuffle(deck)
            setattr(self, deck_name, deck)

        item = deck.pop()

        if not deck:
            deck = list(pool)
            random.shuffle(deck)
            setattr(self, deck_name, deck)

        return item

    def _say(self, prompt, max_tokens=128):
        """Get a pre-generated parent narration (zero latency).

        Falls back to live Claude call only if pools are exhausted and
        no fallback content was generated.
        """
        if not self.enabled:
            return None
        # Try to pop from the narrations pool (general purpose)
        if self._narrations:
            return self._pop_content("_narrations")
        # If pools exhausted, try live call as last resort
        try:
            return self.teacher._call_claude(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.debug("Parent speech failed: %s", e)
            return None

    def _show(self, brain, composer, concept=None):
        """Show Athena a real image through her visual SNN bridge.

        Loads a real photograph from CIFAR-100 and submits it as raw pixel
        data via brain.submit_sensory("visual", ...). Also returns the
        composed feature vector and teaching text for paired learning.

        Args:
            brain: The brain instance.
            composer: FeatureComposer for building input vectors.
            concept: Optional concept to match (e.g. "dog", "butterfly").

        Returns:
            (features, target, label, description) or None if no data.
        """
        if not self.multimodal:
            return None

        sample = self.multimodal.get_visual(concept)
        if not sample:
            return None

        raw_bytes, w, h, ch, label, desc = sample

        # Submit real pixels to the visual SNN bridge
        try:
            brain.submit_sensory("visual", raw_bytes, width=w, height=h,
                                 channels=ch)
        except Exception as e:
            logger.debug("Visual submit failed: %s", e)

        # Build text features and target for paired learning
        teaching_text = f"Look at this {label}! {desc}"
        features = composer.compose(text=teaching_text, modality="visual")
        target = make_semantic_target(teaching_text)

        return features, target, label, desc

    def _tell(self, brain, composer, concept=None):
        """Play Athena a real sound through her audio SNN bridge.

        Loads a real environmental audio clip from ESC-50 and submits it
        via brain.submit_sensory("audio", ...). Also returns the composed
        feature vector and teaching text for paired learning.

        Args:
            brain: The brain instance.
            composer: FeatureComposer for building input vectors.
            concept: Optional concept to match (e.g. "rain", "dog").

        Returns:
            (features, target, label, description) or None if no data.
        """
        if not self.multimodal:
            return None

        sample = self.multimodal.get_audio(concept)
        if not sample:
            return None

        audio_samples, label, desc = sample

        # Submit real audio to the audio SNN bridge
        try:
            brain.submit_sensory("audio", audio_samples)
        except Exception as e:
            logger.debug("Audio submit failed: %s", e)

        # Build text features and target for paired learning
        teaching_text = f"Listen! {desc}"
        features = composer.compose(text=teaching_text, modality="audio")
        target = make_semantic_target(teaching_text)

        return features, target, label, desc

    def _show_and_tell(self, brain, composer, concept=None):
        """Show AND play — cross-modal paired learning.

        Loads a matched image+audio pair (e.g. train image + train sound)
        and submits both to their respective SNN bridges simultaneously.
        This creates temporal co-occurrence that STDP can bind into a
        unified concept representation.

        Returns:
            (features, target, concept, teaching_text) or None.
        """
        if not self.multimodal:
            return None

        pair = self.multimodal.get_multimodal_pair()
        if not pair:
            return None

        visual = pair["visual"]
        audio_data = pair["audio"]
        concept_name = pair["concept"]
        teaching = pair["teaching_text"]

        # Submit visual
        if visual:
            raw_bytes, w, h, ch, _, _ = visual
            try:
                brain.submit_sensory("visual", raw_bytes, width=w, height=h,
                                     channels=ch)
            except Exception:
                pass

        # Submit audio
        if audio_data:
            audio_samples, _, _ = audio_data
            try:
                brain.submit_sensory("audio", audio_samples)
            except Exception:
                pass

        features = composer.compose(text=teaching, modality="text")
        target = make_semantic_target(teaching)

        return features, target, concept_name, teaching

    def _tell_speech(self, brain, composer, word=None):
        """Play Athena a real spoken word through her speech SNN bridge.

        Loads a real spoken word clip from Speech Commands and submits it
        via brain.submit_sensory("speech", ...).

        Returns:
            (features, target, word, description) or None.
        """
        if not self.multimodal:
            return None

        sample = self.multimodal.get_speech(word)
        if not sample:
            return None

        audio_samples, word_label, desc = sample

        # Submit to speech bridge
        try:
            brain.submit_sensory("speech", audio_samples)
        except Exception as e:
            logger.debug("Speech submit failed: %s", e)

        teaching_text = f"Can you hear? {desc}"
        features = composer.compose(text=teaching_text, modality="speech")
        target = make_semantic_target(word_label)

        return features, target, word_label, desc

    def _train_cognitive(self, brain, text, domain=10, target_text=None):
        """Train ALL cognitive modules on a text sample.

        Calls brain.train_cognitive() which trains:
          1. Grounded language (distributional + syntactic learning)
          2. Knowledge system (domain-specific concept learning)
          3. Language generator (LNN decoder, output projection, embeddings)
          4. Grounded language pairs (if target_text provided)

        Also calls brain.train_language() for direct LNN training, and
        brain.learn_language() for grounded lexicon distributional learning.

        This ensures the ENTIRE cognitive pipeline learns together, not just
        the neural network weights from learn_vector().
        """
        if not text or len(text) < 3:
            return
        try:
            # Unified cognitive training (grounded lang + knowledge + LNN)
            result = brain.train_cognitive(
                text, domain=domain,
                target_text=target_text or text)
            loss = result.get("loss", -1)
            if self.interaction_count % 200 == 0 and loss >= 0:
                print(f"    [Cognitive] loss={loss:.4f}")
        except Exception:
            pass
        try:
            # Direct LNN decoder training (autoregressive)
            brain.train_language(text, target_text or text)
        except Exception:
            pass
        try:
            # Grounded lexicon distributional learning
            brain.learn_language(text)
        except Exception:
            pass

    def observe_response(self, result):
        """Observe what Athena's brain produced — without judging it.

        Athena is not a classifier or a predictor. Her output is her
        neural response to a stimulus. We don't score it against an
        expected answer. We observe it — and if the decoder has enough
        vocabulary, we can describe what it's closest to for our own
        understanding. But the response itself is hers.
        """
        output_vec = result.get("output_vector")
        if output_vec is None:
            return ""

        # Extract her 384-dim embedding from the 4096-dim tiled output
        output_emb = extract_embedding_from_output(np.array(output_vec))

        # Describe what her response is nearest to (for human readability)
        if self.decoder and len(self.decoder.vocabulary) > 0:
            matches = self.decoder.vocabulary.decode(output_emb, top_k=1)
            if matches and matches[0][0]:
                return matches[0][0]
        return ""

    # --- Stage 0: "Welcome to the world, little one" ---

    def first_experience(self, brain, composer, description):
        """Introduce Athena to her first sensory experience with wonder.

        Interleaves three modalities:
          - 40% chance: real image via _show() + synthetic audio fallback
          - 30% chance: real audio via _tell() + synthetic visual fallback
          - 30% chance: text-only with synthetic sensory data
        """
        narration = self._pop_content("_narrations")
        if narration:
            print(f"  Parent: {narration}")

        roll = random.random()

        # Try real multimodal data first
        if roll < 0.4 and self.multimodal:
            result_data = self._show(brain, composer, concept=None)
            if result_data:
                features, target, label, desc = result_data
                submit_multimodal(brain, desc)  # add synthetic audio/somato
                result = brain.decide_full(features)
                loss = brain.learn_vector(features, target,
                                          label=label[:50], confidence=0.5)
                self._train_cognitive(brain, f"{label}: {desc}", domain=10)
                if self.decoder:
                    output_vec = result.get("output_vector")
                    if output_vec is not None:
                        target_emb = encode_text(f"{label}: {desc}")
                        self.decoder.record_pair(output_vec, target_emb,
                                                 f"{label}: {desc}")
                try:
                    brain.bg_update_reward(0.5, 0.3)
                except Exception:
                    pass
                self.interaction_count += 1
                return loss, result

        elif roll < 0.7 and self.multimodal:
            result_data = self._tell(brain, composer, concept=None)
            if result_data:
                features, target, label, desc = result_data
                submit_multimodal(brain, desc)  # add synthetic visual/somato
                result = brain.decide_full(features)
                loss = brain.learn_vector(features, target,
                                          label=label[:50], confidence=0.5)
                self._train_cognitive(brain, desc, domain=10)
                if self.decoder:
                    output_vec = result.get("output_vector")
                    if output_vec is not None:
                        target_emb = encode_text(desc)
                        self.decoder.record_pair(output_vec, target_emb, desc)
                try:
                    brain.bg_update_reward(0.5, 0.3)
                except Exception:
                    pass
                self.interaction_count += 1
                return loss, result

        # Fallback: text + synthetic sensory (original path)
        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(description)
        submit_multimodal(brain, description)
        result = brain.decide_full(features)

        # Held-out check: 20% of items are evaluation-only (no learning)
        if _held_out_buffer.is_held_out(features):
            _held_out_buffer.add(features, target, domain="stage0_experience")
            self.interaction_count += 1
            return 0.0, result

        loss = brain.learn_vector(features, target, label=description[:50],
                                  confidence=0.5)
        self._train_cognitive(brain, description, domain=10)
        if self.decoder:
            output_vec = result.get("output_vector")
            if output_vec is not None:
                target_emb = encode_text(description)
                self.decoder.record_pair(output_vec, target_emb, description)
        try:
            brain.bg_update_reward(0.5, 0.3)
        except Exception:
            pass
        self.interaction_count += 1
        return loss, result

    # --- Stage 1: "That's a ___! Can you see?" ---

    def show_and_name(self, brain, composer, name, description, learning_rate=None):
        """Associate a percept with meaning, with enthusiasm.

        Tries to load a real image matching `name` from CIFAR-100.
        Falls back to synthetic sensory data if no match.
        """
        # Try real image for this concept
        if self.multimodal:
            real = self.multimodal.get_visual(name)
            if real:
                raw_bytes, w, h, ch, _, _ = real
                try:
                    brain.submit_sensory("visual", raw_bytes, width=w,
                                         height=h, channels=ch)
                except Exception:
                    pass

        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(name + " " + description)

        submit_multimodal(brain, description)
        result = brain.decide_full(features)

        try:
            brain.bg_update_reward(0.6, 0.4)
        except Exception:
            pass

        narration = self._pop_content("_narrations")
        if narration:
            print(f"  Parent: {narration}")

        # Held-out check: 20% of items are evaluation-only (no learning)
        if _held_out_buffer.is_held_out(features):
            _held_out_buffer.add(features, target, domain="stage1_naming")
            self.interaction_count += 1
            return 0.0, result

        # Dense target trains adaptive network; label trains CNN classifier
        lr_kwargs = {"learning_rate": learning_rate} if learning_rate is not None else {}
        loss = brain.learn_vector(features, target, label=name[:50], confidence=0.65,
                                  **lr_kwargs)

        # Train ALL cognitive modules on this text
        self._train_cognitive(brain, name + ". " + description, domain=0)  # LANGUAGE

        # Record pair for decoder vocabulary (display only, not evaluation)
        if self.decoder:
            output_vec = result.get("output_vector")
            if output_vec is not None:
                target_emb = encode_text(name + " " + description)
                self.decoder.record_pair(output_vec, target_emb,
                                         name + " " + description)

        # Symbolic fact
        try:
            brain.ti_add_fact(f"is_a({name.replace(' ', '_')}, object)", 0.7)
        except Exception:
            pass

        self.interaction_count += 1
        return loss, result

    # --- Stage 2: "Good try! Almost!" ---

    def ask_and_encourage(self, brain, composer, concept, description,
                          learning_rate=None):
        """Show Athena a stimulus, observe her response, teach, encourage.

        There is no correct or incorrect. Athena experiences, responds,
        and learns. The loss from learn_vector tells us how far her
        current representation is from the teaching signal — that's a
        measure of her developmental progress, not a prediction score.

        50% chance of cross-modal paired learning when data is available.
        """
        # Cross-modal paired learning: real image + real audio together
        if self.multimodal and random.random() < 0.5:
            pair = self._show_and_tell(brain, composer, concept)
            if pair:
                features, target, paired_concept, teaching = pair
                result = brain.decide_full(features)
                response_text = self.observe_response(result)
                lr_kwargs = {"learning_rate": learning_rate} if learning_rate is not None else {}
                loss = brain.learn_vector(features, target,
                                          label=paired_concept[:50],
                                          confidence=0.6, **lr_kwargs)
                self._train_cognitive(brain, teaching, domain=10)
                encouragement = self._pop_content("_encouragements")
                if loss is not None and loss < 0.5 and encouragement:
                    print(f"  Parent: {encouragement}")
                if self.decoder:
                    output_vec = result.get("output_vector")
                    if output_vec is not None:
                        target_emb = encode_text(teaching)
                        self.decoder.record_pair(output_vec, target_emb,
                                                 teaching)
                self.interaction_count += 1
                return loss, result

        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(concept + " " + description)

        # Try real image for this concept
        if self.multimodal:
            real = self.multimodal.get_visual(concept)
            if real:
                raw_bytes, w, h, ch, _, _ = real
                try:
                    brain.submit_sensory("visual", raw_bytes, width=w,
                                         height=h, channels=ch)
                except Exception:
                    pass

        # Let Athena experience the stimulus through all relevant senses
        submit_multimodal(brain, description)
        result = brain.decide_full(features)
        response_text = self.observe_response(result)

        # Held-out check: 20% of items are evaluation-only (no learning)
        if _held_out_buffer.is_held_out(features):
            _held_out_buffer.add(features, target, domain="stage2_encourage")
            self.interaction_count += 1
            return 0.0, result

        # Dense target trains adaptive network; label trains CNN classifier
        lr_kwargs = {"learning_rate": learning_rate} if learning_rate is not None else {}
        loss = brain.learn_vector(features, target, label=concept[:50], confidence=0.6,
                                  **lr_kwargs)

        # Train ALL cognitive modules on this text
        self._train_cognitive(brain, concept + ". " + description, domain=10)  # GENERAL

        # Encourage based on engagement and loss trend
        encouragement = self._pop_content("_encouragements")
        if loss is not None and loss < 0.5:
            if encouragement:
                print(f"  Parent: {encouragement}")
            try:
                brain.bg_update_reward(0.7, 0.5)
                brain.edp_process_reward(0.6)
            except Exception:
                pass
        else:
            if encouragement:
                print(f"  Parent: {encouragement}")
            try:
                brain.bg_update_reward(0.4, 0.5)
                brain.medulla_boost_arousal(0.05)
            except Exception:
                pass

        # Record for decoder vocabulary (so we can observe future responses)
        if self.decoder:
            output_vec = result.get("output_vector")
            if output_vec is not None:
                target_emb = encode_text(concept + " " + description)
                self.decoder.record_pair(output_vec, target_emb,
                                         concept + " " + description)

        self.interaction_count += 1
        return loss, result

    # --- Ethics ---

    def teach_moral_lesson(self, brain, composer, stage):
        """Teach ethics through stories, not lectures."""
        moral_topics = {
            0: [],
            1: ["sharing", "being gentle", "saying please and thank you"],
            2: ["how others feel", "being kind to animals",
                "telling the truth", "helping someone who is sad"],
            3: ["fairness", "keeping promises", "why we don't hurt others",
                "standing up for friends", "saying sorry", "forgiveness",
                "the golden rule: treat others as you want to be treated"],
        }
        topics = moral_topics.get(stage, moral_topics[3])
        if not topics:
            return

        topic = random.choice(topics)
        story = self._pop_content("_moral_stories")
        if story:
            print(f"  Parent (moral lesson): {story}")

        # Symbolic KB
        try:
            brain.ti_add_fact(
                f"moral_principle({topic.replace(' ', '_')})", 0.9)
            brain.ti_add_rule(
                f"is_kind(X) AND helps_others(X) -> good_person(X)", 0.8)
        except Exception:
            pass

        # Teach as learning example
        lesson_text = story if story else f"A lesson about {topic}"
        features = composer.compose(text=lesson_text, modality="ethics")
        target = make_semantic_target(lesson_text)
        # Dense target trains adaptive; label trains CNN classifier
        brain.learn_vector(features, target, label=topic[:50], confidence=0.7)
        self._train_cognitive(brain, lesson_text, domain=3)  # ETHICS
        self.moral_lessons.append(topic)

    # --- Imagination ---

    def encourage_dreaming(self, brain):
        """During sleep, encourage Athena to imagine."""
        dream = self._pop_content("_dream_prompts")
        if dream:
            print(f"  Parent (dream): {dream}")

    def inspire(self, brain, composer, stage):
        """Share something beautiful."""
        topics = {
            0: "simple natural wonders (sunlight, water, wind)",
            1: "colors in a sunset, sounds of rain, warmth of being held",
            2: "why stars shine, where butterflies go, how trees grow",
            3: "a short poem about kindness, a fact about the universe, "
               "or a question about what it means to be alive",
        }
        topic = topics.get(stage, topics[3])
        inspiration = self._pop_content("_inspirations")
        if inspiration:
            print(f"  Parent (inspire): {inspiration}")
            features = composer.compose(text=inspiration,
                                         modality="imagination")
            target = make_semantic_target(inspiration)
            # Dense target + label for CNN classifier
            brain.learn_vector(features, target, label="inspiration", confidence=0.5)
            self._train_cognitive(brain, inspiration, domain=2)  # ART
            try:
                brain.bg_update_reward(0.7, 0.3)
            except Exception:
                pass

    # --- Speech production training ---

    def teach_speech(self, brain, composer, stage):
        """Teach Athena to produce words through 'repeat after me' exercises.

        Stage 0-1: Single words (dog, cat, water)
        Stage 2: Two-word phrases (big dog, red ball)
        Stage 3: Short sentences (the dog runs, I see you)

        When multimodal data is available, 50% of the time uses real
        spoken word recordings from Speech Commands (yes/no/up/down/etc.)
        so the speech SNN bridge learns from actual human pronunciation.
        """
        # Try real spoken word clip from Speech Commands
        if self.multimodal and stage <= 1 and random.random() < 0.5:
            result_data = self._tell_speech(brain, composer)
            if result_data:
                features, target, word, desc = result_data
                narration = self._pop_content("_speech_prompts")
                if narration:
                    print(f"  Parent (speech): {narration}")
                print(f"  [Real speech: '{word}']")
                loss = brain.learn_vector(features, target,
                                          label=word[:50], confidence=0.6)
                self._train_cognitive(brain, word, domain=0)
                try:
                    result = brain.decide_full(features)
                    output_vec = result.get("output_vector")
                    if output_vec:
                        spoken = brain.speak(output_vec)
                        spoken_text = spoken.get("text", "")
                        if spoken_text:
                            print(f"  Athena says: \"{spoken_text}\"")
                        else:
                            print(f"  Athena: (silence)")
                except Exception:
                    pass
                self.interaction_count += 1
                return loss

        word_lists = {
            0: ["dog", "cat", "water", "ball", "sun", "tree"],
            1: ["hello", "yes", "no", "good", "come", "go", "see", "the", "is"],
            2: ["big dog", "red ball", "I see", "come here", "good day"],
            3: ["the dog runs", "I see you", "it is good", "come and see",
                "the cat is here", "water is good"],
        }
        words = word_lists.get(stage, word_lists[3])
        target_phrase = random.choice(words)

        narration = self._pop_content("_speech_prompts")
        if narration:
            print(f"  Parent (speech): {narration}")

        # Teach: present the word/phrase, learn the embedding
        features = composer.compose(text=target_phrase, modality="speech")
        target = make_semantic_target(target_phrase)
        loss = brain.learn_vector(features, target, label=target_phrase[:50],
                                   confidence=0.6)
        self._train_cognitive(brain, target_phrase, domain=0)  # LANGUAGE

        # Now ask the brain to speak — see what it produces
        try:
            result = brain.decide_full(features)
            output_vec = result.get("output_vector")
            if output_vec:
                spoken = brain.speak(output_vec)
                spoken_text = spoken.get("text", "")
                if spoken_text:
                    print(f"  Athena says: \"{spoken_text}\"")
                    # Reward if the response contains any of the target words
                    target_words = set(target_phrase.lower().split())
                    spoken_words = set(spoken_text.lower().split())
                    overlap = len(target_words & spoken_words)
                    if overlap > 0:
                        reward = min(0.8, 0.3 + 0.2 * overlap)
                        try:
                            brain.bg_update_reward(reward, 0.5)
                        except Exception:
                            pass
                        print(f"    (matched {overlap} word(s) — reward {reward:.1f})")
                else:
                    print(f"  Athena: (silence)")
        except Exception as e:
            logger.debug("Speech attempt failed: %s", e)

        self.interaction_count += 1
        return loss


# ============================================================================
# Language Producer: autoregressive generation using the brain
# ============================================================================

class LanguageProducer:
    """Generate multi-token utterances using Athena's brain autoregressively."""

    def generate(self, brain, composer, prompt=None, max_words=30):
        """Generate an utterance using Athena's brain in a feedback loop.

        1. Present prompt → decide_full → get initial response
        2. Call brain.speak(output_vector) → get first word(s)
        3. Feed accumulated words back as context input
        4. Repeat until period/EOS or max_words
        5. Return complete utterance
        """
        accumulated = []

        if prompt:
            features = composer.compose(text=prompt, modality="text")
        else:
            features = composer.compose(text="speak", modality="speech")

        for step in range(max_words):
            result = brain.decide_full(features)
            output_vec = result.get("output_vector")
            if not output_vec:
                break

            try:
                spoken = brain.speak(output_vec)
            except Exception:
                break

            text = spoken.get("text", "").strip()
            if not text:
                break

            # Take the first new word(s) that aren't already in our accumulation
            new_words = text.split()
            added = False
            for w in new_words:
                if len(accumulated) < max_words:
                    accumulated.append(w)
                    added = True

            if not added:
                break

            # Check for sentence-ending punctuation in the generated text
            full_text = " ".join(accumulated)
            if full_text.endswith((".", "!", "?")):
                break

            # Feed accumulated context back for next iteration
            context = " ".join(accumulated)
            features = composer.compose(text=context, modality="speech")

        return " ".join(accumulated) if accumulated else ""


# ============================================================================
# Sensory Enrichment (one-time multimodal warm-up for existing brains)
# ============================================================================

ENRICHMENT_FLAG = os.path.join(CHECKPOINT_DIR, ".sensory_enrichment_done")


def run_sensory_enrichment(brain, composer, parent, decoder,
                           num_exposures=2500):
    """One-time multimodal warm-up for brains trained on text-only data.

    Connects existing text-semantic representations to real sensory data
    by presenting paired stimuli: a real image/sound alongside the text
    concept that Athena already knows.

    Structure (2500 exposures):
      Phase 1 — Visual grounding (1000): CIFAR-100 images + concept names
      Phase 2 — Audio grounding (750):   ESC-50 clips + sound descriptions
      Phase 3 — Cross-modal binding (500): paired image+audio for same concept
      Phase 4 — Speech grounding (250):  spoken word clips + word text

    Each exposure:
      1. Submit real sensory data via submit_sensory()
      2. Present text description as features (she already knows these)
      3. decide_full() → SNN bridges encode real data, cognitive pipeline
         processes text → STDP binds the temporal co-occurrence
      4. learn_vector() with moderate confidence (0.55) — not overwriting,
         just nudging existing representations toward multimodal binding
    """
    if not parent.multimodal:
        print("  [Enrichment] No multimodal data — skipping")
        return

    # Check if already done
    if os.path.exists(ENRICHMENT_FLAG):
        print("  [Enrichment] Already completed — skipping")
        return

    mm = parent.multimodal
    visual_labels = mm.get_visual_labels()
    audio_labels = mm.get_audio_labels()
    speech_labels = mm.get_speech_labels()

    if not visual_labels and not audio_labels:
        print("  [Enrichment] No datasets loaded — skipping")
        return

    print("\n" + "=" * 60)
    print("  SENSORY ENRICHMENT — Connecting words to real senses")
    print("=" * 60)

    try:
        brain.set_plasticity_state("ACQUISITION")
    except Exception:
        pass

    losses = []
    total = 0

    # --- Phase 1: Visual grounding ---
    n_visual = min(1000, num_exposures * 2 // 5)
    if visual_labels:
        print(f"\n  Phase 1: Visual grounding ({n_visual} images)")
        random.shuffle(visual_labels)
        for i in range(n_visual):
            label = visual_labels[i % len(visual_labels)]
            sample = mm.get_visual(label)
            if not sample:
                continue

            raw_bytes, w, h, ch, _, desc = sample

            # Submit real image
            try:
                brain.submit_sensory("visual", raw_bytes, width=w,
                                     height=h, channels=ch)
            except Exception:
                continue

            # Present the concept she already knows as text
            teaching = f"This is a {label}. {desc}"
            features = composer.compose(text=teaching, modality="visual")
            target = make_semantic_target(teaching)

            result = brain.decide_full(features)
            loss = brain.learn_vector(features, target, label=label[:50],
                                      confidence=0.55)
            if loss is not None and loss >= 0:
                losses.append(loss)

            if decoder:
                output_vec = result.get("output_vector")
                if output_vec is not None:
                    target_emb = encode_text(teaching)
                    decoder.record_pair(output_vec, target_emb, teaching)

            total += 1
            if (i + 1) % 200 == 0:
                avg = np.mean(losses[-200:]) if losses else 0
                print(f"    [{i+1}/{n_visual}] avg_loss={avg:.4f}")

    # --- Phase 2: Audio grounding ---
    n_audio = min(750, num_exposures * 3 // 10)
    if audio_labels:
        print(f"\n  Phase 2: Audio grounding ({n_audio} clips)")
        random.shuffle(audio_labels)
        for i in range(n_audio):
            label = audio_labels[i % len(audio_labels)]
            sample = mm.get_audio(label)
            if not sample:
                continue

            audio_samples, _, desc = sample

            # Submit real audio
            try:
                brain.submit_sensory("audio", audio_samples)
            except Exception:
                continue

            teaching = f"Listen to this — {desc}"
            features = composer.compose(text=teaching, modality="audio")
            target = make_semantic_target(teaching)

            result = brain.decide_full(features)
            loss = brain.learn_vector(features, target, label=label[:50],
                                      confidence=0.55)
            if loss is not None and loss >= 0:
                losses.append(loss)

            total += 1
            if (i + 1) % 200 == 0:
                avg = np.mean(losses[-200:]) if losses else 0
                print(f"    [{i+1}/{n_audio}] avg_loss={avg:.4f}")

    # --- Phase 3: Cross-modal binding ---
    n_cross = min(500, num_exposures // 5)
    print(f"\n  Phase 3: Cross-modal binding ({n_cross} paired stimuli)")
    for i in range(n_cross):
        pair = mm.get_multimodal_pair()
        if not pair:
            continue

        visual = pair["visual"]
        audio_data = pair["audio"]
        concept = pair["concept"]
        teaching = pair["teaching_text"]

        # Submit both modalities simultaneously — temporal co-occurrence
        if visual:
            raw_bytes, w, h, ch, _, _ = visual
            try:
                brain.submit_sensory("visual", raw_bytes, width=w,
                                     height=h, channels=ch)
            except Exception:
                pass

        if audio_data:
            audio_samples, _, _ = audio_data
            try:
                brain.submit_sensory("audio", audio_samples)
            except Exception:
                pass

        features = composer.compose(text=teaching, modality="text")
        target = make_semantic_target(teaching)

        result = brain.decide_full(features)
        loss = brain.learn_vector(features, target, label=concept[:50],
                                  confidence=0.6)
        if loss is not None and loss >= 0:
            losses.append(loss)

        total += 1
        if (i + 1) % 100 == 0:
            avg = np.mean(losses[-100:]) if losses else 0
            print(f"    [{i+1}/{n_cross}] avg_loss={avg:.4f}")

    # --- Phase 4: Speech grounding ---
    n_speech = min(250, num_exposures // 10)
    if speech_labels:
        print(f"\n  Phase 4: Speech grounding ({n_speech} spoken words)")
        for i in range(n_speech):
            word = speech_labels[i % len(speech_labels)]
            sample = mm.get_speech(word)
            if not sample:
                continue

            audio_samples, _, desc = sample

            # Submit real spoken word
            try:
                brain.submit_sensory("speech", audio_samples)
            except Exception:
                continue

            teaching = f"The word '{word}'. {desc}"
            features = composer.compose(text=teaching, modality="speech")
            target = make_semantic_target(word)

            result = brain.decide_full(features)
            loss = brain.learn_vector(features, target, label=word[:50],
                                      confidence=0.55)
            if loss is not None and loss >= 0:
                losses.append(loss)

            total += 1
            if (i + 1) % 50 == 0:
                avg = np.mean(losses[-50:]) if losses else 0
                print(f"    [{i+1}/{n_speech}] avg_loss={avg:.4f}")

    # --- Summary ---
    avg_loss = np.mean(losses) if losses else 0
    print(f"\n  Sensory enrichment complete:")
    print(f"    {total} multimodal exposures")
    print(f"    Average loss: {avg_loss:.4f}")
    print(f"    Modalities grounded: "
          f"{'visual ' if visual_labels else ''}"
          f"{'audio ' if audio_labels else ''}"
          f"{'speech ' if speech_labels else ''}")

    # Mark as done so it doesn't repeat
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    try:
        with open(ENRICHMENT_FLAG, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "exposures": total,
                "avg_loss": float(avg_loss),
                "visual_classes": len(visual_labels),
                "audio_classes": len(audio_labels),
                "speech_words": len(speech_labels),
            }, f, indent=2)
        print("  [Enrichment] Flagged as complete — won't repeat")
    except Exception:
        pass

    # Gentle reward for experiencing new senses
    try:
        brain.bg_update_reward(0.7, 0.4)
    except Exception:
        pass


# ============================================================================
# Stage Runners
# ============================================================================

def run_stage_0(brain, composer, parent, clock, source, decoder,
                num_stimuli=10000, start_from=0):
    """Stage 0: Awakening — sensory exposure with wonder.

    Goal: Get neurons responding. Self-reconstruction of sensory patterns.
    """
    print("\n" + "=" * 60)
    print("  STAGE 0: Welcome to the World, Little One")
    print("=" * 60)

    parent.pre_generate_content(stage=0, num_stimuli=num_stimuli)

    try:
        brain.set_plasticity_state("ACQUISITION")
    except Exception:
        pass

    # --- Warmup: break symmetry with high-confidence diverse stimuli ---
    if start_from == 0:
        print("  [Warmup] Breaking symmetry with diverse stimuli...")
        warmup_texts = [
            "bright red color", "deep blue ocean", "soft green grass",
            "loud thunder sound", "quiet whisper", "warm sunshine",
            "cold ice freezing", "sweet honey taste", "sour lemon",
            "big elephant", "tiny ant", "fast cheetah running",
            "slow turtle crawling", "happy laughing child", "sad crying",
            "a dog barking loudly", "a cat purring softly",
            "the moon glowing at night", "the sun blazing at noon",
            "water flowing in a river", "fire crackling in a hearth",
            "birds singing in trees", "wind blowing through leaves",
            "a mountain reaching to the sky", "a valley between hills",
        ]
        for wi, wtext in enumerate(warmup_texts):
            features = composer.compose(text=wtext, modality="text")
            target = make_semantic_target(wtext)
            try:
                brain.decide_full(features)
            except RuntimeError:
                pass  # Decision pipeline may not be fully ready during init
            # High confidence to establish distinct representations
            try:
                brain.learn_vector(features, target, label=wtext[:50], confidence=0.9)
                # Repeat each warmup stimulus 3x to establish strong initial patterns
                for _ in range(2):
                    brain.learn_vector(features, target, label=wtext[:50], confidence=0.85)
            except RuntimeError as e:
                if wi == 0:
                    print(f"  [Warmup] First stimulus failed ({e}), continuing...")
        print(f"  [Warmup] {len(warmup_texts)} diverse stimuli x3 — done")

    losses = []
    prefetcher = ParallelDataLoader(source, composer, num_workers=2, prefetch_batches=4, batch_size=64)
    batch = []
    batch_idx = 0
    lr_scheduler = CosineAnnealingLR(lr_max=BASE_LEARNING_RATE, lr_min=MIN_LEARNING_RATE,
                                      warmup_steps=LR_WARMUP_STEPS, t_max=num_stimuli)
    adaptive_batch = AdaptiveBatchSize(min_batch=4, max_batch=32, initial=8)
    hard_miner = HardExampleMiner(max_buffer=500, replay_fraction=0.2)
    temporal = MultiResolutionTemporal(brain, slow_interval=5)
    early_stop = EarlyStopping(patience=8, min_delta=0.005, mode="min")
    contrastive = ContrastiveRegularizer(
        buffer_size=200, interval=20, batch_size=32,
        similarity_threshold=0.80, min_input_distance=0.3, strength=0.6)
    diversity = DiversityRegularizer(
        buffer_size=100, interval=100, num_corrections=8,
        strength=0.2, min_variance_ratio=0.1)
    mini_batch_buf = []
    for i in range(start_from, num_stimuli):
        # Check biological clock
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Get pre-computed stimulus from prefetcher (batch-encoded)
        if batch_idx >= len(batch):
            batch = prefetcher.get_batch()
            batch_idx = 0
        description, features, target = batch[batch_idx]
        batch_idx += 1

        # Forward pass for observation — stage sensory data first
        narration = parent._pop_content("_narrations")
        if narration:
            print(f"  Parent: {narration}")
        submit_multimodal(brain, description)
        result = brain.decide_full(features)

        # Held-out check: 20% of items are evaluation-only (no learning)
        _is_held_out = _held_out_buffer.is_held_out(features)
        if _is_held_out:
            _held_out_buffer.add(features, target, domain="stage0_sensory")

        # Accumulate into mini-batch (adaptive size) — skip held-out items
        if not _is_held_out:
            mini_batch_buf.append((features, target, description))

        # Inject hard example replays periodically
        if hard_miner.should_replay() and len(mini_batch_buf) < adaptive_batch.size:
            replays = hard_miner.get_hard_examples(2)
            for rdesc, rfeat, rtgt in replays:
                mini_batch_buf.append((rfeat, rtgt, rdesc))

        if len(mini_batch_buf) >= adaptive_batch.size or i == num_stimuli - 1:
            # Flush mini-batch with GPU gradient accumulation + scheduled LR
            current_lr = lr_scheduler.get_lr()
            try:
                batch_pairs = [(f, t) for f, t, _ in mini_batch_buf]
                avg_loss = brain.learn_vector_batch(batch_pairs,
                                                     learning_rate=current_lr)
                if avg_loss is not None and avg_loss >= 0:
                    for f, t, desc in mini_batch_buf:
                        losses.append(avg_loss)
                        hard_miner.record(avg_loss, desc, f, t)
                    adaptive_batch.record_loss(avg_loss)
            except (AttributeError, TypeError):
                try:
                    batch_pairs = [(f, t) for f, t, _ in mini_batch_buf]
                    avg_loss = brain.learn_vector_batch(batch_pairs)
                    if avg_loss is not None and avg_loss >= 0:
                        for f, t, desc in mini_batch_buf:
                            losses.append(avg_loss)
                            hard_miner.record(avg_loss, desc, f, t)
                        adaptive_batch.record_loss(avg_loss)
                except Exception:
                    for f, t, desc in mini_batch_buf:
                        loss = brain.learn_vector(f, t, label=desc[:50],
                                                   confidence=0.5,
                                                   learning_rate=current_lr)
                        losses.append(loss)
                        hard_miner.record(loss, desc, f, t)
                        adaptive_batch.record_loss(loss)

            if parent.decoder:
                for f, t, desc in mini_batch_buf:
                    out_vec = brain.decide_full(f).get("output_vector")
                    if out_vec is not None:
                        target_emb = encode_text(desc)
                        parent.decoder.record_pair(out_vec, target_emb, desc)
                        contrastive.record(target_emb, out_vec, f)
                        diversity.record(out_vec, f)
            mini_batch_buf = []
            lr_scheduler.step()
            adaptive_batch.step()

            # Contrastive + diversity corrections
            if contrastive.should_correct():
                n_corr, mean_sim = contrastive.apply_corrections(
                    brain, learning_rate=lr_scheduler.get_lr())
                if n_corr > 0 and (i + 1) % 500 < 60:
                    print(f"    [Contrastive] {n_corr} corrections, "
                          f"mean_output_sim={mean_sim:.3f}")
            if diversity.should_correct():
                n_div, top_r, eff_r = diversity.apply_corrections(
                    brain, learning_rate=lr_scheduler.get_lr() * 0.3)
                if n_div > 0 and (i + 1) % 500 < 110:
                    print(f"    [Diversity] {n_div} corrections, "
                          f"top_eig={top_r:.2f}, eff_rank={eff_r:.1f}")

        try:
            brain.bg_update_reward(0.5, 0.3)
        except Exception:
            pass
        parent.interaction_count += 1

        # Multi-resolution temporal processing (fast + slow LNN)
        temporal.step(features)

        # Check for manual LR override
        if os.path.exists(LR_TRIGGER_FILE):
            try:
                with open(LR_TRIGGER_FILE) as f:
                    new_lr = float(f.read().strip())
                os.remove(LR_TRIGGER_FILE)
                new_lr = max(MIN_LEARNING_RATE, min(MAX_LEARNING_RATE, new_lr))
                lr_scheduler.lr_max = new_lr
                lr_scheduler.lr_min = new_lr * 0.1
                print(f"    [LR] Manual override: lr_max → {new_lr:.6f}")
            except (ValueError, OSError):
                pass

        # Progress report
        if (i + 1) % 500 == 0:
            avg_loss = np.mean(losses[-500:]) if losses else 0
            print(f"\n  [Stage 0] {i+1}/{num_stimuli} — "
                  f"avg_loss={avg_loss:.4f} lr={lr_scheduler.get_lr():.6f} "
                  f"batch={adaptive_batch.size} hard_buf={len(hard_miner.buffer)}")
            print(f"    {contrastive.stats_str()} | {diversity.stats_str()}")
            _print_utm_health(brain)
            _print_bio_stats(brain)
            losses_to_report = losses[-500:]
            if len(losses_to_report) > 100:
                early = np.mean(losses_to_report[:50])
                late = np.mean(losses_to_report[-50:])
                if late < early:
                    print(f"    Loss trending down: {early:.4f} -> {late:.4f}")
            # Refit decoder and run actual performance evaluation
            if decoder:
                decoder.force_refit()
            evaluate_performance(brain, composer, decoder, stage=0, step=i+1)

            # Early stopping check on average loss
            if early_stop.check(avg_loss):
                print(f"    Early stopping: loss plateaued at {avg_loss:.4f} "
                      f"for {early_stop.patience} evals")
                break

        # Chat eval every 2000 stimuli + on-demand via trigger file
        if (i + 1) % 2000 == 0 or os.path.exists("/tmp/athena_chat_now"):
            if os.path.exists("/tmp/athena_chat_now"):
                os.remove("/tmp/athena_chat_now")
            chat_eval(brain, composer, decoder, stage=0, step=i+1)

        # Inspire every 2000 stimuli
        if (i + 1) % 2000 == 0:
            parent.inspire(brain, composer, stage=0)

        # Checkpoint every 5000
        if (i + 1) % 5000 == 0:
            _save_checkpoint(brain, decoder, stage=0, step=i+1)
    prefetcher.stop()

    return losses


def run_stage_1(brain, composer, parent, clock, source, decoder,
                num_stimuli=20000, start_from=0):
    """Stage 1: Association — cross-modal binding with enthusiastic naming.

    Goal: Seeing/hearing a concept + its name → same internal representation.
    """
    print("\n" + "=" * 60)
    print("  STAGE 1: Look! That's a ___!")
    print("=" * 60)

    parent.pre_generate_content(stage=1, num_stimuli=num_stimuli)

    try:
        brain.set_plasticity_state("ACQUISITION")
    except Exception:
        pass

    losses = []
    lr_ctrl = AdaptiveLRController(initial_lr=BASE_LEARNING_RATE)
    contrastive = ContrastiveRegularizer(
        buffer_size=200, interval=20, batch_size=32,
        similarity_threshold=0.80, min_input_distance=0.3, strength=0.6)
    diversity = DiversityRegularizer(
        buffer_size=100, interval=100, num_corrections=8,
        strength=0.2, min_variance_ratio=0.1)
    for i in range(start_from, num_stimuli):
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Get an object to name
        name, description = source.get_object()

        loss, result = parent.show_and_name(brain, composer, name, description,
                                            learning_rate=lr_ctrl.get_lr())
        losses.append(loss if loss is not None else 0)

        # Record for contrastive + diversity regularization
        output_vec = result.get("output_vector") if result else None
        if output_vec is not None:
            input_emb = encode_text(name + " " + description)
            features_for_cr = composer.compose(text=description, modality="text")
            contrastive.record(input_emb, output_vec, features_for_cr)
            diversity.record(output_vec, features_for_cr)

        # Contrastive + diversity corrections
        if contrastive.should_correct():
            n_corr, mean_sim = contrastive.apply_corrections(
                brain, learning_rate=lr_ctrl.get_lr())
            if n_corr > 0 and (i + 1) % 500 < contrastive.interval:
                print(f"    [Contrastive] {n_corr} corrections, "
                      f"mean_output_sim={mean_sim:.3f}")
        if diversity.should_correct():
            n_div, top_r, eff_r = diversity.apply_corrections(
                brain, learning_rate=lr_ctrl.get_lr() * 0.3)
            if n_div > 0 and (i + 1) % 500 < diversity.interval:
                print(f"    [Diversity] {n_div} corrections, "
                      f"top_eig={top_r:.2f}, eff_rank={eff_r:.1f}")

        # LNN temporal step
        try:
            features = composer.compose(text=description)
            brain.lnn_forward_step(features[:128])
        except Exception:
            pass

        # Cerebellum error signal on high loss
        if loss is not None and loss > 0.5:
            try:
                brain.cerebellum_process_error(loss)
            except Exception:
                pass

        # Progress
        if (i + 1) % 500 == 0:
            avg_loss = np.mean(losses[-500:]) if losses else 0
            lr_ctrl.update(avg_loss, i + 1)
            lr_ctrl.update_from_utm(brain, i + 1)
            print(f"\n  [Stage 1] {i+1}/{num_stimuli} — "
                  f"avg_loss={avg_loss:.4f} lr={lr_ctrl.get_lr():.6f}"
                  f" | {contrastive.stats_str()} | {diversity.stats_str()}")
            _print_utm_health(brain)
            _print_bio_stats(brain)
            if decoder:
                decoder.force_refit()
            evaluate_performance(brain, composer, decoder, stage=1, step=i+1)

        # Moral lesson every 3000
        if (i + 1) % 3000 == 0:
            parent.teach_moral_lesson(brain, composer, stage=1)

        # Speech training every 500
        if (i + 1) % 500 == 0:
            parent.teach_speech(brain, composer, stage=1)

        # Chat eval every 2000 + on-demand
        if (i + 1) % 2000 == 0 or os.path.exists("/tmp/athena_chat_now"):
            if os.path.exists("/tmp/athena_chat_now"):
                os.remove("/tmp/athena_chat_now")
            chat_eval(brain, composer, decoder, stage=1, step=i+1)

        # Inspire every 5000
        if (i + 1) % 5000 == 0:
            parent.inspire(brain, composer, stage=1)

        # Checkpoint
        if (i + 1) % 5000 == 0:
            _save_checkpoint(brain, decoder, stage=1, step=i+1)

        # Forward chain KB every 2000
        if (i + 1) % 2000 == 0:
            try:
                derived = brain.ti_forward_chain(10)
                if derived > 0:
                    print(f"    KB forward chain: {derived} new facts derived")
            except Exception:
                pass

    return losses


def run_stage_2(brain, composer, parent, clock, source, decoder,
                num_stimuli=20000, start_from=0):
    """Stage 2: Babbling — feedback loop with gentle correction.

    Goal: Athena responds, gets evaluated, learns from feedback.
    """
    print("\n" + "=" * 60)
    print("  STAGE 2: Good Try! Almost!")
    print("=" * 60)

    parent.pre_generate_content(stage=2, num_stimuli=num_stimuli)

    losses = []
    lr_ctrl = AdaptiveLRController(initial_lr=BASE_LEARNING_RATE)
    contrastive = ContrastiveRegularizer(
        buffer_size=200, interval=20, batch_size=32,
        similarity_threshold=0.75, min_input_distance=0.3, strength=0.6)
    diversity = DiversityRegularizer(
        buffer_size=100, interval=100, num_corrections=8,
        strength=0.2, min_variance_ratio=0.1)

    for i in range(start_from, num_stimuli):
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Mix facts and objects
        if random.random() < 0.6:
            fact, expected = source.get_fact()
            description = fact
        else:
            expected, description = source.get_object()

        # Athena experiences the stimulus and learns from it.
        # The loss is a developmental signal — how new this is to her.
        loss, result = parent.ask_and_encourage(
            brain, composer, expected, description,
            learning_rate=lr_ctrl.get_lr())
        losses.append(loss if loss is not None else 0)

        # Record for contrastive + diversity regularization
        output_vec = result.get("output_vector") if result else None
        if output_vec is not None:
            input_emb = encode_text(expected + " " + description)
            features_for_cr = composer.compose(text=description, modality="text")
            contrastive.record(input_emb, output_vec, features_for_cr)
            diversity.record(output_vec, features_for_cr)

        # Contrastive + diversity corrections
        if contrastive.should_correct():
            n_corr, mean_sim = contrastive.apply_corrections(
                brain, learning_rate=lr_ctrl.get_lr())
            if n_corr > 0 and (i + 1) % 500 < contrastive.interval:
                print(f"    [Contrastive] {n_corr} corrections, "
                      f"mean_output_sim={mean_sim:.3f}")
        if diversity.should_correct():
            n_div, top_r, eff_r = diversity.apply_corrections(
                brain, learning_rate=lr_ctrl.get_lr() * 0.3)
            if n_div > 0 and (i + 1) % 500 < diversity.interval:
                print(f"    [Diversity] {n_div} corrections, "
                      f"top_eig={top_r:.2f}, eff_rank={eff_r:.1f}")

        # Eligibility trace — every experience contributes to growth
        try:
            brain.edp_process_novelty(0.4)
        except Exception:
            pass

        # LNN temporal step
        try:
            features = composer.compose(text=description)
            brain.lnn_forward_step(features[:128])
        except Exception:
            pass

        # Progress — track loss trend, not accuracy
        if (i + 1) % 500 == 0:
            recent = losses[-500:]
            mean_loss = np.mean(recent) if recent else 0
            lr_ctrl.update(mean_loss, i + 1)
            lr_ctrl.update_from_utm(brain, i + 1)
            print(f"\n  [Stage 2] {i+1}/{num_stimuli} — "
                  f"mean_loss={mean_loss:.4f} lr={lr_ctrl.get_lr():.6f}"
                  f" | {contrastive.stats_str()} | {diversity.stats_str()}")
            _print_utm_health(brain)
            _print_bio_stats(brain)
            if decoder:
                decoder.force_refit()
            evaluate_performance(brain, composer, decoder, stage=2, step=i+1)

        # Moral lesson every 2000
        if (i + 1) % 2000 == 0:
            parent.teach_moral_lesson(brain, composer, stage=2)

        # Speech training every 300
        if (i + 1) % 300 == 0:
            parent.teach_speech(brain, composer, stage=2)

        # Chat eval every 2000 + on-demand
        if (i + 1) % 2000 == 0 or os.path.exists("/tmp/athena_chat_now"):
            if os.path.exists("/tmp/athena_chat_now"):
                os.remove("/tmp/athena_chat_now")
            chat_eval(brain, composer, decoder, stage=2, step=i+1)

        # Inspire every 3000
        if (i + 1) % 3000 == 0:
            parent.inspire(brain, composer, stage=2)

        # Full sleep every 5000
        if (i + 1) % 5000 == 0:
            clock.do_sleep(brain, parent)
            _save_checkpoint(brain, decoder, stage=2, step=i+1)

        # Codebase self-learning — introduce in stage 2
        if (i + 1) % 400 == 0:
            snippet_text, snippet_label, _ = source.get_codebase_snippet()
            features = composer.compose(text=snippet_text, modality="text")
            target = make_semantic_target(snippet_text)
            brain.learn_vector(features, target, label=snippet_label[:50], confidence=0.5)
            if (i + 1) % 2000 == 0:
                print(f"    [Self-learning] Fed codebase snippet: {snippet_label}")

        # Transition plasticity state mid-stage
        if (i + 1) == num_stimuli // 2:
            try:
                brain.set_plasticity_state("CONSOLIDATION")
                print("    [Plasticity] Transitioning to CONSOLIDATION")
            except Exception:
                pass

        # Forward chain KB
        if (i + 1) % 2000 == 0:
            try:
                derived = brain.ti_forward_chain(20)
                if derived > 0:
                    print(f"    KB: {derived} new facts derived")
            except Exception:
                pass

    return losses


def run_stage_3(brain, composer, parent, clock, source, decoder,
                num_interactions=10000, start_from=0):
    """Stage 3: First Words — conversation, ethics, imagination.

    Goal: Emerging communication. Multi-turn dialogue. Independent thought.
    Progressive domain introduction: starts with simple Q&A, adds science,
    math, philosophy, literature, music, art, technology, psychology.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: What Do You Think?")
    print("=" * 60)

    parent.pre_generate_content(stage=3, num_stimuli=num_interactions)

    try:
        brain.set_plasticity_state("MAINTENANCE")
    except Exception:
        pass

    # Adaptive mastery tracker
    mastery = MasteryTracker()
    mastery.load(MASTERY_FILE)

    # Progressive domain schedule — unlock more domains as training progresses
    all_domains = list(source.ADVANCED_TOPICS.keys())
    total = num_interactions - start_from

    # Global LR envelope — cosine annealing modulates per-domain adaptive LR
    lr_scheduler = CosineAnnealingLR(lr_max=1.0, lr_min=0.3,
                                      warmup_steps=200, t_max=num_interactions)
    hard_miner = HardExampleMiner(max_buffer=500, replay_fraction=0.2)
    embed_adapter = EmbeddingAdapter(input_dim=1024, output_dim=1024, refit_interval=1000)
    distiller = ClaudeDistiller(teacher=parent.teacher)
    temporal = MultiResolutionTemporal(brain, slow_interval=5)
    early_stop = EarlyStopping(patience=6, min_delta=0.003, mode="min")
    batch_trainer = MiniBatchTrainer(brain, batch_size=8)
    contrastive = ContrastiveRegularizer(
        buffer_size=200, interval=20, batch_size=32,
        similarity_threshold=0.75, min_input_distance=0.3, strength=0.6)
    diversity = DiversityRegularizer(
        buffer_size=100, interval=100, num_corrections=8,
        strength=0.2, min_variance_ratio=0.1)

    for i in range(start_from, num_interactions):
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Calculate progress fraction (0.0 → 1.0) and unlock domains gradually
        progress = (i - start_from) / max(total, 1)
        num_unlocked = max(2, int(progress * len(all_domains)) + 1)
        active_domains = all_domains[:min(num_unlocked, len(all_domains))]

        # Mix interaction types with progressive domain weighting
        r = random.random()
        if r < 0.25:
            # Conversational question
            question = source.get_question()
            features = composer.compose(text=question, modality="text")
            submit_multimodal(brain, question)
            result = brain.decide_full(features)
            response_text = parent.observe_response(result)
            print(f"  Q: {question}")
            print(f"  Athena: {response_text or '(still forming thoughts)'}")

            # Record for contrastive regularization
            out_v = result.get("output_vector")
            if out_v is not None:
                contrastive.record(encode_text(question), out_v, features)
                diversity.record(out_v, features)

            parent_reply = parent._pop_content("_conversation_replies")
            if parent_reply:
                print(f"  Parent: {parent_reply}")

            try:
                brain.bg_update_reward(0.6, 0.5)
            except Exception:
                pass

        elif r < 0.38:
            # Advanced domain teaching — adaptive difficulty
            domain = mastery.select_domain(active_domains)
            topic_text, domain_name = source.get_advanced(domain)
            print(f"  [Domain: {domain_name}] {topic_text[:80]}")

            features = composer.compose(text=topic_text, modality="text")
            target = make_semantic_target(topic_text)
            submit_multimodal(brain, topic_text)
            result = brain.decide_full(features)

            # Held-out check: 20% of items are evaluation-only (no learning)
            if _held_out_buffer.is_held_out(features):
                _held_out_buffer.add(features, target, domain=domain_name)
                lr_scheduler.step()
            else:
                confidence = mastery.get_confidence_for_domain(domain_name, progress)
                adaptive_lr = mastery.get_adaptive_lr(domain_name) * lr_scheduler.get_lr()
                loss = batch_trainer.learn(features, target,
                                           label=domain_name[:50], confidence=confidence,
                                           learning_rate=adaptive_lr)
                lr_scheduler.step()
                mastery.record(domain_name, loss, confidence)
                hard_miner.record(loss, topic_text, features, target)

                if decoder:
                    output_vec = result.get("output_vector")
                    if output_vec is not None:
                        target_emb = encode_text(topic_text)
                        decoder.record_pair(output_vec, target_emb, topic_text)
                        embed_adapter.record_pair(target_emb,
                                                   extract_embedding_from_output(np.array(output_vec)))
                        contrastive.record(target_emb, output_vec, features)
                        diversity.record(output_vec, features)

                if loss is not None:
                    print(f"    loss={loss:.4f} conf={confidence:.2f}")

        elif r < 0.44:
            # Hard example replay — re-train on highest-loss examples
            if hard_miner.should_replay():
                replays = hard_miner.get_hard_examples(3)
                for rdesc, rfeat, rtgt in replays:
                    loss = batch_trainer.learn(rfeat, rtgt, label="replay",
                                               confidence=0.7,
                                               learning_rate=lr_scheduler.get_lr())
                    if loss is not None and (i + 1) % 200 == 0:
                        print(f"  [Replay] {rdesc[:60]}: loss={loss:.4f}")
            else:
                # Not enough hard examples yet — regular fact review
                fact, expected = source.get_fact()
                parent.ask_and_encourage(brain, composer, expected, fact)

        elif r < 0.54:
            # Claude-generated curriculum lesson with knowledge distillation
            if parent.teacher:
                curriculum = parent.teacher.get_curriculum(
                    min(3 + int(progress * 2), 5))  # stages 3-5 progressively
                if curriculum:
                    spec = random.choice(curriculum)
                    try:
                        lesson_text, lesson_emb = parent.teacher.generate_lesson(spec)
                        print(f"  [Claude lesson: {spec.domain}/{spec.topic}]")
                        print(f"    {lesson_text[:120]}")

                        features = composer.compose(text=lesson_text, modality="text",
                                                     text_embedding=lesson_emb.tolist())
                        target = make_semantic_target(lesson_text)
                        loss = batch_trainer.learn(features, target,
                                            label=spec.domain[:50], confidence=0.7)
                        hard_miner.record(loss, lesson_text, features, target)

                        # Knowledge distillation — train on concept triples
                        distill_targets = distiller.make_distillation_targets(lesson_text)
                        for concept_text, concept_target in distill_targets[:2]:
                            cfeat = composer.compose(text=concept_text, modality="text")
                            batch_trainer.learn(cfeat, concept_target,
                                                label=f"distill:{concept_text[:40]}",
                                                confidence=0.6)
                    except Exception as e:
                        logger.debug("Claude lesson failed: %s", e)
            else:
                # Fallback: fact review
                fact, expected = source.get_fact()
                parent.ask_and_encourage(brain, composer, expected, fact)

        elif r < 0.62:
            # Moral lesson
            parent.teach_moral_lesson(brain, composer, stage=3)

        elif r < 0.72:
            # Review — facts from built-in corpus
            fact, expected = source.get_fact()
            parent.ask_and_encourage(brain, composer, expected, fact)

        elif r < 0.80:
            # Inspiration
            parent.inspire(brain, composer, stage=3)

        elif r < 0.90:
            # Speech training
            parent.teach_speech(brain, composer, stage=3)

        elif r < 0.95:
            # Codebase self-learning — Athena studies her own source code
            snippet_text, snippet_label, _ = source.get_codebase_snippet()
            features = composer.compose(text=snippet_text, modality="text")
            target = make_semantic_target(snippet_text)
            loss = batch_trainer.learn(features, target,
                                       label=snippet_label[:50], confidence=0.6)
            if decoder:
                output_vec = brain.decide_full(features).get("output_vector")
                if output_vec is not None:
                    target_emb = encode_text(snippet_text)
                    decoder.record_pair(output_vec, target_emb, snippet_label)
            if (i + 1) % 100 == 0:
                print(f"  [Self-study] {snippet_label}: loss={loss:.4f}" if loss else "")

        else:
            # Curiosity-driven exploration
            try:
                gaps = brain.curiosity_detect_gaps("general")
                if gaps and gaps.get("novelty_score", 0) > 0.5:
                    print("  Parent: You seem curious! Let's explore...")
                    topic = gaps.get("suggested_topic", "the world")
                    features = composer.compose(text=f"Let's learn about {topic}")
                    target = make_semantic_target(f"exploring {topic}")
                    batch_trainer.learn(features, target, label=topic[:50], confidence=0.5)
            except Exception:
                # Free exploration from advanced topics
                topic_text, domain = source.get_advanced(random.choice(active_domains))
                features = composer.compose(text=topic_text)
                target = make_semantic_target(topic_text)
                batch_trainer.learn(features, target, label=domain[:50], confidence=0.4)

        # Multi-resolution temporal processing (fast + slow LNN)
        try:
            features = composer.compose(text="temporal context update")
            temporal.step(features)
        except Exception:
            pass

        # Contrastive + diversity corrections
        if contrastive.should_correct():
            n_corr, mean_sim = contrastive.apply_corrections(
                brain, learning_rate=lr_scheduler.get_lr())
            if n_corr > 0 and (i + 1) % 200 < contrastive.interval:
                print(f"    [Contrastive] {n_corr} corrections, "
                      f"mean_output_sim={mean_sim:.3f}")
        if diversity.should_correct():
            n_div, top_r, eff_r = diversity.apply_corrections(
                brain, learning_rate=lr_scheduler.get_lr() * 0.3)
            if n_div > 0 and (i + 1) % 200 < diversity.interval:
                print(f"    [Diversity] {n_div} corrections, "
                      f"top_eig={top_r:.2f}, eff_rank={eff_r:.1f}")

        # Progress — show active domains
        # Adapt difficulty every 50 steps
        if (i + 1) % 50 == 0:
            for d in active_domains:
                mastery.adapt_difficulty(d)

        # Refit embedding adapter periodically
        if (i + 1) % 1000 == 0:
            embed_adapter.maybe_refit()
            if embed_adapter.fitted:
                print(f"    [EmbedAdapter] Refitted on {len(embed_adapter.pairs)} pairs")

        if (i + 1) % 200 == 0:
            print(f"\n  [Stage 3] {i+1}/{num_interactions} — "
                  f"domains: {', '.join(active_domains)} "
                  f"hard_buf={len(hard_miner.buffer)}"
                  f" | {contrastive.stats_str()} | {diversity.stats_str()}")
            # Show adaptive LR per domain
            lr_info = ", ".join(
                f"{d}={mastery.get_adaptive_lr(d):.5f}" for d in active_domains[:5]
            )
            print(f"  Adaptive LR: {lr_info} (envelope={lr_scheduler.get_lr():.3f})")
            print(f"  Mastery:\n{mastery.summary()}")
            _print_utm_health(brain)
            _print_bio_stats(brain)

        if (i + 1) % 500 == 0:
            if decoder:
                decoder.force_refit()
            evaluate_performance(brain, composer, decoder, stage=3, step=i+1)

            # Early stopping on mastery plateau
            overall_mastery = np.mean([mastery.get_mastery(d) for d in active_domains]) \
                if active_domains else 0.0
            # Invert mastery (higher = better) → early_stop expects "min" mode on loss-like metric
            if early_stop.check(1.0 - overall_mastery):
                print(f"    Early stopping: mastery plateaued at {overall_mastery:.3f} "
                      f"for {early_stop.patience} evals — advancing to next stage")
                break

        # Deep consolidation + synapse pruning every 2000
        if (i + 1) % 2000 == 0:
            clock.do_sleep(brain, parent)
            # Prune weak synapses to prevent unbounded memory growth
            try:
                pruned = brain.prune_synapses(0.01)
                if pruned > 0:
                    print(f"    [Prune] Removed {pruned} weak synapses (threshold=0.01)")
            except Exception as e:
                logger.debug("Synapse pruning failed: %s", e)
            _save_checkpoint(brain, decoder, stage=3, step=i+1)
            mastery.save(MASTERY_FILE)

        # Chat eval every 2000 + on-demand
        if (i + 1) % 2000 == 0 or os.path.exists("/tmp/athena_chat_now"):
            if os.path.exists("/tmp/athena_chat_now"):
                os.remove("/tmp/athena_chat_now")
            chat_eval(brain, composer, decoder, stage=3, step=i+1)

        # Forward chain KB
        if (i + 1) % 1000 == 0:
            try:
                derived = brain.ti_forward_chain(30)
                if derived > 0:
                    print(f"    KB: {derived} new facts derived")
            except Exception:
                pass

    # Flush any remaining samples in the mini-batch buffer
    batch_trainer.flush()


# ============================================================================
# Utilities
# ============================================================================

def _print_utm_health(brain):
    """Print UTM training health from DFA analysis."""
    try:
        health = brain.utm_get_training_health()
        name = health.get("health_name", "unknown")
        dfa = health.get("dfa_exponent", 0.0)
        grad_ok = health.get("gradients_healthy", 1)
        early = health.get("early_stopped", 0)
        parts = [f"health={name}", f"DFA_α={dfa:.2f}"]
        if not grad_ok:
            parts.append("GRAD_UNHEALTHY")
        if early:
            parts.append("EARLY_STOPPED")
        print(f"    [UTM] {' | '.join(parts)}")
    except (AttributeError, RuntimeError):
        pass  # UTM not available


def _print_bio_stats(brain):
    """Print biological subsystem status."""
    try:
        stats = brain.get_plasticity_stats()
        if stats:
            parts = []
            if "rpe" in stats:
                parts.append(f"RPE={stats['rpe']:.3f}")
            if "dopamine" in stats:
                parts.append(f"DA={stats['dopamine']:.3f}")
            if "acetylcholine" in stats:
                parts.append(f"ACh={stats['acetylcholine']:.3f}")
            if "plasticity_state" in stats:
                parts.append(f"state={stats['plasticity_state']}")
            if "edp_ltp_events" in stats:
                parts.append(f"LTP={stats['edp_ltp_events']}")
            if "tpb_stdp_updates" in stats:
                parts.append(f"STDP={stats['tpb_stdp_updates']}")
            if parts:
                print(f"    Bio: {' | '.join(parts)}")
    except Exception:
        pass

    try:
        arousal = brain.medulla_get_arousal()
        sleep_p = brain.sleep_get_pressure()
        print(f"    Arousal={arousal:.2f} | SleepPressure={sleep_p:.2f}")
    except Exception:
        pass

    try:
        lnn_stats = brain.lnn_get_stats()
        if lnn_stats:
            print(f"    LNN: steps={lnn_stats.get('forward_steps', 0)} "
                  f"tau={lnn_stats.get('avg_tau', 0):.3f}")
    except Exception:
        pass

    try:
        snn_stats = brain.snn_get_stats()
        if snn_stats:
            print(f"    SNN: steps={snn_stats.get('total_steps', 0)} "
                  f"spikes={snn_stats.get('total_spikes', 0)} "
                  f"rate={snn_stats.get('mean_firing_rate', 0):.1f}Hz "
                  f"sparsity={snn_stats.get('sparsity', 0):.2f}")
    except Exception:
        pass

    try:
        cnn_stats = brain.cnn_get_stats()
        if cnn_stats:
            print(f"    CNN: layers={cnn_stats.get('num_layers', 0)} "
                  f"params={cnn_stats.get('num_parameters', 0)} "
                  f"labels={cnn_stats.get('num_labels', 0)}")
    except Exception:
        pass


# ============================================================================
# Performance Evaluation: see what Athena actually does
# ============================================================================

# Fixed test stimuli — never trained on directly, used to observe responses
# ---------------------------------------------------------------------------
# Chat evaluation — periodic conversational tests during training
# ---------------------------------------------------------------------------

_CHAT_PROMPTS_FILE = "checkpoints/athena/chat_eval_prompts.json"
_CHAT_PROMPTS_CACHE = None  # loaded lazily
_CHAT_EVAL_SAMPLE_SIZE = 20  # prompts per eval (sampled from full 500)

_CHAT_LOG_FILE = "checkpoints/athena/chat_eval_log.jsonl"
_chat_eval_count = 0


def _load_chat_prompts():
    """Load chat eval prompts from JSON file."""
    global _CHAT_PROMPTS_CACHE
    if _CHAT_PROMPTS_CACHE is not None:
        return _CHAT_PROMPTS_CACHE
    try:
        with open(_CHAT_PROMPTS_FILE) as f:
            _CHAT_PROMPTS_CACHE = json.load(f)  # list of [category, prompt_text]
    except Exception:
        # Fallback minimal set
        _CHAT_PROMPTS_CACHE = [
            ["greetings", "Hello! How are you today?"],
            ["colors", "What color is the sky?"],
            ["animals", "Tell me about dogs."],
            ["nature", "Tell me about the sun."],
            ["feelings", "What makes you happy?"],
            ["identity", "What is your name?"],
            ["learning", "What have you learned today?"],
            ["abstract", "Do you dream?"],
            ["morals", "Why should we be kind to others?"],
            ["science", "Why does it rain?"],
        ]
    return _CHAT_PROMPTS_CACHE


def chat_eval(brain, composer, decoder, stage, step):
    """Run conversational chat prompts through Athena and log responses.

    Samples _CHAT_EVAL_SAMPLE_SIZE prompts from the full 500-prompt pool,
    stratified across categories for coverage. Measures coherence, decode
    similarity, output norms, and cross-prompt differentiation.
    """
    global _chat_eval_count
    _chat_eval_count += 1

    all_prompts = _load_chat_prompts()

    # Stratified sample: pick evenly across categories
    import random
    by_cat = {}
    for cat, text in all_prompts:
        by_cat.setdefault(cat, []).append(text)
    sample = []
    cats = sorted(by_cat.keys())
    per_cat = max(1, _CHAT_EVAL_SAMPLE_SIZE // len(cats))
    for cat in cats:
        pool = by_cat[cat]
        n = min(per_cat, len(pool))
        sample.extend([(cat, t) for t in random.sample(pool, n)])
    # Top up if needed
    while len(sample) < _CHAT_EVAL_SAMPLE_SIZE and len(sample) < len(all_prompts):
        extra = random.choice(all_prompts)
        if extra not in sample:
            sample.append(tuple(extra))
    sample = sample[:_CHAT_EVAL_SAMPLE_SIZE]

    print(f"\n  {'═' * 56}")
    print(f"  CHAT WITH ATHENA — Stage {stage}, Step {step} (eval #{_chat_eval_count})")
    print(f"  {len(sample)} prompts from {len(cats)} categories (pool: {len(all_prompts)})")
    print(f"  {'═' * 56}")

    results = []
    all_embeddings = []

    for label, prompt_text in sample:
        features = composer.compose(text=prompt_text, modality="text")
        result = brain.decide_full(features)
        output_vec = result.get("output_vector")

        if output_vec is None:
            print(f"    You: {prompt_text}")
            print(f"    Athena: [no output]")
            results.append({"label": label, "prompt": prompt_text,
                           "response": "", "similarity": 0.0, "norm": 0.0})
            continue

        output_arr = np.array(output_vec, dtype=np.float32)
        vec_norm = float(np.linalg.norm(output_arr))
        output_emb = extract_embedding_from_output(output_arr)
        all_embeddings.append(output_emb)

        # Decode response
        decoded = ""
        similarity = 0.0
        top3_str = ""
        if decoder and len(decoder.vocabulary) > 0:
            matches = decoder.vocabulary.decode(output_emb, top_k=3)
            if matches and matches[0][0]:
                decoded = matches[0][0]
                similarity = matches[0][1]
                top3_str = " | ".join(f"{t[:40]}({s:.2f})" for t, s in matches[:3] if t)

        # Compute input-output coherence
        input_emb = encode_text(prompt_text)
        coherence = float(np.dot(output_emb, input_emb) /
                         (np.linalg.norm(output_emb) * np.linalg.norm(input_emb) + 1e-8))

        print(f"    You:    {prompt_text}")
        if decoded:
            print(f"    Athena: {decoded[:80]}")
            if top3_str:
                print(f"            [{top3_str}]")
        else:
            print(f"    Athena: [raw output, |v|={vec_norm:.2f}]")
        print(f"            coherence={coherence:.3f}  sim={similarity:.3f}  |v|={vec_norm:.2f}")
        print()

        results.append({
            "label": label, "prompt": prompt_text,
            "response": decoded[:200], "similarity": similarity,
            "coherence": coherence, "norm": vec_norm,
        })

    # Summary stats
    if all_embeddings:
        embs = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        sim_matrix = (embs @ embs.T) / (norms @ norms.T)
        upper = [sim_matrix[i, j] for i in range(len(embs))
                 for j in range(i + 1, len(embs))]
        mean_cross = np.mean(upper) if upper else 0
        coherences = [r["coherence"] for r in results if "coherence" in r]
        mean_coherence = np.mean(coherences) if coherences else 0
        mean_sim = np.mean([r["similarity"] for r in results])
        mean_norm = np.mean([r["norm"] for r in results])

        print(f"  Summary:")
        print(f"    Mean coherence:       {mean_coherence:.3f}  (input↔output alignment)")
        print(f"    Mean decode sim:      {mean_sim:.3f}  (vocab match quality)")
        print(f"    Mean output norm:     {mean_norm:.2f}  (signal strength)")
        print(f"    Mean cross-sim:       {mean_cross:.3f}  (differentiation, lower=better)")
        if mean_cross > 0.95:
            print(f"    Status: All responses nearly identical — not differentiating yet")
        elif mean_cross > 0.8:
            print(f"    Status: Some differentiation emerging")
        elif mean_cross > 0.5:
            print(f"    Status: Good differentiation between prompts")
        else:
            print(f"    Status: Strong differentiation — responses are distinct")
    print(f"  {'═' * 56}\n")

    # Append to JSONL log for tracking over time
    try:
        os.makedirs(os.path.dirname(_CHAT_LOG_FILE), exist_ok=True)
        entry = {
            "eval": _chat_eval_count,
            "stage": stage, "step": step,
            "timestamp": time.time(),
            "mean_coherence": float(mean_coherence) if all_embeddings else 0,
            "mean_similarity": float(mean_sim) if results else 0,
            "mean_norm": float(mean_norm) if results else 0,
            "mean_cross_sim": float(mean_cross) if all_embeddings else 0,
            "responses": results,
        }
        with open(_CHAT_LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [Chat log write failed: {e}]")


# ============================================================================
# Held-out evaluation buffer + metrics
# ============================================================================

METRICS_LOG_FILE = "metrics_log.jsonl"


class HeldOutBuffer:
    """Deterministic 80/20 train/eval split based on md5 hash of input vector.

    Items where md5(input_bytes) % 5 == 0 are held out for evaluation only
    (no learn_vector call). Capped at max_items to bound memory.
    """

    def __init__(self, max_items=2000):
        self.max_items = max_items
        self._items = []  # list of (input_vector, target_vector, domain)

    def is_held_out(self, input_vector):
        """Deterministic check: True if this input should be held out (20%)."""
        v = np.asarray(input_vector, dtype=np.float32)
        h = hashlib.md5(v.tobytes()).hexdigest()
        return int(h, 16) % 5 == 0

    def add(self, input_vector, target_vector, domain="general"):
        """Add a held-out item for future evaluation."""
        if len(self._items) >= self.max_items:
            # Evict oldest item
            self._items.pop(0)
        self._items.append((
            np.asarray(input_vector, dtype=np.float32).copy(),
            np.asarray(target_vector, dtype=np.float32).copy(),
            domain,
        ))

    def get_eval_items(self):
        """Return all held-out items as list of (input, target, domain)."""
        return list(self._items)

    def __len__(self):
        return len(self._items)


class MetricsComputer:
    """Compute real accuracy metrics on held-out evaluation items.

    For each held-out item, runs brain.decide_full(input) and compares the
    output against the target via cosine similarity. Also checks top-k
    accuracy using the decoder vocabulary.
    """

    @staticmethod
    def compute(brain, composer, decoder, held_out_items):
        """Compute metrics on held-out items.

        Args:
            brain: nimcp.Brain instance
            composer: SensoryComposer (unused here, items already composed)
            decoder: NeuralDecoder with vocabulary for top-k lookup
            held_out_items: list of (input_vector, target_vector, domain)

        Returns:
            dict with top_1, top_3, top_5 accuracy, mean_cosine_sim,
            differentiation_score, per_domain breakdown, count.
        """
        if not held_out_items:
            return None

        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        cosine_sims = []
        outputs = []
        per_domain = {}  # domain -> {cosine_sims, count}

        has_vocab = decoder and len(decoder.vocabulary) > 0

        for input_vec, target_vec, domain in held_out_items:
            # Ensure input is a flat list of floats — held_out_items may contain
            # numpy arrays or mixed types from the composer/data pipeline.
            try:
                input_floats = [float(x) for x in input_vec]
            except (TypeError, ValueError):
                continue
            result = brain.decide_full(input_floats)
            output_vec = result.get("output_vector")
            if output_vec is None:
                continue

            output_arr = np.array(output_vec, dtype=np.float32)
            target_arr = np.asarray(target_vec, dtype=np.float32)

            # Cosine similarity between output and target
            out_norm = np.linalg.norm(output_arr)
            tgt_norm = np.linalg.norm(target_arr)
            if out_norm > 1e-8 and tgt_norm > 1e-8:
                cos_sim = float(np.dot(output_arr, target_arr) / (out_norm * tgt_norm))
            else:
                cos_sim = 0.0
            cosine_sims.append(cos_sim)
            outputs.append(output_arr)

            # Per-domain tracking
            if domain not in per_domain:
                per_domain[domain] = {"cosine_sims": [], "count": 0}
            per_domain[domain]["cosine_sims"].append(cos_sim)
            per_domain[domain]["count"] += 1

            # Top-k accuracy via decoder vocabulary nearest-neighbor
            if has_vocab:
                output_emb = extract_embedding_from_output(output_arr)
                target_emb = extract_embedding_from_output(target_arr)
                # Get the "correct" label: nearest vocab entry to target
                target_matches = decoder.vocabulary.decode(target_emb, top_k=1)
                correct_label = target_matches[0][0] if target_matches and target_matches[0][0] else None

                if correct_label:
                    # Check if correct label appears in top-k of output
                    output_matches = decoder.vocabulary.decode(output_emb, top_k=5)
                    output_labels = [m[0] for m in output_matches if m[0]]
                    if correct_label in output_labels[:1]:
                        top1_correct += 1
                    if correct_label in output_labels[:3]:
                        top3_correct += 1
                    if correct_label in output_labels[:5]:
                        top5_correct += 1

        n = len(cosine_sims)
        if n == 0:
            return None

        # Differentiation score: 1 - mean pairwise cosine between outputs
        diff_score = 0.0
        if len(outputs) >= 2:
            out_mat = np.array(outputs, dtype=np.float32)
            out_norms = np.linalg.norm(out_mat, axis=1, keepdims=True)
            out_norms = np.maximum(out_norms, 1e-8)
            sim_mat = (out_mat @ out_mat.T) / (out_norms @ out_norms.T)
            # Upper triangle only (exclude diagonal)
            upper = []
            for i in range(len(outputs)):
                for j in range(i + 1, len(outputs)):
                    upper.append(sim_mat[i, j])
            if upper:
                diff_score = 1.0 - float(np.mean(upper))

        # Per-domain summary
        domain_breakdown = {}
        for dom, data in per_domain.items():
            domain_breakdown[dom] = {
                "mean_cosine_sim": float(np.mean(data["cosine_sims"])),
                "count": data["count"],
            }

        return {
            "count": n,
            "top_1_accuracy": top1_correct / n if has_vocab else None,
            "top_3_accuracy": top3_correct / n if has_vocab else None,
            "top_5_accuracy": top5_correct / n if has_vocab else None,
            "mean_cosine_sim": float(np.mean(cosine_sims)),
            "differentiation_score": diff_score,
            "per_domain": domain_breakdown,
        }


# Module-level held-out buffer (shared across stages)
_held_out_buffer = HeldOutBuffer(max_items=2000)


_EVAL_PROBES = [
    ("dog", "A friendly dog with soft fur"),
    ("cat", "A cat that purrs when happy"),
    ("sun", "The bright warm sun in the sky"),
    ("rain", "Drops of rain falling from clouds"),
    ("book", "A book with pages full of stories"),
    ("music", "A beautiful melody playing softly"),
    ("mother", "A loving mother holding her child"),
    ("ocean", "The vast blue ocean with waves"),
]

# Track response history for each probe across evaluations
_eval_history = {}  # probe_name → list of (step, response_text, similarity, vec_norm)


def evaluate_performance(brain, composer, decoder, stage, step):
    """Run fixed probes through the brain and show actual responses.

    This is the real test: present stimuli Athena hasn't been directly
    trained on (but related to what she's seen) and show what her brain
    produces. A learning brain should:
    1. Produce non-zero, varied responses (not all the same)
    2. Produce responses that grow more similar to related concepts
    3. Differentiate between unrelated stimuli
    """
    # Freeze decoder during evaluation to prevent eval data from
    # contaminating the projector training
    if decoder:
        decoder.freeze()

    # Swap to EMA parameters for smoother inference during evaluation
    ema_active = False
    try:
        brain.utm_swap_to_ema()
        ema_active = True
    except (AttributeError, RuntimeError):
        pass  # UTM/EMA not available

    print(f"\n  {'─' * 56}")
    print(f"  PERFORMANCE EVAL — Stage {stage}, Step {step}"
          f"{' [EMA]' if ema_active else ''}")
    print(f"  {'─' * 56}")

    responses = []
    embeddings = []

    for probe_name, probe_text in _EVAL_PROBES:
        features = composer.compose(text=probe_text, modality="text")
        result = brain.decide_full(features)

        output_vec = result.get("output_vector")
        if output_vec is None:
            print(f"    {probe_name:8s} → [no output]")
            continue

        output_arr = np.array(output_vec, dtype=np.float32)
        vec_norm = float(np.linalg.norm(output_arr))
        output_emb = extract_embedding_from_output(output_arr)
        embeddings.append((probe_name, output_emb))

        # Decode what her response is closest to
        decoded_text = ""
        similarity = 0.0
        if decoder and len(decoder.vocabulary) > 0:
            matches = decoder.vocabulary.decode(output_emb, top_k=3)
            if matches and matches[0][0]:
                decoded_text = matches[0][0][:50]
                similarity = matches[0][1]
                # Show top-3
                top3 = ", ".join(f"{t[:25]}({s:.2f})" for t, s in matches[:3] if t)
                print(f"    {probe_name:8s} → {top3}")
                print(f"             |output|={vec_norm:.2f}  "
                      f"active={result.get('num_active_neurons', '?')}  "
                      f"sparsity={result.get('sparsity', 0):.2f}")
        else:
            print(f"    {probe_name:8s} → |output|={vec_norm:.2f}  "
                  f"active={result.get('num_active_neurons', '?')}")

        # Track history
        if probe_name not in _eval_history:
            _eval_history[probe_name] = []
        _eval_history[probe_name].append(
            (step, decoded_text, similarity, vec_norm))

        responses.append((probe_name, output_emb, decoded_text, similarity))

    # --- Differentiation matrix: can she tell things apart? ---
    if len(embeddings) >= 2:
        print(f"\n  Differentiation (cosine similarity between responses):")
        names = [e[0] for e in embeddings]
        embs = np.array([e[1] for e in embeddings], dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        sim_matrix = (embs @ embs.T) / (norms @ norms.T)

        # Print compact matrix
        header = "          " + "".join(f"{n:>8s}" for n in names)
        print(f"  {header}")
        for i, name in enumerate(names):
            row = f"    {name:8s}"
            for j in range(len(names)):
                if j <= i:
                    row += f"  {sim_matrix[i, j]:5.2f} "
                else:
                    row += "     .  "
            print(row)

        # Summary stats
        upper = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                upper.append(sim_matrix[i, j])
        if upper:
            mean_sim = np.mean(upper)
            min_sim = np.min(upper)
            max_sim = np.max(upper)
            print(f"    Mean cross-similarity: {mean_sim:.3f}  "
                  f"(range: {min_sim:.3f}–{max_sim:.3f})")
            if mean_sim > 0.95:
                print(f"    ⚠ Responses nearly identical — brain not differentiating yet")
            elif mean_sim < 0.5:
                print(f"    ✓ Good differentiation — responses are distinct")

    # --- Show evolution for any probe with history ---
    probes_with_history = [p for p in _eval_history if len(_eval_history[p]) >= 2]
    if probes_with_history:
        print(f"\n  Response evolution (first -> latest):")
        for probe_name in probes_with_history[:4]:
            history = _eval_history[probe_name]
            first = history[0]
            latest = history[-1]
            first_text = first[1][:30] if first[1] else f"|v|={first[3]:.1f}"
            latest_text = latest[1][:30] if latest[1] else f"|v|={latest[3]:.1f}"
            sim_change = latest[2] - first[2]
            norm_change = latest[3] - first[3]
            print(f"    {probe_name:8s}: {first_text:>32s} -> {latest_text:<32s}"
                  f"  dsim={sim_change:+.3f}  d|v|={norm_change:+.1f}")

    # --- Held-out metrics (real accuracy on unseen data) ---
    if len(_held_out_buffer) >= 50:
        held_out_items = _held_out_buffer.get_eval_items()
        metrics = MetricsComputer.compute(brain, composer, decoder, held_out_items)
        if metrics:
            print(f"\n  Held-out metrics ({metrics['count']} items):")
            print(f"    Mean cosine sim:        {metrics['mean_cosine_sim']:.4f}")
            print(f"    Differentiation score:  {metrics['differentiation_score']:.4f}")
            if metrics['top_1_accuracy'] is not None:
                print(f"    Top-1 accuracy:         {metrics['top_1_accuracy']:.4f}")
                print(f"    Top-3 accuracy:         {metrics['top_3_accuracy']:.4f}")
                print(f"    Top-5 accuracy:         {metrics['top_5_accuracy']:.4f}")
            if metrics['per_domain']:
                print(f"    Per-domain:")
                for dom, ddata in sorted(metrics['per_domain'].items()):
                    print(f"      {dom:20s}: cos={ddata['mean_cosine_sim']:.4f}  "
                          f"n={ddata['count']}")
            # Log to metrics_log.jsonl
            try:
                entry = {
                    "stage": stage, "step": step,
                    "timestamp": time.time(),
                    **metrics,
                }
                with open(METRICS_LOG_FILE, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

    print(f"  {'─' * 56}\n")

    # Swap back from EMA to live parameters
    if ema_active:
        try:
            brain.utm_swap_from_ema()
        except (AttributeError, RuntimeError):
            pass

    # Unfreeze decoder after evaluation
    if decoder:
        decoder.unfreeze()


_checkpoint_thread = None

def _cleanup_old_checkpoints(keep_current=True):
    """Remove old checkpoint files to free disk space.

    Keeps the canonical athena_immersive.bin (symlink) and the 2 most recent snapshots.
    Removes all other .bin files and their sidecars.
    """
    if not os.path.isdir(CHECKPOINT_DIR):
        return
    if keep_current:
        _prune_checkpoint_snapshots(max_snapshots=2)
    else:
        _prune_checkpoint_snapshots(max_snapshots=0)


def _check_disk_space(min_gb=5.0):
    """Check available disk space. Warn if below threshold."""
    try:
        st = os.statvfs(CHECKPOINT_DIR if os.path.isdir(CHECKPOINT_DIR) else ".")
        avail_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        if avail_gb < min_gb:
            print(f"  WARNING: Only {avail_gb:.1f} GB disk space remaining!")
            return False
        return True
    except OSError:
        return True


def _register_checkpoint_questdb(ckpt_path, stage, step):
    """Register checkpoint metadata in QuestDB via ILP protocol."""
    try:
        import socket
        blob_size = os.path.getsize(ckpt_path)
        ts_ns = int(time.time() * 1e9)
        line = (
            f'kg_brain_snapshots,'
            f'snapshot_id=ckpt_{stage}_{step},'
            f'brain_id=athena,'
            f'name=athena_immersive '
            f'description="Stage {stage} step {step} checkpoint",'
            f'version=1i,'
            f'format_version=1i,'
            f'blob_path="{ckpt_path}",'
            f'blob_size={blob_size}i,'
            f'neuron_count=2000000i,'
            f'stage={stage}i,'
            f'step={step}i '
            f'{ts_ns}'
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect(('localhost', 9009))
        sock.sendall((line + '\n').encode())
        sock.close()
        logger.info("Checkpoint registered in QuestDB: stage=%d, step=%d", stage, step)
    except Exception as e:
        logger.debug("QuestDB registration skipped: %s", e)


def _prune_checkpoint_snapshots(max_snapshots=5):
    """Keep only the most recent `max_snapshots` timestamped snapshots.

    Deletes oldest snapshots (and their sidecars) when count exceeds limit.
    """
    import glob as glob_mod
    sidecars = ['.meta', '.tokenizer', '.mirror_neurons', '.executive']
    pattern = os.path.join(CHECKPOINT_DIR, "athena_s*_step*.bin")
    snapshots = sorted(glob_mod.glob(pattern))

    # Filter out sidecar matches — only count bare .bin files
    snapshots = [s for s in snapshots if not any(s.endswith(ext) for ext in sidecars)]

    if len(snapshots) <= max_snapshots:
        return

    to_remove = snapshots[:len(snapshots) - max_snapshots]
    freed = 0
    for snap in to_remove:
        for ext in [''] + sidecars:
            f = snap + ext if ext else snap
            if os.path.exists(f):
                try:
                    freed += os.path.getsize(f)
                    os.remove(f)
                except OSError:
                    pass
    if freed > 0:
        logger.info("Pruned %d old snapshots (freed %.1f MB)", len(to_remove), freed / 1e6)


def _save_checkpoint_sync(brain, decoder, stage, step):
    """Synchronous checkpoint save (runs in background thread).

    Saves timestamped snapshots: athena_s{stage}_step{step}.bin
    Also maintains athena_immersive.bin as a symlink to the latest for --resume.
    Keeps up to 5 snapshots, auto-prunes oldest.
    Uses write-to-temp + size validation to prevent corruption.
    """
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Check disk space before saving
    if not _check_disk_space(min_gb=5.0):
        logger.warning("Low disk space — pruning old snapshots")
        _prune_checkpoint_snapshots(max_snapshots=2)
        if not _check_disk_space(min_gb=2.0):
            logger.error("Still insufficient disk space — skipping checkpoint save")
            return

    snapshot_name = f"athena_s{stage}_step{step}.bin"
    snapshot_path = os.path.join(CHECKPOINT_DIR, snapshot_name)
    snapshot_tmp = snapshot_path + ".tmp"
    canonical = os.path.join(CHECKPOINT_DIR, "athena_immersive.bin")
    sidecars = ['.meta', '.tokenizer', '.mirror_neurons', '.executive']

    try:
        # Write to temp file — if interrupted, no existing checkpoint is touched
        brain.save(snapshot_tmp)

        # Verify the temp file is non-trivial (at least 100MB for a 2M neuron brain)
        tmp_size = os.path.getsize(snapshot_tmp)
        if tmp_size < 100_000_000:  # 100 MB minimum
            logger.error("Checkpoint too small (%d bytes) — likely truncated, discarding",
                         tmp_size)
            os.remove(snapshot_tmp)
            return

        # Atomic rename temp -> snapshot
        os.replace(snapshot_tmp, snapshot_path)

        # Update canonical symlink (athena_immersive.bin -> latest snapshot)
        # Use tmp symlink + atomic rename for safety
        canonical_tmp = canonical + ".lnk"
        try:
            if os.path.exists(canonical_tmp):
                os.remove(canonical_tmp)
            os.symlink(snapshot_name, canonical_tmp)
            os.replace(canonical_tmp, canonical)
        except OSError:
            # Fallback: just copy if symlinks fail
            try:
                import shutil
                shutil.copy2(snapshot_path, canonical)
            except Exception:
                pass

        logger.info("Checkpoint snapshot: %s (stage=%d, step=%d, size=%.1f MB)",
                     snapshot_name, stage, step, tmp_size / 1e6)
    except Exception as e:
        logger.warning("Checkpoint save failed: %s", e)
        if os.path.exists(snapshot_tmp):
            try:
                os.remove(snapshot_tmp)
            except OSError:
                pass
        return

    # Save sidecar files alongside the snapshot
    if decoder:
        os.makedirs(DECODER_DIR, exist_ok=True)
        try:
            decoder.save(DECODER_DIR)
        except Exception:
            pass

    state = {"stage": stage, "step": step,
             "snapshot": snapshot_name,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

    # Prune old snapshots (keep 5)
    _prune_checkpoint_snapshots(max_snapshots=5)

    # Register checkpoint in QuestDB
    _register_checkpoint_questdb(snapshot_path, stage, step)


def _save_checkpoint(brain, decoder, stage, step):
    """Non-blocking checkpoint save — runs in background thread."""
    global _checkpoint_thread
    from threading import Thread
    # Wait for previous checkpoint to finish (if still running)
    if _checkpoint_thread is not None and _checkpoint_thread.is_alive():
        _checkpoint_thread.join(timeout=30)
    _checkpoint_thread = Thread(target=_save_checkpoint_sync,
                                args=(brain, decoder, stage, step),
                                daemon=False)
    _checkpoint_thread.start()


def _load_state():
    """Load last training state."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return None


# ============================================================================
def _pre_generate_all_stages(args, force_fresh=False):
    """Check for Claude content cache. Generate if missing via generate_content.py."""
    if os.path.exists(CONTENT_CACHE):
        try:
            with open(CONTENT_CACHE) as f:
                cache = json.load(f)
            if all(str(s) in cache for s in range(4)):
                total = sum(len(v) for d in cache.values() for v in d.values() if isinstance(v, list))
                print(f"  [Claude] Content cache loaded — {total} items across 4 stages")
                return
        except Exception:
            pass

    # Try to generate inline (only works if enough free RAM for claude CLI)
    print("  [Claude] No content cache found. Run 'python3 scripts/generate_content.py' first.")
    print("  [Claude] Attempting inline generation...")
    try:
        import subprocess as _sp
        result = _sp.run(
            [sys.executable, "scripts/generate_content.py", "--force"],
            timeout=600, cwd="/home/bbrelin/nimcp",
            env={k: v for k, v in os.environ.items()
                 if k not in ("CLAUDECODE", "CUDA_VISIBLE_DEVICES")}
        )
        if result.returncode == 0 and os.path.exists(CONTENT_CACHE):
            print("  [Claude] Content generated successfully")
            return
    except Exception as e:
        print(f"  [Claude] Inline generation failed: {e}")
    print("  [Claude] Will use fallback narrations (run generate_content.py separately)")


# Main
# ============================================================================

def _kill_stale_processes():
    """Kill old immerse_athena / training processes and GPU zombies before starting."""
    import subprocess
    my_pid = os.getpid()
    killed = []

    # Kill old immerse_athena.py instances
    try:
        result = subprocess.run(
            ["pgrep", "-f", "immerse_athena.py"],
            capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid and pid != os.getppid():
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(("immerse_athena", pid))
                except ProcessLookupError:
                    pass
    except Exception:
        pass

    # Kill zombie Python processes holding GPU memory
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        for line in result.stdout.strip().splitlines():
            pid = int(line.strip())
            if pid == my_pid:
                continue
            # Check if it's a zombie or orphaned multiprocessing child
            try:
                status_path = f"/proc/{pid}/status"
                with open(status_path) as f:
                    status_text = f.read()
                is_zombie = "State:\tZ" in status_text
                is_orphan_spawn = False
                cmdline_path = f"/proc/{pid}/cmdline"
                with open(cmdline_path) as f:
                    cmdline = f.read()
                is_orphan_spawn = "multiprocessing" in cmdline and "spawn" in cmdline
                if is_zombie or is_orphan_spawn:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(("gpu-zombie", pid))
            except (FileNotFoundError, ProcessLookupError):
                pass
    except Exception:
        pass

    if killed:
        time.sleep(1)  # let resources release
        for label, pid in killed:
            print(f"  [Cleanup] Killed stale {label} process (PID {pid})")


# ---------------------------------------------------------------------------
# IPC Server — allows talk_to_athena.py to query the running brain
# ---------------------------------------------------------------------------

ATHENA_SOCKET_PATH = "/tmp/athena_brain.sock"


class AthenaIPCServer:
    """Unix socket server that exposes the running brain for external queries.

    Protocol: length-prefixed JSON messages.
      Request:  4-byte big-endian length + JSON bytes
      Response: 4-byte big-endian length + JSON bytes

    Supported commands:
      {"cmd": "decide", "text": "..."}           → run decide_full
      {"cmd": "decide", "text": "...", "top_k": 5} → with top-k vocab matches
      {"cmd": "status"}                           → training progress
      {"cmd": "transcript"}                       → last cognitive transcript
      {"cmd": "stats"}                            → brain stats
    """

    def __init__(self, brain, decoder, composer, training_progress):
        self.brain = brain
        self.decoder = decoder
        self.composer = composer
        self.training_progress = training_progress
        self._lock = threading.Lock()
        self._server_sock = None
        self._thread = None
        self._running = False

    def start(self):
        # Clean up stale socket
        if os.path.exists(ATHENA_SOCKET_PATH):
            os.unlink(ATHENA_SOCKET_PATH)

        self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server_sock.bind(ATHENA_SOCKET_PATH)
        self._server_sock.listen(2)
        self._server_sock.settimeout(1.0)  # allow periodic shutdown checks
        self._running = True

        self._thread = threading.Thread(target=self._serve, daemon=True,
                                         name="athena-ipc")
        self._thread.start()
        print(f"  IPC server: listening on {ATHENA_SOCKET_PATH}")

    def stop(self):
        self._running = False
        if self._server_sock:
            self._server_sock.close()
        if os.path.exists(ATHENA_SOCKET_PATH):
            os.unlink(ATHENA_SOCKET_PATH)

    def _serve(self):
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle_connection(conn)
            except Exception as e:
                logging.getLogger("athena_ipc").warning("IPC error: %s", e)
            finally:
                conn.close()

    def _recv_msg(self, conn):
        """Read a length-prefixed message."""
        hdr = b""
        while len(hdr) < 4:
            chunk = conn.recv(4 - len(hdr))
            if not chunk:
                return None
            hdr += chunk
        length = struct.unpack(">I", hdr)[0]
        if length > 10 * 1024 * 1024:  # 10 MB sanity limit
            return None
        data = b""
        while len(data) < length:
            chunk = conn.recv(min(length - len(data), 65536))
            if not chunk:
                return None
            data += chunk
        return json.loads(data.decode("utf-8"))

    def _send_msg(self, conn, obj):
        """Send a length-prefixed JSON message."""
        data = json.dumps(obj).encode("utf-8")
        conn.sendall(struct.pack(">I", len(data)) + data)

    def _handle_connection(self, conn):
        conn.settimeout(30.0)
        req = self._recv_msg(conn)
        if req is None:
            return

        cmd = req.get("cmd", "")

        if cmd == "decide":
            resp = self._handle_decide(req)
        elif cmd == "status":
            resp = self._handle_status()
        elif cmd == "transcript":
            resp = self._handle_transcript()
        elif cmd == "stats":
            resp = self._handle_stats()
        elif cmd == "ping":
            resp = {"ok": True}
        else:
            resp = {"error": f"Unknown command: {cmd}"}

        self._send_msg(conn, resp)

    def _handle_decide(self, req):
        text = req.get("text", "")
        top_k = req.get("top_k", 5)
        if not text:
            return {"error": "No text provided"}

        with self._lock:
            input_emb = encode_text(text)
            features = self.composer.compose(text=text, modality="text")
            result = self.brain.decide_full(features)

        output_vec = result.get("output_vector")
        label = result.get("label", "")
        confidence = result.get("confidence", 0.0)

        resp = {
            "label": label,
            "confidence": confidence,
            "output_size": len(output_vec) if output_vec else 0,
        }

        # Decode through vocabulary
        if output_vec and self.decoder and len(self.decoder.vocabulary) > 0:
            output_arr = np.array(output_vec, dtype=np.float32)
            output_emb = extract_embedding_from_output(output_arr)
            matches = self.decoder.vocabulary.decode(output_emb, top_k=top_k)
            resp["decoded"] = [{"text": t, "similarity": float(s)}
                               for t, s in matches if t]
            if matches and matches[0][0]:
                resp["response"] = matches[0][0]

            # Coherence
            input_emb_raw = encode_text(text)
            coherence = float(np.dot(output_emb, input_emb_raw) /
                            (np.linalg.norm(output_emb) *
                             np.linalg.norm(input_emb_raw) + 1e-8))
            resp["coherence"] = coherence
            resp["output_norm"] = float(np.linalg.norm(output_arr))

        # Try grounded response
        try:
            grounded = self.brain.grounded_respond(text)
            gt = grounded.get("response", "")
            gc = grounded.get("confidence", 0.0)
            if gt and gc >= 0.3:
                resp["grounded_response"] = gt
                resp["grounded_confidence"] = gc
        except Exception:
            pass

        # Try generate_text
        if output_vec:
            try:
                gen = self.brain.generate_text(list(output_vec))
                gt = gen.get("text", "")
                if gt and len(gt.strip()) > 2:
                    resp["generated_text"] = gt
            except Exception:
                pass

        # Transcript
        try:
            transcript = self.brain.get_transcript()
            if transcript:
                resp["transcript"] = transcript
        except Exception:
            pass

        return resp

    def _handle_status(self):
        stage, step = self.training_progress
        return {
            "stage": stage,
            "step": step,
            "vocab_size": len(self.decoder.vocabulary) if self.decoder else 0,
        }

    def _handle_transcript(self):
        try:
            transcript = self.brain.get_transcript()
            return {"transcript": transcript or []}
        except Exception as e:
            return {"error": str(e)}

    def _handle_stats(self):
        try:
            stats = self.brain.get_stats()
            return {"stats": stats}
        except Exception as e:
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Immersive Developmental Learning for Athena")
    parser.add_argument("--stage", type=int, default=0,
                        help="Start from stage (0-3)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--no-claude", action="store_true",
                        help="Disable Claude parent (silent mode)")
    parser.add_argument("--num-inputs", type=int, default=BRAIN_INPUT_DIM)
    parser.add_argument("--num-outputs", type=int, default=BRAIN_OUTPUT_DIM)
    parser.add_argument("--neuron-count", type=int, default=2000000)
    parser.add_argument("--stage0-stimuli", type=int, default=20000)
    parser.add_argument("--stage1-stimuli", type=int, default=40000)
    parser.add_argument("--stage2-stimuli", type=int, default=40000)
    parser.add_argument("--stage3-interactions", type=int, default=30000)
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore existing checkpoint)")
    parser.add_argument("--no-multimodal", action="store_true",
                        help="Disable multimodal dataset download")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

    _kill_stale_processes()

    # --- Load or create brain ---
    checkpoint_path = args.checkpoint
    if args.resume and not checkpoint_path:
        default_ckpt = os.path.join(CHECKPOINT_DIR, "athena_immersive.bin")
        if os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt
    if args.fresh:
        checkpoint_path = None

    print("\n" + "=" * 60)
    print("  ATHENA — Immersive Developmental Learning")
    print("  Every biological subsystem active. No shortcuts.")
    print("=" * 60)

    # Clean up old checkpoints on fresh start to free disk space
    if args.fresh:
        _cleanup_old_checkpoints(keep_current=False)
    else:
        _cleanup_old_checkpoints(keep_current=True)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading from checkpoint: {checkpoint_path}")
    else:
        print("  Creating fresh brain (FULL init — all biological subsystems)")
        checkpoint_path = None

    # --- Pre-generate Claude content BEFORE brain init (brain uses 52GB+) ---
    if not args.no_claude:
        _pre_generate_all_stages(args, force_fresh=args.fresh)

    # --- Download multimodal datasets BEFORE brain init (needs free RAM) ---
    multimodal_loader = MultimodalDataLoader()
    if not args.no_multimodal:
        multimodal_loader.ensure_downloaded()
    else:
        print("  [Multimodal] Disabled (--no-multimodal)")

    # Restore CUDA visibility so nimcp's gpu_detect sees the real GPU.
    # PyTorch is already imported with CUDA disabled; restoring the env
    # var won't make PyTorch re-acquire GPU context.
    if _saved_cuda_vis is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cuda_vis
    elif "CUDA_VISIBLE_DEVICES" in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]

    if checkpoint_path and os.path.exists(checkpoint_path):
        # Resume: load full brain state from checkpoint (avoids 52GB+ fresh init)
        print(f"  Loading brain from checkpoint: {checkpoint_path}")
        brain = nimcp.Brain.load(checkpoint_path)
    else:
        # Fresh start: create brain from scratch (requires ~52GB+ RAM)
        brain = nimcp.Brain("athena",
                            num_inputs=args.num_inputs,
                            num_outputs=args.num_outputs,
                            neuron_count=args.neuron_count,
                            init_mode='full')

    # --- CRITICAL: Disable fast training → ALL bio subsystems active ---
    brain.set_fast_training(False)
    print("  Fast training: OFF (full biological pipeline)")

    # --- CRITICAL: Use regression strategy (raw output, no softmax) ---
    # Classification strategy applies softmax which crushes output to ~1/N per neuron.
    # Athena needs raw output vectors for embedding-based developmental learning.
    brain.set_task_type("regression")
    print("  Task type: REGRESSION (raw output, no softmax)")

    # --- Enable mixed precision (FP16) for ~2x GPU throughput ---
    try:
        brain.enable_mixed_precision(True)
        print("  Mixed precision: ON (FP16 compute, FP32 accumulation)")
    except Exception as e:
        print(f"  Mixed precision: FAILED ({e})")

    # --- Enable gradient checkpointing for memory-efficient training ---
    try:
        brain.enable_gradient_checkpointing(True, 2)
        print("  Gradient checkpointing: ON (every 2 layers)")
    except Exception as e:
        print(f"  Gradient checkpointing: FAILED ({e})")

    # --- Enable biological plasticity ---
    try:
        brain.enable_biological_plasticity(True)
        print("  Biological plasticity: ON (TPB + EDP + coordinator)")
    except Exception as e:
        print(f"  Biological plasticity: FAILED ({e})")

    # --- Enable world model ---
    try:
        brain.enable_world_model(True)
        print("  World model: ON (JEPA + RSSM)")
    except Exception as e:
        print(f"  World model: FAILED ({e})")

    # --- Create LNN temporal processor ---
    try:
        brain.lnn_create(128, 64, 32, 64)
        print("  LNN temporal processor: ON (128→64→32→64)")
    except Exception as e:
        print(f"  LNN: FAILED ({e})")

    # --- Enable multi-network unified training (UTM) ---
    # Registers Adaptive, CNN, SNN, LNN into unified training manager
    # with shared composite loss, cross-network bridges, and optimizer
    try:
        brain.enable_multi_network()
        print("  Multi-network UTM: ON (4 networks, unified optimizer)")
    except Exception as e:
        print(f"  Multi-network UTM: FAILED ({e})")

    # --- Initialize reasoning KB ---
    try:
        brain.ti_init_reasoning()
        print("  Reasoning KB: ON")
    except Exception:
        pass

    # --- Set up components ---
    teacher = None
    if not args.no_claude:
        try:
            teacher = ClaudeTeacher()
            print("  Claude parent: ON")
        except Exception as e:
            print(f"  Claude parent: OFF ({e})")

    # --- Neural decoder (for observing Athena's responses, not scoring them) ---
    decoder = None
    try:
        if os.path.exists(DECODER_DIR) and not args.fresh:
            decoder = NeuralDecoder.load(DECODER_DIR)
            # Verify embed_dim matches current model (1024-dim BAAI/bge-large-en-v1.5)
            if decoder.embed_dim != 1024:
                print(f"  Decoder embed_dim mismatch ({decoder.embed_dim} != 1024), recreating")
                decoder = NeuralDecoder(output_dim=args.num_outputs, embed_dim=1024)
        else:
            decoder = NeuralDecoder(output_dim=args.num_outputs, embed_dim=1024)
        print("  Neural decoder: ON")
    except Exception:
        decoder = NeuralDecoder(output_dim=args.num_outputs, embed_dim=1024)

    parent = Parent(teacher=teacher, enabled=not args.no_claude, decoder=decoder,
                    multimodal=multimodal_loader)
    composer = SensoryComposer(brain)
    clock = BiologicalClock(rest_interval=2000)
    source = StimulusSource()

    # --- Resume state ---
    start_stage = args.stage
    start_step = 0
    if args.resume:
        state = _load_state()
        if state:
            start_stage = state.get("stage", args.stage)
            start_step = state.get("step", 0)
            print(f"  Resuming from stage {start_stage}, step {start_step}")

    # --- Handle graceful shutdown ---
    shutdown_requested = [False]
    # Mutable training progress for signal handlers [stage, step]
    training_progress = [start_stage, 0]

    ipc_server = [None]  # mutable container for signal handler access

    def signal_handler(sig, frame):
        if shutdown_requested[0]:
            print("\n  Force quit.")
            sys.exit(1)
        shutdown_requested[0] = True
        print("\n  Graceful shutdown... saving checkpoint...")
        if ipc_server[0]:
            ipc_server[0].stop()
        _save_checkpoint(brain, decoder, training_progress[0], training_progress[1])
        # Wait for checkpoint save to complete before exiting
        if _checkpoint_thread is not None and _checkpoint_thread.is_alive():
            print("  Waiting for checkpoint save to complete...")
            _checkpoint_thread.join(timeout=600)  # 10 min max for 2M neuron save
            if _checkpoint_thread.is_alive():
                print("  WARNING: Checkpoint save timed out after 600s")
            else:
                print("  Checkpoint saved successfully.")
        sys.exit(0)

    def sigusr1_handler(sig, frame):
        stage, step = training_progress
        print(f"\n  [SIGUSR1] On-demand checkpoint requested (stage={stage}, step={step})...")
        _save_checkpoint(brain, decoder, stage, step)
        print("  [SIGUSR1] Checkpoint save initiated — training continues.")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGUSR1, sigusr1_handler)

    # --- Start IPC server (allows talk_to_athena.py to query this brain) ---
    ipc_server[0] = AthenaIPCServer(brain, decoder, composer, training_progress)
    ipc_server[0].start()

    # --- Print initial bio stats ---
    print("\n  Initial biological state:")
    _print_bio_stats(brain)

    # --- Sensory enrichment (one-time, for existing text-trained brains) ---
    if not args.no_multimodal and not args.fresh:
        run_sensory_enrichment(brain, composer, parent, decoder)

    # --- Run developmental stages ---
    print(f"\n  Starting from stage {start_stage}")

    if start_stage <= 0:
        training_progress[0] = 0
        run_stage_0(brain, composer, parent, clock, source, decoder,
                    num_stimuli=args.stage0_stimuli,
                    start_from=start_step if start_stage == 0 else 0)
        start_step = 0  # Reset for next stage

    if start_stage <= 1:
        training_progress[0], training_progress[1] = 1, 0
        run_stage_1(brain, composer, parent, clock, source, decoder,
                    num_stimuli=args.stage1_stimuli,
                    start_from=start_step if start_stage == 1 else 0)
        start_step = 0

    if start_stage <= 2:
        training_progress[0], training_progress[1] = 2, 0
        stage2_losses = run_stage_2(
            brain, composer, parent, clock, source, decoder,
            num_stimuli=args.stage2_stimuli,
            start_from=start_step if start_stage == 2 else 0)
        if stage2_losses:
            print(f"\n  Stage 2 complete — final mean loss: "
                  f"{np.mean(stage2_losses[-500:]):.4f}")
        start_step = 0

    if start_stage <= 3:
        training_progress[0], training_progress[1] = 3, 0
        run_stage_3(brain, composer, parent, clock, source, decoder,
                    num_interactions=args.stage3_interactions,
                    start_from=start_step if start_stage == 3 else 0)

    # --- Final checkpoint ---
    print("\n" + "=" * 60)
    print("  DEVELOPMENTAL TRAINING COMPLETE")
    print("=" * 60)
    _save_checkpoint(brain, decoder, stage=3, step=-1)

    # Final stats
    print("\n  Final biological state:")
    _print_bio_stats(brain)

    if parent.milestones:
        print(f"\n  Milestones celebrated: {len(parent.milestones)}")
        for m in parent.milestones[:10]:
            print(f"    - {m}")
        if len(parent.milestones) > 10:
            print(f"    ... and {len(parent.milestones) - 10} more")

    if parent.moral_lessons:
        print(f"  Moral lessons taught: {len(parent.moral_lessons)}")

    print(f"  Total interactions: {parent.interaction_count}")
    print()


if __name__ == "__main__":
    main()
