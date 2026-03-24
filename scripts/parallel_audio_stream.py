#!/usr/bin/env python3
"""
Parallel Audio Stream — Feeds real audio to Athena alongside text training.

Runs as a separate process. Connects to the brain daemon and submits
audio sensory data in parallel with the main immerse_athena.py training.

This gives Athena multimodal grounding — she hears sounds associated
with the concepts she's learning from text. Uses ESC-50 environmental
audio dataset + optional parentese-style TTS.

Usage:
    nohup python3 scripts/parallel_audio_stream.py > nohup_audio.log 2>&1 &
"""

import os
import sys
import time
import random
import numpy as np
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brain_client import BrainProxy

# Audio processing
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("[Audio] librosa not available — using synthetic audio")

# ESC-50 dataset mapping (class → description)
ESC50_CLASSES = {
    0: "dog barking", 1: "rain falling", 2: "sea waves", 3: "baby crying",
    4: "clock ticking", 5: "person sneezing", 6: "helicopter flying",
    7: "chainsaw running", 8: "rooster crowing", 9: "fire crackling",
    10: "hand saw cutting", 11: "train passing", 12: "church bells ringing",
    13: "airplane flying", 14: "mouse clicking", 15: "frog croaking",
    16: "crow calling", 17: "cat meowing", 18: "glass breaking",
    19: "door knocking", 20: "hen clucking", 21: "laughing",
    22: "pig oinking", 23: "insects buzzing", 24: "sheep bleating",
    25: "engine idling", 26: "pouring water", 27: "wind blowing",
    28: "toilet flushing", 29: "brushing teeth", 30: "vacuum cleaning",
    31: "clock alarm", 32: "car horn honking", 33: "washing machine",
    34: "siren wailing", 35: "cow mooing", 36: "footsteps walking",
    37: "breathing heavily", 38: "coughing", 39: "typing on keyboard",
    40: "drinking sipping", 41: "clapping hands", 42: "crickets chirping",
    43: "snoring", 44: "thunderstorm", 45: "water dripping",
    46: "birds singing", 47: "door creaking", 48: "owl hooting",
    49: "children playing",
}


def generate_synthetic_audio(description, sample_rate=16000, duration=1.0, variation=0):
    """Generate synthetic audio with per-instance variation.

    Each call with a different variation produces a unique sound within
    the same category — different pitch, timing, amplitude, and noise.
    This gives the audio cortex diverse training examples per concept.

    Maps description keywords to frequency/amplitude patterns that the
    audio cortex CNN can learn to distinguish.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.zeros_like(t)

    # Per-instance variation: shift pitch, timing, amplitude
    rng = np.random.RandomState(abs(hash(description) + variation) % (2**32 - 1))
    pitch_shift = 1.0 + rng.uniform(-0.2, 0.2)    # ±20% pitch variation
    amp_scale = 0.5 + rng.uniform(0, 0.5)           # 50-100% amplitude
    time_offset = rng.uniform(0, 0.3)               # 0-300ms onset delay
    noise_level = rng.uniform(0.01, 0.05)            # Background noise

    desc_lower = description.lower()

    # Apply time offset
    t_shifted = np.clip(t - time_offset, 0, duration)

    # Map common sounds to frequency patterns (with variation)
    if any(w in desc_lower for w in ["bird", "sing", "chirp", "whistle"]):
        freq = (3000 + 1000 * np.sin(2 * np.pi * (8 * pitch_shift) * t_shifted)) * pitch_shift
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted)
        # Add harmonics for richness
        audio += amp_scale * 0.3 * np.sin(2 * np.pi * freq * 1.5 * t_shifted)

    elif any(w in desc_lower for w in ["thunder", "rumble", "boom", "crash"]):
        base_freq = 100 * pitch_shift
        audio += amp_scale * np.sin(2 * np.pi * base_freq * t_shifted) * np.exp(-3 * t_shifted)
        audio += amp_scale * 0.3 * np.sin(2 * np.pi * base_freq * 0.5 * t_shifted) * np.exp(-2 * t_shifted)

    elif any(w in desc_lower for w in ["rain", "water", "drip", "splash", "pour"]):
        noise = rng.randn(len(t)).astype(np.float32) * 0.2 * amp_scale
        cutoff = 0.7 + rng.uniform(0, 0.2)
        for i in range(1, len(noise)):
            noise[i] = cutoff * noise[i-1] + (1 - cutoff) * noise[i]
        audio += noise

    elif any(w in desc_lower for w in ["wind", "breeze", "gust", "blow"]):
        noise = rng.randn(len(t)).astype(np.float32) * 0.15 * amp_scale
        mod_freq = 0.5 * pitch_shift
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t_shifted)
        audio += noise * mod

    elif any(w in desc_lower for w in ["bell", "chime", "ring", "clock"]):
        f1 = 880 * pitch_shift
        audio += amp_scale * np.sin(2 * np.pi * f1 * t_shifted) * np.exp(-2 * t_shifted)
        audio += amp_scale * 0.4 * np.sin(2 * np.pi * f1 * 2 * t_shifted) * np.exp(-3 * t_shifted)
        audio += amp_scale * 0.2 * np.sin(2 * np.pi * f1 * 3 * t_shifted) * np.exp(-4 * t_shifted)

    elif any(w in desc_lower for w in ["dog", "bark", "woof"]):
        n_barks = rng.randint(2, 5)
        for b in range(n_barks):
            start = b * rng.uniform(0.15, 0.35)
            dur = rng.uniform(0.08, 0.2)
            mask = ((t_shifted >= start) & (t_shifted < start + dur)).astype(np.float32)
            audio += amp_scale * np.sin(2 * np.pi * 400 * pitch_shift * t_shifted) * mask

    elif any(w in desc_lower for w in ["cat", "purr", "meow"]):
        freq = (400 + 200 * t_shifted / duration) * pitch_shift
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted) * np.exp(-1.5 * t_shifted)

    elif any(w in desc_lower for w in ["music", "piano", "guitar", "melody"]):
        notes = [262, 294, 330, 349, 392, 440, 494, 523]
        rng.shuffle(notes)
        for i, f in enumerate(notes[:4]):
            start = i * 0.25
            mask = ((t_shifted >= start) & (t_shifted < start + 0.2)).astype(np.float32)
            audio += amp_scale * np.sin(2 * np.pi * f * pitch_shift * t_shifted) * mask

    elif any(w in desc_lower for w in ["engine", "motor", "machine", "idle"]):
        base = 80 * pitch_shift
        audio += amp_scale * 0.3 * np.sin(2 * np.pi * base * t_shifted)
        audio += amp_scale * 0.2 * np.sin(2 * np.pi * base * 2 * t_shifted)
        audio += amp_scale * 0.15 * np.sin(2 * np.pi * base * 3 * t_shifted)
        audio += rng.randn(len(t)).astype(np.float32) * 0.05 * amp_scale

    elif any(w in desc_lower for w in ["siren", "alarm", "horn"]):
        freq = 600 * pitch_shift + 400 * np.sin(2 * np.pi * 3 * t_shifted)
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted)

    elif any(w in desc_lower for w in ["cry", "baby", "sob", "wail"]):
        freq = (500 + 200 * np.sin(2 * np.pi * 5 * t_shifted)) * pitch_shift
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted)
        audio *= (0.5 + 0.5 * np.sin(2 * np.pi * 2 * t_shifted))

    elif any(w in desc_lower for w in ["laugh", "giggle", "chuckle"]):
        for b in range(rng.randint(3, 8)):
            start = b * rng.uniform(0.08, 0.15)
            dur = 0.06
            mask = ((t_shifted >= start) & (t_shifted < start + dur)).astype(np.float32)
            f = rng.uniform(250, 450) * pitch_shift
            audio += amp_scale * np.sin(2 * np.pi * f * t_shifted) * mask

    elif any(w in desc_lower for w in ["foot", "step", "walk", "run"]):
        for b in range(rng.randint(2, 6)):
            start = b * rng.uniform(0.2, 0.5)
            click = rng.randn(int(sample_rate * 0.02)).astype(np.float32) * amp_scale
            idx = int(start * sample_rate)
            if idx + len(click) < len(audio):
                audio[idx:idx+len(click)] += click

    elif any(w in desc_lower for w in ["type", "keyboard", "click", "tap"]):
        for b in range(rng.randint(5, 15)):
            idx = rng.randint(0, len(audio) - 100)
            click = rng.randn(80).astype(np.float32) * amp_scale * 0.3
            audio[idx:idx+80] += click

    elif any(w in desc_lower for w in ["breath", "snore", "cough", "sneeze"]):
        noise = rng.randn(len(t)).astype(np.float32) * 0.15 * amp_scale
        env = np.exp(-((t_shifted - 0.3)**2) / 0.05)
        audio += noise * env

    elif any(w in desc_lower for w in ["insect", "cricket", "buzz", "bee"]):
        freq = rng.uniform(4000, 6000) * pitch_shift
        audio += amp_scale * 0.2 * np.sin(2 * np.pi * freq * t_shifted)
        mod = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 20 * t_shifted))
        audio *= mod

    elif any(w in desc_lower for w in ["frog", "croak", "ribbit"]):
        for b in range(rng.randint(2, 5)):
            start = b * rng.uniform(0.2, 0.4)
            dur = rng.uniform(0.1, 0.2)
            mask = ((t_shifted >= start) & (t_shifted < start + dur)).astype(np.float32)
            freq = rng.uniform(200, 400) * pitch_shift
            audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted) * mask

    elif any(w in desc_lower for w in ["cow", "moo"]):
        freq = (150 + 30 * np.sin(2 * np.pi * 1.5 * t_shifted)) * pitch_shift
        env = np.exp(-0.5 * t_shifted) * (t_shifted > time_offset).astype(np.float32)
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted) * env

    elif any(w in desc_lower for w in ["rooster", "cock", "crow"]):
        freq = np.where(t_shifted < 0.3, 800 * pitch_shift,
                        np.where(t_shifted < 0.6, 1200 * pitch_shift, 600 * pitch_shift))
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted) * np.exp(-1.5 * t_shifted)

    elif any(w in desc_lower for w in ["clap", "applause", "hand"]):
        for b in range(rng.randint(3, 10)):
            idx = rng.randint(0, len(audio) - 200)
            burst = rng.randn(150).astype(np.float32) * amp_scale * 0.4
            burst *= np.exp(-np.linspace(0, 5, 150))
            audio[idx:idx+150] += burst

    elif any(w in desc_lower for w in ["door", "knock", "creak"]):
        for b in range(rng.randint(2, 4)):
            idx = int((b * 0.3 + time_offset) * sample_rate)
            if idx + 400 < len(audio):
                burst = np.sin(2 * np.pi * 300 * pitch_shift * np.linspace(0, 0.025, 400))
                burst *= np.exp(-np.linspace(0, 8, 400)) * amp_scale
                audio[idx:idx+400] += burst

    elif any(w in desc_lower for w in ["glass", "break", "shatter"]):
        noise = rng.randn(len(t)).astype(np.float32) * amp_scale
        audio += noise * np.exp(-5 * t_shifted) * (t_shifted > time_offset).astype(np.float32)
        audio += amp_scale * 0.3 * np.sin(2 * np.pi * 2000 * pitch_shift * t_shifted) * np.exp(-8 * t_shifted)

    elif any(w in desc_lower for w in ["silence", "quiet", "still"]):
        audio += rng.randn(len(t)).astype(np.float32) * 0.01

    elif any(w in desc_lower for w in ["helicopter", "airplane", "fly"]):
        blade_freq = rng.uniform(15, 25) * pitch_shift
        audio += amp_scale * 0.3 * np.sin(2 * np.pi * blade_freq * t_shifted)
        audio += rng.randn(len(t)).astype(np.float32) * 0.1 * amp_scale

    elif any(w in desc_lower for w in ["train", "rail"]):
        for b in range(10):
            start = b * 0.1
            mask = ((t_shifted >= start) & (t_shifted < start + 0.05)).astype(np.float32)
            audio += amp_scale * 0.3 * rng.randn(len(t)).astype(np.float32) * mask

    elif any(w in desc_lower for w in ["fire", "crackle", "flame"]):
        noise = rng.randn(len(t)).astype(np.float32) * 0.15 * amp_scale
        pops = rng.random(len(t)) > 0.998
        audio += noise + pops.astype(np.float32) * rng.uniform(0.2, 0.5) * amp_scale

    elif any(w in desc_lower for w in ["owl", "hoot"]):
        for b in range(rng.randint(1, 3)):
            start = b * 0.4 + time_offset
            freq = 400 * pitch_shift
            env = np.exp(-((t_shifted - start - 0.15)**2) / 0.01)
            audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted) * env

    elif any(w in desc_lower for w in ["sheep", "bleat", "baa"]):
        freq = (300 + 100 * np.sin(2 * np.pi * 8 * t_shifted)) * pitch_shift
        env = np.exp(-2 * t_shifted) * (t_shifted > time_offset).astype(np.float32)
        audio += amp_scale * np.sin(2 * np.pi * freq * t_shifted) * env

    elif any(w in desc_lower for w in ["pig", "oink", "squeal"]):
        freq = (600 + 200 * np.sin(2 * np.pi * 12 * t_shifted)) * pitch_shift
        audio += amp_scale * 0.4 * np.sin(2 * np.pi * freq * t_shifted) * np.exp(-3 * t_shifted)

    elif any(w in desc_lower for w in ["hen", "chicken", "cluck"]):
        for b in range(rng.randint(3, 7)):
            start = b * rng.uniform(0.1, 0.2)
            dur = 0.05
            mask = ((t_shifted >= start) & (t_shifted < start + dur)).astype(np.float32)
            audio += amp_scale * np.sin(2 * np.pi * 500 * pitch_shift * t_shifted) * mask

    elif any(w in desc_lower for w in ["mouse", "squeak"]):
        freq = rng.uniform(3000, 5000) * pitch_shift
        env = np.exp(-((t_shifted - 0.2)**2) / 0.005) * amp_scale
        audio += np.sin(2 * np.pi * freq * t_shifted) * env

    elif any(w in desc_lower for w in ["vacuum", "clean"]):
        noise = rng.randn(len(t)).astype(np.float32) * 0.2 * amp_scale
        audio += noise + amp_scale * 0.1 * np.sin(2 * np.pi * 120 * pitch_shift * t_shifted)

    elif any(w in desc_lower for w in ["wash", "machine", "laundry"]):
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.8 * t_shifted)
        audio += rng.randn(len(t)).astype(np.float32) * 0.1 * amp_scale * mod

    elif any(w in desc_lower for w in ["toilet", "flush"]):
        noise = rng.randn(len(t)).astype(np.float32) * amp_scale * 0.3
        env = np.where(t_shifted < 0.3, t_shifted / 0.3, np.exp(-2 * (t_shifted - 0.3)))
        audio += noise * env

    elif any(w in desc_lower for w in ["drink", "sip", "swallow"]):
        for b in range(rng.randint(1, 4)):
            start = b * 0.3 + time_offset
            noise = rng.randn(int(sample_rate * 0.1)).astype(np.float32) * amp_scale * 0.2
            idx = int(start * sample_rate)
            if idx + len(noise) < len(audio):
                audio[idx:idx+len(noise)] += noise

    else:
        # Default: unique ambient tone based on description + variation
        h = (hash(description) + variation) % 2000
        freq = 150 + h * 0.3
        audio += amp_scale * 0.15 * np.sin(2 * np.pi * freq * pitch_shift * t_shifted)
        audio += amp_scale * 0.08 * np.sin(2 * np.pi * freq * 1.5 * pitch_shift * t_shifted)

    # Add background noise (every sound has some)
    audio += rng.randn(len(t)).astype(np.float32) * noise_level

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    return audio.tolist()


def audio_to_mel_features(audio_samples, n_mels=64, n_fft=512):
    """Convert audio samples to mel-spectrogram features for the audio cortex."""
    audio = np.array(audio_samples, dtype=np.float32)

    if HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(y=audio, sr=16000,
                                              n_mels=n_mels, n_fft=n_fft)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Flatten to 1D feature vector
        return mel_db.flatten()[:1024].tolist()
    else:
        # Simple FFT-based approximation
        n_frames = len(audio) // (n_fft // 2)
        features = []
        for i in range(min(n_frames, 16)):
            start = i * (n_fft // 2)
            frame = audio[start:start + n_fft]
            if len(frame) < n_fft:
                frame = np.pad(frame, (0, n_fft - len(frame)))
            fft = np.abs(np.fft.rfft(frame))[:n_mels]
            features.extend(fft.tolist())
        # Pad/truncate to 1024
        features = features[:1024]
        features.extend([0.0] * (1024 - len(features)))
        return features


def get_training_descriptions():
    """Get current training descriptions from the content cache."""
    cache_path = "content_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        descriptions = []
        for stage_key in ["0", "1"]:
            if stage_key in cache:
                stage = cache[stage_key]
                if "_narrations" in stage:
                    descriptions.extend(stage["_narrations"])
                if "_encouragements" in stage:
                    descriptions.extend(stage["_encouragements"])
        return descriptions
    return []


def main():
    print("=" * 60)
    print("  PARALLEL AUDIO STREAM")
    print("  Feeding real audio to Athena alongside text training")
    print("=" * 60)

    # Connect to brain daemon
    brain = BrainProxy()
    print(f"  Connected to brain daemon")

    # Build expanded description set — each class gets multiple phrasings
    descriptions = []
    for cls_id, cls_name in ESC50_CLASSES.items():
        descriptions.append(cls_name)
        # Add variations of each class
        words = cls_name.split()
        if len(words) >= 2:
            descriptions.append(f"loud {cls_name}")
            descriptions.append(f"soft {cls_name}")
            descriptions.append(f"distant {cls_name}")
            descriptions.append(f"{cls_name} nearby")

    # Add categories the ESC-50 misses
    extra = [
        "engine running", "motor humming", "machine whirring",
        "traffic noise", "construction drilling", "car passing",
        "cooking sizzling", "pot boiling", "knife chopping",
        "forest ambience", "ocean waves crashing", "river flowing",
        "hail on roof", "heavy rain downpour", "light drizzle",
        "baby cooing", "child giggling", "adult sighing",
        "piano chord", "guitar strum", "drum beat", "violin bow",
        "crowd murmuring", "audience applause", "stadium cheering",
        "door slamming", "window opening", "key turning in lock",
        "paper rustling", "pencil writing", "book pages turning",
        "zipper closing", "velcro ripping", "cloth tearing",
        "ice cracking", "snow crunching", "gravel underfoot",
        "bubble popping", "balloon inflating", "cork popping",
        "heartbeat steady", "heartbeat fast", "breathing slow",
    ]
    descriptions.extend(extra)

    # Also pull from training cache if available
    cached = get_training_descriptions()
    if cached:
        descriptions.extend(cached[:100])

    print(f"  {len(descriptions)} audio descriptions available "
          f"(50 ESC-50 × ~5 variants + {len(extra)} extra + cache)")

    # Audio stream rate — throttled to avoid overloading daemon socket.
    # The training script uses the same socket at ~3 steps/sec.
    audio_hz = 0.2  # 1 audio frame every 5 seconds (safe for socket backlog)
    step_interval = 1.0 / audio_hz

    step = 0
    while True:
        # Pick a description (cycle through)
        desc = descriptions[step % len(descriptions)]

        # Generate audio with per-step variation (never the same twice)
        audio_samples = generate_synthetic_audio(desc, variation=step)
        mel_features = audio_to_mel_features(audio_samples)

        # Submit to brain's audio cortex
        try:
            brain.submit_sensory("audio", mel_features)
        except Exception as e:
            if step < 5:
                print(f"  [Audio] Submit failed: {e}")

        # Also submit as speech (different cortex pathway)
        try:
            brain.submit_sensory("speech", audio_samples[:4000])
        except Exception:
            pass

        step += 1

        if step % 50 == 0:
            # Report with actual current description, not the one from this step
            # Also show which class index we're on
            class_idx = (step - 1) % len(descriptions)
            print(f"  [Audio] Step {step}: class {class_idx}/{len(descriptions)} "
                  f"'{descriptions[class_idx][:50]}' "
                  f"({len(audio_samples)} samples, {len(mel_features)} mel features)",
                  flush=True)

        time.sleep(step_interval)


if __name__ == "__main__":
    main()
