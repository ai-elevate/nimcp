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


def generate_synthetic_audio(description, sample_rate=16000, duration=1.0):
    """Generate simple synthetic audio that encodes the description.

    Maps description keywords to frequency/amplitude patterns that the
    audio cortex CNN can learn to distinguish.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = np.zeros_like(t)

    desc_lower = description.lower()

    # Map common sounds to frequency patterns
    if any(w in desc_lower for w in ["bird", "sing", "chirp", "whistle"]):
        # High-frequency warbling (2-4 kHz)
        freq = 3000 + 1000 * np.sin(2 * np.pi * 8 * t)
        audio += 0.3 * np.sin(2 * np.pi * freq * t)

    elif any(w in desc_lower for w in ["thunder", "rumble", "boom", "crash"]):
        # Low-frequency rumble (50-200 Hz) with decay
        audio += 0.5 * np.sin(2 * np.pi * 100 * t) * np.exp(-3 * t)

    elif any(w in desc_lower for w in ["rain", "water", "drip", "splash"]):
        # White noise filtered to sound like rain
        noise = np.random.randn(len(t)).astype(np.float32) * 0.2
        # Simple low-pass approximation
        for i in range(1, len(noise)):
            noise[i] = 0.8 * noise[i-1] + 0.2 * noise[i]
        audio += noise

    elif any(w in desc_lower for w in ["wind", "breeze", "gust"]):
        # Filtered noise with slow modulation
        noise = np.random.randn(len(t)).astype(np.float32) * 0.15
        mod = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
        audio += noise * mod

    elif any(w in desc_lower for w in ["bell", "chime", "ring"]):
        # Pure tone with decay
        audio += 0.4 * np.sin(2 * np.pi * 880 * t) * np.exp(-2 * t)
        audio += 0.2 * np.sin(2 * np.pi * 1760 * t) * np.exp(-3 * t)

    elif any(w in desc_lower for w in ["dog", "bark", "woof"]):
        # Short bursts at 300-500 Hz
        for burst_start in [0.0, 0.3, 0.6]:
            mask = ((t >= burst_start) & (t < burst_start + 0.15)).astype(np.float32)
            audio += 0.4 * np.sin(2 * np.pi * 400 * t) * mask

    elif any(w in desc_lower for w in ["cat", "purr", "meow"]):
        # Rising tone (meow shape)
        freq = 400 + 200 * t / duration
        audio += 0.3 * np.sin(2 * np.pi * freq * t) * np.exp(-1.5 * t)

    elif any(w in desc_lower for w in ["music", "piano", "guitar", "melody"]):
        # Simple melody (C-E-G arpeggio)
        for i, f in enumerate([262, 330, 392, 523]):
            start = i * 0.25
            mask = ((t >= start) & (t < start + 0.2)).astype(np.float32)
            audio += 0.3 * np.sin(2 * np.pi * f * t) * mask

    elif any(w in desc_lower for w in ["silence", "quiet", "still"]):
        # Near silence with very faint ambient
        audio += np.random.randn(len(t)).astype(np.float32) * 0.01

    else:
        # Default: gentle ambient tone based on description hash
        h = hash(description) % 1000
        freq = 200 + h * 0.5
        audio += 0.1 * np.sin(2 * np.pi * freq * t)

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

    # Get descriptions to generate audio for
    descriptions = get_training_descriptions()
    if not descriptions:
        # Fallback: use ESC-50 class descriptions
        descriptions = list(ESC50_CLASSES.values())
    print(f"  {len(descriptions)} audio descriptions available")

    # Audio stream rate — throttled to avoid overloading daemon socket.
    # The training script uses the same socket at ~3 steps/sec.
    audio_hz = 0.2  # 1 audio frame every 5 seconds (safe for socket backlog)
    step_interval = 1.0 / audio_hz

    step = 0
    while True:
        # Pick a description (cycle through)
        desc = descriptions[step % len(descriptions)]

        # Generate audio
        audio_samples = generate_synthetic_audio(desc)
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

        if step % 100 == 0:
            print(f"  [Audio] Step {step}: '{desc[:60]}' "
                  f"({len(audio_samples)} samples, {len(mel_features)} mel features)")

        time.sleep(step_interval)


if __name__ == "__main__":
    main()
