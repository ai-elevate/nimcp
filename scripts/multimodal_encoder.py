#!/usr/bin/env python3
"""Multimodal encoder for Athena's perception training.

Encodes images, audio, speech, and text into the brain's 1024-dim input layout:

    [modality_tag:16 | primary_encoder:512 | text_semantic:384 | low_level_stats:112]

     [0:16]     Modality one-hot (4 bits: text/visual/audio/speech) + 12 metadata
     [16:528]   Primary encoder (CLIP/CLAP/Whisper/sentence-transformer, padded to 512)
     [528:912]  Sentence-transformer text embedding (384-dim, always present)
     [912:1024] Low-level statistics (pixel stats, audio RMS, spectral centroid, etc.)

Uses pre-trained encoders with lazy loading and graceful fallback:
  - CLIP (open-clip-torch) for visual encoding
  - CLAP (laion-clap) for audio encoding
  - Whisper (openai-whisper) for speech-to-text + encoding
  - sentence-transformers for text (always available)

All outputs normalized to [0,1] for brain input compatibility.
"""

import logging
import warnings
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("multimodal_encoder")

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

BRAIN_INPUT_DIM = 1024
TAG_DIM = 16          # [0:16]
PRIMARY_DIM = 512     # [16:528]
TEXT_DIM = 384        # [528:912]
STATS_DIM = 112       # [912:1024]

TAG_START = 0
PRIMARY_START = TAG_DIM
TEXT_START = PRIMARY_START + PRIMARY_DIM
STATS_START = TEXT_START + TEXT_DIM
assert STATS_START + STATS_DIM == BRAIN_INPUT_DIM


class ModalityType(IntEnum):
    TEXT = 0
    VISUAL = 1
    AUDIO = 2
    SPEECH = 3


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class MultimodalEncoder:
    """Encodes multimodal inputs into 1024-dim brain input vectors.

    Lazy-loads heavy models (CLIP, CLAP, Whisper) on first use.
    Falls back to handcrafted features if encoders are not installed.
    """

    def __init__(self, device: str = "cpu"):
        self.device = device

        # Lazy-loaded models (None = not yet attempted)
        self._text_model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_tokenizer = None
        self._clap_model = None
        self._whisper_model = None

        # Availability flags (None = unknown, True/False after first attempt)
        self._clip_available: Optional[bool] = None
        self._clap_available: Optional[bool] = None
        self._whisper_available: Optional[bool] = None

    # --- Text model (always available) ---

    def _get_text_model(self):
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer
            self._text_model = SentenceTransformer("all-MiniLM-L6-v2",
                                                    device=self.device)
            logger.info("Loaded sentence-transformer (all-MiniLM-L6-v2)")
        return self._text_model

    def _encode_text_embedding(self, text: str) -> np.ndarray:
        """Get 384-dim sentence-transformer embedding."""
        model = self._get_text_model()
        emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32).ravel()

    # --- CLIP (visual) ---

    def _load_clip(self) -> bool:
        if self._clip_available is not None:
            return self._clip_available
        try:
            import open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="laion2b_s34b_b79k", device=self.device
            )
            model.eval()
            self._clip_model = model
            self._clip_preprocess = preprocess
            self._clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._clip_available = True
            logger.info("Loaded CLIP (ViT-B-32, laion2b)")
        except (ImportError, Exception) as e:
            logger.warning("CLIP not available, using pixel-stats fallback: %s", e)
            self._clip_available = False
        return self._clip_available

    def _clip_encode_image(self, pil_image) -> np.ndarray:
        """Encode PIL image → 512-dim CLIP features."""
        import torch
        img_tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._clip_model.encode_image(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        out = features.squeeze().cpu().numpy().astype(np.float32)
        return _resize_to(out, PRIMARY_DIM)

    # --- CLAP (audio) ---

    def _load_clap(self) -> bool:
        if self._clap_available is not None:
            return self._clap_available
        try:
            import laion_clap
            self._clap_model = laion_clap.CLAP_Module(enable_fusion=False)
            self._clap_model.load_ckpt()
            self._clap_available = True
            logger.info("Loaded CLAP (laion-clap)")
        except (ImportError, Exception) as e:
            logger.warning("CLAP not available, using MFCC fallback: %s", e)
            self._clap_available = False
        return self._clap_available

    def _clap_encode_audio(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Encode audio samples → 512-dim CLAP features."""
        import torch
        # CLAP expects 48kHz mono
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if sr != 48000:
            samples = _resample(samples, sr, 48000)
        # Pad/truncate to at least 1 second
        min_len = 48000
        if len(samples) < min_len:
            samples = np.pad(samples, (0, min_len - len(samples)))
        audio_tensor = torch.from_numpy(samples).unsqueeze(0)
        with torch.no_grad():
            features = self._clap_model.get_audio_embedding_from_data(
                x=audio_tensor, use_tensor=True
            )
        out = features.squeeze().cpu().numpy().astype(np.float32)
        return _resize_to(out, PRIMARY_DIM)

    # --- Whisper (speech) ---

    def _load_whisper(self) -> bool:
        if self._whisper_available is not None:
            return self._whisper_available
        try:
            import whisper
            self._whisper_model = whisper.load_model("base", device=self.device)
            self._whisper_available = True
            logger.info("Loaded Whisper (base)")
        except (ImportError, Exception) as e:
            logger.warning("Whisper not available, using MFCC fallback: %s", e)
            self._whisper_available = False
        return self._whisper_available

    def _whisper_encode_speech(self, samples: np.ndarray, sr: int
                                ) -> Tuple[np.ndarray, str]:
        """Encode speech → (512-dim features, transcription text)."""
        import whisper
        import torch
        samples = np.asarray(samples, dtype=np.float32).ravel()
        if sr != 16000:
            samples = _resample(samples, sr, 16000)
        # Pad/trim to 30s
        audio = whisper.pad_or_trim(samples)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        # Get encoder features
        with torch.no_grad():
            enc_out = self._whisper_model.encoder(mel.unsqueeze(0))
        # Mean-pool encoder output → feature vector
        features = enc_out.squeeze(0).mean(dim=0).cpu().numpy().astype(np.float32)
        features = _resize_to(features, PRIMARY_DIM)
        # Transcribe
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self._whisper_model.transcribe(samples, fp16=False)
        text = result.get("text", "").strip()
        return features, text

    # --- Public API ---

    def encode_text(self, text: str) -> Tuple[list, list, str]:
        """Encode text → (features_1024, target_384, text).

        For text-only: primary = sentence-transformer padded to 512,
        text_semantic = same 384-dim embedding.
        """
        emb = self._encode_text_embedding(text)
        primary = _resize_to(emb, PRIMARY_DIM)
        features = self.compose_input_features(ModalityType.TEXT, primary, emb)
        target = emb.tolist()
        return features, target, text

    def encode_visual(self, pil_image, caption: Optional[str] = None
                       ) -> Tuple[list, list, str]:
        """Encode PIL image → (features_1024, target_384, caption).

        Uses CLIP encoder if available, else pixel-stats fallback.
        Target embedding always comes from text caption via sentence-transformer.
        """
        if caption is None:
            caption = "an image"

        # Primary encoder
        if self._load_clip():
            primary = self._clip_encode_image(pil_image)
        else:
            primary = self._visual_fallback(pil_image)

        # Text semantic (from caption)
        text_emb = self._encode_text_embedding(caption)

        # Low-level stats
        stats = _image_stats(pil_image)

        features = self.compose_input_features(ModalityType.VISUAL, primary,
                                                text_emb, stats)
        target = text_emb.tolist()
        return features, target, caption

    def encode_audio(self, samples: np.ndarray, sr: int,
                      caption: Optional[str] = None
                      ) -> Tuple[list, list, str]:
        """Encode audio → (features_1024, target_384, caption).

        Uses CLAP encoder if available, else MFCC/spectral fallback.
        """
        if caption is None:
            caption = "a sound"

        if self._load_clap():
            primary = self._clap_encode_audio(samples, sr)
        else:
            primary = self._audio_fallback(samples, sr)

        text_emb = self._encode_text_embedding(caption)
        stats = _audio_stats(samples, sr)

        features = self.compose_input_features(ModalityType.AUDIO, primary,
                                                text_emb, stats)
        target = text_emb.tolist()
        return features, target, caption

    def encode_speech(self, samples: np.ndarray, sr: int
                       ) -> Tuple[list, list, str]:
        """Encode speech → (features_1024, target_384, transcription).

        Uses Whisper for both encoding and transcription if available,
        else MFCC fallback with empty transcription.
        """
        if self._load_whisper():
            primary, transcription = self._whisper_encode_speech(samples, sr)
        else:
            primary = self._audio_fallback(samples, sr)
            transcription = ""

        if not transcription:
            transcription = "speech audio"
        text_emb = self._encode_text_embedding(transcription)
        stats = _audio_stats(samples, sr)

        features = self.compose_input_features(ModalityType.SPEECH, primary,
                                                text_emb, stats)
        target = text_emb.tolist()
        return features, target, transcription

    def compose_input_features(self, modality: ModalityType,
                                primary_emb: np.ndarray,
                                text_emb: np.ndarray,
                                stats: Optional[np.ndarray] = None
                                ) -> list:
        """Compose the 1024-dim brain input from components.

        Layout: [tag:16 | primary:512 | text:384 | stats:112]
        All values normalized to [0,1].
        """
        out = np.zeros(BRAIN_INPUT_DIM, dtype=np.float32)

        # Modality tag: one-hot in first 4 bits, metadata in rest
        tag = np.zeros(TAG_DIM, dtype=np.float32)
        tag[int(modality)] = 1.0
        # Metadata: bit 4 = has_primary (always 1), bit 5 = has_text
        tag[4] = 1.0
        tag[5] = 1.0 if text_emb is not None and len(text_emb) > 0 else 0.0
        # bit 6 = has_stats
        tag[6] = 1.0 if stats is not None and len(stats) > 0 else 0.0
        out[TAG_START:TAG_START + TAG_DIM] = tag

        # Primary encoder output (512-dim, normalized to [0,1])
        primary = np.asarray(primary_emb, dtype=np.float32).ravel()
        primary = _normalize_to_01(primary)
        primary = _resize_to(primary, PRIMARY_DIM)
        out[PRIMARY_START:PRIMARY_START + PRIMARY_DIM] = primary

        # Text semantic (384-dim, normalized to [0,1])
        text = np.asarray(text_emb, dtype=np.float32).ravel()
        text = _normalize_to_01(text)
        text = _resize_to(text, TEXT_DIM)
        out[TEXT_START:TEXT_START + TEXT_DIM] = text

        # Low-level stats (112-dim, already in [0,1])
        if stats is not None:
            s = np.asarray(stats, dtype=np.float32).ravel()
            s = _resize_to(s, STATS_DIM)
            out[STATS_START:STATS_START + STATS_DIM] = np.clip(s, 0.0, 1.0)

        return out.tolist()

    # --- Fallback encoders ---

    def _visual_fallback(self, pil_image) -> np.ndarray:
        """Extract handcrafted visual features when CLIP is not available.

        Computes: color histograms, spatial means, edge density, texture.
        """
        img = pil_image.convert("RGB")
        pixels = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        h, w, c = pixels.shape

        features = []
        # Per-channel histograms (16 bins x 3 channels = 48)
        for ch in range(3):
            hist, _ = np.histogram(pixels[:, :, ch], bins=16, range=(0, 1))
            hist = hist.astype(np.float32) / (h * w)
            features.extend(hist.tolist())

        # Spatial means (divide into 4x4 grid = 48 features)
        gh, gw = max(1, h // 4), max(1, w // 4)
        for gy in range(4):
            for gx in range(4):
                patch = pixels[gy*gh:(gy+1)*gh, gx*gw:(gx+1)*gw]
                for ch in range(3):
                    features.append(float(patch[:, :, ch].mean()))

        # Global stats (mean, std, min, max per channel = 12)
        for ch in range(3):
            chan = pixels[:, :, ch]
            features.extend([
                float(chan.mean()), float(chan.std()),
                float(chan.min()), float(chan.max()),
            ])

        # Edge density (simple Sobel approximation, 1 feature)
        gray = pixels.mean(axis=2)
        dx = np.abs(np.diff(gray, axis=1)).mean()
        dy = np.abs(np.diff(gray, axis=0)).mean()
        features.append(float((dx + dy) / 2.0))

        # Aspect ratio, size
        features.append(float(w) / max(float(h), 1.0))
        features.append(min(float(h * w) / 1e6, 1.0))

        out = np.array(features, dtype=np.float32)
        return _resize_to(out, PRIMARY_DIM)

    def _audio_fallback(self, samples: np.ndarray, sr: int) -> np.ndarray:
        """Extract handcrafted audio features when CLAP is not available.

        Computes: MFCCs, spectral centroid, zero-crossing rate, RMS energy.
        """
        samples = np.asarray(samples, dtype=np.float32).ravel()
        n = len(samples)
        if n == 0:
            return np.zeros(PRIMARY_DIM, dtype=np.float32)

        features = []

        # RMS energy (1)
        rms = float(np.sqrt(np.mean(samples ** 2)))
        features.append(min(rms, 1.0))

        # Zero-crossing rate (1)
        zc = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2.0)
        features.append(min(zc, 1.0))

        # Simple spectral features via FFT
        fft = np.fft.rfft(samples[:min(n, sr)])  # first 1 second
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(min(n, sr), d=1.0 / sr)

        # Spectral centroid (1)
        mag_sum = magnitude.sum()
        if mag_sum > 0:
            centroid = float(np.sum(freqs * magnitude) / mag_sum)
            features.append(min(centroid / (sr / 2), 1.0))
        else:
            features.append(0.0)

        # Spectral bandwidth (1)
        if mag_sum > 0:
            centroid_val = np.sum(freqs * magnitude) / mag_sum
            bw = float(np.sqrt(np.sum(((freqs - centroid_val) ** 2) * magnitude) / mag_sum))
            features.append(min(bw / (sr / 2), 1.0))
        else:
            features.append(0.0)

        # Spectral rolloff (1)
        cumsum = np.cumsum(magnitude)
        if cumsum[-1] > 0:
            rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
            features.append(min(float(freqs[rolloff_idx]) / (sr / 2), 1.0))
        else:
            features.append(0.0)

        # Mel-frequency bins (32 bands, mean energy)
        n_fft = len(magnitude)
        n_mels = 32
        mel_lo = 0.0
        mel_hi = 2595.0 * np.log10(1.0 + (sr / 2) / 700.0)
        mel_points = np.linspace(mel_lo, mel_hi, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
        bin_points = np.floor((min(n, sr) + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft - 1)
        for i in range(n_mels):
            lo, hi = bin_points[i], bin_points[i + 2]
            if hi > lo:
                features.append(float(magnitude[lo:hi].mean()) /
                                max(float(magnitude.max()), 1e-10))
            else:
                features.append(0.0)

        # Temporal envelope (16 windows)
        win_size = max(1, n // 16)
        for i in range(16):
            chunk = samples[i * win_size:(i + 1) * win_size]
            if len(chunk) > 0:
                features.append(min(float(np.sqrt(np.mean(chunk ** 2))), 1.0))
            else:
                features.append(0.0)

        out = np.array(features, dtype=np.float32)
        return _resize_to(out, PRIMARY_DIM)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Normalize embedding from [-1,1] to [0,1]."""
    return (arr + 1.0) * 0.5


def _resize_to(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or truncate array to target length."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if len(arr) >= target_len:
        return arr[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(arr)] = arr
    return out


def _resample(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling."""
    if orig_sr == target_sr:
        return samples
    ratio = target_sr / orig_sr
    n_out = int(len(samples) * ratio)
    indices = np.linspace(0, len(samples) - 1, n_out)
    idx_floor = np.floor(indices).astype(int)
    idx_ceil = np.minimum(idx_floor + 1, len(samples) - 1)
    frac = indices - idx_floor
    return samples[idx_floor] * (1 - frac) + samples[idx_ceil] * frac


def _image_stats(pil_image) -> np.ndarray:
    """Extract low-level image statistics → STATS_DIM-dim vector in [0,1]."""
    img = pil_image.convert("RGB")
    pixels = np.array(img, dtype=np.float32) / 255.0
    h, w, c = pixels.shape

    stats = []
    for ch in range(3):
        chan = pixels[:, :, ch]
        stats.extend([
            float(chan.mean()),
            float(chan.std()),
            float(chan.min()),
            float(chan.max()),
            float(np.median(chan)),
        ])
    # Brightness, contrast
    gray = pixels.mean(axis=2)
    stats.append(float(gray.mean()))
    stats.append(float(gray.std()))
    # Edge density
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    stats.append(float((dx + dy) / 2.0))
    # Aspect ratio, normalized size
    stats.append(min(float(w) / max(float(h), 1.0), 2.0) / 2.0)
    stats.append(min(float(h * w) / 1e6, 1.0))
    # Color dominance (which channel is strongest)
    channel_means = [float(pixels[:, :, ch].mean()) for ch in range(3)]
    total = sum(channel_means) + 1e-10
    for m in channel_means:
        stats.append(m / total)

    out = np.array(stats, dtype=np.float32)
    return _resize_to(out, STATS_DIM)


def _audio_stats(samples: np.ndarray, sr: int) -> np.ndarray:
    """Extract low-level audio statistics → STATS_DIM-dim vector in [0,1]."""
    samples = np.asarray(samples, dtype=np.float32).ravel()
    n = len(samples)
    stats = []

    if n == 0:
        return np.zeros(STATS_DIM, dtype=np.float32)

    # RMS energy
    rms = float(np.sqrt(np.mean(samples ** 2)))
    stats.append(min(rms, 1.0))
    # Peak amplitude
    stats.append(min(float(np.max(np.abs(samples))), 1.0))
    # Zero-crossing rate
    zc = float(np.mean(np.abs(np.diff(np.sign(samples)))) / 2.0)
    stats.append(min(zc, 1.0))
    # Duration (seconds, capped at 1.0 for 30s)
    stats.append(min(float(n) / sr / 30.0, 1.0))
    # Dynamic range
    rms_chunks = []
    chunk_size = max(1, n // 10)
    for i in range(10):
        chunk = samples[i * chunk_size:(i + 1) * chunk_size]
        if len(chunk) > 0:
            rms_chunks.append(float(np.sqrt(np.mean(chunk ** 2))))
    if rms_chunks:
        stats.append(min(max(rms_chunks) - min(rms_chunks), 1.0))
    else:
        stats.append(0.0)
    # Spectral centroid
    fft = np.fft.rfft(samples[:min(n, sr)])
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(min(n, sr), d=1.0 / sr)
    mag_sum = magnitude.sum()
    if mag_sum > 0:
        centroid = float(np.sum(freqs * magnitude) / mag_sum)
        stats.append(min(centroid / (sr / 2), 1.0))
    else:
        stats.append(0.0)

    out = np.array(stats, dtype=np.float32)
    return _resize_to(out, STATS_DIM)
