#!/usr/bin/env python3
"""Dataset loaders for multimodal perception training.

Yields (raw_data, class_label, text_caption) tuples from standard datasets
using HuggingFace datasets in streaming mode.

Supported datasets:
  Visual:   CIFAR-100, Fashion-MNIST
  Audio:    ESC-50 (environmental sounds)
  Speech:   LibriSpeech (read speech + transcriptions)
  Language: Tatoeba parallel sentences (59 language pairs)
"""

import logging
from typing import Iterator, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("multimodal_datasets")


def _extract_audio(audio_obj):
    """Extract (samples_np, sample_rate) from HF audio field.

    Handles both the legacy dict format {'array': ..., 'sampling_rate': ...}
    and the new torchcodec AudioDecoder format.
    Returns (np.ndarray float32, int) or (None, None) on failure.
    """
    if audio_obj is None:
        return None, None
    # Legacy dict format
    if isinstance(audio_obj, dict):
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate", 16000)
        if arr is None or len(arr) == 0:
            return None, None
        return np.asarray(arr, dtype=np.float32), int(sr)
    # torchcodec AudioDecoder format
    if hasattr(audio_obj, "get_all_samples"):
        try:
            samples = audio_obj.get_all_samples()
            sr = samples.sample_rate
            arr = samples.data.numpy().flatten()
            if len(arr) == 0:
                return None, None
            return arr.astype(np.float32), int(sr)
        except Exception as e:
            logger.debug("torchcodec audio extraction failed: %s", e)
            return None, None
    logger.debug("Unknown audio format: %s", type(audio_obj))
    return None, None


# Try importing PIL — needed for visual datasets
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Try importing datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("HuggingFace datasets not available — multimodal datasets disabled")


# ---------------------------------------------------------------------------
# Caption templates
# ---------------------------------------------------------------------------

CIFAR100_CAPTIONS = {
    "default": "a photo of a {label}",
}

FASHION_MNIST_CLASSES = [
    "t-shirt/top", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "ankle boot",
]

ESC50_CAPTION = "the sound of {category}"

# ---------------------------------------------------------------------------
# Visual datasets
# ---------------------------------------------------------------------------

class VisualDatasetLoader:
    """Loads visual datasets, yielding (PIL.Image, class_label, caption).

    Supports: cifar100, fashion_mnist
    """

    def __init__(self, dataset_name: str = "cifar100",
                 split: str = "train",
                 max_examples: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 42):
        self.dataset_name = dataset_name
        self.split = split
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple]:
        """Yields (PIL.Image, class_label: str, caption: str)."""
        if not HF_AVAILABLE:
            logger.error("HuggingFace datasets required for VisualDatasetLoader")
            return
        if not PIL_AVAILABLE:
            logger.error("Pillow required for VisualDatasetLoader")
            return

        if self.dataset_name == "cifar100":
            yield from self._load_cifar100()
        elif self.dataset_name == "fashion_mnist":
            yield from self._load_fashion_mnist()
        else:
            logger.error("Unknown visual dataset: %s", self.dataset_name)

    def _load_cifar100(self) -> Iterator[Tuple]:
        try:
            ds = load_dataset("cifar100", split=self.split, streaming=True)
        except Exception as e:
            logger.error("Failed to load CIFAR-100: %s", e)
            return

        if self.shuffle:
            ds = ds.shuffle(seed=self.seed, buffer_size=1000)

        count = 0
        for example in ds:
            img = example.get("img") or example.get("image")
            fine_label = example.get("fine_label", 0)
            # CIFAR-100 fine labels are ints — get name from features
            try:
                label_name = ds.features["fine_label"].names[fine_label]
            except (AttributeError, IndexError, KeyError):
                label_name = str(fine_label)

            if img is None:
                continue
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(np.array(img))
                except Exception:
                    continue

            caption = f"a photo of a {label_name}"
            yield img, label_name, caption

            count += 1
            if self.max_examples and count >= self.max_examples:
                return

    def _load_fashion_mnist(self) -> Iterator[Tuple]:
        try:
            ds = load_dataset("fashion_mnist", split=self.split, streaming=True)
        except Exception as e:
            logger.error("Failed to load Fashion-MNIST: %s", e)
            return

        if self.shuffle:
            ds = ds.shuffle(seed=self.seed, buffer_size=1000)

        count = 0
        for example in ds:
            img = example.get("image")
            label_idx = example.get("label", 0)

            if img is None:
                continue
            if not isinstance(img, Image.Image):
                try:
                    img = Image.fromarray(np.array(img))
                except Exception:
                    continue

            # Convert grayscale to RGB for consistency
            if img.mode != "RGB":
                img = img.convert("RGB")

            label_name = FASHION_MNIST_CLASSES[label_idx] if label_idx < len(
                FASHION_MNIST_CLASSES) else str(label_idx)
            caption = f"a photo of a {label_name} clothing item"
            yield img, label_name, caption

            count += 1
            if self.max_examples and count >= self.max_examples:
                return


# ---------------------------------------------------------------------------
# Audio datasets
# ---------------------------------------------------------------------------

class AudioDatasetLoader:
    """Loads audio datasets, yielding (samples, sample_rate, class_label, caption).

    Supports: esc50
    """

    def __init__(self, dataset_name: str = "esc50",
                 split: str = "train",
                 max_examples: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 42):
        self.dataset_name = dataset_name
        self.split = split
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple]:
        """Yields (samples: np.ndarray, sample_rate: int, class_label: str, caption: str)."""
        if not HF_AVAILABLE:
            logger.error("HuggingFace datasets required for AudioDatasetLoader")
            return

        if self.dataset_name == "esc50":
            yield from self._load_esc50()
        else:
            logger.error("Unknown audio dataset: %s", self.dataset_name)

    def _load_esc50(self) -> Iterator[Tuple]:
        try:
            ds = load_dataset("ashraq/esc50", split=self.split, streaming=True)
        except Exception as e:
            logger.error("Failed to load ESC-50: %s", e)
            return

        if self.shuffle:
            ds = ds.shuffle(seed=self.seed, buffer_size=500)

        count = 0
        for example in ds:
            samples, sr = _extract_audio(example.get("audio"))
            if samples is None:
                continue

            category = example.get("category", "unknown")
            caption = f"the sound of {category}"
            yield samples, sr, category, caption

            count += 1
            if self.max_examples and count >= self.max_examples:
                return


# ---------------------------------------------------------------------------
# Speech datasets
# ---------------------------------------------------------------------------

class SpeechDatasetLoader:
    """Loads speech datasets, yielding (samples, sample_rate, transcription).

    Supports: librispeech (default), ljspeech (legacy, may fail on newer HF)
    """

    # Map simple names to HF identifiers and split names
    _DATASET_MAP = {
        "librispeech": ("openslr/librispeech_asr", "train.clean.100"),
        "ljspeech": ("keithito/lj_speech", "train"),
    }

    def __init__(self, dataset_name: str = "librispeech",
                 split: str | None = None,
                 max_examples: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 42):
        self.dataset_name = dataset_name
        self._split_override = split
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Tuple]:
        """Yields (samples: np.ndarray, sample_rate: int, transcription: str)."""
        if not HF_AVAILABLE:
            logger.error("HuggingFace datasets required for SpeechDatasetLoader")
            return

        if self.dataset_name in ("librispeech", "ljspeech"):
            yield from self._load_speech()
        else:
            logger.error("Unknown speech dataset: %s", self.dataset_name)

    def _load_speech(self) -> Iterator[Tuple]:
        hf_id, default_split = self._DATASET_MAP[self.dataset_name]
        split = self._split_override or default_split

        try:
            ds = load_dataset(hf_id, split=split, streaming=True)
        except Exception as e:
            logger.error("Failed to load %s (%s): %s", self.dataset_name, hf_id, e)
            return

        if self.shuffle:
            ds = ds.shuffle(seed=self.seed, buffer_size=500)

        count = 0
        for example in ds:
            samples, sr = _extract_audio(example.get("audio"))
            if samples is None:
                continue

            # LibriSpeech uses "text", LJ Speech uses "normalized_text" / "text"
            transcription = (example.get("normalized_text")
                             or example.get("text", ""))
            if not transcription:
                continue

            yield samples, sr, transcription

            count += 1
            if self.max_examples and count >= self.max_examples:
                return


# ---------------------------------------------------------------------------
# Language text datasets (multilingual)
# ---------------------------------------------------------------------------

class LanguageTextDatasetLoader:
    """Loads multilingual text from Tatoeba, yielding (text, lang_code).

    Uses sentence-transformers/parallel-sentences-tatoeba from HuggingFace.
    Each language pair yields BOTH English and target-language sentences.
    """

    _LANG_CONFIGS = {
        "fr": "en-fr", "es": "en-es", "de": "en-de",
        "it": "en-it", "nl": "en-nl", "pt": "en-pt",
        "ru": "en-ru", "ja": "en-ja", "zh": "en-zh",
    }

    def __init__(self, languages=("en", "fr", "es", "de"),
                 max_examples_per_lang=None, shuffle=True, seed=42):
        self.languages = tuple(languages)
        self.max_examples_per_lang = max_examples_per_lang
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[tuple]:
        """Yields (text, lang_code) pairs, interleaving languages."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace datasets not available — "
                           "LanguageTextDatasetLoader returning empty")
            return

        # Collect per-language iterators
        lang_iters = {}
        non_en = [l for l in self.languages if l != "en" and l in self._LANG_CONFIGS]

        if not non_en:
            # English-only: use any available pair for its English side
            non_en = ["fr"]

        for lang in non_en:
            config = self._LANG_CONFIGS[lang]
            try:
                ds = load_dataset(
                    "sentence-transformers/parallel-sentences-tatoeba",
                    config, split="train", streaming=True)
                if self.shuffle:
                    ds = ds.shuffle(seed=self.seed, buffer_size=2000)
                lang_iters[lang] = iter(ds)
            except Exception as e:
                logger.warning("Failed to load Tatoeba %s: %s", config, e)

        if not lang_iters:
            logger.warning("No Tatoeba language pairs loaded — empty iterator")
            return

        # Round-robin interleave with both English + target per example
        import random as _rng
        rng = _rng.Random(self.seed)
        lang_keys = list(lang_iters.keys())
        counts = {l: 0 for l in lang_keys}
        counts["en"] = 0
        exhausted = set()

        while len(exhausted) < len(lang_keys):
            # Pick a random non-exhausted language
            active = [l for l in lang_keys if l not in exhausted]
            if not active:
                break
            lang = rng.choice(active)

            try:
                example = next(lang_iters[lang])
            except StopIteration:
                exhausted.add(lang)
                continue

            # Each example has 'english' and 'non_english' columns
            en_text = example.get("english", "")
            non_en_text = example.get("non_english", "")

            if en_text and "en" in self.languages:
                yield (en_text.strip(), "en")
                counts["en"] += 1

            if non_en_text and lang in self.languages:
                yield (non_en_text.strip(), lang)
                counts[lang] += 1

            # Check per-language cap
            if self.max_examples_per_lang:
                if counts[lang] >= self.max_examples_per_lang:
                    exhausted.add(lang)
