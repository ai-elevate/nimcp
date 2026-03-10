#!/usr/bin/env python3
"""
multimodal_data.py — Multimodal dataset loader for Athena's developmental training.

Downloads and manages real-world image, audio, and speech datasets so Claude
can _show_ (visual), _tell_ (audio), and _say_ (text) to Athena through her
native SNN sensory bridges.

Phase 1 datasets (~950 MB total):
  - CIFAR-100: 60K 32x32 color images, 100 object classes
  - ESC-50:    2,000 environmental audio clips, 50 sound classes
  - Speech Commands mini: 8K one-second spoken word clips, 8 commands

All datasets are lazily downloaded on first use via HuggingFace `datasets`
or direct URL. No API keys required.
"""

import logging
import os
import random
import struct
import tarfile
import pickle
import numpy as np

logger = logging.getLogger("multimodal_data")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "multimodal")


# ============================================================================
# CIFAR-100 loader (32x32 images, 100 fine-grained classes)
# ============================================================================

# CIFAR-100 fine labels → human-readable descriptions for teaching
CIFAR100_LABELS = [
    "apple", "aquarium fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid",
    "otter", "palm tree", "pear", "pickup truck", "pine tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow tree", "wolf", "woman", "worm",
]

# CIFAR-100 coarse labels (20 superclasses)
CIFAR100_COARSE = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals",
    "non-insect invertebrates", "people", "reptiles",
    "small mammals", "trees", "vehicles 1", "vehicles 2",
]

# Teaching descriptions for each label (Claude uses these as context)
CIFAR100_DESCRIPTIONS = {
    "apple": "A round fruit, often red or green, that grows on trees",
    "bear": "A large furry animal that lives in forests and mountains",
    "bee": "A small flying insect that makes honey and pollinates flowers",
    "beetle": "A small insect with hard wing covers",
    "bicycle": "A two-wheeled vehicle you pedal with your legs",
    "bottle": "A container for holding liquids like water or milk",
    "bridge": "A structure built over water or a road to cross it",
    "bus": "A large vehicle that carries many people on roads",
    "butterfly": "A beautiful insect with colorful wings that flutters",
    "camel": "A large animal with humps that lives in deserts",
    "castle": "A large stone building where kings and queens once lived",
    "caterpillar": "A fuzzy worm-like creature that becomes a butterfly",
    "chair": "A piece of furniture you sit on",
    "clock": "A device that shows what time it is",
    "cloud": "A white fluffy shape in the sky made of tiny water drops",
    "couch": "A long soft seat where you can sit or lie down",
    "crab": "A sea creature with a hard shell and pinching claws",
    "crocodile": "A large reptile with a long snout that lives in rivers",
    "cup": "A small container you drink from",
    "dinosaur": "A very large ancient reptile that lived long ago",
    "dolphin": "A playful sea mammal that leaps through waves",
    "elephant": "The largest land animal with a long trunk and big ears",
    "forest": "A large area covered with many tall trees",
    "fox": "A clever wild animal with red fur and a bushy tail",
    "hamster": "A small furry pet that runs on a wheel",
    "house": "A building where people live with their families",
    "kangaroo": "An Australian animal that hops and carries babies in a pouch",
    "lamp": "A device that makes light so you can see in the dark",
    "leopard": "A wild cat with beautiful spotted fur",
    "lion": "A large wild cat known as the king of the jungle",
    "lizard": "A small reptile with four legs and a long tail",
    "lobster": "A large sea creature with claws, often red when cooked",
    "mountain": "A very tall hill reaching up into the clouds",
    "mouse": "A tiny furry animal with a long thin tail",
    "mushroom": "A fungus that grows from the ground, some you can eat",
    "orange": "A round citrus fruit that is orange in color",
    "orchid": "A beautiful exotic flower with delicate petals",
    "rabbit": "A soft furry animal with long ears that hops",
    "raccoon": "A clever animal with a striped tail and a mask-like face",
    "rocket": "A vehicle that flies into space with powerful engines",
    "rose": "A beautiful flower with soft petals and thorns",
    "sea": "A vast body of salt water stretching to the horizon",
    "seal": "A sea mammal with flippers that swims in cold waters",
    "shark": "A large fish with sharp teeth that swims in the ocean",
    "snail": "A small creature that carries its shell home on its back",
    "snake": "A long reptile with no legs that slithers on the ground",
    "spider": "A small creature with eight legs that spins webs",
    "squirrel": "A small animal with a bushy tail that climbs trees",
    "sunflower": "A tall flower with a big yellow face that follows the sun",
    "table": "A flat piece of furniture where you eat or work",
    "tank": "A heavy armored military vehicle with a big gun",
    "telephone": "A device used to talk to people far away",
    "television": "A screen that shows moving pictures and sounds",
    "tiger": "A large wild cat with orange fur and black stripes",
    "tractor": "A powerful farm vehicle used to pull heavy things",
    "train": "A long vehicle that runs on rails and carries people or cargo",
    "turtle": "A reptile with a hard shell that moves slowly",
    "whale": "The largest animal in the ocean that sings deep songs",
    "wolf": "A wild dog-like animal that howls at the moon and lives in packs",
    "worm": "A small soft creature with no legs that lives in soil",
}

# Fill in any missing descriptions with a generic one
for _label in CIFAR100_LABELS:
    if _label not in CIFAR100_DESCRIPTIONS:
        CIFAR100_DESCRIPTIONS[_label] = f"A {_label}"


# ESC-50 class labels → descriptions
ESC50_LABELS = [
    "dog", "rooster", "pig", "cow", "frog",
    "cat", "hen", "insects", "sheep", "crow",
    "rain", "sea waves", "crackling fire", "crickets", "chirping birds",
    "water drops", "wind", "pouring water", "toilet flush", "thunderstorm",
    "crying baby", "sneezing", "clapping", "breathing", "coughing",
    "footsteps", "laughing", "brushing teeth", "snoring", "drinking sipping",
    "door knock", "mouse click", "keyboard typing", "door creaking",
    "can opening",
    "washing machine", "vacuum cleaner", "clock alarm", "clock tick",
    "glass breaking",
    "helicopter", "chainsaw", "siren", "car horn", "engine",
    "train", "church bells", "airplane", "fireworks", "hand saw",
]

ESC50_DESCRIPTIONS = {
    "dog": "A dog barking — the excited, rhythmic sound dogs make",
    "rooster": "A rooster crowing — the loud wake-up call at dawn",
    "pig": "A pig oinking — the snorting sounds pigs make",
    "cow": "A cow mooing — the deep, long call of cattle",
    "frog": "A frog croaking — the rhythmic sound frogs make near water",
    "cat": "A cat meowing — the vocal call cats use to communicate",
    "hen": "A hen clucking — the soft sounds chickens make",
    "insects": "Insects buzzing — the constant hum of flying bugs",
    "sheep": "A sheep bleating — the 'baa' sound sheep make",
    "crow": "A crow cawing — the harsh call of black crows",
    "rain": "Rain falling — drops of water hitting surfaces rhythmically",
    "sea waves": "Ocean waves — the rhythmic crash and retreat of water",
    "crackling fire": "A fire crackling — the snapping and popping of burning wood",
    "crickets": "Crickets chirping — the steady pulse of cricket song at night",
    "chirping birds": "Birds chirping — the cheerful singing of small birds",
    "water drops": "Water dripping — individual drops falling and splashing",
    "wind": "Wind blowing — the rushing sound of air moving",
    "pouring water": "Water pouring — the continuous splash of liquid flowing",
    "thunderstorm": "Thunder rumbling — the deep rolling boom after lightning",
    "crying baby": "A baby crying — the urgent wail of an infant",
    "sneezing": "A sneeze — the sudden explosive burst of air",
    "clapping": "Hands clapping — the sharp rhythmic sound of applause",
    "laughing": "Laughter — the joyful sound of someone finding something funny",
    "footsteps": "Footsteps — the rhythmic sound of walking",
    "door knock": "Knocking on a door — the rap of knuckles on wood",
    "clock tick": "A clock ticking — the steady mechanical pulse of time",
    "glass breaking": "Glass shattering — the sharp crash of breaking glass",
    "church bells": "Church bells ringing — the deep melodic tone of large bells",
    "fireworks": "Fireworks exploding — the boom and crackle of celebrations",
    "chainsaw": "A chainsaw running — the loud buzzing roar of a power saw",
    "siren": "A siren wailing — the rising and falling alarm of emergency vehicles",
    "car horn": "A car horn honking — the sharp warning blast from a vehicle",
    "helicopter": "A helicopter — the rhythmic chopping of spinning rotors",
    "airplane": "An airplane — the roaring whoosh of jet engines overhead",
    "train": "A train passing — the rumble and clatter of wheels on rails",
}

# Fill missing descriptions
for _label in ESC50_LABELS:
    if _label not in ESC50_DESCRIPTIONS:
        ESC50_DESCRIPTIONS[_label] = f"The sound of {_label}"


# Speech Commands labels
SPEECH_COMMANDS_LABELS = ["yes", "no", "up", "down", "left", "right", "go", "stop"]

SPEECH_DESCRIPTIONS = {
    "yes": "Someone saying 'yes' — an affirmative response",
    "no": "Someone saying 'no' — a negative response",
    "up": "Someone saying 'up' — the direction above",
    "down": "Someone saying 'down' — the direction below",
    "left": "Someone saying 'left' — the direction to one side",
    "right": "Someone saying 'right' — the direction to the other side",
    "go": "Someone saying 'go' — a command to begin moving",
    "stop": "Someone saying 'stop' — a command to halt",
}


class MultimodalDataLoader:
    """Lazy-loading multimodal dataset manager.

    Downloads datasets on first access and caches them locally.
    Provides methods to sample visual, audio, and speech data by concept.
    """

    def __init__(self, data_dir=None):
        self.data_dir = data_dir or DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)

        # Lazy-loaded dataset storage
        self._cifar100 = None       # {label: [image_bytes, ...]}
        self._esc50 = None          # {label: [audio_float_array, ...]}
        self._speech_cmds = None    # {label: [audio_float_array, ...]}

        # Track what's been loaded
        self._cifar100_ready = False
        self._esc50_ready = False
        self._speech_ready = False

    # ------------------------------------------------------------------
    # CIFAR-100 (images)
    # ------------------------------------------------------------------

    def _load_cifar100(self):
        """Load CIFAR-100 from HuggingFace datasets or local cache."""
        if self._cifar100_ready:
            return

        cache_path = os.path.join(self.data_dir, "cifar100_by_label.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading CIFAR-100 from local cache...")
            with open(cache_path, "rb") as f:
                self._cifar100 = pickle.load(f)
            self._cifar100_ready = True
            total = sum(len(v) for v in self._cifar100.values())
            logger.info("CIFAR-100 loaded: %d images across %d classes",
                        total, len(self._cifar100))
            return

        logger.info("Downloading CIFAR-100 via HuggingFace datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset("cifar100", split="train", trust_remote_code=True)
        except Exception as e:
            logger.warning("CIFAR-100 download failed: %s", e)
            self._cifar100 = {}
            self._cifar100_ready = True
            return

        # Organize by fine_label
        by_label = {}
        for item in ds:
            label_idx = item["fine_label"]
            label_name = CIFAR100_LABELS[label_idx]
            img = item["img"]  # PIL Image

            # Convert to 32x32x3 raw bytes
            img = img.convert("RGB").resize((32, 32))
            raw = np.array(img, dtype=np.uint8).tobytes()

            if label_name not in by_label:
                by_label[label_name] = []
            by_label[label_name].append(raw)

        self._cifar100 = by_label
        self._cifar100_ready = True

        # Cache locally
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(by_label, f, protocol=4)
            logger.info("CIFAR-100 cached to %s", cache_path)
        except Exception as e:
            logger.warning("Failed to cache CIFAR-100: %s", e)

        total = sum(len(v) for v in by_label.values())
        logger.info("CIFAR-100 loaded: %d images across %d classes",
                    total, len(by_label))

    def get_visual(self, concept=None):
        """Get a visual sample (image bytes + label + description).

        Args:
            concept: Optional label to match (e.g. "dog", "butterfly").
                     If None, picks a random class.

        Returns:
            (image_bytes, width, height, channels, label, description) or None
        """
        self._load_cifar100()
        if not self._cifar100:
            return None

        if concept:
            # Try exact match first, then substring
            concept_lower = concept.lower()
            matching = [k for k in self._cifar100
                        if concept_lower in k or k in concept_lower]
            if matching:
                label = random.choice(matching)
            else:
                label = random.choice(list(self._cifar100.keys()))
        else:
            label = random.choice(list(self._cifar100.keys()))

        images = self._cifar100[label]
        raw = random.choice(images)
        desc = CIFAR100_DESCRIPTIONS.get(label, f"A {label}")
        return raw, 32, 32, 3, label, desc

    def get_visual_labels(self):
        """Get list of available visual labels."""
        self._load_cifar100()
        return list(self._cifar100.keys()) if self._cifar100 else []

    # ------------------------------------------------------------------
    # ESC-50 (environmental audio)
    # ------------------------------------------------------------------

    def _load_esc50(self):
        """Load ESC-50 from GitHub or local cache."""
        if self._esc50_ready:
            return

        cache_path = os.path.join(self.data_dir, "esc50_by_label.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading ESC-50 from local cache...")
            with open(cache_path, "rb") as f:
                self._esc50 = pickle.load(f)
            self._esc50_ready = True
            total = sum(len(v) for v in self._esc50.values())
            logger.info("ESC-50 loaded: %d clips across %d classes",
                        total, len(self._esc50))
            return

        logger.info("Downloading ESC-50 via HuggingFace datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset("ashraq/esc50", split="train",
                              trust_remote_code=True)
        except Exception as e:
            logger.warning("ESC-50 download failed: %s", e)
            self._esc50 = {}
            self._esc50_ready = True
            return

        by_label = {}
        for item in ds:
            label_idx = item["target"]
            if label_idx < len(ESC50_LABELS):
                label_name = ESC50_LABELS[label_idx]
            else:
                label_name = f"sound_{label_idx}"

            # Get audio array (resample to 16kHz, take first 2 seconds)
            audio = item.get("audio")
            if audio is None:
                continue

            samples = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            # Resample to 16kHz if needed
            if sr != 16000:
                from scipy.signal import resample
                target_len = int(len(samples) * 16000 / sr)
                samples = resample(samples, target_len).astype(np.float32)

            # Take first 2 seconds (32000 samples) for consistency
            max_samples = 32000
            if len(samples) > max_samples:
                samples = samples[:max_samples]

            # Normalize to [-1, 1]
            peak = np.max(np.abs(samples))
            if peak > 1e-6:
                samples = samples / peak * 0.9

            if label_name not in by_label:
                by_label[label_name] = []
            by_label[label_name].append(samples.tolist())

        self._esc50 = by_label
        self._esc50_ready = True

        # Cache locally
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(by_label, f, protocol=4)
            logger.info("ESC-50 cached to %s", cache_path)
        except Exception as e:
            logger.warning("Failed to cache ESC-50: %s", e)

        total = sum(len(v) for v in by_label.values())
        logger.info("ESC-50 loaded: %d clips across %d classes",
                    total, len(by_label))

    def get_audio(self, concept=None):
        """Get an audio sample (float array + label + description).

        Args:
            concept: Optional label to match (e.g. "dog", "rain").
                     If None, picks a random class.

        Returns:
            (audio_samples_list, label, description) or None
        """
        self._load_esc50()
        if not self._esc50:
            return None

        if concept:
            concept_lower = concept.lower()
            matching = [k for k in self._esc50
                        if concept_lower in k or k in concept_lower]
            if matching:
                label = random.choice(matching)
            else:
                label = random.choice(list(self._esc50.keys()))
        else:
            label = random.choice(list(self._esc50.keys()))

        clips = self._esc50[label]
        audio = random.choice(clips)
        desc = ESC50_DESCRIPTIONS.get(label, f"The sound of {label}")
        return audio, label, desc

    def get_audio_labels(self):
        """Get list of available audio labels."""
        self._load_esc50()
        return list(self._esc50.keys()) if self._esc50 else []

    # ------------------------------------------------------------------
    # Speech Commands (spoken words)
    # ------------------------------------------------------------------

    def _load_speech_commands(self):
        """Load Google Speech Commands mini from HuggingFace."""
        if self._speech_ready:
            return

        cache_path = os.path.join(self.data_dir, "speech_cmds_by_label.pkl")
        if os.path.exists(cache_path):
            logger.info("Loading Speech Commands from local cache...")
            with open(cache_path, "rb") as f:
                self._speech_cmds = pickle.load(f)
            self._speech_ready = True
            total = sum(len(v) for v in self._speech_cmds.values())
            logger.info("Speech Commands loaded: %d clips across %d words",
                        total, len(self._speech_cmds))
            return

        logger.info("Downloading Speech Commands via HuggingFace datasets...")
        try:
            from datasets import load_dataset
            ds = load_dataset("google/speech_commands", "v0.02",
                              split="train", trust_remote_code=True)
        except Exception as e:
            logger.warning("Speech Commands download failed: %s", e)
            self._speech_cmds = {}
            self._speech_ready = True
            return

        # Filter to 8 core commands and limit samples per class
        target_labels = set(SPEECH_COMMANDS_LABELS)
        by_label = {label: [] for label in target_labels}
        max_per_label = 200  # Keep dataset manageable

        for item in ds:
            label = item.get("label")
            # The dataset uses integer labels; get the string label
            label_str = ds.features["label"].int2str(label)
            if label_str not in target_labels:
                continue
            if len(by_label[label_str]) >= max_per_label:
                continue

            audio = item.get("audio")
            if audio is None:
                continue

            samples = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            # Resample to 16kHz if needed
            if sr != 16000:
                from scipy.signal import resample
                target_len = int(len(samples) * 16000 / sr)
                samples = resample(samples, target_len).astype(np.float32)

            # Normalize
            peak = np.max(np.abs(samples))
            if peak > 1e-6:
                samples = samples / peak * 0.9

            by_label[label_str].append(samples.tolist())

        # Remove empty labels
        by_label = {k: v for k, v in by_label.items() if v}
        self._speech_cmds = by_label
        self._speech_ready = True

        # Cache locally
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(by_label, f, protocol=4)
            logger.info("Speech Commands cached to %s", cache_path)
        except Exception as e:
            logger.warning("Failed to cache Speech Commands: %s", e)

        total = sum(len(v) for v in by_label.values())
        logger.info("Speech Commands loaded: %d clips across %d words",
                    total, len(by_label))

    def get_speech(self, word=None):
        """Get a spoken word sample (float array + word + description).

        Args:
            word: Optional word to match (e.g. "yes", "stop").
                  If None, picks a random word.

        Returns:
            (audio_samples_list, word, description) or None
        """
        self._load_speech_commands()
        if not self._speech_cmds:
            return None

        if word and word.lower() in self._speech_cmds:
            label = word.lower()
        else:
            label = random.choice(list(self._speech_cmds.keys()))

        clips = self._speech_cmds[label]
        audio = random.choice(clips)
        desc = SPEECH_DESCRIPTIONS.get(label, f"Someone saying '{label}'")
        return audio, label, desc

    def get_speech_labels(self):
        """Get list of available spoken words."""
        self._load_speech_commands()
        return list(self._speech_cmds.keys()) if self._speech_cmds else []

    # ------------------------------------------------------------------
    # Cross-modal pairing
    # ------------------------------------------------------------------

    # Concepts that exist in both visual and audio datasets
    CROSS_MODAL_MAP = {
        # CIFAR-100 label → ESC-50 label(s) that correspond
        "bear": ["dog"],  # no bear sound, closest
        "butterfly": ["insects"],
        "clock": ["clock tick"],
        "dolphin": ["sea waves"],
        "elephant": ["dog"],  # no elephant sound
        "fox": ["dog"],
        "leopard": ["cat"],
        "lion": ["cat"],
        "sea": ["sea waves"],
        "shark": ["sea waves"],
        "snake": ["insects"],
        "spider": ["insects"],
        "squirrel": ["chirping birds"],
        "tiger": ["cat"],
        "train": ["train"],
        "turtle": ["sea waves"],
        "whale": ["sea waves"],
        "wolf": ["dog"],
        "forest": ["chirping birds", "wind", "crickets"],
        "mountain": ["wind", "thunderstorm"],
        "plain": ["wind", "crickets"],
        "bridge": ["train", "car horn"],
        "castle": ["church bells"],
        "house": ["door knock", "clock tick"],
        "road": ["car horn", "engine"],
        "skyscraper": ["car horn", "siren"],
        "rocket": ["airplane"],
        "tank": ["engine"],
        "tractor": ["engine"],
        "motorcycle": ["engine"],
        "bus": ["engine", "car horn"],
        "pickup truck": ["engine", "car horn"],
        "streetcar": ["train"],
        "telephone": ["clock alarm"],
        "television": ["laughing", "clapping"],
        "rose": ["chirping birds"],
        "sunflower": ["chirping birds", "insects"],
        "orchid": ["chirping birds"],
        "tulip": ["chirping birds"],
        "poppy": ["wind"],
        "mushroom": ["rain", "crickets"],
        "apple": ["rain"],
        "orange": ["rain"],
        "pear": ["rain"],
        "sweet pepper": ["rain"],
    }

    def get_multimodal_pair(self):
        """Get a paired visual+audio sample for cross-modal learning.

        Returns:
            {
                "visual": (image_bytes, w, h, ch, label, desc),
                "audio": (audio_samples, label, desc),
                "concept": str,
                "teaching_text": str,
            } or None
        """
        self._load_cifar100()
        self._load_esc50()

        if not self._cifar100 or not self._esc50:
            return None

        # Pick a concept that has cross-modal mapping
        available = [k for k in self.CROSS_MODAL_MAP
                     if k in self._cifar100]
        if not available:
            return None

        concept = random.choice(available)
        audio_labels = self.CROSS_MODAL_MAP[concept]
        audio_label = None
        for al in audio_labels:
            if al in self._esc50:
                audio_label = al
                break
        if not audio_label:
            return None

        visual = self.get_visual(concept)
        audio_clips = self._esc50[audio_label]
        audio_samples = random.choice(audio_clips)
        audio_desc = ESC50_DESCRIPTIONS.get(audio_label,
                                             f"The sound of {audio_label}")

        visual_desc = CIFAR100_DESCRIPTIONS.get(concept, f"A {concept}")
        teaching = (f"Look at this {concept}! {visual_desc}. "
                    f"Listen — {audio_desc.lower()}.")

        return {
            "visual": visual,
            "audio": (audio_samples, audio_label, audio_desc),
            "concept": concept,
            "teaching_text": teaching,
        }

    # ------------------------------------------------------------------
    # Status / summary
    # ------------------------------------------------------------------

    def summary(self):
        """Print summary of loaded datasets."""
        lines = ["Multimodal Dataset Status:"]
        if self._cifar100_ready and self._cifar100:
            total = sum(len(v) for v in self._cifar100.values())
            lines.append(f"  CIFAR-100:  {total:,} images, "
                         f"{len(self._cifar100)} classes")
        else:
            lines.append("  CIFAR-100:  not loaded")

        if self._esc50_ready and self._esc50:
            total = sum(len(v) for v in self._esc50.values())
            lines.append(f"  ESC-50:     {total:,} audio clips, "
                         f"{len(self._esc50)} classes")
        else:
            lines.append("  ESC-50:     not loaded")

        if self._speech_ready and self._speech_cmds:
            total = sum(len(v) for v in self._speech_cmds.values())
            lines.append(f"  Speech:     {total:,} word clips, "
                         f"{len(self._speech_cmds)} words")
        else:
            lines.append("  Speech:     not loaded")

        return "\n".join(lines)

    def ensure_downloaded(self):
        """Pre-download all datasets (call before brain init eats RAM)."""
        print("  [Multimodal] Downloading datasets...")
        self._load_cifar100()
        self._load_esc50()
        self._load_speech_commands()
        print(f"  [Multimodal] {self.summary()}")
