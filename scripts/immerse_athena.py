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
    outputs. She is a developing biological brain. Her 2048-dim output
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
import logging
import os
import sys
import time
import random
import json
import signal
import numpy as np

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress noisy progress bars and verbose logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

import nimcp
from claude_teacher import ClaudeTeacher, encode_text
from talk_to_athena import extract_embedding_from_output
from neural_decoder import NeuralDecoder


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

logger = logging.getLogger("immerse_athena")

# ============================================================================
# Constants
# ============================================================================

BRAIN_INPUT_DIM = 1024
BRAIN_OUTPUT_DIM = 2048
TAG_DIM = 16           # [0:16]   modality flags + brain state
PRIMARY_DIM = 512      # [16:528] primary modality features
TEXT_DIM = 384         # [528:912] text semantic embedding
CONTEXT_DIM = 112      # [912:1024] biological context (arousal, sleep, dopamine, etc.)

CHECKPOINT_DIR = "checkpoints/athena"
DECODER_DIR = "checkpoints/athena/decoder"
STATE_FILE = "checkpoints/athena/immersive_state.json"
MASTERY_FILE = "checkpoints/athena/mastery_state.json"

# Adaptive curriculum constants
MASTERY_WINDOW = 20          # Rolling window for accuracy tracking
MASTERY_THRESHOLD_LOW = 0.3  # Below this: remedial (reduce difficulty)
MASTERY_THRESHOLD_HIGH = 0.8 # Above this: advance (increase difficulty)
ZPD_TARGET = 0.65            # Zone of Proximal Development target accuracy
DIFFICULTY_STEP = 0.1        # How much to adjust difficulty per evaluation

# Adaptive learning rate constants
BASE_LEARNING_RATE = 0.001   # Default brain learning rate
MIN_LEARNING_RATE = 0.0001   # Floor — prevents stalling
MAX_LEARNING_RATE = 0.005    # Ceiling — prevents divergence
LR_SCALE_STRUGGLING = 1.5    # Scale up when mastery < THRESHOLD_LOW
LR_SCALE_MASTERED = 0.5      # Scale down when mastery > THRESHOLD_HIGH
LR_EVAL_INTERVAL = 100       # Steps between LR recalculations

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
        with open(filepath) as f:
            data = json.load(f)
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

        Layout: [tag:16 | primary:512 | text:384 | context:112]
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

        # --- Primary features [16:528] ---
        if primary_features is not None:
            pf = np.array(primary_features, dtype=np.float32)
            n = min(len(pf), PRIMARY_DIM)
            vec[TAG_DIM:TAG_DIM + n] = pf[:n]
        elif text is not None:
            emb = encode_text(text)  # 384-dim
            # Pad to 512 with zeros
            n = min(len(emb), PRIMARY_DIM)
            vec[TAG_DIM:TAG_DIM + n] = emb[:n]

        # --- Text semantic embedding [528:912] ---
        if text_embedding is not None:
            te = np.array(text_embedding, dtype=np.float32)
            n = min(len(te), TEXT_DIM)
            vec[TAG_DIM + PRIMARY_DIM:TAG_DIM + PRIMARY_DIM + n] = te[:n]
        elif text is not None:
            emb = encode_text(text)  # 384-dim
            vec[TAG_DIM + PRIMARY_DIM:TAG_DIM + PRIMARY_DIM + len(emb)] = emb

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


def make_semantic_target(text, target_dim=BRAIN_OUTPUT_DIM):
    """Create a target vector by tiling the semantic embedding to target_dim."""
    emb = encode_text(text)  # 384-dim
    target = np.zeros(target_dim, dtype=np.float32)
    # Tile embedding across target
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
                brain.medulla_boost_arousal(-0.1)
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

    def __init__(self, teacher=None, enabled=True, decoder=None):
        self.teacher = teacher
        self.enabled = enabled and teacher is not None
        self.decoder = decoder
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
        """Pre-generate ALL parent narrations for a stage in one Claude call.

        Makes a single large Claude call that returns JSON with all the
        narrations, moral stories, speech prompts, and inspirations needed
        for the entire stage. Zero blocking during training.
        """
        if not self.enabled:
            print("  [Parent] Claude disabled — using silent parenting")
            return

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
            # Single Claude call with generous token budget
            raw = self.teacher._call_claude(prompt, max_tokens=4096)

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
        """Pick a random item from a content pool (non-destructive)."""
        pool = getattr(self, pool_name)
        if not pool:
            return None
        return random.choice(pool)

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

        # Extract her 384-dim embedding from the 2048-dim tiled output
        output_emb = extract_embedding_from_output(np.array(output_vec))

        # Describe what her response is nearest to (for human readability)
        if self.decoder and len(self.decoder.vocabulary) > 0:
            matches = self.decoder.vocabulary.decode(output_emb, top_k=1)
            if matches and matches[0][0]:
                return matches[0][0]
        return ""

    # --- Stage 0: "Welcome to the world, little one" ---

    def first_experience(self, brain, composer, description):
        """Introduce Athena to her first sensory experience with wonder."""
        narration = self._pop_content("_narrations")
        if narration:
            print(f"  Parent: {narration}")

        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(description)

        result = brain.decide_full(features)
        # Dense target trains adaptive network; label trains CNN classifier
        loss = brain.learn_vector(features, target, label=description[:50], confidence=0.5)

        # Record pair for decoder vocabulary (nearest-neighbor for display only)
        if self.decoder:
            output_vec = result.get("output_vector")
            if output_vec is not None:
                target_emb = encode_text(description)
                self.decoder.record_pair(output_vec, target_emb, description)

        # Gentle positive reward — existence is rewarded
        try:
            brain.bg_update_reward(0.5, 0.3)
        except Exception:
            pass

        self.interaction_count += 1
        return loss, result

    # --- Stage 1: "That's a ___! Can you see?" ---

    def show_and_name(self, brain, composer, name, description):
        """Associate a percept with meaning, with enthusiasm."""
        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(name + " " + description)

        result = brain.decide_full(features)

        try:
            brain.bg_update_reward(0.6, 0.4)
        except Exception:
            pass

        narration = self._pop_content("_narrations")
        if narration:
            print(f"  Parent: {narration}")

        # Dense target trains adaptive network; label trains CNN classifier
        loss = brain.learn_vector(features, target, label=name[:50], confidence=0.65)

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

    def ask_and_encourage(self, brain, composer, concept, description):
        """Show Athena a stimulus, observe her response, teach, encourage.

        There is no correct or incorrect. Athena experiences, responds,
        and learns. The loss from learn_vector tells us how far her
        current representation is from the teaching signal — that's a
        measure of her developmental progress, not a prediction score.
        """
        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(concept + " " + description)

        # Let Athena experience the stimulus
        result = brain.decide_full(features)
        response_text = self.observe_response(result)

        # Dense target trains adaptive network; label trains CNN classifier
        loss = brain.learn_vector(features, target, label=concept[:50], confidence=0.6)

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
        """
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
            brain.decide_full(features)
            # High confidence to establish distinct representations
            brain.learn_vector(features, target, label=wtext[:50], confidence=0.9)
            # Repeat each warmup stimulus 3x to establish strong initial patterns
            for _ in range(2):
                brain.learn_vector(features, target, label=wtext[:50], confidence=0.85)
        print(f"  [Warmup] {len(warmup_texts)} diverse stimuli x3 — done")

    losses = []
    for i in range(start_from, num_stimuli):
        # Check biological clock
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Get a sensory stimulus
        if random.random() < 0.5:
            description = source.get_sensory()
        else:
            description, _ = generate_sensory_exposure()

        loss, result = parent.first_experience(brain, composer, description)
        losses.append(loss)

        # Periodic LNN forward step for temporal context
        try:
            features = composer.compose(text=description)
            brain.lnn_forward_step(features[:128])
        except Exception:
            pass

        # Progress report
        if (i + 1) % 500 == 0:
            avg_loss = np.mean(losses[-500:]) if losses else 0
            print(f"\n  [Stage 0] {i+1}/{num_stimuli} — "
                  f"avg_loss={avg_loss:.4f}")
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

        # Inspire every 2000 stimuli
        if (i + 1) % 2000 == 0:
            parent.inspire(brain, composer, stage=0)

        # Checkpoint every 5000
        if (i + 1) % 5000 == 0:
            _save_checkpoint(brain, decoder, stage=0, step=i+1)

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
    for i in range(start_from, num_stimuli):
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Get an object to name
        name, description = source.get_object()

        loss, result = parent.show_and_name(brain, composer, name, description)
        losses.append(loss if loss is not None else 0)

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
            print(f"\n  [Stage 1] {i+1}/{num_stimuli} — "
                  f"avg_loss={avg_loss:.4f}")
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
            brain, composer, expected, description)
        losses.append(loss if loss is not None else 0)

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
            print(f"\n  [Stage 2] {i+1}/{num_stimuli} — "
                  f"mean_loss={mean_loss:.4f}")
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
            result = brain.decide_full(features)
            response_text = parent.observe_response(result)
            print(f"  Q: {question}")
            print(f"  Athena: {response_text or '(still forming thoughts)'}")

            parent_reply = parent._pop_content("_conversation_replies")
            if parent_reply:
                print(f"  Parent: {parent_reply}")

            try:
                brain.bg_update_reward(0.6, 0.5)
            except Exception:
                pass

        elif r < 0.40:
            # Advanced domain teaching — adaptive difficulty
            domain = mastery.select_domain(active_domains)
            topic_text, domain_name = source.get_advanced(domain)
            print(f"  [Domain: {domain_name}] {topic_text[:80]}")

            features = composer.compose(text=topic_text, modality="text")
            target = make_semantic_target(topic_text)
            result = brain.decide_full(features)
            confidence = mastery.get_confidence_for_domain(domain_name, progress)
            adaptive_lr = mastery.get_adaptive_lr(domain_name)
            loss = brain.learn_vector(features, target,
                                       label=domain_name[:50], confidence=confidence,
                                       learning_rate=adaptive_lr)
            mastery.record(domain_name, loss, confidence)

            if decoder:
                output_vec = result.get("output_vector")
                if output_vec is not None:
                    target_emb = encode_text(topic_text)
                    decoder.record_pair(output_vec, target_emb, topic_text)

            if loss is not None:
                print(f"    loss={loss:.4f} conf={confidence:.2f}")

        elif r < 0.52:
            # Claude-generated curriculum lesson (if available)
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
                        brain.learn_vector(features, target,
                                            label=spec.domain[:50], confidence=0.7)
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
            loss = brain.learn_vector(features, target,
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
                    brain.learn_vector(features, target, label=topic[:50], confidence=0.5)
            except Exception:
                # Free exploration from advanced topics
                topic_text, domain = source.get_advanced(random.choice(active_domains))
                features = composer.compose(text=topic_text)
                target = make_semantic_target(topic_text)
                brain.learn_vector(features, target, label=domain[:50], confidence=0.4)

        # LNN step
        try:
            features = composer.compose(text="temporal context update")
            brain.lnn_forward_step(features[:128])
        except Exception:
            pass

        # Progress — show active domains
        # Adapt difficulty every 50 steps
        if (i + 1) % 50 == 0:
            for d in active_domains:
                mastery.adapt_difficulty(d)

        if (i + 1) % 200 == 0:
            print(f"\n  [Stage 3] {i+1}/{num_interactions} — "
                  f"domains: {', '.join(active_domains)}")
            # Show adaptive LR per domain
            lr_info = ", ".join(
                f"{d}={mastery.get_adaptive_lr(d):.5f}" for d in active_domains[:5]
            )
            print(f"  Adaptive LR: {lr_info}")
            print(f"  Mastery:\n{mastery.summary()}")
            _print_bio_stats(brain)

        if (i + 1) % 500 == 0:
            if decoder:
                decoder.force_refit()
            evaluate_performance(brain, composer, decoder, stage=3, step=i+1)

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

        # Forward chain KB
        if (i + 1) % 1000 == 0:
            try:
                derived = brain.ti_forward_chain(30)
                if derived > 0:
                    print(f"    KB: {derived} new facts derived")
            except Exception:
                pass


# ============================================================================
# Utilities
# ============================================================================

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
    print(f"\n  {'─' * 56}")
    print(f"  PERFORMANCE EVAL — Stage {stage}, Step {step}")
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
        print(f"\n  Response evolution (first → latest):")
        for probe_name in probes_with_history[:4]:
            history = _eval_history[probe_name]
            first = history[0]
            latest = history[-1]
            first_text = first[1][:30] if first[1] else f"|v|={first[3]:.1f}"
            latest_text = latest[1][:30] if latest[1] else f"|v|={latest[3]:.1f}"
            sim_change = latest[2] - first[2]
            norm_change = latest[3] - first[3]
            print(f"    {probe_name:8s}: {first_text:>32s} → {latest_text:<32s}"
                  f"  Δsim={sim_change:+.3f}  Δ|v|={norm_change:+.1f}")

    print(f"  {'─' * 56}\n")


def _save_checkpoint(brain, decoder, stage, step):
    """Save brain checkpoint and decoder state."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt = os.path.join(CHECKPOINT_DIR, "athena_immersive.bin")
    try:
        brain.save(ckpt)
        logger.info("Checkpoint saved: %s (stage=%d, step=%d)",
                     ckpt, stage, step)
    except Exception as e:
        logger.warning("Checkpoint save failed: %s", e)

    if decoder:
        os.makedirs(DECODER_DIR, exist_ok=True)
        try:
            decoder.save(DECODER_DIR)
        except Exception:
            pass

    # Save state
    state = {"stage": stage, "step": step,
             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass


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
# Main
# ============================================================================

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
    parser.add_argument("--neuron-count", type=int, default=1500000)
    parser.add_argument("--stage0-stimuli", type=int, default=20000)
    parser.add_argument("--stage1-stimuli", type=int, default=40000)
    parser.add_argument("--stage2-stimuli", type=int, default=40000)
    parser.add_argument("--stage3-interactions", type=int, default=30000)
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (ignore existing checkpoint)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(message)s")

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

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Loading from checkpoint: {checkpoint_path}")
    else:
        print("  Creating fresh brain (full biological init)")
        checkpoint_path = None

    brain = nimcp.Brain("athena",
                        num_inputs=args.num_inputs,
                        num_outputs=args.num_outputs,
                        neuron_count=args.neuron_count,
                        checkpoint=checkpoint_path)

    # --- CRITICAL: Disable fast training → ALL bio subsystems active ---
    brain.set_fast_training(False)
    print("  Fast training: OFF (full biological pipeline)")

    # --- CRITICAL: Use regression strategy (raw output, no softmax) ---
    # Classification strategy applies softmax which crushes output to ~1/N per neuron.
    # Athena needs raw output vectors for embedding-based developmental learning.
    brain.set_task_type("regression")
    print("  Task type: REGRESSION (raw output, no softmax)")

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
        if os.path.exists(DECODER_DIR):
            decoder = NeuralDecoder.load(DECODER_DIR)
        else:
            decoder = NeuralDecoder(output_dim=args.num_outputs)
        print("  Neural decoder: ON")
    except Exception:
        decoder = NeuralDecoder(output_dim=args.num_outputs)

    parent = Parent(teacher=teacher, enabled=not args.no_claude, decoder=decoder)
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

    def signal_handler(sig, frame):
        if shutdown_requested[0]:
            print("\n  Force quit.")
            sys.exit(1)
        shutdown_requested[0] = True
        print("\n  Graceful shutdown... saving checkpoint...")
        _save_checkpoint(brain, decoder, start_stage, -1)
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # --- Print initial bio stats ---
    print("\n  Initial biological state:")
    _print_bio_stats(brain)

    # --- Run developmental stages ---
    print(f"\n  Starting from stage {start_stage}")

    if start_stage <= 0:
        run_stage_0(brain, composer, parent, clock, source, decoder,
                    num_stimuli=args.stage0_stimuli,
                    start_from=start_step if start_stage == 0 else 0)
        start_step = 0  # Reset for next stage

    if start_stage <= 1:
        run_stage_1(brain, composer, parent, clock, source, decoder,
                    num_stimuli=args.stage1_stimuli,
                    start_from=start_step if start_stage == 1 else 0)
        start_step = 0

    if start_stage <= 2:
        stage2_losses = run_stage_2(
            brain, composer, parent, clock, source, decoder,
            num_stimuli=args.stage2_stimuli,
            start_from=start_step if start_stage == 2 else 0)
        if stage2_losses:
            print(f"\n  Stage 2 complete — final mean loss: "
                  f"{np.mean(stage2_losses[-500:]):.4f}")
        start_step = 0

    if start_stage <= 3:
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
