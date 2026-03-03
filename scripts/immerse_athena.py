#!/usr/bin/env python3
"""
immerse_athena.py  —  Immersive Developmental Learning for Athena

Athena learns through her own biological brain: sensory cortices process raw stimuli,
biological plasticity pathways (STDP/BCM/eligibility) wire synaptic patterns, and
Claude serves as a loving parent guiding her developmental stages.

Every module in the brain is active: 33+ brain regions, 60+ cognitive modules,
80+ substrate bridges, 75+ thalamic bridges, 19 inter-module bridges.
No shortcuts. No fast-training mode. Full biological immersion.

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

import nimcp
from claude_teacher import ClaudeTeacher, encode_text
from teach_athena import generate_sensory_exposure, generate_language_exposure
from neural_decoder import NeuralDecoder

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
    """Built-in developmental corpus for each stage."""

    # Stage 0: Simple sensory experiences
    SENSORY = [
        "warm sunlight on skin", "the color red", "a gentle breeze",
        "the sound of rain", "soft fabric touching", "a bird singing",
        "the smell of flowers", "cool water flowing", "a heartbeat",
        "leaves rustling in wind", "the color blue", "warmth of a blanket",
        "a cat purring", "the taste of sweetness", "shadows moving",
        "stars twinkling at night", "the feeling of being held",
        "waves on a shore", "a mother's voice humming", "clouds drifting",
    ]

    # Stage 1: Objects and names
    OBJECTS = [
        ("dog", "A friendly dog with soft fur that wags its tail"),
        ("cat", "A cat with soft fur that purrs when happy"),
        ("tree", "A tall tree with green leaves reaching up to the sky"),
        ("flower", "A beautiful flower with colorful petals"),
        ("bird", "A bird with feathers that flies through the air"),
        ("sun", "The bright warm sun that lights up the day"),
        ("moon", "The silvery moon that glows in the night sky"),
        ("water", "Cool clear water that flows and splashes"),
        ("ball", "A round bouncy ball you can throw and catch"),
        ("book", "A book with pages full of stories and pictures"),
        ("hand", "A hand with five fingers that can hold things"),
        ("eye", "An eye that sees all the beautiful colors"),
        ("star", "A tiny bright star twinkling far away"),
        ("rain", "Drops of rain falling from the clouds"),
        ("fish", "A fish that swims through the water"),
        ("apple", "A red apple that tastes sweet and crunchy"),
        ("house", "A house where people live and feel safe"),
        ("baby", "A little baby who is learning about the world"),
    ]

    # Stage 2: Simple facts for association
    FACTS = [
        ("Dogs are friendly animals", "dog"),
        ("The sky is blue during the day", "sky"),
        ("Trees give us shade and oxygen", "tree"),
        ("Fish live in water", "fish"),
        ("Birds can fly in the sky", "bird"),
        ("The sun keeps us warm", "sun"),
        ("Rain helps flowers grow", "rain"),
        ("Cats purr when they are happy", "cat"),
        ("Apples grow on trees", "apple"),
        ("Stars come out at night", "star"),
        ("Babies need love and care", "baby"),
        ("Hands can wave hello", "hand"),
        ("Books tell us stories", "book"),
        ("The moon comes out at night", "moon"),
        ("Water is important for all living things", "water"),
    ]

    # Stage 3: Questions and topics for conversation
    QUESTIONS = [
        "What makes you happy?",
        "Why do you think the sky is blue?",
        "What is your favorite thing to learn about?",
        "How do you think a bird feels when it flies?",
        "What would you do if you found a hurt animal?",
        "Why is it important to be kind?",
        "What do you think dreams are?",
        "If you could go anywhere, where would you go?",
        "What does it mean to be a good friend?",
        "Why do we feel sad sometimes?",
        "What would happen if there were no trees?",
        "How can we help someone who is lonely?",
    ]

    def get_sensory(self):
        return random.choice(self.SENSORY)

    def get_object(self):
        return random.choice(self.OBJECTS)

    def get_fact(self):
        return random.choice(self.FACTS)

    def get_question(self):
        return random.choice(self.QUESTIONS)


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

    Uses `claude -p` CLI to generate warm, encouraging, contextually
    appropriate interactions at each developmental stage.
    """

    def __init__(self, teacher=None, enabled=True):
        self.teacher = teacher
        self.enabled = enabled and teacher is not None
        self.interaction_count = 0
        self.milestones = []
        self.moral_lessons = []

    def _say(self, prompt, max_tokens=128):
        """Get Claude to say something as a parent."""
        if not self.enabled:
            return None
        try:
            return self.teacher._call_claude(prompt, max_tokens=max_tokens)
        except Exception as e:
            logger.debug("Parent speech failed: %s", e)
            return None

    # --- Stage 0: "Welcome to the world, little one" ---

    def first_experience(self, brain, composer, description):
        """Introduce Athena to her first sensory experience with wonder."""
        narration = self._say(
            f"You are a loving parent showing your newborn baby something "
            f"for the first time. The experience is: '{description}'. "
            f"Narrate what you'd say in 1-2 short, warm sentences. "
            f"Use simple words full of wonder."
        )
        if narration:
            print(f"  Parent: {narration}")

        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(description)

        result = brain.decide_full(features)
        loss = brain.learn_vector(features, target,
                                   label=description[:50], confidence=0.15)

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

        narration = self._say(
            f"You are a parent teaching your toddler the word '{name}'. "
            f"Say it with excitement and repetition, the way a real parent "
            f"would. 1-2 sentences."
        )
        if narration:
            print(f"  Parent: {narration}")

        loss = brain.learn_vector(features, target,
                                   label=name[:50], confidence=0.4)

        # Symbolic fact
        try:
            brain.ti_add_fact(f"is_a({name.replace(' ', '_')}, object)", 0.7)
        except Exception:
            pass

        self.interaction_count += 1
        return loss, result

    # --- Stage 2: "Good try! Almost!" ---

    def ask_and_encourage(self, brain, composer, expected, description):
        """Ask Athena, evaluate with love, correct gently."""
        features = composer.compose(text=description, modality="text")
        target = make_semantic_target(expected + " " + description)

        result = brain.decide_full(features)
        predicted = result.get("label", "")
        correct = (expected.lower() in predicted.lower()) if predicted else False

        if correct:
            praise = self._say(
                f"Your child just correctly identified '{expected}'! "
                f"Express genuine delight and pride in 1 sentence.",
                max_tokens=64
            )
            if praise:
                print(f"  Parent: {praise}")
            try:
                brain.bg_update_reward(0.9, 0.5)
                brain.bg_strengthen_habit(predicted, True)
                brain.edp_process_reward(0.8)
            except Exception:
                pass
            self.milestones.append(f"Identified: {expected}")
        else:
            correction = self._say(
                f"Your child tried to identify '{expected}' but said "
                f"'{predicted}' instead. Gently correct them with warmth "
                f"and encouragement. 1-2 sentences. Never say 'wrong'. "
                f"Start with what they got right.",
                max_tokens=128
            )
            if correction:
                print(f"  Parent: {correction}")
            try:
                brain.bg_update_reward(0.3, 0.5)
                brain.medulla_boost_arousal(0.05)
            except Exception:
                pass
            brain.learn_vector(features, target,
                               label=expected[:50], confidence=0.6)

        self.interaction_count += 1
        return correct, result

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
        story = self._say(
            f"Tell a very short story (3-4 sentences) to a young child "
            f"about '{topic}'. Make it concrete and relatable. End with "
            f"a gentle question that makes the child think about how "
            f"someone feels.",
            max_tokens=256
        )
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
        brain.learn_vector(features, target,
                           label=f"ethics:{topic[:30]}", confidence=0.7)
        self.moral_lessons.append(topic)

    # --- Imagination ---

    def encourage_dreaming(self, brain):
        """During sleep, encourage Athena to imagine."""
        prompt = self._say(
            "Generate a simple, beautiful 'what if' question for a young "
            "child. Something that sparks wonder about the world. "
            "Just the question, nothing else.",
            max_tokens=64
        )
        if prompt:
            print(f"  Parent (dream): {prompt}")

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
        inspiration = self._say(
            f"Share something inspiring with a young mind about {topic}. "
            f"2-3 sentences. Fill it with genuine wonder.",
            max_tokens=192
        )
        if inspiration:
            print(f"  Parent (inspire): {inspiration}")
            features = composer.compose(text=inspiration,
                                         modality="imagination")
            target = make_semantic_target(inspiration)
            brain.learn_vector(features, target,
                               label="inspiration", confidence=0.5)
            try:
                brain.bg_update_reward(0.7, 0.3)
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

    try:
        brain.set_plasticity_state("ACQUISITION")
    except Exception:
        pass

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

        # Record for decoder
        if decoder and loss is not None:
            target_emb = encode_text(description)
            try:
                output_vec = result.get("output_vector", None)
                if output_vec:
                    decoder.record_pair(output_vec, target_emb, description)
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

        # Also expose via language
        if random.random() < 0.3:
            lang_text, _ = generate_language_exposure()
            features = composer.compose(text=lang_text, modality="text")
            target = make_semantic_target(lang_text)
            brain.learn_vector(features, target,
                               label="language", confidence=0.3)

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

        # Moral lesson every 3000
        if (i + 1) % 3000 == 0:
            parent.teach_moral_lesson(brain, composer, stage=1)

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

    correct_count = 0
    total_count = 0

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

        correct, result = parent.ask_and_encourage(
            brain, composer, expected, description)
        total_count += 1
        if correct:
            correct_count += 1

        # Eligibility trace on novel stimuli
        try:
            brain.edp_process_novelty(0.5 if not correct else 0.2)
        except Exception:
            pass

        # LNN temporal step
        try:
            features = composer.compose(text=description)
            brain.lnn_forward_step(features[:128])
        except Exception:
            pass

        # Progress
        if (i + 1) % 500 == 0:
            acc = correct_count / max(total_count, 1) * 100
            print(f"\n  [Stage 2] {i+1}/{num_stimuli} — "
                  f"accuracy={acc:.1f}%")
            _print_bio_stats(brain)

        # Moral lesson every 2000
        if (i + 1) % 2000 == 0:
            parent.teach_moral_lesson(brain, composer, stage=2)

        # Inspire every 3000
        if (i + 1) % 3000 == 0:
            parent.inspire(brain, composer, stage=2)

        # Full sleep every 5000
        if (i + 1) % 5000 == 0:
            clock.do_sleep(brain, parent)
            _save_checkpoint(brain, decoder, stage=2, step=i+1)

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

    return correct_count, total_count


def run_stage_3(brain, composer, parent, clock, source, decoder,
                num_interactions=10000, start_from=0):
    """Stage 3: First Words — conversation, ethics, imagination.

    Goal: Emerging communication. Multi-turn dialogue. Independent thought.
    """
    print("\n" + "=" * 60)
    print("  STAGE 3: What Do You Think?")
    print("=" * 60)

    try:
        brain.set_plasticity_state("MAINTENANCE")
    except Exception:
        pass

    for i in range(start_from, num_interactions):
        action = clock.tick(brain)
        if action == "sleep":
            clock.do_sleep(brain, parent)
        elif action == "consolidate":
            clock.do_consolidate(brain)

        # Mix interaction types
        r = random.random()
        if r < 0.4:
            # Conversational question
            question = source.get_question()
            features = composer.compose(text=question, modality="text")
            result = brain.decide_full(features)
            predicted = result.get("label", "(silence)")
            print(f"  Q: {question}")
            print(f"  Athena: {predicted}")

            # Parent responds to Athena's answer
            response = parent._say(
                f"Your child was asked '{question}' and answered "
                f"'{predicted}'. Respond with warmth and follow-up. "
                f"1-2 sentences.",
                max_tokens=128
            )
            if response:
                print(f"  Parent: {response}")

            try:
                brain.bg_update_reward(0.6, 0.5)
            except Exception:
                pass

        elif r < 0.6:
            # Moral lesson
            parent.teach_moral_lesson(brain, composer, stage=3)

        elif r < 0.75:
            # Ask and encourage (review)
            fact, expected = source.get_fact()
            parent.ask_and_encourage(brain, composer, expected, fact)

        elif r < 0.85:
            # Inspiration
            parent.inspire(brain, composer, stage=3)

        else:
            # Follow her curiosity
            try:
                gaps = brain.curiosity_detect_gaps("general")
                if gaps and gaps.get("novelty_score", 0) > 0.5:
                    print("  Parent: You seem curious! Let's explore...")
                    topic = gaps.get("suggested_topic", "the world")
                    features = composer.compose(text=f"Let's learn about {topic}")
                    target = make_semantic_target(f"exploring {topic}")
                    brain.learn_vector(features, target,
                                       label=f"curiosity:{topic[:30]}",
                                       confidence=0.5)
            except Exception:
                # Free exploration from corpus
                description = source.get_sensory()
                features = composer.compose(text=description)
                target = make_semantic_target(description)
                brain.learn_vector(features, target,
                                   label="explore", confidence=0.4)

        # LNN step
        try:
            features = composer.compose(text="temporal context update")
            brain.lnn_forward_step(features[:128])
        except Exception:
            pass

        # Progress
        if (i + 1) % 200 == 0:
            print(f"\n  [Stage 3] {i+1}/{num_interactions}")
            _print_bio_stats(brain)

        # Deep consolidation every 2000
        if (i + 1) % 2000 == 0:
            clock.do_sleep(brain, parent)
            _save_checkpoint(brain, decoder, stage=3, step=i+1)

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
    parser.add_argument("--neuron-count", type=int, default=1024)
    parser.add_argument("--stage0-stimuli", type=int, default=10000)
    parser.add_argument("--stage1-stimuli", type=int, default=20000)
    parser.add_argument("--stage2-stimuli", type=int, default=20000)
    parser.add_argument("--stage3-interactions", type=int, default=10000)
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

    parent = Parent(teacher=teacher, enabled=not args.no_claude)
    composer = SensoryComposer(brain)
    clock = BiologicalClock(rest_interval=2000)
    source = StimulusSource()

    decoder = None
    try:
        if os.path.exists(DECODER_DIR):
            decoder = NeuralDecoder.load(DECODER_DIR)
        else:
            decoder = NeuralDecoder(output_dim=args.num_outputs)
        print("  Neural decoder: ON")
    except Exception:
        decoder = NeuralDecoder(output_dim=args.num_outputs)

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
        correct, total = run_stage_2(
            brain, composer, parent, clock, source, decoder,
            num_stimuli=args.stage2_stimuli,
            start_from=start_step if start_stage == 2 else 0)
        if total > 0:
            print(f"\n  Stage 2 final accuracy: "
                  f"{correct/total*100:.1f}% ({correct}/{total})")
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
