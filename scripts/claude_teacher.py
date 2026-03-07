#!/usr/bin/env python3
"""Claude Teacher — Uses the Claude CLI (Max subscription) to generate lessons for Athena.

Generates (lesson_text, target_embedding) pairs, evaluates Athena's responses,
and conducts interactive teaching dialogues.  Uses `claude -p` (non-interactive
mode) so all calls run through your existing Max plan — no separate API costs.
"""

import hashlib
import json
import logging
import os
import subprocess
import time
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    import torch
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)
    return _embed_model


def encode_text(text: str):
    """Encode text to 1024-dim embedding using sentence-transformers (BAAI/bge-large-en-v1.5)."""
    import numpy as np
    model = _get_embed_model()
    emb = model.encode(text, convert_to_numpy=True)
    return np.asarray(emb, dtype=np.float32).ravel()


def batch_encode_texts(texts):
    """Batch-encode multiple texts to 1024-dim embeddings — ~10x faster than encoding one-at-a-time."""
    import numpy as np
    model = _get_embed_model()
    embs = model.encode(texts, convert_to_numpy=True, batch_size=64)
    return [np.asarray(e, dtype=np.float32).ravel() for e in embs]


# ---------------------------------------------------------------------------
# Lesson specification
# ---------------------------------------------------------------------------

@dataclass
class LessonSpec:
    domain: str
    topic: str
    difficulty: float  # 0.0 = trivial, 1.0 = expert
    context: str = ""
    lesson_type: str = "teach"  # teach, quiz, dialogue


# ---------------------------------------------------------------------------
# Curriculum definitions for each developmental stage
# ---------------------------------------------------------------------------

STAGE_CURRICULA = {
    0: {  # Birth — sensory grounding
        "name": "Birth",
        "domains": ["perception", "patterns"],
        "topics": [
            LessonSpec("perception", "colors and shapes", 0.1),
            LessonSpec("perception", "spatial relationships", 0.15),
            LessonSpec("perception", "size and quantity", 0.1),
            LessonSpec("patterns", "simple sequences", 0.15),
            LessonSpec("patterns", "repetition and rhythm", 0.2),
            LessonSpec("patterns", "matching and sorting", 0.15),
        ],
    },
    1: {  # Babbling — phoneme patterns, word boundaries
        "name": "Babbling",
        "domains": ["language", "vocabulary"],
        "topics": [
            LessonSpec("language", "common sounds and phonemes", 0.2),
            LessonSpec("language", "syllable patterns", 0.25),
            LessonSpec("language", "word boundaries", 0.25),
            LessonSpec("vocabulary", "concrete nouns", 0.2),
            LessonSpec("vocabulary", "action verbs", 0.25),
            LessonSpec("vocabulary", "descriptive words", 0.25),
        ],
    },
    2: {  # First Words — simple concepts
        "name": "First Words",
        "domains": ["language", "concepts", "reasoning"],
        "topics": [
            LessonSpec("language", "simple sentences", 0.3),
            LessonSpec("concepts", "categories and types", 0.3),
            LessonSpec("concepts", "cause and effect", 0.35),
            LessonSpec("reasoning", "basic comparisons", 0.3),
            LessonSpec("reasoning", "simple analogies", 0.35),
            LessonSpec("concepts", "temporal concepts", 0.35),
        ],
    },
    3: {  # Sentences — grammar, reasoning
        "name": "Sentences",
        "domains": ["language", "reasoning", "knowledge"],
        "topics": [
            LessonSpec("language", "complex sentences", 0.5),
            LessonSpec("language", "grammar patterns", 0.45),
            LessonSpec("reasoning", "logical deduction", 0.5),
            LessonSpec("reasoning", "if-then reasoning", 0.5),
            LessonSpec("knowledge", "basic science facts", 0.45),
            LessonSpec("knowledge", "geography and culture", 0.45),
        ],
    },
    4: {  # Conversation — dialogue, Q&A
        "name": "Conversation",
        "domains": ["dialogue", "knowledge", "reasoning"],
        "topics": [
            LessonSpec("dialogue", "question answering", 0.6, lesson_type="dialogue"),
            LessonSpec("dialogue", "topic discussion", 0.65, lesson_type="dialogue"),
            LessonSpec("knowledge", "history and events", 0.6),
            LessonSpec("knowledge", "science concepts", 0.65),
            LessonSpec("reasoning", "abstract reasoning", 0.7),
            LessonSpec("reasoning", "multi-step problems", 0.7),
        ],
    },
    5: {  # Self-Directed — deep learning
        "name": "Self-Directed",
        "domains": ["philosophy", "creativity", "meta-cognition"],
        "topics": [
            LessonSpec("philosophy", "ethics and values", 0.8),
            LessonSpec("philosophy", "epistemology", 0.8),
            LessonSpec("creativity", "creative writing", 0.75, lesson_type="dialogue"),
            LessonSpec("creativity", "problem solving", 0.8, lesson_type="dialogue"),
            LessonSpec("meta-cognition", "learning strategies", 0.8),
            LessonSpec("meta-cognition", "self-reflection", 0.85, lesson_type="dialogue"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Claude Teacher
# ---------------------------------------------------------------------------

class ClaudeTeacher:
    """Uses the `claude` CLI (Max subscription) to generate lessons and evaluate responses.

    All calls go through `claude -p` (non-interactive one-shot mode), which runs
    on your existing Max plan.  No separate API key or billing required.
    """

    def __init__(self, *,
                 model: str = "sonnet",
                 cache_size: int = 200,
                 timeout: int = 60,
                 max_tokens: int = 1024,
                 # Legacy kwargs accepted but ignored (for load_state compat)
                 routine_model: str = "",
                 eval_model: str = "",
                 token_budget: int = 0,
                 max_api_calls_per_minute: int = 0):
        self.model = model
        self._cache: dict[str, tuple[str, any]] = {}
        self._cache_max = cache_size
        self.timeout = timeout
        self.max_tokens = max_tokens

        # Teaching context
        self.mastered_concepts: list[str] = []
        self.current_stage = 0
        self.lessons_taught = 0
        self.calls_made = 0

    def _call_claude(self, prompt: str, system: str = "",
                     max_tokens: int | None = None) -> str:
        """Call Claude via CLI: `claude -p "prompt" --model model`.

        Uses the Max subscription — no API costs.
        """
        mt = max_tokens or self.max_tokens

        # Build the full prompt with system context prepended
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\n{prompt}"

        cmd = [
            "claude", "-p", full_prompt,
            "--model", self.model,
            "--max-tokens", str(mt),
            "--output-format", "text",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                raise RuntimeError(f"claude CLI failed (rc={result.returncode}): {stderr}")
            text = result.stdout.strip()
            if not text:
                raise RuntimeError("claude CLI returned empty output")
            self.calls_made += 1
            return text
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"claude CLI timed out after {self.timeout}s")
        except FileNotFoundError:
            raise RuntimeError("claude CLI not found — is Claude Code installed?")

    def _cache_key(self, domain: str, topic: str, difficulty: float) -> str:
        raw = f"{domain}:{topic}:{difficulty:.2f}"
        return hashlib.md5(raw.encode()).hexdigest()

    def generate_lesson(self, spec: LessonSpec) -> tuple[str, "np.ndarray"]:
        """Generate a lesson appropriate to Athena's current developmental level.

        Returns (lesson_text, target_embedding).
        """
        import numpy as np

        # Check cache
        key = self._cache_key(spec.domain, spec.topic, spec.difficulty)
        if key in self._cache:
            logger.debug("Cache hit for %s/%s", spec.domain, spec.topic)
            return self._cache[key]

        system = (
            "You are a teacher for an AI student named Athena. "
            "Athena is learning progressively, like a human child. "
            f"Current developmental stage: {self.current_stage}. "
            f"Mastered concepts: {', '.join(self.mastered_concepts[-20:]) or 'none yet'}. "
            "Generate a clear, focused lesson. Respond with ONLY the lesson content — "
            "no meta-commentary about being a teacher."
        )

        prompt = (
            f"Domain: {spec.domain}\n"
            f"Topic: {spec.topic}\n"
            f"Difficulty: {spec.difficulty:.1f}/1.0\n"
        )
        if spec.context:
            prompt += f"Context: {spec.context}\n"
        prompt += (
            "\nGenerate a short teaching lesson (2-4 sentences) appropriate "
            "for this difficulty level. Be concrete and specific."
        )

        try:
            text = self._call_claude(prompt, system=system, max_tokens=256)
        except Exception as e:
            logger.warning("Claude CLI failed: %s — using fallback", e)
            text = self._fallback_lesson(spec)

        embedding = encode_text(text)

        # Cache result
        if len(self._cache) >= self._cache_max:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[key] = (text, embedding)

        self.lessons_taught += 1
        return text, embedding

    def evaluate_response(self, lesson_text: str, athena_response: str) -> tuple[float, str]:
        """Evaluate Athena's decoded response to a lesson.

        Returns (score 0.0-1.0, feedback_text).
        """
        prompt = (
            f"A student named Athena was given this lesson:\n\"{lesson_text}\"\n\n"
            f"Athena's response was:\n\"{athena_response}\"\n\n"
            "Rate the response on a scale of 0.0 to 1.0 where:\n"
            "- 0.0 = completely incoherent/unrelated\n"
            "- 0.3 = shows some relevant patterns but not meaningful\n"
            "- 0.5 = partially relevant, shows understanding of some concepts\n"
            "- 0.7 = good understanding, mostly correct\n"
            "- 1.0 = excellent, fully correct response\n\n"
            "Respond in this exact format:\n"
            "SCORE: <number>\n"
            "FEEDBACK: <one sentence>"
        )

        try:
            response = self._call_claude(prompt, max_tokens=128)
            lines = response.strip().split("\n")
            score = 0.0
            feedback = "No feedback"
            for line in lines:
                if line.startswith("SCORE:"):
                    score = float(line.split(":")[1].strip())
                    score = max(0.0, min(1.0, score))
                elif line.startswith("FEEDBACK:"):
                    feedback = line.split(":", 1)[1].strip()
            return score, feedback
        except Exception as e:
            logger.warning("Evaluation failed: %s", e)
            return 0.0, f"Evaluation error: {e}"

    def generate_dialogue_turn(self, context: str,
                               athena_response: str) -> tuple[str, "np.ndarray"]:
        """Generate the next dialogue turn in response to Athena's output.

        Returns (next_prompt_text, target_embedding).
        """
        import numpy as np

        system = (
            "You are having a conversation with an AI student named Athena. "
            "Respond naturally and teach through dialogue. Keep responses "
            "concise (1-2 sentences). Build on what Athena said."
        )

        prompt = (
            f"Conversation so far:\n{context}\n\n"
            f"Athena said: \"{athena_response}\"\n\n"
            "Respond with your next teaching turn."
        )

        try:
            text = self._call_claude(prompt, system=system, max_tokens=256)
        except Exception as e:
            logger.warning("Dialogue generation failed: %s", e)
            text = "Tell me more about what you think."

        embedding = encode_text(text)
        return text, embedding

    def get_curriculum(self, stage: int) -> list[LessonSpec]:
        """Return curriculum appropriate to developmental stage."""
        if stage not in STAGE_CURRICULA:
            stage = min(stage, max(STAGE_CURRICULA.keys()))
        return list(STAGE_CURRICULA[stage]["topics"])

    def get_stage_name(self, stage: int) -> str:
        if stage in STAGE_CURRICULA:
            return STAGE_CURRICULA[stage]["name"]
        return f"Stage {stage}"

    def _fallback_lesson(self, spec: LessonSpec) -> str:
        """Generate a simple lesson without the CLI (for timeout / error fallback)."""
        fallbacks = {
            "perception": "Look at the world around you. Objects have shapes — circles are round, squares have four equal sides. Colors help us tell things apart.",
            "patterns": "Patterns repeat in predictable ways. The sequence 1, 2, 3 continues with 4. Red, blue, red, blue continues with red.",
            "language": "Words are building blocks of communication. Every sentence has a subject and a verb. 'The cat sits' is a complete thought.",
            "vocabulary": "A 'tree' is a tall plant with a trunk, branches, and leaves. Trees provide shade, oxygen, and homes for birds.",
            "concepts": "Things can be grouped into categories. Dogs, cats, and fish are all animals. Apples, bananas, and oranges are all fruits.",
            "reasoning": "If it is raining, the ground gets wet. We see wet ground. Therefore, it probably rained recently. This is called deduction.",
            "knowledge": "The Earth orbits the Sun. One complete orbit takes about 365 days, which we call a year. The Earth also rotates on its axis once per day.",
            "dialogue": "A good conversation involves listening and responding. When someone asks a question, think about what they want to know before answering.",
            "philosophy": "Ethics is the study of right and wrong. Different people may have different values, but some principles like fairness are widely shared.",
            "creativity": "Creativity means combining ideas in new ways. A story needs a beginning, middle, and end. Characters face challenges and grow.",
            "meta-cognition": "Learning how you learn is called metacognition. When something is hard, try breaking it into smaller pieces.",
        }
        return fallbacks.get(spec.domain, f"Today we learn about {spec.topic} in the field of {spec.domain}.")

    def save_state(self, path: str):
        """Save teacher state (mastered concepts, stats)."""
        state = {
            "mastered_concepts": self.mastered_concepts,
            "current_stage": self.current_stage,
            "lessons_taught": self.lessons_taught,
            "calls_made": self.calls_made,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_state(cls, path: str, **kwargs) -> "ClaudeTeacher":
        """Load teacher state from file."""
        teacher = cls(**kwargs)
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            teacher.mastered_concepts = state.get("mastered_concepts", [])
            teacher.current_stage = state.get("current_stage", 0)
            teacher.lessons_taught = state.get("lessons_taught", 0)
            teacher.calls_made = state.get("calls_made", 0)
        return teacher
