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
_use_onnx = None  # None = not checked, True/False = checked


def _get_embed_model():
    global _embed_model
    if _embed_model is not None:
        return _embed_model
    from sentence_transformers import SentenceTransformer
    # Force CPU for embeddings — GPU VRAM is reserved for nimcp brain training.
    # PyTorch 2.8+: low_cpu_mem_usage=True (default) uses meta tensors which
    # crash on .to('cpu'). Disable to use real tensors from the start.
    _embed_model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='cpu',
                                       model_kwargs={'low_cpu_mem_usage': False})
    return _embed_model


def _try_onnx():
    """Try ONNX encoder first — faster and avoids loading PyTorch."""
    global _use_onnx
    if _use_onnx is not None:
        return _use_onnx
    try:
        from onnx_encoder import encode_text as _onnx_encode
        # Smoke test
        _onnx_encode("test")
        _use_onnx = True
        logger.info("Using ONNX encoder (faster, no PyTorch overhead)")
    except Exception as e:
        _use_onnx = False
        logger.info("ONNX encoder unavailable (%s), falling back to SentenceTransformer", e)
    return _use_onnx


_encode_cache = {}
_ENCODE_CACHE_MAX = 50000
_ENCODE_CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'athena', 'embed_cache.npz')


def _load_encode_cache():
    """Load persistent embedding cache from disk."""
    global _encode_cache
    import numpy as np
    try:
        if os.path.exists(_ENCODE_CACHE_PATH):
            data = np.load(_ENCODE_CACHE_PATH, allow_pickle=True)
            texts = data['texts']
            embeddings = data['embeddings']
            for t, e in zip(texts, embeddings):
                _encode_cache[str(t)] = np.asarray(e, dtype=np.float32)
            logger.info("Loaded %d cached embeddings from %s", len(_encode_cache), _ENCODE_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to load embedding cache: %s", e)


def _save_encode_cache():
    """Persist embedding cache to disk (atomic write to prevent corruption)."""
    import numpy as np
    try:
        os.makedirs(os.path.dirname(_ENCODE_CACHE_PATH), exist_ok=True)
        texts = list(_encode_cache.keys())
        embeddings = [_encode_cache[t] for t in texts]
        tmp_path = _ENCODE_CACHE_PATH + ".tmp.npz"
        np.savez_compressed(tmp_path,
                            texts=np.array(texts, dtype=object),
                            embeddings=np.array(embeddings))
        os.replace(tmp_path, _ENCODE_CACHE_PATH)
        logger.info("Saved %d embeddings to cache %s", len(texts), _ENCODE_CACHE_PATH)
    except Exception as e:
        logger.warning("Failed to save embedding cache: %s", e)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# Load cache on module import
_load_encode_cache()

def encode_text(text: str):
    """Encode text to 1024-dim embedding.

    Prefers ONNX encoder (GPU-accelerated, no PyTorch). Falls back to
    SentenceTransformer if ONNX is unavailable.
    Caches results — same text returns cached embedding instantly.
    """
    import numpy as np
    if text in _encode_cache:
        return _encode_cache[text].copy()

    if _try_onnx():
        from onnx_encoder import encode_text as onnx_encode
        result = onnx_encode(text)
    else:
        model = _get_embed_model()
        emb = model.encode(text, convert_to_numpy=True)
        result = np.asarray(emb, dtype=np.float32).ravel()

    if len(_encode_cache) < _ENCODE_CACHE_MAX:
        _encode_cache[text] = result.copy()
        # Persist every 500 new entries
        if len(_encode_cache) % 500 == 0:
            _save_encode_cache()
    return result


def batch_encode_texts(texts):
    """Batch-encode multiple texts to 1024-dim embeddings."""
    import numpy as np

    # Check cache for all texts first
    results = [None] * len(texts)
    uncached_indices = []
    for i, t in enumerate(texts):
        if t in _encode_cache:
            results[i] = _encode_cache[t].copy()
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        return results  # All cached — skip model entirely

    # Encode only uncached texts
    uncached_texts = [texts[i] for i in uncached_indices]
    if _try_onnx():
        from onnx_encoder import encode_batch as onnx_batch
        embs = onnx_batch(uncached_texts)
        new_embs = [np.asarray(e, dtype=np.float32).ravel() for e in embs]
    else:
        model = _get_embed_model()
        embs = model.encode(uncached_texts, convert_to_numpy=True, batch_size=64)
        new_embs = [np.asarray(e, dtype=np.float32).ravel() for e in embs]

    # Fill results and update cache
    for idx, emb in zip(uncached_indices, new_embs):
        results[idx] = emb
        if len(_encode_cache) < _ENCODE_CACHE_MAX:
            _encode_cache[texts[idx]] = emb.copy()

    # Persist cache to disk after batch encoding
    if uncached_indices:
        _save_encode_cache()

    return results


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
                 timeout: int = 180,
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
        """Call Claude via CLI with minimal startup overhead.

        Uses the Max subscription — no API costs.
        Optimized flags: no MCP servers, no tools, no session persistence,
        no project settings (CLAUDE.md), minimal system prompt.
        """
        mt = max_tokens or self.max_tokens

        # Build the full prompt with system context prepended
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\n{prompt}"

        cmd = [
            "claude", "-p", full_prompt,
            "--model", self.model,
            "--output-format", "text",
            "--system-prompt", "",      # override system prompt (skip CLAUDE.md)
            "--setting-sources", "",    # skip loading project/user settings entirely
            "--allowed-tools", "",      # disable all tools (pure text generation)
        ]

        # Clean env for claude subprocess:
        # - Remove CLAUDECODE to prevent nesting detection
        # - Remove CUDA_VISIBLE_DEVICES (we hide GPU from PyTorch, but claude doesn't need it)
        env = {k: v for k, v in os.environ.items()
               if k not in ("CLAUDECODE", "CUDA_VISIBLE_DEVICES")}

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
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
