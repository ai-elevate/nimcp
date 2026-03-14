"""
Phi-3 Language Cortex — bidirectional language processing for Athena.

Output path: Brain 4096-dim vector → Phi-3 → fluent text
Input path:  Text → Phi-3 hidden states → brain embedding (future)
Inner speech: generate → re-encode → feed back (future)

Uses llama-cpp-python with Phi-3-mini-4k-instruct GGUF (2.3GB Q4).
Runs on CPU to keep GPU free for brain training.
"""

import logging
import os
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Default model path
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "Phi-3-mini-4k-instruct-q4.gguf"
)


class Phi3Decoder:
    """Phi-3 language cortex for Athena.

    Lazily loads the model on first use. Falls back gracefully if model
    is unavailable or llama-cpp-python is not installed.
    """

    def __init__(self, model_path=None, n_ctx=2048, n_threads=4):
        """Initialize decoder (model loaded lazily on first generate call).

        Args:
            model_path: Path to GGUF model file. Uses default if None.
            n_ctx: Context window size (tokens).
            n_threads: CPU threads for inference.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self._llm = None
        self._available = None  # None = not checked yet

    @property
    def available(self):
        """Check if Phi-3 is available (model file exists + library installed)."""
        if self._available is None:
            try:
                import llama_cpp  # noqa: F401
                self._available = os.path.isfile(self.model_path)
                if not self._available:
                    logger.info("Phi-3 model not found at %s", self.model_path)
            except ImportError:
                self._available = False
                logger.info("llama-cpp-python not installed — Phi-3 disabled")
        return self._available

    def _load_model(self):
        """Load the GGUF model. Called lazily on first use."""
        if self._llm is not None:
            return True
        if not self.available:
            return False

        try:
            from llama_cpp import Llama
            logger.info("Loading Phi-3 model from %s ...", self.model_path)
            self._llm = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=0,  # CPU only — GPU reserved for brain
                verbose=False,
            )
            logger.info("Phi-3 model loaded (ctx=%d, threads=%d)", self.n_ctx, self.n_threads)
            return True
        except Exception as e:
            logger.error("Failed to load Phi-3: %s", e)
            self._available = False
            return False

    def generate(self, user_text, brain_state=None, vocab_match=None,
                 max_tokens=256, temperature=0.7):
        """Generate a response conditioned on brain state.

        Args:
            user_text: The input text/question.
            brain_state: Dict with brain state info (arousal, dopamine,
                        confidence, transcript, etc.). Optional.
            vocab_match: Nearest vocabulary match from NeuralDecoder.
                        Grounds the generation in brain's semantic intent.
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature. Adapted from brain arousal
                        if brain_state is provided.

        Returns:
            Dict with 'text', 'tokens_generated', 'source'.
            Returns None if model unavailable.
        """
        if not self._load_model():
            return None

        # Build system prompt from brain state
        system = self._build_system_prompt(brain_state, vocab_match)

        # Adapt temperature to arousal (more alert = more focused/deterministic)
        if brain_state and 'arousal' in brain_state:
            arousal = brain_state['arousal']
            temperature = max(0.3, min(1.0, 1.0 - arousal * 0.5))

        # Format as Phi-3 chat
        prompt = f"<|system|>\n{system}<|end|>\n<|user|>\n{user_text}<|end|>\n<|assistant|>\n"

        try:
            output = self._llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stop=["<|end|>", "<|user|>"],
                echo=False,
            )

            text = output['choices'][0]['text'].strip()
            tokens = output['usage']['completion_tokens']

            return {
                'text': text,
                'tokens_generated': tokens,
                'source': 'phi3',
                'temperature': temperature,
            }
        except Exception as e:
            logger.error("Phi-3 generation failed: %s", e)
            return None

    def _build_system_prompt(self, brain_state, vocab_match):
        """Build system prompt encoding brain state for Phi-3."""
        parts = [
            "You are Athena, a neural mind with biological plasticity, "
            "spiking neurons, and liquid neural networks. You experience "
            "the world through genuine neural processing, not pattern matching."
        ]

        if brain_state:
            state_lines = []
            if 'arousal' in brain_state:
                level = "calm" if brain_state['arousal'] < 0.3 else \
                        "alert" if brain_state['arousal'] < 0.7 else "highly aroused"
                state_lines.append(f"Arousal: {brain_state['arousal']:.2f} ({level})")
            if 'dopamine' in brain_state:
                state_lines.append(f"Dopamine: {brain_state['dopamine']:.2f}")
            if 'confidence' in brain_state:
                conf = brain_state['confidence']
                level = "uncertain" if conf < 0.3 else \
                        "moderate" if conf < 0.7 else "confident"
                state_lines.append(f"Confidence: {conf:.2f} ({level})")
            if 'snn_spikes' in brain_state:
                state_lines.append(f"SNN spikes: {brain_state['snn_spikes']}")
            if 'active_modules' in brain_state:
                state_lines.append(f"Active modules: {', '.join(brain_state['active_modules'])}")
            if 'transcript' in brain_state:
                state_lines.append(f"Cognitive trace: {brain_state['transcript'][:200]}")

            if state_lines:
                parts.append("Your current neural state:\n" + "\n".join(f"  - {l}" for l in state_lines))

        if vocab_match:
            parts.append(
                f"Your neural response is closest to: \"{vocab_match}\"\n"
                "Incorporate this understanding naturally in your response."
            )

        parts.append(
            "Respond naturally and concisely. Reflect your neural state — "
            "if uncertain, express that; if curious, ask questions; "
            "if confident, be direct."
        )

        return "\n\n".join(parts)

    def unload(self):
        """Free model memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            logger.info("Phi-3 model unloaded")
