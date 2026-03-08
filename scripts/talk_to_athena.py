#!/usr/bin/env python3
"""Talk to Athena — Interactive conversation with the trained brain.

Loads Athena from checkpoint, encodes your text to brain input features,
runs inference, and decodes the brain's output back to text using the
NeuralDecoder (embedding projector + vocabulary nearest-neighbor lookup).

Usage:
    python3 scripts/talk_to_athena.py
    python3 scripts/talk_to_athena.py --checkpoint checkpoints/athena/athena_developmental.bin
    python3 scripts/talk_to_athena.py --top-k 5  # show top 5 nearest matches
"""

import argparse
import logging
import math
import os
import sys
import time

import numpy as np

# Ensure scripts/ is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_decoder import NeuralDecoder, VocabularyBank
from claude_teacher import encode_text
from response_composer import ResponseComposer
from conversation_context import ConversationContext

logger = logging.getLogger("talk_to_athena")


class ConversationMemory:
    """Track recent conversation turns for context."""

    def __init__(self, max_turns=10):
        self.turns = []
        self.max_turns = max_turns

    def add(self, user_text, athena_response, confidence=0.0):
        self.turns.append({
            'user': user_text,
            'athena': athena_response,
            'confidence': confidence,
            'time': time.time()
        })
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_context_string(self, last_n=3):
        """Build context from recent turns for input enrichment."""
        if not self.turns:
            return ""
        recent = self.turns[-last_n:]
        parts = []
        for t in recent:
            parts.append(f"User: {t['user']}")
            if t['athena']:
                parts.append(f"Athena: {t['athena']}")
        return " | ".join(parts)

    def get_context_embedding(self, last_n=3):
        """Get embedding of recent conversation context."""
        ctx = self.get_context_string(last_n)
        if not ctx:
            return None
        return encode_text(ctx)

# ---------------------------------------------------------------------------
# Constants (must match teach_athena.py)
# ---------------------------------------------------------------------------

EMBED_DIM = 1024
BRAIN_INPUTS = 1024


def _normalize_embedding(e: np.ndarray) -> np.ndarray:
    """Normalize embedding values from [-1, 1] to [0, 1] for the brain."""
    return (e + 1.0) * 0.5


def _denormalize_embedding(e: np.ndarray) -> np.ndarray:
    """Reverse [0,1] normalization back to [-1, 1] for cosine similarity."""
    return e * 2.0 - 1.0


def tile_to_brain_input(embedding: np.ndarray) -> list:
    """Tile/truncate embedding to 1024-dim brain input."""
    e = np.asarray(embedding, dtype=np.float32).ravel()
    e = _normalize_embedding(e)
    reps = math.ceil(BRAIN_INPUTS / len(e))
    tiled = np.tile(e, reps)[:BRAIN_INPUTS]
    return tiled.tolist()


def extract_embedding_from_output(output_vector: np.ndarray,
                                   embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Extract embedding from brain output by averaging tiled copies.

    The brain was trained with 1024-dim embeddings tiled to fill 2048-dim output.
    To decode, we average the tiled copies to reduce noise, then denormalize.
    """
    v = np.asarray(output_vector, dtype=np.float32).ravel()
    num_outputs = len(v)

    if num_outputs <= embed_dim:
        return _denormalize_embedding(v)

    # Average the tiled copies
    n_full = num_outputs // embed_dim
    remainder = num_outputs % embed_dim
    chunks = [v[i * embed_dim:(i + 1) * embed_dim] for i in range(n_full)]
    if remainder > 0:
        # Partial last tile — only average the overlapping portion
        partial = v[n_full * embed_dim:]
        for i in range(n_full):
            chunks[i][:remainder] = (chunks[i][:remainder] + partial / n_full)

    averaged = np.mean(chunks, axis=0).astype(np.float32)
    return _denormalize_embedding(averaged)


def main():
    parser = argparse.ArgumentParser(description="Talk to Athena")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/athena/athena_initial.bin",
                        help="Path to brain checkpoint")
    parser.add_argument("--decoder-dir", type=str,
                        default="teaching_state/decoder",
                        help="Path to decoder state directory")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of nearest matches to show")
    parser.add_argument("--verbose", action="store_true",
                        help="Show timing and debug info")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )

    # -----------------------------------------------------------------------
    # Load brain with progress feedback
    # -----------------------------------------------------------------------
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    print("[1/3] Importing NIMCP library...", end=" ", flush=True)
    t0 = time.time()
    import nimcp
    print(f"done ({time.time() - t0:.1f}s)")
    print(f"      NIMCP v{nimcp.version()} | ABI hash: {nimcp.ABI_LAYOUT_HASH}")

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        print("Available checkpoints:")
        cp_dir = os.path.dirname(args.checkpoint) or "checkpoints/athena"
        if os.path.isdir(cp_dir):
            for f in sorted(os.listdir(cp_dir)):
                if f.endswith(".bin"):
                    path = os.path.join(cp_dir, f)
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    print(f"  {path}  ({size_mb:.0f} MB)")
        sys.exit(1)

    cp_size_mb = os.path.getsize(args.checkpoint) / (1024 * 1024)
    print(f"[2/3] Loading brain from checkpoint ({cp_size_mb:.0f} MB)...",
          end=" ", flush=True)
    t0 = time.time()
    brain = nimcp.Brain("athena", checkpoint=args.checkpoint)
    load_time = time.time() - t0
    print(f"done ({load_time:.1f}s)")

    # -----------------------------------------------------------------------
    # Load decoder (vocabulary bank for nearest-neighbor lookup)
    # -----------------------------------------------------------------------
    print("[3/3] Loading decoder...", end=" ", flush=True)
    decoder = None
    vocab = None

    if os.path.isdir(args.decoder_dir):
        try:
            decoder = NeuralDecoder.load(args.decoder_dir)
            vocab = decoder.vocabulary
            print(f"done ({len(vocab)} vocab entries, "
                  f"projector={'fitted' if decoder.projector.is_fitted else 'not fitted'})")
        except Exception as e:
            print(f"failed ({e})")
            decoder = None
    else:
        print("skipped (no decoder directory)")

    if vocab is None or len(vocab) == 0:
        print("WARNING: No vocabulary loaded. Responses will show raw similarity scores.")
        print("Run teach_athena.py first to build the vocabulary bank.")

    # -----------------------------------------------------------------------
    # Conversation loop
    # -----------------------------------------------------------------------
    # Check grounded language availability
    has_grounded = hasattr(brain, 'grounded_respond')
    has_generate = hasattr(brain, 'generate_text')

    print()
    print("=" * 60)
    print("  ATHENA CONVERSATION")
    print("  Type your message and press Enter.")
    print("  Commands: /quit /status /raw /similar <text> /teach <text> /probe /transcript /context")
    print("=" * 60)
    print(f"  Inference pipeline:")
    print(f"    System 1 (fast): grounded_respond {'[available]' if has_grounded else '[unavailable]'}")
    print(f"    System 2 (full): decide_full + "
          f"{'generate_text' if has_generate else 'vocab lookup'}")
    if vocab and len(vocab) > 0:
        print(f"  Vocabulary: {len(vocab)} entries")
    if decoder and decoder.projector.is_fitted:
        if decoder.projector.output_dim != 2048:
            print(f"  Note: Projector trained on {decoder.projector.output_dim}-dim "
                  f"(brain outputs 2048-dim). Using direct embedding extraction.")
    print()

    show_raw = False
    turn = 0
    memory = ConversationMemory(max_turns=10)
    composer = ResponseComposer(max_sentences=3, min_salience=0.15)
    ctx_engine = ConversationContext(max_turns=50)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if user_input.lower() == "/raw":
            show_raw = not show_raw
            print(f"[Raw output display: {'ON' if show_raw else 'OFF'}]")
            continue

        if user_input.lower() == "/status":
            print(f"[Turn {turn} | Vocab: {len(vocab) if vocab else 0} | "
                  f"Checkpoint: {args.checkpoint}]")
            continue

        if user_input.lower().startswith("/similar "):
            query = user_input[9:].strip()
            if query and vocab:
                q_emb = encode_text(query)
                results = vocab.decode(q_emb, top_k=args.top_k)
                print(f"[Nearest to '{query}':]")
                for text, sim in results:
                    print(f"  {sim:.4f}  {text[:80]}")
            continue

        if user_input.lower() == "/probe":
            try:
                stats = brain.get_stats()
                print(f"[Brain stats: {stats}]")
            except Exception as e:
                print(f"[Probe error: {e}]")
            continue

        if user_input.lower() == "/transcript":
            try:
                transcript = brain.get_transcript()
                if transcript:
                    print(f"[Cognitive transcript ({len(transcript)} entries):]")
                    for entry in transcript:
                        mod = entry.get('module', '?')
                        sal = entry.get('salience', 0.0)
                        conf = entry.get('confidence', 0.0)
                        summary = entry.get('summary', '')
                        print(f"  {mod:20s}  sal={sal:.3f}  "
                              f"conf={conf:.3f}  {summary}")
                else:
                    print("[No transcript available — run a query first]")
            except Exception as e:
                print(f"[Transcript error: {e}]")
            continue

        if user_input.lower() == "/context":
            summary = ctx_engine.get_summary()
            print(f"[Conversation context:]")
            print(f"  Turns: {summary['turns']}")
            print(f"  Active topic: {', '.join(summary['active_topic']) if summary['active_topic'] else '(none)'}")
            print(f"  Topic threads: {summary['threads']}")
            print(f"  Confidence trend: {summary['confidence_trend']:+.2f}")
            continue

        if user_input.lower().startswith("/teach "):
            # Teach Athena: /teach <text> — learn this text as input AND target
            text = user_input[7:].strip()
            if text:
                emb = encode_text(text)
                features = tile_to_brain_input(emb)
                target_emb = _normalize_embedding(emb)
                reps = math.ceil(2048 / len(target_emb))
                target = np.tile(target_emb, reps)[:2048].tolist()
                try:
                    loss = brain.learn_vector(features, target, label=text[:50])
                    if vocab:
                        vocab.add(text, emb)
                    print(f"[Taught: '{text[:60]}' | loss={loss:.4f}]")
                except Exception as e:
                    print(f"[Teach error: {e}]")
            continue

        # ------- Process conversation context (Phase 3) -------
        # We process context early so both System 1 and System 2 can use it.
        # For System 1, we pass an empty transcript/decision — the context
        # engine still detects greetings, follow-ups, topic shifts from text.
        ctx_signals = ctx_engine.process_turn(user_input, [], {})

        # ------- System 1: Try grounded language (fast path) -------
        grounded_ok = False
        t0 = time.time()
        try:
            grounded = brain.grounded_respond(user_input)
            grounded_time = time.time() - t0
            grounded_text = grounded.get("response", "")
            grounded_conf = grounded.get("confidence", 0.0)

            if grounded_text and grounded_conf >= 0.3:
                grounded_ok = True
                print(f"Athena: {grounded_text}")
                print(f"        [grounded | conf={grounded_conf:.4f} | "
                      f"{grounded_time*1000:.0f}ms]")
                if args.verbose:
                    phase = ctx_signals.conversation_phase
                    print(f"        [System 1 fast path — grounded language | "
                          f"phase={phase}]")
        except Exception:
            pass  # Grounded language not initialized — fall through

        if grounded_ok:
            ctx_engine.record_response(grounded_text)
            memory.add(user_input, grounded_text, grounded_conf)
            print()
            turn += 1
            continue

        # ------- System 2: Full cognitive pipeline (slow path) -------
        athena_response = ""

        t0 = time.time()
        input_embedding = encode_text(user_input)
        # Blend with conversation context for continuity
        ctx_emb = memory.get_context_embedding(last_n=3)
        if ctx_emb is not None:
            # 80% current input, 20% context
            input_embedding = 0.8 * input_embedding + 0.2 * ctx_emb
            input_embedding = input_embedding / max(np.linalg.norm(input_embedding), 1e-8)
        features = tile_to_brain_input(input_embedding)
        encode_time = time.time() - t0

        t0 = time.time()
        try:
            result = brain.decide_full(features)
        except Exception as e:
            print(f"[Brain error: {e}]")
            continue
        infer_time = time.time() - t0

        output_vector = result.get("output_vector", [])
        label = result.get("label", "")
        confidence = result.get("confidence", 0.0)

        if args.verbose or show_raw:
            print(f"[System 2 | encode={encode_time*1000:.0f}ms, "
                  f"infer={infer_time*1000:.0f}ms, "
                  f"label='{label}', conf={confidence:.4f}]")

        if show_raw and output_vector:
            ov = np.array(output_vector)
            print(f"[output: len={len(ov)}, min={ov.min():.4f}, max={ov.max():.4f}, "
                  f"mean={ov.mean():.4f}, std={ov.std():.4f}]")

        # ---- Gather raw decode candidates (used by composer + verbose) ----
        vocab_response = ""
        gen_text = ""

        # Try generate_text from brain output (grounded production)
        if output_vector:
            try:
                gen_result = brain.generate_text(list(output_vector))
                gt = gen_result.get("text", "")
                if gt and len(gt.strip()) > 2:
                    gen_text = gt
            except Exception:
                pass

        # Try vocabulary nearest-neighbor decode
        if output_vector and vocab and len(vocab) > 0:
            brain_emb = extract_embedding_from_output(np.array(output_vector))
            results = vocab.decode(brain_emb, top_k=args.top_k)
            best_text, best_sim = results[0]
            vocab_response = best_text

        # ---- Get cognitive transcript ----
        transcript = []
        try:
            transcript = brain.get_transcript() or []
        except Exception:
            pass

        # ---- Re-process context with transcript + decision (Phase 3) ----
        # The initial process_turn used empty transcript. Now update with
        # real cognitive data for accurate trajectory tracking.
        if transcript or result:
            # Update the last turn's data in-place
            if ctx_engine.turns:
                last_turn = ctx_engine.turns[-1]
                last_turn["confidence"] = confidence
                if transcript:
                    best_entry = max(transcript,
                                     key=lambda e: e.get("salience", 0))
                    last_turn["top_salience"] = best_entry.get("salience", 0.0)
                    last_turn["dominant_module"] = best_entry.get("module", "")
                # Re-compute trajectory with real confidence
                ctx_engine._confidences[-1] = confidence
                valence = ctx_engine._extract_valence(transcript)
                ctx_engine._valences[-1] = valence
                ctx_signals.confidence_trend = ctx_engine._compute_trend(
                    ctx_engine._confidences)
                ctx_signals.emotional_trend = ctx_engine._compute_trend(
                    ctx_engine._valences)

        # ---- Compose response from transcript + decoded signals ----
        # The composer synthesizes from cognitive module outputs,
        # falling back to vocab/generated text when available.
        best_base = gen_text or vocab_response
        composed = composer.compose(
            transcript, result,
            user_input=user_input,
            vocab_response=best_base if confidence > 0.1 else "",
            context=ctx_signals)

        if composed:
            athena_response = composed
            print(f"Athena: {composed}")
            # Show source attribution
            sources = []
            for entry in sorted(transcript,
                                key=lambda e: e.get('salience', 0),
                                reverse=True)[:3]:
                if entry.get('salience', 0) > 0.15:
                    sources.append(entry['module'])
            src_str = ", ".join(sources) if sources else "cognitive"
            # Phase 3: Show context state in attribution
            phase = ctx_signals.conversation_phase
            ctx_tag = ""
            if ctx_signals.is_followup:
                ctx_tag = " followup"
            elif ctx_signals.is_topic_shift:
                ctx_tag = " topic-shift"
            elif ctx_signals.is_returning_topic:
                ctx_tag = " returning"
            print(f"        [composed from: {src_str} | "
                  f"conf={confidence:.4f} | "
                  f"phase={phase}{ctx_tag} | "
                  f"{(encode_time + infer_time)*1000:.0f}ms]")
        elif gen_text:
            athena_response = gen_text
            print(f"Athena: {gen_text}")
            gen_conf = gen_result.get("confidence", 0.0) if 'gen_result' in dir() else 0.0
            print(f"        [generated | conf={confidence:.4f}]")
        elif vocab_response:
            athena_response = vocab_response
            print(f"Athena: {vocab_response}")
            print(f"        [vocab lookup | similarity={best_sim:.4f}]")
        elif label:
            athena_response = label
            print(f"Athena: [{label}] (confidence: {confidence:.4f})")
        else:
            athena_response = ""
            print("Athena: [no decodable output]")

        # Show verbose details
        if args.verbose or show_raw:
            if vocab_response:
                print(f"        [vocab: '{vocab_response[:60]}' sim={best_sim:.4f}]")
                if args.top_k > 1 and len(results) > 1:
                    print("        alternatives:")
                    for text, sim in results[1:]:
                        print(f"          {sim:.4f}  {text[:80]}")
            if transcript:
                print("        [cognitive transcript:]")
                for entry in transcript:
                    mod = entry.get('module', '?')
                    sal = entry.get('salience', 0.0)
                    conf = entry.get('confidence', 0.0)
                    summary = entry.get('summary', '')
                    if sal > 0.1:
                        print(f"          {mod:20s}  sal={sal:.2f}  "
                              f"conf={conf:.2f}  {summary[:60]}")

        ctx_engine.record_response(athena_response)
        memory.add(user_input, athena_response, confidence)
        print()
        turn += 1


if __name__ == "__main__":
    main()
