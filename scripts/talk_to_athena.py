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

logger = logging.getLogger("talk_to_athena")

# ---------------------------------------------------------------------------
# Constants (must match teach_athena.py)
# ---------------------------------------------------------------------------

EMBED_DIM = 384
BRAIN_INPUTS = 1024


def _normalize_embedding(e: np.ndarray) -> np.ndarray:
    """Normalize embedding values from [-1, 1] to [0, 1] for the brain."""
    return (e + 1.0) * 0.5


def _denormalize_embedding(e: np.ndarray) -> np.ndarray:
    """Reverse [0,1] normalization back to [-1, 1] for cosine similarity."""
    return e * 2.0 - 1.0


def tile_to_brain_input(embedding: np.ndarray) -> list:
    """Tile a 384-dim embedding to 1024-dim brain input."""
    e = np.asarray(embedding, dtype=np.float32).ravel()
    e = _normalize_embedding(e)
    reps = math.ceil(BRAIN_INPUTS / len(e))
    tiled = np.tile(e, reps)[:BRAIN_INPUTS]
    return tiled.tolist()


def extract_embedding_from_output(output_vector: np.ndarray,
                                   embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Extract embedding from brain output by averaging tiled copies.

    The brain was trained with targets that tile 384-dim embeddings to 2048.
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
    # Load brain
    # -----------------------------------------------------------------------
    print("Loading Athena from checkpoint...")
    t0 = time.time()

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import nimcp

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

    brain = nimcp.Brain("athena", checkpoint=args.checkpoint)
    load_time = time.time() - t0
    print(f"Brain loaded in {load_time:.1f}s")

    # -----------------------------------------------------------------------
    # Load decoder (vocabulary bank for nearest-neighbor lookup)
    # -----------------------------------------------------------------------
    decoder = None
    vocab = None

    if os.path.isdir(args.decoder_dir):
        print("Loading decoder...")
        try:
            decoder = NeuralDecoder.load(args.decoder_dir)
            vocab = decoder.vocabulary
            print(f"Decoder loaded: vocab={len(vocab)} entries, "
                  f"projector_fitted={decoder.projector.is_fitted}")
        except Exception as e:
            print(f"Warning: Could not load decoder: {e}")
            decoder = None

    if vocab is None or len(vocab) == 0:
        print("WARNING: No vocabulary loaded. Responses will show raw similarity scores.")
        print("Run teach_athena.py first to build the vocabulary bank.")

    # -----------------------------------------------------------------------
    # Conversation loop
    # -----------------------------------------------------------------------
    print()
    print("=" * 60)
    print("  ATHENA CONVERSATION")
    print("  Type your message and press Enter.")
    print("  Commands: /quit /status /raw /similar <text> /teach <text> /probe")
    print("=" * 60)
    if vocab and len(vocab) > 0:
        print(f"  Vocabulary: {len(vocab)} entries")
    if decoder and decoder.projector.is_fitted:
        if decoder.projector.output_dim != 2048:
            print(f"  Note: Projector trained on {decoder.projector.output_dim}-dim "
                  f"(brain outputs 2048-dim). Using direct embedding extraction.")
    print()

    show_raw = False
    turn = 0

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

        # Encode input
        t0 = time.time()
        input_embedding = encode_text(user_input)
        features = tile_to_brain_input(input_embedding)
        encode_time = time.time() - t0

        # Run brain inference
        t0 = time.time()
        try:
            result = brain.decide_full(features)
        except Exception as e:
            print(f"[Brain error: {e}]")
            continue
        infer_time = time.time() - t0

        # Extract output
        output_vector = result.get("output_vector", [])
        label = result.get("label", "")
        confidence = result.get("confidence", 0.0)

        if args.verbose or show_raw:
            print(f"[encode={encode_time*1000:.0f}ms, infer={infer_time*1000:.0f}ms, "
                  f"label='{label}', conf={confidence:.4f}]")

        if show_raw and output_vector:
            ov = np.array(output_vector)
            print(f"[output: len={len(ov)}, min={ov.min():.4f}, max={ov.max():.4f}, "
                  f"mean={ov.mean():.4f}, std={ov.std():.4f}]")

        # Decode brain output to text
        if output_vector and vocab and len(vocab) > 0:
            brain_emb = extract_embedding_from_output(np.array(output_vector))

            results = vocab.decode(brain_emb, top_k=args.top_k)

            # Show best match as "Athena's response"
            best_text, best_sim = results[0]
            print(f"Athena: {best_text}")
            print(f"        [similarity: {best_sim:.4f}]")

            # Show alternatives
            if args.top_k > 1 and len(results) > 1:
                print("        alternatives:")
                for text, sim in results[1:]:
                    print(f"          {sim:.4f}  {text[:80]}")

            # Also compute direct cosine similarity with input
            cos_sim = np.dot(brain_emb, input_embedding) / (
                max(np.linalg.norm(brain_emb), 1e-8) *
                max(np.linalg.norm(input_embedding), 1e-8))
            if args.verbose:
                print(f"        [input↔output cosine: {cos_sim:.4f}]")

        elif label:
            print(f"Athena: [{label}] (confidence: {confidence:.4f})")
        else:
            print("Athena: [no decodable output]")

        print()
        turn += 1


if __name__ == "__main__":
    main()
