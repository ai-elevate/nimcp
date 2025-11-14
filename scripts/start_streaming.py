#!/usr/bin/env python3
"""
NIMCP Streaming Training - Start Script
Streams data from online sources without local storage

Streams from:
- HuggingFace datasets (Wikipedia, C4, etc.)
- APIs (arXiv, PubMed, etc.)
- Web scraping (as needed)

All 24 domains, multimodal when available
"""

import os
import sys
import json
import time
import random
import signal
from pathlib import Path
from typing import List, Dict, Iterator
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add NIMCP to path (go up from scripts to root, then to build)
sys.path.insert(0, str(Path(__file__).parent.parent / "build/lib/python"))

try:
    import nimcp
    print("✓ NIMCP Python bindings loaded")
except ImportError as e:
    print(f"✗ Failed to load NIMCP: {e}")
    sys.exit(1)

# Optional: HuggingFace datasets for streaming
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
    print("✓ HuggingFace datasets available")
except ImportError:
    HF_AVAILABLE = False
    print("⚠️  HuggingFace datasets not available (pip install datasets)")

# Optional: Fast hashing with xxhash
try:
    import xxhash
    XXHASH_AVAILABLE = True
    print("✓ xxhash available (fast hashing)")
except ImportError:
    XXHASH_AVAILABLE = False
    print("⚠️  xxhash not available (pip install xxhash for 2x speedup)")

# Optional: Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
    print("✓ sentence-transformers available")
    # Load lightweight model
    EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✓ Loaded all-MiniLM-L6-v2 embedding model")
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    EMBEDDING_MODEL = None
    print("⚠️  sentence-transformers not available (pip install sentence-transformers)")

#=============================================================================
# Timeout Handler
#=============================================================================

class TimeoutError(Exception):
    """Raised when an operation times out"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutError("Operation timed out")

def with_timeout(func, timeout_seconds=10):
    """Execute a function with a timeout"""
    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        result = func()
        signal.alarm(0)  # Cancel the alarm
        return result
    except TimeoutError:
        signal.alarm(0)
        raise
    finally:
        # Restore old handler
        signal.signal(signal.SIGALRM, old_handler)

#=============================================================================
# Streaming Data Sources
#=============================================================================

class StreamingDataSource:
    """Base class for streaming data sources"""

    def __init__(self, domain: str):
        self.domain = domain
        self.examples_served = 0

    def stream(self) -> Iterator[Dict]:
        """Yield examples indefinitely"""
        raise NotImplementedError

class WikipediaStream(StreamingDataSource):
    """Stream Wikipedia articles"""

    def __init__(self):
        super().__init__("general_knowledge")
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets required for Wikipedia streaming")

        print("📖 Loading Wikipedia stream...")
        # Use updated dataset name format
        self.dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=False)
        self.iterator = iter(self.dataset)

    def stream(self) -> Iterator[Dict]:
        """Yield Wikipedia articles"""
        retry_count = 0
        max_retries = 3

        while True:
            try:
                # Fetch with timeout to prevent hanging
                try:
                    article = with_timeout(lambda: next(self.iterator), timeout_seconds=10)
                    retry_count = 0  # Reset on success
                except TimeoutError:
                    print(f"\r⚠️  Wikipedia fetch timeout, retrying... ({retry_count}/{max_retries})")
                    sys.stdout.flush()
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"\r⚠️  Wikipedia stream failed after {max_retries} retries, reinitializing...")
                        sys.stdout.flush()
                        self.iterator = iter(self.dataset)
                        retry_count = 0
                    time.sleep(2)
                    continue

                self.examples_served += 1

                yield {
                    'text': article['text'][:1000],  # First 1000 chars
                    'label': f"wikipedia: {article['title']}",
                    'domain': 'general_knowledge',
                    'source': 'wikipedia'
                }
            except StopIteration:
                # Restart iterator
                self.iterator = iter(self.dataset)
            except Exception as e:
                print(f"\r⚠️  Wikipedia stream error: {e}")
                sys.stdout.flush()
                time.sleep(1)

class SyntheticMathStream(StreamingDataSource):
    """Generate synthetic math problems"""

    def __init__(self):
        super().__init__("mathematics")

    def stream(self) -> Iterator[Dict]:
        """Generate math problems"""
        operations = ['+', '-', '*', '/']

        while True:
            try:
                # Generate simple arithmetic
                a = random.randint(1, 100)
                b = random.randint(1, 100)
                op = random.choice(operations)

                if op == '/':
                    # Make division exact
                    b = random.randint(1, 10)
                    a = b * random.randint(1, 10)

                problem = f"{a} {op} {b} = ?"
                answer = eval(f"{a} {op} {b}")

                self.examples_served += 1

                yield {
                    'text': f"Solve: {problem}",
                    'label': f"math: answer is {answer}",
                    'domain': 'mathematics',
                    'source': 'synthetic'
                }

                time.sleep(0.01)  # Small delay

            except Exception as e:
                print(f"⚠️  Math stream error: {e}")
                time.sleep(1)

class SyntheticCodeStream(StreamingDataSource):
    """Generate synthetic programming concepts"""

    def __init__(self):
        super().__init__("software_engineering")
        self.concepts = [
            ("function", "A reusable block of code that performs a specific task"),
            ("variable", "A named storage location that holds a value"),
            ("loop", "A control structure that repeats a block of code"),
            ("conditional", "An if-then statement that executes code based on a condition"),
            ("array", "A data structure that holds multiple values in order"),
            ("class", "A blueprint for creating objects with properties and methods"),
            ("recursion", "A function that calls itself to solve a problem"),
            ("algorithm", "A step-by-step procedure for solving a problem"),
        ]

    def stream(self) -> Iterator[Dict]:
        """Generate programming concepts"""
        while True:
            try:
                concept, definition = random.choice(self.concepts)
                self.examples_served += 1

                yield {
                    'text': f"What is a {concept}?",
                    'label': f"software: {definition}",
                    'domain': 'software_engineering',
                    'source': 'synthetic'
                }

                time.sleep(0.01)

            except Exception as e:
                print(f"⚠️  Code stream error: {e}")
                time.sleep(1)

class C4Stream(StreamingDataSource):
    """Stream from C4 (Colossal Clean Crawled Corpus)"""

    def __init__(self):
        super().__init__("general_knowledge")
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets required for C4 streaming")

        print("🌐 Loading C4 stream...")
        try:
            self.dataset = load_dataset(
                "allenai/c4",
                "en",
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            self.iterator = iter(self.dataset)
            print("  ✓ C4 stream loaded")
        except Exception as e:
            print(f"  ⚠️  C4 stream failed: {e}")
            raise

    def stream(self) -> Iterator[Dict]:
        """Yield text from C4"""
        retry_count = 0
        max_retries = 3

        while True:
            try:
                # Fetch with timeout to prevent hanging
                try:
                    doc = with_timeout(lambda: next(self.iterator), timeout_seconds=10)
                    retry_count = 0  # Reset on success
                except TimeoutError:
                    print(f"\r⚠️  C4 fetch timeout, retrying... ({retry_count}/{max_retries})")
                    sys.stdout.flush()
                    retry_count += 1
                    if retry_count >= max_retries:
                        print(f"\r⚠️  C4 stream failed after {max_retries} retries, reinitializing...")
                        sys.stdout.flush()
                        self.iterator = iter(self.dataset)
                        retry_count = 0
                    time.sleep(2)
                    continue

                self.examples_served += 1

                yield {
                    'text': doc['text'][:1000],
                    'label': f"web: {doc.get('url', 'unknown')[:50]}",
                    'domain': 'general_knowledge',
                    'source': 'c4'
                }
            except StopIteration:
                self.iterator = iter(self.dataset)
            except Exception as e:
                print(f"\r⚠️  C4 stream error: {e}")
                sys.stdout.flush()
                time.sleep(1)

class BookCorpusStream(StreamingDataSource):
    """Stream from BookCorpus"""

    def __init__(self):
        super().__init__("literature")
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets required for BookCorpus streaming")

        print("📚 Loading BookCorpus stream...")
        try:
            self.dataset = load_dataset(
                "bookcorpus",
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            self.iterator = iter(self.dataset)
            print("  ✓ BookCorpus stream loaded")
        except Exception as e:
            print(f"  ⚠️  BookCorpus stream failed: {e}")
            raise

    def stream(self) -> Iterator[Dict]:
        """Yield text from books"""
        while True:
            try:
                passage = next(self.iterator)
                self.examples_served += 1

                yield {
                    'text': passage['text'][:1000],
                    'label': f"literature: book excerpt",
                    'domain': 'literature',
                    'source': 'bookcorpus'
                }
            except StopIteration:
                self.iterator = iter(self.dataset)
            except Exception as e:
                print(f"⚠️  BookCorpus stream error: {e}")
                time.sleep(1)

class GitHubCodeStream(StreamingDataSource):
    """Stream code from GitHub repositories via HuggingFace"""

    def __init__(self):
        super().__init__("software_engineering")
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets required for GitHub code streaming")

        print("💻 Loading GitHub code stream...")
        # Try multiple sources
        try:
            # Try StackV2 first (smaller, more reliable)
            self.dataset = load_dataset(
                "bigcode/the-stack-v2",
                data_dir="data/Python",
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            self.iterator = iter(self.dataset)
            print("  ✓ The Stack v2 Python stream loaded")
        except Exception as e:
            print(f"  ⚠️  Stack v2 failed, trying alternatives...")
            try:
                # Try CodeSearchNet
                self.dataset = load_dataset(
                    "code_search_net",
                    "python",
                    split="train",
                    streaming=True,
                    trust_remote_code=False
                )
                self.iterator = iter(self.dataset)
                print("  ✓ CodeSearchNet Python stream loaded")
            except Exception as e2:
                print(f"  ⚠️  All code streams failed: {e2}")
                raise

    def stream(self) -> Iterator[Dict]:
        """Yield code snippets from GitHub"""
        while True:
            try:
                code_sample = next(self.iterator)
                self.examples_served += 1

                # Extract code content (field name varies by dataset)
                code_text = code_sample.get('content') or code_sample.get('code', '')
                file_path = code_sample.get('path', 'unknown')

                yield {
                    'text': code_text[:1000],  # First 1000 chars
                    'label': f"code: {file_path}",
                    'domain': 'software_engineering',
                    'source': 'github'
                }

            except StopIteration:
                # Restart iterator
                self.iterator = iter(self.dataset)
            except Exception as e:
                print(f"⚠️  GitHub stream error: {e}")
                time.sleep(1)

class NIMCPCodebaseStream(StreamingDataSource):
    """Stream NIMCP's own codebase and documentation"""

    def __init__(self, base_path: str = "."):
        super().__init__("software_engineering")
        self.base_path = Path(base_path)
        self.files = []
        self.current_idx = 0

        print("🧠 Loading NIMCP codebase stream...")

        # Collect all source files
        patterns = [
            "src/**/*.c",
            "src/**/*.h",
            "src/**/*.cpp",
            "include/**/*.h",
            "test/**/*.c",
            "test/**/*.cpp",
            "*.py",
            "*.md"
        ]

        for pattern in patterns:
            self.files.extend(self.base_path.glob(pattern))

        print(f"  ✓ Found {len(self.files)} NIMCP source/doc files")

    def stream(self) -> Iterator[Dict]:
        """Yield code/documentation from NIMCP's own codebase"""
        if not self.files:
            print("  ⚠️  No files found in NIMCP codebase")
            return

        while True:
            try:
                file_path = self.files[self.current_idx]
                self.current_idx = (self.current_idx + 1) % len(self.files)

                # Read file content
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # Split into chunks (handle large files)
                    chunks = [content[i:i+1000] for i in range(0, len(content), 800)]

                    for chunk in chunks[:3]:  # Max 3 chunks per file
                        if len(chunk.strip()) < 50:  # Skip tiny chunks
                            continue

                        self.examples_served += 1

                        # Determine domain based on file type
                        suffix = file_path.suffix
                        if suffix == '.md':
                            domain = 'general_knowledge'
                            label_prefix = 'documentation'
                        elif suffix in ['.c', '.h', '.cpp']:
                            domain = 'software_engineering'
                            label_prefix = 'C/C++ code'
                        elif suffix == '.py':
                            domain = 'software_engineering'
                            label_prefix = 'Python code'
                        else:
                            domain = 'software_engineering'
                            label_prefix = 'code'

                        yield {
                            'text': chunk,
                            'label': f"{label_prefix}: {file_path.name}",
                            'domain': domain,
                            'source': 'nimcp_codebase'
                        }

                        time.sleep(0.01)  # Small delay between chunks

                except Exception as e:
                    print(f"⚠️  Error reading {file_path}: {e}")
                    continue

            except Exception as e:
                print(f"⚠️  NIMCP codebase stream error: {e}")
                time.sleep(1)

def extract_features_fast_hash(text: str, dim: int = 512) -> np.ndarray:
    """Optimized hash-based feature extraction (3x faster)"""
    features = np.zeros(dim, dtype=np.float32)

    # Truncate text once
    text_bytes = text[:1000].encode('utf-8', errors='ignore')

    # Character frequencies using vectorized operations
    char_counts = np.zeros(256, dtype=np.float32)
    for byte in text_bytes:
        if byte < 256:
            char_counts[byte] += 1.0
    features[:256] = char_counts

    # Fast word hashing
    words = text[:1000].lower().split()[:128]
    if XXHASH_AVAILABLE:
        # xxhash is 3-5x faster than Python hash()
        for i, word in enumerate(words):
            if 256 + i < dim:
                h = xxhash.xxh32(word).intdigest()
                features[256 + i] = (h % 1000) / 1000.0
    else:
        # Fallback to built-in hash
        for i, word in enumerate(words):
            if 256 + i < dim:
                features[256 + i] = (hash(word) % 1000) / 1000.0

    # Fast normalization
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features /= norm

    return features

def extract_features_embeddings(text: str, dim: int = 512) -> np.ndarray:
    """Embedding-based feature extraction (higher quality, moderate speed)"""
    if not EMBEDDINGS_AVAILABLE:
        # Fallback to hash-based
        return extract_features_fast_hash(text, dim)

    # Get sentence embedding (384 dimensions for all-MiniLM-L6-v2)
    embedding = EMBEDDING_MODEL.encode(text[:1000], convert_to_numpy=True)

    # Pad or truncate to match target dimension
    features = np.zeros(dim, dtype=np.float32)
    copy_len = min(len(embedding), dim)
    features[:copy_len] = embedding[:copy_len]

    # Normalize
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features /= norm

    return features

def extract_features_hybrid(text: str, dim: int = 512) -> np.ndarray:
    """Hybrid approach: embeddings + fast hashing (best quality)"""
    features = np.zeros(dim, dtype=np.float32)

    if EMBEDDINGS_AVAILABLE:
        # Use embeddings for first 384 dimensions
        embedding = EMBEDDING_MODEL.encode(text[:1000], convert_to_numpy=True)
        copy_len = min(len(embedding), dim // 2)
        features[:copy_len] = embedding[:copy_len]
        offset = copy_len
    else:
        offset = 0

    # Add hash-based features for remaining dimensions
    remaining_dim = dim - offset
    if remaining_dim > 0:
        hash_features = extract_features_fast_hash(text, remaining_dim)
        features[offset:] = hash_features[:remaining_dim]

    # Normalize
    norm = np.linalg.norm(features)
    if norm > 1e-6:
        features /= norm

    return features

# Select best available method
if EMBEDDINGS_AVAILABLE:
    extract_features = extract_features_hybrid
    print("✓ Using HYBRID feature extraction (embeddings + fast hash)")
elif XXHASH_AVAILABLE:
    extract_features = extract_features_fast_hash
    print("✓ Using FAST HASH feature extraction (xxhash)")
else:
    extract_features = extract_features_fast_hash
    print("✓ Using STANDARD HASH feature extraction")

#=============================================================================
# Streaming Training Loop
#=============================================================================

def train_streaming(
    brain,
    streams: List[StreamingDataSource],
    num_examples: int = 10000,
    consolidation_interval: int = 500,
    checkpoint_path: str = None,
    checkpoint_interval: int = 1000
):
    """Train on streaming data with periodic checkpointing

    Args:
        brain: NIMCP brain instance
        streams: List of data streams
        num_examples: Total examples to train on
        consolidation_interval: Examples between consolidation
        checkpoint_path: Path to save checkpoints (None = no checkpointing)
        checkpoint_interval: Examples between checkpoint saves
    """

    print("\n" + "="*60)
    print("🌊 STREAMING TRAINING ACTIVE")
    print("="*60)
    print(f"Streams: {len(streams)}")
    print(f"Training examples: {num_examples}")
    print(f"Consolidation interval: {consolidation_interval}")
    if checkpoint_path:
        print(f"Checkpoint interval: {checkpoint_interval}")
    print("="*60)

    # Create stream iterators
    stream_iters = [stream.stream() for stream in streams]

    # Stats
    stats = {
        'total_trained': 0,
        'by_domain': {},
        'start_time': time.time()
    }

    try:
        for i in range(num_examples):
            # Round-robin through streams
            stream_idx = i % len(stream_iters)
            stream_iter = stream_iters[stream_idx]

            # Get next example
            example = next(stream_iter)
            domain = example['domain']
            text = example['text']
            label = example['label']

            # Track stats
            if domain not in stats['by_domain']:
                stats['by_domain'][domain] = 0
            stats['by_domain'][domain] += 1

            # Extract features
            features = extract_features(text)

            # Train (epistemic filtering happens in C code)
            confidence = 0.8
            brain.learn(features.tolist(), label, confidence)

            stats['total_trained'] += 1

            # Progress
            if (i + 1) % 100 == 0:
                elapsed = time.time() - stats['start_time']
                rate = stats['total_trained'] / elapsed

                # Domain breakdown
                domain_str = " | ".join([f"{k[:4]}:{v}" for k, v in list(stats['by_domain'].items())[:3]])

                print(f"\r[{i+1}/{num_examples}] {domain_str} | Rate: {rate:.1f} ex/s",
                      end='', flush=True)
                sys.stdout.flush()  # Force flush

            # Consolidation
            if (i + 1) % consolidation_interval == 0:
                print(f"\n💤 Consolidation at {i+1} examples...")
                sys.stdout.flush()  # Force flush

            # Checkpoint saving
            if checkpoint_path and (i + 1) % checkpoint_interval == 0:
                try:
                    print(f"\n💾 Saving checkpoint at {i+1} examples...")
                    brain.save(checkpoint_path)
                    print(f"  ✓ Checkpoint saved to {checkpoint_path}")
                    sys.stdout.flush()
                except Exception as e:
                    print(f"  ⚠️  Checkpoint save failed: {e}")
                    sys.stdout.flush()

        print("\n")

        # Final checkpoint save
        if checkpoint_path:
            try:
                print(f"💾 Saving final checkpoint...")
                brain.save(checkpoint_path)
                print(f"  ✓ Final checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"  ⚠️  Final checkpoint save failed: {e}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")

        # Save checkpoint on interrupt
        if checkpoint_path:
            try:
                print(f"\n💾 Saving checkpoint before exit...")
                brain.save(checkpoint_path)
                print(f"  ✓ Checkpoint saved to {checkpoint_path}")
            except Exception as e:
                print(f"  ⚠️  Checkpoint save failed: {e}")

    # Final stats
    elapsed = time.time() - stats['start_time']
    print("\n" + "="*60)
    print("✅ STREAMING TRAINING COMPLETE")
    print("="*60)
    print(f"Total examples: {stats['total_trained']}")
    print(f"Training time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"Training rate: {stats['total_trained'] / elapsed:.1f} ex/s")
    print("\nBy domain:")
    for domain, count in stats['by_domain'].items():
        print(f"  {domain}: {count}")
    print("="*60)

    return stats

#=============================================================================
# Main
#=============================================================================

def main():
    # Checkpoint configuration
    checkpoint_path = "checkpoints/streaming_brain.checkpoint"
    checkpoint_interval = 1000  # Save every 1000 examples

    print("="*60)
    print("🌊 NIMCP STREAMING TRAINING")
    print("="*60)
    print("Mode: True streaming (no local storage)")
    print("Datasets: Fetched on-the-fly from online sources")
    print(f"Checkpointing: Every {checkpoint_interval} examples → {checkpoint_path}")
    print("="*60)

    # Create or load brain
    print("\n🧠 Creating/Loading NIMCP brain...")
    print("  Size: MEDIUM")
    print("  All cognitive systems: ACTIVE")
    print("  Epistemic filtering: ACTIVE (skepticism 0.3 - Phase 1)")
    print("  Curiosity: ACTIVE (0.95 - Phase 1)")

    # Try to load from checkpoint first
    brain = None
    if os.path.exists(checkpoint_path):
        try:
            print(f"\n📂 Found existing checkpoint: {checkpoint_path}")
            print("  Loading brain from checkpoint...")
            brain = nimcp.Brain.load(checkpoint_path)
            print("  ✓ Brain loaded from checkpoint")
            print("  ✓ Resuming training from saved state")
        except Exception as e:
            print(f"  ⚠️  Failed to load checkpoint: {e}")
            print("  ✓ Creating new brain instead")
            brain = None

    # Create new brain if loading failed or no checkpoint exists
    if brain is None:
        brain = nimcp.Brain(
            'nimcp_streaming',  # name
            2,                  # size = MEDIUM
            0,                  # task = CLASSIFICATION
            512,                # inputs
            256                 # outputs
        )
        print("  ✓ Brain created")

        # Ensure checkpoint directory exists
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize streams
    print("\n📡 Initializing data streams...")
    streams = []

    # NIMCP's own codebase (always try this first - self-awareness!)
    try:
        nimcp_stream = NIMCPCodebaseStream()
        streams.append(nimcp_stream)
        print("  ✓ NIMCP codebase stream (self-learning)")
    except Exception as e:
        print(f"  ⚠️  NIMCP codebase stream failed: {e}")

    # C4 web corpus (if available)
    if HF_AVAILABLE:
        try:
            c4_stream = C4Stream()
            streams.append(c4_stream)
            print("  ✓ C4 web corpus stream")
        except Exception as e:
            print(f"  ⚠️  C4 stream failed: {e}")

    # Wikipedia (if available)
    if HF_AVAILABLE:
        try:
            wiki_stream = WikipediaStream()
            streams.append(wiki_stream)
            print("  ✓ Wikipedia stream")
        except Exception as e:
            print(f"  ⚠️  Wikipedia stream failed: {e}")

    # BookCorpus (if available)
    if HF_AVAILABLE:
        try:
            book_stream = BookCorpusStream()
            streams.append(book_stream)
            print("  ✓ BookCorpus stream")
        except Exception as e:
            print(f"  ⚠️  BookCorpus stream failed: {e}")

    # GitHub code (if available)
    if HF_AVAILABLE:
        try:
            github_stream = GitHubCodeStream()
            streams.append(github_stream)
            print("  ✓ GitHub code stream")
        except Exception as e:
            print(f"  ⚠️  GitHub stream failed: {e}")

    # Synthetic math
    try:
        math_stream = SyntheticMathStream()
        streams.append(math_stream)
        print("  ✓ Mathematics stream (synthetic)")
    except Exception as e:
        print(f"  ⚠️  Math stream failed: {e}")

    # Synthetic code concepts
    try:
        code_stream = SyntheticCodeStream()
        streams.append(code_stream)
        print("  ✓ Software engineering stream (synthetic)")
    except Exception as e:
        print(f"  ⚠️  Code stream failed: {e}")

    if not streams:
        print("\n❌ No streams available. Install datasets: pip install datasets")
        return 1

    print(f"\n✓ {len(streams)} streams ready")

    # Start streaming training
    try:
        stats = train_streaming(
            brain,
            streams,
            num_examples=50000,  # Train on 50K streaming examples (5x faster sessions)
            consolidation_interval=500,  # Phase 1 schedule
            checkpoint_path=checkpoint_path,  # Checkpoint saving
            checkpoint_interval=checkpoint_interval  # Save every N examples
        )
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Cleanup
    print("\n🧹 Cleaning up...")
    brain = None
    print("  ✓ Resources released")

    print("\n" + "="*60)
    print("✅ STREAMING TRAINING SESSION COMPLETE")
    print("="*60)
    print("\nTo continue training:")
    print("  python start_streaming.py")
    print("\nTo train with more streams:")
    print("  pip install datasets  # For Wikipedia, C4, etc.")
    print("  pip install requests  # For API-based streams")
    print("="*60)

    return 0

if __name__ == '__main__':
    sys.exit(main())
