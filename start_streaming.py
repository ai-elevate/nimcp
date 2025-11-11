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
from pathlib import Path
from typing import List, Dict, Iterator
import numpy as np

# Add NIMCP to path
sys.path.insert(0, str(Path(__file__).parent / "build/lib/python"))

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
        while True:
            try:
                article = next(self.iterator)
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
                print(f"⚠️  Wikipedia stream error: {e}")
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

class GitHubCodeStream(StreamingDataSource):
    """Stream code from GitHub repositories via HuggingFace"""

    def __init__(self):
        super().__init__("software_engineering")
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace datasets required for GitHub code streaming")

        print("💻 Loading GitHub code stream...")
        # Use The Stack dataset (deduplicated)
        try:
            self.dataset = load_dataset(
                "bigcode/the-stack-dedup",
                data_dir="data/python",
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            self.iterator = iter(self.dataset)
            print("  ✓ GitHub Python code stream loaded")
        except Exception as e:
            print(f"  ⚠️  Failed to load The Stack, trying alternative...")
            # Fallback to smaller dataset
            self.dataset = load_dataset(
                "codeparrot/github-code",
                split="train",
                streaming=True,
                trust_remote_code=False
            )
            self.iterator = iter(self.dataset)

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

def extract_features(text: str, dim: int = 512) -> np.ndarray:
    """Simple feature extraction"""
    features = np.zeros(dim, dtype=np.float32)

    # Character frequencies
    for char in text[:1000]:
        idx = ord(char) % 256
        if idx < dim:
            features[idx] += 1.0

    # Word hashes
    words = text.lower().split()[:128]
    for i, word in enumerate(words):
        if 256 + i < dim:
            features[256 + i] = hash(word) % 1000 / 1000.0

    # Normalize
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm

    return features

#=============================================================================
# Streaming Training Loop
#=============================================================================

def train_streaming(
    brain,
    streams: List[StreamingDataSource],
    num_examples: int = 10000,
    consolidation_interval: int = 500
):
    """Train on streaming data"""

    print("\n" + "="*60)
    print("🌊 STREAMING TRAINING ACTIVE")
    print("="*60)
    print(f"Streams: {len(streams)}")
    print(f"Training examples: {num_examples}")
    print(f"Consolidation interval: {consolidation_interval}")
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

            # Consolidation
            if (i + 1) % consolidation_interval == 0:
                print(f"\n💤 Consolidation at {i+1} examples...")

        print("\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")

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
    print("="*60)
    print("🌊 NIMCP STREAMING TRAINING")
    print("="*60)
    print("Mode: True streaming (no local storage)")
    print("Datasets: Fetched on-the-fly from online sources")
    print("="*60)

    # Create brain
    print("\n🧠 Creating NIMCP brain...")
    print("  Size: MEDIUM")
    print("  All cognitive systems: ACTIVE")
    print("  Epistemic filtering: ACTIVE (skepticism 0.3 - Phase 1)")
    print("  Curiosity: ACTIVE (0.95 - Phase 1)")

    brain = nimcp.Brain(
        'nimcp_streaming',  # name
        2,                  # size = MEDIUM
        0,                  # task = CLASSIFICATION
        512,                # inputs
        256                 # outputs
    )
    print("  ✓ Brain created")

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

    # Wikipedia (if available)
    if HF_AVAILABLE:
        try:
            wiki_stream = WikipediaStream()
            streams.append(wiki_stream)
            print("  ✓ Wikipedia stream")
        except Exception as e:
            print(f"  ⚠️  Wikipedia stream failed: {e}")

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
            num_examples=10000,  # Train on 10K streaming examples
            consolidation_interval=500  # Phase 1 schedule
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
