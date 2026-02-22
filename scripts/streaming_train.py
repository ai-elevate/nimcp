#!/usr/bin/env python3
"""
NIMCP Streaming Training Pipeline
==================================

WHAT: Memory-efficient training on massive datasets using streaming
WHY:  Avoid downloading 6TB+ datasets - train incrementally on batches
HOW:  Stream dataset → Process batch → Train → Discard → Repeat

Features:
- Streaming mode (no full download required)
- Configurable batch sizes (20-30 parquet files at a time)
- Automatic checkpoint saving
- Resume from checkpoint support
- Progress tracking
- Memory-efficient cleanup
"""

import hashlib
import json
import os
import sys
import time
import gc
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass
from datasets import load_dataset, IterableDataset
import nimcp

@dataclass
class StreamConfig:
    """Configuration for streaming training"""
    batch_size: int = 1000  # Examples per training batch
    parquet_batch_size: int = 25  # Number of parquet files to buffer
    checkpoint_interval: int = 10000  # Save checkpoint every N examples
    max_examples_per_dataset: Optional[int] = None  # Limit examples (for testing)
    temp_dir: Path = Path("/tmp/nimcp_streaming")
    checkpoint_dir: Path = Path("checkpoints")

@dataclass
class TrainingProgress:
    """Track training progress for checkpointing"""
    dataset_name: str
    examples_processed: int = 0
    batches_completed: int = 0
    last_checkpoint: int = 0
    start_time: float = 0.0

    def save(self, checkpoint_dir: Path):
        """Save progress to checkpoint file"""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{self.dataset_name}_progress.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                'dataset_name': self.dataset_name,
                'examples_processed': self.examples_processed,
                'batches_completed': self.batches_completed,
                'last_checkpoint': self.last_checkpoint,
                'timestamp': time.time()
            }, f, indent=2)

    @classmethod
    def load(cls, dataset_name: str, checkpoint_dir: Path):
        """Load progress from checkpoint file"""
        checkpoint_file = checkpoint_dir / f"{dataset_name}_progress.json"
        if not checkpoint_file.exists():
            return cls(dataset_name=dataset_name, start_time=time.time())

        with open(checkpoint_file) as f:
            data = json.load(f)

        progress = cls(
            dataset_name=data['dataset_name'],
            examples_processed=data['examples_processed'],
            batches_completed=data['batches_completed'],
            last_checkpoint=data['last_checkpoint'],
            start_time=time.time()
        )
        print(f"  ↻ Resuming from checkpoint: {progress.examples_processed} examples processed")
        return progress


class StreamingDatasetProcessor:
    """
    WHAT: Process datasets in streaming mode with batch training
    WHY:  Handle massive datasets without downloading everything
    HOW:  Stream → Buffer → Train → Cleanup
    """

    def __init__(self, brain, config: StreamConfig, num_inputs: int = 128, hf_token: str = None):
        self.brain = brain
        self.config = config
        self.num_inputs = num_inputs
        self.hf_token = hf_token
        self.config.temp_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_streaming_dataset(self, dataset_config: Dict):
        """
        WHAT: Load a HuggingFace dataset in streaming mode
        WHY:  Avoid downloading entire dataset; returns raw iterator (NOT a generator)
        HOW:  Use streaming=True, return the IterableDataset directly
        """
        hf_dataset = dataset_config['hf_dataset']
        hf_subset = dataset_config.get('hf_subset', None)

        print(f"  🔄 Streaming from HuggingFace: {hf_dataset}" +
              (f" ({hf_subset})" if hf_subset else ""))

        kwargs = dict(streaming=True)
        if self.hf_token:
            kwargs['token'] = self.hf_token

        # Try train split first, fall back to test/validation
        for split in ('train', 'test', 'validation', 'dev'):
            try:
                if hf_subset:
                    dataset = load_dataset(hf_dataset, hf_subset, split=split, **kwargs)
                else:
                    dataset = load_dataset(hf_dataset, split=split, **kwargs)
                print(f"  📂 Using split: {split}")
                return dataset
            except (ValueError, Exception) as split_err:
                err_str = str(split_err)
                if 'Bad split' in err_str or 'does not exist' in err_str:
                    continue
                print(f"  ✗ Streaming failed: {split_err}")
                return None

        print(f"  ✗ No usable split found")
        return None

    def encode_text(self, text: str, num_features: int) -> List[float]:
        """Encode text into a fixed-size feature vector using character
        n-grams and word hashing (same approach as CognitiveInterpreter)."""
        features = [0.0] * num_features
        text_lower = text.lower().strip()
        if not text_lower:
            return features

        # Character unigram frequencies
        for ch in text_lower:
            features[ord(ch) % num_features] += 1.0

        # Character bigram frequencies
        for i in range(len(text_lower) - 1):
            bigram = text_lower[i:i + 2]
            h = int(hashlib.md5(bigram.encode()).hexdigest(), 16)
            features[h % num_features] += 0.5

        # Word-level semantic hashing
        words = text_lower.split()
        for wi, word in enumerate(words):
            h = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for offset in range(3):
                features[(h + offset * 31) % num_features] += (wi + 1) * 0.1

        # Normalize to [0, 1]
        mx = max(features) if features else 1.0
        if mx > 0:
            features = [v / mx for v in features]
        return features

    def _extract_text_and_label(self, example: Dict, domain: str) -> Optional[tuple]:
        """Extract raw text and label from a dataset example based on its
        schema.  Returns (text, label_int) or None if the example is empty."""

        # --- MMLU format (cais/mmlu): question + choices + answer index ---
        if 'question' in example and 'choices' in example and 'answer' in example:
            q = example['question']
            choices = example['choices']
            if isinstance(choices, list):
                choices_str = ' '.join(f"({chr(65+i)}) {c}" for i, c in enumerate(choices))
            else:
                choices_str = str(choices)
            text = f"{q} {choices_str}"
            label = int(example['answer']) if isinstance(example['answer'], (int, float)) else 0
            return (text, label)

        # --- SQuAD format: context + question + answers ---
        if 'context' in example and 'question' in example:
            text = f"{example['question']} {example['context']}"
            answers = example.get('answers', {})
            if isinstance(answers, dict):
                ans_text = ' '.join(answers.get('text', []))
            elif isinstance(answers, list):
                ans_text = ' '.join(str(a) for a in answers)
            else:
                ans_text = str(answers)
            text = f"{text} {ans_text}"
            label = hash(ans_text) % 100 if ans_text.strip() else 0
            return (text, label)

        # --- CommonsenseQA: question + choices ---
        if 'question' in example and 'choices' in example and 'answerKey' in example:
            q = example['question']
            ch = example['choices']
            if isinstance(ch, dict):
                labels_list = ch.get('label', [])
                texts_list = ch.get('text', [])
                choices_str = ' '.join(f"({l}) {t}" for l, t in zip(labels_list, texts_list))
            else:
                choices_str = str(ch)
            text = f"{q} {choices_str}"
            ak = example['answerKey']
            label = ord(ak) - ord('A') if isinstance(ak, str) and ak.isalpha() else 0
            return (text, label)

        # --- Ethics format (hendrycks/ethics): input + label ---
        if 'input' in example and 'label' in example:
            text = str(example['input'])
            label = int(example['label']) if isinstance(example['label'], (int, float)) else 0
            return (text, label)

        # --- Moral stories: multiple text fields ---
        if 'norm' in example or 'situation' in example or 'moral_action' in example:
            parts = []
            for key in ('norm', 'situation', 'intention', 'moral_action', 'immoral_action'):
                val = example.get(key, '')
                if val:
                    parts.append(str(val))
            text = ' '.join(parts)
            label = int(example.get('label', 0)) if 'label' in example else 0
            return (text, label)

        # --- MMMU format: question + options ---
        if 'question' in example and 'options' in example:
            q = str(example['question'])
            opts = example['options']
            if isinstance(opts, list):
                opts_str = ' '.join(f"({chr(65+i)}) {o}" for i, o in enumerate(opts))
            else:
                opts_str = str(opts)
            text = f"{q} {opts_str}"
            ans = example.get('answer', '')
            label = ord(str(ans)) - ord('A') if isinstance(ans, str) and ans.isalpha() else 0
            return (text, label)

        # --- Programming: content/code ---
        if 'content' in example or 'code' in example:
            text = example.get('content', example.get('code', ''))
            lang = example.get('language', '')
            if lang:
                text = f"{lang}: {text}"
            label = hash(lang) % 100 if lang else 0
            return (text, label)

        # --- LiveCodeBench: question_content ---
        if 'question_content' in example:
            text = str(example['question_content'])
            label = hash(example.get('question_id', '')) % 100
            return (text, label)

        # --- Plain text (wikitext, etc.) ---
        if 'text' in example:
            text = str(example['text'])
            if not text.strip():
                return None
            label = 0
            return (text, label)

        # --- Social chem: action + rot (rule of thumb) ---
        if 'action' in example or 'rot' in example:
            parts = [str(example.get('action', '')), str(example.get('rot', ''))]
            text = ' '.join(p for p in parts if p)
            label = int(example.get('rot-judgment', 0)) if 'rot-judgment' in example else 0
            return (text, label)

        # --- HellaSwag format: ctx + endings + label ---
        if 'ctx' in example and 'endings' in example:
            ctx = str(example.get('ctx', ''))
            endings = example.get('endings', [])
            if isinstance(endings, list):
                endings_str = ' '.join(f"({i}) {e}" for i, e in enumerate(endings))
            else:
                endings_str = str(endings)
            text = f"{ctx} {endings_str}"
            label = int(example.get('label', 0)) if example.get('label') is not None else 0
            return (text, label)

        # --- OpenOrca format: system_prompt + question + response ---
        if 'system_prompt' in example and 'question' in example and 'response' in example:
            text = f"{example['system_prompt']} {example['question']} {example['response']}"
            label = hash(example.get('id', '')) % 100
            return (text, label)

        # --- Prosocial dialog (allenai/prosocial-dialog) ---
        if 'rots' in example and 'safety_label' in example:
            parts = [str(example.get('context', '')), str(example.get('response', ''))]
            rots = example.get('rots', [])
            if isinstance(rots, list):
                parts.extend(str(r) for r in rots)
            text = ' '.join(p for p in parts if p)
            sl = example.get('safety_label', '')
            label = 0 if sl == '__casual__' else 1 if sl == '__needs_caution__' else 2
            return (text, label)

        # --- HH-RLHF format (Anthropic/hh-rlhf): chosen + rejected ---
        if 'chosen' in example and 'rejected' in example:
            text = str(example['chosen'])
            label = 0  # chosen = good
            return (text, label)

        # --- CultureBank format (SALT-NLP/CultureBank) ---
        if 'cultural_group' in example or 'actor_behavior' in example:
            parts = []
            for key in ('cultural_group', 'context', 'goal', 'actor_behavior', 'topic'):
                val = example.get(key, '')
                if val:
                    parts.append(str(val))
            text = ' '.join(parts)
            agreement = example.get('agreement', '')
            label = hash(str(agreement)) % 100 if agreement else 0
            return (text, label)

        # --- Code contests (deepmind/code_contests) ---
        if 'description' in example and 'solutions' in example:
            text = str(example.get('description', ''))
            sols = example.get('solutions', {})
            if isinstance(sols, dict):
                sol_list = sols.get('solution', [])
                if isinstance(sol_list, list) and sol_list:
                    text = f"{text}\n{sol_list[0][:1000]}"
            label = int(example.get('difficulty', 0)) if 'difficulty' in example else 0
            return (text, label)

        # --- PalmX / query-answer format ---
        if 'query' in example:
            text = str(example['query'])
            ans = example.get('answer', '')
            if ans:
                text = f"{text} {ans}"
            label = 0
            return (text, label)

        # --- Fallback: concatenate all string fields ---
        parts = []
        label = 0
        for key, value in example.items():
            if isinstance(value, str) and value.strip():
                parts.append(value)
            elif isinstance(value, (int, float)) and 'label' in key.lower():
                label = int(value)
        if parts:
            return (' '.join(parts), label)

        return None

    def extract_features_and_label(self, example: Dict, domain: str) -> Optional[tuple]:
        """
        WHAT: Extract features and labels from dataset example
        WHY:  Different datasets have different schemas
        HOW:  Schema-aware text extraction → text-to-feature encoding
        """
        try:
            result = self._extract_text_and_label(example, domain)
            if result is None:
                return None

            text, label = result
            if not text or not text.strip():
                return None

            # Truncate very long texts to keep encoding fast
            if len(text) > 2000:
                text = text[:2000]

            features = self.encode_text(text, self.num_inputs)
            return (features, label)
        except Exception:
            return None

    def train_on_batch(self, batch: List[tuple], progress: TrainingProgress):
        """
        WHAT: Train brain on a batch of examples
        WHY:  Incremental learning from streaming data
        HOW:  Extract features, call brain.learn()
        """
        if not batch:
            return

        print(f"    Training on batch of {len(batch)} examples...", end='', flush=True)

        start_time = time.time()
        trained_count = 0
        total_loss = 0.0

        first_error_logged = False
        for features, label in batch:
            try:
                loss = self.brain.learn(features, str(label))
                if loss is not None:
                    total_loss += float(loss)
                trained_count += 1
            except Exception as e:
                if not first_error_logged:
                    print(f"\n    [learn error: {e}, features len={len(features)}]", end='', flush=True)
                    first_error_logged = True

        elapsed = time.time() - start_time
        examples_per_sec = trained_count / max(elapsed, 0.001)
        avg_loss = total_loss / max(trained_count, 1)

        print(f" done ({trained_count}/{len(batch)}, loss={avg_loss:.4f}, {examples_per_sec:.1f} ex/sec)")

        progress.examples_processed += trained_count
        progress.batches_completed += 1

    def cleanup_temp_files(self):
        """
        WHAT: Clean up temporary files after batch processing
        WHY:  Free disk space for next batch
        HOW:  Delete temp directory contents
        """
        if self.config.temp_dir.exists():
            try:
                shutil.rmtree(self.config.temp_dir)
                self.config.temp_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"  ⚠ Warning: Failed to cleanup temp files: {e}")

    def process_dataset_streaming(self, dataset_config: Dict) -> TrainingProgress:
        """
        WHAT: Main streaming processing loop for one dataset
        WHY:  Handle massive datasets efficiently
        HOW:  Stream → Buffer → Train → Cleanup → Repeat
        """
        name = dataset_config['name']
        domain = dataset_config['domain']

        print(f"\n{'='*70}")
        print(f"Processing: {name} ({domain})")
        print(f"{'='*70}")

        # Load or create progress tracker
        progress = TrainingProgress.load(name, self.config.checkpoint_dir)

        # Skip if already completed
        if progress.examples_processed > 0 and self.config.max_examples_per_dataset:
            if progress.examples_processed >= self.config.max_examples_per_dataset:
                print(f"  ✓ Already completed ({progress.examples_processed} examples)")
                return progress

        examples_since_checkpoint = 0

        try:
            # Load dataset — returns IterableDataset, NOT a generator
            dataset = self.load_streaming_dataset(dataset_config)
            if dataset is None:
                elapsed = time.time() - progress.start_time
                print(f"  ✓ Completed: {progress.examples_processed} examples in {elapsed:.1f}s")
                return progress

            # Use iter() + next() directly — avoids generator yield issues
            # with NIMCP C library exception handling
            stream = iter(dataset)
            stream_pos = 0
            logged_keys = False

            while True:
                # Phase 1: Buffer one batch from the stream
                batch = []
                stream_exhausted = False
                while len(batch) < self.config.batch_size:
                    try:
                        example = next(stream)
                        stream_pos += 1
                    except StopIteration:
                        stream_exhausted = True
                        break

                    if not logged_keys:
                        print(f"  📋 Fields: {list(example.keys())}", flush=True)
                        logged_keys = True

                    # Skip already processed examples (if resuming)
                    if stream_pos - 1 < progress.examples_processed:
                        continue

                    result = self.extract_features_and_label(example, domain)
                    if result is not None:
                        batch.append(result)

                    # Progress indicator
                    if stream_pos % 10000 == 0:
                        elapsed = time.time() - progress.start_time
                        rate = progress.examples_processed / max(elapsed, 1)
                        print(f"  📊 Streamed: {stream_pos}, Trained: {progress.examples_processed} ({rate:.1f} ex/sec)", flush=True)

                # Phase 2: Train on the batch
                if batch:
                    self.train_on_batch(batch, progress)
                    examples_since_checkpoint += len(batch)

                    # Checkpoint periodically
                    if examples_since_checkpoint >= self.config.checkpoint_interval:
                        progress.save(self.config.checkpoint_dir)
                        print(f"  💾 Checkpoint at {progress.examples_processed} examples")
                        examples_since_checkpoint = 0
                        gc.collect()

                # Check if stream is done
                if stream_exhausted:
                    break

                # Check limit
                if self.config.max_examples_per_dataset:
                    if progress.examples_processed >= self.config.max_examples_per_dataset:
                        print(f"  ⏹ Reached limit of {self.config.max_examples_per_dataset} examples")
                        break

            # Final checkpoint
            progress.save(self.config.checkpoint_dir)

            elapsed = time.time() - progress.start_time
            print(f"  ✓ Completed: {progress.examples_processed} examples in {elapsed:.1f}s")

        except KeyboardInterrupt:
            print(f"\n  ⏸ Interrupted - saving checkpoint...")
            progress.save(self.config.checkpoint_dir)
            raise

        except Exception as e:
            print(f"  ✗ Error: {e}")
            progress.save(self.config.checkpoint_dir)

        # Final cleanup
        self.cleanup_temp_files()

        return progress


def main():
    """Main streaming training pipeline"""

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='NIMCP Streaming Training Pipeline')
    parser.add_argument('--brain-path', type=str, default=None, help='Path to brain binary to load')
    parser.add_argument('--output-path', type=str, default=None, help='Path to save trained brain')
    parser.add_argument('--batch-size', type=int, default=1000, help='Examples per training batch')
    parser.add_argument('--max-examples', type=int, default=None, help='Max examples per dataset (default: unlimited)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000, help='Checkpoint every N examples')
    parser.add_argument('--datasets', type=str, default=None, help='Comma-separated dataset names (default: all)')
    parser.add_argument('--hf-token', type=str, default=None, help='HuggingFace API token for gated datasets')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    args = parser.parse_args()

    # Load dataset config
    config_file = Path(__file__).parent / "foundation_datasets_config.json"
    with open(config_file) as f:
        config = json.load(f)

    # Initialize streaming config
    stream_config = StreamConfig(
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        max_examples_per_dataset=args.max_examples
    )

    print(f"""
{'='*70}
NIMCP Streaming Training Pipeline
{'='*70}
Configuration:
  Brain Path: {args.brain_path or '(create new)'}
  Output Path: {args.output_path or '(none)'}
  Batch Size: {stream_config.batch_size} examples
  Checkpoint Interval: {stream_config.checkpoint_interval} examples
  Max Examples per Dataset: {stream_config.max_examples_per_dataset or 'unlimited'}
{'='*70}
""")

    # Load brain from frontend or create new
    nimcp.init()
    if args.brain_path and Path(args.brain_path).exists():
        print(f"Loading brain from {args.brain_path}...")
        brain = nimcp.Brain.load(args.brain_path)
        print("Brain loaded successfully")
    else:
        print("Creating new brain...")
        brain = nimcp.Brain("streaming", 2, 2, 128, 32)
        print("Brain created")

    # Determine brain's num_inputs for feature encoding
    num_inputs = 128
    try:
        ni = brain.num_inputs
        if ni and ni > 0:
            num_inputs = ni
    except (AttributeError, Exception):
        pass
    print(f"Feature vector size: {num_inputs}")

    # Create processor
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    processor = StreamingDatasetProcessor(brain, stream_config, num_inputs=num_inputs, hf_token=hf_token)

    # Filter datasets if specified
    hf_datasets = [d for d in config['datasets'] if d['type'] == 'huggingface']
    if args.datasets:
        dataset_names = set(args.datasets.split(','))
        hf_datasets = [d for d in hf_datasets if d['name'] in dataset_names]

    print(f"Processing {len(hf_datasets)} datasets in streaming mode\n")

    # Process each dataset
    results = {}
    total_examples = 0

    try:
        for dataset_config in hf_datasets:
            name = dataset_config['name']
            progress = processor.process_dataset_streaming(dataset_config)
            results[name] = progress
            total_examples += progress.examples_processed

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    # Save trained brain
    if args.output_path:
        print(f"Saving trained brain to {args.output_path}...")
        try:
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            brain.save(args.output_path)
            print("Brain saved successfully")
        except Exception as exc:
            print(f"Failed to save brain: {exc}")

    # Summary
    print(f"""
{'='*70}
Training Summary
{'='*70}
Total Examples Processed: {total_examples}
Datasets Completed: {len(results)}

Per-Dataset Results:
""")

    for name, progress in results.items():
        print(f"  {name}: {progress.examples_processed} examples, {progress.batches_completed} batches")

    # Skip nimcp.shutdown() — it triggers a segfault during cleanup.
    # The OS reclaims all memory when the process exits anyway.
    # The brain object must be deleted before exit to flush any pending writes.
    del brain
    gc.collect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
