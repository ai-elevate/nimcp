#!/usr/bin/env python3
"""
NIMCP Dataset Preprocessor
Converts raw datasets into NIMCP-friendly training format.

Features:
- Text extraction and cleaning
- Label normalization
- Feature extraction (embeddings, TF-IDF, etc.)
- Train/validation/test splits
- Data normalization and standardization
- Multiple domain support

Output Format:
Each processed dataset contains:
- samples.jsonl: One sample per line {"features": [...], "label": "...", "text": "..."}
- metadata.json: Dataset statistics and configuration
- train.jsonl, val.jsonl, test.jsonl: Data splits
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random


class TextProcessor:
    """Text cleaning and normalization"""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            text = str(text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization"""
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    @staticmethod
    def extract_features_tfidf(text: str, vocab: Dict[str, int],
                               max_features: int = 512) -> List[float]:
        """
        Extract TF-IDF features (simplified version).

        Args:
            text: Input text
            vocab: Vocabulary dict {word: index}
            max_features: Maximum feature vector size

        Returns:
            Feature vector
        """
        tokens = TextProcessor.tokenize(text)

        # Term frequency
        tf = Counter(tokens)
        total_terms = len(tokens) if tokens else 1

        # Create feature vector
        features = [0.0] * max_features

        for word, count in tf.items():
            if word in vocab:
                idx = vocab[word]
                if idx < max_features:
                    # Simple TF (can be extended to TF-IDF)
                    features[idx] = count / total_terms

        return features

    @staticmethod
    def build_vocabulary(texts: List[str], max_vocab: int = 10000) -> Dict[str, int]:
        """Build vocabulary from texts"""
        all_tokens = []
        for text in texts:
            all_tokens.extend(TextProcessor.tokenize(text))

        # Count and sort by frequency
        counter = Counter(all_tokens)
        most_common = counter.most_common(max_vocab)

        # Create vocab dict
        vocab = {word: idx for idx, (word, _) in enumerate(most_common)}

        return vocab


class DatasetPreprocessor:
    """Main preprocessor for datasets"""

    def __init__(self, extracted_dir: str = None, processed_dir: str = None):
        if extracted_dir is None:
            script_dir = Path(__file__).parent
            base_dir = script_dir.parent / "datasets"
            extracted_dir = base_dir / "extracted"
            processed_dir = base_dir / "processed"

        self.extracted_dir = Path(extracted_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_csv(self, filepath: Path, text_col: str, label_col: str,
                 delimiter: str = ',') -> List[Dict]:
        """Load data from CSV file"""
        samples = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                if text_col not in row or label_col not in row:
                    continue

                text = TextProcessor.clean_text(row[text_col])
                label = TextProcessor.clean_text(row[label_col])

                if text and label:
                    samples.append({
                        'text': text,
                        'label': label,
                        'raw': row
                    })

        return samples

    def load_jsonl(self, filepath: Path, text_key: str, label_key: str) -> List[Dict]:
        """Load data from JSONL file"""
        samples = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    data = json.loads(line)

                    text = TextProcessor.clean_text(data.get(text_key, ''))
                    label = TextProcessor.clean_text(data.get(label_key, ''))

                    if text and label:
                        samples.append({
                            'text': text,
                            'label': label,
                            'raw': data
                        })
                except json.JSONDecodeError:
                    continue

        return samples

    def load_text_classification(self, dataset_dir: Path,
                                 config: Dict) -> List[Dict]:
        """
        Load text classification dataset.

        Config format:
        {
            "format": "csv" | "jsonl" | "folders",
            "text_column": "text",
            "label_column": "label",
            "files": ["train.csv", "test.csv"] (optional)
        }
        """
        format_type = config.get('format', 'csv')
        text_col = config.get('text_column', 'text')
        label_col = config.get('label_column', 'label')

        samples = []

        if format_type == 'csv':
            files = config.get('files', [])
            if not files:
                # Auto-detect CSV files
                files = list(dataset_dir.glob('*.csv'))
            else:
                files = [dataset_dir / f for f in files]

            for filepath in files:
                if filepath.exists():
                    print(f"  Loading: {filepath.name}")
                    samples.extend(self.load_csv(filepath, text_col, label_col))

        elif format_type == 'jsonl':
            files = config.get('files', [])
            if not files:
                files = list(dataset_dir.glob('*.jsonl'))
            else:
                files = [dataset_dir / f for f in files]

            for filepath in files:
                if filepath.exists():
                    print(f"  Loading: {filepath.name}")
                    samples.extend(self.load_jsonl(filepath, text_col, label_col))

        elif format_type == 'folders':
            # Each subfolder is a class
            for class_dir in dataset_dir.iterdir():
                if class_dir.is_dir():
                    label = class_dir.name
                    for text_file in class_dir.glob('*.txt'):
                        with open(text_file, 'r', encoding='utf-8', errors='ignore') as f:
                            text = TextProcessor.clean_text(f.read())
                            if text:
                                samples.append({
                                    'text': text,
                                    'label': label,
                                    'raw': {}
                                })

        return samples

    def extract_features(self, samples: List[Dict], max_features: int = 512) -> Tuple[List[Dict], Dict]:
        """
        Extract features from samples.

        Returns:
            Tuple of (processed_samples, metadata)
        """
        print("  Extracting features...")

        # Build vocabulary from all texts
        all_texts = [s['text'] for s in samples]
        vocab = TextProcessor.build_vocabulary(all_texts, max_vocab=10000)

        print(f"    Vocabulary size: {len(vocab)}")

        # Extract features for each sample
        processed = []
        for sample in samples:
            features = TextProcessor.extract_features_tfidf(
                sample['text'], vocab, max_features
            )

            processed.append({
                'text': sample['text'],
                'label': sample['label'],
                'features': features
            })

        metadata = {
            'vocab_size': len(vocab),
            'feature_dim': max_features,
            'vocab': vocab  # Save for later use
        }

        return processed, metadata

    def create_splits(self, samples: List[Dict],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     shuffle: bool = True) -> Dict[str, List[Dict]]:
        """
        Create train/val/test splits.

        Args:
            samples: List of samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            shuffle: Whether to shuffle before splitting

        Returns:
            Dict with 'train', 'val', 'test' keys
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

        if shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        n = len(samples)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        splits = {
            'train': samples[:train_end],
            'val': samples[train_end:val_end],
            'test': samples[val_end:]
        }

        return splits

    def save_jsonl(self, samples: List[Dict], filepath: Path):
        """Save samples to JSONL file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                json.dump(sample, f)
                f.write('\n')

    def get_dataset_stats(self, samples: List[Dict]) -> Dict:
        """Calculate dataset statistics"""
        label_counts = Counter(s['label'] for s in samples)

        stats = {
            'num_samples': len(samples),
            'num_classes': len(label_counts),
            'classes': list(label_counts.keys()),
            'class_distribution': dict(label_counts),
            'avg_text_length': sum(len(s['text']) for s in samples) / len(samples) if samples else 0
        }

        return stats

    def preprocess_dataset(self, dataset_name: str, config: Dict) -> Dict:
        """
        Preprocess a single dataset.

        Config format:
        {
            "name": "dataset_name",
            "domain": "classification" | "sentiment" | "qa" | "translation",
            "source_dir": "optional_subdir",
            "format": "csv" | "jsonl" | "folders",
            "text_column": "text",
            "label_column": "label",
            "max_features": 512,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15
        }

        Returns:
            Processing result dict
        """
        print(f"\n{'='*60}")
        print(f"Preprocessing: {dataset_name}")
        print(f"{'='*60}")

        result = {
            'name': dataset_name,
            'success': False,
            'output_dir': None,
            'stats': None,
            'error': None
        }

        try:
            # Get source directory
            source_dir = self.extracted_dir / config.get('source_dir', dataset_name)
            if not source_dir.exists():
                raise ValueError(f"Source directory not found: {source_dir}")

            # Create output directory
            output_dir = self.processed_dir / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            result['output_dir'] = str(output_dir)

            # Load samples
            print("  Loading samples...")
            samples = self.load_text_classification(source_dir, config)
            print(f"    Loaded {len(samples)} samples")

            if not samples:
                raise ValueError("No samples loaded")

            # Extract features
            max_features = config.get('max_features', 512)
            processed_samples, feature_metadata = self.extract_features(samples, max_features)

            # Create splits
            print("  Creating splits...")
            train_ratio = config.get('train_ratio', 0.7)
            val_ratio = config.get('val_ratio', 0.15)
            test_ratio = config.get('test_ratio', 0.15)

            splits = self.create_splits(
                processed_samples,
                train_ratio, val_ratio, test_ratio
            )

            print(f"    Train: {len(splits['train'])} samples")
            print(f"    Val: {len(splits['val'])} samples")
            print(f"    Test: {len(splits['test'])} samples")

            # Save splits
            print("  Saving splits...")
            self.save_jsonl(splits['train'], output_dir / 'train.jsonl')
            self.save_jsonl(splits['val'], output_dir / 'val.jsonl')
            self.save_jsonl(splits['test'], output_dir / 'test.jsonl')
            self.save_jsonl(processed_samples, output_dir / 'all.jsonl')

            # Get statistics
            stats = self.get_dataset_stats(processed_samples)
            stats['feature_dim'] = max_features
            stats['vocab_size'] = feature_metadata['vocab_size']
            stats['domain'] = config.get('domain', 'classification')

            # Add split stats
            stats['splits'] = {
                'train': len(splits['train']),
                'val': len(splits['val']),
                'test': len(splits['test'])
            }

            # Save metadata
            metadata = {
                'dataset': dataset_name,
                'stats': stats,
                'config': config,
                'feature_metadata': {
                    'vocab_size': feature_metadata['vocab_size'],
                    'feature_dim': feature_metadata['feature_dim']
                }
            }

            with open(output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

            # Save vocabulary separately (can be large)
            with open(output_dir / 'vocab.json', 'w') as f:
                json.dump(feature_metadata['vocab'], f, indent=2)

            result['stats'] = stats
            result['success'] = True
            print(f"SUCCESS: {dataset_name}")
            print(f"  Classes: {stats['num_classes']}")
            print(f"  Features: {stats['feature_dim']}")

        except Exception as e:
            result['error'] = str(e)
            print(f"FAILED: {dataset_name} - {e}")
            import traceback
            traceback.print_exc()

        return result

    def preprocess_all(self, datasets: List[Dict]) -> Dict:
        """Preprocess multiple datasets"""
        results = []

        for dataset_config in datasets:
            result = self.preprocess_dataset(dataset_config['name'], dataset_config)
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print("PREPROCESSING SUMMARY")
        print(f"{'='*60}")

        success_count = sum(1 for r in results if r['success'])
        total_count = len(results)

        print(f"Total: {total_count}")
        print(f"Successful: {success_count}")
        print(f"Failed: {total_count - success_count}")

        if success_count < total_count:
            print("\nFailed datasets:")
            for r in results:
                if not r['success']:
                    print(f"  - {r['name']}: {r['error']}")

        return {
            'datasets': results,
            'total': total_count,
            'success': success_count,
            'failed': total_count - success_count
        }


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for NIMCP training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preprocess from config file
  python preprocess_datasets.py --config preprocess_config.json

  # Preprocess single dataset
  python preprocess_datasets.py --dataset mydata --source mydata --format csv
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='JSON config file with preprocessing instructions'
    )

    parser.add_argument(
        '--extracted-dir',
        type=str,
        help='Directory with extracted datasets (default: ../datasets/extracted)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: ../datasets/processed)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Single dataset name'
    )

    parser.add_argument(
        '--source',
        type=str,
        help='Source subdirectory (use with --dataset)'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['csv', 'jsonl', 'folders'],
        default='csv',
        help='Data format'
    )

    parser.add_argument(
        '--max-features',
        type=int,
        default=512,
        help='Maximum feature dimension (default: 512)'
    )

    args = parser.parse_args()

    # Create preprocessor
    preprocessor = DatasetPreprocessor(args.extracted_dir, args.output_dir)

    print("NIMCP Dataset Preprocessor")
    print(f"Extracted directory: {preprocessor.extracted_dir}")
    print(f"Output directory: {preprocessor.processed_dir}")
    print()

    # Get datasets to preprocess
    datasets = []

    if args.config:
        # Load from config file
        with open(args.config, 'r') as f:
            config = json.load(f)
        datasets = config.get('datasets', [])
        print(f"Loaded {len(datasets)} datasets from {args.config}")

    elif args.dataset:
        # Single dataset from command line
        datasets = [{
            'name': args.dataset,
            'source_dir': args.source or args.dataset,
            'format': args.format,
            'max_features': args.max_features
        }]

    else:
        print("Error: Provide either --config or --dataset")
        parser.print_help()
        return 1

    if not datasets:
        print("No datasets to preprocess")
        return 1

    # Preprocess all datasets
    results = preprocessor.preprocess_all(datasets)

    # Save results
    results_file = preprocessor.processed_dir / "preprocess_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
