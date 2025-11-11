#!/usr/bin/env python3
"""
NIMCP Dataset Downloader
Downloads priority training datasets for the 20-domain curriculum

Priority Datasets (Phase 1 focus):
1. Mathematics (MATH, MathQA, DeepMind Math)
2. Wikipedia (via Hugging Face)
3. Project Gutenberg samples
4. Basic science datasets

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset math
    python download_datasets.py --dataset wikipedia --samples 10000
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# Dataset storage location
DATA_DIR = Path("./datasets")
DATA_DIR.mkdir(exist_ok=True)

class DatasetDownloader:
    """Handles downloading and organizing training datasets"""

    def __init__(self, data_dir: Path = DATA_DIR):
        self.data_dir = data_dir
        self.downloaded = []

    def check_dependencies(self):
        """Check for required tools"""
        required = {
            'git': 'Git (for cloning repositories)',
            'wget': 'wget (for downloading files)',
            'curl': 'curl (alternative downloader)'
        }

        missing = []
        for cmd, desc in required.items():
            if subprocess.run(['which', cmd], capture_output=True).returncode != 0:
                missing.append(f"  - {desc}: {cmd}")

        if missing:
            print("⚠️  Missing dependencies:")
            print("\n".join(missing))
            print("\nInstall with: sudo apt-get install git wget curl")
            return False
        return True

    def download_math_datasets(self):
        """Download mathematics datasets (MATH, MathQA, DeepMind)"""
        print("\n📐 DOWNLOADING MATHEMATICS DATASETS")
        print("=" * 60)

        math_dir = self.data_dir / "mathematics"
        math_dir.mkdir(exist_ok=True)

        # 1. MATH Dataset (Hendrycks et al.)
        print("\n[1/3] Downloading MATH Dataset...")
        math_dataset_dir = math_dir / "MATH"
        if math_dataset_dir.exists():
            print("  ✓ MATH dataset already exists")
        else:
            try:
                subprocess.run([
                    'git', 'clone', '--depth', '1',
                    'https://github.com/hendrycks/math.git',
                    str(math_dataset_dir)
                ], check=True)
                print("  ✓ MATH dataset downloaded (12,500 problems)")
                self.downloaded.append("MATH Dataset")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to download MATH dataset: {e}")

        # 2. DeepMind Mathematics Dataset
        print("\n[2/3] Downloading DeepMind Mathematics Dataset...")
        deepmind_dir = math_dir / "deepmind_math"
        if deepmind_dir.exists():
            print("  ✓ DeepMind Math dataset already exists")
        else:
            try:
                subprocess.run([
                    'git', 'clone', '--depth', '1',
                    'https://github.com/deepmind/mathematics_dataset.git',
                    str(deepmind_dir)
                ], check=True)
                print("  ✓ DeepMind Math dataset downloaded")
                self.downloaded.append("DeepMind Mathematics")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to download DeepMind Math: {e}")

        # 3. MathQA
        print("\n[3/3] Downloading MathQA Dataset...")
        mathqa_dir = math_dir / "MathQA"
        if mathqa_dir.exists():
            print("  ✓ MathQA dataset already exists")
        else:
            mathqa_url = "https://math-qa.github.io/data/MathQA.zip"
            try:
                subprocess.run([
                    'wget', '-P', str(math_dir),
                    mathqa_url
                ], check=True)
                subprocess.run([
                    'unzip', '-q',
                    str(math_dir / 'MathQA.zip'),
                    '-d', str(mathqa_dir)
                ], check=True)
                os.remove(math_dir / 'MathQA.zip')
                print("  ✓ MathQA dataset downloaded (37K problems)")
                self.downloaded.append("MathQA")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to download MathQA: {e}")

        print("\n✓ Mathematics datasets ready")
        return True

    def download_wikipedia_sample(self, num_samples: int = 10000):
        """Download Wikipedia sample using Hugging Face datasets"""
        print(f"\n📖 DOWNLOADING WIKIPEDIA SAMPLE ({num_samples} articles)")
        print("=" * 60)

        wiki_dir = self.data_dir / "wikipedia"
        wiki_dir.mkdir(exist_ok=True)

        # Create Python script to download using Hugging Face
        download_script = wiki_dir / "download_wiki.py"
        with open(download_script, 'w') as f:
            f.write(f'''
import sys
try:
    from datasets import load_dataset
except ImportError:
    print("Error: datasets library not installed")
    print("Install with: pip install datasets")
    sys.exit(1)

print("Loading Wikipedia dataset from Hugging Face...")
print("This may take several minutes on first run...")

# Load Wikipedia dataset (streaming to avoid full download)
dataset = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)

# Take first {num_samples} examples
print(f"Extracting {num_samples} articles...")
samples = []
for i, example in enumerate(dataset):
    if i >= {num_samples}:
        break
    samples.append({{
        'id': example['id'],
        'url': example['url'],
        'title': example['title'],
        'text': example['text']
    }})
    if (i + 1) % 1000 == 0:
        print(f"  Processed {{i+1}} articles...")

# Save to JSON
import json
output_file = "{wiki_dir}/wikipedia_sample_{num_samples}.json"
with open(output_file, 'w') as f:
    json.dump(samples, f, indent=2)

print(f"✓ Saved {{len(samples)}} articles to {{output_file}}")
print(f"Total size: {{os.path.getsize(output_file) / 1024 / 1024:.1f}} MB")
''')

        print(f"\n📝 Created download script: {download_script}")
        print("\nTo download Wikipedia:")
        print(f"  1. Install datasets: pip install datasets")
        print(f"  2. Run script: python {download_script}")
        print(f"\nThis will download {num_samples} Wikipedia articles (~{num_samples * 3 / 1000:.0f} MB)")

        return True

    def download_gutenberg_samples(self):
        """Download sample books from Project Gutenberg"""
        print("\n📚 DOWNLOADING PROJECT GUTENBERG SAMPLES")
        print("=" * 60)

        gutenberg_dir = self.data_dir / "gutenberg"
        gutenberg_dir.mkdir(exist_ok=True)

        # Download a curated selection of important texts
        important_books = [
            # Philosophy
            (1342, "Pride and Prejudice - Jane Austen"),
            (84, "Frankenstein - Mary Shelley"),
            (1661, "Sherlock Holmes - Arthur Conan Doyle"),
            (11, "Alice in Wonderland - Lewis Carroll"),
            (1952, "Yellow Wallpaper - Charlotte Perkins Gilman"),
            # Poetry
            (1065, "The Raven - Edgar Allan Poe"),
            (1260, "Jane Eyre - Charlotte Bronte"),
            # Science
            (1, "United States Declaration of Independence"),
            (2, "US Bill of Rights"),
        ]

        print(f"\nDownloading {len(important_books)} foundational texts...")
        downloaded_count = 0

        for book_id, title in important_books:
            output_file = gutenberg_dir / f"{book_id}.txt"
            if output_file.exists():
                print(f"  ✓ [{book_id}] {title} (cached)")
                downloaded_count += 1
                continue

            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            try:
                subprocess.run([
                    'wget', '-q', '-O', str(output_file), url
                ], check=True, timeout=30)
                print(f"  ✓ [{book_id}] {title}")
                downloaded_count += 1
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Try alternative URL format
                url_alt = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                try:
                    subprocess.run([
                        'wget', '-q', '-O', str(output_file), url_alt
                    ], check=True, timeout=30)
                    print(f"  ✓ [{book_id}] {title}")
                    downloaded_count += 1
                except:
                    print(f"  ✗ [{book_id}] {title} - failed")

        print(f"\n✓ Downloaded {downloaded_count}/{len(important_books)} Gutenberg texts")
        if downloaded_count > 0:
            self.downloaded.append(f"Project Gutenberg ({downloaded_count} books)")
        return True

    def show_manual_downloads(self):
        """Show instructions for datasets requiring manual download"""
        print("\n" + "=" * 60)
        print("📋 DATASETS REQUIRING MANUAL DOWNLOAD")
        print("=" * 60)

        manual_datasets = [
            {
                'name': 'Common Voice (Speech)',
                'url': 'https://commonvoice.mozilla.org/datasets',
                'size': '~12 GB',
                'instructions': 'Download English dataset, extract to datasets/audio/common_voice/'
            },
            {
                'name': 'WikiArt (Visual Art)',
                'url': 'https://paperswithcode.com/dataset/wikiart',
                'size': '~25 GB',
                'instructions': 'Download images, extract to datasets/images/wikiart/'
            },
            {
                'name': 'UCF101 (Video Actions)',
                'url': 'https://www.crcv.ucf.edu/data/UCF101.php',
                'size': '~6.5 GB',
                'instructions': 'Download UCF101.rar, extract to datasets/video/ucf101/'
            },
            {
                'name': 'Perseus Digital Library',
                'url': 'https://www.perseus.tufts.edu/hopper/opensource/download',
                'size': '~500 MB',
                'instructions': 'Download XML corpus, extract to datasets/classics/perseus/'
            },
        ]

        for i, dataset in enumerate(manual_datasets, 1):
            print(f"\n[{i}] {dataset['name']}")
            print(f"    URL:  {dataset['url']}")
            print(f"    Size: {dataset['size']}")
            print(f"    📁    {dataset['instructions']}")

    def show_summary(self):
        """Show download summary"""
        print("\n" + "=" * 60)
        print("✅ DOWNLOAD SUMMARY")
        print("=" * 60)

        if self.downloaded:
            print("\nSuccessfully downloaded:")
            for dataset in self.downloaded:
                print(f"  ✓ {dataset}")

        print(f"\n📁 Datasets location: {self.data_dir.absolute()}")

        # Check total size
        try:
            result = subprocess.run(
                ['du', '-sh', str(self.data_dir)],
                capture_output=True,
                text=True,
                check=True
            )
            size = result.stdout.split()[0]
            print(f"📊 Total size: {size}")
        except:
            pass

        print("\n" + "=" * 60)
        print("NEXT STEPS:")
        print("=" * 60)
        print("1. Review downloaded datasets in ./datasets/")
        print("2. Download manual datasets (see instructions above)")
        print("3. Run: python streaming_trainer.py --phase 1")
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Download NIMCP training datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--all', action='store_true',
        help='Download all automatic datasets'
    )
    parser.add_argument(
        '--dataset', choices=['math', 'wikipedia', 'gutenberg'],
        help='Download specific dataset'
    )
    parser.add_argument(
        '--samples', type=int, default=10000,
        help='Number of Wikipedia samples (default: 10000)'
    )

    args = parser.parse_args()

    # Create downloader
    downloader = DatasetDownloader()

    print("=" * 60)
    print("🧠 NIMCP DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Target: {downloader.data_dir.absolute()}")

    # Check dependencies
    if not downloader.check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return 1

    # Download requested datasets
    if args.all:
        print("\n🎯 Downloading all automatic datasets...")
        downloader.download_math_datasets()
        downloader.download_wikipedia_sample(args.samples)
        downloader.download_gutenberg_samples()
    elif args.dataset == 'math':
        downloader.download_math_datasets()
    elif args.dataset == 'wikipedia':
        downloader.download_wikipedia_sample(args.samples)
    elif args.dataset == 'gutenberg':
        downloader.download_gutenberg_samples()
    else:
        parser.print_help()
        return 0

    # Show manual download instructions
    downloader.show_manual_downloads()

    # Show summary
    downloader.show_summary()

    return 0

if __name__ == '__main__':
    sys.exit(main())
