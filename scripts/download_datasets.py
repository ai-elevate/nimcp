#!/usr/bin/env python3
"""
NIMCP Dataset Downloader
Downloads and organizes datasets for training foundation models.

Features:
- Downloads datasets from URLs
- Extracts archives (zip, tar.gz, tar.bz2)
- Validates downloads with checksums
- Organizes into structured directory layout
- Progress tracking and resumable downloads

Directory Structure:
/datasets/
    /raw/           # Original downloaded files
    /extracted/     # Extracted datasets
    /processed/     # Preprocessed data (created by preprocess_datasets.py)
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from urllib.request import urlretrieve
from urllib.parse import urlparse


class DownloadProgressBar:
    """Simple progress bar for downloads"""

    def __init__(self, filename: str):
        self.filename = filename
        self.last_percent = -1

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if total_size <= 0:
            return

        downloaded = block_num * block_size
        percent = min(100, int(downloaded * 100 / total_size))

        if percent != self.last_percent:
            self.last_percent = percent
            bar_length = 40
            filled = int(bar_length * percent / 100)
            bar = '=' * filled + '-' * (bar_length - filled)

            size_mb = total_size / (1024 * 1024)
            downloaded_mb = downloaded / (1024 * 1024)

            print(f'\r{self.filename}: [{bar}] {percent}% ({downloaded_mb:.1f}/{size_mb:.1f} MB)',
                  end='', flush=True)

        if percent == 100:
            print()  # New line when complete


class DatasetDownloader:
    """Handles dataset downloading, extraction, and validation"""

    def __init__(self, base_dir: str = None):
        if base_dir is None:
            # Default to /home/bbrelin/nimcp/datasets
            script_dir = Path(__file__).parent
            base_dir = script_dir.parent / "datasets"

        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.extracted_dir = self.base_dir / "extracted"
        self.processed_dir = self.base_dir / "processed"

        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, filename: str = None,
                      checksum: str = None) -> Path:
        """
        Download a file from URL with progress tracking.

        Args:
            url: URL to download from
            filename: Optional filename (default: extract from URL)
            checksum: Optional MD5 checksum for validation

        Returns:
            Path to downloaded file
        """
        if filename is None:
            filename = Path(urlparse(url).path).name

        output_path = self.raw_dir / filename

        # Check if already downloaded and valid
        if output_path.exists():
            if checksum is None or self._verify_checksum(output_path, checksum):
                print(f"Already downloaded: {filename}")
                return output_path
            else:
                print(f"Checksum mismatch, re-downloading: {filename}")
                output_path.unlink()

        # Download with progress bar
        print(f"Downloading: {url}")
        progress = DownloadProgressBar(filename)

        try:
            urlretrieve(url, output_path, reporthook=progress)
        except Exception as e:
            print(f"\nError downloading {url}: {e}")
            if output_path.exists():
                output_path.unlink()
            raise

        # Verify checksum if provided
        if checksum and not self._verify_checksum(output_path, checksum):
            output_path.unlink()
            raise ValueError(f"Checksum verification failed for {filename}")

        return output_path

    def _verify_checksum(self, filepath: Path, expected_md5: str) -> bool:
        """Verify MD5 checksum of file"""
        md5 = hashlib.md5()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)

        actual_md5 = md5.hexdigest()
        return actual_md5.lower() == expected_md5.lower()

    def extract_archive(self, archive_path: Path, extract_to: str = None) -> Path:
        """
        Extract archive (zip, tar.gz, tar.bz2).

        Args:
            archive_path: Path to archive file
            extract_to: Optional subdirectory name in extracted_dir

        Returns:
            Path to extracted directory
        """
        if extract_to is None:
            # Use archive name without extension
            extract_to = archive_path.stem
            if extract_to.endswith('.tar'):
                extract_to = extract_to[:-4]

        output_dir = self.extracted_dir / extract_to

        # Check if already extracted
        if output_dir.exists() and any(output_dir.iterdir()):
            print(f"Already extracted: {extract_to}")
            return output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Extracting: {archive_path.name} -> {extract_to}/")

        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    zf.extractall(output_dir)
            elif archive_path.suffix in ['.gz', '.bz2'] or '.tar' in archive_path.name:
                with tarfile.open(archive_path, 'r:*') as tf:
                    tf.extractall(output_dir)
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")

            print(f"Extracted successfully: {extract_to}/")

        except Exception as e:
            print(f"Error extracting {archive_path.name}: {e}")
            if output_dir.exists():
                shutil.rmtree(output_dir)
            raise

        return output_dir

    def download_dataset(self, config: Dict) -> Dict:
        """
        Download and extract a dataset from configuration.

        Config format:
        {
            "name": "dataset_name",
            "url": "https://...",
            "filename": "optional_filename.zip",
            "md5": "optional_checksum",
            "extract_to": "optional_subdir"
        }

        Returns:
            Dict with download info (paths, status)
        """
        name = config['name']
        url = config['url']
        filename = config.get('filename')
        md5 = config.get('md5')
        extract_to = config.get('extract_to')

        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        result = {
            'name': name,
            'success': False,
            'raw_path': None,
            'extracted_path': None,
            'error': None
        }

        try:
            # Download
            raw_path = self.download_file(url, filename, md5)
            result['raw_path'] = str(raw_path)

            # Extract if it's an archive
            if raw_path.suffix in ['.zip', '.gz', '.bz2'] or '.tar' in raw_path.name:
                extracted_path = self.extract_archive(raw_path, extract_to)
                result['extracted_path'] = str(extracted_path)
            else:
                # Not an archive, just copy to extracted dir
                dest = self.extracted_dir / name
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(raw_path, dest / raw_path.name)
                result['extracted_path'] = str(dest)

            result['success'] = True
            print(f"SUCCESS: {name}")

        except Exception as e:
            result['error'] = str(e)
            print(f"FAILED: {name} - {e}")

        return result

    def download_all(self, datasets: List[Dict]) -> Dict:
        """
        Download multiple datasets.

        Args:
            datasets: List of dataset configurations

        Returns:
            Summary dict with results for each dataset
        """
        results = []

        for dataset_config in datasets:
            result = self.download_dataset(dataset_config)
            results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
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


def load_dataset_config(config_file: str) -> List[Dict]:
    """Load dataset configurations from JSON file"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config.get('datasets', [])


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for NIMCP training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from config file
  python download_datasets.py --config datasets.json

  # Download to custom directory
  python download_datasets.py --config datasets.json --output /path/to/datasets

  # Download single dataset
  python download_datasets.py --url https://example.com/data.zip --name mydata
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        help='JSON config file with dataset URLs'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (default: ../datasets)'
    )

    parser.add_argument(
        '--url',
        type=str,
        help='Single dataset URL (use with --name)'
    )

    parser.add_argument(
        '--name',
        type=str,
        help='Dataset name (use with --url)'
    )

    parser.add_argument(
        '--md5',
        type=str,
        help='MD5 checksum for single dataset'
    )

    args = parser.parse_args()

    # Create downloader
    downloader = DatasetDownloader(args.output)

    print("NIMCP Dataset Downloader")
    print(f"Base directory: {downloader.base_dir}")
    print()

    # Get datasets to download
    datasets = []

    if args.config:
        # Load from config file
        datasets = load_dataset_config(args.config)
        print(f"Loaded {len(datasets)} datasets from {args.config}")

    elif args.url and args.name:
        # Single dataset from command line
        datasets = [{
            'name': args.name,
            'url': args.url,
            'md5': args.md5
        }]

    else:
        print("Error: Provide either --config or both --url and --name")
        parser.print_help()
        return 1

    if not datasets:
        print("No datasets to download")
        return 1

    # Download all datasets
    results = downloader.download_all(datasets)

    # Save results
    results_file = downloader.base_dir / "download_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
