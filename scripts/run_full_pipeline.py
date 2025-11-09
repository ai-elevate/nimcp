#!/usr/bin/env python3
"""
NIMCP Full Training Pipeline
Orchestrates the complete training pipeline: download -> preprocess -> train

This script runs all three stages:
1. Download datasets from URLs
2. Preprocess into NIMCP format
3. Train foundation model with progressive learning

Features:
- End-to-end automation
- Progress tracking
- Error handling and recovery
- Configurable stages (can skip completed stages)
- Summary reports
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class PipelineRunner:
    """Orchestrates the full training pipeline"""

    def __init__(self, config_file: str, base_dir: str = None):
        self.config_file = Path(config_file)
        self.config = self.load_config()

        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        self.base_dir = Path(base_dir)

        self.scripts_dir = self.base_dir / "scripts"
        self.datasets_dir = self.base_dir / "datasets"
        self.models_dir = self.base_dir / "models"

        # Stage flags
        self.stages = {
            'download': True,
            'preprocess': True,
            'train': True
        }

        # Results
        self.results = {
            'start_time': None,
            'end_time': None,
            'stages': {},
            'success': False
        }

    def load_config(self) -> Dict:
        """Load pipeline configuration"""
        with open(self.config_file, 'r') as f:
            return json.load(f)

    def run_script(self, script_name: str, args: List[str],
                  stage_name: str) -> bool:
        """
        Run a Python script and capture output.

        Returns:
            True if successful, False otherwise
        """
        script_path = self.scripts_dir / script_name

        if not script_path.exists():
            print(f"Error: Script not found: {script_path}")
            return False

        cmd = [sys.executable, str(script_path)] + args

        print(f"\n{'='*60}")
        print(f"Stage: {stage_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        stage_start = time.time()

        try:
            # Run script
            result = subprocess.run(
                cmd,
                cwd=self.scripts_dir,
                check=False,  # Don't raise on non-zero exit
                capture_output=False,  # Show output in real-time
                text=True
            )

            stage_time = time.time() - stage_start

            # Record result
            self.results['stages'][stage_name] = {
                'success': result.returncode == 0,
                'return_code': result.returncode,
                'duration': stage_time
            }

            if result.returncode == 0:
                print(f"\n[SUCCESS] {stage_name} completed in {stage_time:.1f}s")
                return True
            else:
                print(f"\n[FAILED] {stage_name} failed with code {result.returncode}")
                return False

        except Exception as e:
            stage_time = time.time() - stage_start
            print(f"\n[ERROR] {stage_name} failed: {e}")

            self.results['stages'][stage_name] = {
                'success': False,
                'error': str(e),
                'duration': stage_time
            }

            return False

    def stage_download(self) -> bool:
        """Stage 1: Download datasets"""
        if not self.stages['download']:
            print("\nSkipping download stage (disabled)")
            return True

        # Get dataset config
        dataset_config = self.config.get('dataset_config')
        if not dataset_config:
            print("Error: No 'dataset_config' in pipeline config")
            return False

        # If it's a path to a file, use it directly
        if os.path.exists(dataset_config):
            config_file = dataset_config
        else:
            # Look for it in scripts directory
            config_file = self.scripts_dir / dataset_config

        if not os.path.exists(config_file):
            print(f"Error: Dataset config not found: {config_file}")
            return False

        args = [
            '--config', str(config_file),
            '--output', str(self.datasets_dir)
        ]

        return self.run_script('download_datasets.py', args, 'Download')

    def stage_preprocess(self) -> bool:
        """Stage 2: Preprocess datasets"""
        if not self.stages['preprocess']:
            print("\nSkipping preprocess stage (disabled)")
            return True

        # Get preprocess config
        preprocess_config = self.config.get('preprocess_config')
        if not preprocess_config:
            print("Error: No 'preprocess_config' in pipeline config")
            return False

        # If it's a path to a file, use it directly
        if os.path.exists(preprocess_config):
            config_file = preprocess_config
        else:
            # Look for it in scripts directory
            config_file = self.scripts_dir / preprocess_config

        if not os.path.exists(config_file):
            print(f"Error: Preprocess config not found: {config_file}")
            return False

        args = [
            '--config', str(config_file),
            '--extracted-dir', str(self.datasets_dir / 'extracted'),
            '--output-dir', str(self.datasets_dir / 'processed')
        ]

        return self.run_script('preprocess_datasets.py', args, 'Preprocess')

    def stage_train(self) -> bool:
        """Stage 3: Train foundation model"""
        if not self.stages['train']:
            print("\nSkipping training stage (disabled)")
            return True

        # Get training config
        training_config = self.config.get('training_config')
        if not training_config:
            print("Error: No 'training_config' in pipeline config")
            return False

        # If it's a path to a file, use it directly
        if os.path.exists(training_config):
            config_file = training_config
        else:
            # Look for it in scripts directory
            config_file = self.scripts_dir / training_config

        if not os.path.exists(config_file):
            print(f"Error: Training config not found: {config_file}")
            return False

        # Get output directory from config or use default
        output_dir = self.config.get('output_dir')
        if output_dir:
            output_dir = self.base_dir / output_dir
        else:
            # Generate timestamped output directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = self.models_dir / f"run_{timestamp}"

        args = [
            '--config', str(config_file),
            '--datasets-dir', str(self.datasets_dir / 'processed'),
            '--output', str(output_dir)
        ]

        return self.run_script('train_foundation_model.py', args, 'Train')

    def run(self) -> bool:
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("NIMCP FULL TRAINING PIPELINE")
        print("="*60)
        print(f"Config: {self.config_file}")
        print(f"Base directory: {self.base_dir}")
        print()

        self.results['start_time'] = time.time()

        # Stage 1: Download
        if not self.stage_download():
            print("\n[PIPELINE FAILED] Download stage failed")
            self.results['end_time'] = time.time()
            return False

        # Stage 2: Preprocess
        if not self.stage_preprocess():
            print("\n[PIPELINE FAILED] Preprocess stage failed")
            self.results['end_time'] = time.time()
            return False

        # Stage 3: Train
        if not self.stage_train():
            print("\n[PIPELINE FAILED] Training stage failed")
            self.results['end_time'] = time.time()
            return False

        self.results['end_time'] = time.time()
        self.results['success'] = True

        # Print summary
        self.print_summary()

        return True

    def print_summary(self):
        """Print pipeline summary"""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)

        total_time = self.results['end_time'] - self.results['start_time']

        for stage_name, stage_result in self.results['stages'].items():
            status = "SUCCESS" if stage_result['success'] else "FAILED"
            duration = stage_result.get('duration', 0)
            print(f"  {stage_name:12s}: {status:7s} ({duration:.1f}s)")

        print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Status: {'SUCCESS' if self.results['success'] else 'FAILED'}")

        # Save results
        results_file = self.scripts_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete NIMCP training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python run_full_pipeline.py --config pipeline_config.json

  # Skip download stage (datasets already downloaded)
  python run_full_pipeline.py --config pipeline_config.json --skip-download

  # Only run training (data already prepared)
  python run_full_pipeline.py --config pipeline_config.json --only-train

  # Custom base directory
  python run_full_pipeline.py --config pipeline_config.json --base-dir /path/to/nimcp

Pipeline Config Format:
{
  "dataset_config": "example_datasets_config.json",
  "preprocess_config": "example_preprocess_config.json",
  "training_config": "example_training_config.json",
  "output_dir": "models/my_run"
}
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Pipeline configuration file (JSON)'
    )

    parser.add_argument(
        '--base-dir',
        type=str,
        help='Base directory (default: parent of scripts dir)'
    )

    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip download stage'
    )

    parser.add_argument(
        '--skip-preprocess',
        action='store_true',
        help='Skip preprocess stage'
    )

    parser.add_argument(
        '--only-train',
        action='store_true',
        help='Only run training stage'
    )

    args = parser.parse_args()

    # Create runner
    runner = PipelineRunner(args.config, args.base_dir)

    # Configure stages
    if args.skip_download:
        runner.stages['download'] = False

    if args.skip_preprocess:
        runner.stages['preprocess'] = False

    if args.only_train:
        runner.stages['download'] = False
        runner.stages['preprocess'] = False

    # Run pipeline
    success = runner.run()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
