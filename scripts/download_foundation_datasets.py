#!/usr/bin/env python3
"""
Download Foundation Datasets for NIMCP
Downloads all datasets specified in foundation_datasets_config.json
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

def main():
    # Load config
    config_file = Path(__file__).parent / "foundation_datasets_config.json"
    with open(config_file) as f:
        config = json.load(f)
    
    # Create datasets directory
    datasets_dir = Path(__file__).parent.parent / "datasets" / "foundation"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Downloading {len(config['datasets'])} Foundation Datasets ===\n")
    print(f"Output directory: {datasets_dir}\n")
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, dataset_config in enumerate(config['datasets'], 1):
        name = dataset_config['name']
        domain = dataset_config['domain']
        dataset_type = dataset_config['type']
        
        print(f"[{i}/{len(config['datasets'])}] {name} ({domain})...")
        
        # Skip non-HuggingFace datasets for now
        if dataset_type != 'huggingface':
            print(f"  ⊙ Skipped (type: {dataset_type})\n")
            skip_count += 1
            continue
        
        try:
            # Download HuggingFace dataset
            hf_dataset = dataset_config['hf_dataset']
            hf_subset = dataset_config.get('hf_subset', None)
            
            print(f"  Downloading from HuggingFace: {hf_dataset}", end='')
            if hf_subset:
                print(f" (subset: {hf_subset})", end='')
            print()
            
            # Load dataset (this caches it locally)
            if hf_subset:
                dataset = load_dataset(hf_dataset, hf_subset, cache_dir=str(datasets_dir / ".cache"))
            else:
                dataset = load_dataset(hf_dataset, cache_dir=str(datasets_dir / ".cache"))
            
            # Save to disk in JSONL format
            output_dir = datasets_dir / name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each split
            for split_name, split_data in dataset.items():
                output_file = output_dir / f"{split_name}.jsonl"
                split_data.to_json(output_file, orient='records', lines=True)
                print(f"  ✓ Saved {split_name}: {len(split_data)} examples -> {output_file.name}")
            
            # Save metadata
            metadata = {
                'name': name,
                'domain': domain,
                'hf_dataset': hf_dataset,
                'hf_subset': hf_subset,
                'description': dataset_config['description'],
                'splits': {split: len(data) for split, data in dataset.items()}
            }
            
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Success!\n")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Failed: {e}\n")
            fail_count += 1
    
    # Summary
    print("\n=== Download Summary ===")
    print(f"✓ Success: {success_count}")
    print(f"⊙ Skipped: {skip_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"Total: {len(config['datasets'])}")
    print(f"\nDatasets saved to: {datasets_dir}")
    
    return 0 if fail_count == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
