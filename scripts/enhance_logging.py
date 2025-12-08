#!/usr/bin/env python3
"""
Add comprehensive logging and LOG_MODULE_ID to integrated files.
"""

import os
import re
import sys
from pathlib import Path
from typing import List

# Module ID mappings
MODULE_IDS = {
    'nimcp_circular_buffer': 0x0510,
    'nimcp_integration_buffer': 0x0511,
    'nimcp_phase_coded_buffer': 0x0512,
    'nimcp_sliding_window': 0x0513,
    'nimcp_temporal_accumulator': 0x0514,
    'nimcp_cognitive_adapters': 0x0515,
    'nimcp_working_memory_adapter': 0x0516,
    'nimcp_population_coding': 0x0517,
    'nimcp_rate_coding': 0x0518,
    'nimcp_temporal_coding': 0x0519,
    'nimcp_feature_extractor': 0x051A,
    'nimcp_executive_middleware_adapter': 0x051B,
    'nimcp_flow_tracker': 0x051C,
    'nimcp_middleware_controller': 0x051D,
    'nimcp_quantum_command_propagator': 0x051E,
    'nimcp_shannon_monitor': 0x051F,
    'nimcp_adaptive_normalizer': 0x0520,
    'nimcp_homeostatic_normalizer': 0x0521,
    'nimcp_min_max_normalizer': 0x0522,
    'nimcp_zscore_normalizer': 0x0523,
    'nimcp_oscillation_detector': 0x0524,
    'nimcp_pattern_cow': 0x0525,
    'nimcp_pattern_library': 0x0526,
    'nimcp_sequence_detector': 0x0527,
    'nimcp_synchrony_detector': 0x0528,
    'nimcp_attention_gate': 0x0529,
    'nimcp_routing_table': 0x052A,
    'nimcp_signal_wrapper': 0x052B,
    'nimcp_thalamic_router': 0x052C,
    'nimcp_dataio': 0x052D,
    'nimcp_serialization': 0x052E,
    'nimcp_network_serialization': 0x052E,
    'nimcp_encryption': 0x052E,
    'nimcp_stream': 0x052F,
    'brain_integration': 0x0510,
}


def add_log_module_id(filepath: Path) -> bool:
    """Add LOG_MODULE_ID define if missing."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if LOG_MODULE_ID already exists
        if 'LOG_MODULE_ID' in content:
            return True

        # Get module ID
        module_name = filepath.stem
        module_id = MODULE_IDS.get(module_name, 0x0510)

        lines = content.split('\n')

        # Find MODULE_NAME or LOG_MODULE define
        insert_pos = -1
        for i, line in enumerate(lines):
            if 'MODULE_NAME' in line or ('LOG_MODULE' in line and 'define' in line):
                insert_pos = i + 1
                break

        if insert_pos == -1:
            # Find position after includes
            for i, line in enumerate(lines):
                if line.strip().startswith('#include'):
                    insert_pos = i + 1

        if insert_pos == -1:
            return False

        # Insert LOG_MODULE_ID
        lines.insert(insert_pos, f'#define LOG_MODULE_ID 0x{module_id:04X}')

        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"  ✅ Added LOG_MODULE_ID to {filepath.name}")
        return True

    except Exception as e:
        print(f"  ❌ Failed to add LOG_MODULE_ID to {filepath.name}: {e}")
        return False


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # All middleware and IO directories
    dirs_to_process = [
        'src/middleware/buffering',
        'src/middleware/cognitive',
        'src/middleware/encoding',
        'src/middleware/features',
        'src/middleware/integration',
        'src/middleware/normalization',
        'src/middleware/patterns',
        'src/middleware/routing',
        'src/io/dataio',
        'src/io/serialization',
        'src/io/stream',
    ]

    print("=" * 80)
    print("Adding LOG_MODULE_ID to all integrated files")
    print("=" * 80)

    total = 0
    success = 0

    for dir_path in dirs_to_process:
        full_path = project_root / dir_path
        if not full_path.exists():
            continue

        c_files = list(full_path.glob('*.c'))
        c_files = [f for f in c_files if 'CMakeFiles' not in str(f)]

        if not c_files:
            continue

        print(f"\n📁 {dir_path}")
        for filepath in sorted(c_files):
            total += 1
            if add_log_module_id(filepath):
                success += 1

    # Also process top-level middleware file
    middleware_file = project_root / 'src' / 'middleware' / 'brain_integration.c'
    if middleware_file.exists():
        print(f"\n📁 src/middleware")
        total += 1
        if add_log_module_id(middleware_file):
            success += 1

    print("\n" + "=" * 80)
    print(f"Processed: {success}/{total}")
    print("=" * 80)


if __name__ == '__main__':
    main()
