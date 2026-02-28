#!/usr/bin/env python3
"""Fix const-correctness in header declarations, v2.

For each source file where we removed const from a function parameter that locks
a mutex, find and update the corresponding header declaration.
"""
import re
import os
import sys


def get_fixed_signatures(src_dir):
    """Get dict of func_name -> type_name for functions that were fixed."""
    skip_types = {
        'cognitive_event_data_t', 'brain_event_t', 'resource_metrics_t',
        'rcog_subtask_result_t', 'surprise_thalamic_signal_t',
        'collective_fep_config_t'
    }

    # We know the source files were fixed. Now find functions that:
    # 1. Have non-const type_t* as first meaningful param
    # 2. Call mutex_lock in their body
    # 3. The type_t ends with _bridge_t, _system_t, _engine_t, etc. (struct types that hold mutexes)

    fixed = {}  # func_name -> type_name

    for root, dirs, files in os.walk(src_dir):
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                content = f.read()

            lines = content.split('\n')
            i = 0
            while i < len(lines):
                line = lines[i]
                # Match function defs with non-const type_t* (already fixed)
                # but only the "accessor" pattern functions
                m = re.search(r'^(?:\w+\s+\*?\s*)*(\w+)\s*\((\w+_t)\s*\*\s*(\w+)', line)
                if m:
                    func_name = m.group(1)
                    type_name = m.group(2)
                    param_name = m.group(3)

                    if type_name in skip_types:
                        i += 1
                        continue

                    # Only look at accessor-type functions
                    accessor_patterns = ['get_', 'is_', '_get_', '_is_', '_get_stats', '_get_state',
                                       '_get_score', '_get_level', '_get_count', '_get_config',
                                       '_get_health', '_get_info', '_get_metrics', '_get_status',
                                       '_get_current', '_get_free_energy', '_get_prediction',
                                       '_get_attention', '_is_registered', '_is_degraded',
                                       '_is_impaired', '_is_converging', '_is_at_nash',
                                       '_is_coherent', '_is_efficient', '_is_connected',
                                       '_is_high_', '_is_offline', '_is_consolidation',
                                       '_is_in_refractory', '_get_last_', '_get_sync_stats',
                                       '_get_capabilities', '_get_node_by_name',
                                       '_get_direction', '_get_conditions', '_get_effects',
                                       '_get_omni_effects', '_get_logic_effects', '_get_rcog_effects',
                                       '_get_strategy', '_get_aggregate', '_get_confidence',
                                       '_get_id', '_get_broadcast', '_get_subscriber',
                                       '_get_competitor', '_get_history', '_get_statistics',
                                       '_get_ignition', '_get_replay', '_get_checkpoint',
                                       '_get_pressure_state', '_get_sleep_state',
                                       '_get_last_event', '_get_inference_capacity']

                    if not any(p in func_name for p in accessor_patterns):
                        i += 1
                        continue

                    # Check for mutex_lock in body
                    sig_start = i
                    sig = line
                    j = i
                    while '{' not in sig and j < min(i + 10, len(lines) - 1):
                        j += 1
                        sig += '\n' + lines[j]
                    if '{' not in sig:
                        i += 1
                        continue
                    brace_count = 0
                    for k in range(sig_start, len(lines)):
                        brace_count += lines[k].count('{') - lines[k].count('}')
                        if brace_count == 0 and k > sig_start:
                            break
                    body = '\n'.join(lines[sig_start:k + 1])
                    if 'mutex_lock' in body:
                        fixed[func_name] = type_name
                i += 1

    return fixed


def fix_headers(include_dir, fixed_funcs, base_dir):
    """Fix header declarations."""
    total = 0
    file_count = 0

    for root, dirs, files in os.walk(include_dir):
        for fname in sorted(files):
            if not fname.endswith('.h'):
                continue
            fpath = os.path.join(root, fname)
            with open(fpath, 'r') as f:
                content = f.read()

            original_content = content
            lines = content.split('\n')
            fixes = 0

            for func_name, type_name in fixed_funcs.items():
                # Search for the function declaration in this header
                for li in range(len(lines)):
                    if func_name not in lines[li]:
                        continue

                    # Check if this line or nearby lines have the const type_name* pattern
                    # Headers can have multi-line declarations
                    for offset in range(6):
                        if li + offset >= len(lines):
                            break
                        target = 'const ' + type_name + '*'
                        if target in lines[li + offset]:
                            # Make sure this is actually part of this function's declaration
                            # by checking the function name is nearby
                            context_start = max(0, li - 1)
                            context_end = min(len(lines), li + offset + 2)
                            context = '\n'.join(lines[context_start:context_end])
                            if func_name in context:
                                lines[li + offset] = lines[li + offset].replace(
                                    target, type_name + '*', 1)
                                fixes += 1
                                break

            if fixes > 0:
                new_content = '\n'.join(lines)
                if new_content != original_content:
                    with open(fpath, 'w') as f:
                        f.write(new_content)
                    file_count += 1
                    total += fixes
                    relpath = os.path.relpath(fpath, base_dir)
                    print(f'{relpath}: {fixes} fixes')

    return total, file_count


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(base_dir, 'src', 'cognitive')
    include_dir = os.path.join(base_dir, 'include', 'cognitive')

    print("Phase 1: Collecting fixed accessor functions from source files...")
    fixed_funcs = get_fixed_signatures(src_dir)
    print(f"Found {len(fixed_funcs)} accessor functions that lock mutexes")

    print("\nPhase 2: Fixing header declarations...")
    total, file_count = fix_headers(include_dir, fixed_funcs, base_dir)

    print(f'\nTotal: {total} header fixes in {file_count} files')
    return 0


if __name__ == '__main__':
    sys.exit(main())
