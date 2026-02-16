#!/usr/bin/env python3
"""Replace magic number neural constants with named constants.

Focuses on neural simulation parameters: membrane potentials, time constants,
neuromodulator baselines, synaptic weights, firing rates.

Usage:
    python3 scripts/replace_magic_neural.py [--dry-run]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_neural_constants.h"'

SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_neural_constants.h', 'nimcp_constants.h',
}

RULES = [
    # Membrane potentials
    (re.compile(r'(?:resting_potential|v_rest|rest_mv)\b', re.I), {
        '-70.0f': 'NIMCP_RESTING_POTENTIAL_MV',
    }),
    (re.compile(r'(?:threshold_potential|v_thresh|firing_threshold|spike_threshold)\b', re.I), {
        '-55.0f': 'NIMCP_FIRING_THRESHOLD_MV',
    }),
    (re.compile(r'(?:reset_potential|v_reset|reset_mv)\b', re.I), {
        '-65.0f': 'NIMCP_RESET_POTENTIAL_MV',
    }),
    # Neuromodulator baselines
    (re.compile(r'(?:dopamine_baseline|da_baseline|dopamine_level)\b', re.I), {
        '0.5f': 'NIMCP_DOPAMINE_BASELINE',
    }),
    (re.compile(r'(?:serotonin_baseline|_5ht_baseline|serotonin_level)\b', re.I), {
        '0.5f': 'NIMCP_SEROTONIN_BASELINE',
    }),
    (re.compile(r'(?:norepinephrine_baseline|ne_baseline|norepinephrine_level)\b', re.I), {
        '0.5f': 'NIMCP_NOREPINEPHRINE_BASELINE',
    }),
    (re.compile(r'(?:acetylcholine_baseline|ach_baseline|acetylcholine_level)\b', re.I), {
        '0.5f': 'NIMCP_ACETYLCHOLINE_BASELINE',
    }),
    (re.compile(r'(?:gaba_baseline|gaba_level)\b', re.I), {
        '0.5f': 'NIMCP_GABA_BASELINE',
    }),
    (re.compile(r'(?:glutamate_baseline|glutamate_level)\b', re.I), {
        '0.5f': 'NIMCP_GLUTAMATE_BASELINE',
    }),
    # Membrane time constant
    (re.compile(r'(?:membrane_tau|tau_m|tau_membrane)\b', re.I), {
        '20.0f': 'NIMCP_MEMBRANE_TAU_MS',
    }),
    # Refractory period
    (re.compile(r'(?:refractory_period|refract_ms|refractory_ms)\b', re.I), {
        '2.0f': 'NIMCP_REFRACTORY_PERIOD_MS',
    }),
    # Simulation time step
    (re.compile(r'(?:dt|time_step|delta_t|simulation_dt)\b', re.I), {
        '1.0f': 'NIMCP_SIMULATION_DT_MS',
        '0.1f': 'NIMCP_SIMULATION_DT_FINE_MS',
    }),
    # Initial synaptic weight
    (re.compile(r'(?:initial_weight|weight_init|synapse_weight_init)\b', re.I), {
        '0.5f': 'NIMCP_SYNAPSE_WEIGHT_INIT',
    }),
    # Connection probability
    (re.compile(r'(?:connection_probability|conn_prob|connect_prob)\b', re.I), {
        '0.1f': 'NIMCP_CONNECTION_PROBABILITY',
    }),
    # Noise amplitude
    (re.compile(r'(?:noise_amplitude|noise_amp|neural_noise)\b', re.I), {
        '0.1f': 'NIMCP_NEURAL_NOISE_AMPLITUDE',
    }),
    # Phase coherence
    (re.compile(r'(?:phase_coherence|coherence_threshold)\b', re.I), {
        '0.7f': 'NIMCP_PHASE_COHERENCE_THRESHOLD',
    }),
    # Coupling strength
    (re.compile(r'(?:coupling_strength|coupling)\b', re.I), {
        '0.1f': 'NIMCP_OSCILLATION_COUPLING_DEFAULT',
    }),
]

FIELD_ASSIGN_RE = re.compile(
    r'(\.\s*(\w+)\s*=\s*)'
    r'(-?[\d.]+f)'
    r'(\s*[;,])',
)

VAR_DECL_RE = re.compile(
    r'((?:float|double|_Atomic\s+float)\s+(\w+)\s*=\s*)'
    r'(-?[\d.]+f)'
    r'(\s*;)',
)

DEFINE_RE = re.compile(
    r'(#define\s+(\w+)\s+)'
    r'(-?[\d.]+f)'
    r'(\s*(?://.*|/\*.*)?$)',
    re.MULTILINE
)


def has_include(content):
    return (INCLUDE in content or
            '"constants/nimcp_constants.h"' in content or
            'nimcp_neural_constants.h' in content)


def add_include(content):
    lines = content.split('\n')
    in_block = False
    last_include = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#include'):
            in_block = True
            last_include = i
        elif in_block and stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*') and not stripped.startswith('#'):
            break
    if last_include == -1:
        return content
    lines.insert(last_include + 1, f'#include {INCLUDE}')
    return '\n'.join(lines)


def try_replace(name, value_str):
    for name_pattern, value_map in RULES:
        if name_pattern.search(name):
            if value_str in value_map:
                return value_map[value_str]
    return None


def process_file(filepath, dry_run=False):
    basename = os.path.basename(filepath)
    if basename in SKIP_FILES:
        return 0
    with open(filepath) as f:
        content = f.read()

    replacements = [0]

    def replace_match(m):
        prefix = m.group(1)
        name = m.group(2)
        value = m.group(3)
        suffix = m.group(4)
        const = try_replace(name, value)
        if const:
            if 'NIMCP_' in m.group(0):
                return m.group(0)
            replacements[0] += 1
            return f'{prefix}{const}{suffix}'
        return m.group(0)

    content = FIELD_ASSIGN_RE.sub(replace_match, content)
    content = VAR_DECL_RE.sub(replace_match, content)
    content = DEFINE_RE.sub(replace_match, content)

    if replacements[0] == 0:
        return 0

    if not has_include(content):
        content = add_include(content)

    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return replacements[0]


def main():
    dry_run = '--dry-run' in sys.argv
    total = 0
    files_mod = 0
    files_checked = 0

    for root, dirs, files in os.walk(os.path.join(ROOT, 'src')):
        dirs[:] = [d for d in dirs if d not in ('build', '.git', '__pycache__')]
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            files_checked += 1
            count = process_file(filepath, dry_run)
            if count > 0:
                rel = os.path.relpath(filepath, ROOT)
                action = "WOULD FIX" if dry_run else "FIXED"
                print(f"  {action} ({count}): {rel}")
                total += count
                files_mod += 1

    action = "Would modify" if dry_run else "Modified"
    print(f"\n{action} {files_mod} files with {total} replacements ({files_checked} checked)")


if __name__ == '__main__':
    main()
