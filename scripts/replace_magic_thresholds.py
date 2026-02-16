#!/usr/bin/env python3
"""Replace magic number thresholds with named constants from nimcp_threshold_constants.h.

Focuses on safe patterns where context clearly indicates a threshold:
- struct fields named *threshold*, *confidence*, *similarity*, *temperature*
- Variable declarations with threshold-related names

Usage:
    python3 scripts/replace_magic_thresholds.py [--dry-run]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_threshold_constants.h"'

SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_threshold_constants.h', 'nimcp_constants.h',
}

RULES = [
    # Activation thresholds
    (re.compile(r'(?:activation_threshold|act_threshold)\b', re.I), {
        '0.5f': 'NIMCP_ACTIVATION_THRESHOLD',
    }),
    # Novelty thresholds
    (re.compile(r'novelty_threshold\b', re.I), {
        '0.7f': 'NIMCP_NOVELTY_THRESHOLD',
    }),
    # Salience thresholds
    (re.compile(r'salience_threshold\b', re.I), {
        '0.3f': 'NIMCP_SALIENCE_THRESHOLD',
    }),
    # Attention thresholds
    (re.compile(r'attention_threshold\b', re.I), {
        '0.5f': 'NIMCP_ATTENTION_THRESHOLD',
    }),
    # Conflict thresholds
    (re.compile(r'conflict_threshold\b', re.I), {
        '0.5f': 'NIMCP_CONFLICT_THRESHOLD',
    }),
    # Similarity thresholds
    (re.compile(r'(?:similarity_threshold|sim_threshold)\b', re.I), {
        '0.8f': 'NIMCP_SIMILARITY_THRESHOLD',
    }),
    # Distance thresholds
    (re.compile(r'distance_threshold\b', re.I), {
        '0.5f': 'NIMCP_DISTANCE_THRESHOLD',
    }),
    # Temperature
    (re.compile(r'temperature\b', re.I), {
        '1.0f': 'NIMCP_TEMPERATURE_DEFAULT',
        '2.0f': 'NIMCP_TEMPERATURE_HIGH',
        '0.5f': 'NIMCP_TEMPERATURE_LOW',
    }),
    # Exploration rate
    (re.compile(r'(?:exploration_rate|epsilon_greedy|explore_rate)\b', re.I), {
        '0.1f': 'NIMCP_EXPLORATION_RATE_DEFAULT',
    }),
    # Confidence
    (re.compile(r'(?:confidence_threshold|min_confidence|conf_threshold)\b', re.I), {
        '0.8f': 'NIMCP_CONFIDENCE_HIGH',
        '0.5f': 'NIMCP_CONFIDENCE_MEDIUM',
        '0.3f': 'NIMCP_CONFIDENCE_LOW',
        '0.1f': 'NIMCP_CONFIDENCE_MIN',
    }),
    # Sensitivity
    (re.compile(r'(?:sensitivity)\b', re.I), {
        '1.0f': 'NIMCP_SENSITIVITY_DEFAULT',
    }),
]

FIELD_ASSIGN_RE = re.compile(
    r'(\.\s*(\w+)\s*=\s*)'
    r'([\d.]+f)'
    r'(\s*[;,])',
)

VAR_DECL_RE = re.compile(
    r'((?:float|double|_Atomic\s+float)\s+(\w+)\s*=\s*)'
    r'([\d.]+f)'
    r'(\s*;)',
)

DEFINE_RE = re.compile(
    r'(#define\s+(\w+)\s+)'
    r'([\d.]+f)'
    r'(\s*(?://.*|/\*.*)?$)',
    re.MULTILINE
)


def has_include(content):
    return (INCLUDE in content or
            '"constants/nimcp_constants.h"' in content or
            'nimcp_threshold_constants.h' in content)


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
