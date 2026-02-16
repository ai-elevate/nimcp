#!/usr/bin/env python3
"""Replace magic number frequency constants with named constants.

Focuses on neural oscillation bands, update rates, and audio frequencies.

Usage:
    python3 scripts/replace_magic_frequency.py [--dry-run]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_frequency_constants.h"'

SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_frequency_constants.h', 'nimcp_constants.h',
}

RULES = [
    # Oscillation band boundaries
    (re.compile(r'(?:delta_low|delta_min)\b', re.I), {'0.5f': 'NIMCP_DELTA_LOW_HZ'}),
    (re.compile(r'(?:delta_high|delta_max)\b', re.I), {'4.0f': 'NIMCP_DELTA_HIGH_HZ'}),
    (re.compile(r'(?:delta_center|delta_freq)\b', re.I), {'2.0f': 'NIMCP_DELTA_CENTER_HZ'}),
    (re.compile(r'(?:theta_low|theta_min)\b', re.I), {'4.0f': 'NIMCP_THETA_LOW_HZ'}),
    (re.compile(r'(?:theta_high|theta_max)\b', re.I), {'8.0f': 'NIMCP_THETA_HIGH_HZ'}),
    (re.compile(r'(?:theta_center|theta_freq)\b', re.I), {'6.0f': 'NIMCP_THETA_CENTER_HZ'}),
    (re.compile(r'(?:alpha_low|alpha_min)\b', re.I), {'8.0f': 'NIMCP_ALPHA_LOW_HZ'}),
    (re.compile(r'(?:alpha_high|alpha_max)\b', re.I), {'12.0f': 'NIMCP_ALPHA_HIGH_HZ'}),
    (re.compile(r'(?:alpha_center|alpha_freq)\b', re.I), {'10.0f': 'NIMCP_ALPHA_CENTER_HZ'}),
    (re.compile(r'(?:beta_low|beta_min)\b', re.I), {'12.0f': 'NIMCP_BETA_LOW_HZ'}),
    (re.compile(r'(?:beta_high|beta_max)\b', re.I), {'30.0f': 'NIMCP_BETA_HIGH_HZ'}),
    (re.compile(r'(?:beta_center|beta_freq)\b', re.I), {'20.0f': 'NIMCP_BETA_CENTER_HZ'}),
    (re.compile(r'(?:gamma_low|gamma_min)\b', re.I), {'30.0f': 'NIMCP_GAMMA_LOW_HZ'}),
    (re.compile(r'(?:gamma_high|gamma_max)\b', re.I), {'100.0f': 'NIMCP_GAMMA_HIGH_HZ'}),
    (re.compile(r'(?:gamma_center|gamma_freq|gamma_binding)\b', re.I), {'40.0f': 'NIMCP_GAMMA_CENTER_HZ'}),
    # Update rates
    (re.compile(r'(?:update_rate|tick_rate)\b', re.I), {
        '30.0f': 'NIMCP_DEFAULT_UPDATE_RATE_HZ',
        '60.0f': 'NIMCP_FAST_UPDATE_RATE_HZ',
        '100.0f': 'NIMCP_HIGH_UPDATE_RATE_HZ',
    }),
    # Sample rates for oscillation processing
    (re.compile(r'(?:sample_rate|sampling_rate)\b', re.I), {
        '1000.0f': 'NIMCP_OSC_SAMPLE_RATE_HZ',
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
            'nimcp_frequency_constants.h' in content)


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
