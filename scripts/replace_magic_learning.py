#!/usr/bin/env python3
"""Replace magic number learning constants (learning rates, epsilon, momentum, decay).

Focuses on safe patterns where context clearly indicates learning parameters:
- struct fields named *learning_rate*, *lr*, *epsilon*, *momentum*, *decay*
- Variable declarations/assignments with learning parameter names

Usage:
    python3 scripts/replace_magic_learning.py [--dry-run]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_learning_constants.h"'

SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_learning_constants.h', 'nimcp_constants.h',
}

# Field/var name patterns -> (value, constant) mappings
# Only replace when BOTH the name AND value match
RULES = [
    # Learning rates
    (re.compile(r'(?:learning_rate|lr)\b', re.I), {
        '0.01f':  'NIMCP_LEARNING_RATE_DEFAULT',
        '0.001f': 'NIMCP_LEARNING_RATE_FINE',
        '0.1f':   'NIMCP_LEARNING_RATE_COARSE',
        '0.0001f':'NIMCP_LEARNING_RATE_MICRO',
    }),
    # Epsilon / numerical stability
    (re.compile(r'(?:epsilon|eps)\b', re.I), {
        '1e-6f':  'NIMCP_EPSILON_NUMERICAL',
        '1e-8f':  'NIMCP_EPSILON_ADAM',
        '1e-4f':  'NIMCP_EPSILON_LARGE',
    }),
    # Momentum
    (re.compile(r'momentum\b', re.I), {
        '0.9f':  'NIMCP_MOMENTUM_DEFAULT',
        '0.99f': 'NIMCP_MOMENTUM_HIGH',
    }),
    # Decay
    (re.compile(r'(?:decay|ema_decay)\b', re.I), {
        '0.99f':  'NIMCP_EMA_DECAY_DEFAULT',
        '0.9f':   'NIMCP_EMA_DECAY_FAST',
        '0.999f': 'NIMCP_EMA_DECAY_SLOW',
        '0.95f':  'NIMCP_ELIGIBILITY_DECAY_DEFAULT',
        '0.0001f':'NIMCP_WEIGHT_DECAY_DEFAULT',
    }),
    # Weight decay
    (re.compile(r'weight_decay\b', re.I), {
        '0.0001f':'NIMCP_WEIGHT_DECAY_DEFAULT',
    }),
    # Gradient clip
    (re.compile(r'(?:gradient_clip|grad_clip|clip_norm)\b', re.I), {
        '1.0f':  'NIMCP_GRADIENT_CLIP_DEFAULT',
        '0.5f':  'NIMCP_GRADIENT_CLIP_TIGHT',
    }),
    # Beta1/Beta2 (Adam)
    (re.compile(r'beta1\b', re.I), {
        '0.9f': 'NIMCP_ADAM_BETA1_DEFAULT',
    }),
    (re.compile(r'beta2\b', re.I), {
        '0.999f': 'NIMCP_ADAM_BETA2_DEFAULT',
    }),
    # Discount factor
    (re.compile(r'(?:discount|gamma)\b', re.I), {
        '0.99f': 'NIMCP_REWARD_DISCOUNT_DEFAULT',
    }),
]

# Pattern for field assignment: .field_name = value
FIELD_ASSIGN_RE = re.compile(
    r'(\.\s*(\w+)\s*=\s*)'  # .field_name =
    r'([\d.eE+-]+f?)'        # numeric value
    r'(\s*[;,])',             # ; or ,
)

# Pattern for variable declaration: type var = value
VAR_DECL_RE = re.compile(
    r'((?:float|double|_Atomic\s+float|_Atomic\s+double)\s+(\w+)\s*=\s*)'
    r'([\d.eE+-]+f?)'
    r'(\s*;)',
)

# Pattern for #define: #define NAME value
DEFINE_RE = re.compile(
    r'(#define\s+(\w+)\s+)'
    r'([\d.eE+-]+f?)'
    r'(\s*(?://.*|/\*.*)?$)',
    re.MULTILINE
)


def has_learning_include(content):
    return (INCLUDE in content or
            '"constants/nimcp_constants.h"' in content or
            'nimcp_learning_constants.h' in content)


def add_include(content):
    lines = content.split('\n')
    in_block = False
    last_include_in_block = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#include'):
            in_block = True
            last_include_in_block = i
        elif in_block and stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*') and not stripped.startswith('#'):
            break

    if last_include_in_block == -1:
        return content

    lines.insert(last_include_in_block + 1, f'#include {INCLUDE}')
    return '\n'.join(lines)


def try_replace(name, value_str):
    """Check if name+value matches any rule and return the constant name."""
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

    original = content
    replacements = [0]

    def replace_field(m):
        prefix = m.group(1)
        name = m.group(2)
        value = m.group(3)
        suffix = m.group(4)
        const = try_replace(name, value)
        if const:
            replacements[0] += 1
            return f'{prefix}{const}{suffix}'
        return m.group(0)

    content = FIELD_ASSIGN_RE.sub(replace_field, content)
    content = VAR_DECL_RE.sub(replace_field, content)
    content = DEFINE_RE.sub(replace_field, content)

    if replacements[0] == 0:
        return 0

    if not has_learning_include(content):
        content = add_include(content)

    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return replacements[0]


def main():
    dry_run = '--dry-run' in sys.argv

    total_replacements = 0
    files_modified = 0
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
                print(f"  {action} ({count} replacements): {rel}")
                total_replacements += count
                files_modified += 1

    action = "Would modify" if dry_run else "Modified"
    print(f"\n{action} {files_modified} files with {total_replacements} total replacements (checked {files_checked} files)")


if __name__ == '__main__':
    main()
