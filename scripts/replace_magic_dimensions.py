#!/usr/bin/env python3
"""Replace magic number dimension constants with named constants.

Focuses on grid dimensions, embedding sizes, batch sizes, hidden layer sizes.

Usage:
    python3 scripts/replace_magic_dimensions.py [--dry-run]
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_dimension_constants.h"'

SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_dimension_constants.h', 'nimcp_constants.h',
}

RULES = [
    # Embedding dimensions
    (re.compile(r'(?:embedding_dim|embed_dim|embedding_size)\b', re.I), {
        '256': 'NIMCP_DEFAULT_EMBEDDING_DIM',
        '64':  'NIMCP_SMALL_EMBEDDING_DIM',
        '128': 'NIMCP_MEDIUM_EMBEDDING_DIM',
        '512': 'NIMCP_LARGE_EMBEDDING_DIM',
    }),
    # Hidden layer sizes
    (re.compile(r'(?:hidden_size|hidden_dim|hidden_units)\b', re.I), {
        '256': 'NIMCP_DEFAULT_HIDDEN_SIZE',
        '64':  'NIMCP_SMALL_HIDDEN_SIZE',
        '128': 'NIMCP_MEDIUM_HIDDEN_SIZE',
        '512': 'NIMCP_LARGE_HIDDEN_SIZE',
    }),
    # Grid dimensions
    (re.compile(r'(?:grid_dim|grid_size|grid_width|grid_height)\b', re.I), {
        '64':  'NIMCP_DEFAULT_GRID_DIM',
        '32':  'NIMCP_SMALL_GRID_DIM',
        '128': 'NIMCP_LARGE_GRID_DIM',
    }),
    # Batch sizes
    (re.compile(r'(?:batch_size|batch_sz)\b', re.I), {
        '32':  'NIMCP_DEFAULT_BATCH_SIZE',
        '8':   'NIMCP_SMALL_BATCH_SIZE',
        '64':  'NIMCP_MEDIUM_BATCH_SIZE',
        '128': 'NIMCP_LARGE_BATCH_SIZE',
        '256': 'NIMCP_MAX_BATCH_SIZE',
    }),
    # Working memory slots
    (re.compile(r'(?:wm_slots|working_memory_slots|num_slots)\b', re.I), {
        '7': 'NIMCP_WM_SLOTS',
    }),
    # Attention heads
    (re.compile(r'(?:num_heads|attention_heads|n_heads)\b', re.I), {
        '8': 'NIMCP_DEFAULT_ATTENTION_HEADS',
    }),
    # Thread pool
    (re.compile(r'(?:thread_pool_size|pool_size|num_threads|num_workers)\b', re.I), {
        '4': 'NIMCP_DEFAULT_THREAD_POOL_SIZE',
    }),
    # Feature dimension
    (re.compile(r'(?:feature_dim|feature_size|num_features)\b', re.I), {
        '128': 'NIMCP_FEATURE_DIM',
    }),
    # State dimension
    (re.compile(r'(?:state_dim|state_size)\b', re.I), {
        '64': 'NIMCP_STATE_DIM',
    }),
    # Hypercolumns
    (re.compile(r'(?:num_hypercolumns|hypercolumn_count|n_hypercolumns)\b', re.I), {
        '64':  'NIMCP_DEFAULT_HYPERCOLUMNS',
        '32':  'NIMCP_SMALL_HYPERCOLUMNS',
        '128': 'NIMCP_LARGE_HYPERCOLUMNS',
    }),
    # Swarm agents
    (re.compile(r'(?:num_agents|agent_count|swarm_size)\b', re.I), {
        '32': 'NIMCP_DEFAULT_SWARM_AGENTS',
    }),
    # NOTE: max_iterations (NIMCP_MAX_ITERATIONS_DEFAULT) is in nimcp_learning_constants.h
    # Proof depth
    (re.compile(r'(?:proof_depth|search_depth)\b', re.I), {
        '10': 'NIMCP_DEFAULT_PROOF_DEPTH',
    }),
]

FIELD_ASSIGN_RE = re.compile(
    r'(\.\s*(\w+)\s*=\s*)'
    r'(\d+)'
    r'(\s*[;,])',
)

VAR_DECL_RE = re.compile(
    r'((?:int|uint\d+_t|unsigned|size_t|long)\s+(\w+)\s*=\s*)'
    r'(\d+)'
    r'(\s*;)',
)

DEFINE_RE = re.compile(
    r'(#define\s+(\w+)\s+)'
    r'(\d+)'
    r'(\s*(?://.*|/\*.*)?$)',
    re.MULTILINE
)


def has_include(content):
    return (INCLUDE in content or
            '"constants/nimcp_constants.h"' in content or
            'nimcp_dimension_constants.h' in content)


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
