#!/usr/bin/env python3
"""
Automated NIMCP Mutex Policy Enforcement
Replaces ALL raw pthread_mutex_* usage with nimcp_mutex_* equivalents
across the entire source tree (src/ and include/).

Excludes:
  - include/utils/thread/nimcp_thread.h (the abstraction layer)
  - include/utils/platform/nimcp_platform_mutex.h (platform impl)
  - src/utils/thread/nimcp_thread.c (abstraction impl)
  - src/utils/platform/ (platform implementations)
  - test/ directories
  - Files in utils/thread/ that are part of the threading infrastructure
"""

import os
import re
import sys

# Patterns for mutex conversion
PATTERNS = [
    # Type replacement
    (re.compile(r'\bpthread_mutex_t\b'), 'nimcp_mutex_t'),
    # Function replacements
    (re.compile(r'\bpthread_mutex_init\b'), 'nimcp_mutex_init'),
    (re.compile(r'\bpthread_mutex_lock\b'), 'nimcp_mutex_lock'),
    (re.compile(r'\bpthread_mutex_unlock\b'), 'nimcp_mutex_unlock'),
    (re.compile(r'\bpthread_mutex_destroy\b'), 'nimcp_mutex_destroy'),
    (re.compile(r'\bpthread_mutex_trylock\b'), 'nimcp_mutex_trylock'),
    # Initializer
    (re.compile(r'\bPTHREAD_MUTEX_INITIALIZER\b'), 'NIMCP_MUTEX_INITIALIZER'),
]

# Include to add
THREAD_INCLUDE = '#include "utils/thread/nimcp_thread.h"'

# Paths to exclude completely
EXCLUDED_PATHS = [
    'include/utils/thread/nimcp_thread.h',
    'include/utils/platform/',
    'src/utils/thread/nimcp_thread.c',
    'src/utils/platform/',
    'venv/',
    'test/',
]

# Other pthread features that require <pthread.h> to remain
OTHER_PTHREAD_FEATURES = [
    'pthread_create', 'pthread_join', 'pthread_detach',
    'pthread_cond_t', 'pthread_cond_init', 'pthread_cond_wait',
    'pthread_cond_signal', 'pthread_cond_broadcast', 'pthread_cond_destroy',
    'pthread_cond_timedwait',
    'pthread_rwlock_t', 'pthread_rwlock_init', 'pthread_rwlock_rdlock',
    'pthread_rwlock_wrlock', 'pthread_rwlock_unlock', 'pthread_rwlock_destroy',
    'pthread_key_t', 'pthread_key_create', 'pthread_setspecific',
    'pthread_getspecific',
    'pthread_attr_t', 'pthread_attr_init', 'pthread_attr_destroy',
    'pthread_attr_setdetachstate',
    'pthread_self', 'pthread_equal',
    'pthread_once_t', 'pthread_once',
    'pthread_t',
    'pthread_barrier_t', 'pthread_barrier_init', 'pthread_barrier_wait',
    'pthread_barrier_destroy',
    'pthread_spin_t', 'pthread_spin_init', 'pthread_spin_lock',
    'pthread_spin_unlock', 'pthread_spin_destroy',
]


def should_exclude(filepath):
    for ex in EXCLUDED_PATHS:
        if ex in filepath:
            return True
    return False


def uses_other_pthread(content):
    """Check if content uses other pthread features besides mutex."""
    for feat in OTHER_PTHREAD_FEATURES:
        if re.search(r'\b' + re.escape(feat) + r'\b', content):
            return True
    return False


def is_in_comment(content, match_start):
    """Check if the match position is inside a comment (crude check)."""
    # Check if we're inside a // comment on same line
    line_start = content.rfind('\n', 0, match_start) + 1
    line_prefix = content[line_start:match_start]
    if '//' in line_prefix:
        return True
    return False


def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
    except Exception as e:
        print(f"  [ERROR] Cannot read {filepath}: {e}")
        return False, False

    original_content = content

    # Apply substitutions (including in comments - it's fine to update comments too)
    for pattern, replacement in PATTERNS:
        content = pattern.sub(replacement, content)

    if content == original_content:
        return False, False

    # Add nimcp_thread.h include if not already present
    added_include = False
    if THREAD_INCLUDE not in content and 'nimcp_thread.h' not in content:
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                insert_idx = i + 1
        if insert_idx > 0:
            lines.insert(insert_idx, THREAD_INCLUDE)
            content = '\n'.join(lines)
            added_include = True

    # Check if we can remove <pthread.h>
    # Only remove if no other pthread features are used
    if not uses_other_pthread(content):
        content = re.sub(r'#include\s*<pthread\.h>\s*\n', '', content)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"  [ERROR] Cannot write {filepath}: {e}")
        return False, False

    return True, added_include


def main():
    print("=" * 79)
    print("NIMCP Mutex Policy Enforcement - pthread_mutex -> nimcp_mutex Conversion")
    print("=" * 79)

    root_dirs = ['src/', 'include/']
    fixed = 0
    includes_added = 0
    total_scanned = 0

    for root_dir in root_dirs:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fname in sorted(filenames):
                if not (fname.endswith('.c') or fname.endswith('.h')):
                    continue
                fpath = os.path.join(dirpath, fname)
                if should_exclude(fpath):
                    continue
                total_scanned += 1
                changed, inc_added = fix_file(fpath)
                if changed:
                    fixed += 1
                    status = "[FIXED]"
                    if inc_added:
                        status += " [+include]"
                        includes_added += 1
                    print(f"  {status} {fpath}")

    print()
    print("=" * 79)
    print(f"Summary:")
    print(f"  Scanned: {total_scanned} files")
    print(f"  Fixed:   {fixed} files")
    print(f"  Includes added: {includes_added}")
    print("=" * 79)
    return 0


if __name__ == '__main__':
    sys.exit(main())
