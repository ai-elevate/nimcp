#!/usr/bin/env python3
"""Replace magic number buffer sizes in char array declarations with named constants.

Focuses on the safest replacements: char xxx[N] where N is a known buffer size.
Uses variable name heuristics to choose the right constant.

Usage:
    python3 scripts/replace_magic_buffers.py [--dry-run] [--dir DIR]
"""

import os
import re
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE = '"constants/nimcp_buffer_constants.h"'

# Files that should never be modified
SKIP_FILES = {
    'nimcp_memory.c', 'nimcp_unified_memory.c', 'nimcp_constant_time.c',
    'nimcp_buffer_constants.h', 'nimcp_constants.h',
}

# Variable name patterns for choosing the right constant
PATH_NAMES = re.compile(r'(path|dir|file|fname|filename|filepath|mount|root_dir|backup|snapshot|wal_dir|base_dir|tmp_|dest|source_path|config_file|log_file|sock)', re.IGNORECASE)
ERROR_NAMES = re.compile(r'(err|error|msg_buf|detail|reason|message|desc|what|status_msg|diag)', re.IGNORECASE)
LOG_NAMES = re.compile(r'(log|debug|trace|info_buf|warn_buf)', re.IGNORECASE)
NAME_NAMES = re.compile(r'(name|label|key|tag|id|ident|module_name|type_name|func_name|topic)', re.IGNORECASE)
CMD_NAMES = re.compile(r'(cmd|command|query|sql|stmt|expr)', re.IGNORECASE)
JSON_NAMES = re.compile(r'(json|xml|html|payload|body|content|data_buf)', re.IGNORECASE)

# Size -> constant mapping, with name-based overrides
SIZE_DEFAULTS = {
    64:   'NIMCP_ID_BUFFER_SIZE',
    128:  'NIMCP_LABEL_BUFFER_SIZE',
    256:  'NIMCP_ERROR_BUFFER_SIZE',
    512:  'NIMCP_ERROR_BUFFER_LARGE',
    1024: 'NIMCP_LOG_BUFFER_SIZE',
    2048: 'NIMCP_JSON_BUFFER_SIZE',
    4096: 'NIMCP_PATH_BUFFER_SIZE',
}

# Name-based overrides for specific sizes
SIZE_OVERRIDES = {
    64: [
        (ERROR_NAMES, 'NIMCP_ERROR_BUFFER_SMALL'),
        (NAME_NAMES,  'NIMCP_ID_BUFFER_SIZE'),
    ],
    128: [
        (ERROR_NAMES, 'NIMCP_ERROR_BUFFER_MEDIUM'),
        (NAME_NAMES,  'NIMCP_LABEL_BUFFER_SIZE'),
    ],
    256: [
        (PATH_NAMES,  'NIMCP_SHORT_PATH_SIZE'),
        (ERROR_NAMES, 'NIMCP_ERROR_BUFFER_SIZE'),
        (NAME_NAMES,  'NIMCP_NAME_BUFFER_SIZE'),
    ],
    512: [
        (PATH_NAMES,  'NIMCP_METRICS_PATH_SIZE'),
        (ERROR_NAMES, 'NIMCP_ERROR_BUFFER_LARGE'),
        (CMD_NAMES,   'NIMCP_SHELL_CMD_SIZE'),
    ],
    1024: [
        (PATH_NAMES,  'NIMCP_DWARF_PATH_SIZE'),
        (CMD_NAMES,   'NIMCP_CMD_BUFFER_SIZE'),
        (LOG_NAMES,   'NIMCP_LOG_BUFFER_SIZE'),
        (JSON_NAMES,  'NIMCP_LOG_BUFFER_SIZE'),
    ],
    2048: [
        (LOG_NAMES,   'NIMCP_DEBUG_BUFFER_SIZE'),
        (JSON_NAMES,  'NIMCP_JSON_BUFFER_SIZE'),
    ],
    4096: [
        (PATH_NAMES,  'NIMCP_PATH_BUFFER_SIZE'),
        (CMD_NAMES,   'NIMCP_LINE_BUFFER_SIZE'),
    ],
}

# Regex for char array declaration with magic number size
# Matches: char varname[N], char varname [N], char *varname[N] (not pointers)
CHAR_ARRAY_RE = re.compile(
    r'(\bchar\s+)'              # 'char '
    r'(\w+)'                    # variable name
    r'(\s*\[)\s*'              # '['
    r'(\d+)'                    # the magic number
    r'(\s*\])'                  # ']'
)

# Also match 'static char', 'const char', etc.
QUALIFIED_CHAR_ARRAY_RE = re.compile(
    r'((?:static\s+|const\s+|volatile\s+|_Thread_local\s+)*char\s+)'
    r'(\w+)'                    # variable name
    r'(\s*\[)\s*'              # '['
    r'(\d+)'                    # the magic number
    r'(\s*\])'                  # ']'
)

# snprintf/vsnprintf sizeof replacement:
# snprintf(buf, 256, ...) -> snprintf(buf, sizeof(buf), ...) is a different refactoring
# We focus only on declarations here.

def choose_constant(var_name, size):
    """Choose the best constant name based on variable name and size."""
    size = int(size)
    if size not in SIZE_DEFAULTS:
        return None

    # Check name-based overrides
    overrides = SIZE_OVERRIDES.get(size, [])
    for pattern, const_name in overrides:
        if pattern.search(var_name):
            return const_name

    # Fall back to default for this size
    return SIZE_DEFAULTS[size]


def has_buffer_include(content):
    """Check if file already includes buffer constants."""
    return (INCLUDE in content or
            '"constants/nimcp_constants.h"' in content or
            'nimcp_buffer_constants.h' in content)


def add_include(content):
    """Add buffer constants include after the first contiguous block of includes."""
    lines = content.split('\n')
    # Find the end of the first contiguous #include block (allowing blank lines and comments)
    in_include_block = False
    last_include_in_block = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('#include'):
            in_include_block = True
            last_include_in_block = i
        elif in_include_block and stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*') and not stripped.startswith('#'):
            # Non-include, non-comment, non-empty line ends the block
            break

    if last_include_in_block == -1:
        return content  # No includes found, skip

    # Insert after last include in the first block
    lines.insert(last_include_in_block + 1, f'#include {INCLUDE}')
    return '\n'.join(lines)


def process_file(filepath, dry_run=False):
    """Process a single file, replacing magic buffer sizes."""
    basename = os.path.basename(filepath)
    if basename in SKIP_FILES:
        return 0

    with open(filepath) as f:
        content = f.read()

    original = content
    replacements = 0
    needs_include = False

    def replace_match(m):
        nonlocal replacements, needs_include
        prefix = m.group(1)
        var_name = m.group(2)
        bracket_open = m.group(3)
        size_str = m.group(4)
        bracket_close = m.group(5)

        const = choose_constant(var_name, size_str)
        if const is None:
            return m.group(0)  # No replacement

        # Don't replace if it's already a constant reference
        if not size_str.isdigit():
            return m.group(0)

        replacements += 1
        needs_include = True
        return f'{prefix}{var_name}{bracket_open}{const}{bracket_close}'

    content = QUALIFIED_CHAR_ARRAY_RE.sub(replace_match, content)

    if replacements == 0:
        return 0

    # Add include if needed
    if needs_include and not has_buffer_include(content):
        content = add_include(content)

    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return replacements


def main():
    dry_run = '--dry-run' in sys.argv
    target_dir = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--dir' and i + 1 < len(sys.argv):
            target_dir = sys.argv[i + 1]

    if target_dir:
        search_dir = os.path.join(ROOT, target_dir)
    else:
        search_dir = os.path.join(ROOT, 'src')

    # Find all .c files that DON'T already have the include
    total_replacements = 0
    files_modified = 0
    files_checked = 0

    for root, dirs, files in os.walk(search_dir):
        # Skip build directories
        dirs[:] = [d for d in dirs if d not in ('build', '.git', '__pycache__')]
        for fname in sorted(files):
            if not fname.endswith('.c'):
                continue
            filepath = os.path.join(root, fname)
            files_checked += 1
            count = process_file(filepath, dry_run)
            if count > 0:
                rel = os.path.relpath(filepath, ROOT)
                if dry_run:
                    print(f"  WOULD FIX ({count} replacements): {rel}")
                else:
                    print(f"  FIXED ({count} replacements): {rel}")
                total_replacements += count
                files_modified += 1

    action = "Would modify" if dry_run else "Modified"
    print(f"\n{action} {files_modified} files with {total_replacements} total replacements (checked {files_checked} files)")


if __name__ == '__main__':
    main()
