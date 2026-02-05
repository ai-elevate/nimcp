#!/usr/bin/env python3
"""Fix infinite recursion in memory implementation.

The P2-1 migration converted calloc/malloc/free to nimcp_calloc/nimcp_malloc/nimcp_free
inside the memory implementation files themselves, causing infinite recursion:
  nimcp_calloc -> unified_mem_alloc -> nimcp_calloc -> ... -> stack overflow

Fix: Revert these back to raw C stdlib calls inside the implementation files.
"""

import re

ROOT = "/home/bbrelin/nimcp"

# Files that implement the memory API - must use raw stdlib
files = [
    "src/utils/memory/nimcp_unified_memory.c",
    "src/utils/memory/nimcp_memory.c",
]

for rel_path in files:
    path = f"{ROOT}/{rel_path}"
    with open(path, 'r', errors='replace') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    changed = False

    for line in lines:
        stripped = line.strip()

        # Skip function DEFINITION lines (keep nimcp_ prefix for the API functions)
        # These look like: void* nimcp_malloc(size_t size) {
        # or: void nimcp_free(void* ptr) {
        if re.match(r'^(static\s+)?(void\s*\*?|size_t|int|nimcp_error_t|bool)\s+nimcp_(malloc|calloc|free|realloc)\b', stripped):
            new_lines.append(line)
            continue

        # Skip lines that are #define or declarations of nimcp_malloc etc.
        if re.match(r'^(#define|extern|typedef)\s+.*nimcp_(malloc|calloc|free|realloc)', stripped):
            new_lines.append(line)
            continue

        new_line = line
        # Replace CALLS to nimcp_calloc/malloc/free/realloc with raw versions
        new_line = re.sub(r'\bnimcp_calloc\s*\(', 'calloc(', new_line)
        new_line = re.sub(r'\bnimcp_malloc\s*\(', 'malloc(', new_line)
        new_line = re.sub(r'\bnimcp_free\s*\(', 'free(', new_line)
        new_line = re.sub(r'\bnimcp_realloc\s*\(', 'realloc(', new_line)

        if new_line != line:
            changed = True
        new_lines.append(new_line)

    if changed:
        with open(path, 'w') as f:
            f.write('\n'.join(new_lines))
        print(f"  Fixed: {rel_path}")
    else:
        print(f"  No changes: {rel_path}")

# Also check nimcp_constant_time.c which uses nimcp_calloc during __attribute__((constructor))
ct_file = f"{ROOT}/src/security/nimcp_constant_time.c"
with open(ct_file, 'r', errors='replace') as f:
    content = f.read()

# The constructor function s_ct_module_init calls nimcp_ct_create which uses nimcp_calloc
# This is called before main() when the library loads, so must use raw calloc
lines = content.split('\n')
new_lines = []
changed = False
for line in lines:
    new_line = line
    new_line = re.sub(r'\bnimcp_calloc\s*\(', 'calloc(', new_line)
    new_line = re.sub(r'\bnimcp_malloc\s*\(', 'malloc(', new_line)
    new_line = re.sub(r'\bnimcp_free\s*\(', 'free(', new_line)
    new_line = re.sub(r'\bnimcp_realloc\s*\(', 'realloc(', new_line)
    if new_line != line:
        changed = True
    new_lines.append(new_line)

if changed:
    with open(ct_file, 'w') as f:
        f.write('\n'.join(new_lines))
    print(f"  Fixed: src/security/nimcp_constant_time.c")

print("\nDone.")
