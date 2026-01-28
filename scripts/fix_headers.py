#!/usr/bin/env python3
"""Fix headers that use nimcp_health_agent_t without declaring it."""
import os
import re

INCLUDE_ROOT = "/home/bbrelin/nimcp/include"
FORWARD_DECL = "\n/* Phase 8: Forward declaration for health agent */\ntypedef struct nimcp_health_agent nimcp_health_agent_t;\n"

def find_headers_needing_fix():
    """Find headers with nimcp_health_agent_t but no declaration/include."""
    headers = []
    for root, dirs, files in os.walk(INCLUDE_ROOT):
        for f in files:
            if not f.endswith('.h'):
                continue
            path = os.path.join(root, f)
            with open(path) as fh:
                content = fh.read()
            # Has the type but doesn't define or forward-declare it
            if 'nimcp_health_agent_t' in content:
                if 'typedef struct nimcp_health_agent nimcp_health_agent_t' not in content:
                    if 'nimcp_health_agent.h' not in content:
                        headers.append(path)
    return sorted(headers)

def fix_header(path):
    with open(path) as f:
        content = f.read()

    # Find the struct that uses nimcp_health_agent_t
    # Insert forward declaration before it
    # Look for first occurrence of nimcp_health_agent_t
    idx = content.find('nimcp_health_agent_t')
    if idx < 0:
        return

    # Find the beginning of the struct block containing the first use
    # Go backwards to find the struct keyword or typedef
    # Actually, just insert after the last #include or after header guard
    # Find last #include line
    last_include = -1
    for m in re.finditer(r'^#include\s+.*$', content, re.MULTILINE):
        last_include = m.end()

    if last_include > 0 and last_include < idx:
        insert_pos = last_include
    else:
        # Find after header guard
        m = re.search(r'#define\s+\w+_H\w*\s*\n', content)
        if m:
            insert_pos = m.end()
        else:
            # Insert before first use
            # Find start of line containing first use
            line_start = content.rfind('\n', 0, idx)
            # Go back further to find struct/typedef start
            block_start = content.rfind('struct', 0, idx)
            typedef_start = content.rfind('typedef', 0, idx)
            insert_pos = max(line_start, block_start, typedef_start)
            if insert_pos < 0:
                insert_pos = idx

    content = content[:insert_pos] + FORWARD_DECL + content[insert_pos:]
    with open(path, 'w') as f:
        f.write(content)
    print(f"  FIXED: {os.path.relpath(path, '/home/bbrelin/nimcp')}")

def main():
    headers = find_headers_needing_fix()
    print(f"Found {len(headers)} headers needing nimcp_health_agent_t forward declaration")
    for h in headers:
        fix_header(h)

if __name__ == '__main__':
    main()
