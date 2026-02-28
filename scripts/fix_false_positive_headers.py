#!/usr/bin/env python3
"""
Fix false-positive header const removals by parsing build errors.

Each build error pair tells us:
  error line: src_file:N: error: conflicting types for 'func'; have '...(const type_t *)...'
  note line:  header_file:N: note: previous declaration of 'func' with type '...(type_t *)...'

We need to add 'const' before the type in the header declaration.
"""

import re
import subprocess
import sys

BUILD_DIR = "/home/bbrelin/nimcp/build"

def get_error_pairs():
    """Run make -k and parse error/note pairs."""
    # First run cmake to make sure build files are current
    subprocess.run(["cmake", ".."], capture_output=True, text=True, cwd=BUILD_DIR, timeout=60)

    result = subprocess.run(
        ["make", "nimcp", "-j4", "-k"],
        capture_output=True, text=True, cwd=BUILD_DIR, timeout=300
    )
    output = result.stdout + '\n' + result.stderr
    lines = output.split('\n')

    pairs = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "error: conflicting types for" in line:
            # Extract function name
            func_m = re.search(r"conflicting types for '(\w+)'", line)
            if not func_m:
                i += 1
                continue
            func_name = func_m.group(1)

            # Extract the const type from the "have" part
            type_m = re.search(r"const (\w+_t) \*", line)
            if not type_m:
                i += 1
                continue
            const_type = type_m.group(1)

            # Find the corresponding note line (usually within next 3 lines)
            found = False
            for j in range(i+1, min(i+6, len(lines))):
                if "previous declaration" in lines[j] and func_name in lines[j]:
                    header_m = re.match(r'(\S+\.h):(\d+)', lines[j])
                    if header_m:
                        header_file = header_m.group(1)
                        header_line = int(header_m.group(2))
                        pairs.append((func_name, const_type, header_file, header_line))
                        found = True
                    break

            if not found:
                print(f"  WARNING: No note line found for {func_name}")
        i += 1

    return pairs

def fix_header(header_file, header_line, func_name, const_type):
    """Add const to the specific line in the header file."""
    try:
        with open(header_file, 'r', errors='replace') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ERROR reading {header_file}: {e}")
        return False

    # header_line is 1-indexed
    idx = header_line - 1
    if idx < 0 or idx >= len(lines):
        print(f"  ERROR: line {header_line} out of range in {header_file}")
        return False

    # Search around the target line for the type name
    target_idx = None
    for check_idx in range(max(0, idx-2), min(len(lines), idx+5)):
        if const_type in lines[check_idx] and 'const ' + const_type not in lines[check_idx]:
            target_idx = check_idx
            break

    if target_idx is None:
        # Maybe the type is on the same line as the function name
        if const_type in lines[idx] and 'const ' + const_type not in lines[idx]:
            target_idx = idx
        else:
            print(f"  SKIP: {const_type} not found near line {header_line} in {header_file}")
            return False

    line = lines[target_idx]

    # Don't fix if already has const
    if 'const ' + const_type in line or 'const\t' + const_type in line:
        print(f"  SKIP (already const): {func_name} in {header_file}:{header_line}")
        return False

    # Add const before the type name using regex for precision
    new_line = re.sub(
        r'\b' + re.escape(const_type) + r'(\s*\*)',
        'const ' + const_type + r'\1',
        line,
        count=1
    )

    if new_line == line:
        print(f"  SKIP (regex didn't match): {func_name} in {header_file}:{header_line}")
        return False

    lines[target_idx] = new_line
    with open(header_file, 'w') as f:
        f.writelines(lines)
    return True


def main():
    iteration = 0
    max_iterations = 5  # Safety limit
    total_all = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n=== Iteration {iteration} ===")
        print("Building to find conflicting type errors...")
        pairs = get_error_pairs()

        if not pairs:
            print("No conflicting type errors found - build is clean!")
            break

        print(f"Found {len(pairs)} errors to fix:")
        fixes = 0
        for func_name, const_type, header_file, header_line in pairs:
            if fix_header(header_file, header_line, func_name, const_type):
                print(f"  FIXED: {func_name} ({const_type}) in {header_file}:{header_line}")
                fixes += 1

        total_all += fixes

        if fixes == 0:
            print("No fixes could be applied - manual intervention needed")
            for func_name, const_type, header_file, header_line in pairs:
                print(f"  REMAINING: {func_name} ({const_type}) in {header_file}:{header_line}")
            break

        print(f"Fixed {fixes}/{len(pairs)} errors. Rebuilding...")

    print(f"\nCompleted after {iteration} iteration(s), {total_all} total fixes")


if __name__ == '__main__':
    main()
