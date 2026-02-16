#!/usr/bin/env python3
"""
SRP Refactoring Helper - Split P1 files by Single Responsibility Principle
Analyzes source files and generates split module templates
"""

import re
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

@dataclass
class FunctionInfo:
    """Information about a function in the source file"""
    name: str
    start_line: int
    end_line: int
    signature: str
    is_static: bool
    is_public: bool
    concern: str  # Which concern this function belongs to

def find_functions(filepath: Path) -> List[FunctionInfo]:
    """Parse C file and extract function boundaries"""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    functions = []
    in_function = False
    brace_count = 0
    func_start = 0
    func_signature = ""

    for i, line in enumerate(lines, 1):
        # Skip comments and preprocessor directives
        if line.strip().startswith('//') or line.strip().startswith('#') or line.strip().startswith('/*'):
            continue

        # Look for function definitions (simplified heuristic)
        if not in_function and re.match(r'^[a-zA-Z_].*\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(', line):
            func_start = i
            func_signature = line.strip()
            in_function = True
            brace_count = 0

        if in_function:
            brace_count += line.count('{') - line.count('}')

            if brace_count == 0 and '{' in ''.join(lines[func_start-1:i]):
                # Function complete
                is_static = 'static' in func_signature

                # Extract function name
                match = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', func_signature)
                func_name = match.group(1) if match else "unknown"

                functions.append(FunctionInfo(
                    name=func_name,
                    start_line=func_start,
                    end_line=i,
                    signature=func_signature,
                    is_static=is_static,
                    is_public=not is_static,
                    concern=""  # Will be assigned based on name patterns
                ))

                in_function = False

    return functions

def categorize_neuromodulator_functions(functions: List[FunctionInfo]) -> Dict[str, List[FunctionInfo]]:
    """Categorize neuromodulator functions by concern"""
    categories = {
        'dopamine': [],
        'serotonin': [],
        'norepinephrine': [],
        'acetylcholine': [],
        'core': []  # lifecycle, pool, stats, bio-async
    }

    for func in functions:
        name_lower = func.name.lower()

        if 'dopamine' in name_lower:
            func.concern = 'dopamine'
            categories['dopamine'].append(func)
        elif 'serotonin' in name_lower:
            func.concern = 'serotonin'
            categories['serotonin'].append(func)
        elif 'norepinephrine' in name_lower or 'noradrenaline' in name_lower:
            func.concern = 'norepinephrine'
            categories['norepinephrine'].append(func)
        elif 'acetylcholine' in name_lower or 'ach' in name_lower:
            func.concern = 'acetylcholine'
            categories['acetylcholine'].append(func)
        else:
            # Core functions: create, destroy, get_stats, pool functions, bio-async
            func.concern = 'core'
            categories['core'].append(func)

    return categories

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_srp_refactor.py <file_to_analyze>")
        sys.exit(1)

    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    print(f"Analyzing: {filepath}")
    print("=" * 80)

    functions = find_functions(filepath)
    print(f"Found {len(functions)} functions\n")

    # For neuromodulators, categorize by system
    if 'neuromodulators' in filepath.name:
        categories = categorize_neuromodulator_functions(functions)

        for concern, funcs in categories.items():
            print(f"\n{concern.upper()} ({len(funcs)} functions):")
            print("-" * 40)
            for f in funcs:
                visibility = "static" if f.is_static else "public"
                print(f"  {f.name:50s} (line {f.start_line:4d}-{f.end_line:4d}) [{visibility}]")
    else:
        # Just list all functions for other files
        for f in functions:
            visibility = "static" if f.is_static else "public"
            print(f"{f.name:50s} (line {f.start_line:4d}-{f.end_line:4d}) [{visibility}]")

if __name__ == "__main__":
    main()
