#!/usr/bin/env python3
"""Add NIMCP_THROW_TO_IMMUNE calls before error returns in source files."""
import re, os, sys, glob

DESC_MAP = {
    'NIMCP_ERROR_NULL_POINTER': 'NULL pointer parameter',
    'NIMCP_ERROR_NO_MEMORY': 'memory allocation failed',
    'NIMCP_ERROR_INVALID_PARAM': 'invalid parameter',
    'NIMCP_ERROR_INVALID_STATE': 'invalid state',
    'NIMCP_ERROR_NOT_INITIALIZED': 'not initialized',
    'NIMCP_ERROR_OPERATION_FAILED': 'operation failed',
    'NIMCP_ERROR': 'operation failed',
}

def get_module_name(filepath):
    basename = os.path.basename(filepath)
    return basename.replace('nimcp_', '').replace('.c', '')

def process_file(filepath):
    module_name = get_module_name(filepath)
    with open(filepath, 'r') as f:
        lines = f.readlines()
    content = ''.join(lines)
    
    if 'NIMCP_THROW_TO_IMMUNE' in content:
        return False
    
    has_exception_include = 'nimcp_exception_macros.h' in content or 'nimcp_exception.h' in content
    modified = False
    new_lines = []
    
    # Add include if needed
    if not has_exception_include:
        last_nimcp_include = -1
        for idx, line in enumerate(lines):
            if line.startswith('#include') and 'nimcp_' in line:
                last_nimcp_include = idx
        for idx, line in enumerate(lines):
            new_lines.append(line)
            if idx == last_nimcp_include and last_nimcp_include >= 0:
                new_lines.append('#include "utils/exception/nimcp_exception_macros.h"\n')
                modified = True
        lines = new_lines
        new_lines = []
    
    # Add NIMCP_THROW_TO_IMMUNE before error returns
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        error_code = None
        
        m = re.match(r'\s*return\s+(NIMCP_ERROR_\w+)\s*;', stripped)
        if m:
            error_code = m.group(1)
        elif re.match(r'\s*return\s+NIMCP_ERROR\s*;', stripped):
            error_code = 'NIMCP_ERROR'
        
        if error_code:
            prev_text = ''.join(lines[max(0, i - 3):i])
            if 'NIMCP_THROW_TO_IMMUNE' not in prev_text and 'NIMCP_THROW' not in prev_text:
                desc = DESC_MAP.get(error_code, 'error condition')
                indent = re.match(r'^(\s*)', line).group(1)
                throw_line = f'{indent}NIMCP_THROW_TO_IMMUNE({error_code}, "{module_name}: {desc}");\n'
                new_lines.append(throw_line)
                new_lines.append(line)
                modified = True
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    if modified:
        with open(filepath, 'w') as f:
            f.writelines(new_lines)
        return True
    return False

def main():
    base = '/home/bbrelin/nimcp'
    total_modified = 0
    directories = ['src/mesh', 'src/security', 'src/cognitive', 'src/core']
    
    for d in directories:
        dir_path = os.path.join(base, d)
        files = sorted(glob.glob(os.path.join(dir_path, '**', '*.c'), recursive=True))
        files = [f for f in files if 'CMakeFiles' not in f and 'venv' not in f]
        count = sum(1 for f in files if process_file(f))
        total_modified += count
        print(f'{d}: {count}/{len(files)} modified')
    
    print(f'\nTotal modified: {total_modified}')

if __name__ == '__main__':
    main()
