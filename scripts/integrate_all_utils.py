#!/usr/bin/env python3

import os
import re
import sys
from pathlib import Path

# Module name mapping based on directory
MODULE_MAP = {
    'algorithms': 'ALGORITHMS',
    'cache': 'CACHE',
    'config': 'CONFIG',
    'containers': 'CONTAINERS',
    'error': 'ERROR',
    'geometry': 'GEOMETRY',
    'json': 'JSON',
    'math': 'MATH',
    'metrics': 'METRICS',
    'numerical': 'NUMERICAL',
    'platform': 'PLATFORM',
    'quantum': 'QUANTUM',
    'queue_manager': 'QUEUE_MGR',
    'serialization': 'SERIAL',
    'signal': 'SIGNAL',
    'spatial': 'SPATIAL',
    'spectral': 'SPECTRAL',
    'tensor_networks': 'TENSOR',
    'thread': 'THREAD',
    'time': 'TIME',
    'validation': 'VALIDATE',
    'fault_tolerance': 'FAULT_TOL',
}

class UtilsIntegrator:
    def __init__(self, source_file):
        self.source_file = Path(source_file)
        self.module_name = self._get_module_name()
        self.malloc_count = 0
        self.calloc_count = 0
        self.realloc_count = 0
        self.free_count = 0

    def _get_module_name(self):
        """Extract module name from file path"""
        parts = self.source_file.parts
        for i, part in enumerate(parts):
            if part == 'utils' and i + 1 < len(parts):
                subdir = parts[i + 1]
                return MODULE_MAP.get(subdir, subdir.upper())
        return 'UTILS'

    def _has_integration(self, content):
        """Check if file already has logging and unified memory"""
        has_logging = 'nimcp_logging.h' in content
        has_unified = 'nimcp_unified_memory.h' in content
        has_log_module = 'LOG_MODULE' in content
        return has_logging and has_unified and has_log_module

    def _add_includes(self, lines):
        """Add necessary includes after existing includes"""
        result = []
        include_end = -1
        has_logging = False
        has_unified = False
        has_log_module = False

        # Find where includes end and check what we have
        for i, line in enumerate(lines):
            if line.strip().startswith('#include'):
                include_end = i
                if 'nimcp_logging.h' in line:
                    has_logging = True
                if 'nimcp_unified_memory.h' in line:
                    has_unified = True
            elif line.strip().startswith('#define LOG_MODULE'):
                has_log_module = True

        # Insert includes
        for i, line in enumerate(lines):
            result.append(line)

            # After last include, add our includes
            if i == include_end and not (has_logging and has_unified and has_log_module):
                if not has_logging:
                    result.append('#include "utils/logging/nimcp_logging.h"\n')
                if not has_unified:
                    result.append('#include "utils/memory/nimcp_unified_memory.h"\n')
                result.append('\n')
                if not has_log_module:
                    result.append(f'#define LOG_MODULE "{self.module_name}"\n')
                    result.append('\n')

        return result

    def _replace_memory_calls(self, line):
        """Replace memory allocation calls with unified memory versions"""
        original = line

        # Replace malloc - avoid nimcp_malloc, unified_malloc, etc.
        line = re.sub(r'(?<![_a-zA-Z])malloc\s*\(', 'nimcp_malloc(', line)
        if line != original:
            self.malloc_count += 1
            original = line

        # Replace calloc
        line = re.sub(r'(?<![_a-zA-Z])calloc\s*\(', 'nimcp_calloc(', line)
        if line != original:
            self.calloc_count += 1
            original = line

        # Replace realloc
        line = re.sub(r'(?<![_a-zA-Z])realloc\s*\(', 'nimcp_realloc(', line)
        if line != original:
            self.realloc_count += 1
            original = line

        # Replace free - avoid nimcp_free, etc.
        line = re.sub(r'(?<![_a-zA-Z])free\s*\(', 'nimcp_free(', line)
        if line != original:
            self.free_count += 1

        return line

    def integrate(self):
        """Integrate logging and unified memory into source file"""
        if not self.source_file.exists():
            print(f"ERROR: File not found: {self.source_file}")
            return False

        # Read file
        with open(self.source_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check if already integrated
        if self._has_integration(content):
            print(f"SKIP: {self.source_file.name} (already integrated)")
            return True

        # Process line by line
        lines = content.splitlines(keepends=True)

        # Add includes
        lines = self._add_includes(lines)

        # Replace memory calls
        result_lines = []
        for line in lines:
            result_lines.append(self._replace_memory_calls(line))

        # Write back
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.writelines(result_lines)

        # Report
        total = self.malloc_count + self.calloc_count + self.realloc_count + self.free_count
        if total > 0:
            print(f"OK: {self.source_file.name} - malloc:{self.malloc_count} calloc:{self.calloc_count} realloc:{self.realloc_count} free:{self.free_count}")
        else:
            print(f"OK: {self.source_file.name} - includes added (no memory calls)")

        return True

    def get_stats(self):
        return {
            'malloc': self.malloc_count,
            'calloc': self.calloc_count,
            'realloc': self.realloc_count,
            'free': self.free_count
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: integrate_all_utils.py <source_file>")
        sys.exit(1)

    integrator = UtilsIntegrator(sys.argv[1])
    success = integrator.integrate()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
