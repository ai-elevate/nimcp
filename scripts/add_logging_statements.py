#!/usr/bin/env python3

import re
import sys
from pathlib import Path

class LoggingEnhancer:
    """Add comprehensive logging to utility files"""

    def __init__(self, source_file):
        self.source_file = Path(source_file)
        self.added_logs = 0

    def _is_function_start(self, line):
        """Check if line is a function definition"""
        # Look for function definitions (return_type function_name(...)
        pattern = r'^\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s+\*?\s*)?(\w+)\s*\([^)]*\)\s*\{'
        return re.match(pattern, line)

    def _get_function_name(self, line):
        """Extract function name from function definition"""
        pattern = r'^\s*(?:static\s+)?(?:inline\s+)?(?:\w+\s+\*?\s*)?(\w+)\s*\('
        match = re.match(pattern, line)
        return match.group(1) if match else None

    def _is_error_return(self, line):
        """Check if line is an error return"""
        line = line.strip()
        return (
            line.startswith('return NULL') or
            line.startswith('return NIMCP_ERROR') or
            line.startswith('return -') or
            'return errno' in line or
            'return false' in line
        )

    def _is_success_return(self, line):
        """Check if line is a success return"""
        line = line.strip()
        return (
            line.startswith('return NIMCP_SUCCESS') or
            line.startswith('return 0;') or
            line.startswith('return true') or
            (line.startswith('return ') and 'ERROR' not in line and 'NULL' not in line)
        )

    def _is_allocation(self, line):
        """Check if line contains memory allocation"""
        return 'nimcp_malloc(' in line or 'nimcp_calloc(' in line or 'nimcp_realloc(' in line

    def _is_deallocation(self, line):
        """Check if line contains memory deallocation"""
        return 'nimcp_free(' in line

    def _add_function_entry_log(self, lines, i, func_name):
        """Add function entry log"""
        indent = len(lines[i]) - len(lines[i].lstrip())
        log_line = ' ' * (indent + 4) + f'LOG_DEBUG("Entering {func_name}");\n'
        return [lines[i], log_line]

    def _add_error_log(self, lines, i):
        """Add error log before error return"""
        line = lines[i]
        indent = len(line) - len(line.lstrip())

        # Extract error context
        error_msg = "Operation failed"
        if 'NULL' in line:
            error_msg = "Allocation or operation returned NULL"
        elif 'ERROR' in line:
            error_msg = "Error condition detected"
        elif 'errno' in line:
            error_msg = "System error occurred"

        log_line = ' ' * indent + f'LOG_ERROR("{error_msg}");\n'
        return [log_line, line]

    def _add_allocation_log(self, lines, i):
        """Add debug log for allocations"""
        line = lines[i]
        indent = len(line) - len(line.lstrip())

        # Extract size if possible
        if 'nimcp_malloc(' in line or 'nimcp_calloc(' in line or 'nimcp_realloc(' in line:
            log_line = ' ' * indent + 'LOG_DEBUG("Memory allocation requested");\n'
            return [log_line, line]

        return [line]

    def _add_deallocation_log(self, lines, i):
        """Add debug log for deallocations"""
        line = lines[i]
        indent = len(line) - len(line.lstrip())
        log_line = ' ' * indent + 'LOG_DEBUG("Memory deallocation");\n'
        return [log_line, line]

    def enhance(self):
        """Add comprehensive logging to source file"""
        if not self.source_file.exists():
            print(f"ERROR: File not found: {self.source_file}")
            return False

        # Read file
        with open(self.source_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Check if logging is already added
        if 'LOG_DEBUG' in content and 'LOG_ERROR' in content:
            print(f"SKIP: {self.source_file.name} (logging already present)")
            return True

        lines = content.splitlines(keepends=True)
        result_lines = []

        i = 0
        in_function = False
        function_name = None
        brace_count = 0

        while i < len(lines):
            line = lines[i]

            # Check for function start
            func_match = self._is_function_start(line)
            if func_match and '{' in line:
                function_name = self._get_function_name(line)
                in_function = True
                brace_count = 1

                # Add function entry log
                new_lines = self._add_function_entry_log(lines, i, function_name)
                result_lines.extend(new_lines)
                self.added_logs += 1
                i += 1
                continue

            # Track braces
            if in_function:
                brace_count += line.count('{') - line.count('}')
                if brace_count == 0:
                    in_function = False
                    function_name = None

            # Add logging for specific patterns
            if in_function:
                # Error returns
                if self._is_error_return(line):
                    new_lines = self._add_error_log(lines, i)
                    result_lines.extend(new_lines)
                    self.added_logs += 1
                    i += 1
                    continue

                # Memory allocations
                if self._is_allocation(line):
                    new_lines = self._add_allocation_log(lines, i)
                    result_lines.extend(new_lines)
                    if len(new_lines) > 1:
                        self.added_logs += 1
                    i += 1
                    continue

                # Memory deallocations
                if self._is_deallocation(line):
                    new_lines = self._add_deallocation_log(lines, i)
                    result_lines.extend(new_lines)
                    if len(new_lines) > 1:
                        self.added_logs += 1
                    i += 1
                    continue

            # Default: keep line as-is
            result_lines.append(line)
            i += 1

        # Write back
        with open(self.source_file, 'w', encoding='utf-8') as f:
            f.writelines(result_lines)

        if self.added_logs > 0:
            print(f"OK: {self.source_file.name} - added {self.added_logs} log statements")
        else:
            print(f"OK: {self.source_file.name} - no logging opportunities found")

        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: add_logging_statements.py <source_file>")
        sys.exit(1)

    enhancer = LoggingEnhancer(sys.argv[1])
    success = enhancer.enhance()
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
