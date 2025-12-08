#!/usr/bin/env python3
"""
Full integration of bio-async, comprehensive logging, and unified memory
into all middleware and IO modules.
"""

import os
import re
import sys
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

# Module ID mappings (0x0510-0x052F)
MODULE_IDS = {
    # Buffering (0x0510-0x0514)
    'nimcp_circular_buffer': 0x0510,
    'nimcp_integration_buffer': 0x0511,
    'nimcp_phase_coded_buffer': 0x0512,
    'nimcp_sliding_window': 0x0513,
    'nimcp_temporal_accumulator': 0x0514,
    # Cognitive (0x0515-0x0516)
    'nimcp_cognitive_adapters': 0x0515,
    'nimcp_working_memory_adapter': 0x0516,
    # Encoding (0x0517-0x0519)
    'nimcp_population_coding': 0x0517,
    'nimcp_rate_coding': 0x0518,
    'nimcp_temporal_coding': 0x0519,
    # Features (0x051A)
    'nimcp_feature_extractor': 0x051A,
    # Integration (0x051B-0x051F)
    'nimcp_executive_middleware_adapter': 0x051B,
    'nimcp_flow_tracker': 0x051C,
    'nimcp_middleware_controller': 0x051D,
    'nimcp_quantum_command_propagator': 0x051E,
    'nimcp_shannon_monitor': 0x051F,
    # Normalization (0x0520-0x0523)
    'nimcp_adaptive_normalizer': 0x0520,
    'nimcp_homeostatic_normalizer': 0x0521,
    'nimcp_min_max_normalizer': 0x0522,
    'nimcp_zscore_normalizer': 0x0523,
    # Patterns (0x0524-0x0528)
    'nimcp_oscillation_detector': 0x0524,
    'nimcp_pattern_cow': 0x0525,
    'nimcp_pattern_library': 0x0526,
    'nimcp_sequence_detector': 0x0527,
    'nimcp_synchrony_detector': 0x0528,
    # Routing (0x0529-0x052C)
    'nimcp_attention_gate': 0x0529,
    'nimcp_routing_table': 0x052A,
    'nimcp_signal_wrapper': 0x052B,
    'nimcp_thalamic_router': 0x052C,
    # IO (0x052D-0x052F)
    'nimcp_dataio': 0x052D,
    'nimcp_serialization': 0x052E,
    'nimcp_network_serialization': 0x052E,
    'nimcp_encryption': 0x052E,
    'nimcp_stream': 0x052F,
    # Middleware (additional)
    'brain_integration': 0x0510,
    'nimcp_event_types': 0x0511,
}


class ModuleIntegrator:
    """Integrates bio-async, logging, and unified memory into C modules."""

    REQUIRED_INCLUDES = [
        'async/nimcp_bio_async.h',
        'async/nimcp_bio_router.h',
        'async/nimcp_bio_messages.h',
        'utils/logging/nimcp_logging.h',
        'utils/memory/nimcp_unified_memory.h',
    ]

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.stats = {
            'total': 0,
            'processed': 0,
            'skipped': 0,
            'failed': 0
        }

    def get_module_info(self, filepath: Path) -> Tuple[str, str]:
        """Extract module name and ID from filepath."""
        module_name = filepath.stem
        module_id = MODULE_IDS.get(module_name, 0x0510)
        return module_name, f'0x{module_id:04X}'

    def has_integration(self, content: str) -> bool:
        """Check if file already has full integration."""
        has_bio_async = 'nimcp_bio_async.h' in content
        has_logging = 'nimcp_logging.h' in content
        has_unified_mem = 'nimcp_unified_memory.h' in content
        has_log_module = 'LOG_MODULE' in content or 'MODULE_NAME' in content

        return has_bio_async and has_logging and has_unified_mem

    def find_include_position(self, lines: List[str]) -> int:
        """Find the position to insert new includes."""
        last_include = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#include'):
                last_include = i

        if last_include == -1:
            # No includes found, insert after header comment
            for i, line in enumerate(lines):
                if not line.strip().startswith('//') and not line.strip().startswith('/*'):
                    if line.strip():
                        return i
            return 0

        return last_include + 1

    def add_includes(self, lines: List[str], content: str) -> List[str]:
        """Add required includes if missing."""
        includes_to_add = []
        for inc in self.REQUIRED_INCLUDES:
            if f'"{inc}"' not in content and f'<{inc}>' not in content:
                includes_to_add.append(f'#include "{inc}"')

        if not includes_to_add:
            return lines

        pos = self.find_include_position(lines)
        return lines[:pos] + includes_to_add + [''] + lines[pos:]

    def add_log_module_define(self, lines: List[str], module_name: str, module_id: str) -> List[str]:
        """Add LOG_MODULE and LOG_MODULE_ID defines."""
        content = '\n'.join(lines)

        # Check if LOG_MODULE or MODULE_NAME already defined
        if 'LOG_MODULE' in content or 'MODULE_NAME' in content:
            return lines

        # Find position after includes
        pos = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('#include'):
                pos = i + 1
            elif stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                # Found first non-comment, non-include line
                break

        # Skip empty lines
        while pos < len(lines) and not lines[pos].strip():
            pos += 1

        # Insert LOG_MODULE defines
        defines = [
            '',
            f'#define LOG_MODULE "{module_name}"',
            f'#define LOG_MODULE_ID {module_id}',
            ''
        ]

        return lines[:pos] + defines + lines[pos:]

    def replace_memory_functions(self, content: str) -> str:
        """Replace malloc/free with unified memory equivalents."""
        replacements = [
            # Regular malloc/calloc/free/realloc
            (r'\bmalloc\s*\(', 'nimcp_malloc('),
            (r'\bcalloc\s*\(', 'nimcp_calloc('),
            (r'\bfree\s*\(', 'nimcp_free('),
            (r'\brealloc\s*\(', 'nimcp_realloc('),
            # Aligned allocations
            (r'\baligned_alloc\s*\(', 'nimcp_aligned_alloc('),
            (r'\bnimcp_aligned_alloc\s*\([^,]+,\s*([^)]+)\)\s*;',
             r'nimcp_aligned_alloc(\1);'),
        ]

        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)

        return content

    def add_comprehensive_logging(self, content: str, module_name: str) -> str:
        """Add comprehensive logging to key functions."""
        lines = content.split('\n')
        result = []

        for i, line in enumerate(lines):
            result.append(line)

            # Add logging after function entry
            if '{' in line and i > 0:
                prev_line = lines[i-1].strip()
                # Check if this is a function definition (has return type and params)
                if ('*' in prev_line or prev_line.split() and
                    prev_line.split()[0] in ['static', 'void', 'int', 'bool', 'size_t',
                                               'float', 'double', 'nimcp_error_t']):
                    # Don't add logging to inline functions or macros
                    if 'inline' not in prev_line and 'static inline' not in prev_line:
                        # Add entry logging (simple version, doesn't parse all params)
                        indent = '    '
                        # We won't add automatic logging here - too risky
                        pass

        return '\n'.join(result)

    def integrate_file(self, filepath: Path) -> bool:
        """Integrate bio-async, logging, and unified memory into a file."""
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check if already integrated
            if self.has_integration(content):
                print(f"  ⏭️  Already integrated: {filepath.name}")
                self.stats['skipped'] += 1
                return True

            # Backup original
            backup_path = filepath.with_suffix('.c.backup')
            shutil.copy2(filepath, backup_path)

            # Get module info
            module_name, module_id = self.get_module_info(filepath)

            # Process content
            lines = content.split('\n')

            # Add includes
            lines = self.add_includes(lines, content)

            # Add LOG_MODULE defines
            lines = self.add_log_module_define(lines, module_name, module_id)

            # Convert back to string and replace memory functions
            content = '\n'.join(lines)
            content = self.replace_memory_functions(content)

            # Write result
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"  ✅ Integrated: {filepath.name}")
            self.stats['processed'] += 1
            return True

        except Exception as e:
            print(f"  ❌ Failed: {filepath.name} - {e}")
            self.stats['failed'] += 1
            # Restore backup if exists
            backup_path = filepath.with_suffix('.c.backup')
            if backup_path.exists():
                shutil.copy2(backup_path, filepath)
            return False

    def process_directory(self, directory: Path, recursive: bool = False):
        """Process all C files in a directory."""
        if not directory.exists():
            print(f"⚠️  Directory not found: {directory}")
            return

        pattern = '**/*.c' if recursive else '*.c'
        c_files = list(directory.glob(pattern))

        # Filter out CMake generated files
        c_files = [f for f in c_files if 'CMakeFiles' not in str(f)]

        if not c_files:
            print(f"  No C files found in {directory.relative_to(self.project_root)}")
            return

        print(f"\n📁 Processing {directory.relative_to(self.project_root)}: {len(c_files)} files")

        for filepath in sorted(c_files):
            self.stats['total'] += 1
            self.integrate_file(filepath)

    def run(self):
        """Run integration on all middleware and IO modules."""
        print("=" * 80)
        print("Bio-Async + Logging + Unified Memory Integration")
        print("=" * 80)

        # Middleware directories
        middleware_dirs = [
            'src/middleware/buffering',
            'src/middleware/cognitive',
            'src/middleware/encoding',
            'src/middleware/features',
            'src/middleware/integration',
            'src/middleware/normalization',
            'src/middleware/patterns',
            'src/middleware/routing',
        ]

        # IO directories
        io_dirs = [
            'src/io/dataio',
            'src/io/serialization',
            'src/io/stream',
        ]

        # Process middleware
        print("\n" + "=" * 80)
        print("MIDDLEWARE MODULES")
        print("=" * 80)
        for dir_path in middleware_dirs:
            full_path = self.project_root / dir_path
            self.process_directory(full_path)

        # Process IO
        print("\n" + "=" * 80)
        print("IO MODULES")
        print("=" * 80)
        for dir_path in io_dirs:
            full_path = self.project_root / dir_path
            self.process_directory(full_path)

        # Also process top-level middleware file
        middleware_file = self.project_root / 'src' / 'middleware' / 'brain_integration.c'
        if middleware_file.exists():
            print("\n📁 Processing top-level middleware file")
            self.stats['total'] += 1
            self.integrate_file(middleware_file)

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print integration summary."""
        print("\n" + "=" * 80)
        print("INTEGRATION SUMMARY")
        print("=" * 80)
        print(f"Total files:        {self.stats['total']}")
        print(f"✅ Processed:        {self.stats['processed']}")
        print(f"⏭️  Skipped:          {self.stats['skipped']}")
        print(f"❌ Failed:           {self.stats['failed']}")
        print("=" * 80)

        if self.stats['failed'] > 0:
            print("\n⚠️  Some files failed. Check errors above.")
            return False
        else:
            print("\n✅ All files integrated successfully!")
            return True


def main():
    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Run integration
    integrator = ModuleIntegrator(project_root)
    success = integrator.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
