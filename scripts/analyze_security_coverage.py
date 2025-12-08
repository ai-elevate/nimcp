#!/usr/bin/env python3
"""
Analyze Security Coverage Across NIMCP Codebase

This script analyzes the security integration coverage across all NIMCP
source files and generates a comprehensive report.
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict

class SecurityAnalyzer:
    def __init__(self, nimcp_root):
        self.nimcp_root = nimcp_root
        self.total_files = 0
        self.files_with_security = 0
        self.files_by_category = defaultdict(list)
        self.security_patterns_found = defaultdict(int)
        self.files_needing_security = []

    def analyze_file(self, file_path):
        """Analyze a single C source file for security integration"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()

            rel_path = os.path.relpath(file_path, self.nimcp_root)
            file_info = {
                'path': rel_path,
                'name': os.path.basename(file_path),
                'has_security': False,
                'security_features': [],
                'handles_external_input': False,
                'needs_security': False,
            }

            # Check for security includes
            if ('security/nimcp_security.h' in content or
                'security/nimcp_blood_brain_barrier.h' in content or
                'security/nimcp_security_integration.h' in content):
                file_info['has_security'] = True
                file_info['security_features'].append('security_include')

            # Check for BBB validation calls
            if re.search(r'bbb_validate_(input|string|integer|pointer)', content):
                file_info['security_features'].append('bbb_validation')
                self.security_patterns_found['bbb_validation'] += 1

            # Check for security initialization
            if 'security_init' in content or 'bbb_system_create' in content:
                file_info['security_features'].append('security_init')
                self.security_patterns_found['security_init'] += 1

            # Check for global BBB system
            if 'g_bbb_system' in content or 'bbb_system_t' in content:
                file_info['security_features'].append('global_bbb')
                self.security_patterns_found['global_bbb'] += 1

            # Check if file handles external input
            external_input_patterns = [
                r'\b(read|recv|load|parse|deserialize)\w*\s*\(',
                r'\b(accept|connect|receive)\w*\s*\(',
                r'\bset_buffer\s*\(',
                r'\bnext_batch\s*\(',
                r'\bprocess_\w+\s*\(',
            ]

            for pattern in external_input_patterns:
                if re.search(pattern, content):
                    file_info['handles_external_input'] = True
                    break

            # Determine if file needs security but doesn't have it
            if file_info['handles_external_input'] and not file_info['has_security']:
                file_info['needs_security'] = True
                self.files_needing_security.append(file_info)

            # Categorize file
            if 'src/io/' in rel_path:
                category = 'IO'
            elif 'src/networking/' in rel_path:
                category = 'Networking'
            elif 'src/middleware/' in rel_path:
                category = 'Middleware'
            elif 'src/cognitive/' in rel_path:
                category = 'Cognitive'
            elif 'src/security/' in rel_path:
                category = 'Security'
            elif 'src/core/' in rel_path:
                category = 'Core'
            else:
                category = 'Other'

            self.files_by_category[category].append(file_info)

            if file_info['has_security']:
                self.files_with_security += 1

            return file_info

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}", file=sys.stderr)
            return None

    def scan_directory(self, directory):
        """Recursively scan directory for C source files"""
        for root, dirs, files in os.walk(directory):
            # Skip build directories
            if 'build' in root or 'CMakeFiles' in root:
                continue

            for file in files:
                if file.endswith('.c') and not file.endswith('.backup'):
                    file_path = os.path.join(root, file)
                    self.total_files += 1
                    self.analyze_file(file_path)

    def generate_report(self):
        """Generate comprehensive security coverage report"""
        print("="*80)
        print("NIMCP SECURITY COVERAGE ANALYSIS REPORT")
        print("="*80)
        print()

        # Overall Statistics
        print("OVERALL STATISTICS")
        print("-" * 80)
        print(f"Total C source files analyzed: {self.total_files}")
        print(f"Files with security integration: {self.files_with_security}")
        coverage_pct = (self.files_with_security / self.total_files * 100) if self.total_files > 0 else 0
        print(f"Security coverage: {coverage_pct:.1f}%")
        print(f"Files needing security: {len(self.files_needing_security)}")
        print()

        # Security Patterns Found
        print("SECURITY PATTERNS DETECTED")
        print("-" * 80)
        for pattern, count in sorted(self.security_patterns_found.items()):
            print(f"  {pattern}: {count} files")
        print()

        # Category Breakdown
        print("SECURITY COVERAGE BY CATEGORY")
        print("-" * 80)
        for category in sorted(self.files_by_category.keys()):
            files = self.files_by_category[category]
            total = len(files)
            secured = sum(1 for f in files if f['has_security'])
            needs = sum(1 for f in files if f['needs_security'])
            pct = (secured / total * 100) if total > 0 else 0

            print(f"\n{category}:")
            print(f"  Total files: {total}")
            print(f"  With security: {secured} ({pct:.1f}%)")
            print(f"  Needs security: {needs}")

            if needs > 0 and category in ['IO', 'Networking', 'Middleware']:
                print(f"  Priority files needing security:")
                for f in files:
                    if f['needs_security']:
                        print(f"    - {f['name']}")

        # High Priority Files Needing Security
        print("\n" + "="*80)
        print("HIGH PRIORITY FILES NEEDING SECURITY")
        print("="*80)

        high_priority = [f for f in self.files_needing_security
                        if any(cat in f['path'] for cat in ['io/', 'networking/', 'middleware/'])]

        if high_priority:
            for file_info in high_priority[:20]:  # Top 20
                print(f"\n{file_info['path']}")
                print(f"  Category: {self._get_category(file_info['path'])}")
                print(f"  Handles external input: Yes")
                print(f"  Has security: No")
        else:
            print("\nNo high-priority files identified!")

        # Recently Secured Files
        print("\n" + "="*80)
        print("RECENTLY SECURED FILES (from this session)")
        print("="*80)

        recent_secured = []
        for category, files in self.files_by_category.items():
            for f in files:
                if f['has_security'] and 'security_init' in f['security_features']:
                    recent_secured.append(f)

        if recent_secured:
            for f in recent_secured[:15]:
                print(f"  ✓ {f['name']}")
                features = ', '.join(f['security_features'])
                print(f"    Features: {features}")
        else:
            print("  No recently secured files detected")

        # Recommendations
        print("\n" + "="*80)
        print("RECOMMENDATIONS")
        print("="*80)
        print(f"""
1. IMMEDIATE PRIORITY (High-Risk External Input):
   - {len([f for f in self.files_needing_security if 'networking/' in f['path']])} networking files need security
   - {len([f for f in self.files_needing_security if 'io/' in f['path']])} IO files need security
   - {len([f for f in self.files_needing_security if 'middleware/' in f['path']])} middleware files need security

2. COVERAGE TARGET:
   - Current: {coverage_pct:.1f}%
   - Target: 30%+ (industry standard for critical systems)
   - Gap: {max(0, 30 - coverage_pct):.1f}% ({max(0, int((30 - coverage_pct) / 100 * self.total_files))} files)

3. NEXT STEPS:
   - Add security validation to remaining IO/networking files
   - Implement input validation at all external entry points
   - Add BBB validation to deserialization functions
   - Test security validation with malicious inputs
   - Document security architecture and validation patterns

4. VALIDATION PRIORITIES:
   a) Network packet reception and deserialization
   b) File I/O and data loading operations
   c) Inter-process communication and events
   d) Command parsing and configuration loading
   e) External API boundaries
""")

    def _get_category(self, path):
        """Get category from file path"""
        if 'src/io/' in path:
            return 'IO'
        elif 'src/networking/' in path:
            return 'Networking'
        elif 'src/middleware/' in path:
            return 'Middleware'
        elif 'src/cognitive/' in path:
            return 'Cognitive'
        else:
            return 'Other'


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_security_coverage.py <nimcp_root>")
        return 1

    nimcp_root = sys.argv[1]
    if not os.path.exists(nimcp_root):
        print(f"Error: Directory not found: {nimcp_root}")
        return 1

    analyzer = SecurityAnalyzer(nimcp_root)

    # Scan source directories
    src_dir = os.path.join(nimcp_root, 'src')
    if os.path.exists(src_dir):
        analyzer.scan_directory(src_dir)
    else:
        print(f"Error: Source directory not found: {src_dir}")
        return 1

    # Generate and display report
    analyzer.generate_report()

    return 0


if __name__ == '__main__':
    sys.exit(main())
