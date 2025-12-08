#!/usr/bin/env python3
"""
Fix bio-async integration issues in Plasticity module files.
This script fixes variable names and placement issues from the first pass.
"""

import os
import re
from pathlib import Path

def fix_stdp_file(filepath):
    """Fix STDP file - remove duplicate and fix variable names."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove the incorrectly placed bio-async code from stdp_synapse_init
    content = re.sub(
        r'void stdp_synapse_init\(stdp_synapse_t\* synapse\) \{\s+stdp_config_t config = stdp_config_default\(\);\s+stdp_synapse_init_with_config\(synapse, &config\);\s+/\* Bio-async integration \*/.*?}',
        '''void stdp_synapse_init(stdp_synapse_t* synapse) {
    stdp_config_t config = stdp_config_default();
    stdp_synapse_init_with_config(synapse, &config);
}''',
        content,
        flags=re.DOTALL
    )

    # Note: stdp_synapse_t doesn't have bio_ctx fields, so we'll use module-level
    # registration if there's a module_init function

    with open(filepath, 'w') as f:
        f.write(content)

    return True

def fix_bcm_file(filepath):
    """Fix BCM file - remove unreachable code after return."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Remove the bio-async code that was added after return statement
    content = re.sub(
        r'(bcm_synapse_t bcm_synapse_init\([^)]+\) \{.*?return synapse;)\s+/\* Bio-async integration \*/.*?(\})',
        r'\1\n\2',
        content,
        flags=re.DOTALL
    )

    with open(filepath, 'w') as f:
        f.write(content)

    return True

def fix_variable_names(filepath):
    """Fix incorrect 'result' variable names to appropriate names."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if file has 'result->' references that should be 'synapse->' or 'system->'
    if 'result->bio_ctx' not in content:
        return False

    # Try to determine correct variable name from context
    # Look for common patterns
    if 'synapse->' in content and 'synapse->bio_ctx' not in content:
        content = content.replace('result->bio_ctx', 'synapse->bio_ctx')
        content = content.replace('result->bio_async_enabled', 'synapse->bio_async_enabled')
    elif 'system->' in content and 'system->bio_ctx' not in content:
        content = content.replace('result->bio_ctx', 'system->bio_ctx')
        content = content.replace('result->bio_async_enabled', 'system->bio_async_enabled')
    elif 'state->' in content and 'state->bio_ctx' not in content:
        content = content.replace('result->bio_ctx', 'state->bio_ctx')
        content = content.replace('result->bio_async_enabled', 'state->bio_async_enabled')

    with open(filepath, 'w') as f:
        f.write(content)

    return True

def remove_duplicate_registrations(filepath):
    """Remove duplicate bio-async registration blocks."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Find all bio-async integration blocks
    pattern = r'/\* Bio-async integration \*/.*?bio_router_register_module\([^)]+\);\s+\}'
    matches = list(re.finditer(pattern, content, re.DOTALL))

    if len(matches) > 1:
        # Keep only the first one, remove others
        for match in reversed(matches[1:]):
            content = content[:match.start()] + content[match.end():]

        with open(filepath, 'w') as f:
            f.write(content)
        return True

    return False

def main():
    """Main entry point."""
    base_dir = Path(__file__).parent.parent
    plasticity_dir = base_dir / 'src' / 'plasticity'

    print("Fixing bio-async integration issues...")
    print("=" * 80)

    # Fix specific known issues
    stdp_file = plasticity_dir / 'stdp' / 'nimcp_stdp.c'
    if stdp_file.exists():
        print(f"\nFixing {stdp_file.relative_to(base_dir)}...")
        fix_stdp_file(stdp_file)
        print("  ✓ Fixed STDP file")

    bcm_file = plasticity_dir / 'bcm' / 'nimcp_bcm.c'
    if bcm_file.exists():
        print(f"\nFixing {bcm_file.relative_to(base_dir)}...")
        fix_bcm_file(bcm_file)
        print("  ✓ Fixed BCM file")

    # Fix variable names in all files
    c_files = list(plasticity_dir.rglob('*.c'))
    fixed_files = []

    for c_file in c_files:
        if fix_variable_names(c_file):
            fixed_files.append(c_file)
            print(f"  ✓ Fixed variable names in {c_file.name}")

        if remove_duplicate_registrations(c_file):
            if c_file not in fixed_files:
                fixed_files.append(c_file)
            print(f"  ✓ Removed duplicate registrations in {c_file.name}")

    print("\n" + "=" * 80)
    print(f"Fixed {len(fixed_files)} files")

if __name__ == '__main__':
    main()
