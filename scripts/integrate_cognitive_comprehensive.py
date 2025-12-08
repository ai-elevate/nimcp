#!/usr/bin/env python3
"""
Comprehensive Bio-Async Integration Script for Cognitive Modules
Integrates bio-async, logging, and unified memory into all cognitive reasoning/social modules
"""

import os
import re
import sys
from pathlib import Path

# Module definitions
MODULES = [
    ("src/cognitive/bias/nimcp_bias_detection.c", "cognitive.bias_detection", "0x0340"),
    ("src/cognitive/executive/nimcp_executive.c", "cognitive.executive", "0x0341"),
    ("src/cognitive/explanations/nimcp_explanations.c", "cognitive.explanations", "0x0342"),
    ("src/cognitive/logic/nimcp_symbolic_logic.c", "cognitive.logic", "0x0343"),
    ("src/cognitive/reasoning/nimcp_backward_chaining.c", "cognitive.reasoning.backward", "0x0344"),
    ("src/cognitive/reasoning/nimcp_forward_chaining.c", "cognitive.reasoning.forward", "0x0345"),
    ("src/cognitive/reasoning/nimcp_knowledge_base_interface.c", "cognitive.reasoning.kb", "0x0346"),
    ("src/cognitive/reasoning/nimcp_reasoning_factory.c", "cognitive.reasoning.factory", "0x0347"),
    ("src/cognitive/reasoning/nimcp_reasoning_integration.c", "cognitive.reasoning.integration", "0x0348"),
    ("src/cognitive/reasoning/nimcp_symbolic_logic_attachment.c", "cognitive.reasoning.attachment", "0x0349"),
    ("src/cognitive/reasoning/nimcp_symbolic_logic_brain_integration.c", "cognitive.reasoning.brain_integration", "0x034A"),
    ("src/cognitive/reasoning/nimcp_unification_engine.c", "cognitive.reasoning.unification", "0x034B"),
    ("src/cognitive/reasoning/integration/nimcp_reasoning_attention.c", "cognitive.reasoning.attention", "0x034C"),
    ("src/cognitive/reasoning/integration/nimcp_reasoning_curiosity.c", "cognitive.reasoning.curiosity", "0x034D"),
    ("src/cognitive/theory_of_mind/nimcp_theory_of_mind.c", "cognitive.theory_of_mind", "0x034E"),
    ("src/cognitive/social/nimcp_love_loyalty_friendship.c", "cognitive.social", "0x034F"),
    ("src/cognitive/self_awareness/nimcp_self_awareness_extended.c", "cognitive.self_awareness", "0x0350"),
    ("src/cognitive/self_model/nimcp_self_model.c", "cognitive.self_model", "0x0351"),
    ("src/cognitive/personality/nimcp_personality.c", "cognitive.personality", "0x0352"),
    ("src/cognitive/shadow/nimcp_shadow_emotions.c", "cognitive.shadow", "0x0353"),
    ("src/cognitive/sleep_wake/nimcp_sleep_wake.c", "cognitive.sleep_wake", "0x0354"),
    ("src/cognitive/fault_tolerance/nimcp_emotional_tagging.c", "cognitive.fault.emotional_tag", "0x0355"),
    ("src/cognitive/fault_tolerance/nimcp_failure_prediction.c", "cognitive.fault.prediction", "0x0356"),
    ("src/cognitive/fault_tolerance/nimcp_fault_attention.c", "cognitive.fault.attention", "0x0357"),
    ("src/cognitive/fault_tolerance/nimcp_fault_working_memory.c", "cognitive.fault.working_memory", "0x0358"),
    ("src/cognitive/fault_tolerance/nimcp_metacognition.c", "cognitive.fault.metacognition", "0x0359"),
    ("src/cognitive/fault_tolerance/nimcp_recovery_consolidation.c", "cognitive.fault.recovery_consolidation", "0x035A"),
    ("src/cognitive/fault_tolerance/nimcp_recovery_episodic_memory.c", "cognitive.fault.recovery_episodic", "0x035B"),
    ("src/cognitive/fault_tolerance/nimcp_recovery_executive.c", "cognitive.fault.recovery_executive", "0x035C"),
    ("src/cognitive/mental_health/disorder_detectors.c", "cognitive.mental_health.detectors", "0x035D"),
    ("src/cognitive/mental_health/interventions.c", "cognitive.mental_health.interventions", "0x035E"),
    ("src/cognitive/mental_health/nimcp_mental_health.c", "cognitive.mental_health.core", "0x035F"),
]

BIOASYNC_INCLUDES = """#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"

#define LOG_MODULE "{log_module}"
#define BIO_MODULE_{module_const} {module_id}
"""

def has_bioasync_includes(content):
    """Check if file already has bio-async includes"""
    return 'async/nimcp_bio_async.h' in content

def add_includes(content, log_module, module_id):
    """Add bio-async, logging, and unified memory includes"""
    # Find first #include
    lines = content.split('\n')
    include_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith('#include'):
            include_idx = i
            break

    if include_idx is None:
        print("  WARNING: No #include found, skipping")
        return content

    # Find last consecutive include
    last_include_idx = include_idx
    for i in range(include_idx + 1, len(lines)):
        if not lines[i].strip().startswith('#include') and lines[i].strip():
            break
        if lines[i].strip().startswith('#include'):
            last_include_idx = i

    # Generate module constant name
    module_const = log_module.upper().replace('.', '_')

    # Insert includes after last #include
    insert_idx = last_include_idx + 1
    bioasync_text = BIOASYNC_INCLUDES.format(
        log_module=log_module,
        module_const=module_const,
        module_id=module_id
    )

    lines.insert(insert_idx, bioasync_text)
    return '\n'.join(lines)

def replace_memory_calls(content):
    """Replace malloc/calloc/realloc/free with unified memory equivalents"""
    # Be careful to not replace already-prefixed calls
    content = re.sub(r'\bmalloc\(', 'nimcp_malloc(', content)
    content = re.sub(r'([^_])calloc\(', r'\1nimcp_calloc(', content)
    content = re.sub(r'\brealloc\(', 'nimcp_realloc(', content)
    content = re.sub(r'([^_])free\(', r'\1nimcp_free(', content)

    # Fix potential double replacements
    content = content.replace('nimcp_nimcp_', 'nimcp_')

    return content

def add_function_logging(content):
    """Add LOG_DEBUG/INFO/WARN/ERROR to key functions"""
    # This is a simple heuristic - add LOG_DEBUG to function starts
    # Look for function definitions (simplified pattern)

    # Add logging to _create functions
    content = re.sub(
        r'(\w+\*\s+\w+_create\([^)]*\)\s*\{)',
        r'\1\n    LOG_DEBUG("Creating module");',
        content
    )

    # Add logging to _destroy functions
    content = re.sub(
        r'(void\s+\w+_destroy\([^)]*\)\s*\{)',
        r'\1\n    LOG_DEBUG("Destroying module");',
        content
    )

    # Add error logging for NULL returns
    content = re.sub(
        r'(if\s*\([^)]*\)\s*\{\s*return\s+NULL;)',
        r'LOG_ERROR("NULL parameter or allocation failure");\n        \1',
        content
    )

    return content

def process_file(filepath, log_module, module_id, dry_run=False):
    """Process a single file"""
    print(f"\nProcessing: {filepath}")
    print(f"  Module: {log_module} (ID: {module_id})")

    if not os.path.exists(filepath):
        print(f"  ERROR: File not found")
        return False

    # Read content
    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already integrated
    if has_bioasync_includes(content):
        print("  SKIP: Already has bio-async includes")
        return True

    # Create backup
    backup_path = filepath + '.bioasync_backup'
    if not dry_run and not os.path.exists(backup_path):
        with open(backup_path, 'w') as f:
            f.write(content)
        print(f"  Created backup: {backup_path}")

    # Transform content
    print("  Adding includes...")
    content = add_includes(content, log_module, module_id)

    print("  Replacing memory calls...")
    content = replace_memory_calls(content)

    print("  Adding function logging...")
    content = add_function_logging(content)

    # Write result
    if not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)
        print("  ✓ Integration complete")
    else:
        print("  (Dry run - no changes written)")

    return True

def main():
    script_dir = Path(__file__).parent
    nimcp_root = script_dir.parent

    print("=" * 60)
    print("NIMCP Cognitive Bio-Async Integration")
    print("=" * 60)
    print(f"NIMCP Root: {nimcp_root}")
    print(f"Total modules: {len(MODULES)}")
    print()

    dry_run = '--dry-run' in sys.argv
    if dry_run:
        print("*** DRY RUN MODE - No changes will be written ***\n")

    success_count = 0
    skip_count = 0
    fail_count = 0

    for filepath, log_module, module_id in MODULES:
        full_path = nimcp_root / filepath
        try:
            result = process_file(str(full_path), log_module, module_id, dry_run)
            if result:
                if has_bioasync_includes(open(str(full_path)).read()):
                    skip_count += 1
                else:
                    success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            fail_count += 1

    print("\n" + "=" * 60)
    print("Integration Summary")
    print("=" * 60)
    print(f"Total:      {len(MODULES)}")
    print(f"Success:    {success_count}")
    print(f"Skipped:    {skip_count}")
    print(f"Failed:     {fail_count}")
    print()

    if fail_count == 0:
        print("✓ All files integrated successfully!")
        return 0
    else:
        print("⚠ Some files failed - please review manually")
        return 1

if __name__ == '__main__':
    sys.exit(main())
