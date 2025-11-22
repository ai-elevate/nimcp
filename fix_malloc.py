#!/usr/bin/env python3
"""
Automated NIMCP Memory Policy Enforcement
Replaces malloc/calloc/realloc/free with nimcp_* equivalents
"""

import re
import sys

FILES = [
    "src/cognitive/bias/nimcp_bias_detection.c",
    "src/cognitive/curiosity/nimcp_curiosity_fractal.c",
    "src/cognitive/curiosity/nimcp_curiosity_hyperbolic.c",
    "src/cognitive/emotions/nimcp_emotional_system.c",
    "src/cognitive/epistemic/nimcp_epistemic_filter.c",
    "src/cognitive/ethics/nimcp_ethics_hyperbolic.c",
    "src/cognitive/fault_tolerance/nimcp_fault_attention.c",
    "src/cognitive/global_workspace/nimcp_global_workspace.c",
    "src/cognitive/grief/nimcp_grief_and_loss.c",
    "src/cognitive/joy/nimcp_joy_euphoria.c",
    "src/cognitive/memory/nimcp_engram.c",
    "src/cognitive/memory/nimcp_semantic_memory.c",
    "src/cognitive/memory/nimcp_wm_transfer.c",
    "src/cognitive/mental_health_monitor.c",
    "src/cognitive/reasoning/integration/nimcp_reasoning_attention.c",
    "src/cognitive/reasoning/integration/nimcp_reasoning_curiosity.c",
    "src/cognitive/remorse/nimcp_remorse_regret.c",
    "src/cognitive/shadow/nimcp_shadow_emotions.c",
    "src/cognitive/working_memory/nimcp_working_memory.c",
    "src/core/brain/learning/nimcp_association_learning.c",
    "src/core/brain/learning/nimcp_circuit_compilation.c",
    "src/core/brain/learning/nimcp_rule_learning.c",
    "src/core/brain/nimcp_brain.c",
    "src/core/brain/nimcp_pretrained.c",
    "src/core/events/nimcp_event_bus.c",
    "src/core/synapse_compute/nimcp_synapse_compute.c",
    "src/core/topology/nimcp_community_detection.c",
    "src/core/topology/nimcp_fractal_topology.c",
    "src/core/topology/nimcp_network_builder.c",
    "src/gpu/nimcp_multigpu.c",
    "src/information/nimcp_cross_modal.c",
    "src/information/nimcp_shannon.c",
    "src/middleware/events/nimcp_event_queue.c",
    "src/middleware/integration/nimcp_quantum_command_propagator.c",
    "src/nlp/nimcp_nlp.c",
    "src/plasticity/adaptive/nimcp_adaptive.c",
    "src/plasticity/neuromodulators/nimcp_neuromod_pink_noise.c",
    "src/plasticity/noise/nimcp_pink_noise.c",
    "src/utils/config/nimcp_dynamic_config.c",
    "src/utils/error/nimcp_error_codes.c",
    "src/utils/fault_tolerance/nimcp_brain_recovery_integration.c",
    "src/utils/fault_tolerance/nimcp_diagnostics.c",
    "src/utils/fault_tolerance/nimcp_fault_event_bus.c",
    "src/utils/fault_tolerance/nimcp_fault_state_machine.c",
    "src/utils/fault_tolerance/nimcp_health_monitor.c",
    "src/utils/fault_tolerance/nimcp_lockfree_metrics.c",
    "src/utils/fault_tolerance/nimcp_metrics_aggregator.c",
    "src/utils/fault_tolerance/nimcp_recovery.c",
    "src/utils/fault_tolerance/nimcp_recovery_cache.c",
    "src/utils/fault_tolerance/nimcp_runtime_adaptation.c",
    "src/utils/numerical/nimcp_integration.c",
    "src/utils/platform/nimcp_platform_thread.c",
    "src/utils/queue_manager/nimcp_queue_manager.c",
]

def fix_file(filepath):
    """Fix malloc/free violations in a single file"""
    try:
        with open(filepath, 'r') as f:
            content = f.read()

        original_content = content

        # Add include if not present
        if 'nimcp_memory.h' not in content:
            # Find first #include and add after it
            content = re.sub(
                r'(#include\s+[<"][^>"]+[>"])',
                r'\1\n#include "utils/memory/nimcp_memory.h"',
                content,
                count=1
            )

        # Replace function calls (word boundaries)
        content = re.sub(r'\bmalloc\s*\(', 'nimcp_malloc(', content)
        content = re.sub(r'\bcalloc\s*\(', 'nimcp_calloc(', content)
        content = re.sub(r'\brealloc\s*\(', 'nimcp_realloc(', content)
        content = re.sub(r'\bfree\s*\(', 'nimcp_free(', content)

        if content != original_content:
            with open(filepath, 'w') as f:
                f.write(content)
            return True
        return False

    except FileNotFoundError:
        print(f"  [SKIP] {filepath} (not found)")
        return False
    except Exception as e:
        print(f"  [ERROR] {filepath}: {e}")
        return False

def main():
    print("=" * 79)
    print("NIMCP Memory Policy Enforcement - Automated Fix")
    print("=" * 79)

    fixed = 0
    for filepath in FILES:
        if fix_file(filepath):
            print(f"  [FIXED] {filepath}")
            fixed += 1

    print()
    print("=" * 79)
    print(f"Summary: Fixed {fixed} / {len(FILES)} files")
    print("=" * 79)

    return 0

if __name__ == '__main__':
    sys.exit(main())
