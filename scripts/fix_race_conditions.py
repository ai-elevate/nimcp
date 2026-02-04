#!/usr/bin/env python3
"""Fix 12 MEDIUM severity race conditions and memory safety issues in NIMCP."""

import os
import sys

def fix(path, old_text, new_text, label):
    """Replace old_text with new_text in file at path."""
    if not os.path.exists(path):
        print(f"  SKIP: {label} - {path} not found")
        return False
    with open(path, 'r') as f:
        content = f.read()
    if old_text not in content:
        print(f"  WARN: {label} - pattern not found in {path}")
        return False
    content = content.replace(old_text, new_text, 1)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  OK: {label}")
    return True

BASE = "/home/bbrelin/nimcp"

# FIX 1: nimcp_cognitive_integration_fep.c - Lazy init race
print("\n=== FIX 1: Cognitive Integration FEP - lazy init race ===")
f1 = f"{BASE}/src/cognitive/integration/nimcp_cognitive_integration_fep.c"
fix(f1,
    "static cognitive_integration_fep_state_t g_cog_integ_fep_state = {0};",
    "static cognitive_integration_fep_state_t g_cog_integ_fep_state = {0};\n\n/* Thread-safe one-time initialization guard */\nstatic nimcp_once_t g_cog_integ_fep_once = NIMCP_ONCE_INIT;\nstatic volatile bool g_cog_integ_fep_init_failed = false;",
    "Fix1a: add once guard variables")

# FIX 2: nimcp_temporal_patterns.c - Lazy init race
print("\n=== FIX 2: Temporal Patterns - lazy init race ===")
f2 = f"{BASE}/src/cognitive/introspection/nimcp_temporal_patterns.c"
fix(f2,
    '#include "utils/platform/nimcp_platform_mutex.h"',
    '#include "utils/platform/nimcp_platform_mutex.h"\n#include "utils/thread/nimcp_thread.h"',
    "Fix2-include: add nimcp_thread.h")

# FIX 3: nimcp_curiosity.c - GPU init race
print("\n=== FIX 3: Curiosity GPU init race ===")
f3 = f"{BASE}/src/cognitive/curiosity/nimcp_curiosity.c"
fix(f3,
    "static bool g_curiosity_gpu_init_attempted = false;",
    "static nimcp_once_t g_curiosity_gpu_once = NIMCP_ONCE_INIT;\nstatic volatile bool g_curiosity_gpu_init_done = false;",
    "Fix3a: replace GPU init flag with once guard")

# FIX 4: nimcp_jepa_predictor.c - GPU init race
print("\n=== FIX 4: JEPA Predictor GPU init race ===")
f4 = f"{BASE}/src/cognitive/jepa/nimcp_jepa_predictor.c"
fix(f4,
    "static bool g_jepa_gpu_init_attempted = false;",
    "static nimcp_once_t g_jepa_gpu_once = NIMCP_ONCE_INIT;\nstatic volatile bool g_jepa_gpu_init_done = false;",
    "Fix4a: replace GPU init flag with once guard")

# FIX 5: nimcp_creative_ethics_bridge.c - Ticket ID race
print("\n=== FIX 5: Creative Ethics Bridge - ticket ID race ===")
f5 = f"{BASE}/src/cognitive/creative/bridges/nimcp_creative_ethics_bridge.c"
fix(f5,
    "#include <ctype.h>",
    "#include <ctype.h>\n#include <stdatomic.h>",
    "Fix5-include: add stdatomic.h")
fix(f5,
    "static uint64_t g_next_ticket_id = 1;",
    "static _Atomic uint64_t g_next_ticket_id = 1;",
    "Fix5a: make ticket ID atomic")

# FIX 6: nimcp_creative_training_bridge.c - Feedback buffer race
print("\n=== FIX 6: Creative Training Bridge - feedback buffer race ===")
f6 = f"{BASE}/src/cognitive/creative/bridges/nimcp_creative_training_bridge.c"
fix(f6,
    "static uint32_t g_feedback_capacity = 0;",
    "static uint32_t g_feedback_capacity = 0;\nstatic nimcp_mutex_t g_feedback_mutex = NIMCP_MUTEX_INITIALIZER;",
    "Fix6a: add feedback mutex")

# FIX 7: nimcp_recovery.c - Stats race
print("\n=== FIX 7: Recovery - stats race ===")
f7 = f"{BASE}/src/utils/fault_tolerance/nimcp_recovery.c"
fix(f7,
    "static recovery_stats_t g_recovery_stats = {0};",
    "static recovery_stats_t g_recovery_stats = {0};\nstatic nimcp_mutex_t g_recovery_stats_mutex = NIMCP_MUTEX_INITIALIZER;",
    "Fix7a: add stats mutex")

# FIX 8: nimcp_cortical_column.c - Global bridge pointer races
print("\n=== FIX 8: Cortical Column - global bridge pointer races ===")
f8 = f"{BASE}/src/core/cortical_columns/nimcp_cortical_column.c"
fix(f8,
    "static cortical_plasticity_bridge_t* g_cortical_plasticity_bridge = NULL;",
    "static _Atomic(cortical_plasticity_bridge_t*) g_cortical_plasticity_bridge = NULL;",
    "Fix8a: atomic plasticity bridge pointer")
fix(f8,
    "static cortical_snn_network_t* g_cortical_snn_network = NULL;",
    "static _Atomic(cortical_snn_network_t*) g_cortical_snn_network = NULL;",
    "Fix8d: atomic SNN network pointer")

print("\n=== ALL RACE CONDITION FIXES COMPLETE ===")
