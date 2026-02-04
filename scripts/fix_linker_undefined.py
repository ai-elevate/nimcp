#!/usr/bin/env python3
"""Fix undefined reference linker errors in libnimcp.so.

These are functions/macros referenced but not defined due to migration scripts
removing NIMCP_DECLARE_HEALTH_AGENT_ATOMIC macros or other migration artifacts.
"""

import re
import os

ROOT = "/home/bbrelin/nimcp"
fixed = 0

def read_file(path):
    with open(path, 'r', errors='replace') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

# ============================================================================
# Category 1: Files that call MODULE_heartbeat but the NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
# was removed (because a different-signature heartbeat existed).
# Fix: Add a static inline stub for the standard-signature heartbeat.
# ============================================================================

# Map: heartbeat function name -> file that calls it
heartbeat_stubs_needed = {
    # File -> list of (func_name) that need stubs
    "src/async/bridges/nimcp_bio_async_plasticity_bridge.c": [
        "bio_async_plasticity_heartbeat",
    ],
    "src/training/nimcp_cnn_training.c": [
        "cnn_training_heartbeat",
    ],
    "src/lnn/nimcp_lnn_training.c": [
        "lnn_training_heartbeat",
    ],
    "src/cognitive/salience/nimcp_surprise_amplifier.c": [
        "surprise_amplifier_heartbeat",
        "surprise_amplifier_set_health_agent_internal",
    ],
    "src/cognitive/salience/nimcp_surprise_attention_bridge.c": [
        "surprise_att_bridge_heartbeat",
    ],
    "src/cognitive/salience/nimcp_surprise_fep_bridge.c": [
        "surprise_fep_bridge_heartbeat",
    ],
    "src/cognitive/salience/nimcp_surprise_gw_bridge.c": [
        "surprise_gw_bridge_heartbeat",
    ],
    "src/cognitive/vae/nimcp_vae_decoder.c": [
        "vae_decoder_heartbeat",
    ],
    "src/cognitive/vae/nimcp_vae_encoder.c": [
        "vae_encoder_heartbeat",
    ],
    # Financial modules with _global suffix heartbeats
    "src/cognitive/parietal/nimcp_financial_cognitive_orchestrator.c": [
        "fin_orch_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_basal_ganglia_bridge.c": [
        "fin_bg_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_consolidation_bridge.c": [
        "fin_consolidation_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_metacognition_bridge.c": [
        "fin_metacog_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_motivation_bridge.c": [
        "fin_motivation_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_neuromod_bridge.c": [
        "fin_neuromod_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_reasoning_bridge.c": [
        "fin_reasoning_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_regret_bridge.c": [
        "fin_regret_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_stdp_bridge.c": [
        "fin_stdp_heartbeat_global",
    ],
    "src/cognitive/parietal/nimcp_financial_temporal_credit_bridge.c": [
        "fin_temporal_credit_heartbeat_global",
    ],
}

for rel_path, funcs in heartbeat_stubs_needed.items():
    path = os.path.join(ROOT, rel_path)
    if not os.path.exists(path):
        print(f"  SKIP: {rel_path} not found")
        continue

    content = read_file(path)
    changed = False

    for func_name in funcs:
        # Check if this function is already defined (static inline or otherwise)
        if re.search(rf'(static\s+)?inline\s+void\s+{re.escape(func_name)}\s*\(', content):
            continue
        if re.search(rf'^void\s+{re.escape(func_name)}\s*\(', content, re.MULTILINE):
            continue

        # Check if the function is actually called in this file
        if func_name not in content:
            continue

        # Special case: surprise_amplifier_set_health_agent_internal
        if func_name == "surprise_amplifier_set_health_agent_internal":
            stub = (
                f"\n/* Stub for migration compatibility */\n"
                f"static inline void {func_name}(nimcp_health_agent_t* agent) {{\n"
                f"    (void)agent;\n"
                f"}}\n"
            )
        else:
            stub = (
                f"\n/* Stub heartbeat for migration compatibility */\n"
                f"static inline void {func_name}(const char* op, float progress) {{\n"
                f"    (void)op; (void)progress;\n"
                f"}}\n"
            )

        # Insert after the last #include or after the health agent macro
        # Find a good insertion point
        insert_point = None

        # Try after NIMCP_DECLARE_HEALTH_AGENT line
        m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT\w*\([^)]+\)\s*\n', content)
        if m:
            insert_point = m.end()
        else:
            # Try after static g_MODULE_health_agent line
            m = re.search(r'static\s+(?:nimcp_health_agent_t\s*\*|_Atomic\(nimcp_health_agent_t\s*\*\))\s+g_\w+_health_agent\s*=\s*NULL\s*;\s*\n', content)
            if m:
                insert_point = m.end()
            else:
                # Try after last #include
                for m2 in re.finditer(r'^#include\s+.*\n', content, re.MULTILINE):
                    insert_point = m2.end()

        if insert_point:
            content = content[:insert_point] + stub + content[insert_point:]
            changed = True
            print(f"  Added stub {func_name} in {rel_path}")

    if changed:
        write_file(path, content)
        fixed += 1

# ============================================================================
# Category 2: vae_heartbeat - special signature (takes struct pointer + string)
# ============================================================================

vae_main = os.path.join(ROOT, "src/cognitive/vae/nimcp_vae.c")
if os.path.exists(vae_main):
    content = read_file(vae_main)
    if 'vae_heartbeat' in content and not re.search(r'static\s+(?:inline\s+)?void\s+vae_heartbeat\s*\(', content):
        # vae_heartbeat takes (vae_t*, const char*) based on usage pattern
        stub = (
            "\n/* Stub heartbeat for migration compatibility */\n"
            "static inline void vae_heartbeat(void* vae, const char* op) {\n"
            "    (void)vae; (void)op;\n"
            "}\n"
        )
        # Insert after includes
        insert_point = 0
        for m in re.finditer(r'^#include\s+.*\n', content, re.MULTILINE):
            insert_point = m.end()
        if insert_point:
            content = content[:insert_point] + stub + content[insert_point:]
            write_file(vae_main, content)
            fixed += 1
            print(f"  Added vae_heartbeat stub in nimcp_vae.c")

# ============================================================================
# Category 3: genius_profiles_heartbeat_internal
# ============================================================================

genius_file = os.path.join(ROOT, "src/core/brain/genius/nimcp_genius_profiles.c")
if os.path.exists(genius_file):
    content = read_file(genius_file)
    if 'genius_profiles_heartbeat_internal' in content and \
       not re.search(r'static\s+(?:inline\s+)?void\s+genius_profiles_heartbeat_internal\s*\(', content):
        stub = (
            "\n/* Internal heartbeat stub for migration compatibility */\n"
            "static inline void genius_profiles_heartbeat_internal(const char* op, float progress) {\n"
            "    (void)op; (void)progress;\n"
            "}\n"
        )
        # Insert after health agent macro or includes
        m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT\w*\([^)]+\)\s*\n', content)
        if m:
            insert_point = m.end()
        else:
            insert_point = 0
            for m2 in re.finditer(r'^#include\s+.*\n', content, re.MULTILINE):
                insert_point = m2.end()
        content = content[:insert_point] + stub + content[insert_point:]
        write_file(genius_file, content)
        fixed += 1
        print(f"  Added genius_profiles_heartbeat_internal stub")

# ============================================================================
# Category 4: INFO_THEORY_HEARTBEAT - macro missing
# ============================================================================

info_theory_file = os.path.join(ROOT, "src/utils/statistics/nimcp_information_theory.c")
if os.path.exists(info_theory_file):
    content = read_file(info_theory_file)
    if 'INFO_THEORY_HEARTBEAT' in content and '#define INFO_THEORY_HEARTBEAT' not in content:
        macro = (
            "\n/* Heartbeat macro for migration compatibility */\n"
            "#define INFO_THEORY_HEARTBEAT(op, progress) do { (void)(op); (void)(progress); } while(0)\n"
        )
        # Insert after includes
        insert_point = 0
        for m in re.finditer(r'^#include\s+.*\n', content, re.MULTILINE):
            insert_point = m.end()
        content = content[:insert_point] + macro + content[insert_point:]
        write_file(info_theory_file, content)
        fixed += 1
        print(f"  Added INFO_THEORY_HEARTBEAT macro in information_theory.c")

# ============================================================================
# Category 5: NIMCP_THROW_TO_IMMUNE - missing include
# The macro is defined in include/utils/exception/nimcp_exception_macros.h
# Find files that use it but don't include the header
# ============================================================================

import subprocess

# Find files with NIMCP_THROW_TO_IMMUNE that don't include the macro header
result = subprocess.run(
    ['grep', '-rl', 'NIMCP_THROW_TO_IMMUNE', os.path.join(ROOT, 'src')],
    capture_output=True, text=True
)
throw_immune_files = result.stdout.strip().split('\n') if result.stdout.strip() else []

for fpath in throw_immune_files:
    if not os.path.exists(fpath):
        continue
    content = read_file(fpath)
    # Check if the include is present
    if 'nimcp_exception_macros.h' not in content and 'nimcp_exception.h' not in content:
        # Need to add the include
        # Find after last existing include
        insert_point = 0
        for m in re.finditer(r'^#include\s+.*\n', content, re.MULTILINE):
            insert_point = m.end()
        if insert_point:
            include_line = '#include "utils/exception/nimcp_exception_macros.h"\n'
            content = content[:insert_point] + include_line + content[insert_point:]
            write_file(fpath, content)
            rel = os.path.relpath(fpath, ROOT)
            fixed += 1
            print(f"  Added exception_macros include in {rel}")

# ============================================================================
# Category 6: get_timestamp_us - non-static in file that needs it
# Many files define it as static, but some call it without definition.
# Find files that call get_timestamp_us but don't define it.
# ============================================================================

result = subprocess.run(
    ['grep', '-rl', 'get_timestamp_us', os.path.join(ROOT, 'src')],
    capture_output=True, text=True
)
timestamp_files = result.stdout.strip().split('\n') if result.stdout.strip() else []

for fpath in timestamp_files:
    if not os.path.exists(fpath):
        continue
    content = read_file(fpath)
    # Check if this file DEFINES get_timestamp_us
    if re.search(r'(static\s+)?uint64_t\s+get_timestamp_us\s*\(', content):
        continue  # Already defined
    # Check if it CALLS get_timestamp_us
    if not re.search(r'\bget_timestamp_us\s*\(', content):
        continue  # Not called (just commented?)

    # Add a static inline definition
    stub = (
        "\n/* Timestamp utility stub */\n"
        "#include <time.h>\n"
        "static inline uint64_t get_timestamp_us(void) {\n"
        "    struct timespec ts;\n"
        "    clock_gettime(CLOCK_MONOTONIC, &ts);\n"
        "    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;\n"
        "}\n"
    )
    # Insert after includes
    insert_point = 0
    for m in re.finditer(r'^#include\s+.*\n', content, re.MULTILINE):
        insert_point = m.end()
    if insert_point:
        content = content[:insert_point] + stub + content[insert_point:]
        write_file(fpath, content)
        rel = os.path.relpath(fpath, ROOT)
        fixed += 1
        print(f"  Added get_timestamp_us stub in {rel}")

print(f"\n=== {fixed} files fixed ===")
