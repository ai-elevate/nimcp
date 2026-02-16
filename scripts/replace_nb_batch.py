#!/usr/bin/env python3
"""Process a single batch of files. Usage: python3 replace_nb_batch.py START COUNT"""
import re, os, sys

CATEGORY_MAP = [
    ("src/core/brain/subcortical/", "MESH_ADAPTER_CATEGORY_SUBCORTICAL"),
    ("src/cognitive/", "MESH_ADAPTER_CATEGORY_COGNITIVE"),
    ("src/core/", "MESH_ADAPTER_CATEGORY_COGNITIVE"),
    ("src/dragonfly/", "MESH_ADAPTER_CATEGORY_COGNITIVE"),
    ("src/glial/", "MESH_ADAPTER_CATEGORY_GLIAL"),
    ("src/mesh/", "MESH_ADAPTER_CATEGORY_SYSTEM"),
    ("src/middleware/", "MESH_ADAPTER_CATEGORY_SYSTEM"),
    ("src/networking/", "MESH_ADAPTER_CATEGORY_SYSTEM"),
    ("src/plasticity/", "MESH_ADAPTER_CATEGORY_PLASTICITY"),
    ("src/security/", "MESH_ADAPTER_CATEGORY_SECURITY"),
]

def get_category(fp):
    for prefix, cat in CATEGORY_MAP:
        if fp.startswith(prefix):
            return cat
    return "MESH_ADAPTER_CATEGORY_COGNITIVE"

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    if not re.search(r'static\s+mesh_participant_id_t\s+g_\w+_mesh_id\s*=\s*0\s*;', content):
        return "no_mesh_id"

    if 'BRIDGE_BOILERPLATE' in content:
        return "already_done"

    # Find module name
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if m:
        module, htype = m.group(1), 'atomic'
    else:
        m = re.search(r'static\s+nimcp_health_agent_t\s*\*\s*g_(\w+)_health_agent\s*=\s*NULL\s*;', content)
        if m:
            module, htype = m.group(1), 'manual'
        else:
            return "no_module"

    category = get_category(filepath)
    has_hb = bool(re.search(rf'static\s+inline\s+void\s+{re.escape(module)}_heartbeat_instance\s*\(', content))

    original = content

    # Replace include
    content = content.replace(
        '#include "utils/fault_tolerance/nimcp_health_agent_macros.h"',
        '#include "utils/bridge/nimcp_bridge_boilerplate.h"'
    )

    # Choose macro
    if has_hb:
        macro_line = f'BRIDGE_BOILERPLATE({module}, {category})'
    else:
        macro_line = f'BRIDGE_BOILERPLATE_MESH_ONLY({module}, {category})'

    # Remove NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
    if htype == 'atomic':
        content = re.sub(r'\n?NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\(' + re.escape(module) + r'\)\s*\n?', '\n', content)

    # Remove manual health agent
    if htype == 'manual':
        content = re.sub(r'/\*\s*Health agent:.*?\*/\s*\n', '', content)
        content = re.sub(r'static\s+nimcp_health_agent_t\s*\*\s*g_' + re.escape(module) + r'_health_agent\s*=\s*NULL\s*;\s*\n', '', content)
        content = re.sub(r'/\*\s*Stub heartbeat.*?\*/\s*\nstatic\s+inline\s+void\s+' + re.escape(module) + r'_heartbeat\s*\([^)]*\)\s*\{[^}]*\}\s*\n?', '', content)

    # Remove mesh registration block (comment + statics + register + unregister)
    mesh_block = (
        r'(?://=+\n//\s*Mesh Participant Registration\s*\n//=+\s*\n)?'
        r'\n?static\s+mesh_participant_id_t\s+g_' + re.escape(module) + r'_mesh_id\s*=\s*0\s*;\s*\n'
        r'static\s+mesh_participant_registry_t\s*\*\s*g_' + re.escape(module) + r'_mesh_registry\s*=\s*NULL\s*;\s*\n'
    )
    content = re.sub(mesh_block, '', content)

    # Remove register function
    content = re.sub(
        r'\n*nimcp_error_t\s+' + re.escape(module) + r'_mesh_register\s*\(mesh_participant_registry_t\s*\*\s*registry\s*\)\s*\{[^}]*\}\s*\n',
        '\n', content
    )

    # Remove unregister function
    content = re.sub(
        r'\n*void\s+' + re.escape(module) + r'_mesh_unregister\s*\(void\)\s*\{[^}]*?\n\s*\}\s*\n',
        '\n', content
    )

    # Remove heartbeat_instance
    if has_hb:
        # Try with doc comment first
        content = re.sub(
            r'\n*(?:/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n)?'
            r'(?:/\*\*\s*@brief.*?\*/\s*\n)?'
            r'static\s+inline\s+void\s+' + re.escape(module) + r'_heartbeat_instance\s*\([^)]*\)\s*\n?\{[^}]*\}\s*\n',
            '\n', content
        )

    # Also remove Phase 8 heartbeat_instance section if present (but keep instance_health_agent var)
    # Some files have a "Phase 8 Instance-Level" comment section with heartbeat_instance
    content = re.sub(
        r'/\*\s*=+\s*\n\s*\*\s*Phase 8 Instance-Level Health Agent Support\s*\n\s*\*\s*=+\s*\*/\s*\n'
        r'\n?static\s+nimcp_health_agent_t\s*\*\s*g_\w+_instance_health_agent\s*=\s*NULL\s*;\s*\n'
        r'\n?static\s+inline\s+void\s+' + re.escape(module) + r'_heartbeat_instance\s*\([^)]*\)\s*\n?\{[^}]*\}\s*\n',
        '\n', content
    ) if has_hb else content

    # Insert macro after mesh includes
    bp_inc = '#include "utils/bridge/nimcp_bridge_boilerplate.h"'
    if bp_inc in content:
        idx = content.index(bp_inc) + len(bp_inc)
        after = content[idx:]
        lines = after.split('\n')
        skip = 0
        for line in lines:
            s = line.strip()
            if s == '' or s.startswith('#include "mesh/'):
                skip += 1
            else:
                break
        pos = idx
        for i in range(skip):
            pos = content.index('\n', pos) + 1
        content = content[:pos] + '\n' + macro_line + '\n' + content[pos:]
    else:
        mesh_inc = '#include "mesh/nimcp_mesh_adapter.h"'
        if mesh_inc in content:
            idx = content.index(mesh_inc)
            idx = content.index('\n', idx) + 1
            content = content[:idx] + '\n' + macro_line + '\n' + content[idx:]
        else:
            return "no_insert_point"

    # Clean blank lines
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    if content == original:
        return "no_change"

    with open(filepath, 'w') as f:
        f.write(content)

    return f"OK:{macro_line}"

os.chdir('/home/bbrelin/nimcp')
import subprocess
result = subprocess.run(['grep', '-rl', 'g_.*_mesh_id = 0', 'src/', '--include=*.c'], capture_output=True, text=True)
all_files = sorted([f for f in result.stdout.strip().split('\n') if f and not f.endswith('_bridge.c')])

start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
count = int(sys.argv[2]) if len(sys.argv) > 2 else len(all_files)
batch = all_files[start:start+count]

ok = 0
for fp in batch:
    try:
        r = process_file(fp)
        if r.startswith("OK:"):
            ok += 1
        print(f"{fp}: {r}")
    except Exception as e:
        print(f"{fp}: ERROR:{e}")

print(f"\nProcessed {len(batch)} files, {ok} modified")
