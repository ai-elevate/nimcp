#!/usr/bin/env python3
"""Add mesh participant registration to cognitive and brain modules."""
import os, re, sys, glob

BASE = "/home/bbrelin/nimcp"

def extract_module_name(content):
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_ATOMIC\((\w+)\)', content)
    if m: return m.group(1)
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT\((\w+)\)', content)
    if m: return m.group(1)
    m = re.search(r'NIMCP_DECLARE_HEALTH_AGENT_STATIC\((\w+)\)', content)
    if m: return m.group(1)
    return None

def get_category(filepath):
    if '/cognitive/' in filepath:
        if '/memory/' in filepath: return 'MESH_ADAPTER_CATEGORY_MEMORY'
        if '/immune/' in filepath or '/security/' in filepath: return 'MESH_ADAPTER_CATEGORY_SECURITY'
        return 'MESH_ADAPTER_CATEGORY_COGNITIVE'
    if '/core/brain/' in filepath:
        if '/subcortical/' in filepath: return 'MESH_ADAPTER_CATEGORY_SUBCORTICAL'
        return 'MESH_ADAPTER_CATEGORY_SYSTEM'
    return 'MESH_ADAPTER_CATEGORY_COGNITIVE'

def add_mesh_registration(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    if 'mesh_participant_register' in content or '_mesh_register(' in content:
        return False
    
    mn = extract_module_name(content)
    if not mn: return False
    
    cat = get_category(filepath)
    
    # Add includes
    if 'nimcp_mesh_adapter.h' not in content:
        content = re.sub(
            r'(#include\s+"utils/fault_tolerance/nimcp_health_agent_macros\.h")',
            r'\1\n#include "mesh/nimcp_mesh_participant.h"\n#include "mesh/nimcp_mesh_adapter.h"',
            content, count=1
        )
    
    # Add mesh block after NIMCP_DECLARE_HEALTH_AGENT_ATOMIC
    mesh_block = f'''
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_{mn}_mesh_id = 0;
static mesh_participant_registry_t* g_{mn}_mesh_registry = NULL;

nimcp_error_t {mn}_mesh_register(mesh_participant_registry_t* registry) {{
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_{mn}_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "{mn}", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel({cat});
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "{mn}";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_{mn}_mesh_id);
    if (err == NIMCP_SUCCESS) g_{mn}_mesh_registry = registry;
    return err;
}}

void {mn}_mesh_unregister(void) {{
    if (g_{mn}_mesh_registry && g_{mn}_mesh_id != 0) {{
        mesh_participant_unregister(g_{mn}_mesh_registry, g_{mn}_mesh_id);
        g_{mn}_mesh_id = 0;
        g_{mn}_mesh_registry = NULL;
    }}
}}
'''
    
    pattern = r'(NIMCP_DECLARE_HEALTH_AGENT_(?:ATOMIC|STATIC)\(\w+\))'
    if re.search(pattern, content):
        content = re.sub(pattern, r'\1' + mesh_block, content, count=1)
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    print("=" * 70)
    print("Mesh Participant Registration")
    print("=" * 70)
    
    total = 0
    for d in ['src/cognitive', 'src/core/brain', 'src/mesh', 'src/security']:
        dir_path = os.path.join(BASE, d)
        files = sorted(glob.glob(os.path.join(dir_path, '**', '*.c'), recursive=True))
        files = [f for f in files if 'CMakeFiles' not in f and 'venv' not in f]
        count = sum(1 for f in files if add_mesh_registration(f))
        total += count
        print(f'{d}: {count}/{len(files)} modified')
    
    print(f'\nTotal: {total}')

if __name__ == '__main__':
    main()
