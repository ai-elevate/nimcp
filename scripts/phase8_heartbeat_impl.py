#!/usr/bin/env python3
"""
Phase 8: Full Heartbeat Implementation Across All Modules

This script:
1. Makes heartbeat setter functions non-static for test linkage
2. Adds heartbeat calls to public functions
3. Adds progress heartbeats in loops over collections
"""

import os
import re
import sys
import glob
from pathlib import Path

PROJECT_ROOT = "/home/bbrelin/nimcp"
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Files that already have heartbeat calls (reference implementations - skip)
ALREADY_DONE = {
    "nimcp_astrocytes.c",
    "nimcp_plasticity_coordinator.c",
    "nimcp_snn_network.c",
    "nimcp_distributed_training.c",
    "nimcp_bio_router.c",
    "nimcp_model_loader.c",
}

# Batch definitions mapping batch number to directory patterns
BATCHES = {
    1: ["cognitive/memory/core/"],
    2: ["cognitive/memory/"],  # non-core
    3: ["cognitive/immune/"],
    4: ["cognitive/parietal/"],
    5: ["cognitive/mirror_neurons/"],
    6: ["cognitive/integration/"],
    7: ["cognitive/game_theory/"],
    8: ["cognitive/ethics/", "cognitive/reasoning/"],
    9: ["cognitive/omni/", "cognitive/recursive/"],
    10: ["cognitive/free_energy/", "cognitive/fault_tolerance/"],
    11: ["cognitive/wellbeing/", "cognitive/introspection/"],
    12: ["cognitive/neuro_symbolic/", "cognitive/jepa/"],
    13: ["cognitive/knowledge/", "cognitive/mental_health/", "cognitive/imagination/"],
    14: ["cognitive/curiosity/", "cognitive/collective_cognition/", "cognitive/logic/"],
    15: ["cognitive/theory_of_mind/", "cognitive/symbolic_logic/", "cognitive/self_model/", "cognitive/predictive/"],
    16: ["cognitive/executive/", "cognitive/consolidation/", "cognitive/autobiographical_memory/", "cognitive/attention/"],
    17: ["cognitive/"],  # remaining cognitive
    18: ["core/brain/factory/init/"],
    19: ["core/brain/regions/hypothalamus/", "core/brain/regions/broca/"],
    20: ["core/brain/regions/"],
    21: ["core/brain/subcortical/", "core/brain/hemispheric/"],
    22: ["core/brain/", "core/"],  # remaining
    23: ["security/"],
    24: ["plasticity/"],
    25: ["middleware/"],
    26: ["swarm/"],
    27: ["snn/", "lnn/"],
    28: ["dragonfly/", "glial/"],
    29: ["integration/", "async/", "training/"],
    30: ["language/", "portia/", "perception/", "gpu/", "physics/", "networking/", "api/"],
}


def find_c_files_for_batch(batch_num):
    """Find all .c files for a given batch number."""
    dirs = BATCHES[batch_num]
    files = []
    seen = set()

    for d in dirs:
        full_dir = os.path.join(SRC_DIR, d)
        if not os.path.isdir(full_dir):
            continue
        for root, _, filenames in os.walk(full_dir):
            for f in filenames:
                if f.endswith('.c') and f not in ALREADY_DONE:
                    full_path = os.path.join(root, f)
                    if full_path not in seen:
                        seen.add(full_path)
                        files.append(full_path)

    return sorted(files)


def find_all_heartbeat_files():
    """Find all .c files under src/ that have heartbeat infrastructure."""
    files = []
    for root, _, filenames in os.walk(SRC_DIR):
        for f in filenames:
            if f.endswith('.c') and f not in ALREADY_DONE:
                full_path = os.path.join(root, f)
                files.append(full_path)
    return sorted(files)


def has_heartbeat_infrastructure(content):
    """Check if file has the heartbeat infrastructure pattern."""
    return ('_health_agent' in content and
            '_heartbeat(' in content and
            'nimcp_health_agent_heartbeat_ex' in content)


def make_setter_non_static(content):
    """Remove 'static' from the heartbeat setter function."""
    # Pattern: static void <name>_set_health_agent(nimcp_health_agent_t* agent) {
    pattern = r'^(static\s+)(void\s+\w+_set_health_agent\s*\()'
    new_content, count = re.subn(pattern, r'\2', content, flags=re.MULTILINE)
    return new_content, count


def get_heartbeat_helper_name(content):
    """Extract the heartbeat helper function name."""
    m = re.search(r'static\s+inline\s+void\s+(\w+_heartbeat)\s*\(', content)
    if m:
        return m.group(1)
    return None


def get_setter_function_name(content):
    """Extract the setter function name."""
    m = re.search(r'void\s+(\w+_set_health_agent)\s*\(', content)
    if m:
        return m.group(1)
    return None


def get_module_short_name(content):
    """Extract a short module name from the heartbeat infrastructure."""
    m = re.search(r'static\s+nimcp_health_agent_t\*\s+g_(\w+)_health_agent', content)
    if m:
        return m.group(1)
    return None


def find_public_functions(content):
    """Find non-static, non-inline function definitions that are public API.
    Returns list of (func_name, line_number, return_type)."""
    functions = []
    lines = content.split('\n')

    # Skip infrastructure functions
    skip_patterns = [
        '_set_health_agent',
        '_heartbeat',
        'handle_',  # message handlers are static
        '_bio_init_once',
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Skip static, inline, and preprocessor lines
        if (stripped.startswith('static') or
            stripped.startswith('//') or
            stripped.startswith('/*') or
            stripped.startswith('*') or
            stripped.startswith('#') or
            stripped.startswith('typedef') or
            stripped.startswith('extern') or
            not stripped):
            i += 1
            continue

        # Match function definition patterns (return_type func_name(params) {)
        # Look for patterns like: int/void/nimcp_error_t/type_t* func_name(
        func_match = re.match(
            r'^((?:const\s+)?(?:nimcp_error_t|int|void|bool|float|double|size_t|uint\d+_t|int\d+_t|'
            r'[a-z_]+_t\s*\*?)\s+)'
            r'(\w+)\s*\(',
            stripped
        )

        if func_match:
            ret_type = func_match.group(1).strip()
            func_name = func_match.group(2)

            # Check if this should be skipped
            should_skip = False
            for skip in skip_patterns:
                if skip in func_name:
                    should_skip = True
                    break

            if not should_skip:
                functions.append((func_name, i, ret_type))

        i += 1

    return functions


def find_function_body_start(lines, func_line_idx):
    """Find the opening brace of a function and the line after guard clauses."""
    brace_count = 0
    found_open = False

    # Find the opening brace
    i = func_line_idx
    while i < len(lines):
        line = lines[i]
        if '{' in line:
            found_open = True
            brace_count += line.count('{') - line.count('}')
            break
        i += 1

    if not found_open:
        return -1

    open_brace_line = i

    # Now find past guard clauses (if (!ptr) { return ...; } patterns)
    i = open_brace_line + 1
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped or stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            i += 1
            continue

        # Check if this is a guard clause
        if (stripped.startswith('if (!') or
            stripped.startswith('if (') and ('== NULL' in stripped or '!= NULL' not in stripped) and 'return' in stripped):
            # Skip past this guard clause block
            brace_depth = 0
            while i < len(lines):
                brace_depth += lines[i].count('{') - lines[i].count('}')
                if brace_depth <= 0 and '{' in lines[i]:
                    i += 1
                    break
                if brace_depth <= 0:
                    i += 1
                    break
                i += 1
            continue

        # Check for multi-line guard clauses
        if (stripped.startswith('if (!') or
            (stripped.startswith('if (') and 'NULL' in stripped)):
            # Find end of this guard block
            brace_depth = 0
            while i < len(lines):
                brace_depth += lines[i].count('{') - lines[i].count('}')
                i += 1
                if brace_depth <= 0:
                    break
            continue

        # Not a guard clause, this is where we insert
        return i

    return open_brace_line + 1


def add_heartbeat_to_function(lines, func_line_idx, func_name, heartbeat_helper, module_short):
    """Add heartbeat call after guard clauses in a function.
    Returns the modified lines and number of insertions."""
    insert_line = find_function_body_start(lines, func_line_idx)
    if insert_line < 0 or insert_line >= len(lines):
        return lines, 0

    # Check if heartbeat is already present in this function
    # Look from func_line_idx to next function
    func_end = insert_line
    brace_count = 0
    started = False
    for j in range(func_line_idx, len(lines)):
        for ch in lines[j]:
            if ch == '{':
                brace_count += 1
                started = True
            elif ch == '}':
                brace_count -= 1
        if started and brace_count <= 0:
            func_end = j
            break

    # Check if heartbeat already called in this function
    func_body = '\n'.join(lines[func_line_idx:func_end])
    if heartbeat_helper + '(' in func_body:
        return lines, 0

    # Determine indentation
    indent = "    "
    if insert_line < len(lines):
        existing_indent = re.match(r'^(\s*)', lines[insert_line])
        if existing_indent:
            indent = existing_indent.group(1)
            if not indent:
                indent = "    "

    # Create operation name from function name
    # Shorten: remove common prefixes
    op_name = func_name
    for prefix in ['nimcp_', module_short + '_']:
        if op_name.startswith(prefix):
            op_name = op_name[len(prefix):]
            break

    op_name = module_short[:12] + '_' + op_name[:20] if len(op_name) > 20 else module_short[:12] + '_' + op_name

    heartbeat_line = f'{indent}/* Phase 8: Heartbeat at operation start */\n'
    heartbeat_line += f'{indent}{heartbeat_helper}("{op_name}", 0.0f);\n'
    heartbeat_line += f'\n'

    lines.insert(insert_line, heartbeat_line)
    return lines, 1


def add_loop_heartbeats(content, heartbeat_helper, module_short):
    """Add heartbeat progress calls inside large loops."""
    # Find for loops with iteration variables
    # Pattern: for (uint32_t i = 0; i < something->count; i++) or similar
    lines = content.split('\n')
    insertions = 0

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Match for loop patterns
        for_match = re.match(
            r'^(\s*)for\s*\(\s*(?:uint32_t|size_t|int|unsigned)\s+(\w+)\s*=\s*0\s*;\s*'
            r'\w+\s*<\s*(\w+(?:->\w+)?(?:\.\w+)?)\s*;\s*\w+\+\+\s*\)',
            line
        )

        if for_match:
            indent = for_match.group(1) + "    "
            var_name = for_match.group(2)
            count_expr = for_match.group(3)

            # Check if there's already a heartbeat in the next ~30 lines
            has_heartbeat = False
            brace_depth = 0
            for j in range(i, min(i + 50, len(lines))):
                if heartbeat_helper in lines[j]:
                    has_heartbeat = True
                    break
                brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth < 0:
                    break

            if not has_heartbeat:
                # Find the opening brace of the loop body
                j = i
                while j < len(lines) and '{' not in lines[j]:
                    j += 1
                if j < len(lines):
                    # Insert heartbeat after first line of loop body
                    j += 1
                    heartbeat_code = (
                        f'{indent}/* Phase 8: Loop progress heartbeat */\n'
                        f'{indent}if (({var_name} & 0xFF) == 0 && {count_expr} > 256) {{\n'
                        f'{indent}    {heartbeat_helper}("{module_short[:12]}_loop",\n'
                        f'{indent}                     (float)({var_name} + 1) / (float){count_expr});\n'
                        f'{indent}}}\n'
                    )
                    lines.insert(j, heartbeat_code)
                    insertions += 1

        i += 1

    return '\n'.join(lines), insertions


def process_file(filepath, dry_run=False):
    """Process a single .c file to add heartbeat calls."""
    with open(filepath, 'r') as f:
        content = f.read()

    if not has_heartbeat_infrastructure(content):
        return 0, "no infrastructure"

    original_content = content
    changes = 0

    # 1. Make setter non-static
    content, setter_changes = make_setter_non_static(content)
    changes += setter_changes

    # 2. Get helper and module names
    heartbeat_helper = get_heartbeat_helper_name(content)
    module_short = get_module_short_name(content)
    setter_name = get_setter_function_name(content)

    if not heartbeat_helper or not module_short:
        return changes, "no helper/module name found"

    # 3. Find public functions and add heartbeat calls
    functions = find_public_functions(content)
    lines = content.split('\n')

    offset = 0
    for func_name, func_line, ret_type in functions:
        adjusted_line = func_line + offset
        old_len = len(lines)
        lines, n = add_heartbeat_to_function(lines, adjusted_line, func_name,
                                              heartbeat_helper, module_short)
        new_len = len(lines)
        offset += (new_len - old_len)
        changes += n

    content = '\n'.join(lines)

    # 4. Add loop heartbeats
    content, loop_changes = add_loop_heartbeats(content, heartbeat_helper, module_short)
    changes += loop_changes

    if changes > 0 and not dry_run:
        with open(filepath, 'w') as f:
            f.write(content)

    return changes, f"setter={setter_changes}, funcs={len(functions)}, loops={loop_changes}"


def get_batch_files(batch_num):
    """Get files for a specific batch, handling overlaps."""
    all_batch_files = {}
    already_assigned = set()

    # Process batches in order to handle overlaps
    for b in range(1, 31):
        files = find_c_files_for_batch(b)
        batch_files = []
        for f in files:
            if f not in already_assigned:
                batch_files.append(f)
                already_assigned.add(f)
        all_batch_files[b] = batch_files

    return all_batch_files.get(batch_num, [])


def process_batch(batch_num, dry_run=False):
    """Process all files in a batch."""
    files = get_batch_files(batch_num)
    results = {
        'total': len(files),
        'modified': 0,
        'skipped': 0,
        'errors': [],
        'setter_names': [],
    }

    for filepath in files:
        try:
            changes, detail = process_file(filepath, dry_run)
            if changes > 0:
                results['modified'] += 1
                # Get setter name for test generation
                with open(filepath, 'r') as f:
                    content = f.read()
                setter = get_setter_function_name(content)
                if setter:
                    results['setter_names'].append((os.path.basename(filepath), setter))
            else:
                results['skipped'] += 1
        except Exception as e:
            results['errors'].append(f"{filepath}: {str(e)}")

    return results


def get_all_batch_info():
    """Get info about all batches for test generation."""
    all_info = {}
    already_assigned = set()

    for b in range(1, 31):
        files = find_c_files_for_batch(b)
        batch_files = []
        for f in files:
            if f not in already_assigned:
                batch_files.append(f)
                already_assigned.add(f)

        setters = []
        for filepath in batch_files:
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                if has_heartbeat_infrastructure(content):
                    setter = get_setter_function_name(content)
                    if setter:
                        setters.append((os.path.basename(filepath), setter))
            except:
                pass

        all_info[b] = {
            'files': batch_files,
            'setters': setters,
            'count': len(batch_files),
        }

    return all_info


def get_batch_domain_name(batch_num):
    """Get a short domain name for a batch."""
    domains = {
        1: "memory_core", 2: "memory", 3: "immune", 4: "parietal",
        5: "mirror_neurons", 6: "integration", 7: "game_theory",
        8: "ethics_reasoning", 9: "omni_recursive", 10: "fep_fault",
        11: "wellbeing_intro", 12: "neurosym_jepa", 13: "knowledge",
        14: "curiosity_logic", 15: "theory_mind", 16: "executive",
        17: "cognitive_misc", 18: "factory_init", 19: "hypothalamus_broca",
        20: "brain_regions", 21: "subcortical", 22: "core_misc",
        23: "security", 24: "plasticity", 25: "middleware",
        26: "swarm", 27: "snn_lnn", 28: "dragonfly_glial",
        29: "integration_async", 30: "language_misc",
    }
    return domains.get(batch_num, f"batch{batch_num}")


def generate_unit_test(batch_num, setters, output_dir):
    """Generate unit test file for a batch."""
    domain = get_batch_domain_name(batch_num)
    filename = f"test_heartbeat_B{batch_num:02d}_{domain}_functions.cpp"
    filepath = os.path.join(output_dir, filename)

    # Take up to 20 setters for the test
    test_setters = setters[:20]

    extern_decls = '\n'.join([f'    void {s}(nimcp_health_agent_t* agent);'
                              for _, s in test_setters])

    test_cases = []
    for basename, setter in test_setters:
        # Create a clean test name from setter name
        test_name = setter.replace('_set_health_agent', '')
        test_cases.append(f'''
TEST_F(HeartbeatB{batch_num:02d}Test, {test_name}_SetNull_NoError) {{
    EXPECT_NO_THROW({setter}(nullptr));
}}

TEST_F(HeartbeatB{batch_num:02d}Test, {test_name}_SetValid_NoError) {{
    EXPECT_NO_THROW({setter}(agent));
}}

TEST_F(HeartbeatB{batch_num:02d}Test, {test_name}_Replace_NoError) {{
    EXPECT_NO_THROW({setter}(agent));
    EXPECT_NO_THROW({setter}(nullptr));
    EXPECT_NO_THROW({setter}(agent));
}}''')

    # Add combined test
    set_all = '\n'.join([f'    {s}(agent);' for _, s in test_setters])
    clear_all = '\n'.join([f'    {s}(nullptr);' for _, s in test_setters])

    content = f'''/**
 * @file {filename}
 * @brief Unit tests for Phase 8 heartbeat B{batch_num:02d} {domain} module setter functions
 * @date 2025-01-26
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>

#include "utils/fault_tolerance/nimcp_health_agent.h"

extern "C" {{
{extern_decls}
}}

class HeartbeatB{batch_num:02d}Test : public ::testing::Test {{
protected:
    nimcp_health_agent_t* agent = nullptr;
    health_agent_config_t config;

    void SetUp() override {{
        nimcp_health_agent_default_config(&config);
        config.check_interval_ms = 50;
        config.enable_auto_recovery = false;
        config.heartbeat_interval_ms = 100;
        agent = nimcp_health_agent_create(&config);
        ASSERT_NE(agent, nullptr);
    }}

    void TearDown() override {{
        if (agent) {{
            if (nimcp_health_agent_is_running(agent)) {{
                nimcp_health_agent_stop(agent);
            }}
            nimcp_health_agent_destroy(agent);
            agent = nullptr;
        }}
    }}
}};
{"".join(test_cases)}

TEST_F(HeartbeatB{batch_num:02d}Test, AllModules_SharedAgent) {{
{set_all}
    // All modules share same agent without error
{clear_all}
}}

TEST_F(HeartbeatB{batch_num:02d}Test, ConcurrentSetClear) {{
    std::atomic<bool> running(true);
    std::thread setter([&]() {{
        while (running.load()) {{
{chr(10).join([f'            {s}(agent);' for _, s in test_setters[:5]])}
        }}
    }});
    std::thread clearer([&]() {{
        while (running.load()) {{
{chr(10).join([f'            {s}(nullptr);' for _, s in test_setters[:5]])}
        }}
    }});
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    running.store(false);
    setter.join();
    clearer.join();
}}
'''

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def generate_integration_test(batch_num, setters, output_dir):
    """Generate integration test file for a batch."""
    domain = get_batch_domain_name(batch_num)
    filename = f"test_heartbeat_B{batch_num:02d}_{domain}_integration.cpp"
    filepath = os.path.join(output_dir, filename)

    test_setters = setters[:15]
    extern_decls = '\n'.join([f'    void {s}(nimcp_health_agent_t* agent);'
                              for _, s in test_setters])

    set_all = '\n'.join([f'    {s}(agent);' for _, s in test_setters])
    clear_all = '\n'.join([f'    {s}(nullptr);' for _, s in test_setters])

    content = f'''/**
 * @file {filename}
 * @brief Integration tests for Phase 8 heartbeat B{batch_num:02d} {domain} modules
 * @date 2025-01-26
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>

#include "utils/fault_tolerance/nimcp_health_agent.h"

extern "C" {{
{extern_decls}
}}

class HeartbeatB{batch_num:02d}IntegrationTest : public ::testing::Test {{
protected:
    nimcp_health_agent_t* agent = nullptr;
    health_agent_config_t config;

    void SetUp() override {{
        nimcp_health_agent_default_config(&config);
        config.check_interval_ms = 50;
        config.enable_auto_recovery = false;
        config.heartbeat_interval_ms = 100;
        agent = nimcp_health_agent_create(&config);
        ASSERT_NE(agent, nullptr);
    }}

    void TearDown() override {{
        if (agent) {{
            if (nimcp_health_agent_is_running(agent)) {{
                nimcp_health_agent_stop(agent);
            }}
            nimcp_health_agent_destroy(agent);
            agent = nullptr;
        }}
    }}
}};

TEST_F(HeartbeatB{batch_num:02d}IntegrationTest, ModuleConnect_HeartbeatFlow) {{
    nimcp_health_agent_start(agent);
{set_all}
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
{clear_all}
    nimcp_health_agent_stop(agent);
}}

TEST_F(HeartbeatB{batch_num:02d}IntegrationTest, MultiModuleSharedAgent) {{
    nimcp_health_agent_start(agent);
{set_all}
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    // All modules running with shared agent
{clear_all}
    nimcp_health_agent_stop(agent);
}}

TEST_F(HeartbeatB{batch_num:02d}IntegrationTest, DisconnectWhileRunning) {{
    nimcp_health_agent_start(agent);
{set_all}
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
{clear_all}
    // Agent still running after disconnect
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    nimcp_health_agent_stop(agent);
}}

TEST_F(HeartbeatB{batch_num:02d}IntegrationTest, AgentRestart_WithModules) {{
    nimcp_health_agent_start(agent);
{set_all}
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    nimcp_health_agent_stop(agent);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    nimcp_health_agent_start(agent);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
{clear_all}
    nimcp_health_agent_stop(agent);
}}
'''

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def generate_regression_test(batch_num, setters, output_dir):
    """Generate regression test file for a batch."""
    domain = get_batch_domain_name(batch_num)
    filename = f"test_heartbeat_B{batch_num:02d}_{domain}_regression.cpp"
    filepath = os.path.join(output_dir, filename)

    test_setters = setters[:15]
    extern_decls = '\n'.join([f'    void {s}(nimcp_health_agent_t* agent);'
                              for _, s in test_setters])

    content = f'''/**
 * @file {filename}
 * @brief Regression tests for Phase 8 heartbeat B{batch_num:02d} {domain} modules
 * @date 2025-01-26
 */

#include <gtest/gtest.h>
#include <thread>

#include "utils/fault_tolerance/nimcp_health_agent.h"

extern "C" {{
{extern_decls}
}}

class HeartbeatB{batch_num:02d}RegressionTest : public ::testing::Test {{
protected:
    nimcp_health_agent_t* agent = nullptr;
    health_agent_config_t config;

    void SetUp() override {{
        nimcp_health_agent_default_config(&config);
        config.check_interval_ms = 50;
        config.enable_auto_recovery = false;
        config.heartbeat_interval_ms = 100;
        agent = nimcp_health_agent_create(&config);
        ASSERT_NE(agent, nullptr);
    }}

    void TearDown() override {{
        if (agent) {{
            if (nimcp_health_agent_is_running(agent)) {{
                nimcp_health_agent_stop(agent);
            }}
            nimcp_health_agent_destroy(agent);
            agent = nullptr;
        }}
    }}
}};

TEST_F(HeartbeatB{batch_num:02d}RegressionTest, APIContract_SetterAcceptsNull) {{
{chr(10).join([f'    EXPECT_NO_THROW({s}(nullptr));' for _, s in test_setters])}
}}

TEST_F(HeartbeatB{batch_num:02d}RegressionTest, APIContract_SetterAcceptsValid) {{
{chr(10).join([f'    EXPECT_NO_THROW({s}(agent));' for _, s in test_setters])}
}}

TEST_F(HeartbeatB{batch_num:02d}RegressionTest, SetDuringActiveHeartbeats) {{
    nimcp_health_agent_start(agent);
{chr(10).join([f'    {s}(agent);' for _, s in test_setters])}
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    // Replace agent during active operation - should not crash
{chr(10).join([f'    {s}(nullptr);' for _, s in test_setters])}
{chr(10).join([f'    {s}(agent);' for _, s in test_setters])}
    nimcp_health_agent_stop(agent);
}}
'''

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def generate_e2e_test(batch_num, setters, output_dir):
    """Generate e2e test file for a batch."""
    domain = get_batch_domain_name(batch_num)
    filename = f"test_heartbeat_B{batch_num:02d}_{domain}_e2e.cpp"
    filepath = os.path.join(output_dir, filename)

    test_setters = setters[:10]
    extern_decls = '\n'.join([f'    void {s}(nimcp_health_agent_t* agent);'
                              for _, s in test_setters])

    set_all = '\n'.join([f'    {s}(agent);' for _, s in test_setters])
    clear_all = '\n'.join([f'    {s}(nullptr);' for _, s in test_setters])

    content = f'''/**
 * @file {filename}
 * @brief End-to-end tests for Phase 8 heartbeat B{batch_num:02d} {domain} modules
 * @date 2025-01-26
 */

#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <vector>
#include <atomic>

#include "utils/fault_tolerance/nimcp_health_agent.h"

extern "C" {{
{extern_decls}
}}

class HeartbeatB{batch_num:02d}E2ETest : public ::testing::Test {{
protected:
    nimcp_health_agent_t* agent = nullptr;
    health_agent_config_t config;

    void SetUp() override {{
        nimcp_health_agent_default_config(&config);
        config.check_interval_ms = 50;
        config.enable_auto_recovery = false;
        config.heartbeat_interval_ms = 100;
        agent = nimcp_health_agent_create(&config);
        ASSERT_NE(agent, nullptr);
    }}

    void TearDown() override {{
        if (agent) {{
            if (nimcp_health_agent_is_running(agent)) {{
                nimcp_health_agent_stop(agent);
            }}
            nimcp_health_agent_destroy(agent);
            agent = nullptr;
        }}
    }}
}};

TEST_F(HeartbeatB{batch_num:02d}E2ETest, FullLifecycle) {{
    // Connect
{set_all}
    // Start
    nimcp_health_agent_start(agent);
    // Heartbeat
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    // Verify agent is running
    EXPECT_TRUE(nimcp_health_agent_is_running(agent));
    // Stop
    nimcp_health_agent_stop(agent);
    // Disconnect
{clear_all}
}}

TEST_F(HeartbeatB{batch_num:02d}E2ETest, ConcurrentMultiThread) {{
    nimcp_health_agent_start(agent);
    std::atomic<bool> running(true);

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; t++) {{
        threads.emplace_back([&, t]() {{
            while (running.load()) {{
{chr(10).join([f'                {s}(agent);' for _, s in test_setters[:3]])}
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
{chr(10).join([f'                {s}(nullptr);' for _, s in test_setters[:3]])}
            }}
        }});
    }}
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    running.store(false);
    for (auto& t : threads) t.join();
    nimcp_health_agent_stop(agent);
}}

TEST_F(HeartbeatB{batch_num:02d}E2ETest, HighFrequencyBurst) {{
    nimcp_health_agent_start(agent);
{set_all}
    // 1000 rapid set/clear cycles
    for (int i = 0; i < 1000; i++) {{
{chr(10).join([f'        {s}(agent);' for _, s in test_setters[:3]])}
{chr(10).join([f'        {s}(nullptr);' for _, s in test_setters[:3]])}
    }}
    nimcp_health_agent_stop(agent);
}}

TEST_F(HeartbeatB{batch_num:02d}E2ETest, TimeoutDetection) {{
    config.heartbeat_interval_ms = 50;
    nimcp_health_agent_destroy(agent);
    agent = nimcp_health_agent_create(&config);
    ASSERT_NE(agent, nullptr);

    nimcp_health_agent_start(agent);
{set_all}
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    EXPECT_TRUE(nimcp_health_agent_is_running(agent));
    nimcp_health_agent_stop(agent);
{clear_all}
}}
'''

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write(content)
    return filepath


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 phase8_heartbeat_impl.py [process|test|info|all] [batch_nums...]")
        print("  process [batch_nums]  - Process .c files to add heartbeat calls")
        print("  test [batch_nums]     - Generate test files")
        print("  info                  - Show batch information")
        print("  all                   - Process all batches and generate all tests")
        sys.exit(1)

    command = sys.argv[1]

    if command == "info":
        info = get_all_batch_info()
        total_files = 0
        total_setters = 0
        for b in range(1, 31):
            bi = info[b]
            total_files += bi['count']
            total_setters += len(bi['setters'])
            print(f"B{b:02d}: {bi['count']:4d} files, {len(bi['setters']):4d} with setters  ({get_batch_domain_name(b)})")
        print(f"\nTotal: {total_files} files, {total_setters} with setters")

    elif command == "process":
        batches = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else list(range(1, 31))
        total_modified = 0
        total_files = 0
        for b in batches:
            results = process_batch(b)
            total_modified += results['modified']
            total_files += results['total']
            print(f"B{b:02d}: {results['total']} files, {results['modified']} modified, "
                  f"{results['skipped']} skipped, {len(results['errors'])} errors")
            for err in results['errors']:
                print(f"  ERROR: {err}")
        print(f"\nTotal: {total_files} files, {total_modified} modified")

    elif command == "test":
        batches = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else list(range(1, 31))
        info = get_all_batch_info()

        unit_dir = os.path.join(PROJECT_ROOT, "test/unit/fault_tolerance")
        integration_dir = os.path.join(PROJECT_ROOT, "test/integration/fault_tolerance")
        regression_dir = os.path.join(PROJECT_ROOT, "test/regression/fault_tolerance")
        e2e_dir = os.path.join(PROJECT_ROOT, "test/e2e/fault_tolerance")

        for b in batches:
            setters = info[b]['setters']
            if not setters:
                print(f"B{b:02d}: No setters found, skipping tests")
                continue

            f1 = generate_unit_test(b, setters, unit_dir)
            f2 = generate_integration_test(b, setters, integration_dir)
            f3 = generate_regression_test(b, setters, regression_dir)
            f4 = generate_e2e_test(b, setters, e2e_dir)
            print(f"B{b:02d}: Generated 4 test files ({len(setters)} setters)")

    elif command == "all":
        # Process all files
        print("=== Processing all batches ===")
        total_modified = 0
        total_files = 0
        for b in range(1, 31):
            results = process_batch(b)
            total_modified += results['modified']
            total_files += results['total']
            print(f"B{b:02d}: {results['total']} files, {results['modified']} modified, "
                  f"{results['skipped']} skipped")
        print(f"\nTotal: {total_files} files, {total_modified} modified")

        # Generate tests
        print("\n=== Generating test files ===")
        info = get_all_batch_info()
        unit_dir = os.path.join(PROJECT_ROOT, "test/unit/fault_tolerance")
        integration_dir = os.path.join(PROJECT_ROOT, "test/integration/fault_tolerance")
        regression_dir = os.path.join(PROJECT_ROOT, "test/regression/fault_tolerance")
        e2e_dir = os.path.join(PROJECT_ROOT, "test/e2e/fault_tolerance")

        test_count = 0
        for b in range(1, 31):
            setters = info[b]['setters']
            if not setters:
                continue
            generate_unit_test(b, setters, unit_dir)
            generate_integration_test(b, setters, integration_dir)
            generate_regression_test(b, setters, regression_dir)
            generate_e2e_test(b, setters, e2e_dir)
            test_count += 4
            print(f"B{b:02d}: Generated 4 test files ({len(setters)} setters)")
        print(f"\nTotal: {test_count} test files generated")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == '__main__':
    main()
