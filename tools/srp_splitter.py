#!/usr/bin/env python3
"""
SRP File Splitter for NIMCP Project
====================================
Splits god files into focused single-responsibility modules.

Strategy:
- Parse each .c file to find function definitions
- Classify functions by name patterns into responsibility groups
- Create internal headers with shared declarations
- Write split .c files with grouped functions
- Create thin facade from original
- Generate CMakeLists.txt additions

Usage:
    python3 tools/srp_splitter.py [--dry-run] [--file FILE]
"""

import re
import os
import sys
import argparse
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path("/home/bbrelin/nimcp")
SRC_ROOT = PROJECT_ROOT / "src"

# ============================================================================
# C Function Parser
# ============================================================================

def find_function_definitions(source_lines):
    """
    Find all top-level function definitions in C source.
    Returns list of (func_name, start_line, end_line, is_static, signature, body_lines)
    Lines are 0-indexed.
    """
    functions = []
    i = 0
    n = len(source_lines)

    while i < n:
        line = source_lines[i]

        # Skip preprocessor, comments, blank lines
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
            # Skip multi-line comments
            if stripped.startswith('/*') and '*/' not in stripped:
                while i < n and '*/' not in source_lines[i]:
                    i += 1
            i += 1
            continue

        # Look for function definition pattern:
        # [static] [inline] return_type [*] func_name(
        # This is a heuristic - look for lines with '(' that are followed by '{'
        # and are not typedefs, struct definitions, or declarations ending with ';'

        # Collect potential multi-line signature
        if '(' in line and not stripped.startswith('typedef') and not stripped.startswith('struct ') and not stripped.startswith('enum ') and not stripped.startswith('union '):
            sig_start = i
            sig_lines = [line]

            # Collect until we find the closing ')' of the signature
            paren_depth = line.count('(') - line.count(')')
            j = i + 1
            while paren_depth > 0 and j < n:
                sig_lines.append(source_lines[j])
                paren_depth += source_lines[j].count('(') - source_lines[j].count(')')
                j += 1

            # Now look for '{' after the signature
            # It might be on the same line as ')' or the next line
            full_sig = ''.join(sig_lines)

            # Check if this looks like a function definition (has '{' after ')')
            brace_line = j - 1
            found_brace = False

            # Check if '{' is on the last signature line
            if '{' in sig_lines[-1]:
                found_brace = True
                brace_line = j - 1
            # Check next few lines for '{'
            elif j < n:
                for k in range(j, min(j + 3, n)):
                    sk = source_lines[k].strip()
                    if sk.startswith('{'):
                        found_brace = True
                        brace_line = k
                        break
                    elif sk and not sk.startswith('//') and not sk.startswith('/*') and sk != '':
                        break

            if found_brace:
                # Check it's not just a declaration (would have ';' before '{')
                sig_text = ''.join(s.strip() for s in sig_lines).strip()
                if ';' in sig_text.split(')')[0]:
                    i += 1
                    continue

                # Extract function name from signature
                func_name = extract_func_name(full_sig)
                if func_name and not func_name.startswith('__'):
                    is_static = 'static ' in full_sig.split(func_name)[0] if func_name in full_sig else False

                    # Find the matching closing brace
                    brace_depth = 0
                    end_line = brace_line
                    for k in range(brace_line, n):
                        brace_depth += source_lines[k].count('{') - source_lines[k].count('}')
                        if brace_depth == 0:
                            end_line = k
                            break

                    # Also capture any comment block above the function
                    comment_start = sig_start
                    while comment_start > 0:
                        prev = source_lines[comment_start - 1].strip()
                        if prev.endswith('*/') or prev.startswith('*') or prev.startswith('/**') or prev.startswith('//'):
                            comment_start -= 1
                        elif prev == '':
                            # Allow one blank line between comment and function
                            if comment_start > 1 and (source_lines[comment_start - 2].strip().endswith('*/') or source_lines[comment_start - 2].strip().startswith('//')):
                                comment_start -= 1
                            else:
                                break
                        else:
                            break

                    functions.append({
                        'name': func_name,
                        'start': comment_start,
                        'sig_start': sig_start,
                        'end': end_line,
                        'is_static': is_static,
                        'signature': full_sig.strip(),
                    })

                    i = end_line + 1
                    continue

        i += 1

    return functions


def extract_func_name(signature):
    """Extract function name from a C function signature."""
    # Remove attributes, __attribute__((xxx))
    sig = re.sub(r'__attribute__\s*\(\([^)]*\)\)', '', signature)
    # Remove line breaks
    sig = ' '.join(sig.split())

    # Pattern: [qualifiers] return_type [*] func_name(
    match = re.search(r'(\w+)\s*\(', sig)
    if match:
        name = match.group(1)
        # Filter out keywords that aren't function names
        keywords = {'if', 'for', 'while', 'switch', 'return', 'sizeof', 'typeof',
                    'else', 'do', 'case', 'default', 'goto', 'break', 'continue',
                    'struct', 'enum', 'union', 'typedef', 'extern', 'static',
                    'inline', 'volatile', 'const', 'register', 'restrict',
                    'void', 'int', 'float', 'double', 'char', 'long', 'short',
                    'unsigned', 'signed', 'bool', 'uint32_t', 'uint64_t', 'int32_t',
                    'size_t', 'ssize_t', 'nimcp_error_t', 'nimcp_result_t',
                    '_Atomic', 'atomic_store', 'atomic_load'}
        if name in keywords:
            # Try to find the actual function name after the keyword
            # Look for pattern: keyword(...) or keyword name(
            rest = sig[match.end():]
            match2 = re.search(r'(\w+)\s*\(', rest)
            if match2 and match2.group(1) not in keywords:
                return match2.group(1)
            return None
        return name
    return None


# ============================================================================
# Function Classification Rules
# ============================================================================

# Classification rules: (pattern_list, target_file_suffix)
# Functions matching patterns in order; first match wins

CLASSIFICATION_RULES = {
    # Generic rules applied to all files
    '_default': [
        # Lifecycle
        (['_create', '_destroy', '_init', '_cleanup', '_free', '_new', '_alloc',
          '_shutdown', '_start', '_stop', '_reset', '_configure'],
         'lifecycle'),
        # Stats/metrics
        (['_stats', '_metrics', '_get_stats', '_get_metrics', '_report',
          '_dump', '_print_stats', '_log_stats'],
         'stats'),
        # Serialization/IO
        (['_serialize', '_deserialize', '_save', '_load', '_export', '_import',
          '_to_bytes', '_from_bytes', '_to_json', '_from_json', '_checkpoint',
          '_write_', '_read_', 'crc32', 'write_u8', 'write_u32', 'write_u64',
          'write_float', 'write_double', 'read_u8', 'read_u32', 'read_u64',
          'read_float', 'read_double'],
         'io'),
    ],
}

# File-specific rules (override defaults for specific god files)
FILE_SPECIFIC_RULES = {
    'nimcp_health_agent.c': [
        (['_cognitive', '_failure_predict', '_metacog', '_wellbeing', '_emotion',
          '_ethics', '_mental_health', '_collective', '_rcog', '_gpu_health',
          'agent_run_failure', 'agent_run_metacog', 'agent_run_wellbeing',
          'agent_apply_emotion', 'agent_check_gpu', 'agent_check_ethics',
          'agent_get_collective', 'agent_run_rcog'],
         'cognitive'),
        (['_neural', '_snn', '_lnn', 'agent_run_neural', 'agent_check_snn',
          'agent_check_lnn', 'agent_update_neural'],
         'neural'),
        (['_behavioral', '_dragonfly', '_portia', 'agent_run_behavioral',
          'agent_check_dragonfly', 'agent_check_portia', 'agent_update_behavioral',
          'agent_run_cross_module'],
         'behavioral'),
        (['_hypothalamus', '_homeosta', '_drives', '_hypo_', 'hypo_drive',
          'agent_run_hypothalamus', 'agent_run_homeostatic'],
         'homeostasis'),
        (['_hippocampus', '_mammillary', '_engram', '_consolidat', '_memory',
          'agent_check_memory'],
         'memory'),
        (['_swarm', 'swarm_immune', 'swarm_memory'],
         'swarm'),
        (['_connectivity', '_oscillat', '_gc_', '_checkpoint', '_capacity',
          '_logic', '_substrate', '_thalamic', '_middleware', '_perception',
          '_cortical', '_brain_probe', '_wm_', '_imagination', '_jepa',
          'agent_check_connect', 'agent_check_oscillat', 'agent_auto_gc',
          'agent_auto_checkpoint'],
         'system'),
        (['_consistency', '_check_reference', '_check_pointer', '_check_struct',
          '_check_mutex', '_check_circular', '_check_knowledge', '_check_neuron',
          'agent_run_consistency'],
         'consistency'),
        (['_create', '_destroy', '_init', '_free', '_start', '_stop',
          '_configure', 'agent_thread_main', 'msg_queue_', 'get_timestamp',
          '_heartbeat', '_set_health', '_register', '_connect_'],
         'core'),
    ],
    'nimcp_brain.c': [
        (['_forward', '_predict', '_infer', '_decide', '_process_input'],
         'inference'),
        (['_train', '_backprop', '_learn', '_update_weights', '_gradient',
          '_learning_rate', '_schedule'],
         'learning'),
        (['_layer', '_network', '_connect', '_topology', '_add_layer',
          '_remove_layer'],
         'network'),
        (['_serialize', '_deserialize', '_save', '_load', '_checkpoint',
          '_export', '_import'],
         'io'),
        (['_stats', '_metrics', '_get_stats', '_report', '_dump'],
         'stats'),
        (['_config', '_set_param', '_get_param', '_set_option'],
         'config'),
    ],
    'nimcp_omni_world_model.c': [
        (['_predict', '_rssm', '_dynamics', '_step', '_encode', '_decode',
          '_latent', '_prior', '_posterior'],
         'prediction'),
        (['_simulate', '_counterfactual', '_rollout', '_imagine',
          '_replay', '_experience'],
         'simulation'),
        (['_serialize', '_deserialize', '_save', '_load', '_checkpoint',
          'crc32', 'write_u', 'read_u', 'write_float', 'read_float',
          'write_double', 'read_double', 'serialize_', 'deserialize_'],
         'serialization'),
        (['_bio_async', '_bio_router', '_wiring', '_message_handler',
          '_connect_bio', '_disconnect_bio'],
         'bio_async'),
        (['_state_create', '_state_destroy', '_state_clone', '_state_copy',
          '_rssm_state', '_latent_state'],
         'state'),
    ],
    'nimcp_brain_immune.c': [
        (['_antigen', '_present_', '_bbb_threat', '_byzantine', '_swarm_threat'],
         'antigen'),
        (['_b_cell', '_activate_b', '_plasma', '_memory_b', '_affinity',
          'update_b_cell'],
         'bcell'),
        (['_t_cell', '_helper_t', '_killer_t', '_t_help', '_t_kill',
          'update_t_cell'],
         'tcell'),
        (['_antibody', '_produce_antibody', '_execute_antibody', '_neutralize',
          'decay_antibod'],
         'antibody'),
        (['_cytokine', '_release_cytokine', '_broadcast_alert', '_get_cytokine',
          '_inflammation', '_get_inflammation', '_initiate_inflam', '_escalate_inflam',
          '_resolve_inflam', 'update_cytokine', 'update_inflammation'],
         'signaling'),
        (['_connect_bbb', '_connect_bft', '_connect_swarm', '_connect_hr',
          '_connect_bio', '_bft_handler', 'orchestrator_'],
         'connectors'),
        (['_swarm_', 'swarm_'],
         'swarm'),
        (['_imagination', '_kg_self', '_exception_', '_training_',
          '_health_agent_'],
         'integration'),
        (['_update', '_get_stats', '_get_checkpoint', '_get_phase',
          '_is_neutralized', '_get_antigen', 'find_antigen', 'find_b_cell',
          'find_t_cell', 'find_antibody', 'find_inflammation', 'update_immune_phase',
          'process_pending', 'compute_inflammation'],
         'update'),
    ],
    'nimcp_mirror_neurons.c': [
        (['_simulate', '_observe', '_action_obs'],
         'simulation'),
        (['_empathy', '_emotion', '_mirror_emotion'],
         'empathy'),
        (['_learn', '_imitat', '_motor', '_plan'],
         'learning'),
        (['_social', '_theory_of_mind', '_perspective', '_intent'],
         'social'),
    ],
    'nimcp_knowledge.c': [
        (['_graph', '_node', '_edge', '_add_node', '_add_edge', '_remove_node',
          '_remove_edge', '_get_node', '_get_edge'],
         'graph'),
        (['_query', '_search', '_find', '_match', '_retrieve', '_lookup'],
         'query'),
        (['_reason', '_infer', '_deduc', '_integrat'],
         'reasoning'),
    ],
    'nimcp_salience.c': [
        (['_detect', '_signal', '_compute_salience', '_evaluate'],
         'detection'),
        (['_priority', '_urgency', '_rank', '_score'],
         'priority'),
        (['_filter', '_noise', '_select', '_suppress'],
         'filtering'),
        (['_integrat', '_cross_modal', '_fuse', '_combine'],
         'integration'),
    ],
    'nimcp_bio_router.c': [
        (['_dispatch', '_route', '_send', '_deliver', '_forward'],
         'dispatch'),
        (['_queue', '_enqueue', '_dequeue', '_priority'],
         'queue'),
        (['_subscri', '_topic', '_filter', '_register_handler', '_unregister'],
         'subscription'),
    ],
    'nimcp_introspection.c': [
        (['_metacog', '_monitor', '_assess_cognit'],
         'metacog'),
        (['_self_model', '_update_self', '_represent'],
         'self_model'),
        (['_report', '_generate_report', '_summary'],
         'report'),
        (['_aware', '_conscious', '_level'],
         'awareness'),
    ],
    'nimcp_working_memory.c': [
        (['_store', '_add_item', '_capacity', '_slot'],
         'store'),
        (['_retriev', '_get_item', '_decay', '_forget'],
         'retrieval'),
        (['_update', '_refresh', '_maintain', '_rehearse'],
         'update'),
        (['_attention', '_gate', '_focus', '_select'],
         'attention'),
    ],
    'nimcp_brain_training_integration.c': [
        (['_pipeline', '_run_pipeline', '_step'],
         'pipeline'),
        (['_data', '_batch', '_augment', '_load_data', '_prepare'],
         'data'),
        (['_schedule', '_lr_', '_curriculum', '_epoch'],
         'schedule'),
        (['_checkpoint', '_save', '_restore', '_recover'],
         'checkpoint'),
    ],
    'nimcp_hypergraph.c': [
        (['_node', '_edge', '_hyperedge', '_add_', '_remove_', '_connect'],
         'structure'),
        (['_travers', '_path', '_walk', '_bfs', '_dfs', '_shortest'],
         'traversal'),
        (['_reason', '_infer', '_query', '_match'],
         'reasoning'),
    ],
    'nimcp_global_workspace.c': [
        (['_broadcast', '_publish', '_notify', '_distribute'],
         'broadcast'),
        (['_compet', '_coalition', '_bid', '_winner'],
         'competition'),
        (['_access', '_content', '_workspace', '_conscious'],
         'access'),
    ],
    'nimcp_nlp.c': [
        (['_protocol', '_frame', '_message', '_parse', '_format'],
         'protocol'),
        (['_connect', '_handshake', '_accept', '_close', '_disconnect'],
         'connection'),
        (['_transfer', '_send', '_recv', '_spike_batch', '_encode', '_decode'],
         'transfer'),
    ],
    'nimcp_rcog_orchestrator.c': [
        (['_schedule', '_task', '_assign', '_queue'],
         'scheduling'),
        (['_monitor', '_depth', '_loop', '_detect', '_limit'],
         'monitoring'),
        (['_coordinat', '_level', '_cross', '_sync'],
         'coordination'),
    ],
    'nimcp_rcog_engine.c': [
        (['_execute', '_run', '_process', '_step', '_eval'],
         'execution'),
        (['_stack', '_push', '_pop', '_frame', '_context'],
         'stack'),
    ],
    'nimcp_logging.c': [
        (['_output', '_write', '_sink', '_format', '_print'],
         'output'),
        (['_filter', '_level', '_category', '_should_log'],
         'filter'),
        (['_buffer', '_flush', '_async', '_queue'],
         'buffer'),
    ],
    'nimcp.c': [
        (['_brain_create', '_brain_destroy', '_brain_config', '_brain_get',
          '_brain_set', '_get_brain'],
         'brain'),
        (['_train', '_learn', '_fit', '_epoch'],
         'training'),
        (['_infer', '_predict', '_forward', '_process', '_decide'],
         'inference'),
        (['_save', '_load', '_export', '_import', '_checkpoint'],
         'io'),
        (['_stats', '_status', '_monitor', '_version', '_info', '_query',
          '_get_error', '_get_status'],
         'query'),
    ],
    'nimcp_neuromodulators.c': [
        (['_dopamin', '_reward', '_da_'],
         'dopamine'),
        (['_serotonin', '_mood', '_5ht_'],
         'serotonin'),
        (['_acetylcholine', '_ach_', '_attention', '_cholinergic'],
         'acetylcholine'),
        (['_integrat', '_balance', '_interact', '_modulate'],
         'integration'),
    ],
    'nimcp_fep_orchestrator.c': [
        (['_predict', '_error', '_surprise', '_elbo', '_free_energy'],
         'prediction'),
        (['_action', '_active_infer', '_select', '_plan'],
         'action'),
        (['_integrat', '_sensory', '_update_model', '_update_belief'],
         'integration'),
    ],
    'nimcp_wellbeing.c': [
        (['_assess', '_score', '_evaluat', '_measure', '_compute_wellbeing'],
         'assessment'),
        (['_regulat', '_cope', '_strateg', '_intervene', '_adapt'],
         'regulation'),
        (['_monitor', '_track', '_alert', '_check', '_detect'],
         'monitoring'),
    ],
    'nimcp_swarm_brain.c': [
        (['_coordinat', '_sync', '_communicate', '_exchange'],
         'coordination'),
        (['_consensus', '_vote', '_agree', '_decide', '_elect'],
         'consensus'),
        (['_metric', '_health', '_monitor', '_status'],
         'metrics'),
    ],
    'nimcp_plasticity_orchestrator.c': [
        (['_stdp', '_timing', '_spike', '_pre_post', '_ltp', '_ltd'],
         'stdp'),
        (['_homeostat', '_scale', '_normalize', '_balance'],
         'homeostatic'),
    ],
    'nimcp_lnn_gradient.c': [
        (['_compute', '_calc', '_backward'],
         'compute'),
        (['_accumul', '_average', '_sum', '_aggregate'],
         'accumulate'),
        (['_apply', '_update', '_optim', '_step'],
         'apply'),
    ],
    'nimcp_swarm_consciousness.c': [
        (['_phi', '_iit', '_measure', '_compute_phi'],
         'phi'),
        (['_collective', '_aware', '_gestalt', '_emerge'],
         'collective'),
    ],
    'nimcp_distributed_cognition.c': [
        (['_sync', '_synchroniz', '_state_sync', '_replicate'],
         'sync'),
        (['_partition', '_distribute', '_shard', '_assign'],
         'partition'),
    ],
    'nimcp_collective_cognition.c': [
        (['_vote', '_consensus', '_agree', '_decide'],
         'voting'),
        (['_merge', '_combine', '_integrat', '_reconcile'],
         'merge'),
    ],
    'nimcp_language_orchestrator.c': [
        (['_pipeline', '_process', '_tokenize', '_parse'],
         'pipeline'),
        (['_integrat', '_connect', '_bridge', '_coordinate'],
         'integration'),
    ],
    'nimcp_portia.c': [
        (['_plan', '_route', '_allocat', '_strateg'],
         'planning'),
        (['_execut', '_monitor', '_track', '_adapt'],
         'execution'),
    ],
    'nimcp_ethics.c': [
        (['_evaluat', '_judge', '_dilemma', '_resolve', '_decide'],
         'evaluation'),
        (['_constrain', '_rule', '_enforce', '_check', '_violat'],
         'constraints'),
    ],
    'nimcp_systems_consolidation.c': [
        (['_transfer', '_replay', '_strengthen', '_move'],
         'transfer'),
        (['_schedule', '_sleep', '_cycle', '_trigger', '_when'],
         'scheduling'),
    ],
    'nimcp_corrigibility.c': [
        (['_authority', '_permission', '_authorize', '_check_auth',
          '_is_authorized'],
         'authority'),
        (['_monitor', '_detect', '_violat', '_check'],
         'monitoring'),
        (['_shutdown', '_accept', '_resist', '_verify'],
         'shutdown'),
        (['_goal', '_objective', '_align'],
         'goals'),
    ],
}


def classify_function(func_name, filename, specific_rules):
    """Classify a function into a target module suffix."""
    # First try file-specific rules
    basename = os.path.basename(filename)
    if basename in specific_rules:
        for patterns, target in specific_rules[basename]:
            for pattern in patterns:
                if pattern in func_name.lower():
                    return target

    # Then try default rules
    for patterns, target in CLASSIFICATION_RULES['_default']:
        for pattern in patterns:
            if pattern in func_name.lower():
                return target

    return None  # Unclassified - stays in facade


# ============================================================================
# File Splitter
# ============================================================================

def split_file(filepath, dry_run=False):
    """Split a single god file into SRP modules."""
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"  SKIP: {filepath} does not exist")
        return []

    basename = filepath.name
    module_name = basename.replace('.c', '')

    with open(filepath, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines < 500:
        print(f"  SKIP: {filepath} only {total_lines} lines (under threshold)")
        return []

    print(f"\n{'='*70}")
    print(f"Splitting: {filepath} ({total_lines} lines)")
    print(f"{'='*70}")

    # Find all functions
    functions = find_function_definitions(lines)
    print(f"  Found {len(functions)} functions")

    # Classify functions
    groups = defaultdict(list)
    unclassified = []

    for func in functions:
        target = classify_function(func['name'], basename, FILE_SPECIFIC_RULES)
        if target:
            groups[target].append(func)
        else:
            unclassified.append(func)

    # Proximity-based fallback: assign unclassified functions to nearest classified group
    # Functions near each other in source tend to be related
    if unclassified and groups:
        still_unclassified = []
        classified_positions = []
        for group_name, group_funcs in groups.items():
            for func in group_funcs:
                classified_positions.append((func['start'], group_name))
        classified_positions.sort()

        for func in unclassified:
            # Find nearest classified function by line position
            best_group = None
            best_dist = float('inf')
            for pos, gname in classified_positions:
                dist = abs(func['start'] - pos)
                if dist < best_dist:
                    best_dist = dist
                    best_group = gname
            # Only assign if within 100 lines of a classified function
            if best_group and best_dist < 150:
                groups[best_group].append(func)
            else:
                still_unclassified.append(func)
        unclassified = still_unclassified

    print(f"  Groups: {dict((k, len(v)) for k, v in groups.items())}")
    if unclassified:
        print(f"  Unclassified ({len(unclassified)}): {[f['name'] for f in unclassified[:10]]}...")

    # Don't split if fewer than 2 groups
    if len(groups) < 2:
        print(f"  SKIP: Only {len(groups)} groups found, not enough to split")
        return []

    # Determine the directory and file naming
    src_dir = filepath.parent
    internal_header_name = f"{module_name}_internal.h"
    internal_header_path = src_dir / internal_header_name

    # Collect includes from original file (first N lines before any function)
    first_func_line = functions[0]['start'] if functions else total_lines
    header_section = lines[:first_func_line]

    # Extract #include lines and other top-level declarations
    includes = []
    other_top = []
    for line in header_section:
        stripped = line.strip()
        if stripped.startswith('#include'):
            includes.append(line)
        elif stripped.startswith('#define') or stripped.startswith('#ifdef') or \
             stripped.startswith('#ifndef') or stripped.startswith('#endif') or \
             stripped.startswith('#else') or stripped.startswith('#if ') or \
             stripped.startswith('#pragma') or stripped.startswith('#undef'):
            other_top.append(line)
        elif stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*') or stripped == '':
            other_top.append(line)

    # Find the original file's own header include
    own_header = None
    for inc in includes:
        # The module's own public header is usually the first include
        if module_name.replace('nimcp_', '') in inc.lower() or basename.replace('.c', '.h') in inc:
            own_header = inc.strip()
            break

    # Generate the internal header
    created_files = []

    internal_header_content = generate_internal_header(
        module_name, own_header, includes, functions, groups, unclassified)

    if not dry_run:
        # Check if internal header already exists
        if internal_header_path.exists():
            print(f"  Internal header already exists: {internal_header_path}")
            # Don't overwrite existing internal headers
        else:
            with open(internal_header_path, 'w') as f:
                f.write(internal_header_content)
            print(f"  Created: {internal_header_path}")
            created_files.append(str(internal_header_path))

    # Generate split files
    for group_name, group_funcs in sorted(groups.items()):
        if not group_funcs:
            continue

        split_filename = f"{module_name}_{group_name}.c"
        split_path = src_dir / split_filename

        # Check if already exists (from previous session)
        if split_path.exists():
            print(f"  SKIP (exists): {split_path}")
            created_files.append(str(split_path))
            continue

        split_content = generate_split_file(
            module_name, group_name, group_funcs, lines,
            internal_header_name, includes)

        if not dry_run:
            with open(split_path, 'w') as f:
                f.write(split_content)
            print(f"  Created: {split_path} ({len(split_content.splitlines())} lines)")
            created_files.append(str(split_path))
        else:
            print(f"  Would create: {split_path} ({len(split_content.splitlines())} lines)")
            created_files.append(str(split_path))

    # Generate the thin facade (reduced original)
    facade_content = generate_facade(
        module_name, lines, functions, groups, unclassified,
        internal_header_name, header_section)

    facade_path = filepath.parent / f"{module_name}_facade.c"
    if not dry_run:
        # Write facade as a separate file first for safety
        with open(facade_path, 'w') as f:
            f.write(facade_content)
        print(f"  Created facade: {facade_path} ({len(facade_content.splitlines())} lines)")
        created_files.append(str(facade_path))
    else:
        print(f"  Would create facade: {facade_path} ({len(facade_content.splitlines())} lines)")

    return created_files


def generate_internal_header(module_name, own_header, includes, functions, groups, unclassified):
    """Generate the internal header for shared types and declarations."""
    guard = f"{module_name.upper()}_INTERNAL_H"

    content = f"""/**
 * @file {module_name}_internal.h
 * @brief Internal shared declarations for {module_name} SRP split
 * @date 2026-02-16
 *
 * Auto-generated by tools/srp_splitter.py
 * This header is shared between all split implementation files.
 */

#ifndef {guard}
#define {guard}

"""
    # Add the primary include
    if own_header:
        content += f"{own_header}\n"
    content += '#include "utils/memory/nimcp_memory.h"\n'
    content += '#include "utils/logging/nimcp_logging.h"\n'
    content += '#include "utils/exception/nimcp_exception_macros.h"\n'
    content += '\n'

    # Add remaining relevant includes
    seen = set()
    for inc in includes:
        stripped = inc.strip()
        if stripped in seen or not stripped:
            continue
        if own_header and stripped == own_header:
            continue
        if 'nimcp_memory.h' in stripped or 'nimcp_logging.h' in stripped or 'nimcp_exception_macros.h' in stripped:
            continue
        seen.add(stripped)
        content += f"{stripped}\n"

    content += f"""
#ifdef __cplusplus
extern "C" {{
#endif

/* ============================================================================
 * Forward Declarations for Split Modules
 * ============================================================================ */

"""
    # Declare non-static functions from each group
    for group_name, group_funcs in sorted(groups.items()):
        content += f"/* --- {group_name} module --- */\n"
        for func in group_funcs:
            if not func['is_static']:
                # Extract just the declaration from the signature
                sig = func['signature']
                # Clean up the signature
                sig = ' '.join(sig.split())
                if '{' in sig:
                    sig = sig[:sig.index('{')].strip()
                content += f"/* {func['name']} - defined in {module_name}_{group_name}.c */\n"
        content += "\n"

    content += f"""
#ifdef __cplusplus
}}
#endif

#endif /* {guard} */
"""
    return content


def generate_split_file(module_name, group_name, group_funcs, source_lines,
                         internal_header, includes):
    """Generate a split .c file with the grouped functions."""
    content = f"""/**
 * @file {module_name}_{group_name}.c
 * @brief {module_name} - {group_name} responsibility
 * @date 2026-02-16
 *
 * Auto-generated by tools/srp_splitter.py
 * Part of SRP split of {module_name}.c
 */

#include "{internal_header}"

"""
    # Add any additional includes that might be needed
    # We include the internal header which chains to all needed headers

    # Extract function bodies from source
    for func in sorted(group_funcs, key=lambda f: f['start']):
        # Get all lines from comment start to function end
        func_lines = source_lines[func['start']:func['end'] + 1]
        content += ''.join(func_lines)
        content += '\n\n'

    return content


def generate_facade(module_name, source_lines, functions, groups, unclassified,
                     internal_header, header_section):
    """Generate the thin facade file."""
    content = f"""/**
 * @file {module_name}.c
 * @brief {module_name} - Thin facade (SRP refactored)
 * @date 2026-02-16
 *
 * This file has been refactored for Single Responsibility Principle.
 * Implementation is split across:
"""
    for group_name in sorted(groups.keys()):
        content += f" *   - {module_name}_{group_name}.c\n"
    content += f""" *
 * This facade contains:
 *   - Original includes and top-level declarations
 *   - Bridge boilerplate / health agent registration
 *   - Any unclassified/coordinating functions
 *   - Functions too tightly coupled to split
 */

"""
    # Keep the original header section (includes, macros, etc.)
    content += ''.join(header_section)
    content += f'\n#include "{internal_header}"\n\n'

    # Keep unclassified functions in the facade
    if unclassified:
        content += "/* ============================================================================\n"
        content += " * Coordination / Unclassified Functions\n"
        content += " * ============================================================================ */\n\n"
        for func in sorted(unclassified, key=lambda f: f['start']):
            func_lines = source_lines[func['start']:func['end'] + 1]
            content += ''.join(func_lines)
            content += '\n\n'

    return content


# ============================================================================
# God File Registry
# ============================================================================

GOD_FILES = [
    # P1 - Critical (>2000 lines)
    "src/utils/fault_tolerance/nimcp_health_agent.c",
    "src/core/brain/nimcp_brain.c",
    "src/cognitive/omni/nimcp_omni_world_model.c",
    "src/cognitive/immune/nimcp_brain_immune.c",
    "src/cognitive/mirror_neurons/nimcp_mirror_neurons.c",
    "src/api/nimcp.c",
    "src/cognitive/knowledge/nimcp_knowledge.c",
    "src/cognitive/salience/nimcp_salience.c",
    "src/plasticity/neuromodulators/nimcp_neuromodulators.c",
    "src/cognitive/wellbeing/nimcp_wellbeing.c",
    "src/swarm/nimcp_swarm_brain.c",
    "src/lnn/nimcp_lnn_gradient.c",
    "src/plasticity/nimcp_plasticity_orchestrator.c",
    "src/security/nimcp_corrigibility.c",
    "src/cognitive/free_energy/nimcp_fep_orchestrator.c",

    # P2 - Major (1000-2500 lines)
    "src/async/nimcp_bio_router.c",
    "src/cognitive/introspection/nimcp_introspection.c",
    "src/cognitive/working_memory/nimcp_working_memory.c",
    "src/middleware/training/nimcp_brain_training_integration.c",
    "src/cognitive/neuro_symbolic/nimcp_hypergraph.c",
    "src/cognitive/global_workspace/nimcp_global_workspace.c",
    "src/networking/nlp/nimcp_nlp.c",
    "src/cognitive/recursive/nimcp_rcog_orchestrator.c",
    "src/utils/logging/nimcp_logging.c",
    "src/swarm/nimcp_swarm_consciousness.c",
    "src/cognitive/recursive/nimcp_rcog_engine.c",
    "src/networking/distributed/nimcp_distributed_cognition.c",
    "src/cognitive/collective_cognition/nimcp_collective_cognition.c",
    "src/language/nimcp_language_orchestrator.c",
    "src/portia/nimcp_portia.c",
    "src/cognitive/ethics/nimcp_ethics.c",
    "src/cognitive/memory/nimcp_systems_consolidation.c",
]


def generate_cmake_additions(created_files):
    """Generate the CMakeLists.txt additions needed."""
    additions = []
    for f in created_files:
        if f.endswith('.c') and '_facade.c' not in f:
            # Convert to CMakeLists.txt format
            rel = os.path.relpath(f, str(SRC_ROOT / "lib"))
            cmake_path = f"${{CMAKE_CURRENT_SOURCE_DIR}}/{rel}"
            additions.append(cmake_path)
    return additions


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SRP File Splitter')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without creating files')
    parser.add_argument('--file', type=str,
                       help='Split only this specific file')
    parser.add_argument('--list', action='store_true',
                       help='List all god files and their sizes')
    args = parser.parse_args()

    if args.list:
        for gf in GOD_FILES:
            fp = PROJECT_ROOT / gf
            if fp.exists():
                with open(fp) as f:
                    lines = len(f.readlines())
                print(f"  {lines:>6} {gf}")
            else:
                print(f"  MISSING {gf}")
        return

    all_created = []

    if args.file:
        files_to_split = [args.file]
    else:
        files_to_split = GOD_FILES

    for gf in files_to_split:
        filepath = PROJECT_ROOT / gf
        try:
            created = split_file(filepath, dry_run=args.dry_run)
            all_created.extend(created)
        except Exception as e:
            print(f"  ERROR splitting {gf}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Files created: {len(all_created)}")

    # Generate CMakeLists.txt additions
    cmake_adds = generate_cmake_additions(all_created)
    if cmake_adds:
        cmake_file = PROJECT_ROOT / "tools" / "cmake_additions.txt"
        if not args.dry_run:
            with open(cmake_file, 'w') as f:
                f.write("# Add these to src/lib/CMakeLists.txt NIMCP_CORE_SOURCES\n")
                f.write("# Auto-generated by tools/srp_splitter.py\n\n")
                for add in sorted(cmake_adds):
                    f.write(f"    {add}\n")
            print(f"\nCMake additions written to: {cmake_file}")
        else:
            print(f"\nWould write {len(cmake_adds)} CMake additions")

    print("\nNext steps:")
    print("  1. Review the *_facade.c files")
    print("  2. Replace original .c files with their facades")
    print("  3. Add split files to CMakeLists.txt")
    print("  4. Build: cd build && cmake .. && make nimcp -j4")
    print("  5. Fix any compilation errors")
    print("  6. Run tests: ctest -R regression -j3 --timeout 600")


if __name__ == '__main__':
    main()
