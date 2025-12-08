#!/usr/bin/env python3
"""
Integrate bio-async, logging, and unified memory into brain modules
"""

import re
import sys
from pathlib import Path

def add_includes(content, log_module):
    """Add bio-async, logging, and unified memory includes"""

    # Check if already has bio-async
    if 'nimcp_bio_async.h' in content:
        return content

    # Find first #include statement
    include_pattern = r'^(#include\s+[<"])'
    match = re.search(include_pattern, content, re.MULTILINE)

    if not match:
        print("Warning: No includes found")
        return content

    # Insert our includes before the first include
    pos = match.start()

    new_includes = f"""// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

// Logging integration
#include "utils/logging/nimcp_logging.h"

// Unified memory integration
#include "utils/memory/nimcp_unified_memory.h"

"""

    content = content[:pos] + new_includes + content[pos:]

    # Add LOG_MODULE if not present
    if '#define LOG_MODULE' not in content:
        # Find end of includes
        last_include = list(re.finditer(r'^#include.*$', content, re.MULTILINE))[-1]
        insert_pos = last_include.end()

        # Find next non-empty, non-comment line
        next_line_match = re.search(r'\n\n', content[insert_pos:])
        if next_line_match:
            insert_pos += next_line_match.end() - 1

        log_define = f'\n#define LOG_MODULE "{log_module}"\n'
        content = content[:insert_pos] + log_define + content[insert_pos:]

    return content

def replace_malloc_calls(content):
    """Replace malloc/calloc/free with unified memory versions"""

    # Replace malloc (not inside comments or already nimcp_malloc)
    content = re.sub(r'(?<!nimcp_)(?<![a-zA-Z_])malloc\s*\(', 'nimcp_malloc(', content)

    # Replace calloc (not inside comments or already nimcp_calloc)
    content = re.sub(r'(?<!nimcp_)(?<![a-zA-Z_])calloc\s*\(', 'nimcp_calloc(', content)

    # Replace free (not inside comments or already nimcp_free)
    # More careful with free as it's a common word
    content = re.sub(r'(?<!nimcp_)(?<![a-zA-Z_])free\s*\(', 'nimcp_free(', content)

    return content

def add_logging_to_function(content, func_name, log_module):
    """Add basic logging to a function entry"""

    # Find function definition
    func_pattern = rf'(bool|void|int|nimcp_error_t|nimcp_result_t)\s+{re.escape(func_name)}\s*\([^)]*\)\s*\{{'
    match = re.search(func_pattern, content)

    if not match:
        return content

    # Check if LOG_DEBUG already present
    func_start = match.end()
    next_100_chars = content[func_start:func_start+100]
    if 'LOG_DEBUG' in next_100_chars or 'LOG_INFO' in next_100_chars:
        return content

    # Find first real code line (skip whitespace and comments)
    code_start_match = re.search(r'\n\s*\n\s*', content[func_start:])
    if code_start_match:
        insert_pos = func_start + code_start_match.end() - 2
    else:
        insert_pos = func_start

    # Insert logging
    log_line = f'\n    LOG_DEBUG("{func_name} called");\n'
    content = content[:insert_pos] + log_line + content[insert_pos:]

    return content

def process_file(filepath, log_module, functions_to_log=None):
    """Process a single brain module file"""

    print(f"Processing {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    # Add includes
    content = add_includes(content, log_module)

    # Replace malloc/calloc/free
    content = replace_malloc_calls(content)

    # Add logging to specific functions if provided
    if functions_to_log:
        for func_name in functions_to_log:
            content = add_logging_to_function(content, func_name, log_module)

    # Write back
    with open(filepath, 'w') as f:
        f.write(content)

    print(f"✓ Integrated {filepath}")

def main():
    # Define files to process with their log modules and key functions
    files_to_process = [
        {
            'path': 'src/core/brain/accessors/nimcp_brain_accessors.c',
            'log_module': 'BRAIN_ACCESSORS',
            'functions': []
        },
        {
            'path': 'src/core/brain/information/nimcp_brain_shannon.c',
            'log_module': 'BRAIN_INFO',
            'functions': []
        },
        {
            'path': 'src/core/brain/cognitive/nimcp_brain_cognitive.c',
            'log_module': 'BRAIN_COGNITIVE',
            'functions': []
        },
        {
            'path': 'src/core/brain/pretrained/nimcp_brain_pretrained.c',
            'log_module': 'BRAIN_PRETRAINED',
            'functions': []
        },
        {
            'path': 'src/core/brain/processing/cognitive_processor.c',
            'log_module': 'BRAIN_PROC_COG',
            'functions': []
        },
        {
            'path': 'src/core/brain/processing/multimodal_integrator.c',
            'log_module': 'BRAIN_PROC_MM',
            'functions': []
        },
        {
            'path': 'src/core/brain/processing/sensory_extractor.c',
            'log_module': 'BRAIN_PROC_SENS',
            'functions': []
        },
        {
            'path': 'src/core/brain/regions/broca/nimcp_language_production_bridge.c',
            'log_module': 'BROCA_LANG_PROD',
            'functions': []
        },
        {
            'path': 'src/core/brain/regions/broca/nimcp_syntax_processor.c',
            'log_module': 'BROCA_SYNTAX',
            'functions': []
        },
        {
            'path': 'src/core/brain/regions/broca/nimcp_speech_motor.c',
            'log_module': 'BROCA_SPEECH',
            'functions': []
        },
        {
            'path': 'src/core/brain/regions/broca/nimcp_phonological.c',
            'log_module': 'BROCA_PHONO',
            'functions': []
        },
        {
            'path': 'src/core/brain/learning/nimcp_circuit_compilation.c',
            'log_module': 'BRAIN_LEARN_CIRC',
            'functions': []
        },
        {
            'path': 'src/core/brain/learning/nimcp_reasoning_learning.c',
            'log_module': 'BRAIN_LEARN_REASON',
            'functions': []
        },
        {
            'path': 'src/core/brain/learning/nimcp_association_learning.c',
            'log_module': 'BRAIN_LEARN_ASSOC',
            'functions': []
        },
        {
            'path': 'src/core/brain/learning/nimcp_rule_learning.c',
            'log_module': 'BRAIN_LEARN_RULE',
            'functions': []
        },
        {
            'path': 'src/core/brain/factory/init/nimcp_brain_init.c',
            'log_module': 'BRAIN_INIT',
            'functions': []
        },
        {
            'path': 'src/core/brain/factory/validation/nimcp_brain_validation.c',
            'log_module': 'BRAIN_VALID',
            'functions': []
        },
    ]

    base_path = Path('/home/bbrelin/nimcp')

    for file_info in files_to_process:
        filepath = base_path / file_info['path']
        if filepath.exists():
            process_file(filepath, file_info['log_module'], file_info['functions'])
        else:
            print(f"Warning: File not found: {filepath}")

if __name__ == '__main__':
    main()
