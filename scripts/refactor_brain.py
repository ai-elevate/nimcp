#!/usr/bin/env python3
"""
Refactor nimcp_brain.c into 6 smaller modules
"""

import re
import os

# File paths
BRAIN_C = "/home/bbrelin/nimcp/src/core/brain/nimcp_brain.c"
OUTPUT_DIR = "/home/bbrelin/nimcp/src/core/brain"
INCLUDE_DIR = "/home/bbrelin/nimcp/include/core/brain"

# Read the original file
with open(BRAIN_C, 'r') as f:
    lines = f.readlines()

# Common header for all modules
COMMON_HEADER = """//=============================================================================
// {filename} - {description}
//=============================================================================

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/{header_name}"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/memory/nimcp_memory_guards.h"
#include "utils/validation/nimcp_validate.h"
#include "utils/platform/nimcp_platform_mutex.h"
#include "security/nimcp_security.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define LOG_MODULE "BRAIN"

"""

# Function to extract lines between two line numbers
def extract_lines(start, end):
    """Extract lines from start to end (inclusive, 1-indexed)"""
    return ''.join(lines[start-1:end])

# Function to find function boundaries
def find_function_end(start_line):
    """Find the ending line of a function starting at start_line"""
    brace_count = 0
    in_function = False

    for i in range(start_line - 1, len(lines)):
        line = lines[i]

        # Count braces
        for char in line:
            if char == '{':
                brace_count += 1
                in_function = True
            elif char == '}':
                brace_count -= 1

        # If we've closed all braces and we were in a function, we're done
        if in_function and brace_count == 0:
            return i + 1  # Return 1-indexed line number

    return len(lines)

# Define module extractions
modules = {
    "nimcp_brain_core.c": {
        "description": "Brain allocation, initialization, and destruction",
        "functions": [
            ("allocate_brain", 1037),
            ("create_brain_network", 1103),
            ("init_output_labels", 1140),
            ("init_attention_subsystem", 1219),
            ("init_brain_regions_subsystem", 1310),
            ("init_symbolic_logic_subsystem", 1431),
            ("init_symbolic_reasoning_subsystem", 1473),
            ("init_epistemic_subsystem", 1521),
            ("brain_destroy", 1654),
        ],
        "extra_includes": [
            '#include "plasticity/adaptive/nimcp_adaptive.h"',
            '#include "core/neuralnet/nimcp_neuralnet.h"',
            '#include "utils/memory/nimcp_memory.h"',
            '#include "glial/integration/nimcp_glial_integration.h"',
            '#include "cognitive/nimcp_symbolic_logic.h"',
            '#include "cognitive/epistemic/nimcp_epistemic_filter.h"',
            '#include "core/brain_regions/nimcp_brain_regions.h"',
            '#include "core/brain/factory/init/nimcp_brain_init.h"',
        ]
    },

    "nimcp_brain_processing.c": {
        "description": "Brain forward pass and decision processing",
        "functions": [
            ("determine_output_label", 2860),
            ("update_inference_stats", 2915),
        ],
        "extra_includes": [
            '#include "core/neuralnet/nimcp_neuralnet.h"',
            '#include "plasticity/adaptive/nimcp_adaptive.h"',
        ]
    },

    "nimcp_brain_memory.c": {
        "description": "Working memory state save/load operations",
        "functions": [
            ("save_working_memory_state", 4395),
            ("load_working_memory_item", 4602),
            ("load_working_memory_state", 4656),
        ],
        "extra_includes": [
            '#include "cognitive/nimcp_working_memory.h"',
        ]
    },

    "nimcp_brain_state.c": {
        "description": "Brain state accessors and COW handling",
        "functions": [
            ("brain_get_network", 520),
            ("brain_get_neuromodulator_system", 544),
            ("brain_get_sleep_system", 2276),
            ("brain_get_theory_of_mind", 2299),
            ("brain_get_explanation_generator", 2322),
            ("ensure_writable_network", 2349),
        ],
        "extra_includes": [
            '#include "plasticity/neuromodulators/nimcp_neuromodulators.h"',
            '#include "cognitive/nimcp_sleep_wake.h"',
            '#include "cognitive/nimcp_theory_of_mind.h"',
            '#include "cognitive/nimcp_explanations.h"',
        ]
    },

    "nimcp_brain_io.c": {
        "description": "Brain persistence and JSON import/export",
        "functions": [
            ("save_metadata", 4451),
            ("load_metadata", 4718),
            ("ensure_snapshot_dir", 5018),
            ("brain_get_memory_usage", 5067),
            ("brain_import_json", 6787),
            ("brain_save_json", 6793),
            ("brain_load_json", 6894),
        ],
        "extra_includes": [
            '#include <cjson/cJSON.h>',
            '#include <dirent.h>',
            '#include <sys/stat.h>',
            '#include "io/serialization/nimcp_serialization.h"',
        ]
    },

    "nimcp_brain_multimodal.c": {
        "description": "Multimodal sensory processing and integration",
        "functions": [
            ("apply_attention_to_features", 5412),
            ("process_brain_regions", 5480),
            # brain_process_multimodal is around line 6201
        ],
        "extra_includes": [
            '#include "core/integration/nimcp_multimodal_integration.h"',
            '#include "perception/nimcp_visual_cortex.h"',
            '#include "perception/nimcp_audio_cortex.h"',
            '#include "perception/nimcp_speech_cortex.h"',
            '#include "plasticity/attention/nimcp_attention.h"',
            '#include "cognitive/introspection/nimcp_introspection.h"',
            '#include "cognitive/ethics/nimcp_ethics.h"',
            '#include "cognitive/salience/nimcp_salience.h"',
            '#include "cognitive/curiosity/nimcp_curiosity.h"',
        ]
    },
}

# Extract helper/static functions from the beginning
def extract_helpers():
    """Extract bio-async and strategy helper functions"""
    # Lines 189-520 contain helpers
    return extract_lines(189, 519)

helpers = extract_helpers()

# Generate each module
for module_name, config in modules.items():
    print(f"Generating {module_name}...")

    # Start with header
    header_name = module_name.replace('.c', '.h')
    content = COMMON_HEADER.format(
        filename=module_name,
        description=config["description"],
        header_name=header_name
    )

    # Add extra includes
    if "extra_includes" in config:
        for inc in config["extra_includes"]:
            content += inc + "\n"
        content += "\n"

    # For core module, include helpers
    if module_name == "nimcp_brain_core.c":
        content += "// Bio-Async and Strategy Helpers\n"
        content += helpers + "\n\n"

    # Extract each function
    for func_name, start_line in config["functions"]:
        end_line = find_function_end(start_line)
        func_code = extract_lines(start_line, end_line)
        content += f"// {func_name}\n"
        content += func_code + "\n\n"

    # Write module file
    output_path = os.path.join(OUTPUT_DIR, module_name)
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"  Generated {output_path}")

print("\nModule generation complete!")
print("\nNow generating header files...")

# Generate header files
for module_name in modules.keys():
    header_name = module_name.replace('.c', '.h')
    guard_name = f"NIMCP_BRAIN_{header_name.replace('.h', '').replace('nimcp_brain_', '').upper()}_H"

    header_content = f"""//=============================================================================
// {header_name} - Module interface
//=============================================================================

#ifndef {guard_name}
#define {guard_name}

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"

// Function declarations will be added based on module needs

#endif // {guard_name}
"""

    header_path = os.path.join(INCLUDE_DIR, header_name)
    with open(header_path, 'w') as f:
        f.write(header_content)

    print(f"  Generated {header_path}")

print("\nRefactoring script complete!")
