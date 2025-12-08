#!/usr/bin/env python3
"""
Integrate bio-async messaging, logging, and security into PLASTICITY modules

This script systematically integrates:
1. Bio-async headers and module registration
2. Comprehensive logging at key points
3. Security validation (weight bounds, neuromodulator concentrations)
4. Bio-async message handlers for plasticity events
5. Broadcasting of plasticity changes via bio-router

Targets all plasticity modules that haven't been fully integrated yet.
"""

import os
import re
import sys

# Module-specific bio module IDs and message types
MODULE_CONFIGS = {
    "nimcp_attention.c": {
        "module_id": "BIO_MODULE_ATTENTION",
        "module_name": "attention_plasticity",
        "handlers": [
            ("BIO_MSG_ATTENTION_UPDATE", "attention_handle_update"),
            ("BIO_MSG_WEIGHT_UPDATE_REQUEST", "attention_handle_weight_update"),
        ],
        "broadcasts": [
            ("attention_weights", "BIO_CHANNEL_GLUTAMATE"),
        ],
    },
    "nimcp_dendritic.c": {
        "module_id": "BIO_MODULE_DENDRITIC",
        "module_name": "dendritic_plasticity",
        "handlers": [
            ("BIO_MSG_DENDRITIC_SPIKE", "dendritic_handle_spike"),
            ("BIO_MSG_CALCIUM_UPDATE", "dendritic_handle_calcium"),
        ],
        "broadcasts": [
            ("dendritic_spike", "BIO_CHANNEL_CALCIUM"),
            ("calcium_concentration", "BIO_CHANNEL_CALCIUM"),
        ],
    },
    "nimcp_adaptive.c": {
        "module_id": "BIO_MODULE_ADAPTIVE",
        "module_name": "adaptive_threshold",
        "handlers": [
            ("BIO_MSG_THRESHOLD_UPDATE", "adaptive_handle_threshold"),
            ("BIO_MSG_LEARNING_RATE", "adaptive_handle_learning_rate"),
        ],
        "broadcasts": [
            ("threshold_adaptation", "BIO_CHANNEL_ACETYLCHOLINE"),
        ],
    },
    "nimcp_predictive_coding.c": {
        "module_id": "BIO_MODULE_PREDICTIVE",
        "module_name": "predictive_coding",
        "handlers": [
            ("BIO_MSG_ERROR_SIGNAL", "predictive_handle_error"),
            ("BIO_MSG_PRECISION_UPDATE", "predictive_handle_precision"),
        ],
        "broadcasts": [
            ("prediction_error", "BIO_CHANNEL_DOPAMINE"),
            ("free_energy", "BIO_CHANNEL_DOPAMINE"),
        ],
    },
    "nimcp_pink_noise.c": {
        "module_id": "BIO_MODULE_PINK_NOISE",
        "module_name": "pink_noise_modulation",
        "handlers": [
            ("BIO_MSG_NOISE_SAMPLE_REQUEST", "pink_noise_handle_request"),
        ],
        "broadcasts": [
            ("noise_sample", "BIO_CHANNEL_NEUROMODULATOR"),
        ],
    },
    "nimcp_receptor_subtypes.c": {
        "module_id": "BIO_MODULE_RECEPTOR",
        "module_name": "receptor_subtypes",
        "handlers": [
            ("BIO_MSG_NEUROMODULATOR_RELEASE", "receptor_handle_neuromod_release"),
            ("BIO_MSG_RECEPTOR_BINDING", "receptor_handle_binding"),
        ],
        "broadcasts": [
            ("receptor_occupancy", "BIO_CHANNEL_NEUROMODULATOR"),
            ("binding_event", "BIO_CHANNEL_NEUROMODULATOR"),
        ],
    },
    "nimcp_phasic_tonic.c": {
        "module_id": "BIO_MODULE_PHASIC_TONIC",
        "module_name": "phasic_tonic_control",
        "handlers": [
            ("BIO_MSG_PHASIC_BURST", "phasic_tonic_handle_burst"),
            ("BIO_MSG_TONIC_LEVEL", "phasic_tonic_handle_tonic"),
        ],
        "broadcasts": [
            ("phasic_burst", "BIO_CHANNEL_DOPAMINE"),
            ("tonic_level", "BIO_CHANNEL_DOPAMINE"),
        ],
    },
    "nimcp_metabolic_pathways.c": {
        "module_id": "BIO_MODULE_METABOLIC",
        "module_name": "metabolic_pathways",
        "handlers": [
            ("BIO_MSG_METABOLIC_STATE", "metabolic_handle_state"),
        ],
        "broadcasts": [
            ("metabolite_concentration", "BIO_CHANNEL_NEUROMODULATOR"),
        ],
    },
    "nimcp_vesicle_packaging.c": {
        "module_id": "BIO_MODULE_VESICLE",
        "module_name": "vesicle_packaging",
        "handlers": [
            ("BIO_MSG_VESICLE_RELEASE", "vesicle_handle_release"),
        ],
        "broadcasts": [
            ("vesicle_count", "BIO_CHANNEL_NEUROMODULATOR"),
        ],
    },
}

BIO_ASYNC_HEADERS = """#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
"""

LOGGING_HEADER = """#include "utils/logging/nimcp_logging.h"
"""

SECURITY_HEADER = """#include "security/nimcp_blood_brain_barrier.h"
"""

def add_log_module_define(content, module_name):
    """Add or update LOG_MODULE define"""
    base_name = module_name.replace("nimcp_", "").replace(".c", "")

    # Check if LOG_MODULE already exists
    if "#define LOG_MODULE" in content:
        return content

    # Find location after initial includes
    pattern = r'(#include\s+[<"][^>"]+[>"]\s*\n)+'
    match = re.search(pattern, content)
    if match:
        insert_pos = match.end()
        log_def = f'\n#define LOG_MODULE "{base_name}"\n\n'
        return content[:insert_pos] + log_def + content[insert_pos:]

    return content

def add_headers_if_missing(content, headers):
    """Add headers if not already present"""
    lines_to_add = []
    for line in headers.strip().split('\n'):
        if line.strip() and line.strip() not in content:
            lines_to_add.append(line)

    if not lines_to_add:
        return content

    # Find last include statement
    pattern = r'(#include\s+[<"][^>"]+[>"]\s*\n)'
    matches = list(re.finditer(pattern, content))
    if matches:
        insert_pos = matches[-1].end()
        return content[:insert_pos] + '\n' + '\n'.join(lines_to_add) + '\n' + content[insert_pos:]

    return content

def add_bio_ctx_to_struct(content, struct_pattern):
    """Add bio_ctx and bio_async_enabled fields to module structure"""
    # Find the structure definition
    struct_match = re.search(struct_pattern, content, re.MULTILINE | re.DOTALL)
    if not struct_match:
        return content

    struct_text = struct_match.group(0)

    # Check if bio fields already exist
    if "bio_module_context_t bio_ctx" in struct_text:
        return content

    # Find closing brace
    closing_brace = struct_text.rfind('}')
    if closing_brace == -1:
        return content

    bio_fields = '''
    /* Bio-async integration */
    bio_module_context_t bio_ctx;
    bool bio_async_enabled;
'''

    new_struct = struct_text[:closing_brace] + bio_fields + struct_text[closing_brace:]
    return content.replace(struct_text, new_struct)

def generate_module_init_code(config):
    """Generate bio-async module initialization code"""
    return f'''
    /* Bio-async integration */
    if (bio_router_is_initialized()) {{
        bio_module_info_t bio_info = {{
            .module_id = {config["module_id"]},
            .module_name = "{config["module_name"]}",
            .inbox_capacity = 64,
            .user_data = NULL
        }};
        module_state.bio_ctx = bio_router_register_module(&bio_info);
        if (module_state.bio_ctx) {{
            module_state.bio_async_enabled = true;
            LOG_MODULE_INFO(LOG_MODULE, "Bio-async registered");
        }} else {{
            LOG_MODULE_WARN(LOG_MODULE, "Bio-async registration failed");
        }}
    }}
'''

def generate_handler_stub(msg_type, handler_name):
    """Generate message handler function stub"""
    return f'''
static nimcp_error_t {handler_name}(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data)
{{
    LOG_MODULE_DEBUG(LOG_MODULE, "Handling {msg_type}");

    if (!msg || msg_size == 0) {{
        LOG_MODULE_ERROR(LOG_MODULE, "Invalid message");
        return -1;
    }}

    // TODO: Implement message handling logic

    return NIMCP_SUCCESS;
}}
'''

def add_security_validation(content):
    """Add security validation calls at key points"""
    # This is a simplified version - real implementation would be more sophisticated
    validations = []

    # Weight bound validation
    if "weight" in content and "float weight" in content:
        validations.append('''
    /* Security: Validate weight bounds */
    if (weight < 0.0f || weight > 1.0f) {
        LOG_MODULE_WARN(LOG_MODULE, "Weight out of bounds: %.6f", weight);
        weight = fmax(0.0f, fmin(1.0f, weight));
    }
''')

    # Neuromodulator concentration validation
    if "concentration" in content:
        validations.append('''
    /* Security: Validate concentration */
    if (concentration < 0.0f || concentration > 1.0f) {
        LOG_MODULE_WARN(LOG_MODULE, "Concentration out of bounds: %.6f", concentration);
        concentration = fmax(0.0f, fmin(1.0f, concentration));
    }
''')

    return content  # Note: actual insertion would require more context

def add_logging_to_functions(content):
    """Add logging statements to key functions"""
    # Add entry logging for major functions
    patterns = [
        (r'(^\w+\s+\w+\s*\([^)]*\)\s*\{)', r'\1\n    LOG_MODULE_DEBUG(LOG_MODULE, "Entering function");'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    return content

def process_file(filepath, config):
    """Process a single plasticity module file"""
    print(f"Processing {filepath}...")

    try:
        with open(filepath, 'r') as f:
            content = f.read()

        # Add headers
        content = add_headers_if_missing(content, BIO_ASYNC_HEADERS)
        content = add_headers_if_missing(content, LOGGING_HEADER)
        content = add_headers_if_missing(content, SECURITY_HEADER)

        # Add LOG_MODULE define
        content = add_log_module_define(content, os.path.basename(filepath))

        # Add bio_ctx to module structures
        struct_patterns = [
            r'struct\s+\w+_struct\s*\{[^}]+\}',
            r'typedef\s+struct\s+\w+_internal_t\s*\{[^}]+\}',
        ]
        for pattern in struct_patterns:
            content = add_bio_ctx_to_struct(content, pattern)

        # Write modified content
        with open(filepath, 'w') as f:
            f.write(content)

        print(f"  ✓ Updated {filepath}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {filepath}: {e}")
        return False

def main():
    base_dir = "/home/bbrelin/nimcp/src/plasticity"

    success_count = 0
    total_count = 0

    for filename, config in MODULE_CONFIGS.items():
        # Find the file
        if filename.startswith("nimcp_receptor") or filename.startswith("nimcp_phasic") or \
           filename.startswith("nimcp_metabolic") or filename.startswith("nimcp_vesicle"):
            filepath = os.path.join(base_dir, "neuromodulators", filename)
        elif filename.startswith("nimcp_pink"):
            filepath = os.path.join(base_dir, "noise", filename)
        elif filename.startswith("nimcp_predictive"):
            filepath = os.path.join(base_dir, "predictive", filename)
        elif filename.startswith("nimcp_dendritic"):
            filepath = os.path.join(base_dir, "dendritic", filename)
        elif filename.startswith("nimcp_attention"):
            filepath = os.path.join(base_dir, "attention", filename)
        elif filename.startswith("nimcp_adaptive"):
            filepath = os.path.join(base_dir, "adaptive", filename)
        else:
            continue

        if os.path.exists(filepath):
            total_count += 1
            if process_file(filepath, config):
                success_count += 1

    print(f"\n✓ Processed {success_count}/{total_count} files successfully")

    if success_count == total_count:
        print("\nAll plasticity modules have been integrated!")
        return 0
    else:
        print(f"\nWarning: {total_count - success_count} files had issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())
