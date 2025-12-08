#!/usr/bin/env python3
"""Fix headers missing bio_ctx fields."""

import os
import re

# Headers to fix with their struct names
HEADERS_TO_FIX = {
    'include/cognitive/nimcp_shadow_emotions.h': 'shadow_emotion_system_t',
    'include/cognitive/nimcp_bias_detection.h': 'bias_detection_system_t',
    'include/cognitive/nimcp_empathetic_response.h': 'empathetic_response_system_t',
    'include/cognitive/nimcp_sleep_wake.h': 'sleep_wake_system_t',
    'include/cognitive/nimcp_theory_of_mind.h': 'theory_of_mind_system_t',
    'include/cognitive/nimcp_self_awareness.h': 'self_awareness_system_t',
    'include/cognitive/nimcp_self_model.h': 'self_model_system_t',
    'include/cognitive/nimcp_meta_learning.h': 'meta_learning_system_t',
}

BIO_CTX_FIELDS = """
    // Bio-async integration
    void* bio_ctx;                  /**< bio_module_context_t pointer */
    bool bio_async_enabled;         /**< Bio-async registration status */
"""

def fix_header(filepath, struct_name):
    """Add bio_ctx fields to struct in header."""
    if not os.path.exists(filepath):
        print(f"  File not found: {filepath}")
        return False

    with open(filepath, 'r') as f:
        content = f.read()

    # Check if already has bio_ctx
    if 'bio_ctx' in content:
        print(f"  Already has bio_ctx: {filepath}")
        return False

    # Find the struct ending pattern
    pattern = rf'(\n\}}) ({struct_name});'

    # Add bio_ctx fields before the closing brace
    new_content = re.sub(pattern, BIO_CTX_FIELDS + r'\1 \2;', content)

    if new_content == content:
        print(f"  Pattern not found for {struct_name} in {filepath}")
        return False

    with open(filepath, 'w') as f:
        f.write(new_content)

    print(f"  Fixed: {filepath}")
    return True

def main():
    os.chdir('/home/bbrelin/nimcp')

    fixed = 0
    for header, struct_name in HEADERS_TO_FIX.items():
        print(f"Processing {header}...")
        if fix_header(header, struct_name):
            fixed += 1

    print(f"\nFixed {fixed}/{len(HEADERS_TO_FIX)} headers")

if __name__ == '__main__':
    main()
