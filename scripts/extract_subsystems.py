#!/usr/bin/env python3
"""Extract subsystems and security sections from nimcp_brain_init.c"""

import sys

def main():
    input_file = "/home/bbrelin/nimcp/src/core/brain/factory/init/nimcp_brain_init.c"

    # Read header includes (lines 29-135)
    with open(input_file, 'r') as f:
        lines = f.readlines()

    # Extract subsystems section (lines 803-3821)
    subsystems_content = ''.join(lines[802:3821])

    # Extract security section (lines 3822-4094)
    security_content = ''.join(lines[3821:4094])

    with open("/tmp/subsystems_extract.c", 'w') as f:
        f.write(subsystems_content)

    with open("/tmp/security_extract.c", 'w') as f:
        f.write(security_content)

    print("Extracted subsystems ({} lines) and security ({} lines)".format(
        len(lines[802:3821]), len(lines[3821:4094])))

if __name__ == "__main__":
    main()
