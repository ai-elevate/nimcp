#!/usr/bin/env python3
import subprocess
import sys
import os

def run_command(cmd, cwd=None, description=""):
    """Run a command and print output"""
    if description:
        print(f"\n=== {description} ===")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    return True

# Change to build directory
build_dir = '/home/bbrelin/nimcp/build'
os.chdir(build_dir)

# Reconfigure
if not run_command(['cmake', '..'], cwd=build_dir, description="Reconfiguring CMake"):
    sys.exit(1)

# Build middleware library
if not run_command(['make', 'nimcp_middleware', f'-j{os.cpu_count()}'],
                   cwd=build_dir, description="Building middleware library"):
    sys.exit(1)

# Build the test
if not run_command(['make', 'unit_middleware_events_event_queue', f'-j{os.cpu_count()}'],
                   cwd=build_dir, description="Building event queue test"):
    sys.exit(1)

# Run the test
os.chdir('/home/bbrelin/nimcp')
test_path = './test/unit_middleware_events_test_event_queue'
if not run_command([test_path], cwd='/home/bbrelin/nimcp',
                   description="Running event queue test"):
    print("\n*** TEST FAILED ***")
    sys.exit(1)

print("\n=== ALL TESTS PASSED ===")
