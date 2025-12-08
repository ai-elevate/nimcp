#!/usr/bin/env python3
import subprocess
import os

os.chdir('/home/bbrelin/nimcp/build')

# Run cmake first
print("Configuring...")
result = subprocess.run(['cmake', '..'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Build all
print("\nBuilding all targets...")
result = subprocess.run(['make', '-j4'], capture_output=True, text=True)
print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
if result.stderr:
    print("STDERR:", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)

# Run specific test
print("\nRunning event queue test...")
result = subprocess.run(['ctest', '-R', 'event_queue', '-V'], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

print(f"\nExit code: {result.returncode}")
