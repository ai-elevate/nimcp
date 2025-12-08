#!/usr/bin/env python3
import subprocess
import sys
import os

os.chdir('/home/bbrelin/nimcp')
result = subprocess.run([
    './test/unit_middleware_events_test_event_queue'
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr, file=sys.stderr)
sys.exit(result.returncode)
