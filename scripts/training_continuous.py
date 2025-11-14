#!/usr/bin/env python3
"""
NIMCP Continuous Training - Direct Python Wrapper
Runs streaming training indefinitely with automatic restarts
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Change to nimcp root directory (parent of scripts)
os.chdir(Path(__file__).parent.parent)

SESSION_COUNT = 0
TOTAL_EXAMPLES = 0

print("=" * 60)
print("🌊 NIMCP CONTINUOUS TRAINING STARTED")
print("=" * 60)
print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PID: {os.getpid()}")
print(f"Target: 2,000,000 examples (Phase 1)")
print(f"Session size: 50,000 examples (~20 minutes)")
print("=" * 60)
print()

# Save PID for easy stopping later
with open('training.pid', 'w') as f:
    f.write(str(os.getpid()))

# Loop indefinitely
while True:
    SESSION_COUNT += 1
    TOTAL_EXAMPLES += 50000

    print()
    print("=" * 60)
    print(f"📊 SESSION {SESSION_COUNT}")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cumulative examples: {TOTAL_EXAMPLES}")
    print(f"Phase 1 progress: {TOTAL_EXAMPLES / 2000000 * 100:.2f}%")
    print("=" * 60)

    # Run training session (script is in scripts directory)
    try:
        result = subprocess.run(
            [sys.executable, 'scripts/start_streaming.py'],
            check=True
        )

        if result.returncode == 0:
            print(f"✅ Session {SESSION_COUNT} complete")
        else:
            print(f"⚠️  Session {SESSION_COUNT} failed with exit code {result.returncode}")
            print("Retrying in 10 seconds...")
            time.sleep(10)

    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        break
    except Exception as e:
        print(f"⚠️  Session {SESSION_COUNT} error: {e}")
        print("Retrying in 10 seconds...")
        time.sleep(10)

    # Short pause between sessions
    time.sleep(2)

print("\n🛑 Training stopped")
