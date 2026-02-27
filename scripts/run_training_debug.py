#!/usr/bin/env python3
"""Wrapper to catch signals that kill training."""
import signal, sys, os, subprocess

def handler(signum, frame):
    try:
        signame = signal.Signals(signum).name
    except (ValueError, AttributeError):
        signame = str(signum)
    with open("/home/bbrelin/nimcp/training_death.log", "a") as f:
        f.write(f"CAUGHT SIGNAL: {signame} ({signum})\n")
    print(f"CAUGHT SIGNAL: {signame} ({signum})", flush=True)
    sys.exit(128 + signum)

for s in [signal.SIGTERM, signal.SIGHUP, signal.SIGINT, signal.SIGUSR1, signal.SIGUSR2]:
    signal.signal(s, handler)

os.chdir("/home/bbrelin/nimcp")
proc = subprocess.Popen(
    [sys.executable, "scripts/train_athena.py"],
    stdout=open("athena_training.log", "w"),
    stderr=subprocess.STDOUT,
)
try:
    rc = proc.wait()
    with open("training_death.log", "a") as f:
        f.write(f"NORMAL EXIT: code={rc}\n")
    print(f"Training exited with code {rc}", flush=True)
except Exception as e:
    with open("training_death.log", "a") as f:
        f.write(f"EXCEPTION: {e}\n")
    sys.exit(1)
