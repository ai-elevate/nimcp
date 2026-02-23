#!/bin/bash
# Athena Training Launcher
# Drops all C-level stderr noise, keeps only Python stdout
# Usage: nohup bash scripts/run_athena.sh &

cd /home/bbrelin/nimcp
export PYTHONPATH=build/lib/python:frontend/backend:scripts

LOG_DIR=logs
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/athena_training_$(date +%Y%m%d_%H%M%S).log"

echo "Athena training started at $(date)" | tee "$LOGFILE"
echo "Log: $LOGFILE" | tee -a "$LOGFILE"
echo "PID: $$" | tee -a "$LOGFILE"

# stderr -> /dev/null (drops all C-level noise)
# stdout -> tee to log file
python3 -u scripts/train_athena.py 2>/dev/null | tee -a "$LOGFILE"

echo "Athena training finished at $(date)" | tee -a "$LOGFILE"
