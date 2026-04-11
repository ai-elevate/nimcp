#!/bin/bash
# Sync latest checkpoint from RunPod to dev server
# Runs via cron every 2 hours
# Copies the main checkpoint + all sidecars

RKEY="$HOME/.ssh/id_ed25519_runpod"
RHOST="root@74.2.96.55"
RPORT=17653
REMOTE_DIR="/workspace/nimcp/checkpoints/athena"
LOCAL_DIR="$HOME/nimcp/checkpoints/athena"
LOG="$HOME/nimcp/sync_checkpoint.log"

mkdir -p "$LOCAL_DIR"

# Find the most recent auto-save on RunPod
LATEST=$(ssh -i "$RKEY" -p "$RPORT" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
    "$RHOST" "ls -t $REMOTE_DIR/athena_auto_*.bin 2>/dev/null | head -1" 2>/dev/null)

if [ -z "$LATEST" ]; then
    echo "$(date): No auto-saves found on RunPod" >> "$LOG"
    exit 0
fi

BASENAME=$(basename "$LATEST")

# Check if we already have this version
if [ -f "$LOCAL_DIR/$BASENAME" ]; then
    REMOTE_SIZE=$(ssh -i "$RKEY" -p "$RPORT" -o StrictHostKeyChecking=no \
        "$RHOST" "stat -c%s $LATEST" 2>/dev/null)
    LOCAL_SIZE=$(stat -c%s "$LOCAL_DIR/$BASENAME" 2>/dev/null)
    if [ "$REMOTE_SIZE" = "$LOCAL_SIZE" ]; then
        echo "$(date): Already synced $BASENAME ($LOCAL_SIZE bytes)" >> "$LOG"
        exit 0
    fi
fi

echo "$(date): Syncing $BASENAME from RunPod..." >> "$LOG"

# Sync core + all sidecars
scp -i "$RKEY" -P "$RPORT" -o StrictHostKeyChecking=no -o Compression=yes \
    "$RHOST:$LATEST" "$RHOST:$LATEST.*" \
    "$LOCAL_DIR/" >> "$LOG" 2>&1

if [ $? -eq 0 ]; then
    SIZE=$(du -sh "$LOCAL_DIR/$BASENAME" | cut -f1)
    SIDECARS=$(ls "$LOCAL_DIR/$BASENAME".* 2>/dev/null | wc -l)
    echo "$(date): Synced $BASENAME ($SIZE, $SIDECARS sidecars)" >> "$LOG"
else
    echo "$(date): SYNC FAILED for $BASENAME" >> "$LOG"
fi
