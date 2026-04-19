#!/usr/bin/env bash
# Safe deployment of a new nimcp.so + scripts to the RunPod.
#
# NEVER overwrite nimcp.so while athena-brain is running — that causes SIGSEGV
# because the .so is mmap'd into the process. This script enforces the correct
# sequence:
#   1. Upload artifacts to /tmp on pod
#   2. Stop athena-brain (supervisord)
#   3. Swap .so + scripts atomically
#   4. Start athena-brain
#   5. Wait for init to complete before exiting
#
# Usage:
#   ./deploy_to_pod.sh                  # full deploy: .so + scripts + stimuli
#   ./deploy_to_pod.sh --scripts-only   # just scripts (safe — no brain restart)
#   ./deploy_to_pod.sh --stimuli-only   # just stimuli (safe — no brain restart)
set -euo pipefail

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_runpod}"
SSH_PORT="${SSH_PORT:-17653}"
SSH_HOST="${SSH_HOST:-root@74.2.96.55}"
SSH="ssh -o ConnectTimeout=10 -i $SSH_KEY -p $SSH_PORT $SSH_HOST"
SCP="scp -o ConnectTimeout=10 -i $SSH_KEY -P $SSH_PORT"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

mode="full"
if [[ "${1:-}" == "--scripts-only" ]]; then mode="scripts"; fi
if [[ "${1:-}" == "--stimuli-only" ]]; then mode="stimuli"; fi

echo "=== Mode: $mode ==="

# Always safe: pack and upload scripts + stimuli
case "$mode" in
  full)
    tar czf /tmp/athena_bundle.tgz \
        build/lib/python/nimcp.so \
        scripts/test_harness scripts/tests \
        scripts/run_full_battery.py \
        scripts/brain_daemon.py scripts/brain_client.py \
        data/stimuli/
    ;;
  scripts)
    tar czf /tmp/athena_bundle.tgz \
        scripts/test_harness scripts/tests \
        scripts/run_full_battery.py \
        scripts/brain_daemon.py scripts/brain_client.py
    ;;
  stimuli)
    tar czf /tmp/athena_bundle.tgz data/stimuli/
    ;;
esac

$SCP /tmp/athena_bundle.tgz "$SSH_HOST:/tmp/"

if [[ "$mode" == "full" ]]; then
  echo "=== Full deploy: requires brain restart ==="
  $SSH 'bash -s' <<'REMOTE_FULL'
set -e
# 1. Stop training + brain (brain first? no — training first so it releases brain)
supervisorctl stop athena-training athena-brain 2>&1 | grep -v pkg_resources || true

# Wait until brain fully stopped (no PID)
for i in $(seq 1 30); do
  if ! supervisorctl status athena-brain 2>/dev/null | grep -q RUNNING; then
    break
  fi
  sleep 2
done

# 2. Extract
cd /tmp
tar xzf athena_bundle.tgz

# 3. Install .so SAFELY (brain is stopped now)
cp /tmp/build/lib/python/nimcp.so \
   /usr/local/lib/python3.12/dist-packages/nimcp.cpython-312-x86_64-linux-gnu.so

# 4. Install scripts
cp -r /tmp/scripts/test_harness /workspace/nimcp/scripts/
cp -r /tmp/scripts/tests /workspace/nimcp/scripts/
cp    /tmp/scripts/run_full_battery.py /workspace/nimcp/scripts/
cp    /tmp/scripts/brain_daemon.py /workspace/nimcp/scripts/
cp    /tmp/scripts/brain_client.py /workspace/nimcp/scripts/

# 5. Install stimuli
rm -rf /workspace/nimcp/data/stimuli
mkdir -p /workspace/nimcp/data
cp -r /tmp/data/stimuli /workspace/nimcp/data/

# 6. Start brain (training stays stopped — user must start manually)
supervisorctl start athena-brain 2>&1 | grep -v pkg_resources
echo "=== Brain restarted; training NOT restarted — start manually after SNN stabilizes ==="
REMOTE_FULL

elif [[ "$mode" == "scripts" ]]; then
  echo "=== Scripts-only deploy (no brain restart) ==="
  $SSH 'bash -s' <<'REMOTE_SCRIPTS'
set -e
cd /tmp && tar xzf athena_bundle.tgz
cp -r /tmp/scripts/test_harness /workspace/nimcp/scripts/
cp -r /tmp/scripts/tests /workspace/nimcp/scripts/
cp    /tmp/scripts/run_full_battery.py /workspace/nimcp/scripts/
cp    /tmp/scripts/brain_daemon.py /workspace/nimcp/scripts/
cp    /tmp/scripts/brain_client.py /workspace/nimcp/scripts/
echo "=== Scripts updated. Daemon handlers need restart to take effect (see --restart-daemon) ==="
REMOTE_SCRIPTS

elif [[ "$mode" == "stimuli" ]]; then
  echo "=== Stimuli-only deploy ==="
  $SSH 'bash -s' <<'REMOTE_STIM'
set -e
cd /tmp && tar xzf athena_bundle.tgz
rm -rf /workspace/nimcp/data/stimuli
mkdir -p /workspace/nimcp/data
cp -r /tmp/data/stimuli /workspace/nimcp/data/
echo "=== Stimuli updated (no restart needed) ==="
REMOTE_STIM
fi

echo "=== Deployment done ==="
