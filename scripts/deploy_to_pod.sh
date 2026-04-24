#!/usr/bin/env bash
# Safe deployment of a new nimcp.so + scripts to the RunPod.
#
# NEVER overwrite nimcp.so while the brain daemon is running — the .so is mmap'd
# into the process, and overwriting it causes SIGSEGV. This script enforces:
#   1. Bundle artifacts + upload to /tmp on pod
#   2. Capture the running daemon's cmdline (argv) from /proc
#   3. Send SIGTERM, wait for graceful exit (final checkpoint save)
#   4. Swap .so + scripts atomically
#   5. Relaunch daemon with the same argv, capture new PID
#   6. Wait for "Brain daemon ready" + socket to appear before returning
#
# The pod has no supervisor/systemd/cron watchdog — this script is the sole
# mechanism that restarts the daemon. If it fails mid-way, the daemon stays
# down until you intervene.
#
# Usage:
#   ./deploy_to_pod.sh                  # full deploy: .so + scripts + stimuli
#   ./deploy_to_pod.sh --scripts-only   # just scripts (no daemon restart)
#   ./deploy_to_pod.sh --stimuli-only   # just stimuli (no daemon restart)
#
# Override pod connection with env vars — see scripts/podconfig.sh.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=podconfig.sh
. "$HERE/podconfig.sh"

REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO"

mode="full"
case "${1:-}" in
  --scripts-only) mode="scripts" ;;
  --stimuli-only) mode="stimuli" ;;
  "") ;;
  *) echo "unknown arg: $1" >&2; exit 2 ;;
esac

echo "=== Mode: $mode  Pod: $POD_HOST:$POD_PORT ==="

# Preflight: the brain .so must exist and be readable before we stop anything.
if [[ "$mode" == "full" ]]; then
  if [[ ! -f build/lib/python/nimcp.so ]]; then
    echo "ERROR: build/lib/python/nimcp.so missing. Run: make nimcp_python -j4" >&2
    exit 3
  fi
fi

# ----------------------------------------------------------------------------
# Pack bundle
# ----------------------------------------------------------------------------
BUNDLE=/tmp/athena_bundle.tgz
# Resolve the versioned libnimcp (e.g. libnimcp.so.0.9.0) — the symlinks on
# the pod already point at it by name, so we only need to ship the real file.
LIBNIMCP_REAL="$(readlink -f build/lib/libnimcp.so 2>/dev/null || true)"
if [[ "$mode" == "full" && ! -f "$LIBNIMCP_REAL" ]]; then
    echo "ERROR: build/lib/libnimcp.so → $LIBNIMCP_REAL missing. Run: make nimcp -j4" >&2
    exit 3
fi
case "$mode" in
  full)
    tar czf "$BUNDLE" \
        "${LIBNIMCP_REAL#"$REPO/"}" \
        build/lib/python/nimcp.so \
        scripts/test_harness scripts/tests \
        scripts/run_full_battery.py \
        scripts/brain_daemon.py scripts/brain_client.py \
        scripts/immerse_athena.py \
        data/stimuli/
    ;;
  scripts)
    tar czf "$BUNDLE" \
        scripts/test_harness scripts/tests \
        scripts/run_full_battery.py \
        scripts/brain_daemon.py scripts/brain_client.py \
        scripts/immerse_athena.py
    ;;
  stimuli)
    tar czf "$BUNDLE" data/stimuli/
    ;;
esac

echo "=== Uploading bundle ($(du -h "$BUNDLE" | cut -f1)) ==="
$POD_SCP "$BUNDLE" "$POD_HOST:/tmp/"

# ----------------------------------------------------------------------------
# Remote execution
# ----------------------------------------------------------------------------
if [[ "$mode" == "full" ]]; then
  echo "=== Full deploy: stop daemon, swap, restart ==="
  # Export pod paths so the heredoc can interpolate.
  export POD_DIR POD_PY_SITE POD_SOCKET POD_LOGDIR POD_CKPT
  $POD_SSH "POD_DIR=$POD_DIR POD_PY_SITE=$POD_PY_SITE POD_SOCKET=$POD_SOCKET POD_LOGDIR=$POD_LOGDIR POD_CKPT=$POD_CKPT bash -s" <<'REMOTE_FULL'
set -euo pipefail

# 1) Find running daemon + training PIDs. Match only actual python3 processes
# (comm=python3) — bash wrappers that launched them via `bash -c "python3 ..."`
# also appear in pgrep -f output, and their argv is shell metacharacters, not
# something we can feed back to python.
find_python_pid() {
  # $1 = script basename (e.g. brain_daemon.py)
  # Prints PID of the first python3 process whose argv contains the script, or nothing.
  for pid in $(pgrep -x python3 2>/dev/null); do
    if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "$1"; then
      echo "$pid"
      return
    fi
  done
}
DAEMON_PID="$(find_python_pid brain_daemon.py)"
TRAIN_PID="$(find_python_pid immerse_athena.py)"

DAEMON_ARGS=""
if [[ -n "$DAEMON_PID" ]]; then
  # /proc cmdline is NUL-separated. argv[0]=python3, argv[1]=-u, argv[2]=scripts/brain_daemon.py,
  # argv[3..]=the daemon's real args (--socket, --resume, --checkpoint, ...).
  DAEMON_ARGS="$(tr '\0' ' ' < /proc/$DAEMON_PID/cmdline | cut -d' ' -f4- | sed 's/ *$//')"
  echo "Daemon PID=$DAEMON_PID args: $DAEMON_ARGS"
else
  # No daemon running — use sensible defaults for the first deploy.
  DAEMON_ARGS="--socket $POD_SOCKET --resume --checkpoint $POD_CKPT --init-mode full"
  echo "No running daemon. Will start fresh with: $DAEMON_ARGS"
fi

TRAIN_ARGS=""
if [[ -n "$TRAIN_PID" ]]; then
  TRAIN_ARGS="$(tr '\0' ' ' < /proc/$TRAIN_PID/cmdline | cut -d' ' -f4- | sed 's/ *$//')"
  echo "Training PID=$TRAIN_PID args: $TRAIN_ARGS"
fi

# 2) Stop training first (it holds socket connections to the daemon). Then daemon.
if [[ -n "$TRAIN_PID" ]]; then
  echo "Stopping training (PID=$TRAIN_PID)..."
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  for _ in $(seq 1 20); do
    kill -0 "$TRAIN_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo "Training didn't exit in 20s — SIGKILL"
    kill -KILL "$TRAIN_PID" 2>/dev/null || true
  fi
fi

if [[ -n "$DAEMON_PID" ]]; then
  echo "Stopping daemon (PID=$DAEMON_PID) — this triggers final checkpoint save..."
  kill -TERM "$DAEMON_PID" 2>/dev/null || true
  # Final save can take ~15s for a 2M brain. Wait up to 120s.
  for i in $(seq 1 120); do
    kill -0 "$DAEMON_PID" 2>/dev/null || { echo "Daemon exited after ${i}s"; break; }
    sleep 1
  done
  if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "ERROR: Daemon didn't exit in 120s — refusing to SIGKILL (would lose checkpoint)" >&2
    echo "       Inspect /proc/$DAEMON_PID and decide manually." >&2
    exit 5
  fi
fi

# Remove the socket if left behind (shouldn't happen with clean exit).
rm -f "$POD_SOCKET"

# 2b) Defense in depth: refuse to overwrite .so files if ANY python3 process
# still has libnimcp mmap'd. We already killed the daemon above, but a
# zombie/escaped python3 elsewhere would silently corrupt if we plowed ahead.
#
# NOTE on bash footguns: `grep ... && echo` can return 1 (when grep misses).
# Under `set -e` + `pipefail`, that exit-1 propagates through the pipeline
# and aborts the whole heredoc WITHOUT EMITTING ANYTHING — you'd see a clean
# early exit and think the install succeeded. Use an explicit `if` instead
# so each loop iteration returns 0 regardless of grep's match status.
STILL_MAPPED=""
for p in $(pgrep -x python3 2>/dev/null || true); do
    if grep -q 'libnimcp\.so' "/proc/$p/maps" 2>/dev/null; then
        STILL_MAPPED+="PID=$p "
    fi
done
if [[ -n "$STILL_MAPPED" ]]; then
    echo "ERROR: python3 process(es) still hold libnimcp.so mmap'd — aborting swap:" >&2
    echo "$STILL_MAPPED" >&2
    exit 7
fi

# 3) Extract bundle + swap files via atomic rename. Writing to a temp name
# then `mv` into place means any process that still has the OLD .so mmap'd
# keeps seeing the old inode (unlinked but kept alive by the mapping) — no
# page-level corruption. `cp` over the destination TRUNCATES the existing
# inode, which corrupts any live mmap; never use it for mmap'd files.
cd /tmp
tar xzf athena_bundle.tgz

atomic_install() {
    # $1 = source, $2 = destination.
    local src="$1" dst="$2" tmp
    tmp="$dst.deploy.$$"
    cp "$src" "$tmp"
    chmod --reference="$src" "$tmp" 2>/dev/null || chmod 0755 "$tmp"
    mv -f "$tmp" "$dst"
    echo "Installed $dst"
}

# The versioned C library — daemon mmaps this at runtime. Symlinks
# (libnimcp.so, libnimcp.so.2) are left untouched; they already point at
# the versioned name (e.g. libnimcp.so.0.9.0). If the version tag changes
# between builds you'll need to update them manually.
for f in /tmp/build/lib/libnimcp.so.*; do
    [[ -f "$f" && ! -L "$f" ]] || continue
    atomic_install "$f" "$POD_DIR/build/lib/$(basename "$f")"
done

# The Python binding wrapper — thin .so that dlopens libnimcp.so.2.
atomic_install /tmp/build/lib/python/nimcp.so \
    "$POD_PY_SITE/nimcp.cpython-312-x86_64-linux-gnu.so"

# Scripts (bundle paths are relative to repo root — /tmp/scripts/...).
cp -r /tmp/scripts/test_harness "$POD_DIR/scripts/"
cp -r /tmp/scripts/tests        "$POD_DIR/scripts/"
cp    /tmp/scripts/run_full_battery.py "$POD_DIR/scripts/"
cp    /tmp/scripts/brain_daemon.py     "$POD_DIR/scripts/"
cp    /tmp/scripts/brain_client.py     "$POD_DIR/scripts/"
cp    /tmp/scripts/immerse_athena.py   "$POD_DIR/scripts/"

# Stimuli.
rm -rf "$POD_DIR/data/stimuli"
mkdir -p "$POD_DIR/data"
cp -r /tmp/data/stimuli "$POD_DIR/data/"

# 4) Relaunch daemon with the same argv (detached, stdout/stderr to log dir).
mkdir -p "$POD_LOGDIR" "$(dirname "$POD_SOCKET")"
cd "$POD_DIR"
# Rotate previous logs so the new run starts clean.
ts="$(date +%m-%d-%H%M-deploy)"
for f in daemon.stdout daemon.stderr training.stdout training.stderr; do
  [[ -f "$POD_LOGDIR/$f" ]] && mv "$POD_LOGDIR/$f" "$POD_LOGDIR/$f.$ts"
done

echo "Starting daemon: python3 -u scripts/brain_daemon.py $DAEMON_ARGS"
# shellcheck disable=SC2086
nohup python3 -u scripts/brain_daemon.py $DAEMON_ARGS \
    > "$POD_LOGDIR/daemon.stdout" 2> "$POD_LOGDIR/daemon.stderr" &
NEW_DAEMON_PID=$!
disown || true
echo "Daemon started PID=$NEW_DAEMON_PID"

# 5) Wait for socket + "Brain daemon ready" in stderr. 2M brain init can take 15min.
echo "Waiting for daemon ready (socket + log marker)..."
for i in $(seq 1 120); do
  if [[ -S "$POD_SOCKET" ]] && grep -q "Brain daemon ready" "$POD_LOGDIR/daemon.stderr" 2>/dev/null; then
    echo "Daemon READY after ${i}*10s"
    break
  fi
  # Detect early crash.
  if ! kill -0 "$NEW_DAEMON_PID" 2>/dev/null; then
    echo "ERROR: Daemon died during init. Tail of stderr:" >&2
    tail -40 "$POD_LOGDIR/daemon.stderr" >&2 || true
    exit 6
  fi
  sleep 10
done
if [[ ! -S "$POD_SOCKET" ]]; then
  echo "WARNING: Daemon still initializing after 20 min — letting it continue in background"
  echo "         Watch: tail -f $POD_LOGDIR/daemon.stderr"
fi

# 6) Relaunch training if it was running.
if [[ -n "$TRAIN_ARGS" ]]; then
  echo "Restarting training: python3 -u scripts/immerse_athena.py $TRAIN_ARGS"
  # shellcheck disable=SC2086
  nohup python3 -u scripts/immerse_athena.py $TRAIN_ARGS \
      > "$POD_LOGDIR/training.stdout" 2> "$POD_LOGDIR/training.stderr" &
  disown || true
  echo "Training PID=$!"
else
  echo "Training was not running before deploy — not restarting."
fi

echo "=== Full deploy complete ==="
REMOTE_FULL

elif [[ "$mode" == "scripts" ]]; then
  echo "=== Scripts-only deploy (no daemon restart) ==="
  export POD_DIR
  $POD_SSH "POD_DIR=$POD_DIR bash -s" <<'REMOTE_SCRIPTS'
set -euo pipefail
cd /tmp && tar xzf athena_bundle.tgz
cp -r /tmp/scripts/test_harness "$POD_DIR/scripts/"
cp -r /tmp/scripts/tests        "$POD_DIR/scripts/"
cp    /tmp/scripts/run_full_battery.py "$POD_DIR/scripts/"
cp    /tmp/scripts/brain_daemon.py     "$POD_DIR/scripts/"
cp    /tmp/scripts/brain_client.py     "$POD_DIR/scripts/"
cp    /tmp/scripts/immerse_athena.py   "$POD_DIR/scripts/"
echo "=== Scripts updated. Daemon handlers need a restart to take effect. ==="
REMOTE_SCRIPTS

elif [[ "$mode" == "stimuli" ]]; then
  echo "=== Stimuli-only deploy ==="
  export POD_DIR
  $POD_SSH "POD_DIR=$POD_DIR bash -s" <<'REMOTE_STIM'
set -euo pipefail
cd /tmp && tar xzf athena_bundle.tgz
rm -rf "$POD_DIR/data/stimuli"
mkdir -p "$POD_DIR/data"
cp -r /tmp/data/stimuli "$POD_DIR/data/"
echo "=== Stimuli updated (no restart needed) ==="
REMOTE_STIM
fi

echo "=== Deployment done ==="
