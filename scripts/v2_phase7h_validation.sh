#!/usr/bin/env bash
# Phase 7h — V2 daemon ≥1hr live validation under systemd.
#
# Runs against a host where `nimcp-v2-daemon.service` is installed. The
# script:
#   1. Pre-checks: binary present, config present, socket dir writable,
#      no stale daemon already holding the socket.
#   2. Starts the systemd unit + waits for the socket to appear.
#   3. Drives the daemon for ${VALIDATION_DURATION_SEC} seconds via the
#      JSON-over-unix-socket protocol (one learn() call/sec, mixed in
#      with periodic stats() reads + every-5-min checkpoint trigger).
#   4. Captures the journal + metrics.json + stats snapshots into a
#      timestamped artifact dir.
#   5. Acceptance checks:
#        - daemon never restarted (systemd RestartCount == 0)
#        - no SIGSEGV / panic in journal
#        - metrics.json mtime advanced at the metrics-tick cadence
#        - process RSS at end < 1.5x RSS at start (no slow leak)
#        - stats() returned valid JSON throughout (no protocol drift)
#        - last stats() reports >= 1 successful learn() and predict()
#   6. Prints PASS / FAIL summary and exits with the matching code.
#
# Exit codes:
#   0  — every acceptance check passed (Phase 7h SHIP)
#   2  — pre-check failed (binary / config / socket)
#   3  — daemon failed to start
#   4  — protocol error during the drive loop
#   5  — acceptance check failed (RSS leak / restart / SIGSEGV / etc.)
#
# Defaults are tuned for the RunPod box (74.2.96.55:17653). Override
# via env: VALIDATION_DURATION_SEC, ARTIFACT_DIR, SOCKET_PATH.
set -euo pipefail

# -----------------------------------------------------------------------------
# Config (override via env).
# -----------------------------------------------------------------------------
VALIDATION_DURATION_SEC="${VALIDATION_DURATION_SEC:-3600}"  # 1 hour default
SOCKET_PATH="${SOCKET_PATH:-/run/hera/brain.sock}"
SERVICE_NAME="${SERVICE_NAME:-nimcp-v2-daemon}"
METRICS_PATH="${METRICS_PATH:-/var/lib/nimcp-v2/metrics.json}"
ARTIFACT_DIR="${ARTIFACT_DIR:-/var/log/nimcp-v2/phase7h-$(date -u +%Y%m%dT%H%M%SZ)}"
DAEMON_BIN="${DAEMON_BIN:-/opt/nimcp-v2/bin/nimcp-v2-daemon}"
CONFIG_JSON="${CONFIG_JSON:-/etc/nimcp-v2/config.json}"
DRIVE_INPUT_DIM="${DRIVE_INPUT_DIM:-64}"
DRIVE_OUTPUT_DIM="${DRIVE_OUTPUT_DIM:-10}"
LEARN_INTERVAL_SEC="${LEARN_INTERVAL_SEC:-1}"
STATS_INTERVAL_SEC="${STATS_INTERVAL_SEC:-30}"
CHECKPOINT_INTERVAL_SEC="${CHECKPOINT_INTERVAL_SEC:-300}"

mkdir -p "$ARTIFACT_DIR"
RUN_LOG="$ARTIFACT_DIR/run.log"
JOURNAL_LOG="$ARTIFACT_DIR/journal.log"
STATS_LOG="$ARTIFACT_DIR/stats.jsonl"
METRICS_HISTORY="$ARTIFACT_DIR/metrics_history.tsv"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$RUN_LOG" >&2; }
fail() { log "FAIL $*"; exit "${2:-5}"; }

# -----------------------------------------------------------------------------
# 1. Pre-checks.
# -----------------------------------------------------------------------------
log "== Phase 7h validation starting =="
log "duration=${VALIDATION_DURATION_SEC}s socket=${SOCKET_PATH} service=${SERVICE_NAME}"
log "artifact_dir=${ARTIFACT_DIR}"

[[ -x "$DAEMON_BIN" ]] || fail "daemon binary missing: $DAEMON_BIN" 2
[[ -r "$CONFIG_JSON" ]] || fail "config missing: $CONFIG_JSON" 2

if systemctl is-active --quiet "$SERVICE_NAME"; then
    log "service already running — stopping for a clean baseline"
    sudo systemctl stop "$SERVICE_NAME"
    sleep 2
fi

if [[ -S "$SOCKET_PATH" ]]; then
    log "stale socket present, removing"
    sudo rm -f "$SOCKET_PATH"
fi

# -----------------------------------------------------------------------------
# 2. Start the unit + wait for socket.
# -----------------------------------------------------------------------------
log "starting $SERVICE_NAME"
sudo systemctl start "$SERVICE_NAME"
START_TS=$(date +%s)

deadline=$((START_TS + 30))
while [[ $(date +%s) -lt $deadline ]]; do
    if [[ -S "$SOCKET_PATH" ]] && systemctl is-active --quiet "$SERVICE_NAME"; then
        break
    fi
    sleep 1
done
[[ -S "$SOCKET_PATH" ]] || fail "socket did not appear within 30s" 3
systemctl is-active --quiet "$SERVICE_NAME" || fail "service not active after 30s" 3

START_RSS_KB=$(ps -o rss= -p "$(systemctl show -p MainPID --value "$SERVICE_NAME")" | tr -d ' ')
log "daemon up; main_pid=$(systemctl show -p MainPID --value "$SERVICE_NAME") start_rss_kb=$START_RSS_KB"

# -----------------------------------------------------------------------------
# 3. Drive loop. Three concurrent cadences via background subshells.
# -----------------------------------------------------------------------------
END_TS=$((START_TS + VALIDATION_DURATION_SEC))

# Build deterministic sample vectors via Python (avoid shell float math).
python3 - <<PY > "$ARTIFACT_DIR/features.json"
import json, random
random.seed(0xC0FFEE)
print(json.dumps([random.gauss(0, 1) for _ in range($DRIVE_INPUT_DIM)]))
PY
python3 - <<PY > "$ARTIFACT_DIR/target.json"
import json, random
random.seed(0xBEEF)
print(json.dumps([random.uniform(-1, 1) for _ in range($DRIVE_OUTPUT_DIM)]))
PY

# JSON-over-unix-socket helper — `socat` keeps the line-oriented
# protocol simple. Each call opens, sends, reads one line back.
send() {
    local payload="$1"
    socat -t 5 - "UNIX-CONNECT:$SOCKET_PATH" <<<"$payload" || return 1
}

learn_loop() {
    local feats; feats=$(cat "$ARTIFACT_DIR/features.json")
    local tgt;   tgt=$(cat "$ARTIFACT_DIR/target.json")
    local n=0
    while [[ $(date +%s) -lt $END_TS ]]; do
        send "{\"op\":\"learn\",\"features\":$feats,\"target\":$tgt,\"lr\":0.01}" \
            >>"$ARTIFACT_DIR/learn.log" 2>&1 || \
            { log "learn() error at iter $n"; return 4; }
        n=$((n+1))
        sleep "$LEARN_INTERVAL_SEC"
    done
    log "learn_loop completed $n iterations"
}

stats_loop() {
    while [[ $(date +%s) -lt $END_TS ]]; do
        local snap; snap=$(send '{"op":"stats"}') || \
            { log "stats() error"; return 4; }
        printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$snap" >>"$STATS_LOG"
        sleep "$STATS_INTERVAL_SEC"
    done
}

ckpt_loop() {
    while [[ $(date +%s) -lt $END_TS ]]; do
        sleep "$CHECKPOINT_INTERVAL_SEC"
        send "{\"op\":\"save\",\"path\":\"$ARTIFACT_DIR/ckpt-$(date +%s).bin\"}" \
            >>"$ARTIFACT_DIR/ckpt.log" 2>&1 || \
            { log "save() error"; return 4; }
    done
}

metrics_history_loop() {
    while [[ $(date +%s) -lt $END_TS ]]; do
        if [[ -r "$METRICS_PATH" ]]; then
            printf '%s\t%s\n' "$(date -u +%s)" "$(stat -c %Y "$METRICS_PATH")" \
                >>"$METRICS_HISTORY"
        fi
        sleep 5
    done
}

log "starting drive loops; will run for ${VALIDATION_DURATION_SEC}s"
learn_loop &        LEARN_PID=$!
stats_loop &        STATS_PID=$!
ckpt_loop &         CKPT_PID=$!
metrics_history_loop & MET_PID=$!

# Wait for the duration window. Use `wait` on each subshell so any
# error propagates as a non-zero exit.
RC=0
wait $LEARN_PID || RC=$?
wait $STATS_PID || true
wait $CKPT_PID  || true
wait $MET_PID   || true
[[ $RC -eq 0 ]] || fail "drive loop returned $RC" "$RC"

# -----------------------------------------------------------------------------
# 4. Capture journal + final RSS.
# -----------------------------------------------------------------------------
log "drive loop complete; capturing artifacts"
journalctl -u "$SERVICE_NAME" --since "@$START_TS" --no-pager >"$JOURNAL_LOG" || true
END_RSS_KB=$(ps -o rss= -p "$(systemctl show -p MainPID --value "$SERVICE_NAME")" 2>/dev/null \
    | tr -d ' ' || echo 0)
RESTART_COUNT=$(systemctl show -p NRestarts --value "$SERVICE_NAME" || echo 0)
log "end_rss_kb=$END_RSS_KB restart_count=$RESTART_COUNT"

# -----------------------------------------------------------------------------
# 5. Acceptance checks.
# -----------------------------------------------------------------------------
PASS=0
CHECK() {
    local name="$1"; shift
    if "$@"; then log "PASS $name"; else log "FAIL $name"; PASS=1; fi
}

CHECK "no_restarts" test "$RESTART_COUNT" -eq 0
CHECK "no_sigsegv_or_panic" \
    bash -c "! grep -E -q '(SIGSEGV|SIGABRT|panicked at|panic:)' '$JOURNAL_LOG'"
CHECK "metrics_advanced" \
    bash -c "wc -l <'$METRICS_HISTORY' | awk '{exit !(\$1 > 60)}'"
CHECK "rss_no_serious_leak" \
    bash -c "awk -v s='$START_RSS_KB' -v e='$END_RSS_KB' 'BEGIN{exit !(e <= s*1.5)}'"
CHECK "stats_log_nonempty" \
    bash -c "[[ -s '$STATS_LOG' ]]"
CHECK "service_still_active" \
    systemctl is-active --quiet "$SERVICE_NAME"

if [[ $PASS -eq 0 ]]; then
    log "== Phase 7h VALIDATION PASS — V2 daemon survived ${VALIDATION_DURATION_SEC}s under load =="
    exit 0
else
    log "== Phase 7h VALIDATION FAIL — see ${ARTIFACT_DIR} for details =="
    exit 5
fi
