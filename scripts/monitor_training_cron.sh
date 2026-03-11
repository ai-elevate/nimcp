#!/bin/bash
#
# NIMCP Training Monitor — cron-friendly, persistent across sessions
# Runs all 3 monitors: differentiation, health, checkpoint integrity
#
# Install: crontab -e, then add:
#   */5 * * * * /home/bbrelin/nimcp/scripts/monitor_training_cron.sh
#
# Alerts go to:
#   1. /home/bbrelin/nimcp/monitoring.log (always)
#   2. Desktop notification via notify-send (if DISPLAY available)
#   3. /home/bbrelin/nimcp/TRAINING_ALERT.txt (touch file on critical alerts)
#

set -uo pipefail

NIMCP_DIR="/home/bbrelin/nimcp"
TRAINING_LOG="${NIMCP_DIR}/training.log"
CHECKPOINT_DIR="${NIMCP_DIR}/checkpoints/athena"
MONITOR_LOG="${NIMCP_DIR}/monitoring.log"
ALERT_FILE="${NIMCP_DIR}/TRAINING_ALERT.txt"
STATE_FILE="${CHECKPOINT_DIR}/immersive_state.json"

# Thresholds
OUTPUT_SIM_THRESHOLD=0.95
EFF_RANK_THRESHOLD=10
GRAD_NORM_THRESHOLD=100
CHECKPOINT_MIN_SIZE=104857600  # 100 MB
RSS_WARN_GB=55  # warn if RSS > 55 GB

NOW=$(date '+%Y-%m-%d %H:%M:%S')

# Truncate monitor log if > 10MB
if [ -f "$MONITOR_LOG" ] && [ "$(stat -c%s "$MONITOR_LOG" 2>/dev/null || echo 0)" -gt 10485760 ]; then
    tail -1000 "$MONITOR_LOG" > "${MONITOR_LOG}.tmp" && mv "${MONITOR_LOG}.tmp" "$MONITOR_LOG"
fi

log() {
    echo "[$NOW] $1" >> "$MONITOR_LOG"
}

alert() {
    local level="$1"  # WARN or CRITICAL
    local msg="$2"
    log "[$level] $msg"
    echo "[$NOW] [$level] $msg" >> "$ALERT_FILE"

    # Desktop notification if possible
    if [ -n "${DISPLAY:-}" ] || [ -n "${WAYLAND_DISPLAY:-}" ]; then
        urgency="normal"
        [ "$level" = "CRITICAL" ] && urgency="critical"
        notify-send -u "$urgency" "NIMCP Training $level" "$msg" 2>/dev/null || true
    fi
}

#=============================================================================
# 1. PROCESS HEALTH
#=============================================================================
check_health() {
    log "--- HEALTH CHECK ---"

    # Find training process (python3 binary, not the bash wrapper)
    local pid
    pid=$(pgrep -x -f "python3 scripts/immerse_athena.py" 2>/dev/null | head -1 || true)
    # Fallback: find python3 process with immerse_athena in cmdline
    if [ -z "$pid" ]; then
        pid=$(ps aux | grep "[p]ython3.*immerse_athena" | awk '{print $2}' | head -1 || true)
    fi

    if [ -z "$pid" ]; then
        alert "CRITICAL" "Training process NOT running! immerse_athena.py is dead."
        return
    fi

    # RSS memory (in KB from ps, convert to GB)
    local rss_kb rss_gb
    rss_kb=$(cat /proc/"$pid"/status 2>/dev/null | grep VmRSS | awk '{print $2}' || echo 0)
    rss_kb=${rss_kb:-0}
    rss_gb=$(awk "BEGIN {printf \"%.1f\", $rss_kb / 1048576}")
    local cpu
    cpu=$(ps -p "$pid" -o %cpu= 2>/dev/null | tr -d ' ' || echo "?")

    log "Process alive PID=$pid | RSS=${rss_gb}GB | CPU=${cpu}%"

    if [ "$(awk "BEGIN {print ($rss_gb > $RSS_WARN_GB)}")" = "1" ]; then
        alert "WARN" "RSS memory high: ${rss_gb}GB (threshold: ${RSS_WARN_GB}GB)"
    fi

    # Check for NaN/Inf in recent log (last 500 lines)
    if [ -f "$TRAINING_LOG" ]; then
        local nan_count
        nan_count=$(tail -500 "$TRAINING_LOG" | grep -c -i "nan" 2>/dev/null || true)
        nan_count=${nan_count:-0}
        nan_count=$(echo "$nan_count" | tr -d '[:space:]')
        local inf_count
        inf_count=$(tail -500 "$TRAINING_LOG" | grep -c -iE "\binf\b|INFINITY" 2>/dev/null || true)
        inf_count=${inf_count:-0}
        inf_count=$(echo "$inf_count" | tr -d '[:space:]')

        if [ "$nan_count" -gt 5 ]; then
            alert "CRITICAL" "NaN detected in training: $nan_count occurrences in last 500 log lines"
        fi
        if [ "$inf_count" -gt 5 ]; then
            alert "WARN" "Inf detected in training: $inf_count occurrences in last 500 log lines"
        fi

        # Latest gradient norm from optimizer log
        local latest_grad
        latest_grad=$(grep -oP 'grad_norm=\K[0-9.e+-]+' "$TRAINING_LOG" | tail -1 || true)
        if [ -n "$latest_grad" ]; then
            log "Latest optimizer grad_norm=$latest_grad"
            # Check if > threshold (handle scientific notation)
            if [ "$(awk "BEGIN {print ($latest_grad > $GRAD_NORM_THRESHOLD)}" 2>/dev/null)" = "1" ]; then
                alert "WARN" "Gradient norm high: $latest_grad (threshold: $GRAD_NORM_THRESHOLD)"
            fi
        fi

        # Latest LNN adjoint norm
        local latest_adjoint
        latest_adjoint=$(grep -oP 'Adjoint computation complete: norm=\K[0-9.]+' "$TRAINING_LOG" | tail -1 || true)
        if [ -n "$latest_adjoint" ]; then
            log "Latest LNN adjoint norm=$latest_adjoint"
        fi

        # Latest loss value
        local latest_loss
        latest_loss=$(grep -oP 'loss[=:]\s*\K[0-9.e+-]+' "$TRAINING_LOG" | tail -1 || true)
        if [ -n "$latest_loss" ]; then
            log "Latest loss=$latest_loss"
        fi

        # K-WTA sparsity
        local latest_kwta
        latest_kwta=$(grep -oP 'sparsity[=:]\s*\K[0-9.]+' "$TRAINING_LOG" | tail -1 || true)
        if [ -n "$latest_kwta" ]; then
            log "Latest K-WTA sparsity=$latest_kwta"
        fi

        # Optimizer step count
        local latest_step
        latest_step=$(grep -oP 'Optimizer step \K[0-9]+' "$TRAINING_LOG" | tail -1 || true)
        if [ -n "$latest_step" ]; then
            log "Latest optimizer step=$latest_step"
        fi
    fi
}

#=============================================================================
# 2. PLASTICITY / SNN / NEUROMODULATION
#=============================================================================
check_bio_metrics() {
    log "--- BIO METRICS CHECK ---"

    if [ ! -f "$TRAINING_LOG" ]; then
        log "No training.log found"
        return
    fi

    # Plasticity state + neuromodulators (from _print_bio_stats output)
    local bio_line
    bio_line=$(grep "Bio:" "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$bio_line" ]; then
        log "$bio_line"
        # Extract individual values
        local da ach rpe stdp ltp state
        state=$(echo "$bio_line" | grep -oP 'state=\K\w+' || true)
        da=$(echo "$bio_line" | grep -oP 'DA=\K[0-9.+-]+' || true)
        ach=$(echo "$bio_line" | grep -oP 'ACh=\K[0-9.+-]+' || true)
        rpe=$(echo "$bio_line" | grep -oP 'RPE=\K[0-9.e+-]+' || true)
        stdp=$(echo "$bio_line" | grep -oP 'STDP=\K[0-9]+' || true)
        ltp=$(echo "$bio_line" | grep -oP 'LTP=\K[0-9]+' || true)
        [ -n "$state" ] && log "  Plasticity state: $state"
        [ -n "$da" ] && log "  Dopamine: $da"
        [ -n "$ach" ] && log "  Acetylcholine: $ach"
        [ -n "$rpe" ] && log "  RPE: $rpe"
        [ -n "$stdp" ] && log "  STDP updates: $stdp"
        [ -n "$ltp" ] && log "  LTP events: $ltp"

        # Alert on stuck plasticity state (should progress from ACQUISITION)
        if [ -n "$state" ] && [ "$state" = "ACQUISITION" ]; then
            local opt_step
            opt_step=$(grep -oP 'Optimizer step \K[0-9]+' "$TRAINING_LOG" | tail -1 || echo "0")
            if [ "$opt_step" -gt 5000 ]; then
                alert "WARN" "Plasticity still in ACQUISITION at step $opt_step (expected CONSOLIDATION)"
            fi
        fi
    else
        log "No Bio stats output yet"
    fi

    # SNN metrics
    local snn_line
    snn_line=$(grep "SNN:" "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$snn_line" ]; then
        log "$snn_line"
        local firing_rate spikes snn_sparsity silent
        firing_rate=$(echo "$snn_line" | grep -oP 'rate=\K[0-9.]+' || true)
        spikes=$(echo "$snn_line" | grep -oP 'spikes=\K[0-9]+' || true)
        snn_sparsity=$(echo "$snn_line" | grep -oP 'sparsity=\K[0-9.]+' || true)
        [ -n "$firing_rate" ] && log "  SNN firing rate: ${firing_rate}Hz"
        [ -n "$spikes" ] && log "  SNN total spikes: $spikes"
        [ -n "$snn_sparsity" ] && log "  SNN sparsity: $snn_sparsity"

        # Alert on dead SNN (0 Hz firing after significant training)
        if [ -n "$firing_rate" ] && [ "$(awk "BEGIN {print ($firing_rate == 0)}")" = "1" ]; then
            local opt_step
            opt_step=$(grep -oP 'Optimizer step \K[0-9]+' "$TRAINING_LOG" | tail -1 || echo "0")
            if [ "$opt_step" -gt 1000 ]; then
                alert "WARN" "SNN firing rate is 0 Hz at step $opt_step — network may be dead"
            fi
        fi
        # Alert on hyperactive SNN (>100 Hz mean is suspicious)
        if [ -n "$firing_rate" ] && [ "$(awk "BEGIN {print ($firing_rate > 100)}" 2>/dev/null)" = "1" ]; then
            alert "WARN" "SNN firing rate very high: ${firing_rate}Hz (possible runaway excitation)"
        fi
    else
        log "No SNN stats output yet"
    fi

    # LNN metrics
    local lnn_line
    lnn_line=$(grep "LNN:" "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$lnn_line" ]; then
        log "$lnn_line"
        local tau lnn_steps
        tau=$(echo "$lnn_line" | grep -oP 'tau=\K[0-9.]+' || true)
        lnn_steps=$(echo "$lnn_line" | grep -oP 'steps=\K[0-9]+' || true)
        [ -n "$tau" ] && log "  LNN avg tau: $tau"
        [ -n "$lnn_steps" ] && log "  LNN forward steps: $lnn_steps"

        # Alert on tau collapse (should be > 0.1 for meaningful temporal dynamics)
        if [ -n "$tau" ] && [ "$(awk "BEGIN {print ($tau < 0.1)}" 2>/dev/null)" = "1" ]; then
            alert "WARN" "LNN tau collapsed to $tau (temporal dynamics lost)"
        fi
    else
        log "No LNN stats output yet"
    fi

    # Arousal / Sleep pressure
    local arousal_line
    arousal_line=$(grep "Arousal=" "$TRAINING_LOG" | grep "SleepPressure" | tail -1 || true)
    if [ -n "$arousal_line" ]; then
        log "$arousal_line"
    fi
}

#=============================================================================
# 3. OUTPUT DIFFERENTIATION
#=============================================================================
check_differentiation() {
    log "--- DIFFERENTIATION CHECK ---"

    if [ ! -f "$TRAINING_LOG" ]; then
        log "No training.log found"
        return
    fi

    # output_sim (cosine similarity between outputs)
    local latest_sim
    latest_sim=$(grep -oP 'output_sim[=:]\s*\K[0-9.]+' "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$latest_sim" ]; then
        log "output_sim=$latest_sim"
        if [ "$(awk "BEGIN {print ($latest_sim > $OUTPUT_SIM_THRESHOLD)}")" = "1" ]; then
            alert "CRITICAL" "Mode collapse risk! output_sim=$latest_sim (threshold: $OUTPUT_SIM_THRESHOLD)"
        fi
    else
        log "output_sim: not yet reported"
    fi

    # eff_rank (effective rank of output matrix)
    local latest_rank
    latest_rank=$(grep -oP 'eff_rank[=:]\s*\K[0-9.]+' "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$latest_rank" ]; then
        log "eff_rank=$latest_rank"
        if [ "$(awk "BEGIN {print ($latest_rank < $EFF_RANK_THRESHOLD)}")" = "1" ]; then
            alert "WARN" "Poor output differentiation! eff_rank=$latest_rank (threshold: $EFF_RANK_THRESHOLD)"
        fi
    else
        log "eff_rank: not yet reported"
    fi

    # contrastive loss
    local latest_contrastive
    latest_contrastive=$(grep -oP 'contrastive[=:]\s*\K[0-9.e+-]+' "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$latest_contrastive" ]; then
        log "contrastive=$latest_contrastive"
    else
        log "contrastive: not yet reported"
    fi

    # diversity
    local latest_diversity
    latest_diversity=$(grep -oP 'diversity[=:]\s*\K[0-9.e+-]+' "$TRAINING_LOG" | tail -1 || true)
    if [ -n "$latest_diversity" ]; then
        log "diversity=$latest_diversity"
    else
        log "diversity: not yet reported"
    fi
}

#=============================================================================
# 4. CHECKPOINT INTEGRITY & STAGE PROGRESSION
#=============================================================================
check_checkpoints() {
    log "--- CHECKPOINT CHECK ---"

    if [ ! -d "$CHECKPOINT_DIR" ]; then
        log "Checkpoint dir not found: $CHECKPOINT_DIR"
        return
    fi

    # Find latest .bin checkpoint
    local latest_ckpt
    latest_ckpt=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name "athena_s*.bin" -not -name "*.tmp" \
                  -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}' || true)

    if [ -n "$latest_ckpt" ] && [ -f "$latest_ckpt" ]; then
        local ckpt_size ckpt_size_mb ckpt_age_sec ckpt_name
        ckpt_name=$(basename "$latest_ckpt")
        ckpt_size=$(stat -c%s "$latest_ckpt" 2>/dev/null || echo 0)
        ckpt_size_mb=$(awk "BEGIN {printf \"%.0f\", $ckpt_size / 1048576}")
        ckpt_age_sec=$(( $(date +%s) - $(stat -c%Y "$latest_ckpt") ))

        log "Latest checkpoint: $ckpt_name (${ckpt_size_mb}MB, ${ckpt_age_sec}s ago)"

        # Size check
        if [ "$ckpt_size" -lt "$CHECKPOINT_MIN_SIZE" ]; then
            # Only alert if training has been running a while (check optimizer step)
            local step
            step=$(grep -oP 'Optimizer step \K[0-9]+' "$TRAINING_LOG" 2>/dev/null | tail -1 || echo "0")
            if [ "$step" -gt 1000 ]; then
                alert "WARN" "Checkpoint possibly truncated: $ckpt_name is only ${ckpt_size_mb}MB (expected >100MB)"
            fi
        fi

        # Staleness check — alert if no checkpoint in 2 hours
        if [ "$ckpt_age_sec" -gt 7200 ]; then
            alert "WARN" "Checkpoint stale: $ckpt_name is $(( ckpt_age_sec / 60 ))m old"
        fi
    else
        log "No snapshot checkpoints found yet"
    fi

    # Count total snapshots
    local snap_count
    snap_count=$(find "$CHECKPOINT_DIR" -maxdepth 1 -name "athena_s*.bin" -not -name "*.tmp" 2>/dev/null | wc -l)
    log "Total snapshots: $snap_count"

    # Canonical symlink
    local canonical="${CHECKPOINT_DIR}/athena_checkpoint.bin"
    if [ -L "$canonical" ]; then
        local target
        target=$(readlink "$canonical" 2>/dev/null || true)
        log "Canonical symlink -> $target"
    elif [ -f "$canonical" ]; then
        local can_size_mb
        can_size_mb=$(awk "BEGIN {printf \"%.0f\", $(stat -c%s "$canonical" 2>/dev/null || echo 0) / 1048576}")
        log "Canonical checkpoint: ${can_size_mb}MB (regular file, not symlink)"
    else
        log "No canonical checkpoint yet"
    fi

    # Stage progression from immersive_state.json
    if [ -f "$STATE_FILE" ]; then
        local stage step
        stage=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('current_stage','?'))" 2>/dev/null || echo "?")
        step=$(python3 -c "import json; d=json.load(open('$STATE_FILE')); print(d.get('current_step', d.get('step','?')))" 2>/dev/null || echo "?")
        log "Stage=$stage Step=$step"
    else
        log "No immersive_state.json found"
    fi

    # List all checkpoint files with sizes
    local listing
    listing=$(ls -lhS "$CHECKPOINT_DIR"/*.bin 2>/dev/null || true)
    if [ -n "$listing" ]; then
        log "Checkpoint listing:"
        echo "$listing" | while read -r line; do
            log "  $line"
        done
    fi
}

#=============================================================================
# MAIN
#=============================================================================
log "========== TRAINING MONITOR RUN =========="

check_health
check_bio_metrics
check_differentiation
check_checkpoints

log "========== END MONITOR RUN =========="
log ""
