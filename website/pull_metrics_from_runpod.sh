#!/bin/bash
# Pull metrics.json from RunPod to Hetzner website dir.
#
# Sources podconfig.sh so when the pod's SSH port/host changes (RunPod
# re-issues these on every restart), this script tracks it automatically
# instead of going silently stale.
set -uo pipefail

NIMCP_DIR="/home/bbrelin/nimcp"
. "${NIMCP_DIR}/scripts/podconfig.sh"

$POD_SCP "${POD_HOST}:${POD_DIR}/website/metrics.json" \
         "${NIMCP_DIR}/website/metrics.json" 2>/dev/null
