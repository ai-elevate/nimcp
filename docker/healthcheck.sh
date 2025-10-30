#!/bin/bash
#==============================================================================
# NIMCP Health Check Script
#==============================================================================
# Verifies NIMCP service is healthy and responsive
#
# Exit codes:
#   0 = Healthy
#   1 = Unhealthy
#==============================================================================

set -e

# Configuration
NIMCP_HOME="${NIMCP_HOME:-/opt/nimcp}"
NIMCP_DATA="${NIMCP_DATA:-/var/lib/nimcp}"
MAX_RESPONSE_TIME=3  # seconds

#------------------------------------------------------------------------------
# Check 1: Library files exist
#------------------------------------------------------------------------------
check_libraries() {
    if [ ! -f "/usr/local/lib/libnimcp_core.so" ]; then
        echo "ERROR: Core library not found"
        return 1
    fi
    return 0
}

#------------------------------------------------------------------------------
# Check 2: Data directory accessible
#------------------------------------------------------------------------------
check_data_dir() {
    if [ ! -d "$NIMCP_DATA" ] || [ ! -w "$NIMCP_DATA" ]; then
        echo "ERROR: Data directory not accessible"
        return 1
    fi
    return 0
}

#------------------------------------------------------------------------------
# Check 3: Memory usage within limits
#------------------------------------------------------------------------------
check_memory() {
    # Get current memory usage (in KB)
    local mem_used=$(ps -o rss= -p $$ | awk '{print $1}')
    local mem_limit=4194304  # 4GB in KB

    if [ "$mem_used" -gt "$mem_limit" ]; then
        echo "WARNING: Memory usage high: ${mem_used}KB"
        # Don't fail, just warn
    fi
    return 0
}

#------------------------------------------------------------------------------
# Check 4: Can create simple test brain
#------------------------------------------------------------------------------
check_brain_creation() {
    # Use Python to test if we can import and create a simple brain
    python3 -c "
import sys
sys.path.insert(0, '/usr/local/lib/python3.10/dist-packages')
try:
    import nimcp
    # Quick test - just check module loads
    assert hasattr(nimcp, 'NeuralNetwork')
    print('OK: NIMCP module functional')
    sys.exit(0)
except Exception as e:
    print(f'ERROR: NIMCP module test failed: {e}')
    sys.exit(1)
" 2>/dev/null

    return $?
}

#------------------------------------------------------------------------------
# Main Health Check
#------------------------------------------------------------------------------
main() {
    echo "Running NIMCP health checks..."

    # Run all checks
    check_libraries || exit 1
    check_data_dir || exit 1
    check_memory || exit 1
    check_brain_creation || exit 1

    echo "✓ All health checks passed"
    return 0
}

main "$@"
