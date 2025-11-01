#!/bin/bash
#==============================================================================
# NIMCP Code Coverage Script
#==============================================================================
# Generates code coverage reports using gcov/lcov
#
# Prerequisites:
#   - gcov (usually comes with gcc)
#   - lcov (install: sudo apt-get install lcov)
#   - genhtml (part of lcov)
#
# Usage:
#   ./scripts/coverage.sh [--html] [--clean]
#
# Options:
#   --html    Generate HTML report (default: yes)
#   --clean   Clean coverage data before running
#
# Output:
#   - coverage/coverage.info  - LCOV data file
#   - coverage/html/          - HTML report (if --html)
#   - Terminal summary
#
# Exit codes:
#   0 - Coverage generated successfully
#   1 - Failed to generate coverage
#   2 - Required tools not available
#==============================================================================

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
COVERAGE_DIR="$PROJECT_ROOT/coverage"
SRC_DIR="$PROJECT_ROOT/src"

# Options
GENERATE_HTML=true
CLEAN_FIRST=false
ENFORCE_THRESHOLD=false
MIN_COVERAGE=85.0

# Parse arguments
for arg in "$@"; do
    case $arg in
        --no-html)
            GENERATE_HTML=false
            ;;
        --clean)
            CLEAN_FIRST=true
            ;;
        --enforce)
            ENFORCE_THRESHOLD=true
            ;;
        --threshold)
            shift
            MIN_COVERAGE="$1"
            ;;
        --help)
            head -n 30 "$0" | grep "^#" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
    esac
    shift || true
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

#==============================================================================
# Check Prerequisites
#==============================================================================
log_info "Checking prerequisites..."

if ! command -v gcov &> /dev/null; then
    log_error "gcov not found. Please install gcc/g++."
    exit 2
fi

if ! command -v lcov &> /dev/null; then
    log_warning "lcov not found. Install with: sudo apt-get install lcov"
    log_info "Continuing with gcov only..."
    LCOV_AVAILABLE=false
else
    LCOV_AVAILABLE=true
fi

#==============================================================================
# Check Build Directory
#==============================================================================
if [[ ! -d "$BUILD_DIR" ]]; then
    log_error "Build directory not found: $BUILD_DIR"
    log_info "Please build the project first with coverage flags:"
    log_info "  mkdir -p build && cd build"
    log_info "  cmake -DCMAKE_BUILD_TYPE=Debug .."
    log_info "  make"
    exit 1
fi

#==============================================================================
# Clean Previous Coverage Data
#==============================================================================
if [[ "$CLEAN_FIRST" == true ]]; then
    log_info "Cleaning previous coverage data..."
    find "$BUILD_DIR" -name "*.gcda" -delete
    rm -rf "$COVERAGE_DIR"
fi

#==============================================================================
# Create Coverage Directory
#==============================================================================
log_info "Creating coverage directory..."
mkdir -p "$COVERAGE_DIR"

#==============================================================================
# Run Tests to Generate Coverage Data
#==============================================================================
log_info "Running tests to generate coverage data..."

cd "$BUILD_DIR"

if [[ -f "src/tests/nimcp_tests" ]]; then
    log_info "Running nimcp_tests..."
    ./src/tests/nimcp_tests || log_warning "Some tests failed, but continuing with coverage..."
else
    log_error "Test executable not found: $BUILD_DIR/src/tests/nimcp_tests"
    log_info "Please build tests first."
    exit 1
fi

#==============================================================================
# Check for Coverage Data
#==============================================================================
log_info "Checking for coverage data files..."

GCDA_COUNT=$(find "$BUILD_DIR" -name "*.gcda" | wc -l)
if [[ $GCDA_COUNT -eq 0 ]]; then
    log_error "No .gcda files found. Was the project built with coverage flags?"
    log_info "Rebuild with:"
    log_info "  cd build"
    log_info "  cmake -DCMAKE_BUILD_TYPE=Debug .."
    log_info "  make clean && make"
    exit 1
fi

log_success "Found $GCDA_COUNT coverage data files"

#==============================================================================
# Generate LCOV Report
#==============================================================================
if [[ "$LCOV_AVAILABLE" == true ]]; then
    log_info "Generating LCOV coverage report..."

    # Capture coverage data
    lcov --capture \
         --directory "$BUILD_DIR" \
         --output-file "$COVERAGE_DIR/coverage.info" \
         --rc lcov_branch_coverage=1 \
         2>&1 | grep -v "ignoring data for external file" || true

    # Remove system/external files from coverage
    lcov --remove "$COVERAGE_DIR/coverage.info" \
         '/usr/*' \
         '*/tests/*' \
         '*/examples/*' \
         --output-file "$COVERAGE_DIR/coverage.info" \
         --rc lcov_branch_coverage=1 \
         2>&1 | grep -v "ignoring data for external file" || true

    # Generate summary
    log_info "Coverage Summary:"
    lcov --list "$COVERAGE_DIR/coverage.info" --rc lcov_branch_coverage=1

    #==========================================================================
    # Generate HTML Report
    #==========================================================================
    if [[ "$GENERATE_HTML" == true ]]; then
        log_info "Generating HTML report..."

        if command -v genhtml &> /dev/null; then
            genhtml "$COVERAGE_DIR/coverage.info" \
                    --output-directory "$COVERAGE_DIR/html" \
                    --title "NIMCP Code Coverage" \
                    --branch-coverage \
                    --function-coverage \
                    --rc lcov_branch_coverage=1 \
                    2>&1 | tail -20

            log_success "HTML report generated at: $COVERAGE_DIR/html/index.html"

            # Try to open in browser
            if command -v xdg-open &> /dev/null; then
                log_info "Opening coverage report in browser..."
                xdg-open "$COVERAGE_DIR/html/index.html" 2>/dev/null || true
            fi
        else
            log_warning "genhtml not found, skipping HTML generation"
        fi
    fi

else
    #==========================================================================
    # Fallback: Use gcov directly
    #==========================================================================
    log_info "Using gcov for coverage analysis..."

    cd "$BUILD_DIR/src/tests"
    COVERED=0
    TOTAL=0

    for gcda in *.gcda; do
        if [[ -f "$gcda" ]]; then
            gcov "$gcda" &> /dev/null || true
            COVERED=$((COVERED + 1))
        fi
    done

    log_info "Processed coverage data for $COVERED files"
    log_warning "Install lcov for detailed coverage reports: sudo apt-get install lcov"
fi

#==============================================================================
# Extract Key Metrics
#==============================================================================
if [[ "$LCOV_AVAILABLE" == true && -f "$COVERAGE_DIR/coverage.info" ]]; then
    log_info "Extracting coverage metrics..."

    # Get line coverage percentage
    LINE_COVERAGE=$(lcov --summary "$COVERAGE_DIR/coverage.info" 2>&1 | \
                    grep "lines" | \
                    awk '{print $2}' | \
                    sed 's/%//')

    # Get function coverage percentage
    FUNC_COVERAGE=$(lcov --summary "$COVERAGE_DIR/coverage.info" 2>&1 | \
                    grep "functions" | \
                    awk '{print $2}' | \
                    sed 's/%//')

    echo ""
    echo "========================================"
    echo "Coverage Summary"
    echo "========================================"
    echo "Line Coverage:     ${LINE_COVERAGE}%"
    echo "Function Coverage: ${FUNC_COVERAGE}%"
    echo "Minimum Required:  ${MIN_COVERAGE}%"
    echo "========================================"
    echo ""

    # Check coverage threshold
    if command -v bc &> /dev/null; then
        IS_BELOW=$(echo "$LINE_COVERAGE < $MIN_COVERAGE" | bc -l 2>/dev/null || echo 0)

        if [[ "$IS_BELOW" -eq 1 ]]; then
            log_warning "Line coverage (${LINE_COVERAGE}%) is below threshold (${MIN_COVERAGE}%)"

            if [[ "$ENFORCE_THRESHOLD" == true ]]; then
                log_error "Coverage enforcement failed. Please add more tests."
                exit 1
            fi
        else
            log_success "Coverage ${LINE_COVERAGE}% meets threshold (${MIN_COVERAGE}%)"
        fi
    else
        log_warning "bc not found, cannot compare thresholds"
    fi
fi

#==============================================================================
# Final Status
#==============================================================================
echo ""
log_success "Coverage analysis complete!"
log_info "Coverage data: $COVERAGE_DIR/coverage.info"
if [[ "$GENERATE_HTML" == true && -d "$COVERAGE_DIR/html" ]]; then
    log_info "HTML report:   $COVERAGE_DIR/html/index.html"
fi
echo ""

exit 0
