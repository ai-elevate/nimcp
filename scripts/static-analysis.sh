#!/bin/bash
#
# Comprehensive static analysis runner for NIMCP
# Runs multiple static analysis tools and aggregates results
#
# Usage:
#   ./scripts/static-analysis.sh [--fix] [--report]
#
# Options:
#   --fix     Apply automatic fixes where possible
#   --report  Generate HTML reports
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
REPORTS_DIR="$PROJECT_ROOT/security-reports"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
FIX_MODE=false
REPORT_MODE=false
for arg in "$@"; do
    case $arg in
        --fix)
            FIX_MODE=true
            ;;
        --report)
            REPORT_MODE=true
            ;;
    esac
done

# Create reports directory
mkdir -p "$REPORTS_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         NIMCP Comprehensive Static Analysis Suite         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

EXIT_CODE=0

# ============================================================================
# 1. Clang-Tidy Analysis
# ============================================================================
echo -e "${YELLOW}[1/5] Running clang-tidy...${NC}"
if command -v clang-tidy &> /dev/null; then
    cd "$BUILD_DIR"

    if [ "$FIX_MODE" = true ]; then
        echo "  → Applying automatic fixes..."
        run-clang-tidy -fix -quiet -j "$(nproc)" 2>&1 | tee "$REPORTS_DIR/clang-tidy.log"
    else
        run-clang-tidy -quiet -j "$(nproc)" 2>&1 | tee "$REPORTS_DIR/clang-tidy.log"
    fi

    # Check for errors
    if grep -q "error:" "$REPORTS_DIR/clang-tidy.log"; then
        echo -e "${RED}  ✗ clang-tidy found errors${NC}"
        EXIT_CODE=1
    else
        echo -e "${GREEN}  ✓ clang-tidy passed${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ clang-tidy not found, skipping${NC}"
fi
echo ""

# ============================================================================
# 2. Cppcheck Analysis
# ============================================================================
echo -e "${YELLOW}[2/5] Running cppcheck...${NC}"
if command -v cppcheck &> /dev/null; then
    cd "$PROJECT_ROOT"

    CPPCHECK_ARGS=(
        --project=.cppcheck
        --quiet
        --error-exitcode=1
        --inline-suppr
        --suppress=missingIncludeSystem
        --suppress=unmatchedSuppression
    )

    if [ "$REPORT_MODE" = true ]; then
        CPPCHECK_ARGS+=(
            --xml
            --xml-version=2
        )
        cppcheck "${CPPCHECK_ARGS[@]}" 2> "$REPORTS_DIR/cppcheck.xml" || EXIT_CODE=1

        # Generate HTML report if cppcheck-htmlreport is available
        if command -v cppcheck-htmlreport &> /dev/null; then
            cppcheck-htmlreport --file="$REPORTS_DIR/cppcheck.xml" \
                               --report-dir="$REPORTS_DIR/cppcheck-html" \
                               --source-dir="$PROJECT_ROOT"
            echo -e "${GREEN}  ✓ HTML report: $REPORTS_DIR/cppcheck-html/index.html${NC}"
        fi
    else
        cppcheck "${CPPCHECK_ARGS[@]}" 2>&1 | tee "$REPORTS_DIR/cppcheck.log" || EXIT_CODE=1
    fi

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}  ✓ cppcheck passed${NC}"
    else
        echo -e "${RED}  ✗ cppcheck found issues${NC}"
    fi
else
    echo -e "${YELLOW}  ⚠ cppcheck not found, skipping${NC}"
fi
echo ""

# ============================================================================
# 3. Clang Static Analyzer
# ============================================================================
echo -e "${YELLOW}[3/5] Running clang static analyzer...${NC}"
if command -v scan-build &> /dev/null; then
    cd "$BUILD_DIR"

    # Clean and rebuild with analyzer
    make clean > /dev/null 2>&1 || true

    ANALYZER_OPTS=(
        -o "$REPORTS_DIR/analyzer"
        --status-bugs
        -enable-checker security.insecureAPI.UncheckedReturn
        -enable-checker security.insecureAPI.strcpy
        -enable-checker security.FloatLoopCounter
        -enable-checker alpha.security.ArrayBoundV2
        -enable-checker alpha.security.MallocOverflow
        -enable-checker alpha.security.ReturnPtrRange
        -enable-checker alpha.unix.cstring.OutOfBounds
    )

    scan-build "${ANALYZER_OPTS[@]}" make nimcp_core > "$REPORTS_DIR/scan-build.log" 2>&1 || EXIT_CODE=1

    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}  ✓ scan-build passed${NC}"
    else
        echo -e "${RED}  ✗ scan-build found bugs${NC}"
        if [ "$REPORT_MODE" = true ]; then
            echo -e "${BLUE}  → HTML report: $REPORTS_DIR/analyzer/*/index.html${NC}"
        fi
    fi
else
    echo -e "${YELLOW}  ⚠ scan-build not found, skipping${NC}"
fi
echo ""

# ============================================================================
# 4. Include-What-You-Use (IWYU)
# ============================================================================
echo -e "${YELLOW}[4/5] Running include-what-you-use...${NC}"
if command -v iwyu_tool.py &> /dev/null || command -v iwyu_tool &> /dev/null; then
    cd "$BUILD_DIR"

    IWYU_CMD="iwyu_tool.py"
    if ! command -v iwyu_tool.py &> /dev/null; then
        IWYU_CMD="iwyu_tool"
    fi

    $IWYU_CMD -p . -- -Xiwyu --no_fwd_decls > "$REPORTS_DIR/iwyu.log" 2>&1 || true

    # IWYU always has suggestions, so we don't fail on it
    echo -e "${GREEN}  ✓ IWYU analysis complete${NC}"
    echo -e "${BLUE}  → Report: $REPORTS_DIR/iwyu.log${NC}"
else
    echo -e "${YELLOW}  ⚠ include-what-you-use not found, skipping${NC}"
fi
echo ""

# ============================================================================
# 5. Unsafe Function Audit
# ============================================================================
echo -e "${YELLOW}[5/5] Running unsafe function audit...${NC}"
cd "$PROJECT_ROOT"

if [ -f "scripts/audit-unsafe-functions.sh" ]; then
    if bash scripts/audit-unsafe-functions.sh > "$REPORTS_DIR/unsafe-functions.log" 2>&1; then
        echo -e "${GREEN}  ✓ No unsafe functions found${NC}"
    else
        echo -e "${RED}  ✗ Unsafe functions detected${NC}"
        cat "$REPORTS_DIR/unsafe-functions.log"
        EXIT_CODE=1
    fi
else
    echo -e "${YELLOW}  ⚠ audit-unsafe-functions.sh not found${NC}"
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                      Analysis Summary                      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Reports directory: $REPORTS_DIR"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All static analysis checks passed!${NC}"
else
    echo -e "${RED}✗ Some static analysis checks failed. Review reports above.${NC}"
fi

exit $EXIT_CODE
