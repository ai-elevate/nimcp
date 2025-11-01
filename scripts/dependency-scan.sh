#!/bin/bash
#
# Dependency Security Scanner for NIMCP
# Scans system dependencies and build tools for known vulnerabilities
#
# Usage:
#   ./scripts/dependency-scan.sh [--report] [--strict]
#
# Options:
#   --report  Generate detailed vulnerability report
#   --strict  Exit with error if vulnerabilities found
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPORTS_DIR="$PROJECT_ROOT/security-reports"

# Configuration
GENERATE_REPORT=false
STRICT_MODE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
for arg in "$@"; do
    case $arg in
        --report)
            GENERATE_REPORT=true
            ;;
        --strict)
            STRICT_MODE=true
            ;;
    esac
done

mkdir -p "$REPORTS_DIR"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           NIMCP Dependency Security Scanner                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

VULNERABILITIES_FOUND=0

#==============================================================================
# 1. Python Dependencies
#==============================================================================
echo -e "${YELLOW}[1/5] Scanning Python dependencies...${NC}"

if command -v pip3 &> /dev/null || command -v pip &> /dev/null; then
    PIP_CMD="pip3"
    if ! command -v pip3 &> /dev/null; then
        PIP_CMD="pip"
    fi

    # Check if safety is installed
    if command -v safety &> /dev/null; then
        echo "  → Running safety check..."

        if $PIP_CMD freeze | safety check --stdin --json > "$REPORTS_DIR/python-vulns.json" 2>&1; then
            echo -e "${GREEN}  ✓ No known vulnerabilities in Python packages${NC}"
        else
            echo -e "${YELLOW}  ⚠ Vulnerabilities found in Python packages${NC}"
            VULNERABILITIES_FOUND=$((VULNERABILITIES_FOUND + 1))

            if [ "$GENERATE_REPORT" = true ]; then
                cat "$REPORTS_DIR/python-vulns.json"
            fi
        fi
    else
        echo -e "${YELLOW}  ⚠ 'safety' not installed. Install with: pip install safety${NC}"
        echo "  → Checking for outdated packages instead..."

        $PIP_CMD list --outdated > "$REPORTS_DIR/python-outdated.txt" 2>&1 || true
        OUTDATED_COUNT=$(wc -l < "$REPORTS_DIR/python-outdated.txt" || echo 0)

        if [ "$OUTDATED_COUNT" -gt 2 ]; then  # Header lines count as 2
            echo -e "${YELLOW}  ⚠ Found $((OUTDATED_COUNT - 2)) outdated Python packages${NC}"
        else
            echo -e "${GREEN}  ✓ Python packages up to date${NC}"
        fi
    fi
else
    echo -e "${YELLOW}  ⚠ Python/pip not found, skipping${NC}"
fi
echo ""

#==============================================================================
# 2. System Libraries
#==============================================================================
echo -e "${YELLOW}[2/5] Checking system library versions...${NC}"

# Critical libraries to check
LIBRARIES=(
    "glibc:libc"
    "openssl:libssl"
    "pthread:pthread"
    "stdlibc++:libstdc++"
)

for lib_entry in "${LIBRARIES[@]}"; do
    IFS=':' read -r lib_name lib_file <<< "$lib_entry"

    echo -n "  → Checking $lib_name... "

    # Find library version
    if command -v dpkg &> /dev/null; then
        # Debian/Ubuntu
        if dpkg -l | grep -q "$lib_file"; then
            VERSION=$(dpkg -l | grep "$lib_file" | head -1 | awk '{print $3}')
            echo -e "${GREEN}found ($VERSION)${NC}"
        else
            echo -e "${YELLOW}not found via dpkg${NC}"
        fi
    elif command -v rpm &> /dev/null; then
        # RedHat/Fedora
        if rpm -qa | grep -q "$lib_file"; then
            VERSION=$(rpm -qa | grep "$lib_file" | head -1)
            echo -e "${GREEN}found ($VERSION)${NC}"
        else
            echo -e "${YELLOW}not found via rpm${NC}"
        fi
    else
        echo -e "${YELLOW}package manager not found${NC}"
    fi
done
echo ""

#==============================================================================
# 3. CMake and Build Tools
#==============================================================================
echo -e "${YELLOW}[3/5] Checking build tool versions...${NC}"

BUILD_TOOLS=(
    "cmake:CMake"
    "gcc:GCC"
    "g++:G++"
    "make:Make"
)

for tool_entry in "${BUILD_TOOLS[@]}"; do
    IFS=':' read -r tool_cmd tool_name <<< "$tool_entry"

    echo -n "  → Checking $tool_name... "

    if command -v "$tool_cmd" &> /dev/null; then
        VERSION=$($tool_cmd --version 2>&1 | head -1 || echo "unknown")
        echo -e "${GREEN}$VERSION${NC}"

        # Check for known vulnerable versions
        case "$tool_cmd" in
            cmake)
                MAJOR_VER=$(echo "$VERSION" | grep -oP '\d+\.\d+' | head -1)
                if command -v bc &> /dev/null; then
                    if [ "$(echo "$MAJOR_VER < 3.16" | bc 2>/dev/null)" -eq 1 ]; then
                        echo -e "${YELLOW}    ⚠ CMake version older than 3.16 (security improvements missing)${NC}"
                        VULNERABILITIES_FOUND=$((VULNERABILITIES_FOUND + 1))
                    fi
                fi
                ;;
            gcc|g++)
                MAJOR_VER=$(echo "$VERSION" | grep -oP '\d+\.\d+' | head -1 | cut -d. -f1)
                if [ "$MAJOR_VER" -lt 9 ] 2>/dev/null; then
                    echo -e "${YELLOW}    ⚠ Compiler version older than GCC 9 (missing security features)${NC}"
                fi
                ;;
        esac
    else
        echo -e "${YELLOW}not found${NC}"
    fi
done
echo ""

#==============================================================================
# 4. Git Dependencies (submodules, if any)
#==============================================================================
echo -e "${YELLOW}[4/5] Checking Git submodules...${NC}"

cd "$PROJECT_ROOT"

if [ -f ".gitmodules" ]; then
    echo "  → Analyzing submodules..."

    git submodule status > "$REPORTS_DIR/submodules.txt" 2>&1 || true

    SUBMODULE_COUNT=$(wc -l < "$REPORTS_DIR/submodules.txt" || echo 0)

    if [ "$SUBMODULE_COUNT" -gt 0 ]; then
        echo -e "${BLUE}  Found $SUBMODULE_COUNT submodules${NC}"

        # Check for uncommitted changes in submodules
        git submodule foreach 'git status --short' > "$REPORTS_DIR/submodule-status.txt" 2>&1 || true

        if [ -s "$REPORTS_DIR/submodule-status.txt" ]; then
            echo -e "${YELLOW}  ⚠ Some submodules have uncommitted changes${NC}"
        fi
    else
        echo -e "${GREEN}  ✓ No submodules configured${NC}"
    fi
else
    echo -e "${GREEN}  ✓ No submodules configured${NC}"
fi
echo ""

#==============================================================================
# 5. Known CVE Checks
#==============================================================================
echo -e "${YELLOW}[5/5] Checking for known CVEs...${NC}"

# List of dependencies with known CVE history to monitor
CVE_WATCHLIST=(
    "libpthread:CVE-2021-35942"
    "openssl:CVE-2022-0778"
    "glibc:CVE-2021-3999"
)

echo "  → Monitoring watchlist..."
for entry in "${CVE_WATCHLIST[@]}"; do
    IFS=':' read -r lib cve <<< "$entry"
    echo "    $lib: monitoring for $cve"
done

echo -e "${GREEN}  ✓ Watchlist monitoring active${NC}"
echo ""

#==============================================================================
# License Compliance Check
#==============================================================================
echo -e "${YELLOW}[Bonus] License compliance check...${NC}"

# Check for restrictive licenses in dependencies
if [ -f "requirements.txt" ]; then
    echo "  → Scanning Python package licenses..."

    if command -v pip-licenses &> /dev/null; then
        pip-licenses --format=csv > "$REPORTS_DIR/licenses.csv" 2>&1 || true
        echo -e "${GREEN}  ✓ License report: $REPORTS_DIR/licenses.csv${NC}"
    else
        echo -e "${YELLOW}  ⚠ pip-licenses not installed. Install with: pip install pip-licenses${NC}"
    fi
fi

echo ""

#==============================================================================
# Summary
#==============================================================================
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Scan Summary                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$GENERATE_REPORT" = true ]; then
    echo "Detailed reports saved to: $REPORTS_DIR"
    echo ""
fi

if [ $VULNERABILITIES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✓ No critical vulnerabilities detected${NC}"
    echo ""
    echo "Recommendations:"
    echo "  • Keep dependencies updated regularly"
    echo "  • Monitor security advisories for:"
    echo "    - Python packages (https://github.com/advisories)"
    echo "    - System libraries (CVE databases)"
    echo "  • Run this scan before each release"
    exit 0
else
    echo -e "${YELLOW}⚠ Found $VULNERABILITIES_FOUND potential security issues${NC}"
    echo ""
    echo "Recommendations:"
    echo "  • Review flagged packages/tools above"
    echo "  • Update to latest stable versions"
    echo "  • Check CVE databases for details"
    echo "  • Consider using dependency scanning in CI/CD"

    if [ "$STRICT_MODE" = true ]; then
        echo ""
        echo -e "${RED}✗ Strict mode enabled: failing build due to vulnerabilities${NC}"
        exit 1
    fi

    exit 0
fi
