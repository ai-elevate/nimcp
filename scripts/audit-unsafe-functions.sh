#!/bin/bash
#==============================================================================
# Unsafe Function Auditor for NIMCP
#==============================================================================
# Scans C source files for unsafe string functions and provides remediation
# guidance.
#
# Usage: ./scripts/audit-unsafe-functions.sh [path]
#==============================================================================

set -euo pipefail

# Color codes
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default search path
SEARCH_PATH="${1:-src/lib}"

# Unsafe functions to detect
UNSAFE_FUNCTIONS=(
    "strcpy"
    "strcat"
    "sprintf"
    "gets"
    "scanf"
    "vsprintf"
    "strncpy"  # Can be unsafe if not null-terminated
)

# Safe alternatives
declare -A SAFE_ALTERNATIVES
SAFE_ALTERNATIVES["strcpy"]="snprintf or strlcpy"
SAFE_ALTERNATIVES["strcat"]="strncat with size check or snprintf"
SAFE_ALTERNATIVES["sprintf"]="snprintf"
SAFE_ALTERNATIVES["gets"]="fgets"
SAFE_ALTERNATIVES["scanf"]="fgets + sscanf"
SAFE_ALTERNATIVES["vsprintf"]="vsnprintf"
SAFE_ALTERNATIVES["strncpy"]="snprintf (ensures null termination)"

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}NIMCP Unsafe Function Audit${NC}"
echo -e "${BLUE}===============================================================================${NC}"
echo ""

total_issues=0
files_with_issues=0

# Create temporary file for results
TMPFILE=$(mktemp)
trap "rm -f $TMPFILE" EXIT

echo -e "${YELLOW}Scanning: ${SEARCH_PATH}${NC}"
echo ""

# Search for each unsafe function
for func in "${UNSAFE_FUNCTIONS[@]}"; do
    # Skip snprintf and other safe variants
    pattern="${func}\s*\("

    # Find occurrences
    if grep -rn "$pattern" "$SEARCH_PATH" --include="*.c" --include="*.h" > "$TMPFILE" 2>/dev/null; then
        count=$(wc -l < "$TMPFILE")

        if [ "$count" -gt 0 ]; then
            echo -e "${RED}✗ Found ${count} usage(s) of ${func}()${NC}"
            echo -e "  ${BLUE}Safe alternative: ${SAFE_ALTERNATIVES[$func]}${NC}"
            echo ""

            # Display occurrences with context
            while IFS= read -r line; do
                file=$(echo "$line" | cut -d: -f1)
                linenum=$(echo "$line" | cut -d: -f2)
                code=$(echo "$line" | cut -d: -f3-)

                echo -e "  ${YELLOW}${file}:${linenum}${NC}"
                echo -e "    ${code}"
                echo ""

                ((total_issues++))
            done < "$TMPFILE"

            ((files_with_issues++))
        fi
    fi
done

echo -e "${BLUE}===============================================================================${NC}"
echo -e "${BLUE}Summary${NC}"
echo -e "${BLUE}===============================================================================${NC}"

if [ "$total_issues" -eq 0 ]; then
    echo -e "${GREEN}✓ No unsafe function usage detected!${NC}"
    exit 0
else
    echo -e "${RED}✗ Found ${total_issues} unsafe function call(s) in ${files_with_issues} location(s)${NC}"
    echo ""
    echo -e "${YELLOW}Remediation Guide:${NC}"
    echo ""
    echo "1. Replace strcpy() with snprintf():"
    echo "   Before: strcpy(dest, src);"
    echo "   After:  snprintf(dest, sizeof(dest), \"%s\", src);"
    echo ""
    echo "2. Replace strcat() with snprintf():"
    echo "   Before: strcat(dest, src);"
    echo "   After:  size_t len = strlen(dest);"
    echo "           snprintf(dest + len, sizeof(dest) - len, \"%s\", src);"
    echo ""
    echo "3. Replace sprintf() with snprintf():"
    echo "   Before: sprintf(buffer, format, args...);"
    echo "   After:  snprintf(buffer, sizeof(buffer), format, args...);"
    echo ""
    echo "4. For fixed strings, direct assignment is safe:"
    echo "   strcpy(result->explanation, \"Null engine\");"
    echo "   → Can be replaced with snprintf for consistency."
    echo ""

    exit 1
fi
