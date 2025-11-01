#!/bin/bash
#=============================================================================
# NIMCP Code Formatting Script
#=============================================================================
# Description: Automatically format C/C++ code using clang-format
# Usage: ./scripts/format.sh [--check] [--verbose] [path...]
# Exit codes:
#   0 - Success (all files formatted or already formatted)
#   1 - Formatting needed (in --check mode)
#   2 - clang-format not found
#   3 - Script error
#
# Options:
#   --check     Check formatting without modifying files
#   --verbose   Show detailed output for each file
#   path...     Specific files/directories to format (default: src/ examples/)
#
# Environment variables:
#   NIMCP_FORMAT_STYLE  - Override style (default: file)
#=============================================================================

set -o pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configuration
CHECK_ONLY=0
VERBOSE=0
EXIT_CODE=0
STYLE="${NIMCP_FORMAT_STYLE:-file}"

# Paths to format
DEFAULT_PATHS=("${PROJECT_ROOT}/src" "${PROJECT_ROOT}/examples" "${PROJECT_ROOT}/bindings")
FORMAT_PATHS=()

# Statistics
TOTAL_FILES=0
FORMATTED_FILES=0
ALREADY_FORMATTED=0

#-----------------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------------

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

#-----------------------------------------------------------------------------
# Parse Arguments
#-----------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check)
                CHECK_ONLY=1
                shift
                ;;
            --verbose)
                VERBOSE=1
                shift
                ;;
            --help|-h)
                cat << EOF
Usage: $0 [OPTIONS] [PATHS...]

Options:
  --check        Check formatting without modifying files (exit 1 if changes needed)
  --verbose      Show detailed output for each file processed
  --help, -h     Show this help message

Paths:
  Specific files or directories to format. If not provided, formats:
    - src/
    - examples/
    - bindings/

Environment Variables:
  NIMCP_FORMAT_STYLE  - Override clang-format style (default: file)

Examples:
  $0                              # Format all default paths
  $0 --check                      # Check if formatting is needed
  $0 --verbose src/               # Format src/ with verbose output
  $0 src/lib/nimcp_brain.c        # Format specific file
EOF
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 3
                ;;
            *)
                FORMAT_PATHS+=("$1")
                shift
                ;;
        esac
    done

    # Use default paths if none specified
    if [[ ${#FORMAT_PATHS[@]} -eq 0 ]]; then
        FORMAT_PATHS=("${DEFAULT_PATHS[@]}")
    fi

    # Validate paths exist
    for path in "${FORMAT_PATHS[@]}"; do
        if [[ ! -e "$path" ]]; then
            print_error "Path does not exist: $path"
            exit 3
        fi
    done
}

#-----------------------------------------------------------------------------
# Check Tool Availability
#-----------------------------------------------------------------------------

check_clang_format() {
    print_header "Checking clang-format"

    if ! command -v clang-format &> /dev/null; then
        print_error "clang-format not found!"
        echo ""
        echo "Installation instructions:"
        echo "  Ubuntu/Debian: sudo apt-get install clang-format"
        echo "  Fedora:        sudo dnf install clang-tools-extra"
        echo "  macOS:         brew install clang-format"
        echo ""
        exit 2
    fi

    local version=$(clang-format --version | head -n1)
    print_success "Found: $version"

    # Check for .clang-format config
    if [[ -f "${PROJECT_ROOT}/.clang-format" ]]; then
        print_success "Using config: ${PROJECT_ROOT}/.clang-format"
    else
        print_warning "No .clang-format config found, using defaults"
    fi

    echo ""
}

#-----------------------------------------------------------------------------
# Find Files to Format
#-----------------------------------------------------------------------------

find_source_files() {
    local paths=("$@")
    local files=()

    for path in "${paths[@]}"; do
        if [[ -f "$path" ]]; then
            # Single file - check if it's a C/C++ file
            case "$path" in
                *.c|*.cpp|*.h|*.hpp)
                    files+=("$path")
                    ;;
                *)
                    print_warning "Skipping non-C/C++ file: $path"
                    ;;
            esac
        elif [[ -d "$path" ]]; then
            # Directory - find all C/C++ files
            # Exclude build directories, third-party code, and generated files
            while IFS= read -r -d '' file; do
                files+=("$file")
            done < <(find "$path" -type f \
                \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) \
                ! -path "*/build/*" \
                ! -path "*/CMakeFiles/*" \
                ! -path "*/.git/*" \
                ! -path "*/third_party/*" \
                ! -path "*/vendor/*" \
                -print0)
        fi
    done

    # Remove duplicates and sort
    printf '%s\n' "${files[@]}" | sort -u
}

#-----------------------------------------------------------------------------
# Format Files
#-----------------------------------------------------------------------------

format_files() {
    local files=()
    while IFS= read -r file; do
        files+=("$file")
    done < <(find_source_files "${FORMAT_PATHS[@]}")

    TOTAL_FILES=${#files[@]}

    if [[ $TOTAL_FILES -eq 0 ]]; then
        print_warning "No C/C++ files found to format"
        return 0
    fi

    if [[ $CHECK_ONLY -eq 1 ]]; then
        print_header "Checking Formatting ($TOTAL_FILES files)"
    else
        print_header "Formatting Code ($TOTAL_FILES files)"
    fi

    local needs_formatting=()

    for file in "${files[@]}"; do
        if [[ $CHECK_ONLY -eq 1 ]]; then
            # Check mode - see if file needs formatting
            local formatted_content=$(clang-format -style="$STYLE" "$file" 2>/dev/null)
            local original_content=$(cat "$file")

            if [[ "$formatted_content" != "$original_content" ]]; then
                needs_formatting+=("$file")
                if [[ $VERBOSE -eq 1 ]]; then
                    print_warning "Needs formatting: $file"
                fi
            else
                ALREADY_FORMATTED=$((ALREADY_FORMATTED + 1))
                if [[ $VERBOSE -eq 1 ]]; then
                    print_success "Already formatted: $file"
                fi
            fi
        else
            # Format mode - apply formatting
            local temp_file=$(mktemp)
            if clang-format -style="$STYLE" "$file" > "$temp_file" 2>/dev/null; then
                if ! cmp -s "$file" "$temp_file"; then
                    mv "$temp_file" "$file"
                    FORMATTED_FILES=$((FORMATTED_FILES + 1))
                    if [[ $VERBOSE -eq 1 ]]; then
                        print_success "Formatted: $file"
                    fi
                else
                    ALREADY_FORMATTED=$((ALREADY_FORMATTED + 1))
                    rm "$temp_file"
                    if [[ $VERBOSE -eq 1 ]]; then
                        print_info "No changes: $file"
                    fi
                fi
            else
                print_error "Failed to format: $file"
                rm -f "$temp_file"
                EXIT_CODE=3
            fi
        fi
    done

    echo ""

    # Report results
    if [[ $CHECK_ONLY -eq 1 ]]; then
        local needs_count=${#needs_formatting[@]}
        if [[ $needs_count -gt 0 ]]; then
            print_error "$needs_count file(s) need formatting:"
            for file in "${needs_formatting[@]}"; do
                echo "  - $file"
            done
            echo ""
            print_info "Run without --check to apply formatting"
            EXIT_CODE=1
        else
            print_success "All $TOTAL_FILES files are properly formatted!"
        fi
    else
        if [[ $FORMATTED_FILES -gt 0 ]]; then
            print_success "Formatted $FORMATTED_FILES file(s)"
        fi
        if [[ $ALREADY_FORMATTED -gt 0 ]]; then
            print_info "$ALREADY_FORMATTED file(s) were already formatted"
        fi
        if [[ $EXIT_CODE -eq 0 ]]; then
            print_success "Formatting complete!"
        fi
    fi
}

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------

print_summary() {
    print_header "Summary"

    echo "Total files processed: $TOTAL_FILES"

    if [[ $CHECK_ONLY -eq 1 ]]; then
        local needs_formatting=$((TOTAL_FILES - ALREADY_FORMATTED))
        echo "Files needing formatting: $needs_formatting"
        echo "Files already formatted: $ALREADY_FORMATTED"
    else
        echo "Files formatted: $FORMATTED_FILES"
        echo "Files unchanged: $ALREADY_FORMATTED"
    fi

    echo ""

    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "Success!"
    else
        print_error "Failed - see errors above"
    fi
}

#-----------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------

main() {
    cd "$PROJECT_ROOT" || exit 3

    parse_args "$@"
    check_clang_format
    format_files
    print_summary

    exit $EXIT_CODE
}

main "$@"
