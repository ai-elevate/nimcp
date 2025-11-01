#!/bin/bash
#=============================================================================
# NIMCP Code Quality Lint Script
#=============================================================================
# Description: Comprehensive linting for C/C++ codebase
# Usage: ./scripts/lint.sh [--fix] [--verbose] [path...]
# Exit codes:
#   0 - All checks passed
#   1 - Linting issues found
#   2 - Required tools not found
#   3 - Script error
#
# Options:
#   --fix       Apply fixes where possible (clang-format, clang-tidy)
#   --verbose   Show detailed output
#   path...     Specific files/directories to check (default: src/ examples/)
#
# Environment variables:
#   NIMCP_LINT_SKIP_FORMAT    - Skip clang-format check
#   NIMCP_LINT_SKIP_TIDY      - Skip clang-tidy check
#   NIMCP_LINT_SKIP_CPPCHECK  - Skip cppcheck
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
FIX_MODE=0
VERBOSE=0
EXIT_CODE=0
CHECKS_RUN=0
CHECKS_FAILED=0

# Paths to check
DEFAULT_PATHS=("${PROJECT_ROOT}/src" "${PROJECT_ROOT}/examples")
CHECK_PATHS=()

# Tool availability
HAS_CLANG_FORMAT=0
HAS_CLANG_TIDY=0
HAS_CPPCHECK=0

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
            --fix)
                FIX_MODE=1
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
  --fix          Apply automatic fixes where possible
  --verbose      Show detailed output from all tools
  --help, -h     Show this help message

Paths:
  Specific files or directories to check. If not provided, checks:
    - src/
    - examples/

Environment Variables:
  NIMCP_LINT_SKIP_FORMAT    - Skip clang-format check
  NIMCP_LINT_SKIP_TIDY      - Skip clang-tidy check
  NIMCP_LINT_SKIP_CPPCHECK  - Skip cppcheck

Examples:
  $0                           # Check all default paths
  $0 --fix                     # Check and fix issues
  $0 src/lib/nimcp_brain.c     # Check specific file
  $0 --verbose src/            # Verbose output for src/
EOF
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 3
                ;;
            *)
                CHECK_PATHS+=("$1")
                shift
                ;;
        esac
    done

    # Use default paths if none specified
    if [[ ${#CHECK_PATHS[@]} -eq 0 ]]; then
        CHECK_PATHS=("${DEFAULT_PATHS[@]}")
    fi

    # Validate paths exist
    for path in "${CHECK_PATHS[@]}"; do
        if [[ ! -e "$path" ]]; then
            print_error "Path does not exist: $path"
            exit 3
        fi
    done
}

#-----------------------------------------------------------------------------
# Check Tool Availability
#-----------------------------------------------------------------------------

check_tools() {
    print_header "Checking Required Tools"

    if command -v clang-format &> /dev/null; then
        HAS_CLANG_FORMAT=1
        VERSION=$(clang-format --version | head -n1)
        print_success "clang-format found: $VERSION"
    else
        print_warning "clang-format not found"
        echo "  Install: sudo apt-get install clang-format"
    fi

    if command -v clang-tidy &> /dev/null; then
        HAS_CLANG_TIDY=1
        VERSION=$(clang-tidy --version | head -n1)
        print_success "clang-tidy found: $VERSION"
    else
        print_warning "clang-tidy not found"
        echo "  Install: sudo apt-get install clang-tidy"
    fi

    if command -v cppcheck &> /dev/null; then
        HAS_CPPCHECK=1
        VERSION=$(cppcheck --version)
        print_success "cppcheck found: $VERSION"
    else
        print_warning "cppcheck not found"
        echo "  Install: sudo apt-get install cppcheck"
    fi

    echo ""

    # Check if at least one tool is available
    if [[ $HAS_CLANG_FORMAT -eq 0 && $HAS_CLANG_TIDY -eq 0 && $HAS_CPPCHECK -eq 0 ]]; then
        print_error "No linting tools found!"
        print_info "Please install at least one of: clang-format, clang-tidy, cppcheck"
        exit 2
    fi
}

#-----------------------------------------------------------------------------
# Find Files to Check
#-----------------------------------------------------------------------------

find_source_files() {
    local paths=("$@")
    local files=()

    for path in "${paths[@]}"; do
        if [[ -f "$path" ]]; then
            # Single file
            files+=("$path")
        elif [[ -d "$path" ]]; then
            # Directory - find all C/C++ files
            while IFS= read -r -d '' file; do
                files+=("$file")
            done < <(find "$path" -type f \( -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.hpp" \) -print0)
        fi
    done

    # Remove duplicates and sort
    printf '%s\n' "${files[@]}" | sort -u
}

#-----------------------------------------------------------------------------
# clang-format Check
#-----------------------------------------------------------------------------

run_clang_format() {
    [[ $HAS_CLANG_FORMAT -eq 0 ]] && return 0
    [[ -n "${NIMCP_LINT_SKIP_FORMAT}" ]] && return 0

    CHECKS_RUN=$((CHECKS_RUN + 1))
    print_header "Running clang-format"

    local files=()
    while IFS= read -r file; do
        files+=("$file")
    done < <(find_source_files "${CHECK_PATHS[@]}")

    if [[ ${#files[@]} -eq 0 ]]; then
        print_warning "No files found to format"
        return 0
    fi

    print_info "Checking ${#files[@]} files..."

    local format_issues=0
    local temp_output=$(mktemp)

    for file in "${files[@]}"; do
        if [[ $FIX_MODE -eq 1 ]]; then
            # Apply fixes
            if [[ $VERBOSE -eq 1 ]]; then
                clang-format -i -style=file "$file"
                echo "  Formatted: $file"
            else
                clang-format -i -style=file "$file" 2>/dev/null
            fi
        else
            # Check only
            if ! clang-format --dry-run -Werror -style=file "$file" 2>&1 | tee -a "$temp_output" > /dev/null; then
                format_issues=$((format_issues + 1))
                if [[ $VERBOSE -eq 1 ]]; then
                    echo "  Issues in: $file"
                fi
            fi
        fi
    done

    if [[ $FIX_MODE -eq 1 ]]; then
        print_success "Formatting applied to ${#files[@]} files"
    else
        if [[ $format_issues -gt 0 ]]; then
            print_error "Found formatting issues in $format_issues file(s)"
            print_info "Run with --fix to apply formatting automatically"
            CHECKS_FAILED=$((CHECKS_FAILED + 1))
            EXIT_CODE=1
        else
            print_success "All files are properly formatted"
        fi
    fi

    rm -f "$temp_output"
    echo ""
}

#-----------------------------------------------------------------------------
# clang-tidy Check
#-----------------------------------------------------------------------------

run_clang_tidy() {
    [[ $HAS_CLANG_TIDY -eq 0 ]] && return 0
    [[ -n "${NIMCP_LINT_SKIP_TIDY}" ]] && return 0

    CHECKS_RUN=$((CHECKS_RUN + 1))
    print_header "Running clang-tidy"

    local files=()
    while IFS= read -r file; do
        files+=("$file")
    done < <(find_source_files "${CHECK_PATHS[@]}")

    if [[ ${#files[@]} -eq 0 ]]; then
        print_warning "No files found to analyze"
        return 0
    fi

    print_info "Analyzing ${#files[@]} files..."

    local tidy_issues=0
    local temp_output=$(mktemp)
    local build_dir="${PROJECT_ROOT}/build"

    # Prepare clang-tidy arguments
    local tidy_args=()
    if [[ $FIX_MODE -eq 1 ]]; then
        tidy_args+=("--fix" "--fix-errors")
    fi
    if [[ -d "$build_dir" ]]; then
        tidy_args+=("-p" "$build_dir")
    fi

    for file in "${files[@]}"; do
        if [[ $VERBOSE -eq 1 ]]; then
            echo "  Checking: $file"
            if ! clang-tidy "${tidy_args[@]}" "$file" 2>&1 | tee -a "$temp_output"; then
                tidy_issues=$((tidy_issues + 1))
            fi
        else
            if ! clang-tidy "${tidy_args[@]}" "$file" 2>&1 >> "$temp_output"; then
                tidy_issues=$((tidy_issues + 1))
                echo "  Issues in: $file"
            fi
        fi
    done

    # Check if any warnings were generated
    if grep -q "warning:" "$temp_output" || grep -q "error:" "$temp_output"; then
        tidy_issues=$((tidy_issues + 1))
    fi

    if [[ $tidy_issues -gt 0 ]]; then
        print_error "clang-tidy found issues"
        if [[ $VERBOSE -eq 0 ]]; then
            print_info "Run with --verbose to see detailed output"
        fi
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        EXIT_CODE=1
    else
        print_success "No issues found by clang-tidy"
    fi

    rm -f "$temp_output"
    echo ""
}

#-----------------------------------------------------------------------------
# cppcheck
#-----------------------------------------------------------------------------

run_cppcheck() {
    [[ $HAS_CPPCHECK -eq 0 ]] && return 0
    [[ -n "${NIMCP_LINT_SKIP_CPPCHECK}" ]] && return 0

    CHECKS_RUN=$((CHECKS_RUN + 1))
    print_header "Running cppcheck"

    local paths_str="${CHECK_PATHS[*]}"
    print_info "Analyzing paths: $paths_str"

    local temp_output=$(mktemp)
    local cppcheck_args=(
        "--enable=all"
        "--inconclusive"
        "--std=c99"
        "--std=c++17"
        "--language=c"
        "--language=c++"
        "--suppress=missingIncludeSystem"
        "--suppress=unmatchedSuppression"
        "--suppress=unusedFunction"
        "--inline-suppr"
        "-I${PROJECT_ROOT}/src/include"
        "-I${PROJECT_ROOT}/src/include/utils"
        "-I${PROJECT_ROOT}/src/include/logging"
        "--error-exitcode=1"
    )

    if [[ $VERBOSE -eq 1 ]]; then
        cppcheck_args+=("--verbose")
    else
        cppcheck_args+=("--quiet")
    fi

    # Run cppcheck
    local cppcheck_exit=0
    if [[ $VERBOSE -eq 1 ]]; then
        cppcheck "${cppcheck_args[@]}" "${CHECK_PATHS[@]}" 2>&1 | tee "$temp_output" || cppcheck_exit=$?
    else
        cppcheck "${cppcheck_args[@]}" "${CHECK_PATHS[@]}" 2>&1 > "$temp_output" || cppcheck_exit=$?
    fi

    # Parse results
    local error_count=$(grep -c "error:" "$temp_output" 2>/dev/null || echo 0)
    local warning_count=$(grep -c "warning:" "$temp_output" 2>/dev/null || echo 0)

    if [[ $cppcheck_exit -ne 0 ]] || [[ $error_count -gt 0 ]]; then
        print_error "cppcheck found $error_count error(s) and $warning_count warning(s)"
        if [[ $VERBOSE -eq 0 ]]; then
            cat "$temp_output"
            print_info "Run with --verbose for more details"
        fi
        CHECKS_FAILED=$((CHECKS_FAILED + 1))
        EXIT_CODE=1
    elif [[ $warning_count -gt 0 ]]; then
        print_warning "cppcheck found $warning_count warning(s)"
        if [[ $VERBOSE -eq 0 ]]; then
            cat "$temp_output"
        fi
        # Warnings don't fail the build
    else
        print_success "No issues found by cppcheck"
    fi

    rm -f "$temp_output"
    echo ""
}

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------

print_summary() {
    print_header "Lint Summary"

    echo "Checks run: $CHECKS_RUN"
    echo "Checks failed: $CHECKS_FAILED"
    echo ""

    if [[ $EXIT_CODE -eq 0 ]]; then
        print_success "All checks passed! ✓"
    else
        print_error "Some checks failed. Please fix the issues above."
        if [[ $FIX_MODE -eq 0 ]]; then
            print_info "Tip: Run with --fix to automatically fix some issues"
        fi
    fi
}

#-----------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------

main() {
    cd "$PROJECT_ROOT" || exit 3

    parse_args "$@"
    check_tools
    run_clang_format
    run_clang_tidy
    run_cppcheck
    print_summary

    exit $EXIT_CODE
}

main "$@"
