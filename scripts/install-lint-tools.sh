#!/bin/bash
#=============================================================================
# NIMCP Linting Tools Installation Script
#=============================================================================
# Description: Install all required linting and code quality tools
# Usage: ./scripts/install-lint-tools.sh [--check-only]
#
# Installs:
#   - clang-format (code formatting)
#   - clang-tidy (static analysis)
#   - cppcheck (C/C++ static analysis)
#
# Options:
#   --check-only    Only check if tools are installed (don't install)
#   --help, -h      Show this help message
#=============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CHECK_ONLY=0
TOOLS_MISSING=0

#-----------------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------------

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

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

#-----------------------------------------------------------------------------
# Parse Arguments
#-----------------------------------------------------------------------------

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --check-only)
                CHECK_ONLY=1
                shift
                ;;
            --help|-h)
                cat << EOF
Usage: $0 [OPTIONS]

Options:
  --check-only   Check if tools are installed without installing
  --help, -h     Show this help message

This script installs the required linting tools:
  - clang-format: Code formatting
  - clang-tidy:   Static analysis and modernization
  - cppcheck:     Additional C/C++ static analysis

Platform Detection:
  - Ubuntu/Debian: Uses apt-get
  - Fedora/RHEL:   Uses dnf
  - macOS:         Uses homebrew

Examples:
  $0                  # Install all missing tools
  $0 --check-only     # Check what's installed
EOF
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

#-----------------------------------------------------------------------------
# Detect Platform
#-----------------------------------------------------------------------------

detect_platform() {
    if [[ "$OSTYPE" =~ ^linux ]]; then
        if command -v apt-get &> /dev/null; then
            PLATFORM="debian"
            print_info "Platform: Ubuntu/Debian"
        elif command -v dnf &> /dev/null; then
            PLATFORM="fedora"
            print_info "Platform: Fedora/RHEL"
        elif command -v yum &> /dev/null; then
            PLATFORM="rhel"
            print_info "Platform: RHEL/CentOS (legacy)"
        else
            print_error "Unknown Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" =~ ^darwin ]]; then
        PLATFORM="macos"
        print_info "Platform: macOS"
        if ! command -v brew &> /dev/null; then
            print_error "Homebrew not found. Please install from https://brew.sh"
            exit 1
        fi
    else
        print_error "Unsupported platform: $OSTYPE"
        exit 1
    fi
}

#-----------------------------------------------------------------------------
# Check Tool Status
#-----------------------------------------------------------------------------

check_tool() {
    local tool=$1
    local package=$2
    
    if command -v "$tool" &> /dev/null; then
        local version
        case $tool in
            clang-format)
                version=$(clang-format --version 2>&1 | head -n1)
                ;;
            clang-tidy)
                version=$(clang-tidy --version 2>&1 | head -n1)
                ;;
            cppcheck)
                version=$(cppcheck --version 2>&1)
                ;;
        esac
        print_success "$tool installed: $version"
        return 0
    else
        print_warning "$tool not found (package: $package)"
        TOOLS_MISSING=1
        return 1
    fi
}

check_all_tools() {
    print_header "Checking Installed Tools"
    
    case $PLATFORM in
        debian)
            check_tool "clang-format" "clang-format"
            check_tool "clang-tidy" "clang-tidy"
            check_tool "cppcheck" "cppcheck"
            ;;
        fedora|rhel)
            check_tool "clang-format" "clang-tools-extra"
            check_tool "clang-tidy" "clang-tools-extra"
            check_tool "cppcheck" "cppcheck"
            ;;
        macos)
            check_tool "clang-format" "clang-format"
            check_tool "clang-tidy" "llvm"
            check_tool "cppcheck" "cppcheck"
            ;;
    esac
    
    echo ""
}

#-----------------------------------------------------------------------------
# Install Tools
#-----------------------------------------------------------------------------

install_tools() {
    if [[ $TOOLS_MISSING -eq 0 ]]; then
        print_success "All tools are already installed!"
        return 0
    fi
    
    if [[ $CHECK_ONLY -eq 1 ]]; then
        print_info "Run without --check-only to install missing tools"
        return 0
    fi
    
    print_header "Installing Missing Tools"
    
    case $PLATFORM in
        debian)
            print_info "Updating package lists..."
            sudo apt-get update
            
            print_info "Installing linting tools..."
            sudo apt-get install -y \
                clang-format \
                clang-tidy \
                cppcheck
            ;;
        fedora)
            print_info "Installing linting tools..."
            sudo dnf install -y \
                clang-tools-extra \
                cppcheck
            ;;
        rhel)
            print_info "Installing linting tools..."
            sudo yum install -y \
                clang-tools-extra \
                cppcheck
            ;;
        macos)
            print_info "Installing linting tools..."
            brew install clang-format llvm cppcheck
            
            # Add LLVM to PATH if needed
            if [[ -d "/usr/local/opt/llvm/bin" ]]; then
                print_info "Add to ~/.zshrc or ~/.bash_profile:"
                echo "  export PATH=\"/usr/local/opt/llvm/bin:\$PATH\""
            fi
            ;;
    esac
    
    echo ""
    print_success "Installation complete!"
}

#-----------------------------------------------------------------------------
# Verify Installation
#-----------------------------------------------------------------------------

verify_installation() {
    print_header "Verifying Installation"
    
    local all_ok=1
    
    for tool in clang-format clang-tidy cppcheck; do
        if command -v "$tool" &> /dev/null; then
            print_success "$tool is available"
        else
            print_error "$tool is NOT available"
            all_ok=0
        fi
    done
    
    echo ""
    
    if [[ $all_ok -eq 1 ]]; then
        print_success "All tools verified successfully!"
        echo ""
        print_info "You can now run:"
        echo "  ./scripts/lint.sh       # Run all checks"
        echo "  ./scripts/format.sh     # Format code"
    else
        print_error "Some tools failed to install"
        print_info "Please check the error messages above"
        exit 1
    fi
}

#-----------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------

main() {
    parse_args "$@"
    
    print_header "NIMCP Linting Tools Installation"
    
    detect_platform
    check_all_tools
    
    if [[ $CHECK_ONLY -eq 0 ]]; then
        install_tools
        check_all_tools
        verify_installation
    else
        if [[ $TOOLS_MISSING -eq 1 ]]; then
            exit 1
        fi
    fi
}

main "$@"
