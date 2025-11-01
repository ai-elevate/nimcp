#!/bin/bash
#==============================================================================
# NIMCP Development Environment Setup
#==============================================================================
# Sets up development environment for NIMCP project
# - Installs recommended linting tools
# - Configures pre-commit hooks
# - Verifies build environment
#
# Usage: ./scripts/setup-dev-env.sh [--install-tools]
#==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Options
INSTALL_TOOLS=false

print_header() {
    echo ""
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

# Parse arguments
for arg in "$@"; do
    case $arg in
        --install-tools)
            INSTALL_TOOLS=true
            ;;
        --help|-h)
            cat << EOF
NIMCP Development Environment Setup

Usage: $0 [OPTIONS]

Options:
  --install-tools    Install recommended development tools
  --help, -h         Show this help message

This script will:
  1. Check for required build tools
  2. Check for recommended linting tools
  3. Install pre-commit hooks
  4. Verify project can be built
  5. Run basic tests

EOF
            exit 0
            ;;
    esac
done

cd "$PROJECT_ROOT"

print_header "NIMCP Development Environment Setup"

#==============================================================================
# Check Required Build Tools
#==============================================================================
print_header "Checking Required Build Tools"

MISSING_REQUIRED=()

if ! command -v cmake &> /dev/null; then
    MISSING_REQUIRED+=("cmake")
    print_error "cmake not found"
else
    VERSION=$(cmake --version | head -n1)
    print_success "cmake: $VERSION"
fi

if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
    MISSING_REQUIRED+=("gcc or clang")
    print_error "No C compiler found (gcc or clang)"
else
    if command -v gcc &> /dev/null; then
        VERSION=$(gcc --version | head -n1)
        print_success "gcc: $VERSION"
    fi
    if command -v clang &> /dev/null; then
        VERSION=$(clang --version | head -n1)
        print_success "clang: $VERSION"
    fi
fi

if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    MISSING_REQUIRED+=("g++ or clang++")
    print_error "No C++ compiler found (g++ or clang++)"
else
    if command -v g++ &> /dev/null; then
        VERSION=$(g++ --version | head -n1)
        print_success "g++: $VERSION"
    fi
fi

if ! command -v make &> /dev/null; then
    MISSING_REQUIRED+=("make")
    print_error "make not found"
else
    VERSION=$(make --version | head -n1)
    print_success "make: $VERSION"
fi

if ! command -v python3 &> /dev/null; then
    MISSING_REQUIRED+=("python3")
    print_error "python3 not found"
else
    VERSION=$(python3 --version)
    print_success "python3: $VERSION"
fi

# Check for Python development headers
if ! python3-config --includes &> /dev/null; then
    MISSING_REQUIRED+=("python3-dev")
    print_error "python3 development headers not found"
else
    print_success "python3-dev: installed"
fi

if [ ${#MISSING_REQUIRED[@]} -gt 0 ]; then
    echo ""
    print_error "Missing required tools:"
    for tool in "${MISSING_REQUIRED[@]}"; do
        echo "  - $tool"
    done
    echo ""
    print_info "Install on Ubuntu/Debian:"
    echo "  sudo apt-get install build-essential cmake python3-dev pkg-config"
    exit 1
fi

#==============================================================================
# Check Recommended Linting Tools
#==============================================================================
print_header "Checking Recommended Linting Tools"

MISSING_RECOMMENDED=()

if ! command -v clang-format &> /dev/null; then
    MISSING_RECOMMENDED+=("clang-format")
    print_warning "clang-format not found"
else
    VERSION=$(clang-format --version | head -n1)
    print_success "clang-format: $VERSION"
fi

if ! command -v clang-tidy &> /dev/null; then
    MISSING_RECOMMENDED+=("clang-tidy")
    print_warning "clang-tidy not found"
else
    VERSION=$(clang-tidy --version | head -n1)
    print_success "clang-tidy: $VERSION"
fi

if ! command -v cppcheck &> /dev/null; then
    MISSING_RECOMMENDED+=("cppcheck")
    print_warning "cppcheck not found"
else
    VERSION=$(cppcheck --version)
    print_success "cppcheck: $VERSION"
fi

if ! command -v shellcheck &> /dev/null; then
    MISSING_RECOMMENDED+=("shellcheck")
    print_warning "shellcheck not found"
else
    VERSION=$(shellcheck --version | grep version: | awk '{print $2}')
    print_success "shellcheck: version $VERSION"
fi

if ! command -v lcov &> /dev/null; then
    MISSING_RECOMMENDED+=("lcov")
    print_warning "lcov not found"
else
    VERSION=$(lcov --version | head -n1)
    print_success "lcov: $VERSION"
fi

if ! command -v valgrind &> /dev/null; then
    MISSING_RECOMMENDED+=("valgrind")
    print_warning "valgrind not found"
else
    VERSION=$(valgrind --version)
    print_success "valgrind: $VERSION"
fi

if [ ${#MISSING_RECOMMENDED[@]} -gt 0 ]; then
    echo ""
    print_warning "Missing recommended development tools:"
    for tool in "${MISSING_RECOMMENDED[@]}"; do
        echo "  - $tool"
    done
    echo ""

    if [ "$INSTALL_TOOLS" = true ]; then
        print_info "Installing missing tools..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                clang-format \
                clang-tidy \
                cppcheck \
                shellcheck \
                lcov \
                valgrind
            pip3 install lizard || print_warning "Failed to install lizard"
            print_success "Tools installed successfully"
        else
            print_warning "apt-get not available, please install manually"
        fi
    else
        print_info "Install on Ubuntu/Debian:"
        echo "  sudo apt-get install clang-format clang-tidy cppcheck shellcheck lcov valgrind"
        echo "  pip3 install lizard"
        echo ""
        print_info "Or run with: $0 --install-tools"
    fi
fi

#==============================================================================
# Setup Pre-Commit Hook
#==============================================================================
print_header "Setting Up Pre-Commit Hook"

if [ -d ".git" ]; then
    print_info "Configuring git hooks path..."
    git config core.hooksPath .git-hooks

    if [ -f ".git-hooks/pre-commit" ]; then
        chmod +x .git-hooks/pre-commit
        print_success "Pre-commit hook configured"
        print_info "Hook location: .git-hooks/pre-commit"
        print_info "To bypass: git commit --no-verify"
    else
        print_error "Pre-commit hook not found at .git-hooks/pre-commit"
    fi
else
    print_warning "Not a git repository, skipping hook setup"
fi

#==============================================================================
# Make Scripts Executable
#==============================================================================
print_header "Configuring Scripts"

chmod +x "$SCRIPT_DIR/lint.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/coverage.sh" 2>/dev/null || true
chmod +x "$SCRIPT_DIR/setup-dev-env.sh" 2>/dev/null || true

print_success "Scripts made executable"

#==============================================================================
# Test Build
#==============================================================================
print_header "Testing Build"

if [ ! -d "build" ]; then
    print_info "Creating build directory..."
    mkdir -p build
fi

print_info "Configuring CMake..."
cd build
if cmake -DCMAKE_BUILD_TYPE=Debug .. &> /dev/null; then
    print_success "CMake configuration successful"

    print_info "Building project..."
    if make -j$(nproc) &> /dev/null; then
        print_success "Build successful"

        if [ -f "src/tests/nimcp_tests" ]; then
            print_info "Running quick test..."
            if timeout 30 ./src/tests/nimcp_tests --gtest_filter="-*Disabled*" --gtest_brief=1 &> /dev/null; then
                print_success "Tests passed"
            else
                print_warning "Some tests failed (see build output)"
            fi
        fi
    else
        print_error "Build failed"
        print_info "Run 'cd build && make' for details"
    fi
else
    print_error "CMake configuration failed"
    print_info "Run 'cd build && cmake ..' for details"
fi

cd "$PROJECT_ROOT"

#==============================================================================
# Summary
#==============================================================================
print_header "Setup Summary"

echo "Development environment is ready!"
echo ""
print_info "Next steps:"
echo "  1. Build project:       cd build && make"
echo "  2. Run tests:           cd build && make test"
echo "  3. Run lint:            ./scripts/lint.sh"
echo "  4. Generate coverage:   ./scripts/coverage.sh"
echo ""
print_info "Useful commands:"
echo "  - Format code:          ./scripts/lint.sh --fix"
echo "  - Run specific test:    ./build/src/tests/nimcp_tests --gtest_filter=TestName.*"
echo "  - Memory check:         valgrind ./build/src/tests/nimcp_tests"
echo ""
print_info "Documentation:"
echo "  - Scripts README:       scripts/README.md"
echo "  - CI workflow:          .github/workflows/ci.yml"
echo ""

if [ ${#MISSING_RECOMMENDED[@]} -gt 0 ]; then
    print_warning "Consider installing recommended tools for full functionality"
    echo "  Run: $0 --install-tools"
    echo ""
fi

print_success "Setup complete!"
