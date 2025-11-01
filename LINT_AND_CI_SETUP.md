# NIMCP Lint Tests and CI Configuration

## Overview

This document describes the automated code quality checks, lint tests, and CI configuration implemented for the NIMCP project.

## Files Created

### 1. Scripts (`/home/bbrelin/src/repos/nimcp/scripts/`)

#### `lint.sh`
Comprehensive linting script that performs:
- **Code formatting checks** (clang-format)
- **Static analysis** (clang-tidy, cppcheck)
- **Shell script linting** (shellcheck)
- **File size validation** (max 2000 lines)
- **TODO/FIXME detection**
- **Code complexity analysis** (lizard, optional)

**Usage:**
```bash
./scripts/lint.sh              # Run all checks
./scripts/lint.sh --fix        # Auto-fix formatting issues
./scripts/lint.sh --verbose    # Detailed output
./scripts/lint.sh src/lib/     # Check specific path
```

**Exit codes:**
- 0: All checks passed
- 1: Linting issues found
- 2: Required tools not found
- 3: Script error

#### `coverage.sh`
Code coverage analysis using gcov/lcov:
- Generates coverage reports
- Creates HTML reports with line-by-line coverage
- Checks coverage thresholds (70% minimum)
- Outputs to `coverage/` directory

**Usage:**
```bash
./scripts/coverage.sh          # Generate full report
./scripts/coverage.sh --clean  # Clean and regenerate
./scripts/coverage.sh --no-html # Skip HTML generation
```

**Prerequisites:**
```bash
sudo apt-get install lcov gcov
# Build with coverage flags
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

#### `setup-dev-env.sh`
Development environment setup script:
- Checks for required build tools
- Verifies linting tools installation
- Configures git pre-commit hooks
- Tests project build
- Provides installation guidance

**Usage:**
```bash
./scripts/setup-dev-env.sh                # Check environment
./scripts/setup-dev-env.sh --install-tools # Install missing tools
```

#### `README.md`
Comprehensive documentation for all scripts including:
- Tool installation guides
- Usage examples
- Best practices
- Troubleshooting tips

### 2. Test Files (`/home/bbrelin/src/repos/nimcp/src/tests/`)

#### `test_lint.cpp`
GoogleTest-based lint testing that:
- Verifies lint.sh script exists and is executable
- Tests code formatting compliance
- Checks file size limits
- Validates availability of linting tools (clang-format, cppcheck)
- Detects TODO/FIXME comments
- Runs complexity analysis
- Checks for memory-unsafe patterns

**Key test cases:**
- `LintTest.LintScriptExists` - Verifies lint script presence
- `LintTest.LintScriptExecutes` - Tests script execution
- `LintTest.ClangFormatAvailable` - Checks formatting tool
- `LintTest.FileSizeLimits` - Validates file sizes
- `LintTest.CppCheckSample` - Runs static analysis
- `LintTest.MemorySafetyPatterns` - Detects unsafe functions

**Features:**
- Gracefully skips tests if tools unavailable
- Informational only (doesn't fail build on warnings)
- Provides detailed output for debugging

#### `test_memory_leaks.cpp`
Memory leak detection tests using valgrind:
- Basic allocation/deallocation tests
- Neural network memory leak checks
- Queue allocation tests
- Valgrind integration tests (disabled by default)
- Manual testing instructions

**Key test cases:**
- `MemoryLeakTest.ValgrindAvailable` - Checks for valgrind
- `MemoryLeakTest.NeuralNetworkCreationDestruction` - Tests neural net cleanup
- `MemoryLeakTest.QueueAllocationDeallocation` - Tests queue cleanup
- `DISABLED_ValgrindNeuralNetworkTests` - Full valgrind integration (manual)
- `ManualValgrindInstructions` - Provides usage documentation

**Running tests:**
```bash
# Run basic memory tests
./build/src/tests/nimcp_tests --gtest_filter=MemoryLeakTest.*

# Enable valgrind integration tests
./build/src/tests/nimcp_tests --gtest_filter=MemoryLeakTest.DISABLED_*

# Manual valgrind
valgrind --leak-check=full ./build/src/tests/nimcp_tests
```

### 3. CI Configuration (`/home/bbrelin/src/repos/nimcp/.github/workflows/ci.yml`)

Updated CI workflow with new jobs:

#### Job 2: Code Quality & Linting
- Installs linting tools (clang-format, clang-tidy, cppcheck, shellcheck)
- Runs `scripts/lint.sh` with verbose output
- Performs individual tool checks
- Validates shell scripts
- Fails on critical issues

#### Job 2.5: Code Coverage
- Builds project with coverage flags
- Runs all tests
- Generates coverage reports (lcov/HTML)
- Uploads to Codecov
- Checks 70% coverage threshold
- Uploads HTML report as artifact

#### Job 2.6: Memory Leak Testing (Valgrind)
- Installs valgrind
- Runs memory leak test suite
- Executes critical tests under valgrind
- Generates detailed valgrind reports
- Uploads reports as artifacts

**CI Artifacts:**
- `test-results-*.xml` - Test results
- `coverage-report/` - HTML coverage report
- `valgrind-report.txt` - Memory leak analysis

### 4. Pre-Commit Hook (`/home/bbrelin/src/repos/nimcp/.git-hooks/pre-commit`)

Runs before each commit to ensure code quality:

**Checks performed:**
1. Code formatting (clang-format)
2. File size limits
3. Basic syntax checking (gcc -fsyntax-only)
4. TODO/FIXME detection (warning)
5. Quick lint on staged files

**Installation:**
```bash
# Option 1: Configure hooks path (recommended)
git config core.hooksPath .git-hooks

# Option 2: Copy manually
cp .git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

**Bypass (emergency only):**
```bash
git commit --no-verify
```

### 5. Supporting Files

#### `valgrind.supp`
Valgrind suppressions file for known false positives:
- Python C API allocations
- GTest initialization
- System library allocations

**Usage:**
```bash
valgrind --suppressions=valgrind.supp ./build/src/tests/nimcp_tests
```

### 6. Build Configuration Updates

#### `src/tests/CMakeLists.txt`
Added new test files:
- `test_lint.cpp` - Lint and code quality tests
- `test_memory_leaks.cpp` - Memory leak detection tests

## CI Workflow Summary

### Triggers
- Push to `main`, `master`, `develop` branches
- Pull requests to `main`, `master`
- Version tags (`v*`)

### Jobs
1. **Build & Test** - Multi-OS, multi-Python version testing
2. **Code Quality** - Linting and static analysis
3. **Code Coverage** - Coverage reporting with threshold checks
4. **Memory Leak Testing** - Valgrind-based leak detection
5. **Docker Build** - Container image creation
6. **Benchmarks** - Performance testing
7. **Security Scan** - Trivy security scanning
8. **Documentation** - API docs generation
9. **Release** - Automated releases on tags

### Status Badges (recommended)
Add to README.md:
```markdown
![CI](https://github.com/YOUR_ORG/nimcp/workflows/NIMCP%20CI%2FCD/badge.svg)
![Coverage](https://codecov.io/gh/YOUR_ORG/nimcp/branch/master/graph/badge.svg)
```

## Local Testing Instructions

### 1. Setup Development Environment
```bash
./scripts/setup-dev-env.sh --install-tools
```

### 2. Run Lint Checks
```bash
# Check all code
./scripts/lint.sh

# Auto-fix formatting
./scripts/lint.sh --fix

# Verbose output
./scripts/lint.sh --verbose
```

### 3. Generate Coverage Report
```bash
# Build with coverage
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make

# Run tests and generate report
cd ..
./scripts/coverage.sh
```

### 4. Memory Leak Testing
```bash
# Run basic tests
cd build
./src/tests/nimcp_tests --gtest_filter=MemoryLeakTest.*

# Full valgrind check
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./src/tests/nimcp_tests
```

### 5. Run All Quality Checks (Pre-CI)
```bash
# Lint
./scripts/lint.sh

# Build and test
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
make test

# Coverage
cd ..
./scripts/coverage.sh

# Memory check (sample)
cd build
valgrind --leak-check=full \
         --suppressions=../valgrind.supp \
         ./src/tests/nimcp_tests --gtest_filter=NeuralNetCreate.*
```

## Tool Requirements

### Required (for building)
- cmake (≥3.10)
- gcc/g++ or clang/clang++
- make
- python3
- python3-dev
- pkg-config
- googletest

### Recommended (for quality checks)
- clang-format
- clang-tidy
- cppcheck
- shellcheck
- lcov
- gcov
- valgrind
- lizard (Python package)

### Installation

#### Ubuntu/Debian
```bash
# Required
sudo apt-get install -y build-essential cmake python3-dev pkg-config

# Recommended
sudo apt-get install -y \
    clang-format \
    clang-tidy \
    cppcheck \
    shellcheck \
    lcov \
    valgrind

pip3 install lizard
```

#### macOS
```bash
brew install cmake llvm python3
brew install clang-format cppcheck shellcheck lcov
pip3 install lizard
```

## Best Practices

### Before Committing
1. Run `./scripts/lint.sh --fix`
2. Fix any remaining lint issues
3. Run tests locally
4. Check that pre-commit hook passes
5. Write clear commit messages

### Before Creating PR
1. Ensure all CI checks pass locally
2. Verify coverage doesn't decrease
3. Run memory leak tests on modified code
4. Update tests for new functionality
5. Update documentation

### Code Quality Standards
- **Line Coverage**: 70% minimum
- **File Size**: <2000 lines per file
- **Formatting**: Strict clang-format compliance
- **Static Analysis**: Address all critical issues
- **Memory**: Zero definite leaks in valgrind

## Troubleshooting

### Lint Script Fails
**Issue**: Missing tools
```bash
./scripts/setup-dev-env.sh --install-tools
```

**Issue**: Format violations
```bash
./scripts/lint.sh --fix
```

### Coverage Issues
**Issue**: No .gcda files
```bash
# Rebuild with coverage flags
cd build
rm -rf *
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
./src/tests/nimcp_tests
cd .. && ./scripts/coverage.sh
```

### Valgrind Reports Leaks
**Issue**: Python false positives
```bash
valgrind --suppressions=valgrind.supp ./build/src/tests/nimcp_tests
```

**Issue**: Too slow
```bash
# Run subset
valgrind ./build/src/tests/nimcp_tests --gtest_filter=Critical.*
```

### CI Fails Locally Passes
1. Check CI logs for specific error
2. Verify tool versions match CI
3. Run in clean build directory
4. Check for timing-dependent tests

## Future Enhancements

### Potential Additions
- [ ] Automatic code review comments (GitHub Actions)
- [ ] Benchmark regression detection
- [ ] Fuzz testing integration
- [ ] ASAN/UBSAN sanitizer tests
- [ ] Thread sanitizer (TSAN) tests
- [ ] Static analysis with scan-build
- [ ] Documentation coverage checks
- [ ] API compatibility checks

### Metrics Dashboard
Consider adding:
- Code coverage trends
- Test execution time trends
- Lint violation trends
- Memory leak detection history

## References

- **Scripts Documentation**: `/home/bbrelin/src/repos/nimcp/scripts/README.md`
- **CI Workflow**: `/home/bbrelin/src/repos/nimcp/.github/workflows/ci.yml`
- **Pre-commit Hook**: `/home/bbrelin/src/repos/nimcp/.git-hooks/pre-commit`
- **Test Files**: `/home/bbrelin/src/repos/nimcp/src/tests/test_lint.cpp`, `test_memory_leaks.cpp`

## Support

For issues or questions:
1. Check scripts/README.md for detailed documentation
2. Review CI workflow logs
3. Run with --verbose for detailed output
4. Check tool versions match CI environment

---

**Generated**: 2025-11-01
**NIMCP Version**: 2.5.0
