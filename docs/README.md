# NIMCP Scripts - Code Quality and Testing Tools

This directory contains scripts for code quality checks, testing, and continuous integration.

## Contents

- `lint.sh` - Comprehensive linting and static analysis
- `format.sh` - Automatic code formatting with clang-format
- `coverage.sh` - Code coverage analysis and reporting

## Quick Start

### Running Lint Checks

```bash
# Run all lint checks
./scripts/lint.sh

# Auto-fix formatting issues
./scripts/lint.sh --fix

# Verbose output
./scripts/lint.sh --verbose

# Check specific files
./scripts/lint.sh src/lib/nimcp_brain.c
```

### Running Code Formatter

```bash
# Format all code
./scripts/format.sh

# Check formatting without modifying files
./scripts/format.sh --check

# Format with verbose output
./scripts/format.sh --verbose

# Format specific files
./scripts/format.sh src/lib/nimcp_brain.c
```

### Generating Code Coverage

```bash
# Generate coverage report
./scripts/coverage.sh

# Generate without HTML
./scripts/coverage.sh --no-html

# Clean and regenerate
./scripts/coverage.sh --clean
```

## Lint Script (`lint.sh`)

### Features

- **Code Formatting**: clang-format compliance checking
- **Static Analysis**: clang-tidy and cppcheck
- **Shell Linting**: shellcheck for shell scripts
- **File Size Checks**: Ensures files don't exceed 2000 lines
- **TODO/FIXME Detection**: Finds comments needing attention
- **Complexity Analysis**: lizard for code complexity (optional)

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install -y clang-format clang-tidy cppcheck shellcheck
pip3 install lizard
```

### Usage

```bash
./scripts/lint.sh [OPTIONS] [PATHS...]

Options:
  --fix          Apply automatic fixes where possible
  --verbose      Show detailed output from all tools
  --help, -h     Show help message

Environment Variables:
  NIMCP_LINT_SKIP_FORMAT    - Skip clang-format check
  NIMCP_LINT_SKIP_TIDY      - Skip clang-tidy check
  NIMCP_LINT_SKIP_CPPCHECK  - Skip cppcheck
```

### Exit Codes

- `0` - All checks passed
- `1` - Linting issues found
- `2` - Required tools not found
- `3` - Script error

## Format Script (`format.sh`)

### Features

- **Automatic Formatting**: Apply clang-format rules to C/C++ code
- **Check Mode**: Verify formatting without modifying files
- **Selective Formatting**: Format specific files or directories
- **Smart Filtering**: Automatically excludes build directories and generated files
- **Verbose Output**: Detailed per-file formatting status

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install -y clang-format

# macOS
brew install clang-format

# Fedora/RHEL
sudo dnf install -y clang-tools-extra
```

### Usage

```bash
./scripts/format.sh [OPTIONS] [PATHS...]

Options:
  --check        Check formatting without modifying files (exit 1 if changes needed)
  --verbose      Show detailed output for each file processed
  --help, -h     Show help message

Environment Variables:
  NIMCP_FORMAT_STYLE  - Override clang-format style (default: file)
```

### Exit Codes

- `0` - All files are formatted (or successfully formatted)
- `1` - Files need formatting (in --check mode)
- `2` - clang-format not found
- `3` - Script error

## Coverage Script (`coverage.sh`)

### Features

- **gcov/lcov Integration**: Standard coverage tools
- **HTML Reports**: Interactive coverage browsing
- **Coverage Metrics**: Line and function coverage
- **Threshold Checking**: Warns if coverage < 70%

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install -y lcov gcov

# Project must be built with coverage flags
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### Usage

```bash
./scripts/coverage.sh [OPTIONS]

Options:
  --html     Generate HTML report (default: yes)
  --no-html  Skip HTML generation
  --clean    Clean coverage data before running
  --help     Show help message
```

### Output

- `coverage/coverage.info` - LCOV data file
- `coverage/html/index.html` - HTML report (interactive)
- Terminal summary with key metrics

### Exit Codes

- `0` - Coverage generated successfully
- `1` - Failed to generate coverage
- `2` - Required tools not available

## CI Integration

All scripts are integrated into GitHub Actions CI workflow (`.github/workflows/ci.yml`):

### CI Jobs

1. **Build & Test** - Builds project and runs tests
2. **Code Quality & Linting** - Runs lint.sh and all quality checks
3. **Code Coverage** - Generates coverage reports
4. **Memory Leak Testing** - Runs valgrind on critical tests

### CI Artifacts

- Test results (XML)
- Coverage reports (HTML)
- Valgrind reports (text)

## Pre-Commit Hook

Install the pre-commit hook to run checks before each commit:

```bash
# Option 1: Configure git hooks path
git config core.hooksPath .git-hooks

# Option 2: Copy manually
cp .git-hooks/pre-commit .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### Pre-Commit Checks

- Code formatting (clang-format)
- Basic syntax checks
- File size limits
- TODO/FIXME detection

### Bypassing Pre-Commit Hook

```bash
# Not recommended, but useful in emergencies
git commit --no-verify
```

## Test Integration

Lint and memory leak checks are also available as GoogleTest tests:

```bash
cd build

# Run lint tests
./src/tests/nimcp_tests --gtest_filter=LintTest.*

# Run memory leak tests
./src/tests/nimcp_tests --gtest_filter=MemoryLeakTest.*

# Run all quality tests
./src/tests/nimcp_tests --gtest_filter=LintTest.*:MemoryLeakTest.*
```

## Valgrind Memory Testing

### Manual Valgrind Testing

```bash
cd build

# Full leak check
valgrind --leak-check=full \
         --show-leak-kinds=all \
         --track-origins=yes \
         ./src/tests/nimcp_tests

# Specific test suite
valgrind --leak-check=full \
         ./src/tests/nimcp_tests \
         --gtest_filter=NeuralNetCreate.*

# Save to file
valgrind --leak-check=full \
         --log-file=valgrind.log \
         ./src/tests/nimcp_tests
```

### Valgrind Suppressions

Known false positives are suppressed via `valgrind.supp`:

- Python C API allocations
- GTest initialization
- System library allocations

## Tool Installation Guide

### Ubuntu/Debian

```bash
# All linting tools
sudo apt-get update
sudo apt-get install -y \
    clang-format \
    clang-tidy \
    cppcheck \
    shellcheck \
    lcov \
    gcov \
    valgrind

# Python tools
pip3 install lizard
```

### macOS (Homebrew)

```bash
brew install clang-format llvm cppcheck shellcheck lcov
pip3 install lizard
```

### Fedora/RHEL

```bash
sudo dnf install -y \
    clang-tools-extra \
    cppcheck \
    ShellCheck \
    lcov \
    valgrind

pip3 install lizard
```

## Best Practices

### Before Committing

1. Run `./scripts/lint.sh --fix` to auto-fix formatting
2. Fix any remaining lint issues
3. Run tests: `cd build && make test`
4. Check coverage: `./scripts/coverage.sh`
5. Commit with descriptive message

### Before Pull Request

1. Ensure all CI checks pass locally
2. Run memory leak tests on critical changes
3. Update tests for new functionality
4. Verify coverage doesn't decrease

### Code Quality Standards

- **Line Coverage**: Target 70% minimum
- **File Size**: Keep files under 2000 lines
- **Formatting**: Strictly follow clang-format rules
- **Static Analysis**: Address all clang-tidy warnings
- **Memory Leaks**: Zero definite leaks in valgrind

## Troubleshooting

### Lint Script Issues

**Problem**: clang-format not found
```bash
sudo apt-get install clang-format
```

**Problem**: False positives from cppcheck
```bash
# Skip cppcheck
NIMCP_LINT_SKIP_CPPCHECK=1 ./scripts/lint.sh
```

### Coverage Issues

**Problem**: No .gcda files found
```bash
# Rebuild with coverage flags
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make clean && make
./src/tests/nimcp_tests
```

**Problem**: Low coverage numbers
- Add more unit tests
- Run all test suites before coverage
- Check that tests actually execute code paths

### Valgrind Issues

**Problem**: Python leaks reported
- These are often false positives
- Use suppressions file: `--suppressions=valgrind.supp`

**Problem**: Valgrind too slow
- Run on subset of tests: `--gtest_filter=Critical.*`
- Use lighter checks: `--leak-check=summary`

## Contributing

When adding new scripts:

1. Make executable: `chmod +x scripts/newscript.sh`
2. Add shebang: `#!/bin/bash`
3. Add comprehensive help text
4. Handle missing tools gracefully
5. Use exit codes correctly
6. Update this README

## References

- [clang-format documentation](https://clang.llvm.org/docs/ClangFormat.html)
- [clang-tidy documentation](https://clang.llvm.org/extra/clang-tidy/)
- [cppcheck manual](http://cppcheck.sourceforge.net/manual.pdf)
- [lcov documentation](http://ltp.sourceforge.net/coverage/lcov.php)
- [valgrind manual](https://valgrind.org/docs/manual/manual.html)
