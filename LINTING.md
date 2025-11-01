# NIMCP Code Quality and Linting Infrastructure

This document describes the comprehensive linting and code quality infrastructure for the NIMCP project.

## Overview

The NIMCP project uses a multi-tool approach to ensure code quality:

- **clang-format**: Automatic code formatting
- **clang-tidy**: Static analysis and modernization checks
- **cppcheck**: Additional static analysis for C/C++

All tools are configured with project-specific rules and can be run via convenient wrapper scripts.

## Quick Start

```bash
# Check code quality (no modifications)
./scripts/lint.sh

# Check and auto-fix issues
./scripts/lint.sh --fix

# Format all code
./scripts/format.sh

# Check formatting only
./scripts/format.sh --check
```

## Configuration Files

### `.clang-format`

Location: `/home/bbrelin/src/repos/nimcp/.clang-format`

**Style Base**: Google style with NIMCP-specific customizations

**Key Settings**:
- **IndentWidth**: 4 spaces
- **ColumnLimit**: 100 characters
- **PointerAlignment**: Left (`int* ptr`)
- **BreakBeforeBraces**: Custom (K&R for functions, no braces for control flow)
- **Standard**: C++17 for C++ files, auto-detected C99 for C files

**Rationale**:
- Google style is widely adopted and well-maintained
- 4-space indentation matches existing codebase
- 100-char line limit balances readability and screen width
- Left pointer alignment is standard in modern C/C++

### `.clang-tidy`

Location: `/home/bbrelin/src/repos/nimcp/.clang-tidy`

**Enabled Check Categories**:
- `bugprone-*`: Detect bug-prone patterns
- `performance-*`: Performance optimizations
- `readability-*`: Code readability
- `clang-analyzer-*`: Deep static analysis
- `misc-*`: Miscellaneous useful checks
- `cert-*`: CERT secure coding guidelines
- `portability-*`: Cross-platform compatibility
- `concurrency-*`: Thread-safety checks

**Disabled Checks** (and why):
- `bugprone-easily-swappable-parameters`: Too many false positives
- `readability-magic-numbers`: Too strict for system programming
- `misc-unused-parameters`: Common in callback interfaces
- `misc-no-recursion`: Sometimes needed for tree/graph algorithms

**Naming Conventions**:
- Functions: `lower_case`
- Variables: `lower_case`
- Types/Structs: `lower_case` with `_t` suffix
- Macros: `UPPER_CASE`
- Enum constants: `UPPER_CASE`

## Scripts

### `lint.sh` - Comprehensive Linting

**Purpose**: Run all code quality checks in one command

**Usage**:
```bash
./scripts/lint.sh [OPTIONS] [PATHS...]

Options:
  --fix          Apply automatic fixes where possible
  --verbose      Show detailed output from all tools
  --help         Show help message

Paths:
  Specific files or directories to check
  Default: src/ examples/
```

**Checks Performed**:
1. **clang-format**: Code formatting compliance
2. **clang-tidy**: Static analysis and modernization
3. **cppcheck**: Additional C/C++ static analysis

**Environment Variables**:
- `NIMCP_LINT_SKIP_FORMAT`: Skip clang-format check
- `NIMCP_LINT_SKIP_TIDY`: Skip clang-tidy check
- `NIMCP_LINT_SKIP_CPPCHECK`: Skip cppcheck

**Exit Codes**:
- `0`: All checks passed
- `1`: Issues found
- `2`: Required tools not found
- `3`: Script error

**Example Output**:
```
========================================
Checking Required Tools
========================================
✓ clang-format found: Ubuntu clang-format version 14.0.0
✓ clang-tidy found: LLVM version 14.0.0
✓ cppcheck found: Cppcheck 2.7

========================================
Running clang-format
========================================
ℹ Checking 156 files...
✓ All files are properly formatted

========================================
Running clang-tidy
========================================
ℹ Analyzing 156 files...
✓ No issues found by clang-tidy

========================================
Running cppcheck
========================================
ℹ Analyzing paths: /path/to/nimcp/src /path/to/nimcp/examples
✓ No issues found by cppcheck

========================================
Lint Summary
========================================
Checks run: 3
Checks failed: 0

✓ All checks passed! ✓
```

### `format.sh` - Code Formatting

**Purpose**: Apply or check clang-format rules

**Usage**:
```bash
./scripts/format.sh [OPTIONS] [PATHS...]

Options:
  --check        Check formatting without modifying files
  --verbose      Show detailed output for each file
  --help         Show help message

Paths:
  Specific files or directories to format
  Default: src/ examples/ bindings/
```

**Features**:
- Automatically excludes build directories
- Skips third-party code
- Supports individual files or entire directories
- Color-coded output for easy scanning

**Exit Codes**:
- `0`: Success (all formatted or already formatted)
- `1`: Files need formatting (--check mode)
- `2`: clang-format not found
- `3`: Script error

**Example - Format All Code**:
```bash
$ ./scripts/format.sh
========================================
Checking clang-format
========================================
✓ Found: Ubuntu clang-format version 14.0.0
✓ Using config: /path/to/nimcp/.clang-format

========================================
Formatting Code (156 files)
========================================

========================================
Summary
========================================
Total files processed: 156
Files formatted: 12
Files unchanged: 144

✓ Success!
```

**Example - Check Only**:
```bash
$ ./scripts/format.sh --check
========================================
Checking Formatting (156 files)
========================================

✗ 3 file(s) need formatting:
  - /path/to/nimcp/src/lib/nimcp_brain.c
  - /path/to/nimcp/src/lib/nimcp_adaptive.c
  - /path/to/nimcp/examples/brain_demo.c

ℹ Run without --check to apply formatting
```

## Tool Installation

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y clang-format clang-tidy cppcheck
```

### macOS (Homebrew)

```bash
brew install clang-format llvm cppcheck
```

### Fedora/RHEL

```bash
sudo dnf install -y clang-tools-extra cppcheck
```

### Verify Installation

```bash
$ clang-format --version
Ubuntu clang-format version 14.0.0-1ubuntu1

$ clang-tidy --version
LLVM (http://llvm.org/):
  LLVM version 14.0.0

$ cppcheck --version
Cppcheck 2.7
```

## Tool Status (Current System)

**As of this setup, the following tools are NOT installed**:
- clang-format: NOT FOUND
- clang-tidy: NOT FOUND
- cppcheck: NOT FOUND

**Installation Required**:
```bash
sudo apt-get install -y clang-format clang-tidy cppcheck
```

The scripts are designed to work gracefully when tools are missing:
- They detect available tools and skip unavailable ones
- They provide clear installation instructions
- They exit with appropriate error codes for CI integration

## CI/CD Integration

### GitHub Actions

The linting infrastructure integrates seamlessly with CI/CD pipelines:

```yaml
# .github/workflows/lint.yml
name: Code Quality

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install linting tools
        run: |
          sudo apt-get update
          sudo apt-get install -y clang-format clang-tidy cppcheck
      
      - name: Run linting
        run: ./scripts/lint.sh
      
      - name: Check formatting
        run: ./scripts/format.sh --check
```

### Pre-Commit Hooks

To enforce code quality before commits:

```bash
# .git/hooks/pre-commit
#!/bin/bash
./scripts/format.sh --check
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Code formatting issues detected!"
    echo "Run './scripts/format.sh' to fix"
    exit 1
fi
```

## Best Practices

### Daily Development

1. **Before Starting Work**:
   ```bash
   git pull
   ./scripts/format.sh  # Sync with team's formatting
   ```

2. **During Development**:
   - Use an editor with clang-format integration
   - Format on save (most IDEs support this)
   - Run `./scripts/lint.sh src/lib/myfile.c` on changed files

3. **Before Committing**:
   ```bash
   ./scripts/format.sh          # Auto-format
   ./scripts/lint.sh --fix      # Fix auto-fixable issues
   ./scripts/lint.sh            # Verify all issues resolved
   ```

### Code Review

1. **Automated Checks**: CI should run all linting automatically
2. **Manual Review**: Focus on logic, not formatting (automated)
3. **Zero Tolerance**: No lint issues should be merged

### Performance Considerations

For large codebases:

```bash
# Check only changed files
git diff --name-only main | xargs ./scripts/lint.sh

# Parallel checking (if tools support it)
./scripts/lint.sh --jobs=4  # Future enhancement
```

## Common Issues and Solutions

### Issue: "clang-format changes keep appearing"

**Cause**: Different clang-format versions produce different output

**Solution**:
- Pin clang-format version in CI and docs
- Recommended: clang-format 14 or later
- Check version: `clang-format --version`

### Issue: "Too many clang-tidy warnings"

**Solutions**:
1. **Disable specific checks**: Edit `.clang-tidy`
2. **Suppress in code**:
   ```c
   // NOLINT(check-name)
   int* ptr = malloc(size);
   ```
3. **Skip directories**:
   ```bash
   NIMCP_LINT_SKIP_TIDY=1 ./scripts/lint.sh
   ```

### Issue: "cppcheck false positives"

**Solutions**:
1. **Inline suppression**:
   ```c
   // cppcheck-suppress memleak
   return ptr;
   ```
2. **Global suppression**: Add to cppcheck args in `lint.sh`
3. **Skip cppcheck**:
   ```bash
   NIMCP_LINT_SKIP_CPPCHECK=1 ./scripts/lint.sh
   ```

### Issue: "Scripts fail in CI but work locally"

**Debugging**:
1. Check tool versions: `clang-format --version`
2. Run with `--verbose`: `./scripts/lint.sh --verbose`
3. Check CI logs for missing dependencies
4. Ensure `.clang-format` and `.clang-tidy` are committed

## Advanced Usage

### Custom Checks

Add custom checks to `lint.sh`:

```bash
# Example: Check for large files
check_file_size() {
    local max_lines=2000
    large_files=$(find src -name "*.c" -o -name "*.cpp" | \
                  xargs wc -l | \
                  awk -v max=$max_lines '$1 > max {print $2}')
    
    if [ -n "$large_files" ]; then
        print_error "Files exceed $max_lines lines:"
        echo "$large_files"
        return 1
    fi
}
```

### Integration with IDEs

#### Visual Studio Code

Install extensions:
- **C/C++** (Microsoft)
- **clang-format** (xaver)
- **clangd** (LLVM)

Settings (`.vscode/settings.json`):
```json
{
  "C_Cpp.clang_format_style": "file",
  "editor.formatOnSave": true,
  "clangd.arguments": ["--clang-tidy"]
}
```

#### Vim/Neovim

```vim
" Auto-format on save
autocmd BufWritePre *.c,*.cpp,*.h,*.hpp :silent! !clang-format -i %

" Manual format
nnoremap <leader>f :!clang-format -i %<CR>
```

#### CLion/IntelliJ

Settings → Editor → Code Style → C/C++
- Enable "Enable ClangFormat"
- Scheme: "Project"

## Metrics and Monitoring

### Tracking Code Quality Over Time

```bash
# Generate lint report
./scripts/lint.sh 2>&1 | tee lint-report.txt

# Count warnings
grep -c "warning:" lint-report.txt

# Track over time
echo "$(date),$(grep -c "warning:" lint-report.txt)" >> quality-metrics.csv
```

### Quality Gates

Set thresholds for CI:

```bash
# Fail if more than 10 warnings
warnings=$(./scripts/lint.sh 2>&1 | grep -c "warning:" || echo 0)
if [ $warnings -gt 10 ]; then
    echo "Too many warnings: $warnings"
    exit 1
fi
```

## References

- [clang-format Documentation](https://clang.llvm.org/docs/ClangFormat.html)
- [clang-tidy Documentation](https://clang.llvm.org/extra/clang-tidy/)
- [cppcheck Manual](http://cppcheck.sourceforge.net/)
- [CERT C Coding Standard](https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard)

## Changelog

### 2025-11-01
- Initial setup of linting infrastructure
- Created `.clang-format` with Google base style
- Created `.clang-tidy` with comprehensive checks
- Implemented `lint.sh` wrapper script
- Implemented `format.sh` formatting script
- Updated `scripts/README.md` documentation

## Contributing

When modifying linting configurations:

1. **Test Thoroughly**: Run on entire codebase
2. **Document Changes**: Update this file and inline comments
3. **Communicate**: Announce breaking changes to team
4. **Version Control**: Keep old configs for reference

For questions or issues, see the project's main README or open an issue.
