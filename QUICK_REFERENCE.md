# NIMCP Code Quality - Quick Reference

## Quick Start

```bash
# Setup development environment
./scripts/setup-dev-env.sh --install-tools

# Configure pre-commit hook
git config core.hooksPath .git-hooks
```

## Daily Workflow

### Before Coding
```bash
# Update from main
git pull origin master

# Create feature branch
git checkout -b feature/my-feature
```

### While Coding
```bash
# Auto-format code as you go
./scripts/lint.sh --fix

# Build and test frequently
cd build && make && make test
```

### Before Committing
```bash
# Run full lint check
./scripts/lint.sh

# Run tests
cd build && make test

# Check coverage (optional)
cd .. && ./scripts/coverage.sh
```

### Commit
```bash
# Stage changes
git add <files>

# Commit (pre-commit hook runs automatically)
git commit -m "feat: Add new feature"

# If pre-commit fails, fix and retry
./scripts/lint.sh --fix
git add <fixed-files>
git commit -m "feat: Add new feature"
```

## Common Commands

### Linting
```bash
./scripts/lint.sh              # Run all checks
./scripts/lint.sh --fix        # Auto-fix issues
./scripts/lint.sh --verbose    # Detailed output
./scripts/lint.sh src/lib/     # Specific path
```

### Testing
```bash
cd build

# All tests
make test

# Specific test suite
./src/tests/nimcp_tests --gtest_filter=NeuralNet*

# With valgrind
valgrind --leak-check=full ./src/tests/nimcp_tests

# Lint tests
./src/tests/nimcp_tests --gtest_filter=LintTest.*

# Memory leak tests
./src/tests/nimcp_tests --gtest_filter=MemoryLeakTest.*
```

### Coverage
```bash
./scripts/coverage.sh          # Generate report
./scripts/coverage.sh --clean  # Clean and regenerate
xdg-open coverage/html/index.html  # View report
```

### Fixing Common Issues

#### Format Violations
```bash
./scripts/lint.sh --fix
git add <fixed-files>
```

#### Build Errors
```bash
cd build
make clean
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

#### Test Failures
```bash
cd build
./src/tests/nimcp_tests --gtest_filter=FailingTest.* --gtest_output=xml
```

#### Coverage Below Threshold
```bash
# Add more tests for uncovered code
# Check coverage report for uncovered lines
xdg-open coverage/html/index.html
```

## Pre-Commit Hook

### Status
```bash
git config core.hooksPath  # Should show: .git-hooks
```

### Bypass (Emergency Only)
```bash
git commit --no-verify
```

### Fix and Retry
```bash
# Fix formatting
./scripts/lint.sh --fix

# Fix syntax errors
# (edit files)

# Re-stage and commit
git add <fixed-files>
git commit
```

## CI Status

### Check CI Status
- View on GitHub: Actions tab
- PR shows status checks

### If CI Fails

1. **Check logs** in GitHub Actions
2. **Reproduce locally**:
   ```bash
   ./scripts/lint.sh
   cd build && make test
   ./scripts/coverage.sh
   ```
3. **Fix issues**
4. **Push fix**:
   ```bash
   git add <fixes>
   git commit --amend  # or new commit
   git push --force-with-lease  # if amended
   ```

## File Locations

### Scripts
- `/home/bbrelin/src/repos/nimcp/scripts/lint.sh`
- `/home/bbrelin/src/repos/nimcp/scripts/coverage.sh`
- `/home/bbrelin/src/repos/nimcp/scripts/setup-dev-env.sh`

### Tests
- `/home/bbrelin/src/repos/nimcp/src/tests/test_lint.cpp`
- `/home/bbrelin/src/repos/nimcp/src/tests/test_memory_leaks.cpp`

### Config
- `/home/bbrelin/src/repos/nimcp/.github/workflows/ci.yml`
- `/home/bbrelin/src/repos/nimcp/.git-hooks/pre-commit`
- `/home/bbrelin/src/repos/nimcp/valgrind.supp`

### Documentation
- `/home/bbrelin/src/repos/nimcp/scripts/README.md`
- `/home/bbrelin/src/repos/nimcp/LINT_AND_CI_SETUP.md`

## Tool Installation

### Ubuntu/Debian
```bash
sudo apt-get install -y \
  clang-format clang-tidy cppcheck \
  shellcheck lcov valgrind
pip3 install lizard
```

### Quick Check
```bash
./scripts/setup-dev-env.sh  # Shows what's missing
```

## Standards

| Metric | Requirement |
|--------|-------------|
| Line Coverage | ≥70% |
| File Size | <2000 lines |
| Formatting | clang-format strict |
| Memory Leaks | Zero definite leaks |
| Build Warnings | Zero with -Werror |

## Exit Codes

### lint.sh
- `0` - All checks passed
- `1` - Linting issues found
- `2` - Tools not found
- `3` - Script error

### coverage.sh
- `0` - Success
- `1` - Failed to generate
- `2` - Tools not available

## Emergency Procedures

### Can't Commit (Pre-commit Blocking)
```bash
# Fix the issues (recommended)
./scripts/lint.sh --fix

# OR bypass (not recommended)
git commit --no-verify
```

### CI Failing, Urgent Merge Needed
1. Create hotfix branch
2. Fix CI issues
3. Get approval
4. Merge

Do NOT bypass CI checks in CI configuration.

### Coverage Suddenly Low
1. Check if tests ran: `build/src/tests/nimcp_tests`
2. Rebuild with coverage: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
3. Check .gcda files exist: `find build -name "*.gcda"`

## Getting Help

1. **Documentation**
   - `scripts/README.md` - Detailed script docs
   - `LINT_AND_CI_SETUP.md` - Full setup guide

2. **Logs**
   - CI: GitHub Actions logs
   - Local: Run with `--verbose`

3. **Common Issues**
   - See scripts/README.md Troubleshooting section

4. **Tools**
   ```bash
   clang-format --help
   cppcheck --help
   valgrind --help
   ```

## Tips

- Run `./scripts/lint.sh --fix` before every commit
- Keep coverage above 70%
- Run valgrind on new memory allocations
- Use `--gtest_filter` to run specific tests
- Check CI logs early in PR review
- Update tests when adding features

---

**Quick Links**:
- Full Docs: `LINT_AND_CI_SETUP.md`
- Scripts: `scripts/README.md`
- CI Config: `.github/workflows/ci.yml`
