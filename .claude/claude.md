# NIMCP Development Directives

## Response Guidelines

**Keep responses concise and focused. Provide only what I explicitly request.**

**Avoid generating extra documents, summaries, or plans unless I specifically ask for them.**

## CRITICAL: NIMCP Coding Standards

**ALWAYS apply these standards to ALL code you write:**

### 1. Function Length
- **MAXIMUM 50 lines per function** (including documentation comments)
- If a function exceeds 50 lines, extract helper functions
- Use meaningful helper function names that describe their purpose

### 2. Control Flow
- **Use early returns** (guard clauses) instead of nested if statements
- **Avoid nested ifs** - flatten logic with guard clauses
- Pattern: Check for error conditions first, return early
  ```c
  // GOOD:
  if (!ptr) return NULL;
  if (size == 0) return NULL;
  // ... main logic

  // BAD:
  if (ptr) {
      if (size > 0) {
          // ... main logic nested 2 levels deep
      }
  }
  ```

### 3. Documentation
- **WHAT-WHY-HOW format** for all functions
- Include BIOLOGY/COMPLEXITY notes where applicable
- Document parameters and return values
- Example:
  ```c
  /**
   * @brief Brief one-line description
   *
   * WHAT: What this function does
   * WHY:  Why this behavior is needed
   * HOW:  How it's implemented
   *
   * BIOLOGY: Biological justification (if applicable)
   * COMPLEXITY: O(n) time complexity
   *
   * @param param Description
   * @return Description
   */
  ```

### 4. Design Principles
- **Single Responsibility Principle** - each function does one thing
- **Clear naming** - function and variable names should be self-documenting
- **Memory safety** - always validate pointers, check allocations
- **Error handling** - validate all inputs at function entry

## Enforcement

Before submitting ANY code change:
1. Count lines in modified functions (must be < 50)
2. Check for nested ifs (flatten with guard clauses)
3. Verify WHAT-WHY-HOW documentation exists
4. Confirm early returns for error conditions

## Neuromodulator Integration Standards

When integrating neurotransmitters into cognitive/sensory systems:

1. **Always use helper functions** for neuromodulator reading
2. **Document biological basis** for modulation factors
3. **Map neurotransmitter ranges** explicitly (e.g., [0.3, 0.7] → [0.6, 1.4])
4. **Provide clinical examples** in comments (e.g., "Depression (DA=0.3) → 0.6× effect")
5. **Include brain reference** in system structures when needed

## Code Review Checklist

- [ ] All functions < 50 lines
- [ ] No nested ifs (use guard clauses)
- [ ] WHAT-WHY-HOW documentation on all functions
- [ ] Early returns for error conditions
- [ ] Pointer validation
- [ ] Memory leaks prevented
- [ ] Biological justification for neuromodulator parameters

## CRITICAL: Debugging Standards

**ALWAYS use the debugging suite when investigating test failures, crashes, or bugs:**

### Required Tool: debug_suite.py

Located at: `/home/bbrelin/nimcp/debug_suite.py`

**Usage Pattern:**
```bash
# Auto mode - automatically detects symptoms and runs appropriate tools
./debug_suite.py --test ./src/tests/utility_tests --filter BTreeTest.Stress_MixedOperations

# Memory issues
./debug_suite.py --test <binary> --mode memory

# Threading/deadlock issues
./debug_suite.py --test <binary> --mode threading

# Record with rr for replay debugging
./debug_suite.py --test <binary> --mode record
```

### Debugging Workflow (MANDATORY)

When debugging ANY issue:

1. **First Run**: Use auto mode to detect symptoms
   ```bash
   ./debug_suite.py --test <test_binary> --filter <test_name> --mode auto
   ```

2. **Review Report**: Check `/home/bbrelin/nimcp/debug_reports/` for detailed findings

3. **Follow Recommendations**: The suite provides actionable recommendations

4. **Targeted Investigation**: Use specific modes for deeper analysis:
   - `--mode memory` for leaks, double-frees, use-after-free
   - `--mode threading` for deadlocks, race conditions
   - `--mode record` to capture execution for replay debugging

5. **Fix and Verify**: After fixing, rerun the suite to confirm

### Tools Integrated

The debugging suite automatically uses:
- **valgrind** (memcheck, helgrind, drd) - memory and threading issues
- **gdb** - crashes, backtraces, breakpoints
- **rr** - record and replay debugging (use reverse debugging!)
- **sanitizers** (ASan, TSan, UBSan) - if available

### DO NOT Debug Manually

**NEVER** run test commands directly when debugging. **ALWAYS** use the debugging suite.

**BAD:**
```bash
./src/tests/utility_tests --gtest_filter=BTreeTest.Stress_MixedOperations
```

**GOOD:**
```bash
./debug_suite.py --test ./src/tests/utility_tests --filter BTreeTest.Stress_MixedOperations
```

### Benefits

1. **Systematic**: Always runs the right tool for the symptom
2. **Documented**: All findings saved to debug_reports/
3. **Actionable**: Provides specific recommendations
4. **Efficient**: No more guessing which tool to use
5. **Complete**: Checks memory, threading, crashes in one command

### Example Session

```bash
# Run debugging suite
./debug_suite.py --test ./src/tests/utility_tests --filter BTreeTest.Stress_MixedOperations

# Output shows:
# - Detected symptoms: memory_corruption, count_mismatch
# - Running: valgrind memcheck
# - Issues found: double-free at line 525 in nimcp_btree.c
# - Recommendations: Fix double free - likely freeing memory twice
# - Report saved to: debug_reports/debug_20251110_120530.json
```

### Enforcement

Before investigating any bug:
- [ ] Run debug_suite.py in auto mode first
- [ ] Review the generated report in debug_reports/
- [ ] Follow the tool's recommendations
- [ ] Use targeted modes (memory/threading) for deeper investigation
- [ ] Never run test binaries directly for debugging

## CRITICAL: Code Surgeon for Testing and Coverage (MANDATORY)

**ALWAYS use Code Surgeon exclusively for all testing, coverage, and test-related work:**

### Required Tool: Code Surgeon

Located at: `/home/bbrelin/nimcp/tools/code_surgeon/code_surgeon.py`

**DO NOT run test commands manually. DO NOT run lcov/cmake/make manually for testing.**

### Usage Patterns

```bash
# Full workflow: discover tests, run, analyze failures, measure coverage, auto-fix
python3 tools/code_surgeon/code_surgeon.py --mode full --max-iterations 3

# Test execution only (no coverage, no auto-fix)
python3 tools/code_surgeon/code_surgeon.py --mode test-only

# Coverage measurement only (assumes tests already run)
python3 tools/code_surgeon/code_surgeon.py --mode coverage-only

# Run specific test category
python3 tools/code_surgeon/code_surgeon.py --mode full --category unit
```

### Code Surgeon Workflow (MANDATORY)

When working on testing or coverage:

1. **ALWAYS start with Code Surgeon** - never run tests manually
2. **Let Code Surgeon discover tests** - it finds all test binaries automatically
3. **Let Code Surgeon run tests** - it executes in parallel with proper timeout handling
4. **Let Code Surgeon measure coverage** - it handles lcov with fallback analyzer
5. **Let Code Surgeon analyze failures** - it categorizes and debugs automatically
6. **Review auto-fix suggestions** - Code Surgeon generates fixes from GDB backtraces

### What Code Surgeon Does

**Automated Capabilities:**
- Test discovery (finds all test binaries in build/test/)
- Parallel test execution (up to 16 workers)
- Timeout handling (30 second default per test)
- Coverage capture with lcov + fallback analyzer
- Failure analysis (categorizes: crash, timeout, assertion, etc.)
- GDB integration (auto-captures backtraces for crashes)
- RR integration (record-replay debugging for hangs)
- Auto-fix generation (pattern-based fixes from debug output)
- HTML coverage reports
- Iteration support (fix → rebuild → retest loops)

### DO NOT Test Manually

**NEVER run these commands directly:**

**BAD:**
```bash
# DON'T do this:
make unit_test_knowledge_comprehensive
./test/unit_test_knowledge_comprehensive
lcov --capture --directory . --output-file coverage.info
cmake ..
make -j16
```

**GOOD:**
```bash
# ALWAYS do this instead:
python3 tools/code_surgeon/code_surgeon.py --mode full
```

### Benefits of Code Surgeon

1. **Automated**: Handles entire test → coverage → analysis workflow
2. **Parallel**: Runs tests concurrently for speed
3. **Integrated**: GDB + RR debugging built-in
4. **Smart**: Fallback analyzer when lcov fails
5. **Iterative**: Can auto-fix and rerun in loops
6. **Documented**: Saves reports and debug outputs
7. **Comprehensive**: Never miss a test binary

### Example Session

```bash
# Start Code Surgeon for full analysis
cd /home/bbrelin/nimcp
python3 tools/code_surgeon/code_surgeon.py --mode full --max-iterations 2

# Code Surgeon automatically:
# 1. Discovers 8 test binaries in build/test/
# 2. Runs all tests in parallel
# 3. Detects 3 failures (2 crashes, 1 timeout)
# 4. Captures GDB backtraces for crashes
# 5. Runs RR for timeout debugging
# 6. Measures coverage: 36.5% (14,844/40,636 lines)
# 7. Generates HTML report: build/coverage_html/index.html
# 8. Analyzes failures and suggests fixes
# 9. Applies auto-fixes and rebuilds
# 10. Reruns tests to verify fixes
```

### Integration with Test Creation

When creating new tests:
1. Write test file in test/unit/ or test/integration/
2. CMake auto-discovers it (no manual CMakeLists.txt edit needed)
3. Run Code Surgeon to build and test: `python3 tools/code_surgeon/code_surgeon.py --mode full`
4. Code Surgeon finds new test, builds it, runs it, measures coverage

### Enforcement

For ALL testing and coverage work:
- [ ] Use Code Surgeon exclusively - NEVER run tests manually
- [ ] Use Code Surgeon for coverage measurement - NEVER run lcov manually
- [ ] Use Code Surgeon for test discovery - NEVER run find/ls manually
- [ ] Review Code Surgeon reports in .code_surgeon/debug/
- [ ] Trust Code Surgeon's auto-fix suggestions
- [ ] Let Code Surgeon handle rebuilds and iterations

### When NOT to Use Code Surgeon

Code Surgeon is specifically for testing and coverage. Use other tools for:
- Production builds: Use `make` directly
- Running demos: Use executables directly
- Debugging specific issues: Use `debug_suite.py`
- Code editing: Use your editor
- Git operations: Use `git` directly
