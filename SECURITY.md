# Security Policy

## Supported Versions

The following versions of NIMCP receive security updates:

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 2.5.x   | :white_check_mark: | Active development |
| 2.0-2.4 | :x:                | End of life |
| < 2.0   | :x:                | Not supported |

## Security Features

NIMCP implements multiple layers of security hardening:

### Build-Time Security

1. **Compiler Hardening** (enabled in Release builds)
   - `-D_FORTIFY_SOURCE=2` - Buffer overflow detection at compile and runtime
   - `-fstack-protector-strong` - Stack canary protection against buffer overflows
   - `-fPIE -pie` - Position Independent Executable for ASLR
   - `-Wformat-security` - Format string vulnerability detection
   - `-Wl,-z,relro -Wl,-z,now` - Full RELRO (Read-Only Relocations)
   - `-Wl,-z,noexecstack` - Non-executable stack

2. **Sanitizers** (optional, for development/testing)
   - AddressSanitizer (ASAN) - Memory error detection
   - UndefinedBehaviorSanitizer (UBSAN) - Undefined behavior detection
   - ThreadSanitizer (TSAN) - Race condition detection

3. **Static Analysis**
   - clang-tidy with security-focused checks
   - cppcheck for additional static analysis
   - Custom unsafe function auditing

### Runtime Security

1. **Memory Safety**
   - All string operations use bounds-checked functions (`snprintf`, `strncat`)
   - No usage of unsafe functions (`strcpy`, `strcat`, `sprintf`, `gets`)
   - Comprehensive memory leak testing with Valgrind
   - Position-independent code for library safety

2. **Input Validation**
   - Systematic validation at all public API boundaries
   - NULL pointer checks
   - Range validation for numerical inputs
   - Size limits on buffers and arrays

3. **Thread Safety**
   - Thread-safe data structures with proper locking
   - Tested with ThreadSanitizer for race conditions
   - Deadlock detection in stress tests

## Building with Security Features

### Default Build (Security Hardening Enabled)
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

This automatically enables:
- FORTIFY_SOURCE
- Stack protection
- PIE/PIC
- Full RELRO

### Development Build with Sanitizers

#### AddressSanitizer + UBSanitizer
```bash
mkdir build-asan && cd build-asan
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_UBSAN=ON ..
make
./src/tests/nimcp_tests
```

#### ThreadSanitizer
```bash
mkdir build-tsan && cd build-tsan
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_TSAN=ON ..
make
export TSAN_OPTIONS="halt_on_error=1 history_size=7"
./src/tests/nimcp_tests
```

## Reporting a Vulnerability

### How to Report

**Do NOT** open a public GitHub issue for security vulnerabilities.

Instead, please report security issues to:
- **Email:** [Create a private security advisory on GitHub]
- **GitHub:** Use the "Security" tab → "Report a vulnerability"

### What to Include

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** assessment
4. **Suggested fix** (if available)
5. **Your contact information**

### Response Timeline

- **Initial Response:** Within 48 hours
- **Status Update:** Within 7 days
- **Fix Timeline:** Severity-dependent
  - Critical: 14 days
  - High: 30 days
  - Medium: 60 days
  - Low: 90 days

## Security Testing

### Continuous Integration

Every commit is automatically tested with:

1. **Memory Leak Testing** (Valgrind)
   - Zero tolerance for definite leaks
   - Suppression file for known false positives

2. **ThreadSanitizer**
   - Race condition detection
   - Deadlock detection

3. **AddressSanitizer + UBSanitizer**
   - Buffer overflows
   - Use-after-free
   - Memory corruption
   - Undefined behavior

4. **Static Analysis**
   - clang-tidy security checks
   - cppcheck warnings
   - Unsafe function detection

5. **Code Coverage**
   - Minimum 70% line coverage
   - Focus on critical paths

### Local Security Audit

```bash
# Run comprehensive security audit
./scripts/audit-unsafe-functions.sh

# Run static analysis
./scripts/lint.sh

# Run tests with all sanitizers
cmake -DENABLE_ASAN=ON -DENABLE_UBSAN=ON ..
make && ./src/tests/nimcp_tests

# Memory leak check
valgrind --leak-check=full ./src/tests/nimcp_tests
```

## Known Security Limitations

1. **Python C API Integration**
   - NIMCP integrates with Python's C API
   - Some Python allocations may appear as leaks in Valgrind
   - These are suppressed in `valgrind.supp`

2. **Floating Point Operations**
   - Neural network operations use floating-point math
   - Normal rounding errors are not considered security issues

3. **Test-Only Code**
   - Some test utilities intentionally create edge cases
   - These are isolated to test code only

## Security Best Practices for Users

### For Library Users

1. **Always validate inputs** before passing to NIMCP APIs
2. **Check return values** for error conditions
3. **Use the latest stable release** for security patches
4. **Enable compiler warnings** in your own code
5. **Test with sanitizers** during development

### Example: Safe API Usage

```c
#include <nimcp_neuralnet.h>

// GOOD: Validate configuration before use
nimcp_neuralnet_config_t config;
if (config.num_inputs > 0 && config.num_inputs <= MAX_NEURONS) {
    nimcp_neuralnet_t *net = nimcp_neuralnet_create(&config);
    if (net) {
        // Success
        // ... use network ...
        nimcp_neuralnet_destroy(net);
    } else {
        // Handle error
    }
}

// BAD: No validation
nimcp_neuralnet_t *net = nimcp_neuralnet_create(&config);  // May fail
net->forward(inputs);  // Crash if net is NULL
```

## Security Credits

We appreciate responsible disclosure. Security researchers who report valid vulnerabilities will be credited in:
- Release notes
- CHANGELOG.md
- This SECURITY.md file (with permission)

## Security Changelog

### Version 2.5.0 (2025-11-01)
- ✅ Replaced all unsafe string functions with bounds-checked alternatives
- ✅ Added comprehensive sanitizer support (ASAN, UBSAN, TSAN)
- ✅ Implemented security hardening compiler flags
- ✅ Added ThreadSanitizer CI testing
- ✅ Created unsafe function auditing script
- ✅ Enhanced clang-tidy security checks

### Version 2.0.0
- Initial security framework
- Valgrind integration
- Basic input validation

## References

- [CWE Top 25](https://cwe.mitre.org/top25/)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [SEI CERT C Coding Standard](https://wiki.sei.cmu.edu/confluence/display/c/SEI+CERT+C+Coding+Standard)
- [Google's Sanitizers](https://github.com/google/sanitizers)

---

**Last Updated:** 2025-11-01
**Policy Version:** 1.0
