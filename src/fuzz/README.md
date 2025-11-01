# NIMCP Fuzzing Infrastructure

This directory contains fuzzing targets for NIMCP using libFuzzer. Fuzzing helps discover bugs, crashes, and security vulnerabilities by testing with millions of automatically generated inputs.

## Quick Start

### Build Fuzz Targets

```bash
mkdir build-fuzz && cd build-fuzz
cmake .. \
  -DCMAKE_BUILD_TYPE=Debug \
  -DENABLE_FUZZING=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++
make -j$(nproc)
```

### Run a Fuzzer (5 minutes)

```bash
cd src/fuzz
./fuzz_neuralnet -max_total_time=300
```

## Available Fuzz Targets

### 1. fuzz_neuralnet

**Tests:** Neural network API (nimcp_neuralnet.h)

**Coverage:**
- Network creation with random configurations
- Forward pass with random inputs
- Statistics gathering
- NULL pointer handling
- Edge case parameter values

**Run:**
```bash
./fuzz_neuralnet -max_total_time=600
```

**Expected findings:**
- Invalid parameter handling
- Memory allocation edge cases
- Numerical stability issues

### 2. fuzz_protocol

**Tests:** Protocol serialization/deserialization (nimcp_protocol.h)

**Coverage:**
- Message serialization
- Message deserialization
- Header validation
- Checksum calculation
- Event packet handling
- Malformed message parsing

**Run:**
```bash
./fuzz_protocol -max_total_time=600
```

**Expected findings:**
- Buffer overflow vulnerabilities
- Integer overflow in length fields
- Malformed message handling
- Checksum bypass attempts

### 3. fuzz_queue_manager

**Tests:** Queue manager API (nimcp_queue_manager.h)

**Coverage:**
- Queue creation with random configs
- Enqueue/dequeue operations
- Priority queue handling
- Concurrent access patterns
- NULL pointer handling

**Run:**
```bash
./fuzz_queue_manager -max_total_time=600
```

**Expected findings:**
- Race conditions
- Queue overflow/underflow
- Priority inversion bugs
- Resource leaks

### 4. fuzz_validate

**Tests:** Input validation functions (nimcp_validate.h)

**Coverage:**
- Range validation
- Pointer validation
- String validation
- Buffer validation
- Configuration validation

**Run:**
```bash
./fuzz_validate -max_total_time=600
```

**Expected findings:**
- Validation bypass attempts
- Edge case parameter combinations
- NULL/invalid pointer handling

## Usage Patterns

### Basic Fuzzing

```bash
# Fuzz for 5 minutes
./fuzz_neuralnet -max_total_time=300

# Fuzz with specific seed
./fuzz_neuralnet -seed=12345

# Limit input size
./fuzz_neuralnet -max_len=1024
```

### Corpus-Based Fuzzing

```bash
# Create corpus directory
mkdir corpus_neuralnet

# Fuzz with corpus
./fuzz_neuralnet corpus_neuralnet/ -max_total_time=3600

# Corpus will grow with interesting inputs
ls corpus_neuralnet/
```

### Parallel Fuzzing

```bash
# Run with 4 parallel workers
./fuzz_neuralnet corpus/ -workers=4 -jobs=4 -max_total_time=3600
```

### Dictionary-Guided Fuzzing

Create a dictionary file for protocol fuzzing:

```bash
cat > protocol.dict << EOF
# Magic number
"NIMC"
# Message types
"\x01\x00\x00\x00"
"\x02\x00\x00\x00"
# Common lengths
"\x00\x01\x00\x00"
"\x00\x10\x00\x00"
EOF

./fuzz_protocol -dict=protocol.dict -max_total_time=600
```

### Reproducing Crashes

When fuzzer finds a crash:

```bash
# Fuzzer output:
# ==12345== ERROR: AddressSanitizer: heap-buffer-overflow
# ...
# artifact_prefix='./'; Test unit written to ./crash-abc123

# Reproduce the crash:
./fuzz_neuralnet crash-abc123

# Debug with gdb:
gdb ./fuzz_neuralnet
(gdb) run crash-abc123
```

## Understanding Output

### Successful Fuzzing

```
#1      INITED cov: 245 ft: 312 corp: 1/1b exec/s: 0 rss: 41Mb
#8      NEW    cov: 267 ft: 345 corp: 2/3b lim: 4 exec/s: 0 rss: 41Mb
#1024   REDUCE cov: 267 ft: 345 corp: 2/2b lim: 11 exec/s: 0 rss: 42Mb
#4096   pulse  cov: 289 ft: 432 corp: 15/45b lim: 43 exec/s: 2048 rss: 45Mb
```

**Metrics:**
- `cov`: Code coverage (edges)
- `ft`: Features covered
- `corp`: Corpus size
- `exec/s`: Executions per second
- `rss`: Memory usage

### Crash Found

```
==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x...
    #0 0x... in nimcp_neuralnet_forward src/lib/nimcp_neuralnet.c:123
    #1 0x... in LLVMFuzzerTestOneInput src/fuzz/fuzz_neuralnet.cpp:45
```

**Action items:**
1. Save crash file
2. File bug report with crash info
3. Fix vulnerability
4. Re-run fuzzer to verify fix

## Best Practices

### 1. Start with Short Runs

```bash
# Quick sanity check (1 minute)
./fuzz_neuralnet -max_total_time=60

# Medium run (10 minutes)
./fuzz_neuralnet -max_total_time=600

# Deep fuzzing (1+ hours)
./fuzz_neuralnet corpus/ -max_total_time=3600
```

### 2. Monitor Coverage

```bash
# Generate coverage report after fuzzing
./fuzz_neuralnet corpus/ -max_total_time=600 -print_final_stats=1
```

### 3. Minimize Crashes

```bash
# Minimize a crashing input
./fuzz_neuralnet -minimize_crash=1 crash-abc123
```

### 4. Merge Corpora

```bash
# Merge multiple corpus directories
mkdir corpus_merged
./fuzz_neuralnet -merge=1 corpus_merged/ corpus1/ corpus2/ corpus3/
```

### 5. Use Sanitizers

Fuzzing is most effective with sanitizers enabled (automatically done by CMake):
- **AddressSanitizer**: Detects memory errors
- **UndefinedBehaviorSanitizer**: Detects UB
- **Coverage instrumentation**: Guides fuzzer to new code paths

## Integration with CI/CD

### GitHub Actions Example

```yaml
- name: Build fuzzers
  run: |
    mkdir build-fuzz && cd build-fuzz
    cmake .. -DENABLE_FUZZING=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    make -j$(nproc)

- name: Run short fuzz campaign
  run: |
    cd build-fuzz/src/fuzz
    ./fuzz_neuralnet -max_total_time=300 || true
    ./fuzz_protocol -max_total_time=300 || true
```

### OSS-Fuzz Integration

For continuous fuzzing with [OSS-Fuzz](https://github.com/google/oss-fuzz):

1. Add `oss-fuzz` directory with `build.sh`
2. Submit project to OSS-Fuzz
3. Receive daily fuzzing reports

## Troubleshooting

### Fuzzer Crashes Immediately

```bash
# Error: libFuzzer not found
# Solution: Use clang
cmake .. -DCMAKE_CXX_COMPILER=clang++

# Error: Sanitizer runtime not found
# Solution: Install sanitizer libraries
sudo apt-get install libasan6 libubsan1
```

### No New Coverage

If fuzzer stops finding new paths:
- Increase max input size: `-max_len=8192`
- Use dictionary: `-dict=my.dict`
- Seed corpus with valid inputs
- Check if API is well-fuzzed (plateau is normal)

### Out of Memory

```bash
# Limit memory usage
./fuzz_neuralnet -rss_limit_mb=2048
```

## Performance

| Fuzzer | Avg exec/s | Coverage (edges) | Notes |
|--------|-----------|------------------|-------|
| fuzz_neuralnet | ~5,000 | ~300 | Memory-intensive |
| fuzz_protocol | ~50,000 | ~450 | CPU-bound, fast |
| fuzz_queue_manager | ~20,000 | ~250 | Threading overhead |
| fuzz_validate | ~100,000 | ~150 | Very fast |

## Further Reading

- [libFuzzer Documentation](https://llvm.org/docs/LibFuzzer.html)
- [Efficient Fuzzing Guide](https://github.com/google/fuzzing/blob/master/docs/good-fuzz-target.md)
- [OSS-Fuzz](https://google.github.io/oss-fuzz/)
- [AFL++ (Alternative fuzzer)](https://github.com/AFLplusplus/AFLplusplus)

---

**Questions?** See [BUILD_SECURITY.md](../../BUILD_SECURITY.md) for more security testing information.
