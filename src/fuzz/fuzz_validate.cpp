/**
 * @file fuzz_validate.cpp
 * @brief Fuzzing target for input validation functions
 *
 * Tests validation logic with extreme and malformed inputs to ensure
 * validators properly reject bad data without crashing.
 *
 * Build:
 *   cmake -DENABLE_FUZZING=ON ..
 *   make fuzz_validate
 *
 * Run:
 *   ./fuzz_validate -max_total_time=300
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "utils/nimcp_validate.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < 4) {
        return 0;
    }

    // Test various validation functions with fuzzer data

    // Test 1: Validate range with random values
    {
        if (size >= sizeof(int32_t) * 3) {
            int32_t value = *reinterpret_cast<const int32_t*>(data);
            int32_t min = *reinterpret_cast<const int32_t*>(data + 4);
            int32_t max = *reinterpret_cast<const int32_t*>(data + 8);

            nimcp_validate_range(value, min, max);
        }
    }

    // Test 2: Validate pointer with random addresses
    {
        void* ptr = reinterpret_cast<void*>(*reinterpret_cast<const uintptr_t*>(data));
        nimcp_validate_pointer(ptr);
    }

    // Test 3: Validate string with fuzzer data
    {
        // Create null-terminated string from fuzzer data
        char buffer[256];
        size_t copy_len = (size > 255) ? 255 : size;
        memcpy(buffer, data, copy_len);
        buffer[copy_len] = '\0';

        nimcp_validate_string(buffer, 256);
    }

    // Test 4: Validate buffer with random sizes
    {
        if (size >= sizeof(uint32_t)) {
            uint32_t buffer_size = *reinterpret_cast<const uint32_t*>(data);
            buffer_size = buffer_size % 65536;  // Max 64KB

            nimcp_validate_buffer(data, size, buffer_size);
        }
    }

    // Test 5: Validate configuration structures
    {
        // Test with malformed config data
        if (size >= 32) {
            nimcp_validate_config(data, size);
        }
    }

    return 0;
}
