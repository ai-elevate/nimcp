/**
 * @file fuzz_queue_manager.cpp
 * @brief Fuzzing target for queue manager API
 *
 * Tests queue operations with random inputs to discover race conditions,
 * memory errors, and crash bugs.
 *
 * Build:
 *   cmake -DENABLE_FUZZING=ON ..
 *   make fuzz_queue_manager
 *
 * Run:
 *   ./fuzz_queue_manager -max_total_time=300
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "utils/nimcp_queue_manager.h"

#define MIN_CONFIG_SIZE (sizeof(uint32_t) * 3)

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < MIN_CONFIG_SIZE) {
        return 0;
    }

    // Extract configuration from fuzzer data
    queue_manager_config_t config;
    memcpy(&config, data, sizeof(config));

    // Sanitize config to prevent excessive allocations
    config.num_channels = (config.num_channels % 8) + 1;                     // 1-8 channels
    config.messages_per_channel = (config.messages_per_channel % 100) + 10;  // 10-109
    config.thread_pool_size = (config.thread_pool_size % 4) + 1;             // 1-4 threads
    config.timeout_ms = config.timeout_ms % 1000;                            // Max 1 second

    // Try to create queue manager
    queue_manager_t* manager = queue_manager_create(&config);

    if (manager != nullptr) {
        // Try various operations with remaining data
        size_t offset = sizeof(config);

        // Test enqueue operations
        while (offset + sizeof(uint32_t) + 16 <= size) {
            uint32_t channel = *(reinterpret_cast<const uint32_t*>(data + offset));
            channel = channel % config.num_channels;
            offset += sizeof(uint32_t);

            queue_message_t msg;
            msg.priority = data[offset] % 3;  // 0-2 priority
            msg.type = data[offset + 1] % 256;
            msg.size = (data[offset + 2] % 64) + 1;  // 1-64 bytes
            msg.data = const_cast<uint8_t*>(data + offset + 3);
            offset += 3 + msg.size;

            if (offset <= size) {
                queue_manager_enqueue(manager, channel, &msg);
            } else {
                break;
            }
        }

        // Test dequeue operations
        for (uint32_t ch = 0; ch < config.num_channels; ch++) {
            queue_message_t msg;
            queue_manager_dequeue(manager, ch, &msg, 10);  // 10ms timeout
        }

        // Test query operations
        for (uint32_t ch = 0; ch < config.num_channels; ch++) {
            queue_manager_is_empty(manager, ch);
            queue_manager_is_full(manager, ch);
            queue_manager_get_size(manager, ch);
        }

        // Test stats
        queue_stats_t stats;
        queue_manager_get_stats(manager, 0, &stats);

        // Clean up
        queue_manager_destroy(manager);
    }

    // Test null pointer handling
    queue_manager_destroy(nullptr);

    return 0;
}
