/**
 * @file fuzz_protocol.cpp
 * @brief Fuzzing target for protocol serialization/deserialization
 *
 * Tests protocol message handling with malformed and edge-case inputs
 * to discover parsing vulnerabilities, buffer overflows, and crashes.
 *
 * Build:
 *   cmake -DENABLE_FUZZING=ON ..
 *   make fuzz_protocol
 *
 * Run:
 *   ./fuzz_protocol -max_total_time=300
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include "nimcp_protocol.h"

#define MAX_BUFFER_SIZE 4096

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size < sizeof(msg_header_t)) {
        return 0;
    }

    // Test 1: Deserialization of fuzzer data
    {
        msg_header_t header;
        uint8_t payload[MAX_BUFFER_SIZE];

        int result = protocol_deserialize_message(data, size, &header, payload, sizeof(payload));

        // If deserialization succeeded, validate the header
        if (result > 0) {
            protocol_validate_header(&header);

            // Calculate checksum
            protocol_calculate_checksum(&header, payload, header.length);
        }
    }

    // Test 2: Serialization with fuzzer-generated message type
    if (size >= sizeof(msg_type_t) + 4) {
        msg_type_t type = *reinterpret_cast<const msg_type_t*>(data);
        uint32_t payload_len = *reinterpret_cast<const uint32_t*>(data + sizeof(msg_type_t));

        // Clamp payload length
        payload_len = payload_len % MAX_BUFFER_SIZE;

        const uint8_t* payload = data + sizeof(msg_type_t) + sizeof(uint32_t);
        size_t available = size - sizeof(msg_type_t) - sizeof(uint32_t);

        if (payload_len > available) {
            payload_len = available;
        }

        uint8_t buffer[MAX_BUFFER_SIZE];
        protocol_serialize_message(type, payload, payload_len, buffer, sizeof(buffer));
    }

    // Test 3: Header validation with crafted headers
    if (size >= sizeof(msg_header_t)) {
        msg_header_t header;
        memcpy(&header, data, sizeof(header));
        protocol_validate_header(&header);
    }

    // Test 4: Event packet serialization/deserialization
    if (size >= sizeof(event_packet_t)) {
        event_packet_t packet;
        memcpy(&packet, data, sizeof(packet));

        // Clamp payload length
        packet.payload_length = packet.payload_length % MAX_BUFFER_SIZE;

        uint8_t buffer[MAX_BUFFER_SIZE * 2];
        int result = event_packet_serialize(&packet, data + sizeof(packet), buffer, sizeof(buffer));

        if (result > 0) {
            event_packet_t decoded;
            uint8_t decoded_payload[MAX_BUFFER_SIZE];
            event_packet_deserialize(buffer, result, &decoded, decoded_payload,
                                     sizeof(decoded_payload));
        }
    }

    return 0;
}
