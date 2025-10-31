/**
 * @file nimcp_validate.c
 * @brief Implementation of field validation utilities
 */

#include "utils/nimcp_validate.h"

/* Internal helper function prototypes */
static bool is_valid_utf8_char(const char* str, size_t len);

bool nimcp_validate_integer_field(const void* field_data, size_t size) {
    if (!field_data) {
        return false;
    }

    /* Validate size */
    switch (size) {
        case sizeof(int8_t):
        case sizeof(int16_t):
        case sizeof(int32_t):
        case sizeof(int64_t):
            break;
        default:
            NIMCP_LOG_ERROR("Invalid integer field size: %zu", size);
            return false;
    }

    /* Check alignment */
    if (((uintptr_t)field_data) % size != 0) {
        NIMCP_LOG_ERROR("Integer field misaligned");
        return false;
    }

    /* Validate range constraints if defined */
    switch (size) {
        case sizeof(int32_t): {
            const int32_t value = *(const int32_t*)field_data;
            if (value < NIMCP_INT32_MIN || value > NIMCP_INT32_MAX) {
                NIMCP_LOG_ERROR("Integer value out of range: %d", value);
                return false;
            }
            break;
        }
        case sizeof(int64_t): {
            const int64_t value = *(const int64_t*)field_data;
            if (value < NIMCP_INT64_MIN || value > NIMCP_INT64_MAX) {
                NIMCP_LOG_ERROR("Integer value out of range: %ld", value);
                return false;
            }
            break;
        }
    }

    return true;
}

bool nimcp_validate_float_field(const void* field_data, size_t size) {
    if (!field_data) {
        return false;
    }

    /* Validate size */
    switch (size) {
        case sizeof(float):
        case sizeof(double):
            break;
        default:
            NIMCP_LOG_ERROR("Invalid float field size: %zu", size);
            return false;
    }

    /* Check alignment */
    if (((uintptr_t)field_data) % size != 0) {
        NIMCP_LOG_ERROR("Float field misaligned");
        return false;
    }

    /* Validate value */
    if (size == sizeof(float)) {
        const float value = *(const float*)field_data;
        
        if (isnan(value)) {
            NIMCP_LOG_ERROR("Float field contains NaN");
            return false;
        }

        if (isinf(value)) {
            NIMCP_LOG_ERROR("Float field contains infinity");
            return false;
        }

        if (fabsf(value) > NIMCP_FLOAT_MAX) {
            NIMCP_LOG_ERROR("Float value out of range: %f", value);
            return false;
        }
    } else {
        const double value = *(const double*)field_data;
        
        if (isnan(value)) {
            NIMCP_LOG_ERROR("Double field contains NaN");
            return false;
        }

        if (isinf(value)) {
            NIMCP_LOG_ERROR("Double field contains infinity");
            return false;
        }

        if (fabs(value) > NIMCP_DOUBLE_MAX) {
            NIMCP_LOG_ERROR("Double value out of range: %f", value);
            return false;
        }
    }

    return true;
}

bool nimcp_validate_string_field(const void* field_data, size_t size) {
    if (!field_data || size == 0) {
        return false;
    }

    const char* str = (const char*)field_data;

    /* Check for NULL termination */
    if (str[size - 1] != '\0') {
        NIMCP_LOG_ERROR("String not NULL terminated");
        return false;
    }

    /* Get string length */
    size_t len = strnlen(str, size);
    if (len >= size) {
        NIMCP_LOG_ERROR("String exceeds field size");
        return false;
    }

    /* Validate length constraints */
    if (len > NIMCP_STRING_MAX_LENGTH) {
        NIMCP_LOG_ERROR("String too long: %zu chars", len);
        return false;
    }

    /* Validate character encoding and content */
    for (size_t i = 0; i < len; i++) {
        /* Check for control characters */
        if (iscntrl(str[i]) && str[i] != '\n' && str[i] != '\t') {
            NIMCP_LOG_ERROR("Invalid control character in string at pos %zu", i);
            return false;
        }

        /* Check for valid UTF-8 encoding */
        if (!is_valid_utf8_char(str + i, len - i)) {
            NIMCP_LOG_ERROR("Invalid UTF-8 encoding at pos %zu", i);
            return false;
        }
    }

    return true;
}

bool nimcp_validate_array_field(const void* field_data, size_t size) {
    if (!field_data || size < sizeof(NimcpArrayHeader)) {
        return false;
    }

    const NimcpArrayHeader* header = (const NimcpArrayHeader*)field_data;
    
    /* Validate header */
    if (header->element_count == 0 || 
        header->element_size == 0 ||
        header->element_count > NIMCP_ARRAY_MAX_ELEMENTS) {
        NIMCP_LOG_ERROR("Invalid array header values");
        return false;
    }

    /* Check total size */
    size_t required_size = sizeof(NimcpArrayHeader) + 
                          (header->element_count * header->element_size);
    if (required_size > size) {
        NIMCP_LOG_ERROR("Array size exceeds field size");
        return false;
    }

    /* Get pointer to array data */
    const uint8_t* array_data = (const uint8_t*)field_data + sizeof(NimcpArrayHeader);

    /* Check alignment */
    if (((uintptr_t)array_data) % NIMCP_ARRAY_ALIGNMENT != 0) {
        NIMCP_LOG_ERROR("Array data misaligned");
        return false;
    }

    /* Validate each element based on type */
    for (uint32_t i = 0; i < header->element_count; i++) {
        const void* element = array_data + (i * header->element_size);
        
        switch (header->element_type) {
            case NIMCP_ARRAY_INTEGER:
                if (!nimcp_validate_integer_field(element, header->element_size)) {
                    NIMCP_LOG_ERROR("Invalid integer array element at index %u", i);
                    return false;
                }
                break;

            case NIMCP_ARRAY_FLOAT:
                if (!nimcp_validate_float_field(element, header->element_size)) {
                    NIMCP_LOG_ERROR("Invalid float array element at index %u", i);
                    return false;
                }
                break;

            case NIMCP_ARRAY_STRING:
                if (!nimcp_validate_string_field(element, header->element_size)) {
                    NIMCP_LOG_ERROR("Invalid string array element at index %u", i);
                    return false;
                }
                break;

            default:
                NIMCP_LOG_ERROR("Unknown array element type: %d", 
                              header->element_type);
                return false;
        }
    }

    return true;
}

bool nimcp_validate_state_fields(const void* state_data, size_t size) {
    if (!state_data || size < sizeof(NimcpStateHeader)) {
        NIMCP_LOG_ERROR("Invalid state data pointer or size");
        return false;
    }

    const NimcpStateHeader* header = (const NimcpStateHeader*)state_data;
    
    /* Validate header */
    if (header->magic != NIMCP_STATE_MAGIC) {
        NIMCP_LOG_ERROR("Invalid state magic number");
        return false;
    }

    if (header->field_count == 0 || header->field_count > NIMCP_MAX_FIELDS) {
        NIMCP_LOG_ERROR("Invalid field count: %u", header->field_count);
        return false;
    }

    /* Get field table */
    const NimcpStateField* fields = 
        (const NimcpStateField*)((const uint8_t*)state_data + sizeof(NimcpStateHeader));
    
    /* Track used memory regions to detect overlaps */
    uint8_t* usage_map = calloc(size, sizeof(uint8_t));
    if (!usage_map) {
        NIMCP_LOG_ERROR("Failed to allocate usage map");
        return false;
    }

    /* Mark header region as used */
    memset(usage_map, 1, sizeof(NimcpStateHeader) + 
           header->field_count * sizeof(NimcpStateField));

    /* Validate each field */
    for (uint32_t i = 0; i < header->field_count; i++) {
        const NimcpStateField* field = &fields[i];

        /* Check field boundaries */
        if (field->offset + field->size > size) {
            NIMCP_LOG_ERROR("Field %u exceeds state boundaries", i);
            free(usage_map);
            return false;
        }

        /* Check for overlaps */
        for (size_t j = 0; j < field->size; j++) {
            if (usage_map[field->offset + j]) {
                NIMCP_LOG_ERROR("Field %u overlaps with other data", i);
                free(usage_map);
                return false;
            }
            usage_map[field->offset + j] = 1;
        }

        /* Validate field data based on type */
        const void* field_data = (const uint8_t*)state_data + field->offset;
        bool valid = false;

        switch (field->type) {
            case NIMCP_FIELD_INTEGER:
                valid = nimcp_validate_integer_field(field_data, field->size);
                break;

            case NIMCP_FIELD_FLOAT:
                valid = nimcp_validate_float_field(field_data, field->size);
                break;

            case NIMCP_FIELD_STRING:
                valid = nimcp_validate_string_field(field_data, field->size);
                break;

            case NIMCP_FIELD_ARRAY:
                valid = nimcp_validate_array_field(field_data, field->size);
                break;

            default:
                NIMCP_LOG_ERROR("Unknown field type %d for field %u", 
                              field->type, i);
                free(usage_map);
                return false;
        }

        if (!valid) {
            free(usage_map);
            return false;
        }
    }

    free(usage_map);
    return true;
}

/* Internal helper function implementation */
static bool is_valid_utf8_char(const char* str, size_t len) {
    if (!str || len == 0) {
        return false;
    }

    /* Check first byte to determine sequence length */
    uint8_t first = (uint8_t)str[0];
    size_t seq_len;

    if ((first & 0x80) == 0) {          /* 0xxxxxxx */
        return true;
    } else if ((first & 0xE0) == 0xC0) { /* 110xxxxx */
        seq_len = 2;
    } else if ((first & 0xF0) == 0xE0) { /* 1110xxxx */
        seq_len = 3;
    } else if ((first & 0xF8) == 0xF0) { /* 11110xxx */
        seq_len = 4;
    } else {
        return false;
    }

    /* Check if string has enough bytes */
    if (len < seq_len) {
        return false;
    }

    /* Validate continuation bytes */
    for (size_t i = 1; i < seq_len; i++) {
        if ((str[i] & 0xC0) != 0x80) {
            return false;
        }
    }

    return true;
}
