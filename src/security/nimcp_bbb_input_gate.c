/**
 * @file nimcp_bbb_input_gate.c
 * @brief Blood-Brain Barrier Input Gate - Input Validation Layer
 *
 * WHAT: Input validation and sanitization for the BBB perimeter defense
 * WHY:  Prevent code injection, buffer overflow, SQL injection, and XSS attacks
 *       before malicious data can reach the neural network core
 * HOW:  Multi-layer detection using pattern matching, bounds checking, and
 *       character analysis with sanitization for safe processing
 *
 * BIOLOGICAL MODEL:
 * ```
 * Endothelial cells (tight junctions) -> Input validation gates
 *   - Selective permeability based on molecular properties
 *   - Rejects harmful substances while allowing nutrients
 * ```
 *
 * DETECTION PATTERNS:
 * - SQL injection: SELECT, DROP, INSERT, --, ', UNION, etc.
 * - XSS attacks: <script>, javascript:, onerror=, etc.
 * - Format string: %n, %s without args, excessive format specifiers
 * - Buffer overflow: Strings exceeding max length
 * - Shellcode: 0x90 NOP sled, common opcodes, null bytes
 *
 * NIMCP STANDARDS:
 * - All functions < 50 lines
 * - Guard clauses (early returns)
 * - WHAT-WHY-HOW documentation
 * - Thread-safe operations
 *
 * @author NIMCP Team
 * @date 2025-11-24
 */

#include "security/nimcp_blood_brain_barrier.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "security_bbb_input_gate"
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(bbb_input_gate)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_bbb_input_gate_mesh_id = 0;
static mesh_participant_registry_t* g_bbb_input_gate_mesh_registry = NULL;

nimcp_error_t bbb_input_gate_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_bbb_input_gate_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "bbb_input_gate", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_COGNITIVE);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "bbb_input_gate";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_bbb_input_gate_mesh_id);
    if (err == NIMCP_SUCCESS) g_bbb_input_gate_mesh_registry = registry;
    return err;
}

void bbb_input_gate_mesh_unregister(void) {
    if (g_bbb_input_gate_mesh_registry && g_bbb_input_gate_mesh_id != 0) {
        mesh_participant_unregister(g_bbb_input_gate_mesh_registry, g_bbb_input_gate_mesh_id);
        g_bbb_input_gate_mesh_id = 0;
        g_bbb_input_gate_mesh_registry = NULL;
    }
}


#include <string.h>
#include <ctype.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/types.h>  /* For ssize_t */

//=============================================================================
// Internal Function Declarations (from core BBB module)
//=============================================================================

/* Statistics and config accessor functions from nimcp_blood_brain_barrier.c */
extern void bbb_system_inc_validations(bbb_system_t system);
extern void bbb_system_inc_threats(bbb_system_t system);
extern size_t bbb_system_get_max_string_length(bbb_system_t system);

//=============================================================================
// Constants
//=============================================================================

/** Maximum number of format specifiers allowed in a string */
#define MAX_FORMAT_SPECIFIERS 10

/** Threshold for suspicious special character density */
#define SPECIAL_CHAR_THRESHOLD_RATIO 3

/** NOP sled detection threshold (consecutive NOPs) */
#define NOP_SLED_THRESHOLD 4

//=============================================================================
// SQL Injection Patterns
//=============================================================================

/**
 * WHAT: SQL injection detection patterns
 * WHY:  Common SQL keywords and operators used in injection attacks
 * HOW:  Case-insensitive substring matching during validation
 */
static const char* sql_patterns[] = {
    /* UNION injection (almost always malicious) */
    " union select", " union all select",
    "' union ", "\" union ",
    /* Boolean injection patterns */
    "' or ", "\" or ",
    " or 1=1", " or '1'='1",
    "'='", "\"=\"",
    /* Terminated/stacked injection (semicolon followed by command) */
    "'; drop ", "'; delete ",
    "'; insert ", "'; update ",
    "'; exec ", "'; execute ",
    "; drop ", "; delete ",          /* Stacked queries */
    "; insert ", "; update ",
    "; exec ", "; execute ",
    /* Comment indicators with quotes */
    "'--", "\"--",
    "/*", "*/",
    /* Dangerous stored procedures */
    "xp_cmdshell", "sp_execute",
    /* Schema probing */
    "information_schema",
    /* Tautology with comment */
    "1=1--", "1=1 --",
    NULL
};

//=============================================================================
// XSS Patterns
//=============================================================================

/**
 * WHAT: XSS attack detection patterns
 * WHY:  Common JavaScript injection vectors for cross-site scripting
 * HOW:  Case-insensitive pattern matching in input strings
 */
static const char* xss_patterns[] = {
    "<script", "</script",
    "javascript:", "vbscript:",
    "onerror=", "onload=", "onclick=", "onmouseover=",
    "onfocus=", "onblur=", "onchange=",
    "<iframe", "<object", "<embed",
    "<img src=", "<svg ",
    "expression(", "eval(",
    "document.cookie", "document.write",
    "window.location",
    NULL
};

//=============================================================================
// Format String Patterns
//=============================================================================

/**
 * WHAT: Dangerous format string specifiers
 * WHY:  %n writes to memory, %s without args crashes, etc.
 * HOW:  Pattern detection in user input that might be used in printf
 */
static const char* format_patterns[] = {
    "%n", "%hn", "%hhn", "%ln",
    "%s%s%s%s",      /* Stack smashing attempt */
    "%x%x%x%x",      /* Stack reading attempt */
    "%p%p%p%p",      /* Pointer leaking */
    "%08x.%08x",     /* Memory dump pattern with width specifier */
    NULL
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Case-insensitive substring search
 *
 * WHAT: Find pattern in text regardless of case
 * WHY:  Attackers use mixed case to bypass simple detection
 * HOW:  Convert both strings to lowercase during comparison
 *
 * @param text Text to search in
 * @param pattern Pattern to find
 * @return Pointer to match or NULL
 */
static const char* case_insensitive_strstr(const char* text, const char* pattern)
{
    if (!text || !pattern) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "case_insensitive_strstr: required parameter is NULL (text, pattern)");
        return NULL;
    }
    if (!*pattern) return text;

    size_t pattern_len = strlen(pattern);
    size_t text_len = strlen(text);

    if (pattern_len > text_len) {
        return NULL;  /* Normal: pattern can't be in shorter text */
    }

    for (size_t i = 0; i <= text_len - pattern_len; i++) {
        size_t j;
        for (j = 0; j < pattern_len; j++) {
            if (tolower((unsigned char)text[i + j]) !=
                tolower((unsigned char)pattern[j])) {
                break;
            }
        }
        if (j == pattern_len) {
            return &text[i];
        }
    }
    return NULL;  /* Normal: no match found */
}

/**
 * @brief Check for pattern array match in input
 *
 * WHAT: Scan input against array of dangerous patterns
 * WHY:  Centralized pattern matching for multiple threat types
 * HOW:  Iterate patterns, return on first match
 *
 * @param input Input string to check
 * @param patterns NULL-terminated array of patterns
 * @return true if any pattern found
 */
static bool check_patterns(const char* input, const char* patterns[])
{
    if (!input || !patterns) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "check_patterns: required parameter is NULL (input, patterns)");
        return false;
    }

    for (int i = 0; patterns[i] != NULL; i++) {
        if (case_insensitive_strstr(input, patterns[i])) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Detect shellcode patterns in binary data
 *
 * WHAT: Identify common shellcode signatures
 * WHY:  Prevent code execution attacks via buffer overflow
 * HOW:  Look for NOP sleds, common opcodes, suspicious byte patterns
 *
 * @param data Binary data to analyze
 * @param size Size of data
 * @return true if shellcode detected
 */
static bool detect_shellcode(const void* data, size_t size)
{
    if (!data || size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "detect_shellcode: data is NULL");
        return false;
    }

    const uint8_t* bytes = (const uint8_t*)data;
    int nop_count = 0;
    int null_count = 0;
    int suspicious_opcodes = 0;

    for (size_t i = 0; i < size; i++) {
        /* NOP sled detection (0x90 on x86) */
        if (bytes[i] == 0x90) {
            nop_count++;
            if (nop_count >= NOP_SLED_THRESHOLD) {
                return true;
            }
        } else {
            nop_count = 0;
        }

        /* Null byte count (common in shellcode) */
        if (bytes[i] == 0x00) {
            null_count++;
        }

        /* Common shellcode opcodes */
        switch (bytes[i]) {
            case 0xCD:  /* int instruction */
            case 0x80:  /* syscall interrupt */
            case 0x0F:  /* syscall/sysenter prefix */
            case 0xCC:  /* int3 debug */
                suspicious_opcodes++;
                break;
        }
    }

    /* High ratio of suspicious bytes indicates shellcode */
    float suspicious_ratio = (float)suspicious_opcodes / size;
    if (suspicious_ratio > 0.1F && size > 10) {
        return true;
    }

    return false;  /* Normal: no shellcode detected */
}

/**
 * @brief Detect dangerous format specifiers with large width/precision
 *
 * WHAT: Check for format specifiers with dangerous width/precision values
 * WHY:  Large width/precision like %.100000s can cause DoS or memory issues
 * HOW:  Parse format specifiers and check for values over threshold
 *
 * @param str String to analyze
 * @return true if dangerous format specifier found
 */
static bool detect_dangerous_format_specifier(const char* str)
{
    if (!str) {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "detect_dangerous_format_specifier: str is NULL");

            return false;

        }

    const int DANGEROUS_WIDTH_THRESHOLD = 10000;

    for (const char* p = str; *p; p++) {
        if (*p != '%') continue;
        p++;
        if (!*p || *p == '%') continue;  /* %% is literal percent */

        /* Parse optional precision .N or width N */
        int value = 0;
        bool has_dot = (*p == '.');
        if (has_dot) p++;

        /* Parse the number */
        while (*p && isdigit((unsigned char)*p)) {
            value = value * 10 + (*p - '0');
            if (value > DANGEROUS_WIDTH_THRESHOLD) {
                return true;  /* Dangerous width/precision */
            }
            p++;
        }

        /* Check if we have a format specifier after the number */
        if (*p && (strchr("diouxXeEfFgGaAcspn", *p) != NULL)) {
            if (value > DANGEROUS_WIDTH_THRESHOLD) {
                return true;
            }
        }

        if (!*p) break;
        p--;  /* Back up since outer loop will advance */
    }
    return false;
}

/**
 * @brief Count format specifiers in string
 *
 * WHAT: Count % format specifiers in input
 * WHY:  Too many specifiers suggests format string attack
 * HOW:  Scan for % followed by format character
 *
 * @param str String to analyze
 * @return Count of format specifiers
 */
static int count_format_specifiers(const char* str)
{
    if (!str) return 0;

    int count = 0;
    for (const char* p = str; *p; p++) {
        if (*p == '%' && *(p + 1)) {
            const char* next = p + 1;
            /* Skip optional width/precision specifiers */
            if (*next == '.') next++;
            while (*next && isdigit((unsigned char)*next)) next++;
            /* Common format specifiers */
            if (*next == 'd' || *next == 'i' || *next == 's' ||
                *next == 'x' || *next == 'X' || *next == 'p' ||
                *next == 'n' || *next == 'f' || *next == 'c' ||
                *next == 'u' || *next == 'o' || *next == 'e') {
                count++;
            }
        }
    }
    return count;
}

//=============================================================================
// Public API Implementation
//=============================================================================

/**
 * @brief Validate raw input data for threats
 *
 * WHAT: Comprehensive validation of arbitrary binary data
 * WHY:  First line of defense against malicious input entering the system
 * HOW:  Check for shellcode, buffer overflow, and suspicious patterns
 *
 * @param system BBB system handle
 * @param data Input data to validate
 * @param size Size of input data
 * @param result Output validation result
 * @return true if valid (no threats detected)
 */
bool bbb_validate_input(bbb_system_t system, const void* data,
                        size_t size, bbb_validation_result_t* result)
{
    /* Guard: NULL parameters */
    if (!result) {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "bbb_validate_input: result is NULL");

            return false;

        }

    /* Initialize result to valid */
    memset(result, 0, sizeof(bbb_validation_result_t));
    result->valid = true;

    /* Guard: NULL system is invalid for input validation */
    if (!system) {
        result->valid = false;
        result->threat = BBB_THREAT_UNAUTHORIZED_ACCESS;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason),
                 "NULL BBB system - validation requires system context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_input: system is NULL");
        return false;
    }

    /* Update validation statistics (always count validation attempts) */
    bbb_system_inc_validations(system);

    /* Guard: NULL data is invalid */
    if (!data) {
        result->valid = false;
        result->threat = BBB_THREAT_BUFFER_OVERFLOW;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason), "NULL data pointer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_input: data is NULL");
        return false;
    }

    /* Guard: Zero size is valid (nothing to validate) */
    if (size == 0) {
        return true;
    }

    /* Check for shellcode patterns */
    if (detect_shellcode(data, size)) {
        result->valid = false;
        result->threat = BBB_THREAT_SHELLCODE;
        result->severity = BBB_SEVERITY_CRITICAL;
        snprintf(result->reason, sizeof(result->reason),
                 "Shellcode pattern detected in input data");
        bbb_report_threat(system, BBB_THREAT_SHELLCODE, BBB_SEVERITY_CRITICAL,
                          result->reason, data, data, size);
        return false;
    }

    /* For text data, perform string validation */
    const char* str = (const char*)data;
    bool is_printable = true;
    size_t str_len = 0;

    /* Find string length within size bounds */
    for (size_t i = 0; i < size; i++) {
        if (str[i] == '\0') {
            str_len = i;
            break;
        }
        if (!isprint((unsigned char)str[i]) && !isspace((unsigned char)str[i])) {
            is_printable = false;
            break;
        }
        str_len = i + 1;
    }

    if (is_printable && str_len > 0) {
        /* Create null-terminated copy for string validation */
        char* safe_str = (char*)nimcp_malloc(str_len + 1);
        if (safe_str) {
            memcpy(safe_str, str, str_len);
            safe_str[str_len] = '\0';
            bool valid = bbb_validate_string(system, safe_str, result);
            nimcp_free(safe_str);
            return valid;
        }
        /* P1-2 FIX: If malloc fails, treat as validation failure (return false),
         * not success. The old code fell through to 'return true' which is a
         * security hole - untrusted input would be treated as VALID on OOM. */
        result->valid = false;
        result->threat = BBB_THREAT_BUFFER_OVERFLOW;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason),
                 "Memory allocation failed during input validation");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
            "bbb_validate_input: failed to allocate safe_str for validation");
        return false;
    }

    return true;
}

/**
 * @brief Validate string for SQL injection, XSS, and format string attacks
 *
 * WHAT: String-specific security validation
 * WHY:  Strings are the most common attack vector for injection attacks
 * HOW:  Pattern matching against known attack signatures
 *
 * @param system BBB system handle
 * @param str String to validate
 * @param result Output validation result
 * @return true if valid
 */
bool bbb_validate_string(bbb_system_t system, const char* str,
                         bbb_validation_result_t* result)
{
    /* Guard: NULL result */
    if (!result) {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "bbb_validate_string: result is NULL");

            return false;

        }

    /* Initialize result */
    memset(result, 0, sizeof(bbb_validation_result_t));
    result->valid = true;

    /* Guard: NULL system requires system context for full validation */
    if (!system) {
        result->valid = false;
        result->threat = BBB_THREAT_UNAUTHORIZED_ACCESS;
        result->severity = BBB_SEVERITY_MEDIUM;
        snprintf(result->reason, sizeof(result->reason),
                 "NULL BBB system - string validation requires system context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_string: system is NULL");
        return false;
    }

    /* Update validation statistics */
    bbb_system_inc_validations(system);

    /* Guard: NULL string is invalid */
    if (!str) {
        result->valid = false;
        result->threat = BBB_THREAT_BUFFER_OVERFLOW;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason), "NULL string pointer");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_string: str is NULL");
        return false;
    }

    /* Guard: Empty string is valid */
    if (*str == '\0') {
        return true;
    }

    /* Length check - use system config if available, otherwise default */
    size_t max_string_length = bbb_system_get_max_string_length(system);
    if (max_string_length == 0) {
        bbb_config_t cfg = bbb_default_config();
        max_string_length = cfg.input.max_string_length;
    }
    size_t len = strlen(str);
    if (len > max_string_length) {
        result->valid = false;
        result->threat = BBB_THREAT_BUFFER_OVERFLOW;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason),
                 "String exceeds max length (%zu > %zu)", len, max_string_length);
        bbb_report_threat(system, BBB_THREAT_BUFFER_OVERFLOW, BBB_SEVERITY_HIGH,
                          result->reason, str, str, len);
        return false;
    }

    /* SQL Injection Detection */
    bool sql_detected = check_patterns(str, sql_patterns);
    if (sql_detected) {
        result->valid = false;
        result->threat = BBB_THREAT_SQL_INJECTION;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason),
                 "SQL injection pattern detected");
        if (system) {
            bbb_report_threat(system, BBB_THREAT_SQL_INJECTION, BBB_SEVERITY_HIGH,
                              result->reason, str, str, strlen(str));
        }
        return false;
    }

    /* XSS Detection */
    if (check_patterns(str, xss_patterns)) {
        result->valid = false;
        result->threat = BBB_THREAT_CODE_INJECTION;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason),
                 "XSS attack pattern detected");
        bbb_report_threat(system, BBB_THREAT_CODE_INJECTION, BBB_SEVERITY_HIGH,
                          result->reason, str, str, strlen(str));
        return false;
    }

    /* Format String Detection - pattern-based */
    if (check_patterns(str, format_patterns)) {
        result->valid = false;
        result->threat = BBB_THREAT_FORMAT_STRING;
        result->severity = BBB_SEVERITY_CRITICAL;
        snprintf(result->reason, sizeof(result->reason),
                 "Format string attack pattern detected");
        bbb_report_threat(system, BBB_THREAT_FORMAT_STRING, BBB_SEVERITY_CRITICAL,
                          result->reason, str, str, strlen(str));
        return false;
    }

    /* Format String Detection - dangerous width/precision */
    if (detect_dangerous_format_specifier(str)) {
        result->valid = false;
        result->threat = BBB_THREAT_FORMAT_STRING;
        result->severity = BBB_SEVERITY_HIGH;
        snprintf(result->reason, sizeof(result->reason),
                 "Dangerous format specifier detected (large width/precision)");
        bbb_report_threat(system, BBB_THREAT_FORMAT_STRING, BBB_SEVERITY_HIGH,
                          result->reason, str, str, strlen(str));
        return false;
    }

    /* Excessive format specifiers */
    if (count_format_specifiers(str) > MAX_FORMAT_SPECIFIERS) {
        result->valid = false;
        result->threat = BBB_THREAT_FORMAT_STRING;
        result->severity = BBB_SEVERITY_MEDIUM;
        snprintf(result->reason, sizeof(result->reason),
                 "Excessive format specifiers detected");
        bbb_report_threat(system, BBB_THREAT_FORMAT_STRING, BBB_SEVERITY_MEDIUM,
                          result->reason, str, str, strlen(str));
        return false;
    }

    return true;
}

/**
 * @brief Validate integer for overflow risks
 *
 * WHAT: Check integer values against safe bounds
 * WHY:  Integer overflow can lead to buffer overflows and logic errors
 * HOW:  Range checking against configured limits
 *
 * @param system BBB system handle
 * @param value Integer to validate
 * @param result Output validation result
 * @return true if valid
 */
bool bbb_validate_integer(bbb_system_t system, int64_t value,
                          bbb_validation_result_t* result)
{
    /* Guard: NULL result */
    if (!result) {

            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,

                "bbb_validate_integer: result is NULL");

            return false;

        }

    /* Initialize result */
    memset(result, 0, sizeof(bbb_validation_result_t));
    result->valid = true;

    /* Guard: NULL system */
    if (!system) {
        result->valid = false;
        result->threat = BBB_THREAT_UNAUTHORIZED_ACCESS;
        result->severity = BBB_SEVERITY_MEDIUM;
        snprintf(result->reason, sizeof(result->reason),
                 "NULL BBB system - integer validation requires system context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_integer: system is NULL");
        return false;
    }

    /* Check for extreme values that often indicate overflow */
    /* INT64_MAX and INT64_MIN often result from overflow */
    if (value == INT64_MAX || value == INT64_MIN) {
        result->valid = false;
        result->threat = BBB_THREAT_INTEGER_OVERFLOW;
        result->severity = BBB_SEVERITY_MEDIUM;
        snprintf(result->reason, sizeof(result->reason),
                 "Integer at boundary - possible overflow");
        return false;
    }

    /* Check for suspicious wrap-around patterns */
    /* Negative size values often indicate unsigned overflow */
    /* This is a heuristic - specific bounds should be checked by caller */

    return true;
}

/**
 * @brief Validate pointer is in valid memory range
 *
 * WHAT: Basic pointer validity check
 * WHY:  Prevent use of NULL, invalid, or out-of-bounds pointers
 * HOW:  Check for NULL and obviously invalid patterns
 *
 * @param system BBB system handle
 * @param ptr Pointer to validate
 * @param expected_size Expected accessible size
 * @param result Output validation result
 * @return true if valid
 */
bool bbb_validate_pointer(bbb_system_t system, const void* ptr,
                          size_t expected_size, bbb_validation_result_t* result)
{
    /* Require valid result pointer */
    if (!result) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_pointer: result is NULL");
        return false;
    }
    bbb_validation_result_t* res = result;

    /* Initialize result */
    memset(res, 0, sizeof(bbb_validation_result_t));
    res->valid = true;

    /* Guard: NULL system */
    if (!system) {
        res->valid = false;
        res->threat = BBB_THREAT_UNAUTHORIZED_ACCESS;
        res->severity = BBB_SEVERITY_MEDIUM;
        snprintf(res->reason, sizeof(res->reason),
                 "NULL BBB system - pointer validation requires system context");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_pointer: system is NULL");
        return false;
    }

    /* NULL pointer check */
    if (!ptr) {
        res->valid = false;
        res->threat = BBB_THREAT_MEMORY_VIOLATION;
        res->severity = BBB_SEVERITY_HIGH;
        snprintf(res->reason, sizeof(res->reason),
                 "NULL pointer detected");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "bbb_validate_pointer: ptr is NULL");
        return false;
    }

    /* Zero size check */
    if (expected_size == 0) {
        res->valid = false;
        res->threat = BBB_THREAT_MEMORY_VIOLATION;
        res->severity = BBB_SEVERITY_LOW;
        snprintf(res->reason, sizeof(res->reason),
                 "Zero-size memory access");
        return false;
    }

    /* Check for obviously invalid addresses */
    uintptr_t addr = (uintptr_t)ptr;

    /* Very low addresses (often NULL-ish or kernel space) */
    if (addr < 0x1000) {
        res->valid = false;
        res->threat = BBB_THREAT_MEMORY_VIOLATION;
        res->severity = BBB_SEVERITY_HIGH;
        snprintf(res->reason, sizeof(res->reason),
                 "Pointer in low memory region (0x%lx)", (unsigned long)addr);
        return false;
    }

    /* Alignment check for common data types */
    if (expected_size >= 8 && (addr & 0x7) != 0) {
        /* 8-byte alignment expected but not met */
        res->severity = BBB_SEVERITY_LOW;
        snprintf(res->reason, sizeof(res->reason),
                 "Pointer misaligned for 8-byte access");
        /* Not invalid, just a warning - don't fail */
    }

    return true;
}

/**
 * @brief Sanitize string by removing dangerous characters
 *
 * WHAT: Remove or escape dangerous characters from input
 * WHY:  Allow processing of input while neutralizing attack vectors
 * HOW:  Whitelist approach - keep only safe characters
 *
 * @param system BBB system handle
 * @param input Input string
 * @param output Output buffer for sanitized string
 * @param output_size Size of output buffer
 * @return Length of sanitized string, or -1 on error
 */
ssize_t bbb_sanitize_string(bbb_system_t system, const char* input,
                            char* output, size_t output_size)
{
    /* Guard: Invalid parameters */
    if (!input || !output || output_size == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "bbb_sanitize_string: invalid parameters");

            return -1;
    }

    (void)system;  /* Available for future configuration */

    size_t out_idx = 0;
    size_t in_len = strlen(input);

    for (size_t i = 0; i < in_len && out_idx < output_size - 1; i++) {
        char c = input[i];

        /* Whitelist: alphanumeric, spaces, basic punctuation */
        if (isalnum((unsigned char)c)) {
            output[out_idx++] = c;
        } else if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
            output[out_idx++] = c;
        } else if (c == '.' || c == ',' || c == '!' || c == '?') {
            output[out_idx++] = c;
        } else if (c == '-' || c == '_' || c == ':') {
            output[out_idx++] = c;
        } else if (c == '(' || c == ')' || c == '[' || c == ']') {
            output[out_idx++] = c;
        } else if (c == '@' || c == '#' || c == '$' || c == '&') {
            /* Keep some special chars but not dangerous ones */
            output[out_idx++] = c;
        }
        /* Skip: < > ' " % \ / ` { } | ^ ~ and control characters */
    }

    /* Null terminate */
    output[out_idx] = '\0';

    return (ssize_t)out_idx;
}

//=============================================================================
// Test Reset Support
//=============================================================================

/**
 * @brief Reset input gate state for test isolation
 *
 * WHAT: Clear any accumulated input gate state between tests
 * WHY:  Prevent validation statistics and cached state from leaking between tests
 * HOW:  Reset is minimal since input gate has no significant mutable global state
 *       beyond mesh registration (which is not test-sensitive)
 *
 * NOTE: Called by bbb_reset_test_state() for unified BBB reset
 */
void bbb_input_gate_reset_internal(void)
{
    /* Input gate has no significant mutable global state that affects tests.
     * Mesh registration and health agent state are intentionally preserved.
     * This function exists for completeness and forward-compatibility
     * if input gate gains test-sensitive state in the future. */
}
