/**
 * @file nimcp_error_codes.h
 * @brief Comprehensive error code system for NIMCP
 *
 * WHAT: Structured error codes replacing bool returns
 * WHY:  Better diagnostics, error propagation, and debugging
 * HOW:  Enum-based error codes with category grouping
 *
 * ERROR CODE RANGES:
 * 0000-0999: Success codes
 * 1000-1999: Generic errors
 * 2000-2999: Memory errors
 * 3000-3999: Brain/Network errors
 * 4000-4999: I/O errors
 * 5000-5999: Configuration errors
 * 6000-6999: Threading/Concurrency errors
 * 7000-7999: Signal/Crash errors
 * 8000-8999: Phase 10 cognitive errors
 * 9000-9999: Reserved for future use
 *
 * @author NIMCP Team
 * @date 2025-11-09
 */

#ifndef NIMCP_ERROR_CODES_H
#define NIMCP_ERROR_CODES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Error Code Type
//=============================================================================

typedef int32_t nimcp_error_t;

//=============================================================================
// Success Codes (0-999)
//=============================================================================

#define NIMCP_SUCCESS                   0    /**< Operation successful */
#define NIMCP_SUCCESS_WITH_WARNINGS     1    /**< Success but with warnings */
#define NIMCP_SUCCESS_PARTIAL           2    /**< Partial success */

//=============================================================================
// Generic Errors (1000-1999)
//=============================================================================

#define NIMCP_ERROR_UNKNOWN             1000 /**< Unknown error */
#define NIMCP_ERROR_NOT_IMPLEMENTED     1001 /**< Feature not implemented */
#define NIMCP_ERROR_INVALID_PARAMETER   1002 /**< Invalid function parameter */
#define NIMCP_ERROR_NULL_POINTER        1003 /**< Unexpected NULL pointer */
#define NIMCP_ERROR_OUT_OF_RANGE        1004 /**< Value out of valid range */
#define NIMCP_ERROR_INVALID_STATE       1005 /**< Invalid object state */
#define NIMCP_ERROR_OPERATION_FAILED    1006 /**< Generic operation failure */
#define NIMCP_ERROR_NOT_INITIALIZED     1007 /**< Object not initialized */
#define NIMCP_ERROR_ALREADY_EXISTS      1008 /**< Resource already exists */
#define NIMCP_ERROR_NOT_FOUND           1009 /**< Resource not found */
#define NIMCP_ERROR_TIMEOUT             1010 /**< Operation timed out */
#define NIMCP_ERROR_CANCELLED           1011 /**< Operation cancelled */
#define NIMCP_ERROR_PERMISSION_DENIED   1012 /**< Permission denied */

//=============================================================================
// Memory Errors (2000-2999)
//=============================================================================

#define NIMCP_ERROR_NO_MEMORY           2000 /**< Memory allocation failed */
#define NIMCP_ERROR_BUFFER_TOO_SMALL    2001 /**< Buffer too small */
#define NIMCP_ERROR_BUFFER_OVERFLOW     2002 /**< Buffer overflow detected */
#define NIMCP_ERROR_MEMORY_CORRUPTION   2003 /**< Memory corruption detected */
#define NIMCP_ERROR_INVALID_ADDRESS     2004 /**< Invalid memory address */
#define NIMCP_ERROR_MEMORY_LEAK         2005 /**< Memory leak detected */
#define NIMCP_ERROR_DOUBLE_FREE         2006 /**< Double free detected */

//=============================================================================
// Brain/Network Errors (3000-3999)
//=============================================================================

#define NIMCP_ERROR_BRAIN_CREATION      3000 /**< Brain creation failed */
#define NIMCP_ERROR_BRAIN_INVALID       3001 /**< Invalid brain instance */
#define NIMCP_ERROR_NETWORK_CREATION    3002 /**< Neural network creation failed */
#define NIMCP_ERROR_NETWORK_INVALID     3003 /**< Invalid network structure */
#define NIMCP_ERROR_DIMENSION_MISMATCH  3004 /**< Dimension mismatch */
#define NIMCP_ERROR_WEIGHT_INIT         3005 /**< Weight initialization failed */
#define NIMCP_ERROR_FORWARD_PASS        3006 /**< Forward pass failed */
#define NIMCP_ERROR_BACKWARD_PASS       3007 /**< Backward pass failed */
#define NIMCP_ERROR_LEARNING_FAILED     3008 /**< Learning step failed */
#define NIMCP_ERROR_INFERENCE_FAILED    3009 /**< Inference failed */
#define NIMCP_ERROR_COW_FAILED          3010 /**< Copy-on-write operation failed */
#define NIMCP_ERROR_CLONE_FAILED        3011 /**< Brain clone failed */

//=============================================================================
// I/O Errors (4000-4999)
//=============================================================================

#define NIMCP_ERROR_FILE_NOT_FOUND      4000 /**< File not found */
#define NIMCP_ERROR_FILE_READ           4001 /**< File read error */
#define NIMCP_ERROR_FILE_WRITE          4002 /**< File write error */
#define NIMCP_ERROR_FILE_OPEN           4003 /**< File open error */
#define NIMCP_ERROR_FILE_CLOSE          4004 /**< File close error */
#define NIMCP_ERROR_FILE_CORRUPT        4005 /**< File corrupted */
#define NIMCP_ERROR_SERIALIZATION       4006 /**< Serialization failed */
#define NIMCP_ERROR_DESERIALIZATION     4007 /**< Deserialization failed */
#define NIMCP_ERROR_NETWORK_IO          4008 /**< Network I/O error */
#define NIMCP_ERROR_SOCKET_ERROR        4009 /**< Socket operation failed */

//=============================================================================
// Configuration Errors (5000-5999)
//=============================================================================

#define NIMCP_ERROR_CONFIG_INVALID      5000 /**< Invalid configuration */
#define NIMCP_ERROR_CONFIG_PARSE        5001 /**< Config parse error */
#define NIMCP_ERROR_CONFIG_MISSING      5002 /**< Required config missing */
#define NIMCP_ERROR_CONFIG_TYPE         5003 /**< Config type mismatch */
#define NIMCP_ERROR_CONFIG_RANGE        5004 /**< Config value out of range */
#define NIMCP_ERROR_CONFIG_RELOAD       5005 /**< Config reload failed */

//=============================================================================
// Threading/Concurrency Errors (6000-6999)
//=============================================================================

#define NIMCP_ERROR_THREAD_CREATE       6000 /**< Thread creation failed */
#define NIMCP_ERROR_THREAD_JOIN         6001 /**< Thread join failed */
#define NIMCP_ERROR_MUTEX_LOCK          6002 /**< Mutex lock failed */
#define NIMCP_ERROR_MUTEX_UNLOCK        6003 /**< Mutex unlock failed */
#define NIMCP_ERROR_MUTEX_INIT          6004 /**< Mutex init failed */
#define NIMCP_ERROR_DEADLOCK            6005 /**< Deadlock detected */
#define NIMCP_ERROR_RACE_CONDITION      6006 /**< Race condition detected */
#define NIMCP_ERROR_THREAD_SYNC         6007 /**< Thread synchronization failed */

//=============================================================================
// Signal/Crash Errors (7000-7999)
//=============================================================================

#define NIMCP_ERROR_SIGNAL_RECEIVED     7000 /**< Signal received */
#define NIMCP_ERROR_SIGSEGV             7001 /**< Segmentation fault */
#define NIMCP_ERROR_SIGABRT             7002 /**< Abort signal */
#define NIMCP_ERROR_SIGFPE              7003 /**< Floating point exception */
#define NIMCP_ERROR_SIGBUS              7004 /**< Bus error */
#define NIMCP_ERROR_SIGILL              7005 /**< Illegal instruction */
#define NIMCP_ERROR_CRASH_RECOVERY      7006 /**< Crash recovery failed */
#define NIMCP_ERROR_CHECKPOINT_SAVE     7007 /**< Checkpoint save failed */
#define NIMCP_ERROR_CHECKPOINT_LOAD     7008 /**< Checkpoint load failed */

//=============================================================================
// Phase 10 Cognitive Errors (8000-8999)
//=============================================================================

#define NIMCP_ERROR_WORKING_MEMORY      8000 /**< Working memory error */
#define NIMCP_ERROR_EMOTIONAL_TAGGING   8001 /**< Emotional tagging error */
#define NIMCP_ERROR_EXECUTIVE_CONTROL   8002 /**< Executive control error */
#define NIMCP_ERROR_SLEEP_WAKE          8003 /**< Sleep/wake cycle error */
#define NIMCP_ERROR_MENTAL_HEALTH       8004 /**< Mental health monitor error */
#define NIMCP_ERROR_THEORY_OF_MIND      8005 /**< Theory of mind error */
#define NIMCP_ERROR_EXPLANATIONS        8006 /**< Natural explanations error */
#define NIMCP_ERROR_META_LEARNING       8007 /**< Meta-learning error */
#define NIMCP_ERROR_PREDICTIVE          8008 /**< Predictive processing error */

//=============================================================================
// Error Information Structure
//=============================================================================

/**
 * @brief Detailed error information
 */
typedef struct {
    nimcp_error_t code;        /**< Error code */
    const char* message;       /**< Human-readable message */
    const char* file;          /**< Source file where error occurred */
    int line;                  /**< Line number where error occurred */
    const char* function;      /**< Function where error occurred */
    void* context;             /**< Optional context data */
} nimcp_error_info_t;

//=============================================================================
// Error Handling API
//=============================================================================

/**
 * @brief Check if error code indicates success
 */
static inline bool nimcp_error_is_success(nimcp_error_t code)
{
    return (code >= 0 && code < 1000);
}

/**
 * @brief Check if error code indicates failure
 */
static inline bool nimcp_error_is_failure(nimcp_error_t code)
{
    return (code >= 1000);
}

/**
 * @brief Get error category from error code
 */
static inline int nimcp_error_get_category(nimcp_error_t code)
{
    return (code / 1000);
}

/**
 * @brief Get human-readable error message
 *
 * @param code Error code
 * @return Error message string
 */
const char* nimcp_error_to_string(nimcp_error_t code);

/**
 * @brief Get error category name
 *
 * @param code Error code
 * @return Category name (e.g., "Memory Error")
 */
const char* nimcp_error_get_category_name(nimcp_error_t code);

/**
 * @brief Set last error (thread-local)
 *
 * @param code Error code
 * @param file Source file (use __FILE__)
 * @param line Line number (use __LINE__)
 * @param function Function name (use __func__)
 * @param message Optional message (can be NULL)
 */
void nimcp_error_set(nimcp_error_t code, const char* file, int line,
                     const char* function, const char* message);

/**
 * @brief Get last error info (thread-local)
 *
 * @return Pointer to error info (valid until next error_set)
 */
const nimcp_error_info_t* nimcp_error_get_last(void);

/**
 * @brief Clear last error
 */
void nimcp_error_clear(void);

/**
 * @brief Print error to stderr
 *
 * @param code Error code
 */
void nimcp_error_print(nimcp_error_t code);

/**
 * @brief Print detailed error info to stderr
 *
 * @param info Error information
 */
void nimcp_error_print_detailed(const nimcp_error_info_t* info);

//=============================================================================
// Convenience Macros
//=============================================================================

/**
 * @brief Set error with file/line/function info
 */
#define NIMCP_SET_ERROR(code, msg) \
    nimcp_error_set((code), __FILE__, __LINE__, __func__, (msg))

/**
 * @brief Return error with auto-set
 */
#define NIMCP_RETURN_ERROR(code, msg) \
    do { \
        NIMCP_SET_ERROR((code), (msg)); \
        return (code); \
    } while (0)

/**
 * @brief Check condition and return error if false
 */
#define NIMCP_CHECK(cond, code, msg) \
    do { \
        if (!(cond)) { \
            NIMCP_RETURN_ERROR((code), (msg)); \
        } \
    } while (0)

/**
 * @brief Propagate error if not success
 */
#define NIMCP_PROPAGATE_ERROR(expr) \
    do { \
        nimcp_error_t _err = (expr); \
        if (nimcp_error_is_failure(_err)) { \
            return _err; \
        } \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif // NIMCP_ERROR_CODES_H
