/**
 * @file nimcp_common.h
 * @brief NIMCP common definitions and utilities
 *
 * Provides common definitions, types, macros, and utility functions used
 * throughout the NIMCP protocol implementation. Includes error handling,
 * logging, memory management, thread synchronization, and network utilities.
 *
 * @version 1.0.0
 * @date 2024
 */

#ifndef NIMCP_COMMON_H
#define NIMCP_COMMON_H


#include <ctype.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
    #include <stdatomic.h>
#endif
#include <math.h>
#include <string.h>
#include <time.h>

#ifdef __linux__
    #include <dirent.h>
    #include <dlfcn.h>
    #include <lz4.h>
#elif defined(_WIN32)
    #include <direct.h>
    #include <lz4.h>
    #include <windows.h>
#endif

typedef struct timespec nimcp_timespec_t;

/* Message Priority Levels */
typedef enum NimcpMessagePriority {
    NIMCP_PRIORITY_LOW = 0,
    NIMCP_PRIORITY_NORMAL = 1,
    NIMCP_PRIORITY_HIGH = 2,
    NIMCP_PRIORITY_URGENT = 3
} NimcpMessagePriority;

/* Result/Error Codes */
typedef int32_t nimcp_result_t;
/* Message type definition */
typedef struct nimcp_message {
    void* data;     /* Message payload */
    size_t size;    /* Size of payload */
    uint32_t type;  /* Message type */
    uint32_t flags; /* Message flags */
} nimcp_message_t;


/* Result/Error Codes */
typedef int32_t nimcp_result_t;

/* Protocol-Specific Errors (-40 to -49) */
#ifndef NIMCP_ERROR_INVALID_PACKET
#define NIMCP_ERROR_INVALID_PACKET -40   /* Invalid packet structure or format */
#endif
#ifndef NIMCP_ERROR_INVALID_HEADER
#define NIMCP_ERROR_INVALID_HEADER -41   /* Invalid packet header */
#endif
#ifndef NIMCP_ERROR_INVALID_VERSION
#define NIMCP_ERROR_INVALID_VERSION -43  /* Unsupported protocol version */
#endif
#ifndef NIMCP_ERROR_INVALID_TYPE
#define NIMCP_ERROR_INVALID_TYPE -44     /* Invalid message type */
#endif
#ifndef NIMCP_ERROR_INVALID_FLAGS
#define NIMCP_ERROR_INVALID_FLAGS -45    /* Invalid packet flags */
#endif
#ifndef NIMCP_ERROR_INVALID_SEQUENCE
#define NIMCP_ERROR_INVALID_SEQUENCE -46 /* Invalid sequence number */
#endif
#ifndef NIMCP_ERROR_INVALID_SIZE
#define NIMCP_ERROR_INVALID_SIZE -47     /* Invalid payload size */
#endif
#ifndef NIMCP_ERROR_SERIALIZATION
#define NIMCP_ERROR_SERIALIZATION -48    /* Serialization error */
#endif
#ifndef NIMCP_ERROR_DESERIALIZATION
#define NIMCP_ERROR_DESERIALIZATION -49  /* Deserialization error */
#endif

/* Memory and Buffer Errors (-50 to -59) */
#ifndef NIMCP_ERROR_MEMORY
#define NIMCP_ERROR_MEMORY -50           /* Memory allocation error */
#endif
#ifndef NIMCP_ERROR_BUFFER_OVERFLOW
#define NIMCP_ERROR_BUFFER_OVERFLOW -51  /* Buffer overflow error */
#endif
#ifndef NIMCP_ERROR_BUFFER_UNDERFLOW
#define NIMCP_ERROR_BUFFER_UNDERFLOW -52 /* Buffer underflow error */
#endif
#ifndef NIMCP_ERROR_BUFFER_TOO_SMALL
#define NIMCP_ERROR_BUFFER_TOO_SMALL -53 /* Buffer too small for operation */
#endif
#ifndef NIMCP_ERROR_NULL_POINTER
#define NIMCP_ERROR_NULL_POINTER -54     /* Null pointer error */
#endif

#define MAX_ROUTES 256

/**
 * @defgroup error_codes Error Codes
 * @{
 */

/* NIMCP uses canonical error codes from nimcp_error_codes.h (values 1000+)
 * The NIMCP_ERROR_INVALID_PARAM alias maps to NIMCP_ERROR_INVALID_PARAMETER (1002)
 * for consistency across the codebase. */
#ifndef NIMCP_SUCCESS
#define NIMCP_SUCCESS 0                     /**< Operation completed successfully */
#endif
#ifndef NIMCP_ERROR_INVALID_PARAM
#define NIMCP_ERROR_INVALID_PARAM 1002      /**< Invalid parameter - maps to NIMCP_ERROR_INVALID_PARAMETER */
#endif
#ifndef NIMCP_ERROR_INVALID_MAGIC
#define NIMCP_ERROR_INVALID_MAGIC -2        /**< Invalid magic number in packet */
#endif
#ifndef NIMCP_ERROR_VERSION_MISMATCH
#define NIMCP_ERROR_VERSION_MISMATCH -3     /**< Protocol version mismatch */
#endif
#ifndef NIMCP_ERROR_PAYLOAD_TOO_LARGE
#define NIMCP_ERROR_PAYLOAD_TOO_LARGE -4    /**< Payload exceeds maximum size */
#endif
#ifndef NIMCP_ERROR_INVALID_SIGNATURE
#define NIMCP_ERROR_INVALID_SIGNATURE -5    /**< Invalid or corrupt signature */
#endif
#ifndef NIMCP_ERROR_SECURITY_REQUIRED
#define NIMCP_ERROR_SECURITY_REQUIRED -6    /**< Security context required but not provided */
#endif
#ifndef NIMCP_ERROR_ENCRYPTION_FAILED
#define NIMCP_ERROR_ENCRYPTION_FAILED -7    /**< Encryption operation failed */
#endif
#ifndef NIMCP_ERROR_DECRYPTION_FAILED
#define NIMCP_ERROR_DECRYPTION_FAILED -8    /**< Decryption operation failed */
#endif
#ifndef NIMCP_ERROR_SIGNATURE_FAILED
#define NIMCP_ERROR_SIGNATURE_FAILED -9     /**< Signature generation failed */
#endif
#ifndef NIMCP_ERROR_VERIFICATION_FAILED
#define NIMCP_ERROR_VERIFICATION_FAILED -10 /**< Signature verification failed */
#endif
#ifndef NIMCP_ERROR_SERIALIZER
#define NIMCP_ERROR_SERIALIZER -11          /**< Serializer operation failed */
#endif
#ifndef NIMCP_ERROR_NOT_IMPLEMENTED
#define NIMCP_ERROR_NOT_IMPLEMENTED -13     /**< Feature not implemented */
#endif
/* Version Information - Use nimcp.h version if already defined */
#ifndef NIMCP_VERSION_MAJOR
#define NIMCP_VERSION_MAJOR 2
#endif
#ifndef NIMCP_VERSION_MINOR
#define NIMCP_VERSION_MINOR 6
#endif
#ifndef NIMCP_VERSION_PATCH
#define NIMCP_VERSION_PATCH 1
#endif
#ifndef NIMCP_VERSION_STRING
#define NIMCP_VERSION_STRING "2.6.1"
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define NIMCP_ALIGN(x, a) (((x) + ((a) -1)) & ~((a) -1))

/* Platform Detection - Use nimcp_platform.h if already defined */
#ifndef NIMCP_PLATFORM_LINUX
#if defined(_WIN32) || defined(_WIN64)
    #define NIMCP_PLATFORM_WINDOWS
#elif defined(__linux__)
    #define NIMCP_PLATFORM_LINUX
#elif defined(__APPLE__)
    #define NIMCP_PLATFORM_MACOS
#else
    #error "Unsupported platform"
#endif
#endif


/* Success Codes (>= 0) - with guards to prevent redefinition */
#ifndef NIMCP_PENDING
#define NIMCP_PENDING 1        /* Operation is in progress */
#endif
#ifndef NIMCP_WOULD_BLOCK
#define NIMCP_WOULD_BLOCK 2    /* Non-blocking operation would block */
#endif
#ifndef NIMCP_TIMEOUT
#define NIMCP_TIMEOUT 3        /* Operation timed out */
#endif
#ifndef NIMCP_NOT_FOUND
#define NIMCP_NOT_FOUND 4      /* Requested item not found */
#endif
#ifndef NIMCP_BUFFER_FULL
#define NIMCP_BUFFER_FULL 5    /* Buffer capacity reached */
#endif
#ifndef NIMCP_END_OF_STREAM
#define NIMCP_END_OF_STREAM 6  /* End of stream reached */
#endif
#ifndef NIMCP_ALREADY_EXISTS
#define NIMCP_ALREADY_EXISTS 7 /* Item already exists */
#endif

/* Error Codes (< 0) */
/* General Errors (-1 to -9) */
#ifndef NIMCP_ERROR
#define NIMCP_ERROR -1              /* Generic error */
#endif
#ifndef NIMCP_INVALID_PARAM
#define NIMCP_INVALID_PARAM -2      /* Invalid parameter */
#endif
#ifndef NIMCP_ERROR_NULL_ARG
#define NIMCP_ERROR_NULL_ARG -2     /* Null argument provided */
#endif
#ifndef NIMCP_ERROR_INVALID
#define NIMCP_ERROR_INVALID -3      /* Invalid argument value (distinct from NULL_ARG) */
#endif
#ifndef NIMCP_NO_MEMORY
#define NIMCP_NO_MEMORY -3          /* Memory allocation failed */
#endif
#ifndef NIMCP_NOT_INITIALIZED
#define NIMCP_NOT_INITIALIZED -4    /* Component not initialized */
#endif
#ifndef NIMCP_NOT_IMPLEMENTED
#define NIMCP_NOT_IMPLEMENTED -5    /* Feature not implemented */
#endif
#ifndef NIMCP_INVALID_STATE
#define NIMCP_INVALID_STATE -6      /* Invalid state for operation */
#endif
#ifndef NIMCP_BUFFER_TOO_SMALL
#define NIMCP_BUFFER_TOO_SMALL -7   /* Buffer too small */
#endif
#ifndef NIMCP_OPERATION_CANCELED
#define NIMCP_OPERATION_CANCELED -8 /* Operation was canceled */
#endif
#ifndef NIMCP_PERMISSION_DENIED
#define NIMCP_PERMISSION_DENIED -9  /* Permission denied */
#endif

/* System Errors (-10 to -19) */
#ifndef NIMCP_SYSTEM_ERROR
#define NIMCP_SYSTEM_ERROR -10      /* System call failed */
#endif
#ifndef NIMCP_IO_ERROR
#define NIMCP_IO_ERROR -11          /* I/O error */
#endif
#ifndef NIMCP_NETWORK_ERROR
#define NIMCP_NETWORK_ERROR -12     /* Network operation failed */
#endif
#ifndef NIMCP_SOCKET_ERROR
#define NIMCP_SOCKET_ERROR -13      /* Socket operation failed */
#endif
#ifndef NIMCP_THREAD_ERROR
#define NIMCP_THREAD_ERROR -14      /* Thread operation failed */
#endif
#ifndef NIMCP_LOCK_ERROR
#define NIMCP_LOCK_ERROR -15        /* Lock operation failed */
#endif
#ifndef NIMCP_TIMEOUT_ERROR
#define NIMCP_TIMEOUT_ERROR -16     /* Timeout error */
#endif
#ifndef NIMCP_ERROR_NOT_FOUND
#define NIMCP_ERROR_NOT_FOUND -17   /* Item not found error */
#endif
#ifndef NIMCP_ERROR_NOT_SUPPORTED
#define NIMCP_ERROR_NOT_SUPPORTED -18 /* Feature not supported */
#endif

/* Protocol Errors (-20 to -29) */
#ifndef NIMCP_INVALID_MSG
#define NIMCP_INVALID_MSG -20      /* Invalid message format */
#endif
#ifndef NIMCP_VERSION_MISMATCH
#define NIMCP_VERSION_MISMATCH -21 /* Protocol version mismatch */
#endif
#ifndef NIMCP_QUEUE_FULL
#define NIMCP_QUEUE_FULL -22       /* Message queue is full */
#endif
#ifndef NIMCP_QUEUE_EMPTY
#define NIMCP_QUEUE_EMPTY -23      /* Message queue is empty */
#endif
#ifndef NIMCP_PROTO_ERROR
#define NIMCP_PROTO_ERROR -24      /* Protocol error */
#endif
#ifndef NIMCP_HANDSHAKE_FAILED
#define NIMCP_HANDSHAKE_FAILED -25 /* Protocol handshake failed */
#endif
#ifndef NIMCP_AUTH_FAILED
#define NIMCP_AUTH_FAILED -26      /* Authentication failed */
#endif
#ifndef NIMCP_INVALID_SEQUENCE
#define NIMCP_INVALID_SEQUENCE -27 /* Invalid message sequence */
#endif
#ifndef NIMCP_CRYPTO_ERROR
#define NIMCP_CRYPTO_ERROR -28     /* Cryptographic operation failed */
#endif
#ifndef NIMCP_INVALID_CHECKSUM
#define NIMCP_INVALID_CHECKSUM -29 /* Invalid message checksum */
#endif

/* Initialization Errors (-30 to -39) */
#ifndef NIMCP_INIT_FAILED
#define NIMCP_INIT_FAILED -30     /* Initialization failed */
#endif
#ifndef NIMCP_CONFIG_ERROR
#define NIMCP_CONFIG_ERROR -31    /* Configuration error */
#endif
#ifndef NIMCP_ALREADY_RUNNING
#define NIMCP_ALREADY_RUNNING -32 /* Already running */
#endif
#ifndef NIMCP_NOT_RUNNING
#define NIMCP_NOT_RUNNING -33     /* Not running */
#endif

// Add necessary constants
#ifndef MAX_MESSAGE_SIZE
#define MAX_MESSAGE_SIZE (1024 * 1024)  // 1MB
#endif
#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1000
#endif
#ifndef MAX_PENDING_MESSAGES
#define MAX_PENDING_MESSAGES 10000
#endif
#ifndef MESSAGE_QUEUE_TIMEOUT_MS
#define MESSAGE_QUEUE_TIMEOUT_MS 1000
#endif

// Add error codes
#ifndef NIMCP_ERROR_QUEUE_FULL
#define NIMCP_ERROR_QUEUE_FULL -100
#endif
#ifndef NIMCP_ERROR_MESSAGE_TOO_LARGE
#define NIMCP_ERROR_MESSAGE_TOO_LARGE -101
#endif
#ifndef NIMCP_ERROR_INVALID_MESSAGE
#define NIMCP_ERROR_INVALID_MESSAGE -102
#endif
#ifndef NIMCP_ERROR_TIMEOUT
#define NIMCP_ERROR_TIMEOUT -103
#endif

/* Common Type Definitions */
typedef uint32_t nimcp_handle_t;
#define NIMCP_INVALID_HANDLE ((nimcp_handle_t) 0)

/* Time-related Definitions */
#define NIMCP_TIMEOUT_INFINITE ((uint32_t) -1)
#define NIMCP_TIMEOUT_IMMEDIATE ((uint32_t) 0)

/* Size-related Definitions */
#define NIMCP_MAX_ERROR_LENGTH 128
#define NIMCP_DEFAULT_BUFFER_SIZE 4096


/* Common Utility Functions */
/**
 * @brief Convert result code to string description
 * @param result Result code
 * @return String description
 */
const char* nimcp_result_to_string(nimcp_result_t result);

/**
 * @brief Get version string
 * @return Version string
 */
const char* nimcp_version_string(void);

/**
 * @brief Sleep for specified milliseconds
 * @param milliseconds Sleep duration
 */
void nimcp_sleep(uint32_t milliseconds);

/* Error Handling */
/**
 * @brief Error information structure
 * Conditionally defined to avoid conflict with nimcp_error_codes.h
 */
#ifndef NIMCP_ERROR_INFO_T_DEFINED
#define NIMCP_ERROR_INFO_T_DEFINED
typedef struct nimcp_error_info_struct {
    nimcp_result_t code;
    char message[NIMCP_MAX_ERROR_LENGTH];
} nimcp_error_info_t;
#endif

/**
 * @brief Get current error information
 * @param info Error info structure to fill
 */
void nimcp_get_error_info(nimcp_error_info_t* info);

/**
 * @brief Set error information
 * @param code Error code
 * @param format Error message format
 * @param ... Format arguments
 */
void nimcp_set_error_info(nimcp_result_t code, const char* format, ...);

/**
 * @brief Clear current error state
 */
void nimcp_clear_error(void);

/* Basic Logging Levels */
typedef enum {
    NIMCP_LOG_NONE = 0,
    NIMCP_LOG_ERROR,
    NIMCP_LOG_WARN,
    NIMCP_LOG_INFO,
    NIMCP_LOG_DEBUG,
    NIMCP_LOG_TRACE
} nimcp_log_level_t;

/* Logging Function Type */
typedef void (*nimcp_log_callback_t)(nimcp_log_level_t level, const char* file, int line,
                                     const char* func, const char* message);

/* Logging Configuration */
/**
 * @brief Set logging level
 * @param level Log level
 */
void nimcp_set_log_level(nimcp_log_level_t level);

/**
 * @brief Set logging callback
 * @param callback Log callback function
 */
void nimcp_set_log_callback(nimcp_log_callback_t callback);


/* Endian Conversion Utilities */
/**
 * @brief Convert host to network byte order (16-bit)
 */
uint16_t nimcp_htons(uint16_t value);

/**
 * @brief Convert host to network byte order (32-bit)
 */
uint32_t nimcp_htonl(uint32_t value);

/**
 * @brief Convert host to network byte order (64-bit)
 */
uint64_t nimcp_htonll(uint64_t value);

/**
 * @brief Convert network to host byte order (16-bit)
 */
uint16_t nimcp_ntohs(uint16_t value);

/**
 * @brief Convert network to host byte order (32-bit)
 */
uint32_t nimcp_ntohl(uint32_t value);

/**
 * @brief Convert network to host byte order (64-bit)
 */
uint64_t nimcp_ntohll(uint64_t value);

/* ============================================================================
 * State Validation Types and Constants
 * ============================================================================ */

/* Field type enumeration */
typedef enum {
    NIMCP_FIELD_INTEGER = 0,
    NIMCP_FIELD_FLOAT = 1,
    NIMCP_FIELD_STRING = 2,
    NIMCP_FIELD_ARRAY = 3
} NimcpFieldType;

/* Array element type enumeration */
typedef enum {
    NIMCP_ARRAY_INTEGER = 0,
    NIMCP_ARRAY_FLOAT = 1,
    NIMCP_ARRAY_STRING = 2
} NimcpArrayElementType;

/* State field descriptor */
typedef struct {
    NimcpFieldType type; /* Field type */
    uint32_t offset;     /* Offset from start of state data */
    uint32_t size;       /* Size of field in bytes */
} NimcpStateField;

/* State header */
typedef struct {
    uint32_t magic;       /* Magic number for validation */
    uint32_t field_count; /* Number of fields */
} NimcpStateHeader;

/* Array field header */
typedef struct {
    NimcpArrayElementType element_type; /* Type of array elements */
    uint32_t element_size;              /* Size of each element */
    uint32_t element_count;             /* Number of elements */
    uint32_t _padding;                  /* Padding to ensure 8-byte alignment of array data */
} NimcpArrayHeader;

/* State validation constants */
#define NIMCP_STATE_MAGIC 0x4E494D43 /* 'NIMC' */
#define NIMCP_MAX_FIELDS 1024
#define NIMCP_INT32_MIN (-2147483647 - 1)
#define NIMCP_INT32_MAX 2147483647
#define NIMCP_INT64_MIN (-9223372036854775807LL - 1)
#define NIMCP_INT64_MAX 9223372036854775807LL
#define NIMCP_FLOAT_MAX 3.40282347e+38F
#define NIMCP_DOUBLE_MAX 1.7976931348623157e+308
#define NIMCP_STRING_MAX_LENGTH 65536
#define NIMCP_ARRAY_MAX_ELEMENTS 100000
#define NIMCP_ARRAY_ALIGNMENT 8

/* ============================================================================
 * Validation Macros
 * ============================================================================ */

/**
 * @brief Validate that enum value is within valid range
 *
 * WHAT: Checks if enum value is non-negative and less than max count
 * WHY:  Prevent array out-of-bounds access when using enums as array indices
 * HOW:  Evaluates to true if value is in range [0, max), false otherwise
 *
 * @param val Enum value to validate
 * @param max Maximum valid value (exclusive, typically ENUM_COUNT)
 * @return true if valid, false otherwise
 *
 * EXAMPLE:
 *   if (!NIMCP_VALIDATE_ENUM(threat->type, THREAT_COUNT)) {
 *       LOG_ERROR("Invalid threat type: %d", threat->type);
 *       return NIMCP_INVALID_PARAM;
 *   }
 */
#define NIMCP_VALIDATE_ENUM(val, max) ((val) >= 0 && (val) < (max))

#endif /* NIMCP_COMMON_H */
