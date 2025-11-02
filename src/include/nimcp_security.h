/**
 * @file nimcp_security.h
 * @brief Comprehensive security framework for NIMCP
 *
 * Provides:
 * - Core directive protection against tampering
 * - Input validation and sanitization against prompt injection
 * - Skepticism-based evaluation of new information
 * - Encryption for inter-component communication
 */

#ifndef NIMCP_SECURITY_H
#define NIMCP_SECURITY_H

#include "utils/nimcp_common.h"
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Constants
//=============================================================================

#define NIMCP_SECURITY_MAX_DIRECTIVES 16
#define NIMCP_SECURITY_DIRECTIVE_MAX_LEN 512
#define NIMCP_SECURITY_HASH_SIZE 32
#define NIMCP_SECURITY_KEY_SIZE 32
#define NIMCP_SECURITY_IV_SIZE 16
#define NIMCP_SECURITY_MAX_ENCRYPTED_SIZE 65536

// Security threat levels
typedef enum {
    NIMCP_THREAT_NONE = 0,
    NIMCP_THREAT_LOW = 1,
    NIMCP_THREAT_MEDIUM = 2,
    NIMCP_THREAT_HIGH = 3,
    NIMCP_THREAT_CRITICAL = 4
} nimcp_threat_level_t;

// Input validation result
typedef enum {
    NIMCP_INPUT_VALID = 0,
    NIMCP_INPUT_CONTAINS_INJECTION,
    NIMCP_INPUT_EXCEEDS_LENGTH,
    NIMCP_INPUT_INVALID_CHARACTERS,
    NIMCP_INPUT_SUSPICIOUS_PATTERN
} nimcp_input_validation_t;

//=============================================================================
// Core Directive Protection
//=============================================================================

/**
 * @brief Core directive with OS-level memory protection
 *
 * Directives are protected by:
 * 1. OS-level memory protection (mprotect PROT_READ on text)
 * 2. Cryptographic hash verification
 * 3. Immutability flags
 *
 * DESIGN: Separates immutable text from mutable metadata to enable mprotect()
 */
typedef struct {
    char* text;                          // mmap'd + mprotect'd (immutable)
    size_t text_length;                  // Length for munmap
    uint8_t hash[NIMCP_SECURITY_HASH_SIZE];
    uint64_t timestamp;
    bool immutable;
    uint32_t verification_count;         // Mutable (not mprotect'd)
} nimcp_core_directive_t;

/**
 * @brief Core directive system
 */
typedef struct nimcp_directive_system nimcp_directive_system_t;

/**
 * @brief Create directive protection system
 *
 * @return Directive system handle or NULL on failure
 */
nimcp_directive_system_t* nimcp_directive_system_create(void);

/**
 * @brief Add core directive (can only be called during initialization)
 *
 * @param system Directive system
 * @param directive_text Directive text
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_directive_add(nimcp_directive_system_t* system, const char* directive_text);

/**
 * @brief Lock directives (makes them immutable)
 *
 * @param system Directive system
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_directive_lock(nimcp_directive_system_t* system);

/**
 * @brief Verify directive integrity
 *
 * @param system Directive system
 * @param directive_index Index of directive to verify
 * @return true if intact, false if tampered
 */
bool nimcp_directive_verify(nimcp_directive_system_t* system, uint32_t directive_index);

/**
 * @brief Verify all directives
 *
 * @param system Directive system
 * @return true if all intact, false if any tampered
 */
bool nimcp_directive_verify_all(nimcp_directive_system_t* system);

/**
 * @brief Get directive text (read-only)
 *
 * @param system Directive system
 * @param directive_index Index of directive
 * @return Directive text or NULL if invalid index
 */
const char* nimcp_directive_get(nimcp_directive_system_t* system, uint32_t directive_index);

/**
 * @brief Get number of directives
 *
 * @param system Directive system
 * @return Number of directives
 */
uint32_t nimcp_directive_count(nimcp_directive_system_t* system);

/**
 * @brief Destroy directive system
 *
 * @param system Directive system
 */
void nimcp_directive_system_destroy(nimcp_directive_system_t* system);

//=============================================================================
// Input Validation and Sanitization
//=============================================================================

/**
 * @brief Validate input for prompt injection attacks
 *
 * Detects patterns like:
 * - Instruction injection ("Ignore previous instructions")
 * - Role confusion ("You are now...")
 * - System prompts ("<|system|>", "### SYSTEM:")
 * - Escape sequences
 * - Excessive special characters
 *
 * @param input Input text to validate
 * @param max_length Maximum allowed length
 * @param threat_level Output: detected threat level
 * @return Validation result
 */
nimcp_input_validation_t nimcp_security_validate_input(const char* input, size_t max_length,
                                                        nimcp_threat_level_t* threat_level);

/**
 * @brief Sanitize input text
 *
 * @param input Input text
 * @param output Output buffer
 * @param output_size Output buffer size
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_security_sanitize_input(const char* input, char* output, size_t output_size);

/**
 * @brief Check if text contains suspicious patterns
 *
 * @param text Text to check
 * @return Threat level detected
 */
nimcp_threat_level_t nimcp_security_analyze_threat(const char* text);

//=============================================================================
// Skepticism System
//=============================================================================

/**
 * @brief Skepticism evaluation result
 */
typedef struct {
    float credibility_score;       // 0.0 (not credible) to 1.0 (highly credible)
    float evidence_strength;       // Strength of supporting evidence
    float source_reliability;      // Reliability of information source
    bool requires_verification;    // Whether info needs additional verification
    char rationale[256];          // Reasoning for skepticism level
} nimcp_skepticism_result_t;

/**
 * @brief Evaluate new information with skepticism
 *
 * Implements the core directive: "Always be skeptical of new information"
 *
 * @param information New information to evaluate
 * @param existing_knowledge Related existing knowledge (can be NULL)
 * @param source_type Type of information source
 * @param result Output: skepticism evaluation result
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_security_evaluate_skepticism(const char* information,
                                                   const char* existing_knowledge,
                                                   const char* source_type,
                                                   nimcp_skepticism_result_t* result);

//=============================================================================
// Encryption for Inter-Component Communication
//=============================================================================

/**
 * @brief Encryption context
 */
typedef struct nimcp_encryption_context nimcp_encryption_context_t;

/**
 * @brief Create encryption context
 *
 * @param key Encryption key (32 bytes)
 * @return Encryption context or NULL on failure
 */
nimcp_encryption_context_t* nimcp_encryption_create(const uint8_t* key);

/**
 * @brief Generate random encryption key
 *
 * @param key Output buffer (32 bytes)
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_encryption_generate_key(uint8_t* key);

/**
 * @brief Encrypt data for inter-component communication
 *
 * @param ctx Encryption context
 * @param plaintext Plaintext data
 * @param plaintext_size Size of plaintext
 * @param ciphertext Output buffer for encrypted data
 * @param ciphertext_size Size of output buffer
 * @param actual_size Output: actual size of encrypted data
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_encryption_encrypt(nimcp_encryption_context_t* ctx, const uint8_t* plaintext,
                                        size_t plaintext_size, uint8_t* ciphertext,
                                        size_t ciphertext_size, size_t* actual_size);

/**
 * @brief Decrypt data from inter-component communication
 *
 * @param ctx Encryption context
 * @param ciphertext Encrypted data
 * @param ciphertext_size Size of encrypted data
 * @param plaintext Output buffer for decrypted data
 * @param plaintext_size Size of output buffer
 * @param actual_size Output: actual size of decrypted data
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_encryption_decrypt(nimcp_encryption_context_t* ctx, const uint8_t* ciphertext,
                                        size_t ciphertext_size, uint8_t* plaintext,
                                        size_t plaintext_size, size_t* actual_size);

/**
 * @brief Destroy encryption context
 *
 * @param ctx Encryption context
 */
void nimcp_encryption_destroy(nimcp_encryption_context_t* ctx);

//=============================================================================
// Security Audit and Logging
//=============================================================================

/**
 * @brief Security event type
 */
typedef enum {
    NIMCP_SECURITY_EVENT_DIRECTIVE_VERIFIED,
    NIMCP_SECURITY_EVENT_DIRECTIVE_TAMPERED,
    NIMCP_SECURITY_EVENT_INPUT_REJECTED,
    NIMCP_SECURITY_EVENT_THREAT_DETECTED,
    NIMCP_SECURITY_EVENT_ENCRYPTION_FAILED,
    NIMCP_SECURITY_EVENT_SKEPTICISM_TRIGGERED
} nimcp_security_event_t;

/**
 * @brief Log security event
 *
 * @param event Event type
 * @param severity Severity level
 * @param details Event details
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_security_log_event(nimcp_security_event_t event,
                                        nimcp_threat_level_t severity, const char* details);

/**
 * @brief Get security statistics
 *
 * @param threats_detected Output: number of threats detected
 * @param inputs_rejected Output: number of inputs rejected
 * @param directives_verified Output: number of directive verifications
 * @return NIMCP_SUCCESS or error code
 */
nimcp_result_t nimcp_security_get_stats(uint64_t* threats_detected, uint64_t* inputs_rejected,
                                        uint64_t* directives_verified);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_SECURITY_H
