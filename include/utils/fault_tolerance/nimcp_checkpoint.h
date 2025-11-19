/**
 * @file nimcp_checkpoint.h
 * @brief Brain checkpoint and recovery system for fault tolerance
 *
 * WHAT: Save and restore brain state to/from disk with integrity checking
 * WHY:  Enable crash recovery, periodic backups, and state persistence
 * HOW:  Serialize brain weights/config/metadata with atomic writes and checksums
 *
 * FEATURES:
 * - Full checkpoint: Save complete brain state (weights, config, metadata)
 * - Incremental checkpoint: Save only changed data (faster, smaller)
 * - Atomic writes: Write to temp file, then rename (prevents corruption)
 * - Compression: Optional zlib compression (configurable)
 * - Integrity: CRC32 checksums for validation
 * - Thread-safe: Uses NIMCP thread primitives for concurrent access
 * - Auto-recovery: Automatic restoration on crash detection
 *
 * CHECKPOINT FORMAT:
 * ```
 * [Header: 64 bytes]
 *   - Magic: "NIMCP-CKP" (9 bytes)
 *   - Version: major.minor (2 bytes)
 *   - Flags: compression, incremental, etc. (4 bytes)
 *   - Timestamp: Unix timestamp (8 bytes)
 *   - CRC32: Checksum of entire file (4 bytes)
 *   - Reserved: (37 bytes)
 *
 * [Brain Metadata: variable]
 *   - Config: brain_config_t
 *   - Stats: brain_stats_t
 *   - Dimensions: inputs, outputs, neurons
 *
 * [Network Weights: variable]
 *   - Layer weights (compressed if enabled)
 *   - Biases
 *   - Activations (optional)
 *
 * [Subsystem State: variable] (optional)
 *   - Glial state
 *   - Working memory
 *   - Emotional state
 *   - etc.
 * ```
 *
 * USAGE:
 * ```c
 * // Create checkpoint on periodic timer
 * checkpoint_save(brain, "/var/lib/nimcp/checkpoints/brain_001.ckpt");
 *
 * // Restore from checkpoint
 * brain_t brain = NULL;
 * checkpoint_load(&brain, "/var/lib/nimcp/checkpoints/brain_001.ckpt");
 *
 * // Validate checkpoint integrity
 * if (!checkpoint_validate("/path/to/checkpoint.ckpt")) {
 *     // Corrupted, try older checkpoint
 * }
 *
 * // Auto-recovery after crash
 * recovery_auto_restore(&brain, "/var/lib/nimcp/checkpoints");
 * ```
 *
 * @author NIMCP Team
 * @date 2025-11-19
 */

#ifndef NIMCP_CHECKPOINT_H
#define NIMCP_CHECKPOINT_H

#include <stdint.h>
#include <stdbool.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Forward Declarations
//=============================================================================

typedef struct brain_struct* brain_t;

//=============================================================================
// Checkpoint Header Format
//=============================================================================

/**
 * @brief Checkpoint file header (64 bytes)
 *
 * WHAT: Fixed-size header at start of checkpoint file
 * WHY:  Enable quick validation and version detection
 * HOW:  Magic bytes + version + flags + checksum
 */
typedef struct {
    char magic[9];          /**< Magic: "NIMCP-CKP" (null-terminated) */
    uint8_t version_major;  /**< Major version (incompatible changes) */
    uint8_t version_minor;  /**< Minor version (compatible changes) */
    uint32_t flags;         /**< Format flags (compressed, incremental, etc.) */
    uint64_t timestamp;     /**< Unix timestamp (seconds since epoch) */
    uint32_t crc32;         /**< CRC32 checksum of entire file (excluding this field) */
    uint32_t data_size;     /**< Size of data section (bytes) */
    uint32_t reserved[7];   /**< Reserved for future use */
} checkpoint_header_t;

// Magic bytes for checkpoint files
#define CHECKPOINT_MAGIC "NIMCP-CKP"

// Current checkpoint format version
#define CHECKPOINT_VERSION_MAJOR 1
#define CHECKPOINT_VERSION_MINOR 0

// Checkpoint flags
#define CHECKPOINT_FLAG_COMPRESSED   0x00000001  /**< Data is zlib-compressed */
#define CHECKPOINT_FLAG_INCREMENTAL  0x00000002  /**< Incremental checkpoint (delta) */
#define CHECKPOINT_FLAG_ENCRYPTED    0x00000004  /**< Data is encrypted */
#define CHECKPOINT_FLAG_SUBSYSTEMS   0x00000008  /**< Includes subsystem state */

//=============================================================================
// Checkpoint Information
//=============================================================================

/**
 * @brief Checkpoint metadata
 *
 * WHAT: Information about a saved checkpoint
 * WHY:  List/compare/select checkpoints for recovery
 * HOW:  Extracted from checkpoint header
 */
typedef struct {
    char path[256];         /**< Full path to checkpoint file */
    uint64_t timestamp;     /**< When checkpoint was created */
    uint32_t size_bytes;    /**< File size in bytes */
    bool is_valid;          /**< Passed integrity check */
    bool is_compressed;     /**< Is compressed */
    bool is_incremental;    /**< Is incremental */
    uint8_t version_major;  /**< Format version */
    uint8_t version_minor;  /**< Format version */
} checkpoint_info_t;

//=============================================================================
// Checkpoint Configuration
//=============================================================================

/**
 * @brief Checkpoint save options
 *
 * WHAT: Configuration for checkpoint creation
 * WHY:  Control compression, incremental saves, subsystems
 * HOW:  Pass to checkpoint_save_ex()
 */
typedef struct {
    bool enable_compression;    /**< Enable zlib compression (slower, smaller) */
    bool incremental;           /**< Save only changed data */
    bool save_subsystems;       /**< Include subsystem state (glial, emotional, etc.) */
    bool save_activations;      /**< Include current activations */
    int compression_level;      /**< Zlib compression level (0-9, default: 6) */
    const char* temp_dir;       /**< Directory for temp files (NULL = same as output) */
} checkpoint_options_t;

//=============================================================================
// Checkpoint Creation
//=============================================================================

/**
 * @brief Save brain checkpoint with default options
 *
 * WHAT: Serialize brain state to disk with atomic write
 * WHY:  Create recovery point for crash recovery
 * HOW:  Write to temp file, verify, rename (atomic on POSIX)
 *
 * DEFAULT OPTIONS:
 * - Compression: enabled (level 6)
 * - Incremental: disabled (full checkpoint)
 * - Subsystems: enabled (save all subsystem state)
 * - Activations: disabled (save weights only)
 *
 * THREAD-SAFE: Yes (acquires brain mutex)
 *
 * ERROR HANDLING:
 * - Returns false on I/O error, allocation failure, or invalid brain
 * - Original file unchanged on error (atomic write guarantees)
 * - Temp file cleaned up on error
 *
 * @param brain Brain instance to checkpoint
 * @param path Output path for checkpoint file
 * @return true on success, false on failure
 */
bool checkpoint_save(brain_t brain, const char* path);

/**
 * @brief Save brain checkpoint with custom options
 *
 * WHAT: Serialize brain state with configurable compression/incrementality
 * WHY:  Fine control over checkpoint size vs speed tradeoff
 * HOW:  Same as checkpoint_save() but with custom options
 *
 * THREAD-SAFE: Yes (acquires brain mutex)
 *
 * @param brain Brain instance to checkpoint
 * @param path Output path for checkpoint file
 * @param options Checkpoint creation options
 * @return true on success, false on failure
 */
bool checkpoint_save_ex(brain_t brain, const char* path, const checkpoint_options_t* options);

/**
 * @brief Save incremental checkpoint (delta only)
 *
 * WHAT: Save only weights/state that changed since last checkpoint
 * WHY:  Faster saves, smaller files for frequent checkpointing
 * HOW:  Compare to previous checkpoint, save only deltas
 *
 * REQUIREMENTS:
 * - Previous full checkpoint must exist at base_path
 * - Brain state must be tracked (internal delta tracking)
 *
 * PERFORMANCE:
 * - 10-100x faster than full checkpoint (depends on change rate)
 * - 10-100x smaller file size
 *
 * THREAD-SAFE: Yes (acquires brain mutex)
 *
 * @param brain Brain instance to checkpoint
 * @param path Output path for incremental checkpoint
 * @param base_path Path to base (full) checkpoint
 * @return true on success, false on failure
 */
bool checkpoint_save_incremental(brain_t brain, const char* path, const char* base_path);

//=============================================================================
// Checkpoint Loading
//=============================================================================

/**
 * @brief Load brain from checkpoint
 *
 * WHAT: Deserialize brain state from checkpoint file
 * WHY:  Restore brain after crash or for inference
 * HOW:  Validate checksum, parse format, reconstruct brain
 *
 * VALIDATION:
 * - Magic bytes and version check
 * - CRC32 checksum verification
 * - Format compatibility check
 *
 * ALLOCATION:
 * - Creates new brain instance (caller must free with brain_destroy)
 * - Allocates network, subsystems, buffers
 *
 * THREAD-SAFE: Yes (new brain is independent)
 *
 * ERROR HANDLING:
 * - Returns false on corruption, version mismatch, I/O error
 * - *brain is NULL on failure
 * - No partial restoration (all-or-nothing)
 *
 * @param brain Output parameter for loaded brain (NULL on failure)
 * @param path Path to checkpoint file
 * @return true on success, false on failure
 */
bool checkpoint_load(brain_t* brain, const char* path);

/**
 * @brief Validate checkpoint integrity
 *
 * WHAT: Check if checkpoint file is valid and uncorrupted
 * WHY:  Quick check before attempting restore
 * HOW:  Verify magic, version, CRC32 checksum
 *
 * PERFORMANCE:
 * - Fast: O(file size) for checksum
 * - No allocation or parsing
 * - Safe to call on any file
 *
 * CHECKS:
 * 1. File exists and readable
 * 2. Magic bytes match
 * 3. Version is supported
 * 4. CRC32 checksum matches
 *
 * THREAD-SAFE: Yes (read-only operation)
 *
 * @param path Path to checkpoint file
 * @return true if valid, false if corrupted/invalid
 */
bool checkpoint_validate(const char* path);

//=============================================================================
// Checkpoint Management
//=============================================================================

/**
 * @brief List all checkpoints in directory
 *
 * WHAT: Enumerate checkpoint files and extract metadata
 * WHY:  Select checkpoint for recovery, cleanup old checkpoints
 * HOW:  Scan directory, parse headers, sort by timestamp
 *
 * OUTPUT:
 * - *list: Array of checkpoint_info_t (sorted newest first)
 * - *count: Number of checkpoints found
 * - Caller must free *list with nimcp_free()
 *
 * THREAD-SAFE: Yes (read-only directory scan)
 *
 * @param dir Directory to scan for checkpoints
 * @param list Output parameter for checkpoint array
 * @param count Output parameter for checkpoint count
 * @return true on success, false on error
 */
bool checkpoint_list(const char* dir, checkpoint_info_t** list, uint32_t* count);

/**
 * @brief Delete old checkpoints, keeping N most recent
 *
 * WHAT: Cleanup old checkpoints to save disk space
 * WHY:  Prevent unbounded checkpoint accumulation
 * HOW:  List checkpoints, sort by timestamp, delete oldest
 *
 * POLICY:
 * - Keep keep_count most recent valid checkpoints
 * - Delete all older checkpoints
 * - Never delete if keep_count = 0 (safety)
 *
 * THREAD-SAFE: Yes (uses file locks if available)
 *
 * @param dir Directory containing checkpoints
 * @param keep_count Number of checkpoints to keep
 * @return true on success, false on error
 */
bool checkpoint_cleanup_old(const char* dir, uint32_t keep_count);

//=============================================================================
// Recovery Operations
//=============================================================================

/**
 * @brief Automatically restore from latest valid checkpoint
 *
 * WHAT: Find and load most recent valid checkpoint in directory
 * WHY:  Automatic crash recovery without manual intervention
 * HOW:  List checkpoints, validate, load newest valid
 *
 * RECOVERY STRATEGY:
 * 1. List all checkpoints in directory (sorted newest first)
 * 2. Validate each checkpoint (CRC32, version)
 * 3. Load first valid checkpoint
 * 4. Return false if no valid checkpoints found
 *
 * THREAD-SAFE: Yes (creates new brain instance)
 *
 * @param brain Output parameter for restored brain
 * @param checkpoint_dir Directory containing checkpoints
 * @return true if brain restored, false if no valid checkpoints
 */
bool recovery_auto_restore(brain_t* brain, const char* checkpoint_dir);

/**
 * @brief Rollback brain to previous checkpoint
 *
 * WHAT: Replace brain state with checkpoint state
 * WHY:  Undo bad training, recover from corruption
 * HOW:  Load checkpoint, replace brain internals, preserve handle
 *
 * WARNING: Destructive operation!
 * - Current brain state is discarded
 * - Cannot be undone (unless you checkpoint first)
 *
 * THREAD-SAFE: Yes (acquires brain mutex)
 *
 * @param brain Brain instance to rollback (modified in-place)
 * @param checkpoint_path Path to checkpoint to restore
 * @return true on success, false on failure
 */
bool recovery_rollback(brain_t brain, const char* checkpoint_path);

/**
 * @brief Recover partial state from corrupted checkpoint
 *
 * WHAT: Load as much as possible from corrupted checkpoint
 * WHY:  Salvage brain when full restore fails
 * HOW:  Parse file, skip corrupted sections, load valid data
 *
 * RECOVERY LEVELS:
 * - Full: All data valid (same as checkpoint_load)
 * - Partial: Some subsystems corrupted, weights OK
 * - Minimal: Only config/dimensions valid, weights corrupted
 * - Failed: Unable to recover anything
 *
 * USE CASE:
 * - Disk corruption during save
 * - Partial file write (power loss)
 * - Format version mismatch
 *
 * THREAD-SAFE: Yes (creates new brain instance)
 *
 * @param brain Output parameter for recovered brain
 * @param path Path to potentially corrupted checkpoint
 * @param recovery_level Output parameter for recovery level (0-3)
 * @return true if any recovery succeeded, false on total failure
 */
bool recovery_partial(brain_t* brain, const char* path, int* recovery_level);

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get default checkpoint options
 *
 * WHAT: Return recommended checkpoint configuration
 * WHY:  Simplify API usage with sensible defaults
 * HOW:  Static initialization
 *
 * DEFAULTS:
 * - Compression: enabled (level 6)
 * - Incremental: false
 * - Subsystems: true
 * - Activations: false
 *
 * @return Default checkpoint options
 */
checkpoint_options_t checkpoint_default_options(void);

/**
 * @brief Get last checkpoint error message
 *
 * WHAT: Return human-readable error from last failed operation
 * WHY:  Debugging and user-facing error messages
 * HOW:  Thread-local error buffer
 *
 * THREAD-SAFE: Yes (thread-local storage)
 *
 * @return Error message string (valid until next call)
 */
const char* checkpoint_get_error(void);

/**
 * @brief Clear checkpoint error state
 *
 * WHAT: Reset error message to empty string
 * WHY:  Clear stale errors between operations
 * HOW:  Zero thread-local buffer
 *
 * THREAD-SAFE: Yes (thread-local storage)
 */
void checkpoint_clear_error(void);

/**
 * @brief Get checkpoint format version string
 *
 * WHAT: Return version string for checkpoint format
 * WHY:  Version reporting and compatibility checking
 * HOW:  Static string from version constants
 *
 * FORMAT: "major.minor" (e.g., "1.0")
 *
 * @return Version string (static storage)
 */
const char* checkpoint_get_version(void);

#ifdef __cplusplus
}
#endif

#endif  // NIMCP_CHECKPOINT_H
