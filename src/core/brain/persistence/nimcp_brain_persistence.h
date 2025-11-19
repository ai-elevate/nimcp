//=============================================================================
// nimcp_brain_persistence.h - Brain Persistence & Snapshot Management
//=============================================================================
/**
 * @file nimcp_brain_persistence.h
 * @brief Brain state persistence, snapshot management, and recovery
 *
 * WHAT: Provides save/load/snapshot APIs for brain state persistence
 * WHY:  Enable model checkpointing, disaster recovery, A/B testing, version control
 * HOW:  Serialize brain state to files with versioning, compression, encryption support
 *
 * ARCHITECTURE:
 * - Persistence API: Save/load complete brain state (network + metadata + subsystems)
 * - Snapshot API: Named, timestamped snapshots with metadata
 * - Format Versioning: Support for format evolution and backward compatibility
 * - Security: Optional compression and encryption (AES-256)
 *
 * PERFORMANCE:
 * - Save: O(n*c + k) where n=neurons, c=connections, k=labels
 * - Load: O(n*c + k) with validation
 * - Snapshot operations: Same as save/load + directory management
 *
 * DESIGN PRINCIPLES:
 * - Atomic operations: Partial writes don't corrupt existing state
 * - Validation: Strict checks on load to prevent buffer overflows
 * - Modularity: Each subsystem saves to separate file
 * - Extensibility: Version header supports format evolution
 */

#ifndef NIMCP_BRAIN_PERSISTENCE_H
#define NIMCP_BRAIN_PERSISTENCE_H

#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Persistence API
//=============================================================================

/**
 * @brief Save brain to file
 *
 * WHAT: Persist complete brain state to disk
 * WHY:  Enable model checkpointing, transfer learning, disaster recovery
 * HOW:  Save network → metadata → subsystems to separate files
 *
 * FILES CREATED:
 * - {filepath}         : Network structure and weights
 * - {filepath}.meta    : Brain configuration, labels, stats, timestamps
 * - {filepath}.knowledge : Knowledge system state (if exists)
 * - {filepath}.executive : Executive controller state (if exists)
 * - {filepath}.pink_noise : Pink noise neuromodulator state (if exists)
 * - {filepath}.mirror_neurons : Mirror neuron system state (if exists)
 *
 * COMPLEXITY: O(n*c + k + m) where:
 *   n = neurons, c = connections per neuron
 *   k = output labels, m = working memory items
 *
 * @param brain Brain instance (non-NULL)
 * @param filepath Base filepath for saving (non-NULL)
 * @return true on success, false on error
 */
bool brain_save(brain_t brain, const char* filepath);

/**
 * @brief Load brain from file
 *
 * WHAT: Restore complete brain state from disk
 * WHY:  Enable model reuse, transfer learning, disaster recovery
 * HOW:  Load network → metadata → subsystems, validate all inputs
 *
 * SECURITY: All loaded data is validated to prevent buffer overflows:
 * - Config fields checked for NaN/Inf
 * - Array sizes checked against maximum limits
 * - String lengths validated before allocation
 * - File format version compatibility checked
 *
 * COMPLEXITY: O(n*c + k + m) where:
 *   n = neurons, c = connections per neuron
 *   k = output labels, m = working memory items
 *
 * @param filepath Base filepath for loading (non-NULL)
 * @return Brain instance on success, NULL on error
 */
brain_t brain_load(const char* filepath);

//=============================================================================
// Snapshot API - Named State Snapshots
//=============================================================================

/**
 * @brief Snapshot metadata structure
 *
 * WHAT: Metadata for a brain snapshot
 * WHY:  Enable snapshot discovery, version tracking, audit trails
 * HOW:  Stored in .snapshot.info file alongside snapshot
 *
 * NOTE: brain_snapshot_info_t is defined in nimcp_brain.h (lines 1251-1258)
 *       and included via "core/brain/nimcp_brain.h" above
 */

/**
 * @brief Save brain snapshot with compression and encryption
 *
 * WHAT: Create a named, timestamped snapshot of complete brain state
 * WHY:  Enable backups, A/B testing, version control, disaster recovery
 * HOW:  Save to snapshot_dir with optional compression/encryption
 *
 * SNAPSHOT FORMAT:
 * - Filename: {snapshot_dir}/{name}_{timestamp}.snapshot
 * - Metadata: {snapshot_dir}/{name}_{timestamp}.snapshot.info
 * - Includes: All brain state (network, subsystems, knowledge)
 * - Compression: zlib (if enabled in config)
 * - Encryption: AES-256 (if enabled in config)
 *
 * EXAMPLE:
 * ```c
 * brain_save_snapshot(brain, "experiment_v1", "Before parameter tuning");
 * // Creates: snapshots/experiment_v1_1642531200.snapshot
 * //          snapshots/experiment_v1_1642531200.snapshot.info
 * ```
 *
 * @param brain Brain instance (non-NULL)
 * @param name Snapshot name (non-NULL, e.g., "before_experiment", "v1.0")
 * @param description Optional description (can be NULL)
 * @return true on success, false on error
 */
bool brain_save_snapshot(brain_t brain, const char* name, const char* description);

/**
 * @brief Restore brain from snapshot
 *
 * WHAT: Load brain state from most recent snapshot with given name
 * WHY:  Restore previous state, rollback changes, A/B testing
 * HOW:  Find latest snapshot → decompress/decrypt → load state
 *
 * BEHAVIOR:
 * - Finds most recent snapshot matching {name}
 * - If brain parameter is non-NULL, warns that in-place restore not yet implemented
 * - Returns new brain instance with restored state
 *
 * EXAMPLE:
 * ```c
 * brain_t restored = brain_restore_snapshot(NULL, "experiment_v1");
 * // Loads: snapshots/experiment_v1_<latest_timestamp>.snapshot
 * ```
 *
 * @param brain Brain instance to restore into (currently ignored, pass NULL)
 * @param name Snapshot name to restore (non-NULL)
 * @return Brain instance with restored state, or NULL on error
 */
brain_t brain_restore_snapshot(brain_t brain, const char* name);

/**
 * @brief List available snapshots
 *
 * WHAT: Enumerate all snapshots in snapshot directory
 * WHY:  Allow users to see available restore points
 * HOW:  Scan snapshot_dir, parse .snapshot.info metadata files
 *
 * EXAMPLE:
 * ```c
 * brain_snapshot_info_t infos[100];
 * uint32_t count;
 * brain_list_snapshots(brain, infos, 100, &count);
 * for (uint32_t i = 0; i < count; i++) {
 *     printf("%s: %s (size: %u)\n", infos[i].name,
 *            infos[i].description, infos[i].file_size);
 * }
 * ```
 *
 * @param brain Brain instance (for snapshot_dir config)
 * @param infos Output array of snapshot info (allocated by caller)
 * @param max_count Maximum number of snapshots to return
 * @param out_count Output: actual number of snapshots found
 * @return true on success, false on error
 */
bool brain_list_snapshots(brain_t brain, brain_snapshot_info_t* infos,
                         uint32_t max_count, uint32_t* out_count);

/**
 * @brief Delete snapshot and associated files
 *
 * WHAT: Remove snapshot and all associated files
 * WHY:  Free disk space, remove obsolete restore points
 * HOW:  Find latest snapshot matching name → delete all files
 *
 * FILES DELETED:
 * - {name}_{timestamp}.snapshot
 * - {name}_{timestamp}.snapshot.info
 * - {name}_{timestamp}.snapshot.meta
 * - {name}_{timestamp}.snapshot.knowledge (if exists)
 *
 * @param brain Brain instance (for snapshot_dir config)
 * @param name Snapshot name to delete (non-NULL)
 * @return true on success, false on error
 */
bool brain_delete_snapshot(brain_t brain, const char* name);

//=============================================================================
// Internal Helper Functions (for nimcp_brain.c use only)
//=============================================================================

/**
 * @brief Save working memory state to file
 *
 * INTERNAL USE ONLY
 *
 * @param wm Working memory instance (nullable)
 * @param file File handle (non-NULL)
 * @return true on success, false on error
 */
bool nimcp_brain_save_working_memory_state(working_memory_t* wm, FILE* file);

/**
 * @brief Load working memory state from file
 *
 * INTERNAL USE ONLY
 *
 * @param brain Brain instance (non-NULL)
 * @param file File handle (non-NULL)
 * @return true on success (non-fatal on WM failure)
 */
bool nimcp_brain_load_working_memory_state(brain_t brain, FILE* file);

/**
 * @brief Save metadata file
 *
 * INTERNAL USE ONLY
 *
 * @param brain Brain instance (non-NULL)
 * @param filepath Base filepath (non-NULL)
 * @return true on success, false on error
 */
bool nimcp_brain_save_metadata(brain_t brain, const char* filepath);

/**
 * @brief Load metadata file
 *
 * INTERNAL USE ONLY
 *
 * @param brain Brain instance (non-NULL)
 * @param filepath Base filepath (non-NULL)
 * @return true on success, false on error
 */
bool nimcp_brain_load_metadata(brain_t brain, const char* filepath);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_BRAIN_PERSISTENCE_H
