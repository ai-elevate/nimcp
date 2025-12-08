//=============================================================================
// nimcp_api_internal.h - Internal API Structure Definitions
//=============================================================================
/**
 * @file nimcp_api_internal.h
 * @brief Internal structure definitions for NIMCP API handles
 *
 * This header is shared between API implementation files.
 * NOT for external use - these structures are opaque to users.
 */

#ifndef NIMCP_API_INTERNAL_H
#define NIMCP_API_INTERNAL_H

#include "core/brain/nimcp_brain.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "cognitive/ethics/nimcp_ethics.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Internal Handle Structures
//=============================================================================

struct nimcp_brain_handle {
    brain_t internal_brain;  // Wraps internal brain_t
};

struct nimcp_network_handle {
    neural_network_t internal_network;  // Wraps internal neural_network_t
};

struct nimcp_ethics_handle {
    ethics_engine_t internal_ethics;  // Wraps internal ethics_engine_t
};

struct nimcp_knowledge_handle {
    knowledge_system_t internal_knowledge;  // Wraps internal knowledge_system_t
};

/**
 * @brief Brain snapshot handle for COW save/restore
 */
struct nimcp_brain_snapshot_handle {
    brain_t internal_brain_snapshot;  // Snapshot of brain state
    uint64_t timestamp_us;            // Snapshot creation time
    size_t shared_memory_size;        // Size of shared memory (for tracking)
    uint32_t snapshot_refcount;       // Reference count for this snapshot
    bool is_isolated;                 // Isolation flag (true if snapshot is independent)
};

#ifdef __cplusplus
}
#endif

#endif // NIMCP_API_INTERNAL_H
