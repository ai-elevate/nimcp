//=============================================================================
// nimcp_synapse_embeddings.c - Semantic Embeddings for Synapses
//=============================================================================
/**
 * @file nimcp_synapse_embeddings.c
 * @brief ENHANCEMENT 1: Semantic embeddings for intelligent synaptic routing
 *
 * WHAT: Per-synapse semantic vectors enable context-aware information routing
 * WHY: Biological synapses are functionally specialized - this models that
 * HOW: Each synapse has embedding vector, updated via learning, used for routing
 *
 * USE CASES:
 * 1. **Semantic Routing:** Route visual info through vision-tuned synapses
 * 2. **Zero-Shot Transfer:** Connect new concepts via embedding similarity
 * 3. **Rapid Learning:** Initialize weights based on semantic match
 * 4. **Pruning:** Remove synapses with low relevance to network function
 *
 * PERFORMANCE:
 * - Memory: 512 bytes per synapse (128D * 4 bytes)
 * - Similarity: O(dim) dot product
 * - Update: O(dim) vector addition
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 */

#include "core/neuralnet/nimcp_neuralnet.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// === BIO-ASYNC + LOGGING + UNIFIED MEMORY INTEGRATION ===
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/exception/nimcp_exception_macros.h"

#define LOG_MODULE "synapse_embeddings"
#define BIO_MODULE_ID 0x013C


//=============================================================================
// Initialization
//=============================================================================

bool synapse_init_embedding(synapse_t *synapse, uint16_t dim) {
    if (!synapse || dim == 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "synapse_init_embedding: invalid parameters");

            return false;
    }

    // Free existing embedding if present
    if (synapse->semantic_embedding) {
        nimcp_free(synapse->semantic_embedding);
    }

    // Allocate embedding vector
    synapse->semantic_embedding = (float*)nimcp_malloc(dim * sizeof(float));
    if (!synapse->semantic_embedding) {
        return false;
    }

    synapse->embedding_dim = dim;

    // Initialize with random values (Xavier/Glorot initialization)
    float scale = sqrtf(2.0F / dim);
    for (uint16_t i = 0; i < dim; i++) {
        synapse->semantic_embedding[i] = ((float)rand() / RAND_MAX - 0.5F) * 2.0F * scale;
    }

    // Normalize to unit length
    float norm = 0.0F;
    for (uint16_t i = 0; i < dim; i++) {
        norm += synapse->semantic_embedding[i] * synapse->semantic_embedding[i];
    }
    norm = sqrtf(norm);

    if (norm > 1e-6F) {
        for (uint16_t i = 0; i < dim; i++) {
            synapse->semantic_embedding[i] /= norm;
        }
    }

    synapse->semantic_relevance = 0.5F;  // Neutral initial relevance

    return true;
}

//=============================================================================
// Similarity Computation
//=============================================================================

float synapse_semantic_similarity(const synapse_t *syn1, const synapse_t *syn2) {
    if (!syn1 || !syn2) {
        return 0.0F;
    }

    // Check both have embeddings
    if (!syn1->semantic_embedding || !syn2->semantic_embedding) {
        return 0.0F;
    }

    // Check dimensions match
    if (syn1->embedding_dim != syn2->embedding_dim) {
        return 0.0F;
    }

    // Compute cosine similarity: dot(a,b) / (||a|| * ||b||)
    // Since embeddings are normalized, this simplifies to just dot(a,b)
    float dot_product = 0.0F;
    for (uint16_t i = 0; i < syn1->embedding_dim; i++) {
        dot_product += syn1->semantic_embedding[i] * syn2->semantic_embedding[i];
    }

    return dot_product;  // Already in [-1, 1] if normalized
}

//=============================================================================
// Embedding Update
//=============================================================================

bool synapse_update_embedding(synapse_t *synapse, const float *target_embedding, float learning_rate) {
    if (!synapse || !target_embedding) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM,

                "synapse_update_embedding: invalid parameters");

            return false;
    }

    // Check synapse has embedding
    if (!synapse->semantic_embedding || synapse->embedding_dim == 0) {
        return false;
    }

    // Gradient descent step: emb += lr * (target - emb)
    for (uint16_t i = 0; i < synapse->embedding_dim; i++) {
        float error = target_embedding[i] - synapse->semantic_embedding[i];
        synapse->semantic_embedding[i] += learning_rate * error;
    }

    // Renormalize to unit length
    float norm = 0.0F;
    for (uint16_t i = 0; i < synapse->embedding_dim; i++) {
        norm += synapse->semantic_embedding[i] * synapse->semantic_embedding[i];
    }
    norm = sqrtf(norm);

    if (norm > 1e-6F) {
        for (uint16_t i = 0; i < synapse->embedding_dim; i++) {
            synapse->semantic_embedding[i] /= norm;
        }
    }

    return true;
}

//=============================================================================
// Relevance Computation
//=============================================================================

float synapse_compute_relevance(synapse_t *synapse, const float *context_embedding, uint16_t context_dim) {
    if (!synapse || !context_embedding) {
        return 0.0F;
    }

    // Check synapse has embedding
    if (!synapse->semantic_embedding || synapse->embedding_dim == 0) {
        return 0.0F;
    }

    // Check dimensions match
    if (synapse->embedding_dim != context_dim) {
        return 0.0F;
    }

    // Compute cosine similarity with context
    float dot_product = 0.0F;
    float context_norm = 0.0F;

    for (uint16_t i = 0; i < context_dim; i++) {
        dot_product += synapse->semantic_embedding[i] * context_embedding[i];
        context_norm += context_embedding[i] * context_embedding[i];
    }

    context_norm = sqrtf(context_norm);

    if (context_norm < 1e-6F) {
        return 0.0F;
    }

    // Cosine similarity (synapse embedding already normalized)
    float similarity = dot_product / context_norm;

    // Map from [-1, 1] to [0, 1]
    float relevance = (similarity + 1.0F) / 2.0F;

    // Cache relevance in synapse for fast access
    synapse->semantic_relevance = relevance;

    return relevance;
}

//=============================================================================
// Cleanup
//=============================================================================

void synapse_destroy_embedding(synapse_t *synapse) {
    if (!synapse) {
        return;
    }

    if (synapse->semantic_embedding) {
        nimcp_free(synapse->semantic_embedding);
        synapse->semantic_embedding = NULL;
    }

    synapse->embedding_dim = 0;
    synapse->semantic_relevance = 0.0F;
}
