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

#include "nimcp_neuralnet.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

//=============================================================================
// Initialization
//=============================================================================

bool synapse_init_embedding(synapse_t *synapse, uint16_t dim) {
    if (!synapse || dim == 0) {
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
    float scale = sqrtf(2.0f / dim);
    for (uint16_t i = 0; i < dim; i++) {
        synapse->semantic_embedding[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }

    // Normalize to unit length
    float norm = 0.0f;
    for (uint16_t i = 0; i < dim; i++) {
        norm += synapse->semantic_embedding[i] * synapse->semantic_embedding[i];
    }
    norm = sqrtf(norm);

    if (norm > 1e-6f) {
        for (uint16_t i = 0; i < dim; i++) {
            synapse->semantic_embedding[i] /= norm;
        }
    }

    synapse->semantic_relevance = 0.5f;  // Neutral initial relevance

    return true;
}

//=============================================================================
// Similarity Computation
//=============================================================================

float synapse_semantic_similarity(const synapse_t *syn1, const synapse_t *syn2) {
    if (!syn1 || !syn2) {
        return 0.0f;
    }

    // Check both have embeddings
    if (!syn1->semantic_embedding || !syn2->semantic_embedding) {
        return 0.0f;
    }

    // Check dimensions match
    if (syn1->embedding_dim != syn2->embedding_dim) {
        return 0.0f;
    }

    // Compute cosine similarity: dot(a,b) / (||a|| * ||b||)
    // Since embeddings are normalized, this simplifies to just dot(a,b)
    float dot_product = 0.0f;
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
    float norm = 0.0f;
    for (uint16_t i = 0; i < synapse->embedding_dim; i++) {
        norm += synapse->semantic_embedding[i] * synapse->semantic_embedding[i];
    }
    norm = sqrtf(norm);

    if (norm > 1e-6f) {
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
        return 0.0f;
    }

    // Check synapse has embedding
    if (!synapse->semantic_embedding || synapse->embedding_dim == 0) {
        return 0.0f;
    }

    // Check dimensions match
    if (synapse->embedding_dim != context_dim) {
        return 0.0f;
    }

    // Compute cosine similarity with context
    float dot_product = 0.0f;
    float context_norm = 0.0f;

    for (uint16_t i = 0; i < context_dim; i++) {
        dot_product += synapse->semantic_embedding[i] * context_embedding[i];
        context_norm += context_embedding[i] * context_embedding[i];
    }

    context_norm = sqrtf(context_norm);

    if (context_norm < 1e-6f) {
        return 0.0f;
    }

    // Cosine similarity (synapse embedding already normalized)
    float similarity = dot_product / context_norm;

    // Map from [-1, 1] to [0, 1]
    float relevance = (similarity + 1.0f) / 2.0f;

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
    synapse->semantic_relevance = 0.0f;
}
