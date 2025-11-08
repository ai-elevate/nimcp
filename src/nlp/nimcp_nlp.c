//=============================================================================
// nimcp_nlp.c - NLP Integration Module Implementation
//=============================================================================
/**
 * @file nimcp_nlp.c
 * @brief NLP integration combining synapses, attention, and neuromodulation
 *
 * ARCHITECTURE:
 * This module acts as a Facade over three major NIMCP subsystems:
 * 1. Neural Network: Programmable synapses with custom compute functions
 * 2. Multihead Attention: Query-key-value semantic routing
 * 3. Neuromodulators: Reward-based learning and gating
 *
 * DESIGN PATTERN: Facade Pattern
 * - Provides unified interface to complex subsystem interactions
 * - Hides internal wiring complexity from users
 * - Manages lifecycle of multiple interdependent components
 *
 * CRITICAL INTEGRATION POINTS:
 * - synapse_compute_context_t bridges synapses ↔ attention ↔ neuromodulation
 * - Attention output becomes global_state for synapse computation
 * - Neuromodulator levels modulate synaptic transmission and learning
 *
 * @author Claude Code + NIMCP Development Team
 * @date 2025-11-07
 * @version 2.7.0
 */

#include "nimcp_nlp.h"
#include "core/synapse_compute/nimcp_synapse_compute.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Internal NLP network structure
 *
 * WHAT: Complete state for integrated NLP system
 * WHY: Encapsulates all subsystems and their interconnections
 * HOW: Aggregates handles to neural net, attention, neuromodulators
 *
 * DESIGN: Composite Pattern - treats network as unified entity
 */
struct nlp_network_struct {
    // Core subsystems
    neural_network_t network;              /**< Base neural network */
    multihead_attention_t attention;       /**< Multihead attention system */
    neuromodulator_system_t neuromodulators; /**< Neuromodulator system */

    // Configuration
    nlp_network_config_t config;           /**< Configuration (copy) */

    // Embedding matrix: vocab_size × embedding_dim
    float* embeddings;                     /**< Word embeddings [vocab_size × embedding_dim] */

    // Attention context (shared with synapses via global_state)
    float* attention_output;               /**< Last attention output [sequence_length × output_dim] */
    uint32_t attention_output_size;        /**< Size of attention output buffer */

    // Sequence processing state
    uint32_t last_sequence_length;         /**< Length of last processed sequence */
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/**
 * @brief Initialize embedding matrix
 *
 * WHAT: Allocate and randomly initialize word embeddings
 * WHY: Need continuous representations for discrete tokens
 * HOW: Xavier/Glorot initialization for stable gradients
 *
 * COMPLEXITY: O(v × d) where v=vocab_size, d=embedding_dim
 *
 * @param network NLP network
 * @return true on success, false on allocation failure
 */
static bool init_embeddings(nlp_network_t network) {
    if (!network) return false;

    uint32_t vocab_size = network->config.vocab_size;
    uint32_t embedding_dim = network->config.embedding_dim;

    // Allocate embedding matrix
    size_t size = vocab_size * embedding_dim * sizeof(float);
    network->embeddings = (float*)malloc(size);
    if (!network->embeddings) {
        return false;
    }

    // Xavier initialization: uniform(-sqrt(6/(vocab+dim)), +sqrt(6/(vocab+dim)))
    float range = sqrtf(6.0f / (vocab_size + embedding_dim));
    for (uint32_t i = 0; i < vocab_size * embedding_dim; i++) {
        float r = (float)rand() / RAND_MAX;  // [0,1]
        network->embeddings[i] = 2.0f * range * r - range;  // [-range, +range]
    }

    return true;
}

/**
 * @brief Configure synapses with NLP-specific compute functions
 *
 * WHAT: Attach attention/neuromodulation compute functions to synapses
 * WHY: Enable context-dependent transmission for language
 * HOW: Iterate through neurons, set compute_function pointers
 *
 * COMPLEXITY: O(n × s) where n=neurons, s=synapses_per_neuron
 *
 * @param network NLP network
 * @return Number of synapses configured
 */
static uint32_t configure_synapses(nlp_network_t network) {
    if (!network || !network->network) return 0;

    uint32_t configured = 0;
    // NOTE: This requires access to internal network structure
    // We'll implement this after updating neural_network_struct

    // TODO: Iterate through neurons and configure synapses based on config flags:
    // - network->config.use_attention_synapses
    // - network->config.use_semantic_synapses
    // - network->config.use_gating_synapses
    // - network->config.use_neuromodulated_synapses

    return configured;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

nlp_network_t nlp_network_create(const nlp_network_config_t* config) {
    // Guard: Validate input
    if (!config) return NULL;
    if (config->vocab_size == 0 || config->embedding_dim == 0) return NULL;

    // Allocate network structure
    nlp_network_t network = (nlp_network_t)calloc(1, sizeof(struct nlp_network_struct));
    if (!network) return NULL;

    // Copy configuration
    memcpy(&network->config, config, sizeof(nlp_network_config_t));

    // Create base neural network
    network->network = neural_network_create(&config->network_config);
    if (!network->network) {
        free(network);
        return NULL;
    }

    // Create multihead attention system
    network->attention = multihead_attention_create(&config->attention_config);
    if (!network->attention) {
        neural_network_destroy(network->network);
        free(network);
        return NULL;
    }

    // Create neuromodulator system
    network->neuromodulators = neuromodulator_system_create(&config->neuromod_config);
    if (!network->neuromodulators) {
        multihead_attention_destroy(network->attention);
        neural_network_destroy(network->network);
        free(network);
        return NULL;
    }

    // CRITICAL: Wire neuromodulator system to neural network
    // This enables synapses to access neuromodulator levels during computation
    neural_network_set_neuromodulator_system(network->network, network->neuromodulators);

    // Initialize embeddings
    if (!init_embeddings(network)) {
        neuromodulator_system_destroy(network->neuromodulators);
        multihead_attention_destroy(network->attention);
        neural_network_destroy(network->network);
        free(network);
        return NULL;
    }

    // Allocate attention output buffer
    network->attention_output_size = config->max_sequence_length *
                                    config->attention_config.output_dim;
    network->attention_output = (float*)calloc(network->attention_output_size, sizeof(float));
    if (!network->attention_output) {
        free(network->embeddings);
        neuromodulator_system_destroy(network->neuromodulators);
        multihead_attention_destroy(network->attention);
        neural_network_destroy(network->network);
        free(network);
        return NULL;
    }

    // Configure synapses with NLP compute functions
    configure_synapses(network);

    return network;
}

void nlp_network_destroy(nlp_network_t network) {
    // Guard: Null check
    if (!network) return;

    // Destroy in reverse creation order
    if (network->attention_output) {
        free(network->attention_output);
    }

    if (network->embeddings) {
        free(network->embeddings);
    }

    if (network->neuromodulators) {
        neuromodulator_system_destroy(network->neuromodulators);
    }

    if (network->attention) {
        multihead_attention_destroy(network->attention);
    }

    if (network->network) {
        neural_network_destroy(network->network);
    }

    free(network);
}

//=============================================================================
// Forward Pass Functions
//=============================================================================

bool nlp_network_forward(
    nlp_network_t network,
    const uint32_t* token_ids,
    uint32_t sequence_length,
    float* output,
    uint32_t output_dim
) {
    // Guard: Validate inputs
    if (!network || !token_ids || !output) return false;
    if (sequence_length == 0 || sequence_length > network->config.max_sequence_length) return false;
    if (output_dim != network->config.attention_config.output_dim) return false;

    // Step 1: Convert tokens to embeddings
    uint32_t embedding_dim = network->config.embedding_dim;
    float* sequence_embeddings = (float*)malloc(sequence_length * embedding_dim * sizeof(float));
    if (!sequence_embeddings) return false;

    for (uint32_t i = 0; i < sequence_length; i++) {
        uint32_t token = token_ids[i];
        if (token >= network->config.vocab_size) {
            free(sequence_embeddings);
            return false;
        }

        // Copy embedding for this token
        memcpy(
            &sequence_embeddings[i * embedding_dim],
            &network->embeddings[token * embedding_dim],
            embedding_dim * sizeof(float)
        );
    }

    // Step 2: Run multihead attention over embeddings
    // Note: Attention expects input of shape [sequence_length × input_dim]
    bool attention_success = multihead_attention_forward(
        network->attention,
        sequence_embeddings,
        sequence_length,
        NULL,  // No salience weighting (can be added later)
        network->attention_output
    );

    free(sequence_embeddings);

    if (!attention_success) {
        return false;
    }

    // Store sequence length for context
    network->last_sequence_length = sequence_length;

    // Step 3: Wire attention output into network as global state
    // CRITICAL: This makes attention output available to all synapses
    // Synapses with attention-based compute functions can access this via
    // synapse_compute_context_t.global_state
    neural_network_set_global_state(
        network->network,
        network->attention_output,
        network->attention_output_size
    );

    // Step 4: Run neural network forward pass
    // The attention output is now available to all synapse compute functions
    // In a more sophisticated implementation, we would inject initial activations
    // and let the network propagate them

    // Step 4: Extract output from network
    // For now, copy attention output directly
    // In full implementation, this would come from output neurons
    uint32_t output_size = sequence_length * output_dim;
    if (output_size > network->attention_output_size) {
        output_size = network->attention_output_size;
    }

    memcpy(output, network->attention_output, output_size * sizeof(float));

    return true;
}

bool nlp_network_get_embedding(
    nlp_network_t network,
    uint32_t token_id,
    float* embedding
) {
    // Guard: Validate inputs
    if (!network || !embedding) return false;
    if (token_id >= network->config.vocab_size) return false;

    // Copy embedding
    uint32_t offset = token_id * network->config.embedding_dim;
    memcpy(
        embedding,
        &network->embeddings[offset],
        network->config.embedding_dim * sizeof(float)
    );

    return true;
}

bool nlp_network_set_embedding(
    nlp_network_t network,
    uint32_t token_id,
    const float* embedding
) {
    // Guard: Validate inputs
    if (!network || !embedding) return false;
    if (token_id >= network->config.vocab_size) return false;

    // Update embedding
    uint32_t offset = token_id * network->config.embedding_dim;
    memcpy(
        &network->embeddings[offset],
        embedding,
        network->config.embedding_dim * sizeof(float)
    );

    return true;
}

//=============================================================================
// Attention Control Functions
//=============================================================================

bool nlp_network_set_attention_gate(nlp_network_t network, float gate_signal) {
    // Guard: Validate inputs
    if (!network || !network->attention) return false;

    // Forward to attention system
    return multihead_attention_set_gate(network->attention, gate_signal);
}

bool nlp_network_get_attention_weights(
    nlp_network_t network,
    uint32_t head_idx,
    float* weights
) {
    // Guard: Validate inputs
    if (!network || !network->attention || !weights) return false;
    if (head_idx >= network->config.attention_config.num_heads) return false;

    // TODO: Implement attention weight extraction
    // The current attention system doesn't expose individual head weights
    // Would need to add multihead_attention_get_head_weights() to nimcp_attention.h

    return false;  // Not implemented yet
}

//=============================================================================
// Neuromodulation Functions
//=============================================================================

float nlp_network_release_dopamine(
    nlp_network_t network,
    float reward_magnitude,
    float predicted_reward
) {
    // Guard: Validate inputs
    if (!network || !network->neuromodulators) return 0.0f;

    // Release dopamine and return prediction error
    return neuromodulator_release_dopamine(
        network->neuromodulators,
        reward_magnitude,
        predicted_reward
    );
}

float nlp_network_release_acetylcholine(
    nlp_network_t network,
    float salience
) {
    // Guard: Validate inputs
    if (!network || !network->neuromodulators) return 0.0f;

    // Release acetylcholine
    return neuromodulator_release_acetylcholine(network->neuromodulators, salience);
}

bool nlp_network_get_neuromodulator_levels(
    nlp_network_t network,
    float* dopamine,
    float* serotonin,
    float* acetylcholine,
    float* norepinephrine
) {
    // Guard: Validate inputs
    if (!network || !network->neuromodulators) return false;

    // Get neuromodulator pool
    neuromodulator_pool_t pool;
    bool success = neuromodulator_get_levels(network->neuromodulators, &pool);
    if (!success) return false;

    // Extract individual levels if requested
    if (dopamine) *dopamine = pool.dopamine;
    if (serotonin) *serotonin = pool.serotonin;
    if (acetylcholine) *acetylcholine = pool.acetylcholine;
    if (norepinephrine) *norepinephrine = pool.norepinephrine;

    return true;
}

//=============================================================================
// Training Functions
//=============================================================================

float nlp_network_train(
    nlp_network_t network,
    const uint32_t* token_ids,
    uint32_t sequence_length,
    const float* target,
    uint32_t output_dim,
    float learning_rate
) {
    // Guard: Validate inputs
    if (!network || !token_ids || !target) return -1.0f;

    // Step 1: Forward pass
    float* output = (float*)malloc(sequence_length * output_dim * sizeof(float));
    if (!output) return -1.0f;

    bool forward_success = nlp_network_forward(
        network,
        token_ids,
        sequence_length,
        output,
        output_dim
    );

    if (!forward_success) {
        free(output);
        return -1.0f;
    }

    // Step 2: Compute loss (mean squared error)
    float loss = 0.0f;
    uint32_t total_elements = sequence_length * output_dim;
    for (uint32_t i = 0; i < total_elements; i++) {
        float error = output[i] - target[i];
        loss += error * error;
    }
    loss /= total_elements;

    // Step 3: Release dopamine based on loss (negative reward)
    float reward = -loss;  // Lower loss = higher reward
    nlp_network_release_dopamine(network, reward, 0.0f);

    // Step 4: Apply learning (TODO: implement backpropagation)
    // For now, this is a placeholder - full implementation would:
    // - Compute gradients
    // - Apply STDP with neuromodulation
    // - Update attention weights
    // - Update embeddings

    free(output);
    return loss;
}

//=============================================================================
// Utility Functions
//=============================================================================

bool nlp_network_get_stats(nlp_network_t network, network_stats_t* stats) {
    // Guard: Validate inputs
    if (!network || !stats || !network->network) return false;

    // Forward to base network
    return neural_network_get_stats(network->network, stats);
}

bool nlp_network_save(nlp_network_t network, const char* filepath) {
    // Guard: Validate inputs
    if (!network || !filepath) return false;

    // TODO: Implement serialization
    // Would write:
    // - Config
    // - Embeddings
    // - Network weights
    // - Attention parameters
    // - Neuromodulator state

    return false;  // Not implemented yet
}

nlp_network_t nlp_network_load(const char* filepath) {
    // Guard: Validate inputs
    if (!filepath) return NULL;

    // TODO: Implement deserialization
    // Would read and reconstruct all components

    return NULL;  // Not implemented yet
}
