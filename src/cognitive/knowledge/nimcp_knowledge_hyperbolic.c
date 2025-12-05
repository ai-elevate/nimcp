//=============================================================================
// nimcp_knowledge_hyperbolic.c - Hyperbolic Knowledge Embeddings Implementation
//=============================================================================
/**
 * @file nimcp_knowledge_hyperbolic.c
 * @brief Implementation of hyperbolic embeddings for hierarchical knowledge
 *
 * PART B1.1: Hyperbolic Knowledge Graph Embeddings
 *
 * WHAT: Embed hierarchical knowledge in hyperbolic space (Poincaré ball)
 * WHY: 200x memory reduction (5D hyperbolic vs 1000D Euclidean)
 * HOW: Use Poincaré ball model with Riemannian optimization
 *
 * MATHEMATICAL FOUNDATION:
 * - Hyperbolic space has exponential growth matching tree hierarchies
 * - Distance d(x,y) = acosh(1 + 2||x-y||²/((1-||x||²)(1-||y||²)))
 * - Learning via Riemannian SGD on manifold
 *
 * PERFORMANCE:
 * - Memory: O(dim) per concept (dim = 5-10 for hyperbolic)
 * - k-NN query: O(n * dim) brute force, O(log n * dim) with ball tree
 * - Learning: O(epochs * n² * dim) for full optimization
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 * @version 1.0.0
 */

#include "cognitive/knowledge/nimcp_knowledge.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_messages.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/geometry/nimcp_hyperbolic.h"
#include "utils/memory/nimcp_memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

//=============================================================================
// Internal Helper Structures
//=============================================================================

/**
 * @brief Item-distance pair for k-NN sorting
 */
typedef struct {
    knowledge_item_t *item;
    float distance;
} knn_candidate_t;

/**
 * @brief Comparison function for qsort
 */
static int compare_knn_candidates(const void *a, const void *b) {
    const knn_candidate_t *ca = (const knn_candidate_t*)a;
    const knn_candidate_t *cb = (const knn_candidate_t*)b;

    if (ca->distance < cb->distance) return -1;
    if (ca->distance > cb->distance) return 1;
    return 0;
}

//=============================================================================
// Embedding Initialization
//=============================================================================

bool knowledge_init_hyperbolic_embedding(knowledge_item_t *item, uint32_t dim,
                                         float hierarchical_level,
                                         const knowledge_item_t *parent) {
    if (!item || dim == 0) {
        return false;
    }

    // If already has hyperbolic embedding, destroy it first
    if (item->hyperbolic_embedding) {
        poincare_point_destroy(item->hyperbolic_embedding);
        item->hyperbolic_embedding = NULL;
    }

    // Compute radius based on hierarchical level
    // Root concepts (level 0) → radius ≈ 0
    // Intermediate concepts → radius ≈ 0.5
    // Specific concepts → radius ≈ 0.8-0.9
    float radius = tanhf(hierarchical_level * 0.5f);
    radius = fminf(radius, 0.95f);  // Cap at 0.95 for numerical stability

    // Allocate coordinates
    float *coords = (float*)nimcp_malloc(dim * sizeof(float));
    if (!coords) {
        return false;
    }

    // If has parent, initialize near parent with small random offset
    if (parent && parent->hyperbolic_embedding) {
        // Start from parent's position
        memcpy(coords, parent->hyperbolic_embedding->coords, dim * sizeof(float));

        // Add small random offset (0.1-0.2 in random direction)
        float offset_magnitude = 0.1f + 0.1f * ((float)rand() / RAND_MAX);
        for (uint32_t i = 0; i < dim; i++) {
            float offset = (2.0f * ((float)rand() / RAND_MAX) - 1.0f) * offset_magnitude;
            coords[i] += offset;
        }
    } else {
        // No parent - initialize at random position on sphere of given radius
        // First generate random unit vector
        float norm = 0.0f;
        for (uint32_t i = 0; i < dim; i++) {
            coords[i] = 2.0f * ((float)rand() / RAND_MAX) - 1.0f;
            norm += coords[i] * coords[i];
        }
        norm = sqrtf(norm);

        // Normalize and scale to target radius
        if (norm > 1e-6f) {
            for (uint32_t i = 0; i < dim; i++) {
                coords[i] = (coords[i] / norm) * radius;
            }
        } else {
            // Degenerate case - just use first axis
            coords[0] = radius;
            for (uint32_t i = 1; i < dim; i++) {
                coords[i] = 0.0f;
            }
        }
    }

    // Create Poincaré point
    item->hyperbolic_embedding = poincare_point_create(dim, coords, -1.0f);
    nimcp_free(coords);

    if (!item->hyperbolic_embedding) {
        return false;
    }

    // Update item metadata
    item->embedding_dim = dim;
    item->use_hyperbolic = true;
    item->hierarchical_level = hierarchical_level;

    return true;
}

//=============================================================================
// Distance Computation
//=============================================================================

float knowledge_hyperbolic_distance(const knowledge_item_t *item1,
                                    const knowledge_item_t *item2) {
    // Validate inputs
    if (!item1 || !item2) {
        return -1.0f;
    }

    // Check both have hyperbolic embeddings
    if (!item1->hyperbolic_embedding || !item2->hyperbolic_embedding) {
        return -1.0f;
    }

    // Check dimensions match
    if (item1->hyperbolic_embedding->dim != item2->hyperbolic_embedding->dim) {
        return -1.0f;
    }

    // Compute hyperbolic distance
    return poincare_distance(item1->hyperbolic_embedding, item2->hyperbolic_embedding);
}

//=============================================================================
// k-Nearest Neighbors Search
//=============================================================================

uint32_t knowledge_hyperbolic_knn(knowledge_system_t system,
                                  const knowledge_item_t *query_item,
                                  uint32_t k,
                                  knowledge_item_t **neighbors_out,
                                  float *distances_out) {
    if (!system || !query_item || k == 0 || !neighbors_out) {
        return 0;
    }

    // Check query has hyperbolic embedding
    if (!query_item->hyperbolic_embedding) {
        return 0;
    }

    // Get all knowledge items from system
    // NOTE: This requires access to internal knowledge_system structure
    // For now, we'll use the confidence-based retrieval as a workaround
    knowledge_item_t *all_items = NULL;
    uint32_t num_items = knowledge_get_all_ordered_by_confidence(system, &all_items);

    if (num_items == 0 || !all_items) {
        return 0;
    }

    // Allocate candidates array
    knn_candidate_t *candidates = (knn_candidate_t*)nimcp_malloc(num_items * sizeof(knn_candidate_t));
    if (!candidates) {
        nimcp_free(all_items);
        return 0;
    }

    // Compute distances to all items
    uint32_t valid_candidates = 0;
    for (uint32_t i = 0; i < num_items; i++) {
        // Skip query item itself
        if (&all_items[i] == query_item) {
            continue;
        }

        // Skip items without hyperbolic embeddings
        if (!all_items[i].hyperbolic_embedding) {
            continue;
        }

        // Compute distance
        float dist = poincare_distance(query_item->hyperbolic_embedding,
                                      all_items[i].hyperbolic_embedding);

        if (dist >= 0.0f) {  // Valid distance
            candidates[valid_candidates].item = &all_items[i];
            candidates[valid_candidates].distance = dist;
            valid_candidates++;
        }
    }

    // Sort candidates by distance
    qsort(candidates, valid_candidates, sizeof(knn_candidate_t), compare_knn_candidates);

    // Copy top k to output
    uint32_t num_neighbors = (k < valid_candidates) ? k : valid_candidates;
    for (uint32_t i = 0; i < num_neighbors; i++) {
        neighbors_out[i] = candidates[i].item;
        if (distances_out) {
            distances_out[i] = candidates[i].distance;
        }
    }

    // Cleanup
    nimcp_free(candidates);
    // NOTE: Don't free all_items - it's owned by the system

    return num_neighbors;
}

//=============================================================================
// Riemannian SGD Update
//=============================================================================

bool knowledge_hyperbolic_sgd_step(knowledge_item_t *item,
                                   const float *euclidean_gradient,
                                   float learning_rate) {
    if (!item || !euclidean_gradient) {
        return false;
    }

    // Check item has hyperbolic embedding
    if (!item->hyperbolic_embedding) {
        return false;
    }

    // Use Poincaré SGD step from hyperbolic library
    return poincare_sgd_step(item->hyperbolic_embedding, euclidean_gradient, learning_rate);
}

//=============================================================================
// Hyperbolic Embedding Learning
//=============================================================================

float knowledge_learn_hyperbolic_embeddings(knowledge_system_t system,
                                            uint32_t num_epochs,
                                            float learning_rate) {
    if (!system || num_epochs == 0) {
        return -1.0f;
    }

    // Get all knowledge items
    knowledge_item_t *all_items = NULL;
    uint32_t num_items = knowledge_get_all_ordered_by_confidence(system, &all_items);

    if (num_items < 2 || !all_items) {
        return -1.0f;
    }

    // Count items with hyperbolic embeddings
    uint32_t num_hyperbolic = 0;
    for (uint32_t i = 0; i < num_items; i++) {
        if (all_items[i].hyperbolic_embedding) {
            num_hyperbolic++;
        }
    }

    if (num_hyperbolic < 2) {
        return -1.0f;  // Need at least 2 items with embeddings
    }

    // Get embedding dimension from first item
    uint32_t dim = 0;
    for (uint32_t i = 0; i < num_items; i++) {
        if (all_items[i].hyperbolic_embedding) {
            dim = all_items[i].hyperbolic_embedding->dim;
            break;
        }
    }

    if (dim == 0) {
        return -1.0f;
    }

    // Allocate gradient buffer
    float *gradient = (float*)nimcp_malloc(dim * sizeof(float));
    if (!gradient) {
        return -1.0f;
    }

    float total_loss = 0.0f;
    float initial_lr = learning_rate;

    // Training loop
    for (uint32_t epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;
        uint32_t num_pairs = 0;

        // Decay learning rate
        float current_lr = initial_lr * (1.0f / (1.0f + 0.01f * epoch));

        // For each pair of items
        for (uint32_t i = 0; i < num_items; i++) {
            if (!all_items[i].hyperbolic_embedding) continue;

            for (uint32_t j = i + 1; j < num_items; j++) {
                if (!all_items[j].hyperbolic_embedding) continue;

                // Compute target distance based on hierarchical relationship
                float target_distance;

                // If parent-child relationship
                if (all_items[i].parent_index == j || all_items[j].parent_index == i) {
                    target_distance = 1.0f;  // Parent-child should be close
                }
                // If same hierarchical level (siblings)
                else if (fabsf(all_items[i].hierarchical_level - all_items[j].hierarchical_level) < 0.5f) {
                    target_distance = 2.0f;  // Siblings moderately close
                }
                // If different domains
                else if (all_items[i].domain != all_items[j].domain) {
                    target_distance = 5.0f;  // Different domains far apart
                }
                // Default
                else {
                    target_distance = 3.0f;  // Related but distinct
                }

                // Compute current distance
                float current_distance = poincare_distance(all_items[i].hyperbolic_embedding,
                                                          all_items[j].hyperbolic_embedding);

                // Compute loss: (d_current - d_target)²
                float error = current_distance - target_distance;
                float loss = error * error;
                epoch_loss += loss;
                num_pairs++;

                // Compute gradient (simplified version)
                // grad_loss = 2 * error * grad_distance
                // For now, use approximate gradient descent

                if (fabsf(error) > 0.01f) {  // Only update if significant error
                    // Move points closer or farther based on error
                    float direction = (error > 0) ? -1.0f : 1.0f;
                    float magnitude = fminf(fabsf(error) * 0.1f, 0.5f);

                    // Compute direction vector: (item_i - item_j)
                    for (uint32_t d = 0; d < dim; d++) {
                        float diff = all_items[i].hyperbolic_embedding->coords[d] -
                                   all_items[j].hyperbolic_embedding->coords[d];
                        gradient[d] = direction * magnitude * diff;
                    }

                    // Apply gradient to item i
                    knowledge_hyperbolic_sgd_step(&all_items[i], gradient, current_lr);

                    // Apply negative gradient to item j
                    for (uint32_t d = 0; d < dim; d++) {
                        gradient[d] = -gradient[d];
                    }
                    knowledge_hyperbolic_sgd_step(&all_items[j], gradient, current_lr);
                }
            }
        }

        // Compute average loss for this epoch
        if (num_pairs > 0) {
            epoch_loss /= num_pairs;
        }

        total_loss = epoch_loss;

        // Print progress every 10 epochs
        if ((epoch + 1) % 10 == 0) {
            printf("Epoch %u/%u: Loss = %.6f, LR = %.6f\n",
                   epoch + 1, num_epochs, epoch_loss, current_lr);
        }
    }

    // Cleanup
    nimcp_free(gradient);

    return total_loss;
}

//=============================================================================
// Euclidean to Hyperbolic Conversion
//=============================================================================

bool knowledge_euclidean_to_hyperbolic(knowledge_item_t *item, uint32_t target_dim) {
    if (!item || target_dim == 0) {
        return false;
    }

    // Check item has Euclidean embedding
    if (!item->euclidean_embedding || item->embedding_dim == 0) {
        return false;
    }

    // For now, use simple projection approach:
    // 1. Take first target_dim dimensions of Euclidean embedding
    // 2. Normalize to fit in Poincaré ball

    uint32_t source_dim = item->embedding_dim;
    uint32_t copy_dim = (target_dim < source_dim) ? target_dim : source_dim;

    // Allocate hyperbolic coordinates
    float *hyp_coords = (float*)nimcp_malloc(target_dim * sizeof(float));
    if (!hyp_coords) {
        return false;
    }

    // Copy and normalize
    float norm = 0.0f;
    for (uint32_t i = 0; i < copy_dim; i++) {
        hyp_coords[i] = item->euclidean_embedding[i];
        norm += hyp_coords[i] * hyp_coords[i];
    }

    // Fill remaining dimensions with zeros
    for (uint32_t i = copy_dim; i < target_dim; i++) {
        hyp_coords[i] = 0.0f;
    }

    // Normalize to fit in ball (||x|| < 1)
    // Use tanh to squash to appropriate radius
    norm = sqrtf(norm);
    if (norm > 1e-6f) {
        float target_radius = tanhf(norm * 0.5f);  // Map R^n norm to ball
        target_radius = fminf(target_radius, 0.95f);

        for (uint32_t i = 0; i < target_dim; i++) {
            hyp_coords[i] = (hyp_coords[i] / norm) * target_radius;
        }
    }

    // Create Poincaré point
    if (item->hyperbolic_embedding) {
        poincare_point_destroy(item->hyperbolic_embedding);
    }

    item->hyperbolic_embedding = poincare_point_create(target_dim, hyp_coords, -1.0f);
    nimcp_free(hyp_coords);

    if (!item->hyperbolic_embedding) {
        return false;
    }

    // Update metadata
    item->use_hyperbolic = true;

    return true;
}

//=============================================================================
// Hierarchical Path Tracing
//=============================================================================

uint32_t knowledge_get_hierarchical_path(knowledge_system_t system,
                                         const char *concept_name,
                                         knowledge_item_t **path_out,
                                         uint32_t max_depth) {
    if (!system || !concept_name || !path_out || max_depth == 0) {
        return 0;
    }

    // Find the concept
    knowledge_item_t query_item;
    if (!knowledge_retrieve(system, concept_name, &query_item)) {
        return 0;  // Concept not found
    }

    // Trace upward to root
    uint32_t path_length = 0;
    uint32_t current_index = query_item.parent_index;

    // Add the query item itself
    path_out[path_length++] = &query_item;

    // Get all items to look up parents
    knowledge_item_t *all_items = NULL;
    uint32_t num_items = knowledge_get_all_ordered_by_confidence(system, &all_items);

    if (num_items == 0 || !all_items) {
        return path_length;
    }

    // Follow parent links
    while (current_index != UINT32_MAX && path_length < max_depth) {
        // Find parent item
        if (current_index >= num_items) {
            break;  // Invalid parent index
        }

        path_out[path_length++] = &all_items[current_index];
        current_index = all_items[current_index].parent_index;
    }

    return path_length;
}
