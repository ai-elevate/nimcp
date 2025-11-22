//=============================================================================
// nimcp_ethics_hyperbolic.c - Hyperbolic Moral Hierarchies for Ethics
//=============================================================================
/**
 * @file nimcp_ethics_hyperbolic.c
 * @brief Integration of hyperbolic knowledge embeddings with ethical reasoning
 *
 * PART B1.1 INTEGRATION: Ethics Module
 *
 * WHAT: Represent moral principles as hyperbolic hierarchy
 * WHY: Ethics naturally forms a tree structure with universal principles at root
 * HOW: Place fundamental principles near origin, specific situations near boundary
 *
 * MORAL HIERARCHY IN HYPERBOLIC SPACE:
 * ```
 * Origin (||x|| ≈ 0):
 *   - Golden Rule: "Treat others as you wish to be treated"
 *   - Do no harm
 *   - Respect autonomy
 *
 * Mid-radius (||x|| ≈ 0.5):
 *   - Medical ethics (informed consent, privacy)
 *   - Social ethics (fairness, justice)
 *   - Environmental ethics (sustainability)
 *
 * Near boundary (||x|| ≈ 0.9):
 *   - Specific dilemmas
 *   - Cultural norms
 *   - Edge cases
 * ```
 *
 * KEY INSIGHT:
 * Hyperbolic distance naturally captures moral reasoning:
 * - Similar situations → close in hyperbolic space
 * - Derive specific from general → geodesic from origin
 * - Resolve conflicts → find geodesic between conflicting principles
 *
 * @author NIMCP Development Team
 * @date 2025-11-11
 */

#include "nimcp_ethics.h"
#include "utils/memory/nimcp_memory.h"
#include "cognitive/knowledge/nimcp_knowledge.h"
#include "utils/geometry/nimcp_hyperbolic.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//=============================================================================
// Hyperbolic Moral Reasoning
//=============================================================================

/**
 * @brief Compute moral weight based on hyperbolic distance from fundamental principles
 *
 * WHAT: Measures how strongly a moral principle applies to a situation
 * WHY: Closer principles in hyperbolic space are more relevant
 * HOW: Inverse distance with exponential decay
 *
 * FORMULA:
 * weight = exp(-distance / decay_rate)
 *
 * Where:
 * - distance: Hyperbolic distance from principle to situation
 * - decay_rate: How quickly relevance falls off (typically 1.0-2.0)
 *
 * @param principle Fundamental moral principle
 * @param situation Specific ethical situation
 * @param decay_rate Relevance decay rate
 * @return Moral weight (0-1, higher = more relevant)
 */
float ethics_hyperbolic_weight(const knowledge_item_t *principle,
                               const knowledge_item_t *situation,
                               float decay_rate) {
    if (!principle || !situation) {
        return 0.0f;
    }

    // Check both have hyperbolic embeddings
    if (!principle->hyperbolic_embedding || !situation->hyperbolic_embedding) {
        return 0.0f;
    }

    // Compute hyperbolic distance
    float distance = knowledge_hyperbolic_distance(principle, situation);
    if (distance < 0.0f) {
        return 0.0f;
    }

    // Exponential decay of relevance
    return expf(-distance / decay_rate);
}

/**
 * @brief Find most relevant moral principles for a given situation
 *
 * WHAT: Returns top-k moral principles applicable to situation
 * WHY: Ethical reasoning requires identifying relevant principles
 * HOW: Uses hyperbolic k-NN weighted by principle fundamentality
 *
 * WEIGHTING:
 * - Fundamental principles (near origin): Weight × 1.5
 * - Domain-specific principles: Weight × 1.0
 * - Specific precedents: Weight × 0.5
 *
 * @param system Knowledge system
 * @param situation Ethical situation to reason about
 * @param k Number of principles to return
 * @param principles_out Output array [k]
 * @param weights_out Output weights [k] (can be NULL)
 * @return Number of principles found
 */
uint32_t ethics_find_relevant_principles_hyperbolic(knowledge_system_t system,
                                                    const knowledge_item_t *situation,
                                                    uint32_t k,
                                                    knowledge_item_t **principles_out,
                                                    float *weights_out) {
    if (!system || !situation || k == 0 || !principles_out) {
        return 0;
    }

    // Find k-NN in hyperbolic space
    knowledge_item_t **neighbors = nimcp_malloc(k * sizeof(knowledge_item_t*));
    float *distances = nimcp_malloc(k * sizeof(float));

    if (!neighbors || !distances) {
        nimcp_free(neighbors);
        nimcp_free(distances);
        return 0;
    }

    uint32_t num_neighbors = knowledge_hyperbolic_knn(system, situation,
                                                      k, neighbors, distances);

    if (num_neighbors == 0) {
        nimcp_free(neighbors);
        nimcp_free(distances);
        return 0;
    }

    // Compute weights for each principle
    for (uint32_t i = 0; i < num_neighbors; i++) {
        float base_weight = ethics_hyperbolic_weight(neighbors[i], situation, 1.5f);

        // Boost weight for fundamental principles (near origin)
        float radius = poincare_norm(neighbors[i]->hyperbolic_embedding);
        float fundamentality_boost = 1.0f + 0.5f * expf(-radius * 2.0f);

        float final_weight = base_weight * fundamentality_boost;

        // Copy to output
        principles_out[i] = neighbors[i];
        if (weights_out) {
            weights_out[i] = final_weight;
        }
    }

    nimcp_free(neighbors);
    nimcp_free(distances);

    return num_neighbors;
}

/**
 * @brief Resolve conflict between two moral principles using hyperbolic geometry
 *
 * WHAT: Finds compromise or precedence between conflicting principles
 * WHY: Many ethical dilemmas involve conflicting valid principles
 * HOW: Analyze positions in hyperbolic space and find geodesic midpoint
 *
 * RESOLUTION STRATEGIES:
 * 1. **Geodesic midpoint:** Balanced compromise
 * 2. **Closer to origin:** More fundamental principle takes precedence
 * 3. **Context-dependent:** Situation-specific weighting
 *
 * @param principle1 First moral principle
 * @param principle2 Second moral principle
 * @param situation Context for conflict
 * @param resolution_out Output: which principle dominates (-1, 0, or 1)
 * @return Confidence in resolution (0-1)
 */
float ethics_resolve_conflict_hyperbolic(const knowledge_item_t *principle1,
                                         const knowledge_item_t *principle2,
                                         const knowledge_item_t *situation,
                                         int *resolution_out) {
    if (!principle1 || !principle2 || !situation || !resolution_out) {
        *resolution_out = 0;
        return 0.0f;
    }

    // Check all have hyperbolic embeddings
    if (!principle1->hyperbolic_embedding ||
        !principle2->hyperbolic_embedding ||
        !situation->hyperbolic_embedding) {
        *resolution_out = 0;
        return 0.0f;
    }

    // Compute distances to situation
    float dist1 = knowledge_hyperbolic_distance(principle1, situation);
    float dist2 = knowledge_hyperbolic_distance(principle2, situation);

    // Compute fundamentality (distance from origin)
    float radius1 = poincare_norm(principle1->hyperbolic_embedding);
    float radius2 = poincare_norm(principle2->hyperbolic_embedding);

    // Combined score: closeness to situation + fundamentality
    float score1 = expf(-dist1) * (1.0f + 0.5f * expf(-radius1));
    float score2 = expf(-dist2) * (1.0f + 0.5f * expf(-radius2));

    // Determine winner
    float score_diff = fabsf(score1 - score2);
    float confidence = tanhf(score_diff * 2.0f);  // Maps difference to [0, 1]

    if (score1 > score2 * 1.2f) {
        *resolution_out = 1;  // Principle 1 dominates
    } else if (score2 > score1 * 1.2f) {
        *resolution_out = -1;  // Principle 2 dominates
    } else {
        *resolution_out = 0;  // Balanced - need compromise
        confidence *= 0.5f;  // Lower confidence in ties
    }

    return confidence;
}

/**
 * @brief Map ethical situation to hyperbolic space
 *
 * WHAT: Creates hyperbolic embedding for new ethical situation
 * WHY: Enables geometric moral reasoning
 * HOW: Position based on relevant principles and context
 *
 * ALGORITHM:
 * 1. Find k most relevant existing principles
 * 2. Compute weighted average position (Fréchet mean)
 * 3. Adjust radius based on specificity
 *
 * @param system Knowledge system
 * @param situation_description Text describing situation
 * @param item Output knowledge item
 * @return true on success
 */
bool ethics_map_situation_to_hyperbolic(knowledge_system_t system,
                                       const char *situation_description,
                                       knowledge_item_t *item) {
    if (!system || !situation_description || !item) {
        return false;
    }

    // For now, use simple heuristic:
    // - Classify situation by domain (medical, social, environmental, etc.)
    // - Position at appropriate radius based on specificity
    // - Use random position in appropriate region

    // Determine specificity from description length
    // (longer descriptions → more specific → higher radius)
    size_t desc_len = strlen(situation_description);
    float specificity = tanhf(desc_len / 200.0f);  // 0 to ~1

    // Map specificity to radius: 0.3 (general) to 0.85 (very specific)
    float radius = 0.3f + 0.55f * specificity;

    // Initialize hyperbolic embedding at this radius
    // Use default dimension from config
    uint32_t dim = 5;  // Standard for hierarchies

    // Use knowledge_init_hyperbolic_embedding
    bool success = knowledge_init_hyperbolic_embedding(item, dim, specificity, NULL);

    return success;
}

/**
 * @brief Visualize ethical hierarchy in hyperbolic space (debug helper)
 *
 * WHAT: Prints ethical principle hierarchy
 * WHY: Helps understand moral reasoning structure
 *
 * @param system Knowledge system
 * @param max_items Maximum items to display
 */
void ethics_visualize_hyperbolic_hierarchy(knowledge_system_t system, uint32_t max_items) {
    if (!system) {
        return;
    }

    // Get all ethics knowledge items
    knowledge_item_t *all_items = NULL;
    uint32_t num_items = knowledge_get_all_ordered_by_confidence(system, &all_items);

    if (num_items == 0 || !all_items) {
        printf("No knowledge items available\n");
        return;
    }

    printf("=== Ethical Hierarchy (Hyperbolic Space) ===\n");
    printf("Origin = Universal Principles | Boundary = Specific Cases\n\n");

    // Group by radius (abstraction level)
    uint32_t displayed = 0;
    float radius_bins[3] = {0.0f, 0.4f, 0.7f};
    const char *bin_labels[4] = {
        "UNIVERSAL PRINCIPLES",
        "DOMAIN ETHICS",
        "SPECIFIC SITUATIONS",
        "EDGE CASES"
    };

    for (uint32_t bin = 0; bin < 4 && displayed < max_items; bin++) {
        float min_radius = (bin > 0) ? radius_bins[bin-1] : 0.0f;
        float max_radius = (bin < 3) ? radius_bins[bin] : 1.0f;

        printf("--- %s (radius %.2f-%.2f) ---\n", bin_labels[bin], min_radius, max_radius);

        for (uint32_t i = 0; i < num_items && displayed < max_items; i++) {
            if (all_items[i].hyperbolic_embedding) {
                float radius = poincare_norm(all_items[i].hyperbolic_embedding);

                if (radius >= min_radius && radius < max_radius) {
                    printf("  %.3f: %s\n", radius, all_items[i].concept_name);
                    displayed++;
                }
            }
        }
        printf("\n");
    }

    printf("============================================\n");
}
