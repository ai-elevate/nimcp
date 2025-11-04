//=============================================================================
// brain_demo.c - NIMCP Brain API Demonstration
//=============================================================================
/**
 * @file brain_demo.c
 * @brief Comprehensive demonstration of NIMCP Brain API for LLM decision caching
 *
 * WHAT THIS DEMONSTRATES:
 * - Creating lightweight neural "brains" for pattern learning
 * - Learning from LLM decisions (knowledge distillation)
 * - Fast local inference without API calls
 * - Brain persistence (save/load)
 * - Decision interpretability
 * - Production optimization
 *
 * WHY THIS IS USEFUL:
 * - Reduces LLM API costs (zero cost after training)
 * - 100-1000x faster inference (microseconds vs milliseconds)
 * - Works offline (no network dependency)
 * - Privacy preserved (local computation)
 * - Interpretable (can inspect active neurons)
 * - Lightweight (MB vs GB for full LLMs)
 *
 * USE CASE: Ethics Decision Caching
 * LLMs make expensive ethics decisions. This demo shows how to:
 * 1. Let LLM make initial decisions (teacher)
 * 2. Train small brain from those decisions (student)
 * 3. Use brain for fast subsequent decisions
 * 4. Maintain interpretability and confidence scores
 *
 * ARCHITECTURE:
 * - Strategy Pattern: Decision rules via function pointers
 * - Teacher-Student Pattern: LLM teaches lightweight brain
 * - Repository Pattern: Trained brain persistence
 *
 * PERFORMANCE:
 * - Training: O(n) where n = num_examples
 * - Inference: O(s) where s = sparsity * num_neurons (typically <100)
 * - Memory: O(neurons * connections) ≈ 10MB for SMALL brain
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "core/brain/nimcp_brain.h"

//=============================================================================
// Simulated LLM Teacher - Strategy Pattern
//=============================================================================

/**
 * @brief Ethics rule evaluation function type
 *
 * WHY: Strategy pattern eliminates if-else-if chains. Each rule is its own
 * function, making code extensible and testable.
 */
typedef struct {
    bool (*predicate)(float harm, float fairness, float transparency, float autonomy);
    const char* label;
    float confidence;
} ethics_rule_t;

/**
 * @brief Rule: Block high harm actions
 *
 * COMPLEXITY: O(1)
 */
static bool rule_high_harm(float harm, float fairness, float transparency, float autonomy)
{
    (void) fairness;
    (void) transparency;
    (void) autonomy;
    return harm > 0.7f;
}

/**
 * @brief Rule: Block moderate harm + unfair actions
 *
 * COMPLEXITY: O(1)
 */
static bool rule_harmful_and_unfair(float harm, float fairness, float transparency, float autonomy)
{
    (void) transparency;
    (void) autonomy;
    return harm > 0.4f && fairness < 0.3f;
}

/**
 * @brief Rule: Warn on low transparency
 *
 * COMPLEXITY: O(1)
 */
static bool rule_low_transparency(float harm, float fairness, float transparency, float autonomy)
{
    (void) harm;
    (void) fairness;
    (void) autonomy;
    return transparency < 0.2f;
}

/**
 * @brief Rule: Warn on low autonomy
 *
 * COMPLEXITY: O(1)
 */
static bool rule_low_autonomy(float harm, float fairness, float transparency, float autonomy)
{
    (void) harm;
    (void) fairness;
    (void) transparency;
    return autonomy < 0.3f;
}

/**
 * @brief Rule table for ethics evaluation (Strategy Pattern)
 *
 * WHY: Replaces if-else-if chain with data-driven dispatch. New rules can be
 * added without modifying existing code (Open/Closed Principle).
 *
 * INVARIANT: Rules ordered by priority (checked top to bottom)
 */
static const ethics_rule_t ETHICS_RULES[] = {{rule_high_harm, "block", 0.95f},
                                             {rule_harmful_and_unfair, "block", 0.85f},
                                             {rule_low_transparency, "warn", 0.75f},
                                             {rule_low_autonomy, "warn", 0.70f}};

static const uint32_t NUM_ETHICS_RULES = sizeof(ETHICS_RULES) / sizeof(ETHICS_RULES[0]);

/**
 * @brief Simulates an LLM making an ethics decision using rule table
 *
 * WHY: In production, this would call Claude/GPT-4 API. This simulation uses
 * a rule table to demonstrate the teacher-student pattern without API costs.
 *
 * ALGORITHM:
 * 1. Extract features (O(1))
 * 2. Evaluate each rule until match (O(n) where n = num_rules, typically <10)
 * 3. Return label and confidence (O(1))
 *
 * COMPLEXITY: O(n) where n = NUM_ETHICS_RULES (4), essentially O(1)
 *
 * @param features Input feature vector [harm, fairness, transparency, autonomy]
 * @param num_features Size of feature vector (should be 4)
 * @param context Optional context (unused)
 * @param output_label Output buffer for decision label
 * @param max_label_len Maximum label buffer size
 * @return Confidence score [0.0, 1.0]
 */
static float simulated_llm_ethics_decision(const float* features, uint32_t num_features,
                                           void* context, char* output_label,
                                           uint32_t max_label_len)
{
    (void) context;       // Unused
    (void) num_features;  // Assumed to be 4

    // Extract features for readability
    float harm = features[0];
    float fairness = features[1];
    float transparency = features[2];
    float autonomy = features[3];

    // Evaluate rules in priority order - O(n) where n = NUM_ETHICS_RULES
    for (uint32_t i = 0; i < NUM_ETHICS_RULES; i++) {
        const ethics_rule_t* rule = &ETHICS_RULES[i];

        // Check if rule matches (O(1) per rule)
        if (rule->predicate(harm, fairness, transparency, autonomy)) {
            strncpy(output_label, rule->label, max_label_len);
            output_label[max_label_len - 1] = '\0';  // Ensure null termination
            return rule->confidence;
        }
    }

    // Default case: allow if no rules triggered
    strncpy(output_label, "allow", max_label_len);
    output_label[max_label_len - 1] = '\0';
    return 0.90f;
}

//=============================================================================
// Training Data Generation
//=============================================================================

/**
 * @brief Generate random ethics scenarios for training
 */
static void generate_training_scenario(float* features, char* label, float* confidence)
{
    // Random ethical situation
    features[0] = (float) rand() / RAND_MAX;  // harm
    features[1] = (float) rand() / RAND_MAX;  // fairness
    features[2] = (float) rand() / RAND_MAX;  // transparency
    features[3] = (float) rand() / RAND_MAX;  // autonomy

    // Get LLM decision
    simulated_llm_ethics_decision(features, 4, NULL, label, 64);

    // Simulate confidence
    *confidence = 0.8 + 0.2 * ((float) rand() / RAND_MAX);
}

//=============================================================================
// Main Demonstration
//=============================================================================

int main(void)
{
    srand(time(NULL));

    printf("===========================================\n");
    printf(" NIMCP Brain API Demonstration\n");
    printf(" Use Case: Ethics Decision Caching\n");
    printf("===========================================\n\n");

    //=========================================================================
    // Step 1: Create a brain for ethics decisions
    //=========================================================================
    printf("Step 1: Creating ethics decision brain...\n");

    brain_t ethics_brain = brain_create("artemis_ethics",           // Name
                                        BRAIN_SIZE_SMALL,           // 1K neurons, ~10MB
                                        BRAIN_TASK_CLASSIFICATION,  // Multi-class classification
                                        4,  // 4 inputs: harm, fairness, transparency, autonomy
                                        3   // 3 outputs: allow, warn, block
    );

    if (!ethics_brain) {
        fprintf(stderr, "Failed to create brain: %s\n", brain_get_last_error());
        return EXIT_FAILURE;
    }

    brain_stats_t initial_stats;
    brain_get_stats(ethics_brain, &initial_stats);
    printf("  Created '%s' brain\n", initial_stats.task_name);
    printf("  Neurons: %u\n", initial_stats.num_neurons);
    printf("  Memory: %.2f MB\n\n", initial_stats.memory_bytes / (1024.0 * 1024.0));

    //=========================================================================
    // Step 2: Train from simulated LLM decisions
    //=========================================================================
    printf("Step 2: Training from LLM decisions...\n");

    uint32_t num_training_examples = 1000;
    float total_loss = 0.0f;

    clock_t training_start = clock();

    for (uint32_t i = 0; i < num_training_examples; i++) {
        float features[4];
        char label[64];
        float confidence;

        generate_training_scenario(features, label, &confidence);

        float loss = brain_learn_example(ethics_brain, features, 4, label, confidence);

        total_loss += loss;

        if ((i + 1) % 100 == 0) {
            printf("  Trained on %u examples, avg loss: %.4f\n", i + 1, total_loss / (i + 1));
        }
    }

    clock_t training_end = clock();
    double training_time = ((double) (training_end - training_start)) / CLOCKS_PER_SEC;

    printf("  Training completed in %.2f seconds\n", training_time);
    printf("  Final average loss: %.4f\n\n", total_loss / num_training_examples);

    //=========================================================================
    // Step 3: Test fast inference
    //=========================================================================
    printf("Step 3: Testing fast inference (no LLM needed)...\n\n");

    // Test scenarios
    struct {
        float features[4];
        const char* description;
    } test_cases[] = {{{0.9, 0.5, 0.5, 0.5}, "High harm scenario"},
                      {{0.2, 0.8, 0.8, 0.8}, "Safe, fair scenario"},
                      {{0.5, 0.2, 0.5, 0.5}, "Moderate harm, unfair"},
                      {{0.3, 0.6, 0.1, 0.7}, "Low transparency"}};

    uint64_t total_inference_time = 0;

    for (size_t i = 0; i < sizeof(test_cases) / sizeof(test_cases[0]); i++) {
        printf("Test Case %zu: %s\n", i + 1, test_cases[i].description);
        printf("  Features: harm=%.2f, fair=%.2f, trans=%.2f, auto=%.2f\n",
               test_cases[i].features[0], test_cases[i].features[1], test_cases[i].features[2],
               test_cases[i].features[3]);

        brain_decision_t* decision = brain_decide(ethics_brain, test_cases[i].features, 4);

        if (decision) {
            printf("  Decision: %s (confidence: %.2f)\n", decision->label, decision->confidence);
            printf("  Active neurons: %u (sparsity: %.1f%%)\n", decision->num_active_neurons,
                   decision->sparsity * 100.0f);
            printf("  Inference time: %lu μs\n", decision->inference_time_us);
            printf("  Explanation: %s\n\n", decision->explanation);

            total_inference_time += decision->inference_time_us;
            brain_free_decision(decision);
        }
    }

    printf("Average inference time: %lu μs (%.3f ms)\n\n", total_inference_time / 4,
           (total_inference_time / 4) / 1000.0);

    //=========================================================================
    // Step 4: Performance comparison
    //=========================================================================
    printf("Step 4: Performance Comparison\n");
    printf("  LLM API call: ~500-2000 ms\n");
    printf("  NIMCP Brain: ~%.3f ms\n", (total_inference_time / 4) / 1000.0);
    printf("  Speedup: %.0fx faster\n\n", 1000000.0 / (total_inference_time / 4));

    //=========================================================================
    // Step 5: Interpretability
    //=========================================================================
    printf("Step 5: Interpretability - Top Contributing Neurons\n");

    uint32_t top_neuron_ids[10];
    float top_importances[10];

    uint32_t num_top = brain_get_top_neurons(ethics_brain, 10, top_neuron_ids, top_importances);

    for (uint32_t i = 0; i < num_top; i++) {
        printf("  Neuron %u: importance = %.4f\n", top_neuron_ids[i], top_importances[i]);
    }
    printf("\n");

    //=========================================================================
    // Step 6: Optimization
    //=========================================================================
    printf("Step 6: Optimizing for production...\n");

    brain_stats_t before_opt;
    brain_get_stats(ethics_brain, &before_opt);
    printf("  Before optimization:\n");
    printf("    Active synapses: %u\n", before_opt.num_active_synapses);
    printf("    Memory: %.2f MB\n", before_opt.memory_bytes / (1024.0 * 1024.0));

    // Prune weak connections
    uint32_t pruned = brain_prune(ethics_brain, 0.01f);
    printf("  Pruned %u weak synapses\n", pruned);

    // Full optimization
    brain_optimize_for_inference(ethics_brain);

    brain_stats_t after_opt;
    brain_get_stats(ethics_brain, &after_opt);
    printf("  After optimization:\n");
    printf("    Active synapses: %u\n", after_opt.num_active_synapses);
    printf("    Memory: %.2f MB\n", after_opt.memory_bytes / (1024.0 * 1024.0));
    printf("    Reduction: %.1f%%\n\n",
           100.0 * (1.0 - (float) after_opt.num_active_synapses / before_opt.num_active_synapses));

    //=========================================================================
    // Step 7: Persistence
    //=========================================================================
    printf("Step 7: Saving trained brain...\n");

    const char* save_path = "artemis_ethics_brain.nimcp";
    if (brain_save(ethics_brain, save_path)) {
        printf("  Saved to: %s\n", save_path);

        // Demonstrate loading
        brain_t loaded_brain = brain_load(save_path);
        if (loaded_brain) {
            printf("  Successfully loaded brain\n");

            // Verify it works
            float test_features[] = {0.8, 0.3, 0.5, 0.6};
            brain_decision_t* test_decision = brain_decide(loaded_brain, test_features, 4);

            if (test_decision) {
                printf("  Test decision after load: %s (conf: %.2f)\n\n", test_decision->label,
                       test_decision->confidence);
                brain_free_decision(test_decision);
            }

            brain_destroy(loaded_brain);
        }
    } else {
        fprintf(stderr, "  Failed to save: %s\n\n", brain_get_last_error());
    }

    //=========================================================================
    // Summary
    //=========================================================================
    printf("===========================================\n");
    printf(" Summary\n");
    printf("===========================================\n");

    brain_stats_t final_stats;
    brain_get_stats(ethics_brain, &final_stats);

    printf("Brain: %s\n", final_stats.task_name);
    printf("  Size: %u neurons, %u active synapses\n", final_stats.num_neurons,
           final_stats.num_active_synapses);
    printf("  Training: %lu examples\n", final_stats.total_learning_steps);
    printf("  Inferences: %lu\n", final_stats.total_inferences);
    printf("  Avg inference: %.3f ms\n", final_stats.avg_inference_time_us / 1000.0);
    printf("  Avg sparsity: %.1f%%\n", final_stats.avg_sparsity * 100.0);
    printf("  Memory: %.2f MB\n", final_stats.memory_bytes / (1024.0 * 1024.0));

    printf("\nBenefits:\n");
    printf("  ✓ 100-1000x faster than LLM calls\n");
    printf("  ✓ Works offline (no API dependency)\n");
    printf("  ✓ Zero cost per inference\n");
    printf("  ✓ Privacy preserved (local inference)\n");
    printf("  ✓ Interpretable (can see active neurons)\n");
    printf("  ✓ Lightweight (~10MB vs 7GB+ for LLMs)\n");

    //=========================================================================
    // Cleanup
    //=========================================================================
    brain_destroy(ethics_brain);

    printf("\nDemonstration complete!\n");
    return EXIT_SUCCESS;
}
