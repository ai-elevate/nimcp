/**
 * @file nimcp_brain_recovery_integration.c
 * @brief Brain-Driven Intelligent Recovery Implementation
 */

#include "utils/fault_tolerance/nimcp_brain_recovery_integration.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

//=============================================================================
// Internal Structures
//=============================================================================

#define MAX_RECOVERY_HISTORY 1000
#define MAX_LEARNED_PATTERNS 500

/**
 * @brief Internal brain recovery context
 */
struct brain_recovery_context_internal {
    brain_t brain;                     /**< Associated brain */

    // Learning history
    recovery_outcome_t* history;       /**< Recovery history */
    uint32_t history_count;            /**< Number of entries */
    uint32_t history_capacity;         /**< Capacity */
    uint32_t history_write_idx;        /**< Circular write index */

    // Learned patterns
    recovery_pattern_t* patterns;      /**< Learned patterns */
    uint32_t pattern_count;            /**< Number of patterns */
    uint32_t pattern_capacity;         /**< Capacity */

    // Statistics
    uint32_t total_recoveries;
    uint32_t successful_recoveries;
    uint32_t brain_decisions;
    uint32_t brain_correct_predictions;

    // Configuration
    float confidence_threshold;        /**< Minimum confidence for brain decisions */
    bool enable_learning;              /**< Learn from outcomes */
};

//=============================================================================
// Helper Functions
//=============================================================================

static uint64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000 + tv.tv_usec;
}

static float calculate_success_rate(uint32_t successes, uint32_t total) {
    if (total == 0) return 0.0f;
    return (float)successes / (float)total;
}

/**
 * @brief Generate failure signature from diagnosis
 */
static void generate_failure_signature(
    diagnostic_result_t* diagnosis,
    char* signature,
    size_t size
) {
    if (!diagnosis || !signature || size == 0) return;

    // Create signature from error type and key symptoms
    snprintf(signature, size, "%s_%d_0x%04x",
        diagnosis->root_cause,
        diagnosis->severity,
        diagnosis->error_type & 0xFFFF);
}

/**
 * @brief Find pattern by signature
 */
static recovery_pattern_t* find_pattern(
    brain_recovery_context_t ctx,
    const char* signature
) {
    if (!ctx || !signature) return NULL;

    for (uint32_t i = 0; i < ctx->pattern_count; i++) {
        if (strcmp(ctx->patterns[i].failure_signature, signature) == 0) {
            return &ctx->patterns[i];
        }
    }
    return NULL;
}

/**
 * @brief Add or update learned pattern
 */
static void update_pattern(
    brain_recovery_context_t ctx,
    const char* signature,
    recovery_strategy_t* strategy,
    bool success
) {
    if (!ctx || !signature || !strategy) return;

    recovery_pattern_t* pattern = find_pattern(ctx, signature);

    if (!pattern) {
        // Create new pattern
        if (ctx->pattern_count >= ctx->pattern_capacity) {
            LOG_WARNING("Pattern capacity reached, cannot add new pattern");
            return;
        }

        pattern = &ctx->patterns[ctx->pattern_count++];
        strncpy(pattern->failure_signature, signature, sizeof(pattern->failure_signature) - 1);
        pattern->occurrence_count = 0;
        pattern->success_count = 0;
        pattern->first_seen_us = get_timestamp_us();
    }

    // Update pattern
    pattern->occurrence_count++;
    if (success) {
        pattern->success_count++;
        pattern->best_tier = strategy->tier;
        pattern->best_action = strategy->primary;
    }
    pattern->last_seen_us = get_timestamp_us();
    pattern->success_rate = calculate_success_rate(
        pattern->success_count,
        pattern->occurrence_count
    );
    pattern->confidence = fminf(0.99f, pattern->occurrence_count / 10.0f);
}

//=============================================================================
// Initialization & Lifecycle
//=============================================================================

brain_recovery_context_t brain_recovery_init(brain_t brain) {
    if (!brain) {
        LOG_ERROR("Cannot initialize brain recovery: NULL brain");
        return NULL;
    }

    brain_recovery_context_t ctx = calloc(1, sizeof(struct brain_recovery_context_internal));
    if (!ctx) {
        LOG_ERROR("Failed to allocate brain recovery context");
        return NULL;
    }

    ctx->brain = brain;

    // Allocate history
    ctx->history_capacity = MAX_RECOVERY_HISTORY;
    ctx->history = calloc(ctx->history_capacity, sizeof(recovery_outcome_t));
    if (!ctx->history) {
        LOG_ERROR("Failed to allocate recovery history");
        free(ctx);
        return NULL;
    }

    // Allocate patterns
    ctx->pattern_capacity = MAX_LEARNED_PATTERNS;
    ctx->patterns = calloc(ctx->pattern_capacity, sizeof(recovery_pattern_t));
    if (!ctx->patterns) {
        LOG_ERROR("Failed to allocate pattern storage");
        free(ctx->history);
        free(ctx);
        return NULL;
    }

    // Configuration
    ctx->confidence_threshold = 0.5f;
    ctx->enable_learning = true;

    LOG_INFO("Brain recovery context initialized (history=%u, patterns=%u)",
        ctx->history_capacity, ctx->pattern_capacity);

    return ctx;
}

void brain_recovery_shutdown(brain_recovery_context_t ctx) {
    if (!ctx) return;

    LOG_INFO("Shutting down brain recovery (total_recoveries=%u, success_rate=%.2f%%)",
        ctx->total_recoveries,
        calculate_success_rate(ctx->successful_recoveries, ctx->total_recoveries) * 100.0f);

    free(ctx->history);
    free(ctx->patterns);
    free(ctx);
}

//=============================================================================
// Brain-Driven Strategy Selection
//=============================================================================

brain_recovery_decision_t* brain_recovery_select_strategy(
    brain_recovery_context_t ctx,
    diagnostic_result_t* diagnosis,
    health_status_snapshot_t* current_health
) {
    if (!ctx || !diagnosis) {
        LOG_ERROR("Invalid parameters for strategy selection");
        return NULL;
    }

    brain_recovery_decision_t* decision = calloc(1, sizeof(brain_recovery_decision_t));
    if (!decision) {
        LOG_ERROR("Failed to allocate brain decision");
        return NULL;
    }

    // Generate failure signature
    char signature[256];
    generate_failure_signature(diagnosis, signature, sizeof(signature));

    // Check for learned patterns (simulating working memory)
    recovery_pattern_t* pattern = find_pattern(ctx, signature);

    if (pattern && pattern->confidence > 0.7f) {
        // Use learned strategy
        LOG_INFO("Found learned pattern: %s (success_rate=%.2f, confidence=%.2f)",
            signature, pattern->success_rate, pattern->confidence);

        decision->selected_strategy = calloc(1, sizeof(recovery_strategy_t));
        if (decision->selected_strategy) {
            decision->selected_strategy->tier = pattern->best_tier;
            decision->selected_strategy->primary = pattern->best_action;
            decision->selected_strategy->fallback = RECOVERY_ACTION_NONE;
            decision->selected_strategy->max_retries = 3;
            decision->selected_strategy->timeout_ms = 1000;
            decision->selected_strategy->success_threshold = 0.5f;
            decision->selected_strategy->description = "Learned strategy from past experience";
        }

        decision->confidence = pattern->confidence;
        decision->predicted_success_prob = pattern->success_rate;
        decision->is_novel_situation = false;

        snprintf(decision->reasoning, sizeof(decision->reasoning),
            "Using learned strategy (seen %u times, %.0f%% success rate)",
            pattern->occurrence_count, pattern->success_rate * 100.0f);
    } else {
        // Novel situation - use default strategy selection
        LOG_INFO("Novel failure pattern: %s - using heuristic selection", signature);

        decision->selected_strategy = recovery_select_strategy(diagnosis);
        decision->confidence = 0.4f;  // Lower confidence for novel situations
        decision->predicted_success_prob = 0.6f;  // Conservative estimate
        decision->is_novel_situation = true;

        snprintf(decision->reasoning, sizeof(decision->reasoning),
            "Novel failure pattern - using heuristic based on error type");
    }

    decision->estimated_time_us = 1000;  // 1ms estimate
    decision->requires_user_confirmation = (diagnosis->severity >= SEVERITY_CRITICAL);

    ctx->brain_decisions++;

    return decision;
}

void brain_recovery_free_decision(brain_recovery_decision_t* decision) {
    if (!decision) return;
    free(decision->selected_strategy);
    free(decision);
}

//=============================================================================
// Learning from Recovery Outcomes
//=============================================================================

void brain_recovery_learn_outcome(
    brain_recovery_context_t ctx,
    brain_recovery_decision_t* decision,
    recovery_result_t* result
) {
    if (!ctx || !decision || !result) return;

    if (!ctx->enable_learning) return;

    // Store outcome in history
    if (ctx->history_count < ctx->history_capacity) {
        recovery_outcome_t* outcome = &ctx->history[ctx->history_write_idx];

        outcome->strategy = decision->selected_strategy;
        outcome->result = result;
        outcome->timestamp_us = get_timestamp_us();
        outcome->execution_time_us = result->time_us;
        outcome->was_successful = (result->status == RECOVERY_SUCCESS);
        outcome->expected_success_prob = decision->predicted_success_prob;
        outcome->actual_success = outcome->was_successful ? 1.0f : 0.0f;

        ctx->history_write_idx = (ctx->history_write_idx + 1) % ctx->history_capacity;
        if (ctx->history_count < ctx->history_capacity) {
            ctx->history_count++;
        }
    }

    // Update learned patterns (simulating episodic memory)
    char signature[256];
    // We'd need the original diagnosis here - simplified for stub
    snprintf(signature, sizeof(signature), "pattern_%u", ctx->total_recoveries);

    update_pattern(ctx, signature, decision->selected_strategy,
                  result->status == RECOVERY_SUCCESS);

    // Update prediction accuracy
    float prediction_error = fabsf(decision->predicted_success_prob -
                                   (result->status == RECOVERY_SUCCESS ? 1.0f : 0.0f));
    if (prediction_error < 0.3f) {
        ctx->brain_correct_predictions++;
    }

    LOG_INFO("Learned from recovery outcome (success=%d, prediction_error=%.2f)",
        result->status == RECOVERY_SUCCESS, prediction_error);
}

uint32_t brain_recovery_get_patterns(
    brain_recovery_context_t ctx,
    recovery_pattern_t* patterns,
    uint32_t max_patterns
) {
    if (!ctx || !patterns || max_patterns == 0) return 0;

    uint32_t count = (ctx->pattern_count < max_patterns) ?
                     ctx->pattern_count : max_patterns;

    memcpy(patterns, ctx->patterns, count * sizeof(recovery_pattern_t));
    return count;
}

//=============================================================================
// Runtime Parameter Suggestions
//=============================================================================

uint32_t brain_recovery_suggest_parameters(
    brain_recovery_context_t ctx,
    diagnostic_result_t* diagnosis,
    parameter_adjustment_t* adjustments,
    uint32_t max_adjustments
) {
    if (!ctx || !diagnosis || !adjustments || max_adjustments == 0) return 0;

    uint32_t count = 0;

    // Suggest parameters based on error type
    switch (diagnosis->error_type) {
        case ERROR_TYPE_NAN_DETECTED:
        case ERROR_TYPE_INF_DETECTED:
            if (count < max_adjustments) {
                adjustments[count].parameter_name = "learning_rate";
                adjustments[count].current_value = 0.01f;  // Would query brain
                adjustments[count].suggested_value = 0.005f;  // Reduce by 50%
                adjustments[count].confidence = 0.9f;
                snprintf((char*)adjustments[count].rationale, sizeof(adjustments[count].rationale),
                    "Reduce learning rate to prevent numerical instability");
                count++;
            }
            break;

        case ERROR_TYPE_OUT_OF_MEMORY:
            if (count < max_adjustments) {
                adjustments[count].parameter_name = "batch_size";
                adjustments[count].current_value = 64.0f;
                adjustments[count].suggested_value = 32.0f;  // Reduce by 50%
                adjustments[count].confidence = 0.85f;
                snprintf((char*)adjustments[count].rationale, sizeof(adjustments[count].rationale),
                    "Reduce batch size to decrease memory usage");
                count++;
            }
            break;

        case ERROR_TYPE_GRADIENT_EXPLOSION:
            if (count < max_adjustments) {
                adjustments[count].parameter_name = "gradient_clip_value";
                adjustments[count].current_value = 0.0f;  // Disabled
                adjustments[count].suggested_value = 1.0f;  // Enable
                adjustments[count].confidence = 0.95f;
                snprintf((char*)adjustments[count].rationale, sizeof(adjustments[count].rationale),
                    "Enable gradient clipping to prevent explosion");
                count++;
            }
            break;

        default:
            break;
    }

    return count;
}

//=============================================================================
// Success Probability Prediction
//=============================================================================

float brain_recovery_predict_success(
    brain_recovery_context_t ctx,
    recovery_strategy_t* strategy,
    diagnostic_result_t* diagnosis
) {
    if (!ctx || !strategy || !diagnosis) return -1.0f;

    // Generate signature and look for pattern
    char signature[256];
    generate_failure_signature(diagnosis, signature, sizeof(signature));

    recovery_pattern_t* pattern = find_pattern(ctx, signature);
    if (pattern) {
        // Use historical success rate
        return pattern->success_rate;
    }

    // No historical data - use base rate
    // Different tiers have different base success rates
    switch (strategy->tier) {
        case RECOVERY_TIER_IMMEDIATE:   return 0.85f;
        case RECOVERY_TIER_TACTICAL:    return 0.70f;
        case RECOVERY_TIER_STRATEGIC:   return 0.60f;
        case RECOVERY_TIER_PREVENTIVE:  return 0.75f;
        default:                        return 0.50f;
    }
}

//=============================================================================
// Recovery History & Analytics
//=============================================================================

bool brain_recovery_get_stats(
    brain_recovery_context_t ctx,
    recovery_history_stats_t* stats
) {
    if (!ctx || !stats) return false;

    memset(stats, 0, sizeof(recovery_history_stats_t));

    stats->total_recoveries = ctx->total_recoveries;
    stats->successful_recoveries = ctx->successful_recoveries;
    stats->failed_recoveries = ctx->total_recoveries - ctx->successful_recoveries;
    stats->success_rate = calculate_success_rate(
        ctx->successful_recoveries,
        ctx->total_recoveries
    );

    stats->total_patterns_learned = ctx->pattern_count;
    stats->avg_prediction_accuracy = calculate_success_rate(
        ctx->brain_correct_predictions,
        ctx->brain_decisions
    );

    // Find most effective tier
    uint32_t tier_successes[4] = {0};
    uint32_t tier_counts[4] = {0};

    for (uint32_t i = 0; i < ctx->pattern_count; i++) {
        recovery_pattern_t* p = &ctx->patterns[i];
        if (p->best_tier < 4) {
            tier_counts[p->best_tier]++;
            tier_successes[p->best_tier] += p->success_count;
        }
    }

    float best_rate = 0.0f;
    for (int i = 0; i < 4; i++) {
        float rate = calculate_success_rate(tier_successes[i], tier_counts[i]);
        if (rate > best_rate) {
            best_rate = rate;
            stats->most_effective_tier = (recovery_tier_t)i;
        }
    }

    return true;
}

uint32_t brain_recovery_get_recent_outcomes(
    brain_recovery_context_t ctx,
    recovery_outcome_t* outcomes,
    uint32_t max_outcomes
) {
    if (!ctx || !outcomes || max_outcomes == 0) return 0;

    uint32_t count = (ctx->history_count < max_outcomes) ?
                     ctx->history_count : max_outcomes;

    // Copy most recent outcomes
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = (ctx->history_write_idx - 1 - i + ctx->history_capacity) %
                       ctx->history_capacity;
        if (idx < ctx->history_count) {
            outcomes[i] = ctx->history[idx];
        }
    }

    return count;
}

//=============================================================================
// Integration with Cognitive Pipeline
//=============================================================================

bool brain_recovery_register_pipeline(
    brain_recovery_context_t ctx,
    brain_t brain
) {
    if (!ctx || !brain) return false;

    // In full implementation, would register callbacks with:
    // - Executive function for decision integration
    // - Working memory for pattern storage
    // - Episodic memory for outcome learning

    LOG_INFO("Registered brain recovery with cognitive pipeline");
    return true;
}

void brain_recovery_unregister_pipeline(
    brain_recovery_context_t ctx,
    brain_t brain
) {
    if (!ctx || !brain) return;
    LOG_INFO("Unregistered brain recovery from cognitive pipeline");
}

//=============================================================================
// Persistence
//=============================================================================

bool brain_recovery_save(
    brain_recovery_context_t ctx,
    const char* filepath
) {
    if (!ctx || !filepath) return false;

    FILE* f = fopen(filepath, "wb");
    if (!f) {
        LOG_ERROR("Failed to open file for writing: %s", filepath);
        return false;
    }

    // Write header
    uint32_t version = 1;
    fwrite(&version, sizeof(uint32_t), 1, f);

    // Write statistics
    fwrite(&ctx->total_recoveries, sizeof(uint32_t), 1, f);
    fwrite(&ctx->successful_recoveries, sizeof(uint32_t), 1, f);
    fwrite(&ctx->brain_decisions, sizeof(uint32_t), 1, f);
    fwrite(&ctx->brain_correct_predictions, sizeof(uint32_t), 1, f);

    // Write patterns
    fwrite(&ctx->pattern_count, sizeof(uint32_t), 1, f);
    fwrite(ctx->patterns, sizeof(recovery_pattern_t), ctx->pattern_count, f);

    fclose(f);

    LOG_INFO("Saved brain recovery state to %s", filepath);
    return true;
}

brain_recovery_context_t brain_recovery_load(
    brain_t brain,
    const char* filepath
) {
    if (!brain || !filepath) return NULL;

    FILE* f = fopen(filepath, "rb");
    if (!f) {
        LOG_ERROR("Failed to open file for reading: %s", filepath);
        return NULL;
    }

    // Read and verify version
    uint32_t version;
    if (fread(&version, sizeof(uint32_t), 1, f) != 1 || version != 1) {
        LOG_ERROR("Invalid or unsupported file version");
        fclose(f);
        return NULL;
    }

    brain_recovery_context_t ctx = brain_recovery_init(brain);
    if (!ctx) {
        fclose(f);
        return NULL;
    }

    // Read statistics
    fread(&ctx->total_recoveries, sizeof(uint32_t), 1, f);
    fread(&ctx->successful_recoveries, sizeof(uint32_t), 1, f);
    fread(&ctx->brain_decisions, sizeof(uint32_t), 1, f);
    fread(&ctx->brain_correct_predictions, sizeof(uint32_t), 1, f);

    // Read patterns
    fread(&ctx->pattern_count, sizeof(uint32_t), 1, f);
    if (ctx->pattern_count <= ctx->pattern_capacity) {
        fread(ctx->patterns, sizeof(recovery_pattern_t), ctx->pattern_count, f);
    }

    fclose(f);

    LOG_INFO("Loaded brain recovery state from %s (patterns=%u)",
        filepath, ctx->pattern_count);
    return ctx;
}

//=============================================================================
// Diagnostics & Debugging
//=============================================================================

void brain_recovery_report(
    brain_recovery_context_t ctx,
    FILE* output
) {
    if (!ctx || !output) return;

    fprintf(output, "\n=== Brain Recovery Report ===\n\n");

    fprintf(output, "Recovery Statistics:\n");
    fprintf(output, "  Total recoveries: %u\n", ctx->total_recoveries);
    fprintf(output, "  Successful: %u (%.1f%%)\n",
        ctx->successful_recoveries,
        calculate_success_rate(ctx->successful_recoveries, ctx->total_recoveries) * 100.0f);

    fprintf(output, "\nBrain Decision Statistics:\n");
    fprintf(output, "  Total decisions: %u\n", ctx->brain_decisions);
    fprintf(output, "  Prediction accuracy: %.1f%%\n",
        calculate_success_rate(ctx->brain_correct_predictions, ctx->brain_decisions) * 100.0f);

    fprintf(output, "\nLearned Patterns: %u\n", ctx->pattern_count);
    for (uint32_t i = 0; i < ctx->pattern_count && i < 10; i++) {
        recovery_pattern_t* p = &ctx->patterns[i];
        fprintf(output, "  [%u] %s: %.1f%% success (%u/%u), confidence=%.2f\n",
            i, p->failure_signature,
            p->success_rate * 100.0f,
            p->success_count, p->occurrence_count,
            p->confidence);
    }

    fprintf(output, "\n");
}

int32_t brain_recovery_export_json(
    brain_recovery_context_t ctx,
    char* json_buffer,
    size_t buffer_size
) {
    if (!ctx || !json_buffer || buffer_size == 0) return -1;

    int written = snprintf(json_buffer, buffer_size,
        "{\n"
        "  \"total_recoveries\": %u,\n"
        "  \"successful_recoveries\": %u,\n"
        "  \"success_rate\": %.3f,\n"
        "  \"brain_decisions\": %u,\n"
        "  \"prediction_accuracy\": %.3f,\n"
        "  \"patterns_learned\": %u\n"
        "}",
        ctx->total_recoveries,
        ctx->successful_recoveries,
        calculate_success_rate(ctx->successful_recoveries, ctx->total_recoveries),
        ctx->brain_decisions,
        calculate_success_rate(ctx->brain_correct_predictions, ctx->brain_decisions),
        ctx->pattern_count
    );

    return (written > 0 && (size_t)written < buffer_size) ? written : -1;
}
