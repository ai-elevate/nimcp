//=============================================================================
// nimcp_gt_neuromod.c - Game Payoff to Neuromodulator Mapping
//=============================================================================
/**
 * @file nimcp_gt_neuromod.c
 * @brief Map game-theoretic outcomes to neuromodulator release
 *
 * WHAT: Payoff-based neuromodulator release
 * WHY:  Game outcomes should affect learning and motivation
 * HOW:  Win -> DA, Loss -> 5-HT, Unfair -> NE
 *
 * BIOLOGICAL INSPIRATION:
 * - Dopamine encodes reward prediction error
 * - Serotonin modulates patience and impulse control
 * - Norepinephrine signals threat and arousal
 * - Acetylcholine drives attention and encoding
 *
 * @author NIMCP Development Team
 * @date 2024-12-27
 * @version 1.0.0
 */

#include "cognitive/game_theory/integration/nimcp_gt_neuromod.h"
#include "utils/memory/nimcp_memory.h"
#include "core/nimcp_error.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Constants
//=============================================================================

// Neuromodulator indices
#define NM_DOPAMINE 0
#define NM_SEROTONIN 1
#define NM_NOREPINEPHRINE 2
#define NM_ACETYLCHOLINE 3
#define NUM_NEUROMODULATORS 4

// Default scaling factors
#define DEFAULT_DA_GAIN 1.0f
#define DEFAULT_5HT_GAIN 0.5f
#define DEFAULT_NE_GAIN 0.8f
#define DEFAULT_ACH_GAIN 0.6f

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Opaque bridge structure
 */
struct gt_neuromod_bridge_struct {
    neuromodulator_system_t neuromod;
    gt_neuromod_config_t config;

    // Cumulative release tracking
    gt_neuromod_release_t cumulative;
    uint64_t outcomes_processed;

    bool active;
};

//=============================================================================
// Helper Functions
//=============================================================================

/**
 * @brief Clamp value to [0, 1] range
 */
static float clamp01(float x) {
    if (x < 0.0f) return 0.0f;
    if (x > 1.0f) return 1.0f;
    return x;
}

/**
 * @brief Compute sigmoid for smooth response curve
 */
static float sigmoid(float x, float steepness) {
    return 1.0f / (1.0f + expf(-steepness * x));
}

//=============================================================================
// Lifecycle
//=============================================================================

gt_neuromod_config_t gt_neuromod_default_config(void) {
    gt_neuromod_config_t config = {
        // Dopamine (reward/motivation)
        .payoff_dopamine_gain = DEFAULT_DA_GAIN,
        .win_dopamine_bonus = 0.3f,
        .cooperation_dopamine_bonus = 0.2f,

        // Serotonin (patience/inhibition)
        .loss_serotonin_gain = DEFAULT_5HT_GAIN,
        .unfair_serotonin_boost = 0.4f,
        .timeout_serotonin_gain = 0.3f,

        // Norepinephrine (arousal/threat)
        .competition_ne_baseline = 0.2f,
        .fairness_violation_ne_gain = DEFAULT_NE_GAIN,
        .uncertainty_ne_gain = 0.5f,

        // Acetylcholine (attention/salience)
        .strategic_ach_gain = DEFAULT_ACH_GAIN,
        .novel_opponent_ach_boost = 0.4f,

        // Thresholds
        .payoff_threshold = 0.0f,
        .fairness_threshold = 0.3f,
        .uncertainty_threshold = 0.6f
    };
    return config;
}

gt_neuromod_bridge_t gt_neuromod_bridge_create(
    neuromodulator_system_t neuromod,
    const gt_neuromod_config_t* config
) {
    // neuromod can be NULL for standalone testing
    gt_neuromod_bridge_t bridge = nimcp_calloc(1, sizeof(struct gt_neuromod_bridge_struct));
    if (!bridge) {
        return NULL;
    }

    bridge->neuromod = neuromod;
    bridge->config = config ? *config : gt_neuromod_default_config();

    memset(&bridge->cumulative, 0, sizeof(gt_neuromod_release_t));
    bridge->outcomes_processed = 0;
    bridge->active = true;

    return bridge;
}

void gt_neuromod_bridge_destroy(gt_neuromod_bridge_t bridge) {
    if (!bridge) {
        return;
    }
    nimcp_free(bridge);
}

//=============================================================================
// Core Operations
//=============================================================================

nimcp_error_t gt_neuromod_process_outcome(
    gt_neuromod_bridge_t bridge,
    nimcp_player_id_t player,
    const nimcp_game_outcome_t* outcome,
    gt_neuromod_release_t* release
) {
    if (!bridge || !outcome || !release) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    memset(release, 0, sizeof(gt_neuromod_release_t));

    // Find player's payoff
    float payoff = 0.0f;
    bool player_found = false;
    for (uint32_t i = 0; i < outcome->num_players; i++) {
        if (outcome->player_ids[i] == player) {
            payoff = outcome->payoffs[i];
            player_found = true;
            break;
        }
    }

    if (!player_found) {
        return NIMCP_GT_ERROR_PLAYER_NOT_FOUND;
    }

    const gt_neuromod_config_t* cfg = &bridge->config;

    // Compute dopamine release (positive outcomes)
    if (payoff > cfg->payoff_threshold) {
        release->dopamine_released = (payoff - cfg->payoff_threshold) * cfg->payoff_dopamine_gain;

        // Bonus for winning
        if (outcome->winner_id == player) {
            release->dopamine_released += cfg->win_dopamine_bonus;
        }

        // Bonus for cooperation
        if (outcome->cooperated) {
            release->dopamine_released += cfg->cooperation_dopamine_bonus;
        }
    }

    // Compute serotonin release (negative outcomes, patience)
    if (payoff < cfg->payoff_threshold) {
        float loss = cfg->payoff_threshold - payoff;
        release->serotonin_released = loss * cfg->loss_serotonin_gain;
    }

    // Check fairness and compute NE for unfairness
    float fairness = nimcp_compute_fairness_index(
        outcome->payoffs,
        outcome->num_players
    );

    if (fairness < cfg->fairness_threshold) {
        float unfairness = 1.0f - fairness;
        release->serotonin_released += unfairness * cfg->unfair_serotonin_boost;
        release->norepinephrine_released = unfairness * cfg->fairness_violation_ne_gain;
    }

    // Baseline NE for competitive games
    if (outcome->num_players > 1) {
        release->norepinephrine_released += cfg->competition_ne_baseline;
    }

    // Compute net valence
    float positive = release->dopamine_released;
    float negative = release->serotonin_released + release->norepinephrine_released;
    float total = positive + negative;
    if (total > 0.0f) {
        release->net_valence = (positive - negative) / total;
    } else {
        release->net_valence = 0.0f;
    }

    // Clamp all values to reasonable range
    release->dopamine_released = clamp01(release->dopamine_released);
    release->serotonin_released = clamp01(release->serotonin_released);
    release->norepinephrine_released = clamp01(release->norepinephrine_released);
    release->acetylcholine_released = clamp01(release->acetylcholine_released);

    // Update cumulative
    bridge->cumulative.dopamine_released += release->dopamine_released;
    bridge->cumulative.serotonin_released += release->serotonin_released;
    bridge->cumulative.norepinephrine_released += release->norepinephrine_released;
    bridge->cumulative.acetylcholine_released += release->acetylcholine_released;
    bridge->outcomes_processed++;

    // TODO: Actually release to neuromodulator system when available
    // if (bridge->neuromod) {
    //     neuromodulator_release(bridge->neuromod, NM_DOPAMINE, release->dopamine_released);
    //     neuromodulator_release(bridge->neuromod, NM_SEROTONIN, release->serotonin_released);
    //     neuromodulator_release(bridge->neuromod, NM_NOREPINEPHRINE, release->norepinephrine_released);
    //     neuromodulator_release(bridge->neuromod, NM_ACETYLCHOLINE, release->acetylcholine_released);
    // }

    return NIMCP_SUCCESS;
}

float gt_neuromod_auction_win(
    gt_neuromod_bridge_t bridge,
    float winning_bid,
    float payment
) {
    if (!bridge) {
        return 0.0f;
    }

    // Consumer surplus = bid - payment
    float surplus = winning_bid - payment;
    if (surplus < 0.0f) {
        surplus = 0.0f;
    }

    // DA release proportional to surplus
    float da = surplus * bridge->config.payoff_dopamine_gain + bridge->config.win_dopamine_bonus;
    da = clamp01(da);

    bridge->cumulative.dopamine_released += da;
    bridge->outcomes_processed++;

    return da;
}

float gt_neuromod_bargaining_success(
    gt_neuromod_bridge_t bridge,
    float agreement_value,
    float fairness_index
) {
    if (!bridge) {
        return 0.0f;
    }

    // DA based on value and fairness
    float da = agreement_value * bridge->config.payoff_dopamine_gain;

    // Bonus for fair agreements
    if (fairness_index > 0.8f) {
        da += bridge->config.cooperation_dopamine_bonus;
    }

    da = clamp01(da);

    bridge->cumulative.dopamine_released += da;
    bridge->outcomes_processed++;

    return da;
}

float gt_neuromod_bargaining_failure(
    gt_neuromod_bridge_t bridge,
    uint32_t rounds_taken,
    uint32_t max_rounds
) {
    if (!bridge || max_rounds == 0) {
        return 0.0f;
    }

    // 5-HT release for patience (more rounds = more 5-HT)
    float patience_factor = (float)rounds_taken / max_rounds;
    float serotonin = patience_factor * bridge->config.timeout_serotonin_gain +
                      bridge->config.loss_serotonin_gain;

    serotonin = clamp01(serotonin);

    bridge->cumulative.serotonin_released += serotonin;
    bridge->outcomes_processed++;

    return serotonin;
}

float gt_neuromod_signal_unfairness(
    gt_neuromod_bridge_t bridge,
    float unfairness_magnitude
) {
    if (!bridge) {
        return 0.0f;
    }

    float ne = unfairness_magnitude * bridge->config.fairness_violation_ne_gain;
    ne = clamp01(ne);

    bridge->cumulative.norepinephrine_released += ne;
    bridge->outcomes_processed++;

    return ne;
}

float gt_neuromod_strategic_attention(
    gt_neuromod_bridge_t bridge,
    uint32_t num_options,
    float uncertainty
) {
    if (!bridge) {
        return 0.0f;
    }

    // ACh for strategic complexity
    float option_factor = logf(1.0f + (float)num_options) / logf(10.0f);  // Log scale
    float ach = option_factor * bridge->config.strategic_ach_gain;

    // Extra ACh for uncertainty
    if (uncertainty > bridge->config.uncertainty_threshold) {
        ach += (uncertainty - bridge->config.uncertainty_threshold) *
               bridge->config.uncertainty_ne_gain;
    }

    ach = clamp01(ach);

    bridge->cumulative.acetylcholine_released += ach;
    bridge->outcomes_processed++;

    return ach;
}

float gt_neuromod_shapley_reward(
    gt_neuromod_bridge_t bridge,
    float credit,
    float total_value
) {
    if (!bridge || total_value <= 0.0f) {
        return 0.0f;
    }

    // DA proportional to credit share
    float credit_share = credit / total_value;
    float da = credit_share * bridge->config.payoff_dopamine_gain;

    // Bonus for significant contribution
    if (credit_share > 0.5f) {
        da += bridge->config.win_dopamine_bonus * (credit_share - 0.5f) * 2.0f;
    }

    da = clamp01(da);

    bridge->cumulative.dopamine_released += da;
    bridge->outcomes_processed++;

    return da;
}

//=============================================================================
// Query Functions
//=============================================================================

neuromodulator_system_t gt_neuromod_get_system(const gt_neuromod_bridge_t bridge) {
    return bridge ? bridge->neuromod : NULL;
}

nimcp_error_t gt_neuromod_get_cumulative_release(
    const gt_neuromod_bridge_t bridge,
    gt_neuromod_release_t* cumulative
) {
    if (!bridge || !cumulative) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    *cumulative = bridge->cumulative;

    // Compute average valence
    float positive = cumulative->dopamine_released;
    float negative = cumulative->serotonin_released + cumulative->norepinephrine_released;
    float total = positive + negative;
    if (total > 0.0f) {
        cumulative->net_valence = (positive - negative) / total;
    } else {
        cumulative->net_valence = 0.0f;
    }

    return NIMCP_SUCCESS;
}

void gt_neuromod_reset_stats(gt_neuromod_bridge_t bridge) {
    if (!bridge) {
        return;
    }

    memset(&bridge->cumulative, 0, sizeof(gt_neuromod_release_t));
    bridge->outcomes_processed = 0;
}
