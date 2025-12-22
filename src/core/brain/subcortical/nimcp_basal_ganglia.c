//=============================================================================
// nimcp_basal_ganglia.c - Basal Ganglia Implementation
//=============================================================================
/**
 * @file nimcp_basal_ganglia.c
 * @brief Main basal ganglia implementation integrating all nuclei
 *
 * WHAT: Basal ganglia model for action selection and habit learning
 * WHY:  Provides biologically-inspired action selection mechanism
 * HOW:  Integrates striatum, GP, STN, and SN for pathway-based competition
 *
 * @author NIMCP Development Team
 * @date 2025-12-22
 */

#include "core/brain/subcortical/nimcp_basal_ganglia.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "async/nimcp_bio_router.h"

#include <string.h>
#include <stdio.h>
#include <math.h>

//=============================================================================
// Internal Helpers
//=============================================================================

/**
 * @brief Find action with highest thalamic output
 */
static uint32_t find_winning_action(const basal_ganglia_t* bg) {
    uint32_t winner = 0;
    float max_val = bg->thalamic_output[0];

    for (uint32_t a = 1; a < bg->num_actions; a++) {
        if (bg->thalamic_output[a] > max_val) {
            max_val = bg->thalamic_output[a];
            winner = a;
        }
    }

    return winner;
}

/**
 * @brief Compute conflict level between competing actions
 */
static float compute_conflict(const basal_ganglia_t* bg) {
    if (bg->num_actions < 2) return 0.0f;

    /* Find top two actions */
    float first = 0, second = 0;
    for (uint32_t a = 0; a < bg->num_actions; a++) {
        float val = bg->thalamic_output[a];
        if (val > first) {
            second = first;
            first = val;
        } else if (val > second) {
            second = val;
        }
    }

    /* Conflict is high when top two are close */
    if (first < 0.001f) return 0.0f;
    return 1.0f - (first - second) / first;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

void basal_ganglia_default_config(basal_ganglia_config_t* config) {
    if (!config) return;

    config->num_actions = 8;
    config->dopamine_baseline = BG_DOPAMINE_BASELINE;
    config->action_threshold = BG_ACTION_THRESHOLD;
    config->habit_learning_rate = 0.01f;
    config->exploration_rate = 0.1f;
    config->enable_hyperdirect = true;
    config->enable_habit_learning = true;

    /* Component configurations */
    striatum_default_config(&config->striatum_config);
    config->striatum_config.num_actions = config->num_actions;

    globus_pallidus_default_config(&config->gpi_config, GP_SEGMENT_INTERNAL);
    config->gpi_config.num_actions = config->num_actions;

    globus_pallidus_default_config(&config->gpe_config, GP_SEGMENT_EXTERNAL);
    config->gpe_config.num_actions = config->num_actions;

    subthalamic_default_config(&config->stn_config);
    config->stn_config.num_actions = config->num_actions;

    substantia_nigra_default_config(&config->snc_config, SN_PART_COMPACTA);

    substantia_nigra_default_config(&config->snr_config, SN_PART_RETICULATA);
    config->snr_config.num_actions = config->num_actions;
}

basal_ganglia_t* basal_ganglia_create(const basal_ganglia_config_t* config) {
    basal_ganglia_config_t cfg;
    if (config) {
        cfg = *config;
    } else {
        basal_ganglia_default_config(&cfg);
    }

    /* Ensure all child configs have matching num_actions */
    cfg.striatum_config.num_actions = cfg.num_actions;
    cfg.gpi_config.num_actions = cfg.num_actions;
    cfg.gpe_config.num_actions = cfg.num_actions;
    cfg.stn_config.num_actions = cfg.num_actions;
    cfg.snr_config.num_actions = cfg.num_actions;

    basal_ganglia_t* bg = nimcp_malloc(sizeof(basal_ganglia_t));
    if (!bg) {
        NIMCP_LOGGING_ERROR("Failed to allocate basal ganglia");
        return NULL;
    }
    memset(bg, 0, sizeof(basal_ganglia_t));

    bg->config = cfg;
    bg->num_actions = cfg.num_actions;
    bg->dopamine_level = cfg.dopamine_baseline;
    bg->max_habits = BG_MAX_HABITS;

    /* Create striatum */
    bg->striatum = striatum_create(&cfg.striatum_config);
    if (!bg->striatum) {
        NIMCP_LOGGING_ERROR("Failed to create striatum");
        nimcp_free(bg);
        return NULL;
    }

    /* Create GPi */
    bg->gpi = globus_pallidus_create(&cfg.gpi_config);
    if (!bg->gpi) {
        NIMCP_LOGGING_ERROR("Failed to create GPi");
        striatum_destroy(bg->striatum);
        nimcp_free(bg);
        return NULL;
    }

    /* Create GPe */
    bg->gpe = globus_pallidus_create(&cfg.gpe_config);
    if (!bg->gpe) {
        NIMCP_LOGGING_ERROR("Failed to create GPe");
        globus_pallidus_destroy(bg->gpi);
        striatum_destroy(bg->striatum);
        nimcp_free(bg);
        return NULL;
    }

    /* Create STN */
    bg->stn = subthalamic_create(&cfg.stn_config);
    if (!bg->stn) {
        NIMCP_LOGGING_ERROR("Failed to create STN");
        globus_pallidus_destroy(bg->gpe);
        globus_pallidus_destroy(bg->gpi);
        striatum_destroy(bg->striatum);
        nimcp_free(bg);
        return NULL;
    }

    /* Create SNc (dopamine) */
    bg->snc = substantia_nigra_create(&cfg.snc_config);
    if (!bg->snc) {
        NIMCP_LOGGING_ERROR("Failed to create SNc");
        subthalamic_destroy(bg->stn);
        globus_pallidus_destroy(bg->gpe);
        globus_pallidus_destroy(bg->gpi);
        striatum_destroy(bg->striatum);
        nimcp_free(bg);
        return NULL;
    }

    /* Create SNr (output) */
    bg->snr = substantia_nigra_create(&cfg.snr_config);
    if (!bg->snr) {
        NIMCP_LOGGING_ERROR("Failed to create SNr");
        substantia_nigra_destroy(bg->snc);
        subthalamic_destroy(bg->stn);
        globus_pallidus_destroy(bg->gpe);
        globus_pallidus_destroy(bg->gpi);
        striatum_destroy(bg->striatum);
        nimcp_free(bg);
        return NULL;
    }

    /* Allocate action candidates */
    bg->actions = nimcp_calloc(cfg.num_actions, sizeof(action_candidate_t));
    if (!bg->actions) {
        NIMCP_LOGGING_ERROR("Failed to allocate action candidates");
        goto cleanup;
    }

    /* Initialize action candidates */
    for (uint32_t a = 0; a < cfg.num_actions; a++) {
        bg->actions[a].action_id = a;
        snprintf(bg->actions[a].name, sizeof(bg->actions[a].name), "Action_%u", a);
    }

    /* Allocate pathway arrays */
    bg->direct_pathway = nimcp_calloc(cfg.num_actions, sizeof(float));
    bg->indirect_pathway = nimcp_calloc(cfg.num_actions, sizeof(float));
    bg->hyperdirect_pathway = nimcp_calloc(cfg.num_actions, sizeof(float));
    bg->thalamic_output = nimcp_calloc(cfg.num_actions, sizeof(float));

    if (!bg->direct_pathway || !bg->indirect_pathway ||
        !bg->hyperdirect_pathway || !bg->thalamic_output) {
        NIMCP_LOGGING_ERROR("Failed to allocate pathway arrays");
        goto cleanup;
    }

    /* Allocate habits */
    bg->habits = nimcp_calloc(BG_MAX_HABITS, sizeof(habit_t));
    if (!bg->habits) {
        NIMCP_LOGGING_ERROR("Failed to allocate habits");
        goto cleanup;
    }

    /* Allocate mutex */
    bg->mutex = nimcp_malloc(sizeof(nimcp_mutex_t));
    if (!bg->mutex) {
        NIMCP_LOGGING_ERROR("Failed to allocate BG mutex");
        goto cleanup;
    }
    nimcp_mutex_init(bg->mutex, NULL);

    NIMCP_LOGGING_DEBUG("Created basal ganglia with %u actions", cfg.num_actions);

    return bg;

cleanup:
    if (bg->actions) nimcp_free(bg->actions);
    if (bg->direct_pathway) nimcp_free(bg->direct_pathway);
    if (bg->indirect_pathway) nimcp_free(bg->indirect_pathway);
    if (bg->hyperdirect_pathway) nimcp_free(bg->hyperdirect_pathway);
    if (bg->thalamic_output) nimcp_free(bg->thalamic_output);
    if (bg->habits) nimcp_free(bg->habits);
    substantia_nigra_destroy(bg->snr);
    substantia_nigra_destroy(bg->snc);
    subthalamic_destroy(bg->stn);
    globus_pallidus_destroy(bg->gpe);
    globus_pallidus_destroy(bg->gpi);
    striatum_destroy(bg->striatum);
    nimcp_free(bg);
    return NULL;
}

void basal_ganglia_destroy(basal_ganglia_t* bg) {
    if (!bg) return;

    nimcp_mutex_lock(bg->mutex);

    /* Destroy nuclei */
    striatum_destroy(bg->striatum);
    globus_pallidus_destroy(bg->gpi);
    globus_pallidus_destroy(bg->gpe);
    subthalamic_destroy(bg->stn);
    substantia_nigra_destroy(bg->snc);
    substantia_nigra_destroy(bg->snr);

    /* Free arrays */
    nimcp_free(bg->actions);
    nimcp_free(bg->direct_pathway);
    nimcp_free(bg->indirect_pathway);
    nimcp_free(bg->hyperdirect_pathway);
    nimcp_free(bg->thalamic_output);
    nimcp_free(bg->habits);

    nimcp_mutex_unlock(bg->mutex);
    nimcp_mutex_destroy(bg->mutex);
    nimcp_free(bg->mutex);

    nimcp_free(bg);

    NIMCP_LOGGING_DEBUG("Destroyed basal ganglia");
}

int basal_ganglia_reset(basal_ganglia_t* bg) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);

    /* Reset nuclei */
    striatum_reset(bg->striatum);
    globus_pallidus_reset(bg->gpi);
    globus_pallidus_reset(bg->gpe);
    subthalamic_reset(bg->stn);
    substantia_nigra_reset(bg->snc);
    substantia_nigra_reset(bg->snr);

    /* Reset state */
    bg->dopamine_level = bg->config.dopamine_baseline;
    bg->selected_action = 0;
    bg->action_state = ACTION_STATE_IDLE;
    bg->mode = BG_MODE_GOAL_DIRECTED;
    bg->conflict_level = 0;
    bg->habit_mode = false;

    /* Clear arrays */
    memset(bg->direct_pathway, 0, bg->num_actions * sizeof(float));
    memset(bg->indirect_pathway, 0, bg->num_actions * sizeof(float));
    memset(bg->hyperdirect_pathway, 0, bg->num_actions * sizeof(float));
    memset(bg->thalamic_output, 0, bg->num_actions * sizeof(float));

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

//=============================================================================
// Action Selection Functions
//=============================================================================

int basal_ganglia_set_action_value(basal_ganglia_t* bg, uint32_t action_id,
                                    float value, float urgency, float cost) {
    if (!bg || action_id >= bg->num_actions) return -1;

    nimcp_mutex_lock(bg->mutex);

    bg->actions[action_id].value = value;
    bg->actions[action_id].urgency = fmaxf(0.0f, fminf(1.0f, urgency));
    bg->actions[action_id].cost = fmaxf(0.0f, fminf(1.0f, cost));

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

int basal_ganglia_select_action(basal_ganglia_t* bg,
                                 const float* cortical_input,
                                 uint32_t* selected_action) {
    if (!bg || !cortical_input || !selected_action) return -1;

    nimcp_mutex_lock(bg->mutex);

    bg->action_state = ACTION_STATE_COMPETING;
    uint64_t start_time = nimcp_time_get_ms();

    /* 1. Process through striatum (D1 and D2 pathways) */
    striatum_process_input(bg->striatum, cortical_input, bg->dopamine_level);
    striatum_get_d1_output(bg->striatum, bg->direct_pathway);
    striatum_get_d2_output(bg->striatum, bg->indirect_pathway);

    /* 2. Process indirect pathway: D2 → GPe */
    globus_pallidus_set_striatal_input(bg->gpe, bg->indirect_pathway);
    globus_pallidus_process(bg->gpe);

    /* 3. Get GPe output for STN */
    float gpe_output[BG_MAX_ACTIONS];
    globus_pallidus_get_output(bg->gpe, gpe_output);

    /* 4. Process STN (receives GPe and optional hyperdirect) */
    subthalamic_set_gpe_input(bg->stn, gpe_output);
    if (bg->config.enable_hyperdirect) {
        /* Send cortical input as hyperdirect signal */
        subthalamic_set_cortical_input(bg->stn, cortical_input, false);
    }
    subthalamic_process(bg->stn);

    /* 5. Get STN output */
    float stn_output[BG_MAX_ACTIONS];
    subthalamic_get_output(bg->stn, stn_output);

    /* 6. Process GPi (receives D1 direct and STN) */
    globus_pallidus_set_striatal_input(bg->gpi, bg->direct_pathway);
    globus_pallidus_set_stn_input(bg->gpi, stn_output);
    globus_pallidus_set_gpe_input(bg->gpi, gpe_output);
    globus_pallidus_process(bg->gpi);

    /* 7. Also process SNr similarly */
    snr_set_striatal_input(bg->snr, bg->direct_pathway);
    snr_set_stn_input(bg->snr, stn_output);
    snr_process(bg->snr);

    /* 8. Compute thalamic output (disinhibition) */
    /* Low GPi/SNr output = high thalamic activity = action enabled */
    float gpi_output[BG_MAX_ACTIONS];
    float snr_output[BG_MAX_ACTIONS];
    globus_pallidus_get_output(bg->gpi, gpi_output);
    snr_get_output(bg->snr, snr_output);

    for (uint32_t a = 0; a < bg->num_actions; a++) {
        /* Average of GPi and SNr inhibition */
        float inhibition = (gpi_output[a] + snr_output[a]) / 2.0f;
        /* Disinhibition = 1 - inhibition */
        bg->thalamic_output[a] = 1.0f - inhibition;
    }

    /* 9. Find winning action */
    uint32_t winner = find_winning_action(bg);
    bg->selected_action = winner;

    /* 10. Check if above threshold */
    if (bg->thalamic_output[winner] >= bg->config.action_threshold) {
        bg->action_state = ACTION_STATE_SELECTED;
        *selected_action = winner;
    } else {
        bg->action_state = ACTION_STATE_IDLE;
        *selected_action = winner;  /* Return best candidate anyway */
    }

    /* 11. Compute conflict level */
    bg->conflict_level = compute_conflict(bg);

    /* 12. Determine operating mode */
    if (bg->conflict_level > 0.7f) {
        bg->mode = BG_MODE_GOAL_DIRECTED;  /* High conflict = need deliberation */
    } else if (bg->habit_mode && bg->conflict_level < 0.3f) {
        bg->mode = BG_MODE_HABITUAL;
    }

    /* Update statistics */
    uint64_t end_time = nimcp_time_get_ms();
    bg->stats.total_selections++;
    float selection_time = (float)(end_time - start_time);
    float count = (float)bg->stats.total_selections;
    bg->stats.avg_selection_time_ms =
        (bg->stats.avg_selection_time_ms * (count - 1.0f) + selection_time) / count;

    if (bg->mode == BG_MODE_GOAL_DIRECTED) bg->stats.goal_directed_count++;
    else if (bg->mode == BG_MODE_HABITUAL) bg->stats.habitual_count++;

    if (bg->conflict_level > 0.5f) bg->stats.conflict_ratio++;

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

int basal_ganglia_get_thalamic_output(const basal_ganglia_t* bg, float* output) {
    if (!bg || !output) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    memcpy(output, bg->thalamic_output, bg->num_actions * sizeof(float));
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return 0;
}

int basal_ganglia_suppress_action(basal_ganglia_t* bg, float suppression_strength) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);

    /* Activate STN emergency stop */
    subthalamic_emergency_stop(bg->stn, suppression_strength);
    bg->mode = BG_MODE_SUPPRESSED;
    bg->action_state = ACTION_STATE_CANCELLED;
    bg->stats.suppression_count++;

    NIMCP_LOGGING_DEBUG("Suppressed action with strength %.2f", suppression_strength);

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

int basal_ganglia_action_completed(basal_ganglia_t* bg, uint32_t action_id,
                                    bool success) {
    if (!bg || action_id >= bg->num_actions) return -1;

    nimcp_mutex_lock(bg->mutex);

    bg->action_state = ACTION_STATE_COMPLETED;

    /* Update habit strength if this was a habit */
    if (bg->actions[action_id].is_habit) {
        for (uint32_t h = 0; h < bg->num_habits; h++) {
            if (bg->habits[h].action_id == action_id) {
                basal_ganglia_strengthen_habit(bg, h, success);
                break;
            }
        }
    }

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

//=============================================================================
// Dopamine Modulation Functions
//=============================================================================

int basal_ganglia_update_dopamine(basal_ganglia_t* bg, float reward,
                                   float expected_reward) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);

    /* Update SNc dopamine based on reward */
    snc_update_reward(bg->snc, reward, expected_reward);

    /* Get new dopamine level */
    bg->dopamine_level = snc_get_dopamine(bg->snc);
    bg->reward_prediction_error = snc_get_rpe(bg->snc);

    /* Determine dopamine signal type */
    if (bg->reward_prediction_error > 0.1f) {
        bg->dopamine_signal = DOPAMINE_SIGNAL_REWARD;
    } else if (bg->reward_prediction_error < -0.1f) {
        bg->dopamine_signal = DOPAMINE_SIGNAL_PUNISHMENT;
    } else {
        bg->dopamine_signal = DOPAMINE_SIGNAL_TONIC;
    }

    /* Update striatum with new dopamine */
    striatum_set_dopamine(bg->striatum, bg->dopamine_level);

    /* Update statistics */
    float count = (float)(bg->stats.total_selections + 1);
    bg->stats.avg_dopamine_level =
        (bg->stats.avg_dopamine_level * (count - 1.0f) + bg->dopamine_level) / count;

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

int basal_ganglia_set_dopamine(basal_ganglia_t* bg, float level) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);
    bg->dopamine_level = fmaxf(0.0f, fminf(1.0f, level));
    striatum_set_dopamine(bg->striatum, bg->dopamine_level);
    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

float basal_ganglia_get_dopamine(const basal_ganglia_t* bg) {
    if (!bg) return 0.5f;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    float da = bg->dopamine_level;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return da;
}

float basal_ganglia_get_rpe(const basal_ganglia_t* bg) {
    if (!bg) return 0.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    float rpe = bg->reward_prediction_error;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return rpe;
}

//=============================================================================
// Habit Learning Functions
//=============================================================================

int basal_ganglia_register_habit(basal_ganglia_t* bg, uint32_t context,
                                  uint32_t action_id, uint32_t* habit_id) {
    if (!bg || !habit_id || action_id >= bg->num_actions) return -1;

    nimcp_mutex_lock(bg->mutex);

    if (bg->num_habits >= bg->max_habits) {
        NIMCP_LOGGING_WARN("Maximum habits reached");
        nimcp_mutex_unlock(bg->mutex);
        return -2;
    }

    uint32_t id = bg->num_habits;
    bg->habits[id].habit_id = id;
    bg->habits[id].trigger_context = context;
    bg->habits[id].action_id = action_id;
    bg->habits[id].strength = 0.1f;  /* Initial strength */
    bg->habits[id].repetitions = 0;
    bg->habits[id].last_executed_ms = 0;
    bg->habits[id].success_rate = 0.0f;

    bg->actions[action_id].is_habit = true;
    bg->actions[action_id].habit_strength = 0.1f;

    bg->num_habits++;
    *habit_id = id;

    NIMCP_LOGGING_DEBUG("Registered habit %u: context %u -> action %u",
                        id, context, action_id);

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

int basal_ganglia_strengthen_habit(basal_ganglia_t* bg, uint32_t habit_id,
                                    bool success) {
    if (!bg || habit_id >= bg->num_habits) return -1;

    nimcp_mutex_lock(bg->mutex);

    habit_t* habit = &bg->habits[habit_id];

    /* Update repetitions */
    habit->repetitions++;
    habit->last_executed_ms = nimcp_time_get_ms();

    /* Update success rate */
    float count = (float)habit->repetitions;
    habit->success_rate =
        (habit->success_rate * (count - 1.0f) + (success ? 1.0f : 0.0f)) / count;

    /* Strengthen or weaken based on success */
    float delta = bg->config.habit_learning_rate;
    if (success) {
        habit->strength = fminf(1.0f, habit->strength + delta);
    } else {
        habit->strength = fmaxf(0.0f, habit->strength - delta * 0.5f);
    }

    /* Update action's habit strength */
    bg->actions[habit->action_id].habit_strength = habit->strength;

    /* Check if habit is now strong enough for automatic mode */
    if (habit->strength > BG_HABIT_STRENGTH_THRESHOLD) {
        NIMCP_LOGGING_DEBUG("Habit %u reached threshold strength %.2f",
                            habit_id, habit->strength);
    }

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

bool basal_ganglia_check_habit(const basal_ganglia_t* bg, uint32_t context,
                                uint32_t* action_id) {
    if (!bg || !action_id) return false;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);

    for (uint32_t h = 0; h < bg->num_habits; h++) {
        if (bg->habits[h].trigger_context == context &&
            bg->habits[h].strength > BG_HABIT_STRENGTH_THRESHOLD) {
            *action_id = bg->habits[h].action_id;
            nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);
            return true;
        }
    }

    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);
    return false;
}

float basal_ganglia_get_habit_strength(const basal_ganglia_t* bg,
                                        uint32_t habit_id) {
    if (!bg || habit_id >= bg->num_habits) return -1.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    float strength = bg->habits[habit_id].strength;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return strength;
}

int basal_ganglia_set_habit_mode(basal_ganglia_t* bg, bool habitual) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);
    bg->habit_mode = habitual;
    bg->mode = habitual ? BG_MODE_HABITUAL : BG_MODE_GOAL_DIRECTED;
    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

bool basal_ganglia_is_habit_mode(const basal_ganglia_t* bg) {
    if (!bg) return false;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    bool mode = bg->habit_mode;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return mode;
}

//=============================================================================
// Pathway Analysis Functions
//=============================================================================

float basal_ganglia_get_direct_activation(const basal_ganglia_t* bg,
                                           uint32_t action_id) {
    if (!bg || action_id >= bg->num_actions) return -1.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    float activation = bg->direct_pathway[action_id];
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return activation;
}

float basal_ganglia_get_indirect_activation(const basal_ganglia_t* bg,
                                             uint32_t action_id) {
    if (!bg || action_id >= bg->num_actions) return -1.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    float activation = bg->indirect_pathway[action_id];
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return activation;
}

float basal_ganglia_get_conflict(const basal_ganglia_t* bg) {
    if (!bg) return 0.0f;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    float conflict = bg->conflict_level;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return conflict;
}

bg_operating_mode_t basal_ganglia_get_mode(const basal_ganglia_t* bg) {
    if (!bg) return BG_MODE_GOAL_DIRECTED;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    bg_operating_mode_t mode = bg->mode;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return mode;
}

//=============================================================================
// Update Functions
//=============================================================================

int basal_ganglia_step(basal_ganglia_t* bg, float dt) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);

    /* Step all nuclei */
    striatum_step(bg->striatum, dt);
    globus_pallidus_step(bg->gpi, dt);
    globus_pallidus_step(bg->gpe, dt);
    subthalamic_step(bg->stn, dt);
    snc_step(bg->snc, dt);
    snr_step(bg->snr, dt);

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

int basal_ganglia_process_input(basal_ganglia_t* bg, const float* cortical_input) {
    if (!bg || !cortical_input) return -1;

    uint32_t selected;
    return basal_ganglia_select_action(bg, cortical_input, &selected);
}

//=============================================================================
// Bio-Async Integration
//=============================================================================

int basal_ganglia_connect_bio_async(basal_ganglia_t* bg) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);

    if (bg->bio_async_enabled) {
        nimcp_mutex_unlock(bg->mutex);
        return 0;
    }

    bio_module_info_t info = {
        .module_id = BIO_MODULE_BASAL_GANGLIA,
        .module_name = "basal_ganglia",
        .inbox_capacity = 32,
        .user_data = bg
    };

    bg->bio_ctx = bio_router_register_module(&info);
    if (bg->bio_ctx) {
        bg->bio_async_enabled = true;
        NIMCP_LOGGING_INFO("Basal ganglia connected to bio-async router");
    }

    nimcp_mutex_unlock(bg->mutex);

    return bg->bio_async_enabled ? 0 : -1;
}

int basal_ganglia_disconnect_bio_async(basal_ganglia_t* bg) {
    if (!bg) return -1;

    nimcp_mutex_lock(bg->mutex);

    if (bg->bio_async_enabled && bg->bio_ctx) {
        bio_router_unregister_module(bg->bio_ctx);
        bg->bio_ctx = NULL;
        bg->bio_async_enabled = false;
    }

    nimcp_mutex_unlock(bg->mutex);

    return 0;
}

bool basal_ganglia_is_bio_async_connected(const basal_ganglia_t* bg) {
    if (!bg) return false;
    return bg->bio_async_enabled;
}

//=============================================================================
// Statistics and Debugging
//=============================================================================

int basal_ganglia_get_stats(const basal_ganglia_t* bg,
                             basal_ganglia_stats_t* stats) {
    if (!bg || !stats) return -1;

    nimcp_mutex_lock((nimcp_mutex_t*)bg->mutex);
    *stats = bg->stats;
    nimcp_mutex_unlock((nimcp_mutex_t*)bg->mutex);

    return 0;
}

const char* basal_ganglia_mode_name(bg_operating_mode_t mode) {
    switch (mode) {
        case BG_MODE_GOAL_DIRECTED: return "Goal-Directed";
        case BG_MODE_HABITUAL: return "Habitual";
        case BG_MODE_EXPLORATORY: return "Exploratory";
        case BG_MODE_SUPPRESSED: return "Suppressed";
        default: return "Unknown";
    }
}

const char* basal_ganglia_action_state_name(action_state_t state) {
    switch (state) {
        case ACTION_STATE_IDLE: return "Idle";
        case ACTION_STATE_COMPETING: return "Competing";
        case ACTION_STATE_SELECTED: return "Selected";
        case ACTION_STATE_EXECUTING: return "Executing";
        case ACTION_STATE_COMPLETED: return "Completed";
        case ACTION_STATE_CANCELLED: return "Cancelled";
        default: return "Unknown";
    }
}
