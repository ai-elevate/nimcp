/**
 * @file nimcp_raphe_adapter.c
 * @brief Raphe Adapter implementation with bidirectional training integration
 * @date 2026-01-11
 */

#include "core/brain/regions/raphe/nimcp_raphe_adapter.h"
#include "core/brain/regions/raphe/nimcp_raphe.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

//=============================================================================
#include <stddef.h>  /* for NULL */
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for raphe_adapter module */
static nimcp_health_agent_t* g_raphe_adapter_health_agent = NULL;

/**
 * @brief Set health agent for raphe_adapter heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void raphe_adapter_set_health_agent(nimcp_health_agent_t* agent) {
    g_raphe_adapter_health_agent = agent;
}

/** @brief Send heartbeat from raphe_adapter module */
static inline void raphe_adapter_heartbeat(const char* operation, float progress) {
    if (g_raphe_adapter_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_raphe_adapter_health_agent, operation, progress);
    }
}


/*=============================================================================
 * Internal Structures
 *===========================================================================*/

typedef struct {
    nimcp_raphe_callback_fn callback;
    void* user_data;
} callback_entry_t;

/**
 * @brief Internal adapter structure
 */
struct nimcp_raphe_adapter {
    bool is_active;
    nimcp_raphe_adapter_config_t config;

    /* Core Raphe system */
    nimcp_raphe_system_t raphe;

    /* Message queue */
    nimcp_raphe_message_t message_queue[RAPHE_ADAPTER_MSG_QUEUE_SIZE];
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;

    /* Callbacks */
    callback_entry_t callbacks[RAPHE_MSG_COUNT][RAPHE_ADAPTER_MAX_CALLBACKS];
    uint32_t callback_counts[RAPHE_MSG_COUNT];

    /* Training integration */
    nimcp_raphe_training_state_t last_training_state;
    nimcp_raphe_training_modulation_t current_modulation;
    nimcp_raphe_training_callback_fn training_callback;
    void* training_callback_data;
    float training_stress;         /* Accumulated stress from training */
    float training_patience;       /* Remaining patience for training */
    bool training_connected;

    /* Connected systems */
    struct nimcp_brain* brain;
    struct nimcp_bio_router* router;
    struct nimcp_immune_system* immune;
    struct nimcp_training_hub* training_hub;

    /* Statistics */
    uint32_t messages_sent;
    uint32_t messages_received;
    uint32_t updates_processed;
    float total_active_time;
    uint32_t training_events_received;
    uint32_t training_modulations_sent;
};

/*=============================================================================
 * Helper Functions
 *===========================================================================*/

static float clamp_f(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static bool queue_push(struct nimcp_raphe_adapter* adapter, const nimcp_raphe_message_t* msg) {
    if (adapter->queue_count >= RAPHE_ADAPTER_MSG_QUEUE_SIZE) {
        return false;
    }

    adapter->message_queue[adapter->queue_tail] = *msg;
    adapter->queue_tail = (adapter->queue_tail + 1) % RAPHE_ADAPTER_MSG_QUEUE_SIZE;
    adapter->queue_count++;
    return true;
}

static bool queue_pop(struct nimcp_raphe_adapter* adapter, nimcp_raphe_message_t* msg) {
    if (adapter->queue_count == 0) {
        return false;
    }

    *msg = adapter->message_queue[adapter->queue_head];
    adapter->queue_head = (adapter->queue_head + 1) % RAPHE_ADAPTER_MSG_QUEUE_SIZE;
    adapter->queue_count--;
    return true;
}

/*=============================================================================
 * Configuration
 *===========================================================================*/

nimcp_raphe_adapter_config_t nimcp_raphe_adapter_default_config(void) {
    nimcp_raphe_adapter_config_t config = {
        .enable_bio_async = true,
        .auto_create_projections = true,
        .enable_training_integration = true,
        .message_rate_limit = 100.0f,
        .loss_stress_sensitivity = 0.5f,
        .gradient_stress_threshold = 10.0f,
        .patience_decay_rate = 0.01f
    };
    return config;
}

/*=============================================================================
 * Lifecycle API
 *===========================================================================*/

nimcp_raphe_adapter_t nimcp_raphe_adapter_create(const nimcp_raphe_adapter_config_t* config) {
    struct nimcp_raphe_adapter* adapter = calloc(1, sizeof(struct nimcp_raphe_adapter));
    if (!adapter) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;

    }

    /* Apply configuration */
    if (config) {
        adapter->config = *config;
    } else {
        adapter->config = nimcp_raphe_adapter_default_config();
    }

    /* Initialize Raphe system */
    nimcp_raphe_config_t raphe_config = nimcp_raphe_default_config();
    if (nimcp_raphe_init(&adapter->raphe, &raphe_config) != RAPHE_OK) {
        free(adapter);
        return NULL;
    }

    /* Initialize message queue */
    adapter->queue_head = 0;
    adapter->queue_tail = 0;
    adapter->queue_count = 0;

    /* Initialize callbacks */
    for (int i = 0; i < RAPHE_MSG_COUNT; i++) {
        adapter->callback_counts[i] = 0;
    }

    /* Initialize training integration */
    memset(&adapter->last_training_state, 0, sizeof(nimcp_raphe_training_state_t));
    adapter->training_stress = 0.0f;
    adapter->training_patience = 1.0f;  /* Full patience initially */
    adapter->training_callback = NULL;
    adapter->training_callback_data = NULL;
    adapter->training_connected = false;

    /* Initialize modulation to neutral */
    adapter->current_modulation.lr_multiplier = 1.0f;
    adapter->current_modulation.exploration_rate = 0.5f;
    adapter->current_modulation.patience_factor = 0.5f;
    adapter->current_modulation.impulse_inhibition = 0.5f;
    adapter->current_modulation.suggest_consolidation = false;
    adapter->current_modulation.suggest_break = false;

    /* Initialize statistics */
    adapter->messages_sent = 0;
    adapter->messages_received = 0;
    adapter->updates_processed = 0;
    adapter->total_active_time = 0.0f;
    adapter->training_events_received = 0;
    adapter->training_modulations_sent = 0;

    adapter->is_active = true;
    return adapter;
}

void nimcp_raphe_adapter_destroy(nimcp_raphe_adapter_t adapter) {
    if (!adapter) return;

    nimcp_raphe_shutdown(&adapter->raphe);
    free(adapter);
}

/*=============================================================================
 * Connection API
 *===========================================================================*/

int nimcp_raphe_adapter_connect_brain(nimcp_raphe_adapter_t adapter,
                                      struct nimcp_brain* brain) {
    if (!adapter || !brain) return -1;

    adapter->brain = brain;

    /* Auto-create projections if enabled */
    if (adapter->config.auto_create_projections) {
        uint32_t id;
        nimcp_raphe_add_projection(&adapter->raphe, RAPHE_TARGET_PFC, "raphe_pfc", 0.8f, &id);
        nimcp_raphe_add_projection(&adapter->raphe, RAPHE_TARGET_AMYGDALA, "raphe_amygdala", 0.7f, &id);
        nimcp_raphe_add_projection(&adapter->raphe, RAPHE_TARGET_HIPPOCAMPUS, "raphe_hipp", 0.6f, &id);
        nimcp_raphe_add_projection(&adapter->raphe, RAPHE_TARGET_VTA, "raphe_vta", 0.5f, &id);
    }

    return 0;
}

int nimcp_raphe_adapter_disconnect(nimcp_raphe_adapter_t adapter) {
    if (!adapter) return -1;

    adapter->brain = NULL;
    adapter->router = NULL;
    adapter->immune = NULL;
    adapter->training_hub = NULL;
    adapter->training_connected = false;

    return 0;
}

int nimcp_raphe_adapter_set_router(nimcp_raphe_adapter_t adapter,
                                   struct nimcp_bio_router* router) {
    if (!adapter) return -1;

    adapter->router = router;
    return 0;
}

int nimcp_raphe_adapter_connect_immune(nimcp_raphe_adapter_t adapter,
                                       struct nimcp_immune_system* immune) {
    if (!adapter) return -1;

    adapter->immune = immune;
    return 0;
}

int nimcp_raphe_adapter_connect_training(nimcp_raphe_adapter_t adapter,
                                         struct nimcp_training_hub* training_hub) {
    if (!adapter) return -1;

    adapter->training_hub = training_hub;
    adapter->training_connected = (training_hub != NULL);
    return 0;
}

/*=============================================================================
 * Raphe Access API
 *===========================================================================*/

nimcp_raphe_system_t* nimcp_raphe_adapter_get_raphe(nimcp_raphe_adapter_t adapter) {
    if (!adapter) {

        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "adapter is NULL");

        return NULL;

    }
    return &adapter->raphe;
}

/*=============================================================================
 * Messaging API
 *===========================================================================*/

int nimcp_raphe_adapter_send_message(nimcp_raphe_adapter_t adapter,
                                     const nimcp_raphe_message_t* msg) {
    if (!adapter || !msg) return -1;

    if (!queue_push(adapter, msg)) {
        return -1;  /* Queue full */
    }

    adapter->messages_sent++;
    return 0;
}

int nimcp_raphe_adapter_process_messages(nimcp_raphe_adapter_t adapter, int max_messages) {
    if (!adapter) return -1;

    int processed = 0;
    nimcp_raphe_message_t msg;

    while (processed < max_messages && queue_pop(adapter, &msg)) {
        /* Invoke callbacks for this message type */
        if (msg.type < RAPHE_MSG_COUNT) {
            for (uint32_t i = 0; i < adapter->callback_counts[msg.type]; i++) {
                callback_entry_t* entry = &adapter->callbacks[msg.type][i];
                if (entry->callback) {
                    entry->callback(adapter, &msg, entry->user_data);
                }
            }
        }

        adapter->messages_received++;
        processed++;
    }

    return processed;
}

int nimcp_raphe_adapter_register_callback(nimcp_raphe_adapter_t adapter,
                                          nimcp_raphe_msg_type_t type,
                                          nimcp_raphe_callback_fn callback,
                                          void* user_data) {
    if (!adapter || !callback) return -1;
    if (type >= RAPHE_MSG_COUNT) return -1;
    if (adapter->callback_counts[type] >= RAPHE_ADAPTER_MAX_CALLBACKS) return -1;

    uint32_t idx = adapter->callback_counts[type];
    adapter->callbacks[type][idx].callback = callback;
    adapter->callbacks[type][idx].user_data = user_data;
    adapter->callback_counts[type]++;

    return 0;
}

/*=============================================================================
 * Update API
 *===========================================================================*/

int nimcp_raphe_adapter_update(nimcp_raphe_adapter_t adapter, float dt) {
    if (!adapter) return -1;

    float dt_sec = dt / 1000.0f;

    /* Update core Raphe system */
    nimcp_raphe_update(&adapter->raphe, dt);

    /* Update training-related state */
    if (adapter->config.enable_training_integration) {
        /* Decay training stress */
        adapter->training_stress *= (1.0f - 0.05f * dt_sec);

        /* Update training patience based on 5-HT */
        float ht;
        nimcp_raphe_get_5ht(&adapter->raphe, &ht);
        float ht_ratio = ht / 20.0f;  /* Baseline 5-HT */

        /* Higher 5-HT -> more patience */
        float patience_recovery = 0.01f * ht_ratio * dt_sec;
        adapter->training_patience += patience_recovery;
        adapter->training_patience = clamp_f(adapter->training_patience, 0.0f, 1.0f);

        /* Update current modulation based on Raphe state */
        float mood, inhibition, patience_raphe;
        nimcp_raphe_get_mood(&adapter->raphe, &mood);
        nimcp_raphe_get_inhibition(&adapter->raphe, &inhibition);
        nimcp_raphe_get_patience(&adapter->raphe, &patience_raphe);

        /* Learning rate modulation: positive mood -> slightly higher LR */
        adapter->current_modulation.lr_multiplier = 1.0f + mood * 0.2f;
        adapter->current_modulation.lr_multiplier = clamp_f(
            adapter->current_modulation.lr_multiplier, 0.5f, 1.5f);

        /* Exploration: lower 5-HT -> more exploration */
        adapter->current_modulation.exploration_rate = 0.5f - (ht_ratio - 1.0f) * 0.3f;
        adapter->current_modulation.exploration_rate = clamp_f(
            adapter->current_modulation.exploration_rate, 0.1f, 0.9f);

        /* Patience factor from Raphe patience */
        adapter->current_modulation.patience_factor = patience_raphe;

        /* Impulse inhibition from Raphe inhibition */
        adapter->current_modulation.impulse_inhibition = inhibition;

        /* Suggest break if training stress is high and mood is low */
        adapter->current_modulation.suggest_break =
            (adapter->training_stress > 0.7f && mood < -0.3f);

        /* Suggest consolidation if patience is very low */
        adapter->current_modulation.suggest_consolidation =
            (adapter->training_patience < 0.2f);

        /* Notify training layer if callback registered */
        if (adapter->training_callback) {
            adapter->training_callback(&adapter->current_modulation,
                                       adapter->training_callback_data);
            adapter->training_modulations_sent++;
        }
    }

    /* Process pending messages */
    nimcp_raphe_adapter_process_messages(adapter, 10);

    adapter->updates_processed++;
    adapter->total_active_time += dt_sec;

    return 0;
}

/*=============================================================================
 * Training Integration API (BIDIRECTIONAL)
 *===========================================================================*/

int nimcp_raphe_adapter_on_training_event(nimcp_raphe_adapter_t adapter,
                                          nimcp_raphe_train_event_t event,
                                          const nimcp_raphe_training_state_t* state) {
    if (!adapter || !state) return -1;

    /* Store training state */
    adapter->last_training_state = *state;
    adapter->training_events_received++;

    /* Process event -> Raphe effects */
    switch (event) {
        case RAPHE_TRAIN_EVENT_LOSS: {
            /* High loss -> stress -> lower 5-HT */
            float loss_stress = state->current_loss * adapter->config.loss_stress_sensitivity;
            if (state->loss_trend > 0) {
                /* Worsening loss -> more stress */
                loss_stress *= (1.0f + state->loss_trend);
            }
            adapter->training_stress += loss_stress * 0.1f;
            adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);

            /* Apply stress to Raphe mood */
            nimcp_raphe_modulate_anxiety(&adapter->raphe, loss_stress * 0.2f);
            break;
        }

        case RAPHE_TRAIN_EVENT_GRADIENT: {
            /* Large gradients -> stress */
            if (state->gradient_norm > adapter->config.gradient_stress_threshold) {
                float gradient_stress = (state->gradient_norm -
                    adapter->config.gradient_stress_threshold) * 0.1f;
                adapter->training_stress += gradient_stress * 0.05f;
                adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);
            }
            break;
        }

        case RAPHE_TRAIN_EVENT_LR_CHANGE: {
            /* LR changes can be stressful */
            adapter->training_stress += 0.05f;
            adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);
            break;
        }

        case RAPHE_TRAIN_EVENT_EPOCH_START: {
            /* New epoch -> slight patience recovery */
            adapter->training_patience += 0.1f;
            adapter->training_patience = clamp_f(adapter->training_patience, 0.0f, 1.0f);
            break;
        }

        case RAPHE_TRAIN_EVENT_EPOCH_END: {
            /* Epoch completion -> reward, reduce stress */
            adapter->training_stress -= 0.1f;
            adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);

            /* Positive mood input */
            nimcp_raphe_apply_mood_input(&adapter->raphe, 0.1f);
            break;
        }

        case RAPHE_TRAIN_EVENT_REWARD: {
            /* Positive reward -> mood boost, stress reduction */
            adapter->training_stress -= 0.15f;
            adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);

            nimcp_raphe_apply_mood_input(&adapter->raphe, 0.2f);
            break;
        }

        case RAPHE_TRAIN_EVENT_TIMEOUT: {
            /* Training timeout -> patience drain, stress */
            adapter->training_patience -= adapter->config.patience_decay_rate * 10.0f;
            adapter->training_patience = clamp_f(adapter->training_patience, 0.0f, 1.0f);

            adapter->training_stress += 0.2f;
            adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);
            break;
        }

        default:
            break;
    }

    /* Convert training stress to Raphe inhibition */
    nimcp_raphe_apply_inhibition(&adapter->raphe, adapter->training_stress * 0.3f);

    return 0;
}

int nimcp_raphe_adapter_get_training_modulation(nimcp_raphe_adapter_t adapter,
                                                nimcp_raphe_training_modulation_t* modulation) {
    if (!adapter || !modulation) return -1;

    *modulation = adapter->current_modulation;
    return 0;
}

int nimcp_raphe_adapter_register_training_callback(nimcp_raphe_adapter_t adapter,
                                                   nimcp_raphe_training_callback_fn callback,
                                                   void* user_data) {
    if (!adapter) return -1;

    adapter->training_callback = callback;
    adapter->training_callback_data = user_data;
    return 0;
}

int nimcp_raphe_adapter_compute_impulse_control(nimcp_raphe_adapter_t adapter,
                                                float action_urgency,
                                                float* inhibition_signal) {
    if (!adapter || !inhibition_signal) return -1;

    return nimcp_raphe_compute_inhibition(&adapter->raphe, action_urgency, inhibition_signal);
}

int nimcp_raphe_adapter_compute_reward_discount(nimcp_raphe_adapter_t adapter,
                                                float reward,
                                                float delay,
                                                float* discounted_reward) {
    if (!adapter || !discounted_reward) return -1;

    return nimcp_raphe_discount_value(&adapter->raphe, reward, delay, discounted_reward);
}

/*=============================================================================
 * Mood/Anxiety Processing API
 *===========================================================================*/

int nimcp_raphe_adapter_process_stress(nimcp_raphe_adapter_t adapter, float stress_level) {
    if (!adapter) return -1;

    /* External stress input */
    nimcp_raphe_modulate_anxiety(&adapter->raphe, stress_level);

    /* Also affects training patience */
    adapter->training_patience -= stress_level * 0.1f;
    adapter->training_patience = clamp_f(adapter->training_patience, 0.0f, 1.0f);

    return 0;
}

int nimcp_raphe_adapter_process_positive_feedback(nimcp_raphe_adapter_t adapter,
                                                  float feedback_magnitude) {
    if (!adapter) return -1;

    /* Positive feedback -> mood improvement */
    nimcp_raphe_apply_mood_input(&adapter->raphe, feedback_magnitude);

    /* Reduces training stress */
    adapter->training_stress -= feedback_magnitude * 0.2f;
    adapter->training_stress = clamp_f(adapter->training_stress, 0.0f, 1.0f);

    return 0;
}

/*=============================================================================
 * Integration API
 *===========================================================================*/

int nimcp_raphe_adapter_process_immune(nimcp_raphe_adapter_t adapter,
                                       float inflammation,
                                       const float* cytokines,
                                       uint32_t num_cytokines) {
    if (!adapter) return -1;

    /* Inflammation affects 5-HT (sickness behavior) */
    /* Pro-inflammatory cytokines reduce 5-HT synthesis */
    float immune_stress = inflammation * 0.5f;

    if (cytokines && num_cytokines > 0) {
        /* Average cytokine effect */
        float cytokine_sum = 0.0f;
        for (uint32_t i = 0; i < num_cytokines; i++) {
            cytokine_sum += cytokines[i];
        }
        immune_stress += (cytokine_sum / num_cytokines) * 0.3f;
    }

    /* Apply as inhibition (reduces 5-HT firing) */
    nimcp_raphe_apply_inhibition(&adapter->raphe, immune_stress);

    return 0;
}

int nimcp_raphe_adapter_apply_vta_modulation(nimcp_raphe_adapter_t adapter, float da_level) {
    if (!adapter) return -1;

    /* DA-5HT interaction:
     * High DA can facilitate 5-HT release via D2 receptors
     * But excessive DA can inhibit via different pathways
     */
    float da_ratio = da_level / 20.0f;  /* Assume 20nM baseline */

    if (da_ratio > 1.5f) {
        /* High DA -> slight inhibition */
        nimcp_raphe_apply_inhibition(&adapter->raphe, (da_ratio - 1.5f) * 0.2f);
    } else if (da_ratio < 0.5f) {
        /* Low DA -> slight excitation (compensatory) */
        nimcp_raphe_apply_excitation(&adapter->raphe, (0.5f - da_ratio) * 0.1f);
    }

    return 0;
}

int nimcp_raphe_adapter_apply_habenula_input(nimcp_raphe_adapter_t adapter, float input) {
    if (!adapter) return -1;

    /* Habenula provides strong inhibitory input to Raphe */
    /* This is important for aversive learning - habenula inhibits 5-HT */
    nimcp_raphe_apply_inhibition(&adapter->raphe, input);

    return 0;
}

/*=============================================================================
 * State API
 *===========================================================================*/

int nimcp_raphe_adapter_get_state(nimcp_raphe_adapter_t adapter,
                                  nimcp_raphe_adapter_state_t* state) {
    if (!adapter || !state) return -1;

    state->is_active = adapter->is_active;

    float ht, mood, anxiety;
    nimcp_raphe_get_5ht(&adapter->raphe, &ht);
    nimcp_raphe_get_mood(&adapter->raphe, &mood);
    nimcp_raphe_get_anxiety(&adapter->raphe, &anxiety);

    float patience;
    nimcp_raphe_get_patience(&adapter->raphe, &patience);

    state->ht_level = ht;
    state->mood_valence = mood;
    state->anxiety = anxiety;
    state->patience = patience;
    state->messages_sent = adapter->messages_sent;
    state->messages_received = adapter->messages_received;
    state->updates_processed = adapter->updates_processed;
    state->total_active_time = adapter->total_active_time;
    state->training_events_received = adapter->training_events_received;
    state->training_modulations_sent = adapter->training_modulations_sent;

    return 0;
}

int nimcp_raphe_adapter_reset_stats(nimcp_raphe_adapter_t adapter) {
    if (!adapter) return -1;

    adapter->messages_sent = 0;
    adapter->messages_received = 0;
    adapter->updates_processed = 0;
    adapter->total_active_time = 0.0f;
    adapter->training_events_received = 0;
    adapter->training_modulations_sent = 0;

    nimcp_raphe_reset_metrics(&adapter->raphe);

    return 0;
}
