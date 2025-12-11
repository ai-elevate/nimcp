/**
 * @file nimcp_training_immune.c
 * @brief Training-Immune System Integration Implementation
 * @version 1.0.0
 * @date 2025-12-11
 *
 * Implementation of bidirectional training-immune coordination.
 * See header for architecture and biological basis.
 */

#include "middleware/immune/nimcp_training_immune.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#include <pthread.h>
#endif

/* ============================================================================
 * Platform-Specific Implementations
 * ============================================================================ */

#ifdef _WIN32
typedef CRITICAL_SECTION platform_mutex_t;

static void platform_mutex_init(platform_mutex_t* mutex) {
    InitializeCriticalSection(mutex);
}

static void platform_mutex_destroy(platform_mutex_t* mutex) {
    DeleteCriticalSection(mutex);
}

static void platform_mutex_lock(platform_mutex_t* mutex) {
    EnterCriticalSection(mutex);
}

static void platform_mutex_unlock(platform_mutex_t* mutex) {
    LeaveCriticalSection(mutex);
}

#else
typedef pthread_mutex_t platform_mutex_t;

static void platform_mutex_init(platform_mutex_t* mutex) {
    pthread_mutex_init(mutex, NULL);
}

static void platform_mutex_destroy(platform_mutex_t* mutex) {
    pthread_mutex_destroy(mutex);
}

static void platform_mutex_lock(platform_mutex_t* mutex) {
    pthread_mutex_lock(mutex);
}

static void platform_mutex_unlock(platform_mutex_t* mutex) {
    pthread_mutex_unlock(mutex);
}
#endif

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Get current timestamp in milliseconds
 */
static uint64_t get_timestamp_ms(void) {
#ifdef _WIN32
    return GetTickCount64();
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
#endif
}

/**
 * @brief Compute learning rate factor for inflammation level
 *
 * WHAT: Map inflammation to LR reduction
 * WHY:  Model fever-induced learning suppression
 * HOW:  Use configured factors per inflammation level
 */
static float compute_lr_factor(
    const training_immune_system_t* system,
    brain_inflammation_level_t inflammation
) {
    if (!system) return 1.0f;

    switch (inflammation) {
        case INFLAMMATION_NONE:
            return TRAINING_IMMUNE_LR_FACTOR_NONE;
        case INFLAMMATION_LOCAL:
            return system->config.lr_factor_local;
        case INFLAMMATION_REGIONAL:
            return system->config.lr_factor_regional;
        case INFLAMMATION_SYSTEMIC:
            return system->config.lr_factor_systemic;
        case INFLAMMATION_STORM:
            return system->config.lr_factor_storm;
        default:
            return 1.0f;
    }
}

/**
 * @brief Check if value is NaN
 */
static bool is_nan_value(float value) {
    return value != value;
}

/**
 * @brief Check if value is infinite
 */
static bool is_inf_value(float value) {
    return !is_nan_value(value) && (value > FLT_MAX || value < -FLT_MAX);
}

/**
 * @brief Add metrics to history buffer
 *
 * WHAT: Store metrics in circular buffer
 * WHY:  Track training history for divergence detection
 * HOW:  Circular buffer with fixed capacity
 */
static void add_metrics_to_history(
    training_immune_system_t* system,
    const training_immune_metrics_t* metrics
) {
    if (!system || !metrics || !system->history) return;

    /* Circular buffer */
    system->history[system->history_index] = *metrics;
    system->history_index = (system->history_index + 1) % system->history_capacity;

    if (system->history_count < system->history_capacity) {
        system->history_count++;
    }
}

/**
 * @brief Create instability epitope signature
 *
 * WHAT: Convert instability to threat signature
 * WHY:  Present training failure as immune antigen
 * HOW:  Encode type and severity as byte pattern
 */
static void create_instability_epitope(
    training_instability_type_t type,
    uint32_t severity,
    uint8_t* epitope,
    size_t* epitope_len
) {
    if (!epitope || !epitope_len) return;

    /* Simple encoding: type ID + severity */
    epitope[0] = (uint8_t)type;
    epitope[1] = (uint8_t)(severity & 0xFF);
    epitope[2] = (uint8_t)((severity >> 8) & 0xFF);
    epitope[3] = 0xAA; /* Magic marker for training instability */

    *epitope_len = 4;
}

/* ============================================================================
 * Lifecycle Implementation
 * ============================================================================ */

int training_immune_default_config(training_immune_config_t* config) {
    if (!config) return -1;

    memset(config, 0, sizeof(training_immune_config_t));

    /* Immune modulation settings */
    config->enable_lr_modulation = true;
    config->enable_grad_scaling = true;
    config->min_lr_factor = 0.01f; /* Never reduce below 1% */

    /* Inflammation LR factors */
    config->lr_factor_local = TRAINING_IMMUNE_LR_FACTOR_LOCAL;
    config->lr_factor_regional = TRAINING_IMMUNE_LR_FACTOR_REGIONAL;
    config->lr_factor_systemic = TRAINING_IMMUNE_LR_FACTOR_SYSTEMIC;
    config->lr_factor_storm = TRAINING_IMMUNE_LR_FACTOR_STORM;

    /* Divergence detection thresholds */
    config->loss_explosion_ratio = TRAINING_IMMUNE_LOSS_EXPLOSION_RATIO;
    config->grad_explosion_threshold = TRAINING_IMMUNE_GRAD_EXPLOSION_THRESHOLD;
    config->grad_vanishing_threshold = 1e-7f;
    config->plateau_steps = TRAINING_IMMUNE_LOSS_PLATEAU_STEPS;
    config->plateau_delta = 1e-4f;

    /* Immune response settings */
    config->enable_auto_immune_response = true;
    config->min_response_duration_ms = TRAINING_IMMUNE_MIN_RESPONSE_DURATION_MS;
    config->inflammation_ema_alpha = TRAINING_IMMUNE_INFLAMMATION_EMA_ALPHA;

    /* Monitoring */
    config->history_size = TRAINING_IMMUNE_MAX_HISTORY;
    config->enable_logging = false;

    return 0;
}

training_immune_system_t* training_immune_create(
    const training_immune_config_t* config
) {
    training_immune_config_t default_config;

    /* Use defaults if no config provided */
    if (!config) {
        training_immune_default_config(&default_config);
        config = &default_config;
    }

    /* Allocate system */
    training_immune_system_t* system =
        (training_immune_system_t*)nimcp_malloc(sizeof(training_immune_system_t));
    if (!system) return NULL;

    memset(system, 0, sizeof(training_immune_system_t));

    /* Copy config */
    system->config = *config;
    system->phase = TRAINING_IMMUNE_PHASE_HEALTHY;

    /* Allocate history buffer */
    system->history_capacity = config->history_size;
    system->history = (training_immune_metrics_t*)nimcp_malloc(
        system->history_capacity * sizeof(training_immune_metrics_t)
    );
    if (!system->history) {
        nimcp_free(system);
        return NULL;
    }

    /* Allocate event buffer */
    system->event_capacity = 64;
    system->events = (training_instability_event_t*)nimcp_malloc(
        system->event_capacity * sizeof(training_instability_event_t)
    );
    if (!system->events) {
        nimcp_free(system->history);
        nimcp_free(system);
        return NULL;
    }

    /* Initialize state */
    system->inflammation = INFLAMMATION_NONE;
    system->inflammation_ema = 0.0f;
    system->current_lr_factor = 1.0f;
    system->best_loss = FLT_MAX;
    system->next_event_id = 1;

    /* Create mutex */
    system->mutex = nimcp_malloc(sizeof(platform_mutex_t));
    if (!system->mutex) {
        nimcp_free(system->events);
        nimcp_free(system->history);
        nimcp_free(system);
        return NULL;
    }
    platform_mutex_init((platform_mutex_t*)system->mutex);

    system->start_time_ms = get_timestamp_ms();

    if (config->enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Training immune system created");
    }

    return system;
}

void training_immune_destroy(training_immune_system_t* system) {
    if (!system) return;

    /* Stop if running */
    if (system->running) {
        training_immune_stop(system);
    }

    /* Destroy mutex */
    if (system->mutex) {
        platform_mutex_destroy((platform_mutex_t*)system->mutex);
        nimcp_free(system->mutex);
    }

    /* Free buffers */
    nimcp_free(system->events);
    nimcp_free(system->history);
    nimcp_free(system);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Training immune system destroyed");
    }
}

int training_immune_start(training_immune_system_t* system) {
    if (!system) return -1;
    if (system->running) return 0;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    system->running = true;
    system->phase = TRAINING_IMMUNE_PHASE_MONITORING;
    system->start_time_ms = get_timestamp_ms();

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Training immune monitoring started");
    }

    return 0;
}

int training_immune_stop(training_immune_system_t* system) {
    if (!system) return -1;
    if (!system->running) return 0;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    system->running = false;
    system->phase = TRAINING_IMMUNE_PHASE_RESOLVED;

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Training immune monitoring stopped");
    }

    return 0;
}

/* ============================================================================
 * Integration Implementation
 * ============================================================================ */

int training_immune_connect_brain_immune(
    training_immune_system_t* system,
    brain_immune_system_t* brain_immune
) {
    if (!system || !brain_immune) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    system->brain_immune = brain_immune;
    system->config.has_brain_immune = true;

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Connected to brain immune system");
    }

    return 0;
}

int training_immune_connect_optimizer(
    training_immune_system_t* system,
    nimcp_optimizer_context_t* optimizer
) {
    if (!system || !optimizer) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    system->optimizer = optimizer;
    system->config.has_optimizer = true;

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Connected to optimizer");
    }

    return 0;
}

int training_immune_connect_gradient_manager(
    training_immune_system_t* system,
    nimcp_gradient_manager_ctx_t* grad_manager
) {
    if (!system || !grad_manager) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    system->grad_manager = grad_manager;
    system->config.has_gradient_manager = true;

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Connected to gradient manager");
    }

    return 0;
}

int training_immune_connect_callbacks(
    training_immune_system_t* system,
    tcb_context_t* callbacks
) {
    if (!system || !callbacks) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    system->callbacks = callbacks;
    system->config.has_callbacks = true;

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Connected to training callbacks");
    }

    return 0;
}

/* ============================================================================
 * Immune → Training Implementation
 * ============================================================================ */

int training_immune_update_inflammation(
    training_immune_system_t* system,
    brain_inflammation_level_t inflammation
) {
    if (!system) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    brain_inflammation_level_t prev_inflammation = system->inflammation;
    system->inflammation = inflammation;

    /* Update EMA for smoothing */
    float alpha = system->config.inflammation_ema_alpha;
    system->inflammation_ema =
        alpha * (float)inflammation + (1.0f - alpha) * system->inflammation_ema;

    /* Compute new LR factor */
    float new_factor = compute_lr_factor(system, inflammation);

    /* Apply minimum LR factor constraint */
    if (new_factor < system->config.min_lr_factor) {
        new_factor = system->config.min_lr_factor;
    }

    system->current_lr_factor = new_factor;

    /* Update phase */
    if (inflammation > INFLAMMATION_NONE) {
        system->phase = TRAINING_IMMUNE_PHASE_RESPONDING;
        system->in_immune_response = true;

        if (!system->immune_response_start_ms) {
            system->immune_response_start_ms = get_timestamp_ms();
        }

        system->stats.inflamed_steps++;
    } else if (prev_inflammation > INFLAMMATION_NONE && inflammation == INFLAMMATION_NONE) {
        /* Check minimum response duration */
        uint64_t duration = get_timestamp_ms() - system->immune_response_start_ms;
        if (duration >= system->config.min_response_duration_ms) {
            system->phase = TRAINING_IMMUNE_PHASE_RECOVERING;
            system->in_immune_response = false;
            system->immune_response_start_ms = 0;
        }
    } else {
        system->stats.healthy_steps++;
    }

    /* Update statistics */
    if (inflammation != prev_inflammation) {
        system->stats.lr_modulations++;
    }

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging && inflammation != prev_inflammation) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Inflammation changed: %d -> %d, LR factor: %.3f",
                  prev_inflammation, inflammation, new_factor);
    }

    return 0;
}

float training_immune_get_effective_lr(
    const training_immune_system_t* system,
    float base_lr
) {
    if (!system || !system->config.enable_lr_modulation) {
        return base_lr;
    }

    return base_lr * system->current_lr_factor;
}

float training_immune_get_lr_factor(
    const training_immune_system_t* system
) {
    if (!system) return 1.0f;
    return system->current_lr_factor;
}

int training_immune_apply_lr_modulation(
    training_immune_system_t* system
) {
    if (!system) return -1;
    if (!system->config.enable_lr_modulation) return 0;
    if (!system->config.has_optimizer) return -1;

    /* Get current base LR from optimizer */
    float base_lr = nimcp_optimizer_get_lr(system->optimizer);

    /* Compute effective LR */
    float effective_lr = training_immune_get_effective_lr(system, base_lr);

    /* Set modulated LR */
    nimcp_optimizer_set_lr(system->optimizer, effective_lr);

    return 0;
}

float training_immune_get_gradient_scale(
    const training_immune_system_t* system
) {
    if (!system || !system->config.enable_grad_scaling) {
        return 1.0f;
    }

    /* Increase numerical stability during inflammation */
    /* Scale down gradients proportional to inflammation */
    return system->current_lr_factor;
}

/* ============================================================================
 * Training → Immune Implementation
 * ============================================================================ */

int training_immune_update_metrics(
    training_immune_system_t* system,
    float loss,
    float grad_norm,
    float learning_rate
) {
    if (!system) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    /* Update metrics */
    training_immune_metrics_t* m = &system->current_metrics;

    m->loss_prev = m->loss;
    m->loss = loss;
    m->grad_norm = grad_norm;
    m->learning_rate = learning_rate;
    m->effective_lr = training_immune_get_effective_lr(system, learning_rate);
    m->timestamp_ms = get_timestamp_ms();

    /* Update step counter */
    m->step++;

    /* Update min loss and check for improvement */
    if (loss < system->best_loss) {
        system->best_loss = loss;
        system->steps_without_improvement = 0;
    } else {
        system->steps_without_improvement++;
    }

    /* Check for NaN/Inf */
    m->has_nan = is_nan_value(loss) || is_nan_value(grad_norm);
    m->has_inf = is_inf_value(loss) || is_inf_value(grad_norm);

    /* Check for explosion/vanishing */
    m->is_exploding = grad_norm > system->config.grad_explosion_threshold;
    m->is_vanishing = grad_norm < system->config.grad_vanishing_threshold;
    m->is_plateau = system->steps_without_improvement >= system->config.plateau_steps;

    /* Update EMA */
    if (m->step == 1) {
        m->loss_ema = loss;
    } else {
        float alpha = 0.1f;
        m->loss_ema = alpha * loss + (1.0f - alpha) * m->loss_ema;
    }

    /* Add to history */
    add_metrics_to_history(system, m);

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    return 0;
}

training_instability_type_t training_immune_check_stability(
    training_immune_system_t* system
) {
    if (!system) return TRAINING_INSTABILITY_NONE;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    const training_immune_metrics_t* m = &system->current_metrics;
    training_instability_type_t instability = TRAINING_INSTABILITY_NONE;

    /* Check for critical failures first */
    if (m->has_nan) {
        instability = TRAINING_INSTABILITY_LOSS_NAN;
    } else if (m->has_inf) {
        instability = TRAINING_INSTABILITY_LOSS_INF;
    } else if (m->loss_prev > 0 &&
               m->loss / m->loss_prev > system->config.loss_explosion_ratio) {
        instability = TRAINING_INSTABILITY_LOSS_EXPLOSION;
    } else if (m->is_exploding) {
        instability = TRAINING_INSTABILITY_GRAD_EXPLOSION;
    } else if (m->is_vanishing) {
        instability = TRAINING_INSTABILITY_GRAD_VANISHING;
    } else if (m->is_plateau) {
        instability = TRAINING_INSTABILITY_LOSS_PLATEAU;
    }

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    /* Auto-trigger immune response if enabled */
    if (instability != TRAINING_INSTABILITY_NONE &&
        system->config.enable_auto_immune_response) {

        uint32_t event_id;
        uint32_t severity = (instability <= TRAINING_INSTABILITY_LOSS_INF) ? 10 :
                           (instability <= TRAINING_INSTABILITY_LOSS_EXPLOSION) ? 8 :
                           (instability <= TRAINING_INSTABILITY_GRAD_EXPLOSION) ? 6 : 3;

        training_immune_report_instability(system, instability, severity, &event_id);
        training_immune_trigger_immune_response(system, event_id);
    }

    return instability;
}

int training_immune_report_instability(
    training_immune_system_t* system,
    training_instability_type_t type,
    uint32_t severity,
    uint32_t* event_id
) {
    if (!system || !event_id) return -1;
    if (type >= TRAINING_INSTABILITY_COUNT) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    /* Check capacity */
    if (system->event_count >= system->event_capacity) {
        platform_mutex_unlock((platform_mutex_t*)system->mutex);
        return -1;
    }

    /* Create event */
    training_instability_event_t* event = &system->events[system->event_count++];
    memset(event, 0, sizeof(training_instability_event_t));

    event->event_id = system->next_event_id++;
    event->type = type;
    event->severity = severity;
    event->confidence = 1.0f;
    event->metrics = system->current_metrics;
    event->detection_time_ms = get_timestamp_ms();

    *event_id = event->event_id;

    /* Update statistics */
    system->stats.total_instabilities++;
    system->stats.instabilities_by_type[type]++;

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging) {
        nimcp_log(LOG_LEVEL_WARN, TRAINING_IMMUNE_MODULE_NAME,
                  "Training instability detected: %s (severity=%u)",
                  training_instability_type_to_string(type), severity);
    }

    return 0;
}

int training_immune_trigger_immune_response(
    training_immune_system_t* system,
    uint32_t event_id
) {
    if (!system) return -1;
    if (!system->config.has_brain_immune) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    /* Find event */
    training_instability_event_t* event = NULL;
    for (size_t i = 0; i < system->event_count; i++) {
        if (system->events[i].event_id == event_id) {
            event = &system->events[i];
            break;
        }
    }

    if (!event) {
        platform_mutex_unlock((platform_mutex_t*)system->mutex);
        return -1;
    }

    /* Create epitope */
    uint8_t epitope[BRAIN_IMMUNE_EPITOPE_SIZE];
    size_t epitope_len;
    create_instability_epitope(event->type, event->severity, epitope, &epitope_len);

    /* Present as antigen to brain immune system */
    uint32_t antigen_id;
    int result = brain_immune_present_antigen(
        system->brain_immune,
        ANTIGEN_SOURCE_ANOMALY,
        epitope,
        epitope_len,
        event->severity,
        0, /* No specific node */
        &antigen_id
    );

    if (result == 0) {
        event->antigen_id = antigen_id;
        event->immune_triggered = true;

        /* Update statistics */
        system->stats.immune_responses_triggered++;
        system->phase = TRAINING_IMMUNE_PHASE_RESPONDING;
    }

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    if (system->config.enable_logging && result == 0) {
        nimcp_log(NIMCP_LOG_INFO, TRAINING_IMMUNE_MODULE_NAME,
                  "Immune response triggered for instability event %u (antigen=%u)",
                  event_id, antigen_id);
    }

    return result;
}

/* ============================================================================
 * Query and Statistics Implementation
 * ============================================================================ */

training_immune_phase_t training_immune_get_phase(
    const training_immune_system_t* system
) {
    return system ? system->phase : TRAINING_IMMUNE_PHASE_HEALTHY;
}

brain_inflammation_level_t training_immune_get_inflammation(
    const training_immune_system_t* system
) {
    return system ? system->inflammation : INFLAMMATION_NONE;
}

bool training_immune_is_responding(
    const training_immune_system_t* system
) {
    return system ? system->in_immune_response : false;
}

int training_immune_get_stats(
    const training_immune_system_t* system,
    training_immune_stats_t* stats
) {
    if (!system || !stats) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);

    *stats = system->stats;

    /* Update current state */
    stats->current_phase = system->phase;
    stats->current_inflammation = system->inflammation;
    stats->current_lr_factor = system->current_lr_factor;

    /* Compute averages */
    if (system->stats.lr_modulations > 0) {
        stats->avg_lr_reduction_factor =
            (float)system->stats.healthy_steps /
            (float)(system->stats.healthy_steps + system->stats.inflamed_steps);
    }

    uint64_t total_steps = system->stats.healthy_steps + system->stats.inflamed_steps;
    if (total_steps > 0) {
        stats->time_in_inflammation_pct =
            100.0f * (float)system->stats.inflamed_steps / (float)total_steps;
    }

    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    return 0;
}

int training_immune_get_current_metrics(
    const training_immune_system_t* system,
    training_immune_metrics_t* metrics
) {
    if (!system || !metrics) return -1;

    platform_mutex_lock((platform_mutex_t*)system->mutex);
    *metrics = system->current_metrics;
    platform_mutex_unlock((platform_mutex_t*)system->mutex);

    return 0;
}

const training_instability_event_t* training_immune_get_event(
    const training_immune_system_t* system,
    uint32_t event_id
) {
    if (!system) return NULL;

    for (size_t i = 0; i < system->event_count; i++) {
        if (system->events[i].event_id == event_id) {
            return &system->events[i];
        }
    }

    return NULL;
}

/* ============================================================================
 * String Conversion Utilities
 * ============================================================================ */

const char* training_immune_phase_to_string(training_immune_phase_t phase) {
    switch (phase) {
        case TRAINING_IMMUNE_PHASE_HEALTHY:    return "HEALTHY";
        case TRAINING_IMMUNE_PHASE_MONITORING: return "MONITORING";
        case TRAINING_IMMUNE_PHASE_RESPONDING: return "RESPONDING";
        case TRAINING_IMMUNE_PHASE_RECOVERING: return "RECOVERING";
        case TRAINING_IMMUNE_PHASE_RESOLVED:   return "RESOLVED";
        default:                               return "UNKNOWN";
    }
}

const char* training_instability_type_to_string(training_instability_type_t type) {
    switch (type) {
        case TRAINING_INSTABILITY_NONE:           return "NONE";
        case TRAINING_INSTABILITY_LOSS_NAN:       return "LOSS_NAN";
        case TRAINING_INSTABILITY_LOSS_INF:       return "LOSS_INF";
        case TRAINING_INSTABILITY_LOSS_EXPLOSION: return "LOSS_EXPLOSION";
        case TRAINING_INSTABILITY_GRAD_EXPLOSION: return "GRAD_EXPLOSION";
        case TRAINING_INSTABILITY_GRAD_VANISHING: return "GRAD_VANISHING";
        case TRAINING_INSTABILITY_LOSS_PLATEAU:   return "LOSS_PLATEAU";
        case TRAINING_INSTABILITY_OSCILLATION:    return "OSCILLATION";
        default:                                  return "UNKNOWN";
    }
}
