/**
 * @file nimcp_motor_adapter.c
 * @brief Implementation of Motor Cortex brain adapter
 *
 * WHAT: Unified adapter connecting Motor Cortex sub-modules to the brain system
 * WHY:  Enable seamless integration with cognitive layers, training, and event system
 * HOW:  Orchestrates M1, premotor, and SMA processors
 *
 * @version Phase M1: Motor Cortex Brain Integration
 * @date 2025-12-30
 */

#include "core/brain/regions/motor/nimcp_motor_adapter.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/memory/nimcp_memory_pool.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"
#include <string.h>
#include <math.h>

/*=============================================================================
 * LOGGING MODULE IDENTIFIER
 *===========================================================================*/

#define MOTOR_LOG_MODULE "MOTOR"

/*=============================================================================
 * INTERNAL STRUCTURES
 *===========================================================================*/

/**
 * @brief Stored motor program
 */
typedef struct motor_program {
    motor_program_info_t info;
    motor_command_t* commands;       /**< Command sequence */
    struct motor_program* next;      /**< Hash chain */
} motor_program_node_t;

/**
 * @brief Command queue entry
 */
typedef struct {
    motor_command_t command;
    bool is_valid;
} command_queue_entry_t;

/**
 * @brief Trajectory plan
 */
typedef struct {
    trajectory_waypoint_t* waypoints;
    uint32_t num_waypoints;
    uint32_t current_waypoint;
    float elapsed_ms;
    motor_region_t region;
    bool is_active;
} trajectory_plan_t;

/**
 * @brief Internal adapter structure
 */
struct motor_adapter {
    /* Configuration */
    motor_config_t config;

    /* Effector states */
    motor_effector_state_t* effectors;
    uint32_t num_effectors;

    /* Motor program storage (hash table) */
    motor_program_node_t** programs;
    uint32_t program_capacity;
    uint32_t program_count;
    uint32_t next_program_id;

    /* Trajectory planning */
    trajectory_plan_t* trajectories;
    uint32_t num_trajectories;
    trajectory_plan_t* active_trajectory;

    /* Command output queue */
    command_queue_entry_t* command_queue;
    uint32_t queue_capacity;
    uint32_t queue_head;
    uint32_t queue_tail;
    uint32_t queue_count;

    /* Current movement state */
    motor_goal_t current_goal;
    motor_result_t last_result;
    bool has_active_goal;
    float execution_time_ms;

    /* Callbacks */
    motor_command_callback_t command_callback;
    void* command_user_data;
    motor_complete_callback_t complete_callback;
    void* complete_user_data;
    motor_event_callback_t event_callback;
    void* event_user_data;

    /* State */
    motor_status_t status;
    motor_error_t last_error;
    double current_time_ms;

    /* Memory pool for trajectory waypoints */
    memory_pool_t waypoint_pool;

    /* Bio-async communication context */
    bio_module_context_t bio_ctx;
    nimcp_bio_channel_type_t default_channel;

    /* Statistics */
    motor_stats_t stats;
};

/*=============================================================================
 * INTERNAL HELPERS
 *===========================================================================*/

/**
 * @brief Hash program ID for storage
 */
static uint32_t hash_program_id(uint32_t program_id, uint32_t capacity) {
    return program_id % capacity;
}

/**
 * @brief Emit event to callback
 */
static void emit_event(motor_adapter_t* adapter, uint32_t event_type, const void* data) {
    if (adapter->config.enable_events && adapter->event_callback) {
        adapter->event_callback(event_type, data, adapter->event_user_data);
    }
}

/**
 * @brief Set error state
 */
static void set_error(motor_adapter_t* adapter, motor_error_t error) {
    if (!adapter) return;
    adapter->last_error = error;
    if (error != MOTOR_ERROR_NONE) {
        adapter->status = MOTOR_STATUS_ERROR;
        LOG_ERROR("[%s] Error set: %d", MOTOR_LOG_MODULE, error);
    }
}

/**
 * @brief Compute distance between two points
 */
static float compute_distance(const motor_vec3_t* a, const motor_vec3_t* b) {
    float dx = b->x - a->x;
    float dy = b->y - a->y;
    float dz = b->z - a->z;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

/**
 * @brief Linear interpolation between points
 */
static motor_vec3_t lerp_vec3(const motor_vec3_t* a, const motor_vec3_t* b, float t) {
    motor_vec3_t result;
    result.x = a->x + (b->x - a->x) * t;
    result.y = a->y + (b->y - a->y) * t;
    result.z = a->z + (b->z - a->z) * t;
    return result;
}

/**
 * @brief Enqueue a motor command
 */
static bool enqueue_command(motor_adapter_t* adapter, const motor_command_t* command) {
    if (!adapter || !command) return false;

    if (adapter->queue_count >= adapter->queue_capacity) {
        LOG_WARN("[%s] Command queue full", MOTOR_LOG_MODULE);
        return false;
    }

    adapter->command_queue[adapter->queue_tail].command = *command;
    adapter->command_queue[adapter->queue_tail].is_valid = true;
    adapter->queue_tail = (adapter->queue_tail + 1) % adapter->queue_capacity;
    adapter->queue_count++;

    /* Invoke callback if set */
    if (adapter->command_callback) {
        adapter->command_callback(command, adapter->command_user_data);
    }

    return true;
}

/*=============================================================================
 * BIO-ASYNC MESSAGE HANDLERS (Forward declarations)
 *===========================================================================*/

static nimcp_error_t handle_motor_command_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

static nimcp_error_t handle_motor_stop_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

static nimcp_error_t handle_bg_action_selection(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

static nimcp_error_t handle_cerebellar_correction(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data);

/*=============================================================================
 * LIFECYCLE FUNCTIONS
 *===========================================================================*/

motor_config_t motor_default_config(void) {
    motor_config_t config;
    memset(&config, 0, sizeof(config));

    config.max_motor_programs = MOTOR_DEFAULT_MAX_PROGRAMS;
    config.max_effectors = MOTOR_DEFAULT_MAX_EFFECTORS;
    config.max_trajectories = MOTOR_DEFAULT_MAX_TRAJECTORIES;
    config.planning_horizon_ms = MOTOR_DEFAULT_PLANNING_HORIZON_MS;
    config.execution_rate_hz = MOTOR_DEFAULT_EXECUTION_RATE_HZ;
    config.reaction_time_ms = 150.0f;  /* Typical human reaction time */

    config.enable_premotor = true;
    config.enable_sma = true;
    config.enable_trajectory_opt = true;
    config.enable_feedforward = true;
    config.enable_feedback = true;

    config.enable_basal_ganglia = true;
    config.enable_cerebellum = true;
    config.enable_thalamus = true;

    config.enable_events = true;
    config.enable_training = false;
    config.learning_rate = 0.01f;

    /* Bio-async: enabled by default, use acetylcholine for motor control */
    config.enable_bio_async = true;
    config.default_channel = BIO_CHANNEL_ACETYLCHOLINE;

    return config;
}

motor_adapter_t* motor_create(const motor_config_t* config) {
    LOG_INFO("[%s] Creating motor cortex adapter", MOTOR_LOG_MODULE);

    motor_adapter_t* adapter = (motor_adapter_t*)nimcp_calloc(1, sizeof(motor_adapter_t));
    if (!adapter) {
        LOG_ERROR("[%s] Failed to allocate adapter memory", MOTOR_LOG_MODULE);
        return NULL;
    }

    /* Set configuration */
    if (config) {
        adapter->config = *config;
        LOG_DEBUG("[%s] Using provided configuration", MOTOR_LOG_MODULE);
    } else {
        adapter->config = motor_default_config();
        LOG_DEBUG("[%s] Using default configuration", MOTOR_LOG_MODULE);
    }

    /* Initialize effector states */
    LOG_DEBUG("[%s] Initializing effectors (max=%u)", MOTOR_LOG_MODULE,
              adapter->config.max_effectors);
    adapter->effectors = (motor_effector_state_t*)nimcp_calloc(
        adapter->config.max_effectors, sizeof(motor_effector_state_t));
    if (!adapter->effectors) {
        LOG_ERROR("[%s] Failed to allocate effector array", MOTOR_LOG_MODULE);
        motor_destroy(adapter);
        return NULL;
    }

    /* Initialize effector IDs by region */
    for (uint32_t i = 0; i < MOTOR_REGION_COUNT && i < adapter->config.max_effectors; i++) {
        adapter->effectors[i].effector_id = i;
        adapter->effectors[i].region = (motor_region_t)i;
        adapter->effectors[i].stiffness = 0.5f;
        adapter->num_effectors++;
    }

    /* Initialize motor program storage */
    LOG_DEBUG("[%s] Initializing motor program storage (capacity=%u)", MOTOR_LOG_MODULE,
              adapter->config.max_motor_programs);
    adapter->program_capacity = adapter->config.max_motor_programs;
    adapter->programs = (motor_program_node_t**)nimcp_calloc(
        adapter->program_capacity, sizeof(motor_program_node_t*));
    if (!adapter->programs) {
        LOG_ERROR("[%s] Failed to allocate program storage", MOTOR_LOG_MODULE);
        motor_destroy(adapter);
        return NULL;
    }
    adapter->next_program_id = 1;

    /* Initialize trajectory planning */
    LOG_DEBUG("[%s] Initializing trajectory plans (max=%u)", MOTOR_LOG_MODULE,
              adapter->config.max_trajectories);
    adapter->trajectories = (trajectory_plan_t*)nimcp_calloc(
        adapter->config.max_trajectories, sizeof(trajectory_plan_t));
    if (!adapter->trajectories) {
        LOG_ERROR("[%s] Failed to allocate trajectory array", MOTOR_LOG_MODULE);
        motor_destroy(adapter);
        return NULL;
    }

    /* Initialize command queue */
    uint32_t queue_size = 256;  /* Commands per execution cycle */
    LOG_DEBUG("[%s] Initializing command queue (capacity=%u)", MOTOR_LOG_MODULE, queue_size);
    adapter->queue_capacity = queue_size;
    adapter->command_queue = (command_queue_entry_t*)nimcp_calloc(
        queue_size, sizeof(command_queue_entry_t));
    if (!adapter->command_queue) {
        LOG_ERROR("[%s] Failed to allocate command queue", MOTOR_LOG_MODULE);
        motor_destroy(adapter);
        return NULL;
    }

    /* Initialize memory pool for waypoints */
    LOG_DEBUG("[%s] Creating waypoint memory pool", MOTOR_LOG_MODULE);
    memory_pool_config_t pool_config = {
        .block_size = 64 * sizeof(trajectory_waypoint_t),  /* 64 waypoints per block */
        .num_blocks = 4,
        .alignment = 16,
        .enable_tracking = false,
        .enable_guard_pages = false
    };
    adapter->waypoint_pool = memory_pool_create(&pool_config);
    if (!adapter->waypoint_pool) {
        LOG_ERROR("[%s] Failed to create waypoint memory pool", MOTOR_LOG_MODULE);
        motor_destroy(adapter);
        return NULL;
    }

    /* Initialize bio-async communication */
    adapter->bio_ctx = NULL;
    adapter->default_channel = adapter->config.default_channel;

    if (adapter->config.enable_bio_async && bio_router_is_initialized()) {
        LOG_DEBUG("[%s] Registering with bio-async router", MOTOR_LOG_MODULE);

        bio_module_info_t bio_info = {
            .module_id = BIO_MODULE_MOTOR_CORTEX,
            .module_name = "motor_cortex",
            .inbox_capacity = 64,
            .user_data = adapter
        };

        adapter->bio_ctx = bio_router_register_module(&bio_info);
        if (adapter->bio_ctx) {
            /* Register message handlers */
            bio_router_register_handler(adapter->bio_ctx,
                BIO_MSG_MOTOR_COMMAND_REQUEST, handle_motor_command_request);
            bio_router_register_handler(adapter->bio_ctx,
                BIO_MSG_MOTOR_STOP_REQUEST, handle_motor_stop_request);
            bio_router_register_handler(adapter->bio_ctx,
                BIO_MSG_BG_ACTION_SELECTION, handle_bg_action_selection);
            bio_router_register_handler(adapter->bio_ctx,
                BIO_MSG_CEREBELLAR_CORRECTION, handle_cerebellar_correction);

            LOG_INFO("[%s] Bio-async handlers registered successfully", MOTOR_LOG_MODULE);
        } else {
            LOG_WARNING("[%s] Failed to register with bio-async router", MOTOR_LOG_MODULE);
        }
    }

    /* Initialize state */
    adapter->status = MOTOR_STATUS_IDLE;
    adapter->last_error = MOTOR_ERROR_NONE;
    adapter->current_time_ms = 0.0;
    adapter->has_active_goal = false;

    LOG_INFO("[%s] Motor cortex adapter created successfully", MOTOR_LOG_MODULE);
    return adapter;
}

void motor_destroy(motor_adapter_t* adapter) {
    if (!adapter) return;

    LOG_INFO("[%s] Destroying motor cortex adapter", MOTOR_LOG_MODULE);

    /* Unregister from bio-async router */
    if (adapter->bio_ctx) {
        LOG_DEBUG("[%s] Unregistering from bio-async router", MOTOR_LOG_MODULE);
        bio_router_unregister_module(adapter->bio_ctx);
        adapter->bio_ctx = NULL;
    }

    /* Free motor programs */
    if (adapter->programs) {
        LOG_DEBUG("[%s] Freeing motor programs", MOTOR_LOG_MODULE);
        for (uint32_t i = 0; i < adapter->program_capacity; i++) {
            motor_program_node_t* node = adapter->programs[i];
            while (node) {
                motor_program_node_t* next = node->next;
                if (node->commands) {
                    nimcp_free(node->commands);
                }
                nimcp_free(node);
                node = next;
            }
        }
        nimcp_free(adapter->programs);
    }

    /* Free trajectories */
    if (adapter->trajectories) {
        LOG_DEBUG("[%s] Freeing trajectories", MOTOR_LOG_MODULE);
        for (uint32_t i = 0; i < adapter->config.max_trajectories; i++) {
            if (adapter->trajectories[i].waypoints) {
                nimcp_free(adapter->trajectories[i].waypoints);
            }
        }
        nimcp_free(adapter->trajectories);
    }

    /* Free effectors */
    if (adapter->effectors) {
        LOG_DEBUG("[%s] Freeing effectors", MOTOR_LOG_MODULE);
        nimcp_free(adapter->effectors);
    }

    /* Free command queue */
    if (adapter->command_queue) {
        LOG_DEBUG("[%s] Freeing command queue", MOTOR_LOG_MODULE);
        nimcp_free(adapter->command_queue);
    }

    /* Destroy memory pool */
    if (adapter->waypoint_pool) {
        LOG_DEBUG("[%s] Destroying waypoint memory pool", MOTOR_LOG_MODULE);
        memory_pool_destroy(adapter->waypoint_pool);
    }

    LOG_DEBUG("[%s] Motor cortex adapter destroyed", MOTOR_LOG_MODULE);
    nimcp_free(adapter);
}

bool motor_reset(motor_adapter_t* adapter) {
    if (!adapter) return false;

    LOG_DEBUG("[%s] Resetting adapter state", MOTOR_LOG_MODULE);

    /* Clear command queue */
    adapter->queue_head = 0;
    adapter->queue_tail = 0;
    adapter->queue_count = 0;

    /* Clear active trajectory */
    adapter->active_trajectory = NULL;
    for (uint32_t i = 0; i < adapter->config.max_trajectories; i++) {
        adapter->trajectories[i].is_active = false;
    }

    /* Clear movement state */
    adapter->has_active_goal = false;
    adapter->execution_time_ms = 0.0f;
    memset(&adapter->current_goal, 0, sizeof(motor_goal_t));
    memset(&adapter->last_result, 0, sizeof(motor_result_t));

    /* Reset state */
    adapter->status = MOTOR_STATUS_IDLE;
    adapter->last_error = MOTOR_ERROR_NONE;

    LOG_DEBUG("[%s] Adapter reset complete", MOTOR_LOG_MODULE);
    return true;
}

/*=============================================================================
 * MOTOR PROGRAM MANAGEMENT
 *===========================================================================*/

uint32_t motor_store_program(motor_adapter_t* adapter,
                              const char* name,
                              const motor_command_t* commands,
                              uint32_t num_commands,
                              movement_type_t type) {
    if (!adapter || !name || !commands || num_commands == 0) return 0;

    if (adapter->program_count >= adapter->program_capacity) {
        LOG_WARN("[%s] Program storage full", MOTOR_LOG_MODULE);
        return 0;
    }

    /* Create new program node */
    motor_program_node_t* node = (motor_program_node_t*)nimcp_calloc(
        1, sizeof(motor_program_node_t));
    if (!node) return 0;

    /* Copy commands */
    node->commands = (motor_command_t*)nimcp_calloc(num_commands, sizeof(motor_command_t));
    if (!node->commands) {
        nimcp_free(node);
        return 0;
    }
    memcpy(node->commands, commands, num_commands * sizeof(motor_command_t));

    /* Fill program info */
    node->info.program_id = adapter->next_program_id++;
    strncpy(node->info.name, name, sizeof(node->info.name) - 1);
    node->info.type = type;
    node->info.num_commands = num_commands;
    node->info.is_learned = true;

    /* Calculate duration and complexity */
    float duration = 0.0f;
    for (uint32_t i = 0; i < num_commands; i++) {
        duration = fmaxf(duration, commands[i].timestamp_ms + commands[i].duration_ms);
    }
    node->info.total_duration_ms = duration;
    node->info.complexity = fminf(1.0f, (float)num_commands / 100.0f);

    /* Get primary region from first command */
    if (num_commands > 0 && commands[0].effector_id < adapter->num_effectors) {
        node->info.primary_region = adapter->effectors[commands[0].effector_id].region;
    }

    /* Insert into hash table */
    uint32_t idx = hash_program_id(node->info.program_id, adapter->program_capacity);
    node->next = adapter->programs[idx];
    adapter->programs[idx] = node;
    adapter->program_count++;

    LOG_DEBUG("[%s] Stored motor program '%s' (ID=%u, commands=%u)",
              MOTOR_LOG_MODULE, name, node->info.program_id, num_commands);

    return node->info.program_id;
}

bool motor_get_program(const motor_adapter_t* adapter,
                        uint32_t program_id,
                        motor_program_info_t* info) {
    if (!adapter || !info || program_id == 0) return false;

    uint32_t idx = hash_program_id(program_id, adapter->program_capacity);
    motor_program_node_t* node = adapter->programs[idx];

    while (node) {
        if (node->info.program_id == program_id) {
            *info = node->info;
            return true;
        }
        node = node->next;
    }

    return false;
}

bool motor_delete_program(motor_adapter_t* adapter, uint32_t program_id) {
    if (!adapter || program_id == 0) return false;

    uint32_t idx = hash_program_id(program_id, adapter->program_capacity);
    motor_program_node_t** pp = &adapter->programs[idx];

    while (*pp) {
        if ((*pp)->info.program_id == program_id) {
            motor_program_node_t* node = *pp;
            *pp = node->next;
            if (node->commands) nimcp_free(node->commands);
            nimcp_free(node);
            adapter->program_count--;
            LOG_DEBUG("[%s] Deleted motor program ID=%u", MOTOR_LOG_MODULE, program_id);
            return true;
        }
        pp = &(*pp)->next;
    }

    return false;
}

/*=============================================================================
 * MOTOR PLANNING
 *===========================================================================*/

bool motor_plan_movement(motor_adapter_t* adapter, const motor_goal_t* goal) {
    if (!adapter || !goal) {
        set_error(adapter, MOTOR_ERROR_INVALID_INPUT);
        return false;
    }

    LOG_DEBUG("[%s] Planning movement to region %d", MOTOR_LOG_MODULE, goal->region);

    adapter->status = MOTOR_STATUS_PLANNING;
    adapter->current_goal = *goal;
    adapter->has_active_goal = true;
    adapter->stats.movements_planned++;

    /* Get current effector state */
    uint32_t effector_id = (uint32_t)goal->region;
    if (effector_id >= adapter->num_effectors) {
        set_error(adapter, MOTOR_ERROR_INVALID_INPUT);
        return false;
    }

    motor_effector_state_t* effector = &adapter->effectors[effector_id];

    /* Calculate movement parameters */
    float distance = compute_distance(&effector->position, &goal->target_position);
    float duration = goal->max_duration_ms;

    /* Apply Fitts' law for duration estimation if not specified */
    if (duration <= 0.0f) {
        float precision = fmaxf(0.01f, goal->precision_required);
        float a = 50.0f;   /* Intercept (ms) */
        float b = 150.0f;  /* Slope (ms) */
        duration = a + b * log2f(distance / precision + 1.0f);
        duration = fmaxf(duration, adapter->config.reaction_time_ms);
    }

    /* Create simple trajectory with start and end */
    if (adapter->num_trajectories >= adapter->config.max_trajectories) {
        set_error(adapter, MOTOR_ERROR_BUFFER_OVERFLOW);
        return false;
    }

    trajectory_plan_t* traj = &adapter->trajectories[adapter->num_trajectories++];
    traj->waypoints = (trajectory_waypoint_t*)nimcp_calloc(2, sizeof(trajectory_waypoint_t));
    if (!traj->waypoints) {
        set_error(adapter, MOTOR_ERROR_INTERNAL);
        return false;
    }

    /* Start waypoint */
    traj->waypoints[0].position = effector->position;
    traj->waypoints[0].velocity = effector->velocity;
    traj->waypoints[0].time_ms = 0.0f;

    /* End waypoint */
    traj->waypoints[1].position = goal->target_position;
    traj->waypoints[1].velocity = goal->target_velocity;
    traj->waypoints[1].time_ms = duration;

    traj->num_waypoints = 2;
    traj->current_waypoint = 0;
    traj->elapsed_ms = 0.0f;
    traj->region = goal->region;
    traj->is_active = true;

    adapter->active_trajectory = traj;
    adapter->status = MOTOR_STATUS_PREPARING;

    emit_event(adapter, 1 /* MOTOR_EVENT_PLAN_COMPLETE */, goal);

    LOG_DEBUG("[%s] Movement planned: distance=%.2f, duration=%.2fms",
              MOTOR_LOG_MODULE, distance, duration);

    return true;
}

bool motor_plan_trajectory(motor_adapter_t* adapter,
                            motor_region_t region,
                            const trajectory_waypoint_t* waypoints,
                            uint32_t num_waypoints) {
    if (!adapter || !waypoints || num_waypoints < 2) {
        set_error(adapter, MOTOR_ERROR_INVALID_INPUT);
        return false;
    }

    if (adapter->num_trajectories >= adapter->config.max_trajectories) {
        set_error(adapter, MOTOR_ERROR_BUFFER_OVERFLOW);
        return false;
    }

    LOG_DEBUG("[%s] Planning trajectory with %u waypoints", MOTOR_LOG_MODULE, num_waypoints);

    adapter->status = MOTOR_STATUS_PLANNING;
    adapter->stats.movements_planned++;

    trajectory_plan_t* traj = &adapter->trajectories[adapter->num_trajectories++];
    traj->waypoints = (trajectory_waypoint_t*)nimcp_calloc(
        num_waypoints, sizeof(trajectory_waypoint_t));
    if (!traj->waypoints) {
        set_error(adapter, MOTOR_ERROR_INTERNAL);
        return false;
    }

    memcpy(traj->waypoints, waypoints, num_waypoints * sizeof(trajectory_waypoint_t));
    traj->num_waypoints = num_waypoints;
    traj->current_waypoint = 0;
    traj->elapsed_ms = 0.0f;
    traj->region = region;
    traj->is_active = true;

    adapter->active_trajectory = traj;
    adapter->status = MOTOR_STATUS_PREPARING;
    adapter->has_active_goal = true;

    return true;
}

bool motor_plan_program_execution(motor_adapter_t* adapter,
                                   uint32_t program_id,
                                   float speed_factor) {
    if (!adapter || program_id == 0) {
        set_error(adapter, MOTOR_ERROR_INVALID_INPUT);
        return false;
    }

    /* Find program */
    uint32_t idx = hash_program_id(program_id, adapter->program_capacity);
    motor_program_node_t* node = adapter->programs[idx];

    while (node) {
        if (node->info.program_id == program_id) break;
        node = node->next;
    }

    if (!node) {
        set_error(adapter, MOTOR_ERROR_INVALID_INPUT);
        return false;
    }

    LOG_DEBUG("[%s] Planning execution of program '%s' (speed=%.2fx)",
              MOTOR_LOG_MODULE, node->info.name, speed_factor);

    adapter->status = MOTOR_STATUS_PLANNING;
    adapter->stats.movements_planned++;

    /* Clear command queue */
    adapter->queue_head = 0;
    adapter->queue_tail = 0;
    adapter->queue_count = 0;

    /* Scale commands by speed factor and enqueue */
    float scale = (speed_factor > 0.0f) ? (1.0f / speed_factor) : 1.0f;

    for (uint32_t i = 0; i < node->info.num_commands; i++) {
        motor_command_t cmd = node->commands[i];
        cmd.timestamp_ms *= scale;
        cmd.duration_ms *= scale;

        if (!enqueue_command(adapter, &cmd)) {
            set_error(adapter, MOTOR_ERROR_BUFFER_OVERFLOW);
            return false;
        }
    }

    adapter->status = MOTOR_STATUS_PREPARING;
    adapter->has_active_goal = true;

    return true;
}

/*=============================================================================
 * MOTOR EXECUTION
 *===========================================================================*/

bool motor_begin_execution(motor_adapter_t* adapter) {
    if (!adapter) return false;

    if (adapter->status != MOTOR_STATUS_PREPARING) {
        LOG_WARN("[%s] Cannot begin execution from status %d", MOTOR_LOG_MODULE, adapter->status);
        return false;
    }

    LOG_DEBUG("[%s] Beginning movement execution", MOTOR_LOG_MODULE);

    adapter->status = MOTOR_STATUS_EXECUTING;
    adapter->execution_time_ms = 0.0f;
    adapter->stats.movements_executed++;

    emit_event(adapter, 2 /* MOTOR_EVENT_EXECUTION_STARTED */, NULL);

    return true;
}

bool motor_update_execution(motor_adapter_t* adapter, float dt_ms) {
    if (!adapter || adapter->status != MOTOR_STATUS_EXECUTING) {
        return false;
    }

    adapter->execution_time_ms += dt_ms;
    adapter->current_time_ms += dt_ms;

    /* Process active trajectory */
    if (adapter->active_trajectory && adapter->active_trajectory->is_active) {
        trajectory_plan_t* traj = adapter->active_trajectory;
        traj->elapsed_ms += dt_ms;

        /* Find current segment */
        uint32_t seg = 0;
        while (seg + 1 < traj->num_waypoints &&
               traj->elapsed_ms > traj->waypoints[seg + 1].time_ms) {
            seg++;
        }

        /* Check if complete */
        if (seg + 1 >= traj->num_waypoints) {
            traj->is_active = false;
            adapter->status = MOTOR_STATUS_COMPLETE;

            /* Fill result */
            adapter->last_result.success = true;
            adapter->last_result.final_position = traj->waypoints[traj->num_waypoints - 1].position;
            adapter->last_result.actual_duration_ms = traj->elapsed_ms;
            adapter->last_result.accuracy = 1.0f;

            /* Invoke completion callback */
            if (adapter->complete_callback) {
                adapter->complete_callback(&adapter->last_result, adapter->complete_user_data);
            }

            emit_event(adapter, 3 /* MOTOR_EVENT_EXECUTION_COMPLETE */, &adapter->last_result);
            adapter->stats.successful_movements++;

            return false;  /* Execution complete */
        }

        /* Interpolate position within segment */
        float seg_start = traj->waypoints[seg].time_ms;
        float seg_end = traj->waypoints[seg + 1].time_ms;
        float t = (traj->elapsed_ms - seg_start) / (seg_end - seg_start);
        t = fmaxf(0.0f, fminf(1.0f, t));

        motor_vec3_t pos = lerp_vec3(&traj->waypoints[seg].position,
                                      &traj->waypoints[seg + 1].position, t);

        /* Generate motor command */
        uint32_t effector_id = (uint32_t)traj->region;
        if (effector_id < adapter->num_effectors) {
            motor_command_t cmd;
            memset(&cmd, 0, sizeof(cmd));
            cmd.effector_id = effector_id;
            cmd.target_position = pos;
            cmd.duration_ms = dt_ms;
            cmd.timestamp_ms = adapter->current_time_ms;

            enqueue_command(adapter, &cmd);
            adapter->stats.commands_generated++;
        }

        return true;  /* Still executing */
    }

    /* Process queued commands */
    if (adapter->queue_count > 0) {
        /* Just dequeue - actual output happens through get_next_command */
        return true;
    }

    /* No more work */
    adapter->status = MOTOR_STATUS_COMPLETE;
    return false;
}

bool motor_stop_execution(motor_adapter_t* adapter) {
    if (!adapter) return false;

    LOG_DEBUG("[%s] Stopping movement execution", MOTOR_LOG_MODULE);

    /* Clear active trajectory */
    if (adapter->active_trajectory) {
        adapter->active_trajectory->is_active = false;
        adapter->active_trajectory = NULL;
    }

    /* Clear command queue */
    adapter->queue_head = 0;
    adapter->queue_tail = 0;
    adapter->queue_count = 0;

    /* Generate stop commands for all active effectors */
    for (uint32_t i = 0; i < adapter->num_effectors; i++) {
        if (adapter->effectors[i].is_active) {
            motor_command_t stop_cmd;
            memset(&stop_cmd, 0, sizeof(stop_cmd));
            stop_cmd.effector_id = i;
            stop_cmd.target_position = adapter->effectors[i].position;
            stop_cmd.target_velocity.x = 0.0f;
            stop_cmd.target_velocity.y = 0.0f;
            stop_cmd.target_velocity.z = 0.0f;
            stop_cmd.duration_ms = 0.0f;
            stop_cmd.timestamp_ms = adapter->current_time_ms;

            enqueue_command(adapter, &stop_cmd);
        }
    }

    adapter->status = MOTOR_STATUS_IDLE;
    adapter->has_active_goal = false;

    emit_event(adapter, 4 /* MOTOR_EVENT_STOPPED */, NULL);

    return true;
}

bool motor_get_next_command(motor_adapter_t* adapter, motor_command_t* command) {
    if (!adapter || !command) return false;

    if (adapter->queue_count == 0) {
        return false;  /* Queue empty */
    }

    *command = adapter->command_queue[adapter->queue_head].command;
    adapter->command_queue[adapter->queue_head].is_valid = false;
    adapter->queue_head = (adapter->queue_head + 1) % adapter->queue_capacity;
    adapter->queue_count--;

    return true;
}

bool motor_get_result(const motor_adapter_t* adapter, motor_result_t* result) {
    if (!adapter || !result) return false;

    if (adapter->status != MOTOR_STATUS_COMPLETE &&
        adapter->status != MOTOR_STATUS_ERROR) {
        return false;
    }

    *result = adapter->last_result;
    return true;
}

/*=============================================================================
 * SENSORY FEEDBACK
 *===========================================================================*/

bool motor_update_feedback(motor_adapter_t* adapter,
                            uint32_t effector_id,
                            const motor_effector_state_t* state) {
    if (!adapter || !state || effector_id >= adapter->num_effectors) {
        return false;
    }

    /* Update internal state */
    adapter->effectors[effector_id] = *state;

    /* Apply corrections if executing with feedback enabled */
    if (adapter->status == MOTOR_STATUS_EXECUTING &&
        adapter->config.enable_feedback &&
        adapter->has_active_goal) {

        /* Compute position error */
        float error = compute_distance(&state->position,
                                        &adapter->current_goal.target_position);

        if (error > 0.1f) {  /* Error threshold */
            adapter->status = MOTOR_STATUS_CORRECTING;
            adapter->stats.corrections_applied++;

            /* Generate corrective command */
            motor_command_t correction;
            memset(&correction, 0, sizeof(correction));
            correction.effector_id = effector_id;
            correction.target_position = adapter->current_goal.target_position;
            correction.duration_ms = error * 10.0f;  /* Simple proportional */
            correction.timestamp_ms = adapter->current_time_ms;

            enqueue_command(adapter, &correction);
        }
    }

    return true;
}

bool motor_update_visual_feedback(motor_adapter_t* adapter,
                                   uint32_t effector_id,
                                   const motor_vec3_t* visual_position,
                                   float confidence) {
    if (!adapter || !visual_position || effector_id >= adapter->num_effectors) {
        return false;
    }

    /* Blend visual feedback with proprioception based on confidence */
    motor_effector_state_t* eff = &adapter->effectors[effector_id];

    float blend = confidence * 0.3f;  /* Visual feedback weight */
    eff->position.x = eff->position.x * (1.0f - blend) + visual_position->x * blend;
    eff->position.y = eff->position.y * (1.0f - blend) + visual_position->y * blend;
    eff->position.z = eff->position.z * (1.0f - blend) + visual_position->z * blend;

    return true;
}

/*=============================================================================
 * INTEGRATION INTERFACES
 *===========================================================================*/

bool motor_receive_bg_selection(motor_adapter_t* adapter,
                                 uint32_t action_id,
                                 float vigor) {
    if (!adapter) return false;

    LOG_DEBUG("[%s] Received BG action selection: action=%u, vigor=%.2f",
              MOTOR_LOG_MODULE, action_id, vigor);

    /* Scale movement parameters by vigor */
    if (adapter->has_active_goal) {
        adapter->current_goal.urgency = vigor;

        /* Vigor affects movement speed */
        if (adapter->active_trajectory && adapter->active_trajectory->is_active) {
            /* Would scale trajectory timing by vigor here */
        }
    }

    return true;
}

bool motor_receive_cerebellar_correction(motor_adapter_t* adapter,
                                          uint32_t effector_id,
                                          float timing_correction,
                                          float amplitude_correction) {
    if (!adapter || effector_id >= adapter->num_effectors) return false;

    LOG_DEBUG("[%s] Received cerebellar correction: effector=%u, timing=%.2fms, amp=%.2f",
              MOTOR_LOG_MODULE, effector_id, timing_correction, amplitude_correction);

    /* Apply correction to current trajectory if active */
    if (adapter->active_trajectory && adapter->active_trajectory->is_active) {
        trajectory_plan_t* traj = adapter->active_trajectory;

        /* Adjust timing */
        traj->elapsed_ms += timing_correction;

        /* Amplitude correction would scale velocity/force */
        adapter->stats.corrections_applied++;
    }

    return true;
}

/*=============================================================================
 * TRAINING INTERFACE
 *===========================================================================*/

bool motor_train_movement(motor_adapter_t* adapter,
                           const motor_vec3_t* target_position,
                           float learning_rate) {
    if (!adapter || !target_position) return false;
    if (!adapter->config.enable_training) return false;

    if (learning_rate <= 0.0f) {
        learning_rate = adapter->config.learning_rate;
    }

    /* Compute error from last movement */
    if (adapter->status == MOTOR_STATUS_COMPLETE) {
        motor_vec3_t error;
        error.x = target_position->x - adapter->last_result.final_position.x;
        error.y = target_position->y - adapter->last_result.final_position.y;
        error.z = target_position->z - adapter->last_result.final_position.z;

        float error_magnitude = sqrtf(error.x * error.x +
                                      error.y * error.y +
                                      error.z * error.z);

        adapter->stats.training_iterations++;
        adapter->stats.training_loss = error_magnitude;

        LOG_DEBUG("[%s] Training: error=%.4f, lr=%.4f",
                  MOTOR_LOG_MODULE, error_magnitude, learning_rate);
    }

    return true;
}

uint32_t motor_train_from_demonstration(motor_adapter_t* adapter,
                                         const trajectory_waypoint_t* waypoints,
                                         uint32_t num_waypoints,
                                         const char* program_name) {
    if (!adapter || !waypoints || num_waypoints < 2 || !program_name) {
        return 0;
    }

    LOG_DEBUG("[%s] Learning from demonstration: '%s' (%u waypoints)",
              MOTOR_LOG_MODULE, program_name, num_waypoints);

    /* Convert waypoints to motor commands */
    motor_command_t* commands = (motor_command_t*)nimcp_calloc(
        num_waypoints, sizeof(motor_command_t));
    if (!commands) return 0;

    for (uint32_t i = 0; i < num_waypoints; i++) {
        commands[i].effector_id = 0;  /* Default effector */
        commands[i].target_position = waypoints[i].position;
        commands[i].target_velocity = waypoints[i].velocity;
        commands[i].timestamp_ms = waypoints[i].time_ms;

        if (i + 1 < num_waypoints) {
            commands[i].duration_ms = waypoints[i + 1].time_ms - waypoints[i].time_ms;
        } else {
            commands[i].duration_ms = 0.0f;
        }
    }

    /* Store as motor program */
    uint32_t program_id = motor_store_program(adapter, program_name,
                                               commands, num_waypoints,
                                               MOVEMENT_TYPE_CONTINUOUS);

    nimcp_free(commands);

    return program_id;
}

/*=============================================================================
 * STATUS AND DIAGNOSTICS
 *===========================================================================*/

motor_status_t motor_get_status(const motor_adapter_t* adapter) {
    if (!adapter) return MOTOR_STATUS_ERROR;
    return adapter->status;
}

motor_error_t motor_get_last_error(const motor_adapter_t* adapter) {
    if (!adapter) return MOTOR_ERROR_INTERNAL;
    return adapter->last_error;
}

const char* motor_error_string(motor_error_t error) {
    switch (error) {
        case MOTOR_ERROR_NONE: return "No error";
        case MOTOR_ERROR_INVALID_INPUT: return "Invalid input";
        case MOTOR_ERROR_PLANNING_FAILURE: return "Planning failed";
        case MOTOR_ERROR_EXECUTION_FAILURE: return "Execution failed";
        case MOTOR_ERROR_TRAJECTORY_INFEASIBLE: return "Trajectory infeasible";
        case MOTOR_ERROR_EFFECTOR_CONFLICT: return "Effector conflict";
        case MOTOR_ERROR_TIMING_VIOLATION: return "Timing violation";
        case MOTOR_ERROR_BUFFER_OVERFLOW: return "Buffer overflow";
        case MOTOR_ERROR_INTERNAL: return "Internal error";
        default: return "Unknown error";
    }
}

const char* motor_status_string(motor_status_t status) {
    switch (status) {
        case MOTOR_STATUS_IDLE: return "Idle";
        case MOTOR_STATUS_PLANNING: return "Planning";
        case MOTOR_STATUS_PREPARING: return "Preparing";
        case MOTOR_STATUS_EXECUTING: return "Executing";
        case MOTOR_STATUS_CORRECTING: return "Correcting";
        case MOTOR_STATUS_COMPLETE: return "Complete";
        case MOTOR_STATUS_ERROR: return "Error";
        default: return "Unknown";
    }
}

bool motor_get_stats(const motor_adapter_t* adapter, motor_stats_t* stats) {
    if (!adapter || !stats) return false;
    *stats = adapter->stats;
    return true;
}

bool motor_get_config(const motor_adapter_t* adapter, motor_config_t* config) {
    if (!adapter || !config) return false;
    *config = adapter->config;
    return true;
}

bool motor_get_effector_state(const motor_adapter_t* adapter,
                               uint32_t effector_id,
                               motor_effector_state_t* state) {
    if (!adapter || !state || effector_id >= adapter->num_effectors) {
        return false;
    }
    *state = adapter->effectors[effector_id];
    return true;
}

/*=============================================================================
 * CALLBACK REGISTRATION
 *===========================================================================*/

bool motor_set_command_callback(motor_adapter_t* adapter,
                                 motor_command_callback_t callback,
                                 void* user_data) {
    if (!adapter) return false;
    adapter->command_callback = callback;
    adapter->command_user_data = user_data;
    return true;
}

bool motor_set_complete_callback(motor_adapter_t* adapter,
                                  motor_complete_callback_t callback,
                                  void* user_data) {
    if (!adapter) return false;
    adapter->complete_callback = callback;
    adapter->complete_user_data = user_data;
    return true;
}

bool motor_set_event_callback(motor_adapter_t* adapter,
                               motor_event_callback_t callback,
                               void* user_data) {
    if (!adapter) return false;
    adapter->event_callback = callback;
    adapter->event_user_data = user_data;
    return true;
}

/*=============================================================================
 * BIO-ASYNC COMMUNICATION API
 *===========================================================================*/

bio_module_context_t motor_get_bio_context(motor_adapter_t* adapter) {
    if (!adapter) return NULL;
    return adapter->bio_ctx;
}

uint32_t motor_process_bio_messages(motor_adapter_t* adapter, uint32_t max_messages) {
    if (!adapter || !adapter->bio_ctx) return 0;

    uint32_t processed = bio_router_process_inbox(adapter->bio_ctx, max_messages);
    if (processed > 0) {
        LOG_DEBUG("[%s] Processed %u bio-async messages", MOTOR_LOG_MODULE, processed);
    }
    return processed;
}

nimcp_bio_future_t motor_request_movement_async(
    motor_adapter_t* adapter,
    const motor_goal_t* goal) {

    if (!adapter || !adapter->bio_ctx || !goal) {
        LOG_WARNING("[%s] Cannot request movement: invalid arguments", MOTOR_LOG_MODULE);
        return NULL;
    }

    LOG_DEBUG("[%s] Requesting movement async to region %d", MOTOR_LOG_MODULE, goal->region);

    /* Create motor command request message */
    bio_msg_motor_command_request_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.type = BIO_MSG_MOTOR_COMMAND_REQUEST;
    msg.header.source_module = BIO_MODULE_MOTOR_CORTEX;
    msg.header.target_module = BIO_MODULE_MOTOR_CORTEX;
    msg.header.payload_size = sizeof(msg);
    msg.header.channel = adapter->default_channel;

    /* Encode goal in message */
    msg.phoneme = (uint8_t)goal->region;  /* Repurpose phoneme field for region */
    msg.duration_ms = goal->max_duration_ms;
    msg.intensity = goal->urgency;

    nimcp_bio_promise_t promise = bio_router_send_async(
        adapter->bio_ctx, &msg, sizeof(msg), adapter->default_channel);

    if (!promise) {
        LOG_ERROR("[%s] Failed to send movement request", MOTOR_LOG_MODULE);
        return NULL;
    }

    return nimcp_bio_promise_get_future(promise);
}

nimcp_error_t motor_broadcast_completion(
    motor_adapter_t* adapter,
    const motor_result_t* result) {

    if (!adapter || !result) {
        return NIMCP_BIO_ERROR_NOT_INITIALIZED;
    }

    if (!adapter->bio_ctx) {
        LOG_DEBUG("[%s] Cannot broadcast: bio-async not available", MOTOR_LOG_MODULE);
        return NIMCP_SUCCESS;
    }

    LOG_INFO("[%s] Broadcasting movement completion (success=%d, accuracy=%.2f)",
             MOTOR_LOG_MODULE, result->success, result->accuracy);

    bio_msg_motor_command_result_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.type = BIO_MSG_MOTOR_COMMAND_RESULT;
    msg.header.source_module = BIO_MODULE_MOTOR_CORTEX;
    msg.header.target_module = 0;  /* Broadcast */
    msg.header.payload_size = sizeof(msg);
    msg.header.channel = adapter->default_channel;
    msg.header.flags = BIO_MSG_FLAG_BROADCAST;

    /* Encode result */
    msg.timestamp_ms = adapter->current_time_ms;

    return bio_router_broadcast(adapter->bio_ctx, &msg, sizeof(msg));
}

/*=============================================================================
 * BIO-ASYNC MESSAGE HANDLERS (Implementation)
 *===========================================================================*/

static nimcp_error_t handle_motor_command_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data) {

    motor_adapter_t* adapter = (motor_adapter_t*)user_data;
    const bio_msg_motor_command_request_t* req = (const bio_msg_motor_command_request_t*)msg;

    if (!adapter || !req || msg_size < sizeof(bio_msg_motor_command_request_t)) {
        LOG_ERROR("[%s] Invalid motor command request", MOTOR_LOG_MODULE);
        return NIMCP_BIO_ERROR_NOT_INITIALIZED;
    }

    LOG_DEBUG("[%s] Handling motor command request", MOTOR_LOG_MODULE);

    /* Create goal from request */
    motor_goal_t goal;
    memset(&goal, 0, sizeof(goal));
    goal.region = (motor_region_t)req->phoneme;
    goal.max_duration_ms = req->duration_ms;
    goal.urgency = req->intensity;
    goal.precision_required = 0.1f;
    goal.type = MOVEMENT_TYPE_DISCRETE;

    /* Plan and begin execution */
    bool success = motor_plan_movement(adapter, &goal);
    if (success) {
        motor_begin_execution(adapter);
    }

    /* Build response */
    bio_msg_motor_command_result_t response;
    memset(&response, 0, sizeof(response));

    response.header.type = BIO_MSG_MOTOR_COMMAND_RESULT;
    response.header.source_module = BIO_MODULE_MOTOR_CORTEX;
    response.header.target_module = req->header.source_module;
    response.header.payload_size = sizeof(response);
    response.header.channel = req->header.channel;

    response.timestamp_ms = adapter->current_time_ms;

    if (response_promise) {
        nimcp_bio_promise_complete(response_promise, &response);
    }

    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_motor_stop_request(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data) {

    motor_adapter_t* adapter = (motor_adapter_t*)user_data;

    if (!adapter) {
        return NIMCP_BIO_ERROR_NOT_INITIALIZED;
    }

    LOG_DEBUG("[%s] Handling motor stop request", MOTOR_LOG_MODULE);

    motor_stop_execution(adapter);

    (void)msg;
    (void)msg_size;
    (void)response_promise;

    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_bg_action_selection(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data) {

    motor_adapter_t* adapter = (motor_adapter_t*)user_data;

    if (!adapter || !msg || msg_size == 0) {
        return NIMCP_BIO_ERROR_NOT_INITIALIZED;
    }

    /* Would decode BG action selection message and call motor_receive_bg_selection */
    LOG_DEBUG("[%s] Handling BG action selection", MOTOR_LOG_MODULE);

    (void)response_promise;

    return NIMCP_SUCCESS;
}

static nimcp_error_t handle_cerebellar_correction(
    const void* msg, size_t msg_size,
    nimcp_bio_promise_t response_promise, void* user_data) {

    motor_adapter_t* adapter = (motor_adapter_t*)user_data;

    if (!adapter || !msg || msg_size == 0) {
        return NIMCP_BIO_ERROR_NOT_INITIALIZED;
    }

    /* Would decode cerebellar correction message and apply */
    LOG_DEBUG("[%s] Handling cerebellar correction", MOTOR_LOG_MODULE);

    (void)response_promise;

    return NIMCP_SUCCESS;
}
