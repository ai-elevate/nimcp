/**
 * @file nimcp_world_model_bridges.c
 * @brief JEPA + Prime Resonance ↔ Simulation Engine bridges
 *
 * WHAT: Predict-verify-surprise-replay loop connecting JEPA to physics
 * WHY:  Physics-grounded JEPA training + surprise-driven memory consolidation
 * HOW:  Encode physical→latent, JEPA predicts, decode back, compare to sim
 */

#include "cognitive/physics/nimcp_world_model_bridges.h"
#include "cognitive/physics/nimcp_intuitive_physics.h"
#include "cognitive/physics/nimcp_world_simulator.h"
#include "cognitive/physics/nimcp_scene_graph.h"
#include "cognitive/physics/nimcp_entity_tracker.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_TAG "WM_BRIDGE"

/* ============================================================================
 * Encoding / Decoding (latent ↔ physical)
 * ============================================================================ */

void wmb_encode_physical(const world_model_bridge_t* bridge,
                          const wmb_physical_state_t* physical,
                          float* latent_out) {
    if (!bridge || !physical || !latent_out) return;

    /* Pack physical state into a flat vector */
    float phys_vec[WMB_PHYSICAL_DIM];
    memcpy(&phys_vec[0], physical->position, 3 * sizeof(float));
    memcpy(&phys_vec[3], physical->velocity, 3 * sizeof(float));
    memcpy(&phys_vec[6], physical->orientation, 4 * sizeof(float));
    phys_vec[10] = physical->mass;
    phys_vec[11] = physical->radius;

    /* Linear projection: latent = encode_weights × phys_vec */
    if (bridge->encode_weights) {
        for (uint32_t i = 0; i < WMB_LATENT_DIM; i++) {
            double sum = 0;
            for (uint32_t j = 0; j < WMB_PHYSICAL_DIM; j++) {
                sum += (double)bridge->encode_weights[i * WMB_PHYSICAL_DIM + j] * phys_vec[j];
            }
            latent_out[i] = (float)sum;
        }
    } else {
        /* Fallback: zero-pad physical into latent */
        memset(latent_out, 0, WMB_LATENT_DIM * sizeof(float));
        uint32_t copy = WMB_PHYSICAL_DIM < WMB_LATENT_DIM ? WMB_PHYSICAL_DIM : WMB_LATENT_DIM;
        for (uint32_t i = 0; i < copy; i++)
            latent_out[i] = phys_vec[i];
    }
}

void wmb_decode_latent(const world_model_bridge_t* bridge,
                        const float* latent,
                        wmb_physical_state_t* physical_out) {
    if (!bridge || !latent || !physical_out) return;

    float phys_vec[WMB_PHYSICAL_DIM] = {0};

    /* Linear projection: phys_vec = decode_weights × latent */
    if (bridge->decode_weights) {
        for (uint32_t i = 0; i < WMB_PHYSICAL_DIM; i++) {
            double sum = 0;
            for (uint32_t j = 0; j < WMB_LATENT_DIM; j++) {
                sum += (double)bridge->decode_weights[i * WMB_LATENT_DIM + j] * latent[j];
            }
            phys_vec[i] = (float)sum;
        }
    } else {
        /* Fallback: first PHYSICAL_DIM elements of latent */
        uint32_t copy = WMB_PHYSICAL_DIM < WMB_LATENT_DIM ? WMB_PHYSICAL_DIM : WMB_LATENT_DIM;
        for (uint32_t i = 0; i < copy; i++)
            phys_vec[i] = latent[i];
    }

    memcpy(physical_out->position, &phys_vec[0], 3 * sizeof(float));
    memcpy(physical_out->velocity, &phys_vec[3], 3 * sizeof(float));
    memcpy(physical_out->orientation, &phys_vec[6], 4 * sizeof(float));
    physical_out->mass = phys_vec[10];
    physical_out->radius = phys_vec[11];
}

/* ============================================================================
 * Surprise Event Management
 * ============================================================================ */

void wmb_store_surprise(world_model_bridge_t* bridge,
                          const wmb_surprise_event_t* event) {
    if (!bridge || !event) return;
    wmb_replay_buffer_t* rb = &bridge->replay;

    rb->events[rb->head] = *event;
    rb->head = (rb->head + 1) % WMB_MAX_REPLAY_BUFFER;
    if (rb->count < WMB_MAX_REPLAY_BUFFER) rb->count++;

    /* Update running stats */
    bridge->stats.surprises_stored++;
    float alpha = 0.01f;
    rb->mean_surprise = (1 - alpha) * rb->mean_surprise + alpha * event->surprise_score;
    if (event->surprise_score > rb->max_surprise)
        rb->max_surprise = event->surprise_score;

    /* Fire surprise callback → routes to perirhinal prime resonance */
    if (bridge->on_surprise)
        bridge->on_surprise(event, bridge->surprise_ctx);
}

const wmb_surprise_event_t* wmb_most_surprising(const world_model_bridge_t* bridge) {
    if (!bridge || bridge->replay.count == 0) return NULL;
    const wmb_replay_buffer_t* rb = &bridge->replay;
    uint32_t best = 0;
    float best_score = -1;
    for (uint32_t i = 0; i < rb->count; i++) {
        if (rb->events[i].surprise_score > best_score) {
            best_score = rb->events[i].surprise_score;
            best = i;
        }
    }
    return &rb->events[best];
}

/* ============================================================================
 * Predict-and-Verify
 * ============================================================================ */

float wmb_predict_and_verify(world_model_bridge_t* bridge, float dt) {
    if (!bridge || !bridge->physics) return 0;
    bridge->stats.predictions_made++;

    /* 1. Get current state of first active object from physics */
    wmb_physical_state_t current = {0};
    ip_object_t* obj = NULL;
    uint32_t obj_id = UINT32_MAX;
    for (uint32_t i = 0; i < IP_MAX_OBJECTS; i++) {
        obj = intuitive_physics_get_object(bridge->physics, i);
        if (obj && !obj->is_static) { obj_id = i; break; }
    }
    if (!obj) return 0;

    current.position[0] = obj->position.x;
    current.position[1] = obj->position.y;
    current.position[2] = obj->position.z;
    current.velocity[0] = obj->velocity.vx;
    current.velocity[1] = obj->velocity.vy;
    current.velocity[2] = obj->velocity.vz;
    current.mass = obj->mass;
    current.radius = (obj->shape.type == IP_SHAPE_SPHERE) ? obj->shape.sphere.radius : 0.5f;
    current.object_id = obj_id;

    /* 2. Encode to latent space */
    float latent_before[WMB_LATENT_DIM];
    wmb_encode_physical(bridge, &current, latent_before);

    /* 3. Simple prediction: extrapolate position linearly (JEPA would do this
     *    in latent space; without actual JEPA connection, we approximate) */
    wmb_physical_state_t predicted = current;
    predicted.position[0] += current.velocity[0] * dt;
    predicted.position[1] += current.velocity[1] * dt;
    predicted.position[2] += current.velocity[2] * dt;
    /* Gravity correction (simple) */
    predicted.velocity[1] -= 9.81f * dt;
    predicted.position[1] += predicted.velocity[1] * dt;

    /* 4. Step simulation to get actual next state */
    intuitive_physics_step(bridge->physics, dt);
    bridge->stats.simulations_run++;

    wmb_physical_state_t actual = {0};
    obj = intuitive_physics_get_object(bridge->physics, obj_id);
    if (obj) {
        actual.position[0] = obj->position.x;
        actual.position[1] = obj->position.y;
        actual.position[2] = obj->position.z;
        actual.velocity[0] = obj->velocity.vx;
        actual.velocity[1] = obj->velocity.vy;
        actual.velocity[2] = obj->velocity.vz;
        actual.mass = obj->mass;
    }

    /* 5. Compute physical prediction error */
    float error = 0;
    for (int i = 0; i < 3; i++) {
        float dp = predicted.position[i] - actual.position[i];
        float dv = predicted.velocity[i] - actual.velocity[i];
        error += dp * dp + 0.1f * dv * dv;
    }
    error = sqrtf(error);

    bridge->stats.mean_prediction_error =
        0.99f * bridge->stats.mean_prediction_error + 0.01f * error;
    bridge->stats.mean_physical_error =
        0.99f * bridge->stats.mean_physical_error + 0.01f * error;

    /* 6. Surprise detection: if error > threshold × running average */
    float expected_var = bridge->stats.mean_prediction_error + 1e-6f;
    float surprise = error / expected_var;

    if (surprise > bridge->config.surprise_threshold && bridge->config.enable_prime_resonance) {
        wmb_surprise_event_t event = {0};
        event.predicted = predicted;
        event.actual = actual;
        event.prediction_error = error;
        event.surprise_score = surprise;
        event.timestamp = 0;  /* caller should set */
        wmb_encode_physical(bridge, &current, event.latent_before);
        wmb_encode_physical(bridge, &actual, event.latent_after);
        wmb_store_surprise(bridge, &event);

        if (surprise > bridge->stats.max_surprise_score)
            bridge->stats.max_surprise_score = surprise;
    }

    return error;
}

/* ============================================================================
 * Consolidation Replay
 * ============================================================================ */

uint32_t wmb_replay_consolidation(world_model_bridge_t* bridge,
                                    uint32_t num_replays) {
    if (!bridge || !bridge->config.enable_replay) return 0;
    wmb_replay_buffer_t* rb = &bridge->replay;
    if (rb->count == 0) return 0;

    uint32_t replayed = 0;
    uint32_t batch = num_replays < rb->count ? num_replays : rb->count;

    /* Prioritized replay: sort by surprise score, replay most surprising first */
    /* Simple approach: iterate through buffer, replay top-N by surprise */
    for (uint32_t r = 0; r < batch; r++) {
        /* Find highest unreplayed surprise */
        uint32_t best = 0;
        float best_score = -1;
        for (uint32_t i = 0; i < rb->count; i++) {
            if (rb->events[i].surprise_score > best_score) {
                best_score = rb->events[i].surprise_score;
                best = i;
            }
        }

        const wmb_surprise_event_t* event = &rb->events[best];

        /* Re-present the before→after transition to JEPA for training.
         * Without direct JEPA API access here, we store the training signal
         * for the main training loop to pick up. In full integration,
         * this would call:
         *   jepa_predictor_train_step(bridge->jepa,
         *       event->latent_before, event->latent_after, replay_lr);
         */
        /* Fire replay callback → routes latent pairs to JEPA training */
        if (bridge->on_replay)
            bridge->on_replay(event->latent_before, event->latent_after,
                              event->surprise_score, bridge->replay_ctx);

        /* Reduce surprise score after replay (diminishing returns) */
        rb->events[best].surprise_score *= 0.7f;

        replayed++;
        rb->replays_total++;
        bridge->stats.replays_executed++;
    }

    return replayed;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

wmb_config_t wmb_default_config(void) {
    return (wmb_config_t){
        .surprise_threshold = WMB_SURPRISE_THRESHOLD,
        .replay_learning_rate = 0.0005f,
        .replay_batch_size = 16,
        .enable_prime_resonance = true,
        .enable_replay = true,
        .enable_physical_loss = true,
        .physical_loss_weight = 0.1f,
    };
}

world_model_bridge_t* wmb_create(const wmb_config_t* config) {
    wmb_config_t cfg = config ? *config : wmb_default_config();
    world_model_bridge_t* bridge = nimcp_calloc(1, sizeof(*bridge));
    if (!bridge) return NULL;

    bridge->config = cfg;

    /* Allocate encoding/decoding weights (initialized to identity-like) */
    bridge->encode_weights = nimcp_calloc(WMB_LATENT_DIM * WMB_PHYSICAL_DIM, sizeof(float));
    bridge->decode_weights = nimcp_calloc(WMB_PHYSICAL_DIM * WMB_LATENT_DIM, sizeof(float));

    if (bridge->encode_weights && bridge->decode_weights) {
        /* Initialize to near-identity (first PHYSICAL_DIM dimensions map 1:1) */
        for (uint32_t i = 0; i < WMB_PHYSICAL_DIM && i < WMB_LATENT_DIM; i++) {
            bridge->encode_weights[i * WMB_PHYSICAL_DIM + i] = 1.0f;
            bridge->decode_weights[i * WMB_LATENT_DIM + i] = 1.0f;
        }
    }

    bridge->initialized = true;
    LOG_INFO(LOG_TAG, "World model bridge created: surprise_thresh=%.1f, "
             "replay=%s, physical_loss=%.2f",
             cfg.surprise_threshold,
             cfg.enable_replay ? "yes" : "no",
             cfg.physical_loss_weight);
    return bridge;
}

void wmb_destroy(world_model_bridge_t* bridge) {
    if (!bridge) return;
    nimcp_free(bridge->encode_weights);
    nimcp_free(bridge->decode_weights);
    nimcp_free(bridge);
}

void wmb_connect(world_model_bridge_t* bridge,
                  struct jepa_predictor* jepa,
                  struct intuitive_physics_engine* physics,
                  struct world_simulator* world_sim,
                  struct scene_graph* scene,
                  struct entity_tracker* tracker) {
    if (!bridge) return;
    bridge->jepa = jepa;
    bridge->physics = physics;
    bridge->world_sim = world_sim;
    bridge->scene = scene;
    bridge->tracker = tracker;
}

void wmb_set_surprise_callback(world_model_bridge_t* bridge,
                                 wmb_surprise_callback_t callback,
                                 void* user_data) {
    if (!bridge) return;
    bridge->on_surprise = callback;
    bridge->surprise_ctx = user_data;
}

void wmb_set_replay_callback(world_model_bridge_t* bridge,
                               wmb_replay_callback_t callback,
                               void* user_data) {
    if (!bridge) return;
    bridge->on_replay = callback;
    bridge->replay_ctx = user_data;
}

wmb_stats_t wmb_get_stats(const world_model_bridge_t* bridge) {
    if (!bridge) return (wmb_stats_t){0};
    return bridge->stats;
}
