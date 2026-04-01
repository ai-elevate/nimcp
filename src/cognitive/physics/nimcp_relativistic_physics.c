/**
 * @file nimcp_relativistic_physics.c
 * @brief Relativistic Physics — special and general relativistic corrections
 *
 * WHAT: Lorentz transformations, relativistic Hamiltonian dynamics, time dilation
 * WHY:  High-energy physics, astrophysics, GPS, particle physics reasoning
 * HOW:  Relativistic Hamiltonian H = sqrt(p^2*c^2 + m^2*c^4) + V with
 *       symplectic integration, Schwarzschild weak-field for GR
 */

#include "cognitive/physics/nimcp_relativistic_physics.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>

#define LOG_TAG "RELATIVISTIC"

/* ============================================================================
 * Helpers
 * ============================================================================ */

static inline float rv_dot(wm_parietal_vec3_t a, wm_parietal_vec3_t b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline float rv_len2(wm_parietal_vec3_t v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

static inline float rv_len(wm_parietal_vec3_t v) {
    return sqrtf(rv_len2(v));
}

static inline wm_parietal_vec3_t rv_scale(wm_parietal_vec3_t v, float s) {
    return (wm_parietal_vec3_t){ v.x * s, v.y * s, v.z * s };
}

static inline wm_parietal_vec3_t rv_add(wm_parietal_vec3_t a, wm_parietal_vec3_t b) {
    return (wm_parietal_vec3_t){ a.x + b.x, a.y + b.y, a.z + b.z };
}

static inline wm_parietal_vec3_t rv_sub(wm_parietal_vec3_t a, wm_parietal_vec3_t b) {
    return (wm_parietal_vec3_t){ a.x - b.x, a.y - b.y, a.z - b.z };
}

/* ============================================================================
 * Core Physics Functions
 * ============================================================================ */

float relativistic_gamma(wm_parietal_vec3_t velocity) {
    float v2 = rv_len2(velocity);
    float beta2 = v2 / REL_C2;
    if (beta2 >= 1.0f) beta2 = 0.9999999f;  /* clamp to sub-luminal */
    return 1.0f / sqrtf(1.0f - beta2);
}

wm_parietal_vec3_t relativistic_momentum(float rest_mass, wm_parietal_vec3_t velocity) {
    float gamma = relativistic_gamma(velocity);
    return rv_scale(velocity, gamma * rest_mass);
}

float relativistic_total_energy(float rest_mass, wm_parietal_vec3_t velocity) {
    float gamma = relativistic_gamma(velocity);
    return gamma * rest_mass * REL_C2;
}

float relativistic_kinetic_energy(float rest_mass, wm_parietal_vec3_t velocity) {
    float gamma = relativistic_gamma(velocity);
    return (gamma - 1.0f) * rest_mass * REL_C2;
}

float relativistic_rest_energy(float rest_mass) {
    return rest_mass * REL_C2;
}

float relativistic_proper_time(float coord_dt, wm_parietal_vec3_t velocity) {
    float gamma = relativistic_gamma(velocity);
    return coord_dt / gamma;  /* proper time < coordinate time */
}

float relativistic_velocity_addition(float u, float v) {
    return (u + v) / (1.0f + u * v / REL_C2);
}

float relativistic_invariant_mass(rel_four_vector_t p) {
    float m2c2 = p.t * p.t - (p.x * p.x + p.y * p.y + p.z * p.z);
    if (m2c2 < 0) return 0;  /* massless or numerical error */
    return sqrtf(m2c2) / REL_C;
}

/* ============================================================================
 * Lorentz Boost
 * ============================================================================ */

rel_lorentz_boost_t relativistic_build_boost(wm_parietal_vec3_t frame_velocity) {
    rel_lorentz_boost_t boost;
    memset(&boost, 0, sizeof(boost));

    float v = rv_len(frame_velocity);
    if (v < 1e-10f) {
        /* Identity transform */
        boost.gamma = 1.0f;
        for (int i = 0; i < 4; i++) boost.matrix[i][i] = 1.0f;
        return boost;
    }

    float beta = v / REL_C;
    if (beta >= 1.0f) beta = 0.9999999f;
    float gamma = 1.0f / sqrtf(1.0f - beta * beta);

    boost.beta_x = frame_velocity.x / REL_C;
    boost.beta_y = frame_velocity.y / REL_C;
    boost.beta_z = frame_velocity.z / REL_C;
    boost.gamma = gamma;

    float bx = boost.beta_x, by = boost.beta_y, bz = boost.beta_z;
    float b2 = bx * bx + by * by + bz * bz;

    /* General Lorentz boost matrix (not just along one axis) */
    /* Lambda^0_0 = gamma */
    boost.matrix[0][0] = gamma;
    /* Lambda^0_i = -gamma * beta_i */
    boost.matrix[0][1] = -gamma * bx;
    boost.matrix[0][2] = -gamma * by;
    boost.matrix[0][3] = -gamma * bz;
    /* Lambda^i_0 = -gamma * beta_i */
    boost.matrix[1][0] = -gamma * bx;
    boost.matrix[2][0] = -gamma * by;
    boost.matrix[3][0] = -gamma * bz;
    /* Lambda^i_j = delta_ij + (gamma-1) * beta_i * beta_j / beta^2 */
    float g1_b2 = (b2 > 1e-20f) ? (gamma - 1.0f) / b2 : 0;
    boost.matrix[1][1] = 1.0f + g1_b2 * bx * bx;
    boost.matrix[1][2] = g1_b2 * bx * by;
    boost.matrix[1][3] = g1_b2 * bx * bz;
    boost.matrix[2][1] = g1_b2 * by * bx;
    boost.matrix[2][2] = 1.0f + g1_b2 * by * by;
    boost.matrix[2][3] = g1_b2 * by * bz;
    boost.matrix[3][1] = g1_b2 * bz * bx;
    boost.matrix[3][2] = g1_b2 * bz * by;
    boost.matrix[3][3] = 1.0f + g1_b2 * bz * bz;

    return boost;
}

rel_four_vector_t relativistic_boost_transform(const rel_lorentz_boost_t* boost,
                                                rel_four_vector_t vec) {
    rel_four_vector_t result;
    float in[4] = { vec.t, vec.x, vec.y, vec.z };
    float out[4] = {0};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            out[i] += boost->matrix[i][j] * in[j];
    result.t = out[0];
    result.x = out[1];
    result.y = out[2];
    result.z = out[3];
    return result;
}

/* ============================================================================
 * Gravitational Time Dilation (weak-field Schwarzschild)
 * ============================================================================ */

float relativistic_gravitational_dilation(const relativistic_engine_t* engine,
                                           wm_parietal_vec3_t position) {
    if (!engine) return 1.0f;
    float dilation = 1.0f;

    for (uint32_t i = 0; i < engine->num_gravity_sources; i++) {
        const rel_gravity_source_t* src = &engine->gravity_sources[i];
        float r = rv_len(rv_sub(position, src->position));
        if (r < 1e-6f) r = 1e-6f;
        /* dt_proper/dt_coord = sqrt(1 - r_s/r) where r_s = 2GM/c^2 */
        float rs_over_r = src->schwarzschild_r / r;
        if (rs_over_r >= 1.0f) rs_over_r = 0.999f; /* inside horizon clamp */
        dilation *= sqrtf(1.0f - rs_over_r);
    }
    return dilation;
}

/* ============================================================================
 * Update Particle State
 * ============================================================================ */

static void update_particle(rel_particle_t* p, const relativistic_engine_t* engine,
                             wm_parietal_vec3_t force, float dt) {
    if (!p->active || p->rest_mass <= 0) return;

    /* Relativistic equation of motion: dp/dt = F where p = gamma*m*v */

    /* Update momentum: p += F * dt */
    p->momentum.x += force.x * dt;
    p->momentum.y += force.y * dt;
    p->momentum.z += force.z * dt;

    /* Derive velocity from momentum: v = p / (gamma * m) */
    /* gamma = sqrt(1 + |p|^2 / (m^2 * c^2)) */
    float p2 = rv_len2(p->momentum);
    float m2c2 = p->rest_mass * p->rest_mass * REL_C2;
    float gamma = sqrtf(1.0f + p2 / m2c2);

    p->gamma = gamma;
    float inv_gamma_m = 1.0f / (gamma * p->rest_mass);
    p->velocity.x = p->momentum.x * inv_gamma_m;
    p->velocity.y = p->momentum.y * inv_gamma_m;
    p->velocity.z = p->momentum.z * inv_gamma_m;

    /* Clamp velocity to < c */
    float v = rv_len(p->velocity);
    if (v >= REL_C * 0.9999f) {
        float scale = REL_C * 0.9999f / v;
        p->velocity = rv_scale(p->velocity, scale);
    }

    /* Update position: x += v * dt */
    p->position.x += p->velocity.x * dt;
    p->position.y += p->velocity.y * dt;
    p->position.z += p->velocity.z * dt;

    /* Update derived quantities */
    p->kinetic_energy = (gamma - 1.0f) * p->rest_mass * REL_C2;
    p->total_energy = gamma * p->rest_mass * REL_C2;

    /* 4-momentum */
    p->four_momentum.t = p->total_energy / REL_C;
    p->four_momentum.x = p->momentum.x;
    p->four_momentum.y = p->momentum.y;
    p->four_momentum.z = p->momentum.z;

    /* Proper time: dtau = dt / gamma (special) * gravitational factor */
    float grav_dilation = relativistic_gravitational_dilation(engine, p->position);
    p->proper_time += dt / gamma * grav_dilation;
}

/* ============================================================================
 * Public API
 * ============================================================================ */

rel_config_t relativistic_default_config(void) {
    return (rel_config_t){
        .enable_special_relativity = true,
        .enable_general_relativity = false,
        .enable_mass_energy = true,
        .velocity_threshold = REL_THRESHOLD,
        .dt = 1e-9f,  /* 1 nanosecond default timestep */
    };
}

relativistic_engine_t* relativistic_create(const rel_config_t* config) {
    rel_config_t cfg = config ? *config : relativistic_default_config();

    relativistic_engine_t* engine = nimcp_calloc(1, sizeof(*engine));
    if (!engine) return NULL;

    engine->config = cfg;
    engine->initialized = true;

    LOG_INFO(LOG_TAG, "Relativistic engine created: SR=%s, GR=%s, E=mc2=%s",
             cfg.enable_special_relativity ? "yes" : "no",
             cfg.enable_general_relativity ? "yes" : "no",
             cfg.enable_mass_energy ? "yes" : "no");
    return engine;
}

void relativistic_destroy(relativistic_engine_t* engine) {
    if (!engine) return;
    nimcp_free(engine);
}

uint32_t relativistic_add_particle(relativistic_engine_t* engine,
                                    const rel_particle_t* particle) {
    if (!engine || !particle || engine->num_particles >= REL_MAX_PARTICLES)
        return UINT32_MAX;

    uint32_t id = engine->num_particles;
    engine->particles[id] = *particle;
    engine->particles[id].id = id;
    engine->particles[id].active = true;

    /* Compute initial derived quantities */
    rel_particle_t* p = &engine->particles[id];
    p->gamma = relativistic_gamma(p->velocity);
    p->momentum = relativistic_momentum(p->rest_mass, p->velocity);
    p->kinetic_energy = relativistic_kinetic_energy(p->rest_mass, p->velocity);
    p->total_energy = relativistic_total_energy(p->rest_mass, p->velocity);
    p->four_momentum.t = p->total_energy / REL_C;
    p->four_momentum.x = p->momentum.x;
    p->four_momentum.y = p->momentum.y;
    p->four_momentum.z = p->momentum.z;

    engine->num_particles = id + 1;
    return id;
}

uint32_t relativistic_add_gravity_source(relativistic_engine_t* engine,
                                          wm_parietal_vec3_t position, float mass) {
    if (!engine || engine->num_gravity_sources >= REL_MAX_GRAVITY_SOURCES)
        return UINT32_MAX;

    uint32_t id = engine->num_gravity_sources;
    engine->gravity_sources[id].position = position;
    engine->gravity_sources[id].mass = mass;
    engine->gravity_sources[id].schwarzschild_r = 2.0f * REL_G_NEWTON * mass / REL_C2;
    engine->num_gravity_sources = id + 1;
    return id;
}

int relativistic_step(relativistic_engine_t* engine, float dt) {
    if (!engine || !engine->initialized) return -1;
    if (dt <= 0) dt = engine->config.dt;

    float max_gamma = 1.0f;
    float max_beta = 0;
    float total_ke = 0, total_re = 0;
    uint32_t active = 0;

    for (uint32_t i = 0; i < engine->num_particles; i++) {
        rel_particle_t* p = &engine->particles[i];
        if (!p->active) continue;
        active++;

        /* Compute forces */
        wm_parietal_vec3_t force = {0, 0, 0};

        /* Gravitational force (Newtonian + optional GR corrections) */
        for (uint32_t g = 0; g < engine->num_gravity_sources; g++) {
            const rel_gravity_source_t* src = &engine->gravity_sources[g];
            wm_parietal_vec3_t r = rv_sub(p->position, src->position);
            float dist = rv_len(r);
            if (dist < 1e-6f) continue;

            /* F = -G*M*m/r^2 * r_hat */
            float F_mag = REL_G_NEWTON * src->mass * p->rest_mass / (dist * dist);

            /* GR correction: Schwarzschild precession term (1 + 3*rs/(2*r)) */
            if (engine->config.enable_general_relativity) {
                float rs = src->schwarzschild_r;
                F_mag *= (1.0f + 1.5f * rs / dist);
            }

            wm_parietal_vec3_t r_hat = rv_scale(r, -1.0f / dist);
            force = rv_add(force, rv_scale(r_hat, F_mag));
        }

        /* Electromagnetic force handled by EM engine (external coupling) */

        update_particle(p, engine, force, dt);

        /* Track statistics */
        if (p->gamma > max_gamma) max_gamma = p->gamma;
        float beta = rv_len(p->velocity) / REL_C;
        if (beta > max_beta) max_beta = beta;
        total_ke += p->kinetic_energy;
        total_re += p->rest_mass * REL_C2;
    }

    engine->coordinate_time += dt;
    engine->stats.step_count++;
    engine->stats.max_gamma = max_gamma;
    engine->stats.max_velocity_fraction = max_beta;
    engine->stats.total_kinetic_energy = total_ke;
    engine->stats.total_rest_energy = total_re;
    engine->stats.total_energy = total_ke + total_re;
    engine->stats.active_particles = active;

    if (engine->stats.step_count == 1) {
        engine->stats.energy_drift = 0;
    } else if (fabsf(engine->stats.total_energy) > 1e-30f) {
        static float initial_E = 0;
        if (engine->stats.step_count == 2) initial_E = engine->stats.total_energy;
        engine->stats.energy_drift = (engine->stats.total_energy - initial_E) / fabsf(initial_E);
    }

    return 0;
}

rel_stats_t relativistic_get_stats(const relativistic_engine_t* engine) {
    if (!engine) return (rel_stats_t){0};
    return engine->stats;
}
