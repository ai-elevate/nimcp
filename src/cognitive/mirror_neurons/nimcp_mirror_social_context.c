/**
 * @file nimcp_mirror_social_context.c
 * @brief Social Context Modulation for Mirror Neurons - Implementation
 * @version 1.0.0
 * @date 2025-01-05
 *
 * WHAT: Social context modulation of mirror neuron activation
 * WHY:  Mirror neuron response varies with social context - in-group, hierarchy, cultural factors
 * HOW:  Modulation factors scale mirror neuron activation based on social relationships
 */

#include "cognitive/mirror_neurons/nimcp_mirror_social_context.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/time/nimcp_time.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define EPSILON                    1e-6f
#define DECAY_TIME_CONSTANT_MS     86400000  /* 24 hours in ms */
#define TRUST_UPDATE_RATE          0.05f
#define COMPETENCE_UPDATE_RATE     0.1f
#define AGENT_HASH_PRIME           37

/* ============================================================================
 * Internal Data Structures
 * ============================================================================ */

/**
 * @brief Internal social context system structure
 */
struct social_context_system {
    /* Agent tracking */
    agent_social_context_t agents[NIMCP_SOCIAL_MAX_AGENTS];
    uint32_t agent_count;
    bool agent_slots_used[NIMCP_SOCIAL_MAX_AGENTS];

    /* Configuration */
    social_context_config_t config;

    /* Statistics */
    social_context_stats_t stats;

    /* Integration */
    mirror_neurons_t connected_mirror;
};

/* ============================================================================
 * Internal Helper Functions
 * ============================================================================ */

/**
 * @brief Clamp float value to range
 */
static inline float clamp_f(float value, float min_val, float max_val) {
    if (value < min_val) return min_val;
    if (value > max_val) return max_val;
    return value;
}

/**
 * @brief Simple hash function for agent ID to slot index
 */
static uint32_t hash_agent_id(uint32_t agent_id) {
    return (agent_id * AGENT_HASH_PRIME) % NIMCP_SOCIAL_MAX_AGENTS;
}

/**
 * @brief Find agent slot index (returns -1 if not found)
 */
static int32_t find_agent_slot(const struct social_context_system* sys,
                               uint32_t agent_id) {
    if (!sys) return -1;

    /* Start at hash position, linear probe on collision */
    uint32_t start = hash_agent_id(agent_id);
    for (uint32_t i = 0; i < NIMCP_SOCIAL_MAX_AGENTS; i++) {
        uint32_t idx = (start + i) % NIMCP_SOCIAL_MAX_AGENTS;
        if (sys->agent_slots_used[idx] && sys->agents[idx].agent_id == agent_id) {
            return (int32_t)idx;
        }
    }
    return -1;
}

/**
 * @brief Find or create agent slot (returns -1 if full)
 */
static int32_t find_or_create_agent_slot(struct social_context_system* sys,
                                          uint32_t agent_id) {
    if (!sys) return -1;

    /* First try to find existing */
    int32_t existing = find_agent_slot(sys, agent_id);
    if (existing >= 0) return existing;

    /* Check capacity */
    if (sys->agent_count >= NIMCP_SOCIAL_MAX_AGENTS) {
        nimcp_log(LOG_LEVEL_WARN, "Social context: max agents (%d) reached",
                  NIMCP_SOCIAL_MAX_AGENTS);
        return -1;
    }

    /* Find empty slot starting at hash position */
    uint32_t start = hash_agent_id(agent_id);
    for (uint32_t i = 0; i < NIMCP_SOCIAL_MAX_AGENTS; i++) {
        uint32_t idx = (start + i) % NIMCP_SOCIAL_MAX_AGENTS;
        if (!sys->agent_slots_used[idx]) {
            /* Initialize new agent */
            memset(&sys->agents[idx], 0, sizeof(agent_social_context_t));
            sys->agents[idx].agent_id = agent_id;
            sys->agents[idx].modulation = social_modulation_init();
            sys->agents[idx].trust_level = 0.5f;
            sys->agents[idx].competence_estimate = 0.5f;

            sys->agent_slots_used[idx] = true;
            sys->agent_count++;
            sys->stats.num_tracked_agents = sys->agent_count;

            return (int32_t)idx;
        }
    }
    return -1;
}

/**
 * @brief Compute gain from modulation factors
 */
static float compute_gain_internal(const social_modulation_t* mod,
                                    const social_context_config_t* cfg) {
    if (!mod || !cfg) return 1.0f;

    float gain = 1.0f;

    /* In-group affinity contribution (0-1 maps to 0.5-1.5 range) */
    if (cfg->enable_ingroup_bias) {
        float affinity_contrib = mod->in_group_affinity;
        gain += cfg->ingroup_weight * (affinity_contrib - 0.5f);
    }

    /* Hierarchy contribution (-1 to 1 maps to boost/reduction) */
    if (cfg->enable_hierarchy_effects) {
        /* Positive hierarchy (dominant/expert) boosts mirroring */
        gain += cfg->hierarchy_weight * mod->social_hierarchy * 0.5f;
    }

    /* Cultural familiarity contribution */
    if (cfg->enable_cultural_effects) {
        float cultural_contrib = mod->cultural_familiarity;
        gain += cfg->cultural_weight * (cultural_contrib - 0.5f);
    }

    /* Emotional valence contribution */
    if (cfg->enable_emotional_effects) {
        /* Positive emotions boost, negative reduce */
        gain += cfg->emotional_weight * mod->emotional_valence * 0.5f;
    }

    /* Clamp to configured range */
    return clamp_f(gain, cfg->min_gain, cfg->max_gain);
}

/**
 * @brief Map continuous affinity to categorical group
 */
static social_group_t affinity_to_group(float affinity) {
    if (affinity >= 0.8f) return SOCIAL_GROUP_ALLY;
    if (affinity >= 0.6f) return SOCIAL_GROUP_INGROUP;
    if (affinity >= 0.4f) return SOCIAL_GROUP_NEUTRAL;
    if (affinity >= 0.2f) return SOCIAL_GROUP_OUTGROUP;
    return SOCIAL_GROUP_RIVAL;
}

/**
 * @brief Map continuous hierarchy to categorical rank
 */
static social_rank_t hierarchy_to_rank(float hierarchy) {
    if (hierarchy >= 0.6f) return SOCIAL_RANK_EXPERT;
    if (hierarchy >= 0.3f) return SOCIAL_RANK_DOMINANT;
    if (hierarchy >= -0.3f) return SOCIAL_RANK_PEER;
    if (hierarchy >= -0.6f) return SOCIAL_RANK_SUBORDINATE;
    return SOCIAL_RANK_NOVICE;
}

/**
 * @brief Map continuous cultural to categorical context
 */
static cultural_context_t cultural_to_context(float familiarity) {
    if (familiarity >= 0.8f) return CULTURAL_NATIVE;
    if (familiarity >= 0.5f) return CULTURAL_FAMILIAR;
    if (familiarity >= 0.2f) return CULTURAL_FOREIGN;
    return CULTURAL_UNKNOWN;
}

/**
 * @brief Map continuous emotional to categorical context
 */
static emotional_context_t emotional_to_context(float valence) {
    if (valence >= 0.5f) return EMOTIONAL_FRIENDLY;
    if (valence >= 0.2f) return EMOTIONAL_POSITIVE;
    if (valence >= -0.2f) return EMOTIONAL_NEUTRAL;
    if (valence >= -0.5f) return EMOTIONAL_NEGATIVE;
    return EMOTIONAL_HOSTILE;
}

/* ============================================================================
 * Core API - Lifecycle Management
 * ============================================================================ */

social_context_t social_context_create(const social_context_config_t* config) {
    struct social_context_system* sys = nimcp_malloc(sizeof(struct social_context_system));
    if (!sys) {
        nimcp_log(LOG_LEVEL_ERROR, "Social context: failed to allocate system");
        return NULL;
    }

    memset(sys, 0, sizeof(struct social_context_system));

    /* Apply configuration */
    sys->config = config ? *config : social_context_get_default_config();

    /* Initialize statistics */
    sys->stats.num_tracked_agents = 0;
    sys->stats.total_lookups = 0;
    sys->stats.cache_hits = 0;
    sys->stats.avg_gain = 1.0f;
    sys->stats.avg_affinity = sys->config.default_affinity;
    sys->stats.avg_hierarchy = sys->config.default_hierarchy;
    sys->stats.last_update = nimcp_time_now_us() / 1000;

    nimcp_log(LOG_LEVEL_INFO, "Social context system created");
    return sys;
}

void social_context_destroy(social_context_t ctx) {
    if (!ctx) return;

    nimcp_log(LOG_LEVEL_INFO, "Social context system destroyed (tracked %u agents)",
              ctx->agent_count);
    nimcp_free(ctx);
}

social_context_config_t social_context_get_default_config(void) {
    social_context_config_t cfg;

    cfg.ingroup_weight = NIMCP_SOCIAL_INGROUP_WEIGHT;
    cfg.hierarchy_weight = NIMCP_SOCIAL_HIERARCHY_WEIGHT;
    cfg.cultural_weight = NIMCP_SOCIAL_CULTURAL_WEIGHT;
    cfg.emotional_weight = NIMCP_SOCIAL_EMOTIONAL_WEIGHT;

    cfg.min_gain = NIMCP_SOCIAL_MIN_GAIN;
    cfg.max_gain = NIMCP_SOCIAL_MAX_GAIN;

    cfg.learning_rate = NIMCP_SOCIAL_LEARNING_RATE;
    cfg.decay_rate = 0.001f;

    cfg.enable_ingroup_bias = true;
    cfg.enable_hierarchy_effects = true;
    cfg.enable_cultural_effects = true;
    cfg.enable_emotional_effects = true;

    cfg.default_affinity = NIMCP_SOCIAL_DEFAULT_AFFINITY;
    cfg.default_hierarchy = NIMCP_SOCIAL_DEFAULT_HIERARCHY;
    cfg.default_cultural = NIMCP_SOCIAL_DEFAULT_CULTURAL;

    return cfg;
}

/* ============================================================================
 * Core API - Modulation Factor Setting
 * ============================================================================ */

bool social_context_set_affinity(social_context_t ctx,
                                  uint32_t agent_id,
                                  float affinity,
                                  float confidence) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    ctx->agents[slot].modulation.in_group_affinity = clamp_f(affinity, 0.0f, 1.0f);
    ctx->agents[slot].modulation.affinity_confidence = clamp_f(confidence, 0.0f, 1.0f);
    ctx->agents[slot].modulation.group_type = affinity_to_group(affinity);
    ctx->agents[slot].modulation.gain_valid = false;

    return true;
}

bool social_context_set_hierarchy(social_context_t ctx,
                                   uint32_t agent_id,
                                   float hierarchy,
                                   float confidence) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    ctx->agents[slot].modulation.social_hierarchy = clamp_f(hierarchy, -1.0f, 1.0f);
    ctx->agents[slot].modulation.hierarchy_confidence = clamp_f(confidence, 0.0f, 1.0f);
    ctx->agents[slot].modulation.rank_type = hierarchy_to_rank(hierarchy);
    ctx->agents[slot].modulation.gain_valid = false;

    return true;
}

bool social_context_set_cultural(social_context_t ctx,
                                  uint32_t agent_id,
                                  float familiarity,
                                  float confidence) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    ctx->agents[slot].modulation.cultural_familiarity = clamp_f(familiarity, 0.0f, 1.0f);
    ctx->agents[slot].modulation.cultural_confidence = clamp_f(confidence, 0.0f, 1.0f);
    ctx->agents[slot].modulation.cultural_type = cultural_to_context(familiarity);
    ctx->agents[slot].modulation.gain_valid = false;

    return true;
}

bool social_context_set_emotional(social_context_t ctx,
                                   uint32_t agent_id,
                                   float valence,
                                   float confidence) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    ctx->agents[slot].modulation.emotional_valence = clamp_f(valence, -1.0f, 1.0f);
    ctx->agents[slot].modulation.emotional_confidence = clamp_f(confidence, 0.0f, 1.0f);
    ctx->agents[slot].modulation.emotional_type = emotional_to_context(valence);
    ctx->agents[slot].modulation.gain_valid = false;

    return true;
}

bool social_context_set_modulation(social_context_t ctx,
                                    uint32_t agent_id,
                                    const social_modulation_t* modulation) {
    if (!ctx || !modulation) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    ctx->agents[slot].modulation = *modulation;
    ctx->agents[slot].modulation.gain_valid = false;

    return true;
}

/* ============================================================================
 * Core API - Modulation Application
 * ============================================================================ */

bool social_context_get_modulation(social_context_t ctx,
                                    uint32_t agent_id,
                                    social_modulation_t* out_modulation) {
    if (!ctx || !out_modulation) return false;

    ctx->stats.total_lookups++;

    int32_t slot = find_agent_slot(ctx, agent_id);
    if (slot < 0) {
        /* Return defaults for unknown agent */
        *out_modulation = social_modulation_init();
        return true;
    }

    ctx->stats.cache_hits++;
    *out_modulation = ctx->agents[slot].modulation;
    return true;
}

float social_context_compute_gain(social_context_t ctx, uint32_t agent_id) {
    if (!ctx) return 1.0f;

    int32_t slot = find_agent_slot(ctx, agent_id);
    if (slot < 0) {
        /* Use defaults for unknown agent */
        social_modulation_t defaults = social_modulation_init();
        return compute_gain_internal(&defaults, &ctx->config);
    }

    /* Check cached gain */
    if (ctx->agents[slot].modulation.gain_valid) {
        return ctx->agents[slot].modulation.computed_gain;
    }

    /* Compute and cache */
    float gain = compute_gain_internal(&ctx->agents[slot].modulation, &ctx->config);
    ctx->agents[slot].modulation.computed_gain = gain;
    ctx->agents[slot].modulation.gain_valid = true;

    return gain;
}

float social_context_apply_modulation(social_context_t ctx,
                                       uint32_t agent_id,
                                       float activation) {
    float gain = social_context_compute_gain(ctx, agent_id);
    return clamp_f(activation * gain, 0.0f, 1.0f);
}

/* ============================================================================
 * Core API - Social Learning
 * ============================================================================ */

bool social_context_learn_interaction(social_context_t ctx,
                                       uint32_t agent_id,
                                       float outcome,
                                       uint32_t interaction_type) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    (void)interaction_type;  /* Reserved for future context-specific learning */

    float lr = ctx->config.learning_rate;
    agent_social_context_t* agent = &ctx->agents[slot];

    /* Positive outcomes increase trust and affinity */
    float delta = clamp_f(outcome, -1.0f, 1.0f);
    agent->trust_level = clamp_f(
        agent->trust_level + TRUST_UPDATE_RATE * delta, 0.0f, 1.0f);

    /* Update affinity based on outcome */
    agent->modulation.in_group_affinity = clamp_f(
        agent->modulation.in_group_affinity + lr * delta * 0.5f, 0.0f, 1.0f);
    agent->modulation.group_type = affinity_to_group(agent->modulation.in_group_affinity);

    /* Increase confidence with interaction */
    agent->modulation.affinity_confidence = clamp_f(
        agent->modulation.affinity_confidence + 0.1f, 0.0f, 1.0f);

    /* Invalidate cached gain */
    agent->modulation.gain_valid = false;
    agent->interaction_count++;

    return true;
}

bool social_context_update_competence(social_context_t ctx,
                                       uint32_t agent_id,
                                       float demonstrated_skill) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    agent_social_context_t* agent = &ctx->agents[slot];
    float skill = clamp_f(demonstrated_skill, 0.0f, 1.0f);

    /* Exponential moving average of competence */
    agent->competence_estimate = agent->competence_estimate * (1.0f - COMPETENCE_UPDATE_RATE) +
                                  skill * COMPETENCE_UPDATE_RATE;

    /* High competence increases perceived hierarchy */
    if (skill > 0.7f) {
        agent->modulation.social_hierarchy = clamp_f(
            agent->modulation.social_hierarchy + 0.05f, -1.0f, 1.0f);
        agent->modulation.rank_type = hierarchy_to_rank(agent->modulation.social_hierarchy);
        agent->modulation.hierarchy_confidence = clamp_f(
            agent->modulation.hierarchy_confidence + 0.1f, 0.0f, 1.0f);
        agent->modulation.gain_valid = false;
    }

    return true;
}

bool social_context_decay(social_context_t ctx, uint32_t delta_time_ms) {
    if (!ctx) return false;

    float decay_factor = expf(-(float)delta_time_ms / (float)DECAY_TIME_CONSTANT_MS *
                               ctx->config.decay_rate);

    for (uint32_t i = 0; i < NIMCP_SOCIAL_MAX_AGENTS; i++) {
        if (!ctx->agent_slots_used[i]) continue;

        agent_social_context_t* agent = &ctx->agents[i];

        /* Decay confidence values toward uncertainty */
        agent->modulation.affinity_confidence *= decay_factor;
        agent->modulation.hierarchy_confidence *= decay_factor;
        agent->modulation.cultural_confidence *= decay_factor;
        agent->modulation.emotional_confidence *= decay_factor;

        /* Decay factors toward defaults */
        float affinity_diff = agent->modulation.in_group_affinity - ctx->config.default_affinity;
        agent->modulation.in_group_affinity -= affinity_diff * (1.0f - decay_factor);

        float hierarchy_diff = agent->modulation.social_hierarchy - ctx->config.default_hierarchy;
        agent->modulation.social_hierarchy -= hierarchy_diff * (1.0f - decay_factor);

        float cultural_diff = agent->modulation.cultural_familiarity - ctx->config.default_cultural;
        agent->modulation.cultural_familiarity -= cultural_diff * (1.0f - decay_factor);

        /* Emotional state decays faster toward neutral */
        agent->modulation.emotional_valence *= decay_factor;

        /* Invalidate cached gains */
        agent->modulation.gain_valid = false;
    }

    ctx->stats.last_update = nimcp_time_now_us() / 1000;
    return true;
}

/* ============================================================================
 * Core API - Query Functions
 * ============================================================================ */

bool social_context_get_agent(social_context_t ctx,
                               uint32_t agent_id,
                               agent_social_context_t* out_context) {
    if (!ctx || !out_context) return false;

    int32_t slot = find_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    *out_context = ctx->agents[slot];
    return true;
}

bool social_context_get_stats(social_context_t ctx,
                               social_context_stats_t* out_stats) {
    if (!ctx || !out_stats) return false;

    /* Update computed stats */
    if (ctx->agent_count > 0) {
        float sum_gain = 0.0f;
        float sum_affinity = 0.0f;
        float sum_hierarchy = 0.0f;
        uint32_t count = 0;

        for (uint32_t i = 0; i < NIMCP_SOCIAL_MAX_AGENTS; i++) {
            if (!ctx->agent_slots_used[i]) continue;

            sum_gain += social_context_compute_gain(ctx, ctx->agents[i].agent_id);
            sum_affinity += ctx->agents[i].modulation.in_group_affinity;
            sum_hierarchy += ctx->agents[i].modulation.social_hierarchy;
            count++;
        }

        if (count > 0) {
            ctx->stats.avg_gain = sum_gain / (float)count;
            ctx->stats.avg_affinity = sum_affinity / (float)count;
            ctx->stats.avg_hierarchy = sum_hierarchy / (float)count;
        }
    }

    *out_stats = ctx->stats;
    return true;
}

bool social_context_has_agent(social_context_t ctx, uint32_t agent_id) {
    if (!ctx) return false;
    return find_agent_slot(ctx, agent_id) >= 0;
}

/* ============================================================================
 * Integration API
 * ============================================================================ */

bool social_context_connect_mirror(social_context_t ctx,
                                    mirror_neurons_t mirror) {
    if (!ctx) return false;

    ctx->connected_mirror = mirror;
    nimcp_log(LOG_LEVEL_INFO, "Social context connected to mirror neuron system");
    return true;
}

bool social_context_observe_agent(social_context_t ctx,
                                   uint32_t agent_id,
                                   uint64_t timestamp) {
    if (!ctx) return false;

    int32_t slot = find_or_create_agent_slot(ctx, agent_id);
    if (slot < 0) return false;

    agent_social_context_t* agent = &ctx->agents[slot];

    if (agent->first_seen == 0) {
        agent->first_seen = timestamp;
    }
    agent->last_seen = timestamp;

    return true;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

const char* social_group_name(social_group_t type) {
    switch (type) {
        case SOCIAL_GROUP_UNKNOWN:  return "unknown";
        case SOCIAL_GROUP_INGROUP:  return "in-group";
        case SOCIAL_GROUP_OUTGROUP: return "out-group";
        case SOCIAL_GROUP_ALLY:     return "ally";
        case SOCIAL_GROUP_RIVAL:    return "rival";
        case SOCIAL_GROUP_NEUTRAL:  return "neutral";
        default:                    return "invalid";
    }
}

const char* social_rank_name(social_rank_t type) {
    switch (type) {
        case SOCIAL_RANK_UNKNOWN:     return "unknown";
        case SOCIAL_RANK_DOMINANT:    return "dominant";
        case SOCIAL_RANK_PEER:        return "peer";
        case SOCIAL_RANK_SUBORDINATE: return "subordinate";
        case SOCIAL_RANK_EXPERT:      return "expert";
        case SOCIAL_RANK_NOVICE:      return "novice";
        default:                      return "invalid";
    }
}

const char* cultural_context_name(cultural_context_t type) {
    switch (type) {
        case CULTURAL_UNKNOWN:   return "unknown";
        case CULTURAL_NATIVE:    return "native";
        case CULTURAL_FAMILIAR:  return "familiar";
        case CULTURAL_FOREIGN:   return "foreign";
        case CULTURAL_UNIVERSAL: return "universal";
        default:                 return "invalid";
    }
}

const char* emotional_context_name(emotional_context_t type) {
    switch (type) {
        case EMOTIONAL_NEUTRAL:  return "neutral";
        case EMOTIONAL_POSITIVE: return "positive";
        case EMOTIONAL_NEGATIVE: return "negative";
        case EMOTIONAL_FEARFUL:  return "fearful";
        case EMOTIONAL_FRIENDLY: return "friendly";
        case EMOTIONAL_HOSTILE:  return "hostile";
        default:                 return "invalid";
    }
}

social_modulation_t social_modulation_init(void) {
    social_modulation_t mod;
    memset(&mod, 0, sizeof(mod));

    mod.in_group_affinity = NIMCP_SOCIAL_DEFAULT_AFFINITY;
    mod.social_hierarchy = NIMCP_SOCIAL_DEFAULT_HIERARCHY;
    mod.cultural_familiarity = NIMCP_SOCIAL_DEFAULT_CULTURAL;
    mod.emotional_valence = NIMCP_SOCIAL_DEFAULT_EMOTIONAL;

    mod.group_type = SOCIAL_GROUP_UNKNOWN;
    mod.rank_type = SOCIAL_RANK_UNKNOWN;
    mod.cultural_type = CULTURAL_UNKNOWN;
    mod.emotional_type = EMOTIONAL_NEUTRAL;

    mod.affinity_confidence = 0.0f;
    mod.hierarchy_confidence = 0.0f;
    mod.cultural_confidence = 0.0f;
    mod.emotional_confidence = 0.0f;

    mod.computed_gain = 1.0f;
    mod.gain_valid = false;

    return mod;
}

void social_modulation_print(const social_modulation_t* modulation,
                             const char* prefix) {
    if (!modulation) return;
    const char* pfx = prefix ? prefix : "";

    nimcp_log(LOG_LEVEL_DEBUG, "%sSocial Modulation:", pfx);
    nimcp_log(LOG_LEVEL_DEBUG, "%s  Affinity: %.2f (conf=%.2f) [%s]",
              pfx, modulation->in_group_affinity, modulation->affinity_confidence,
              social_group_name(modulation->group_type));
    nimcp_log(LOG_LEVEL_DEBUG, "%s  Hierarchy: %.2f (conf=%.2f) [%s]",
              pfx, modulation->social_hierarchy, modulation->hierarchy_confidence,
              social_rank_name(modulation->rank_type));
    nimcp_log(LOG_LEVEL_DEBUG, "%s  Cultural: %.2f (conf=%.2f) [%s]",
              pfx, modulation->cultural_familiarity, modulation->cultural_confidence,
              cultural_context_name(modulation->cultural_type));
    nimcp_log(LOG_LEVEL_DEBUG, "%s  Emotional: %.2f (conf=%.2f) [%s]",
              pfx, modulation->emotional_valence, modulation->emotional_confidence,
              emotional_context_name(modulation->emotional_type));
    nimcp_log(LOG_LEVEL_DEBUG, "%s  Gain: %.2f (valid=%s)",
              pfx, modulation->computed_gain, modulation->gain_valid ? "yes" : "no");
}
