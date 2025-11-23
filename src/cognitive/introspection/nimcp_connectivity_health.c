/**
 * @file nimcp_connectivity_health.c
 * @brief Brain Connectivity Health Assessment Implementation - Phase 1.5.4
 *
 * WHAT: Implementation of introspection-based connectivity health monitoring
 * WHY:  Enable self-awareness of brain's organizational quality
 * HOW:  Combines community detection, hub analysis, topology metrics, Shannon flow
 *
 * PHASE: 1.5.4 - Introspection + Community Detection Health Monitoring
 *
 * @author NIMCP Development Team
 * @date 2025-11-23
 */

#include "cognitive/introspection/nimcp_connectivity_health.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain_regions/nimcp_brain_regions.h"
#include "core/topology/nimcp_community_detection.h"
#include "utils/algorithms/nimcp_graph_metrics.h"
#include "utils/algorithms/nimcp_centrality.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "utils/logging/nimcp_logging.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define LOG2_EPSILON 1e-10f
#define HEALTH_SCORE_EPSILON 0.001f

/* Map conceptual region names to brain_region_type_t values */
#define BRAIN_REGION_EXECUTIVE  REGION_PREFRONTAL    /* PFC is executive control */
#define BRAIN_REGION_WORKSPACE  REGION_PARIETAL      /* Parietal for working memory */
#define BRAIN_REGION_SALIENCE   REGION_TEMPORAL      /* Temporal for salience/memory */

//=============================================================================
// Internal Declarations
//=============================================================================

/* Forward declare introspection context structure to access brain field */
struct introspection_context_struct {
    brain_t brain;
    /* ... other fields not needed here ... */
};

/* Helper to get brain from introspection context */
static inline brain_t introspection_get_brain(introspection_context_t ctx) {
    if (!ctx) return NULL;
    return ((struct introspection_context_struct*)ctx)->brain;
}

/* Use NIMCP time utilities */
#define nimcp_get_time_ms() nimcp_time_monotonic_ms()

//=============================================================================
// Stub/Fallback Functions (Until full brain API integration)
//=============================================================================

/**
 * @brief Get neuron's brain region (fallback: compute from neuron ID)
 *
 * WHAT: Maps neuron ID to brain region type
 * WHY:  Required for hub regional distribution analysis
 * NOTE: Uses heuristic based on neuron ID ranges until full API available
 */
static inline int brain_get_neuron_region(brain_t brain, uint32_t neuron_id) {
    if (!brain) return 0;
    uint32_t neuron_count = brain_get_neuron_count(brain);
    if (neuron_count == 0) return 0;

    /* Heuristic: divide neurons into regions by ID ranges */
    float region_fraction = (float)neuron_id / (float)neuron_count;
    if (region_fraction < 0.15f) return REGION_VISUAL_V1;
    if (region_fraction < 0.30f) return REGION_AUDITORY_A1;
    if (region_fraction < 0.45f) return REGION_MOTOR_M1;
    if (region_fraction < 0.60f) return REGION_PREFRONTAL;
    if (region_fraction < 0.75f) return REGION_PARIETAL;
    if (region_fraction < 0.90f) return REGION_TEMPORAL;
    return REGION_HIPPOCAMPUS;
}

/**
 * @brief Get modularity Q from community detection (fallback)
 */
static inline float brain_get_modularity(brain_t brain) {
    if (!brain) return 0.0f;
    /* TODO: Get actual modularity from community detection subsystem */
    return 0.35f;  /* Default healthy modularity */
}

/**
 * @brief Get number of detected communities (fallback)
 */
static inline uint32_t brain_get_num_communities(brain_t brain) {
    if (!brain) return 0;
    uint32_t neuron_count = brain_get_neuron_count(brain);
    /* Estimate: roughly sqrt(N/10) communities for healthy networks */
    return (uint32_t)sqrtf((float)neuron_count / 10.0f) + 1;
}

/**
 * @brief Get community sizes array (fallback: returns NULL)
 */
static inline uint32_t* brain_get_community_sizes(brain_t brain) {
    (void)brain;
    return NULL;  /* Full implementation requires community detection results */
}

/**
 * @brief Get hub neuron IDs (fallback: returns top-degree neurons)
 */
static inline uint32_t* brain_get_hub_ids(brain_t brain, uint32_t* num_hubs) {
    (void)brain;
    if (num_hubs) *num_hubs = 0;
    return NULL;  /* Full implementation requires centrality computation */
}

/**
 * @brief Get hub centrality scores (fallback)
 */
static inline float* brain_get_hub_centrality(brain_t brain, uint32_t* num_hubs) {
    (void)brain;
    if (num_hubs) *num_hubs = 0;
    return NULL;  /* Full implementation requires centrality computation */
}

/**
 * @brief Get total synapse count (fallback: estimate from neurons)
 */
static inline uint32_t brain_get_synapse_count(brain_t brain) {
    if (!brain) return 0;
    uint32_t neurons = brain_get_neuron_count(brain);
    /* Estimate: average 100 synapses per neuron (biological range: 1000-10000) */
    return neurons * 100;
}

//=============================================================================
// Configuration Functions
//=============================================================================

/**
 * @brief Get default connectivity health configuration
 *
 * WHAT: Returns standard configuration with biological defaults
 * WHY:  Provide sensible starting point based on neuroscience literature
 */
connectivity_health_config_t connectivity_health_default_config(void)
{
    connectivity_health_config_t config = {
        /* Community structure thresholds */
        .min_modularity = CONNECTIVITY_MIN_MODULARITY,
        .max_community_imbalance = 10.0f,

        /* Hub detection parameters */
        .hub_threshold_stddev = CONNECTIVITY_HUB_THRESHOLD,
        .require_executive_hubs = true,
        .require_workspace_hubs = true,

        /* Topology thresholds */
        .min_clustering_coefficient = CONNECTIVITY_MIN_CLUSTERING,
        .max_path_length = CONNECTIVITY_MAX_PATH_LENGTH,
        .small_world_threshold = CONNECTIVITY_SMALL_WORLD_THRESHOLD,

        /* Information flow thresholds */
        .min_flow_efficiency = CONNECTIVITY_MIN_FLOW_EFFICIENCY,
        .min_layer_connectivity = 0.5f,

        /* Weight factors (sum to 1.0) */
        .weight_modularity = 0.25f,
        .weight_hubs = 0.20f,
        .weight_topology = 0.25f,
        .weight_flow = 0.30f,

        /* Assessment control */
        .assessment_interval_ms = CONNECTIVITY_DEFAULT_ASSESSMENT_INTERVAL_MS,
        .enable_detailed_hub_analysis = true,
        .enable_community_balance = true
    };

    return config;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Safe log2 calculation avoiding log(0)
 */
static float safe_log2f(float x)
{
    if (x < LOG2_EPSILON) {
        return 0.0f;
    }
    return log2f(x);
}

/**
 * @brief Calculate community size balance entropy
 *
 * WHAT: Measure evenness of community size distribution
 * WHY:  Extremely imbalanced communities indicate problems
 * HOW:  Shannon entropy of normalized community sizes
 */
float calculate_community_balance(
    const uint32_t* community_sizes,
    uint32_t num_communities)
{
    if (!community_sizes || num_communities == 0) {
        return 0.0f;
    }

    if (num_communities == 1) {
        return 1.0f;  /* Single community is trivially balanced */
    }

    /* Calculate total neurons */
    uint32_t total = 0;
    for (uint32_t i = 0; i < num_communities; i++) {
        total += community_sizes[i];
    }

    if (total == 0) {
        return 0.0f;
    }

    /* Calculate entropy: H = -Σ p(i) log₂ p(i) */
    float entropy = 0.0f;
    float inv_total = 1.0f / (float)total;

    for (uint32_t i = 0; i < num_communities; i++) {
        float p = (float)community_sizes[i] * inv_total;
        if (p > LOG2_EPSILON) {
            entropy -= p * safe_log2f(p);
        }
    }

    /* Normalize by maximum entropy (uniform distribution) */
    float max_entropy = safe_log2f((float)num_communities);
    if (max_entropy < LOG2_EPSILON) {
        return 1.0f;
    }

    return entropy / max_entropy;
}

/**
 * @brief Check if neuron is in specified brain region
 */
bool is_neuron_in_region(
    brain_t brain,
    uint32_t neuron_id,
    int region)
{
    if (!brain) {
        return false;
    }

    /* Get neuron's region from brain topology (cast to int to avoid type issues) */
    int neuron_region = (int)brain_get_neuron_region(brain, neuron_id);
    return (neuron_region == region);
}

/**
 * @brief Count hubs in a specific region
 */
static uint32_t count_hubs_in_region(
    brain_t brain,
    const uint32_t* hub_ids,
    uint32_t num_hubs,
    int region)
{
    if (!brain || !hub_ids || num_hubs == 0) {
        return 0;
    }

    uint32_t count = 0;
    for (uint32_t i = 0; i < num_hubs; i++) {
        if (is_neuron_in_region(brain, hub_ids[i], region)) {
            count++;
        }
    }
    return count;
}

//=============================================================================
// Component Assessment Functions
//=============================================================================

/**
 * @brief Assess community structure health
 */
community_health_t introspection_assess_community_health(
    introspection_context_t introspection,
    const connectivity_health_config_t* config)
{
    community_health_t health = {0};

    if (!introspection) {
        return health;
    }

    /* Use default config if not provided */
    connectivity_health_config_t cfg = config ? *config : connectivity_health_default_config();

    /* Get brain from introspection context */
    brain_t brain = introspection_get_brain(introspection);
    if (!brain) {
        return health;
    }

    /* Run community detection if not already done */
    if (!brain_detect_communities(brain)) {
        return health;
    }

    /* Get modularity */
    health.modularity_q = brain_get_modularity(brain);
    health.num_communities = brain_get_num_communities(brain);

    /* Calculate community balance if enabled */
    if (cfg.enable_community_balance && health.num_communities > 0) {
        uint32_t* sizes = brain_get_community_sizes(brain);
        if (sizes) {
            health.community_balance = calculate_community_balance(
                sizes, health.num_communities);

            /* Find largest community ratio */
            uint32_t total = 0;
            uint32_t largest = 0;
            for (uint32_t i = 0; i < health.num_communities; i++) {
                total += sizes[i];
                if (sizes[i] > largest) {
                    largest = sizes[i];
                }
            }
            if (total > 0) {
                health.largest_community_ratio = (float)largest / (float)total;
            }
        }
    }

    /* Determine health status */
    health.is_healthy = (health.modularity_q >= cfg.min_modularity);

    return health;
}

/**
 * @brief Assess hub neuron health
 */
hub_health_t introspection_assess_hub_health(
    introspection_context_t introspection,
    const connectivity_health_config_t* config)
{
    hub_health_t health = {0};

    if (!introspection) {
        return health;
    }

    connectivity_health_config_t cfg = config ? *config : connectivity_health_default_config();
    brain_t brain = introspection_get_brain(introspection);
    if (!brain) {
        return health;
    }

    /* Detect hub neurons */
    if (!brain_detect_hubs(brain, cfg.hub_threshold_stddev)) {
        return health;
    }

    /* Get hub information */
    uint32_t num_hubs = 0;
    uint32_t* hub_ids = brain_get_hub_ids(brain, &num_hubs);
    float* centrality = brain_get_hub_centrality(brain, &num_hubs);

    if (num_hubs > 0 && hub_ids) {
        health.num_hubs = (num_hubs > CONNECTIVITY_MAX_HUBS) ?
                          CONNECTIVITY_MAX_HUBS : num_hubs;

        /* Copy hub data */
        for (uint32_t i = 0; i < health.num_hubs; i++) {
            health.hub_neuron_ids[i] = hub_ids[i];
            if (centrality) {
                health.hub_centrality[i] = centrality[i];
            }
        }

        /* Calculate average centrality */
        if (centrality) {
            float sum = 0.0f;
            for (uint32_t i = 0; i < health.num_hubs; i++) {
                sum += health.hub_centrality[i];
            }
            health.avg_hub_centrality = sum / (float)health.num_hubs;
        }

        /* Check regional hub distribution */
        health.executive_has_hubs =
            count_hubs_in_region(brain, hub_ids, num_hubs, BRAIN_REGION_EXECUTIVE) > 0;
        health.workspace_has_hubs =
            count_hubs_in_region(brain, hub_ids, num_hubs, BRAIN_REGION_WORKSPACE) > 0;
        health.salience_has_hubs =
            count_hubs_in_region(brain, hub_ids, num_hubs, BRAIN_REGION_SALIENCE) > 0;

        /* Count hubs per region */
        for (int r = 0; r < 16; r++) {
            health.hubs_per_region[r] =
                count_hubs_in_region(brain, hub_ids, num_hubs, r);
        }

        /* Calculate hub distribution entropy */
        uint32_t regions_with_hubs = 0;
        for (int r = 0; r < 16; r++) {
            if (health.hubs_per_region[r] > 0) {
                regions_with_hubs++;
            }
        }
        if (regions_with_hubs > 0) {
            health.hub_distribution_entropy =
                (float)regions_with_hubs / 16.0f;  /* Simple spread metric */
        }
    }

    /* Determine health status */
    bool required_hubs_present = true;
    if (cfg.require_executive_hubs && !health.executive_has_hubs) {
        required_hubs_present = false;
    }
    if (cfg.require_workspace_hubs && !health.workspace_has_hubs) {
        required_hubs_present = false;
    }
    health.is_healthy = (health.num_hubs > 0) && required_hubs_present;

    return health;
}

/**
 * @brief Assess graph topology health
 */
topology_health_t introspection_assess_topology_health(
    introspection_context_t introspection,
    const connectivity_health_config_t* config)
{
    topology_health_t health = {0};

    if (!introspection) {
        return health;
    }

    connectivity_health_config_t cfg = config ? *config : connectivity_health_default_config();
    brain_t brain = introspection_get_brain(introspection);
    if (!brain) {
        return health;
    }

    /* Compute topology metrics */
    if (!brain_compute_topology_metrics(brain)) {
        return health;
    }

    /* Get cached metrics */
    topology_validation_t* metrics = brain->topology_metrics;
    if (!metrics) {
        return health;
    }

    health.clustering_coefficient = metrics->clustering_coefficient;
    health.avg_path_length = metrics->characteristic_path;
    health.small_world_sigma = metrics->small_world_sigma;
    health.network_diameter = 0;  /* Would need separate computation */
    health.assortativity = 0.0f;  /* Would need separate computation */

    /* Calculate normalized scores */
    health.clustering_score = health.clustering_coefficient;  /* Already [0,1] */

    /* Path length score: inverse, shorter is better */
    if (health.avg_path_length > 0) {
        health.path_length_score = 1.0f / health.avg_path_length;
        if (health.path_length_score > 1.0f) {
            health.path_length_score = 1.0f;
        }
    }

    /* Combined topology score */
    health.topology_score = 0.4f * health.clustering_score +
                            0.3f * health.path_length_score +
                            0.3f * (health.small_world_sigma > 1.0f ? 1.0f : health.small_world_sigma);

    /* Check small-world property (sigma must be strictly > threshold) */
    health.is_small_world = (health.small_world_sigma > cfg.small_world_threshold);

    /* Determine overall topology health */
    health.is_healthy =
        (health.clustering_coefficient >= cfg.min_clustering_coefficient) &&
        (health.avg_path_length <= cfg.max_path_length) &&
        health.is_small_world;

    return health;
}

/**
 * @brief Assess information flow health
 */
information_flow_health_t introspection_assess_flow_health(
    introspection_context_t introspection,
    const connectivity_health_config_t* config)
{
    information_flow_health_t health = {0};

    if (!introspection) {
        return health;
    }

    connectivity_health_config_t cfg = config ? *config : connectivity_health_default_config();
    brain_t brain = introspection_get_brain(introspection);
    if (!brain) {
        return health;
    }

    /* Get Shannon metrics if available */
    if (brain->enable_shannon_monitoring) {
        shannon_network_metrics_t shannon = brain->last_shannon_metrics;

        health.transfer_efficiency = shannon.average_efficiency;
        health.bottleneck_score = shannon.bottleneck_score;
        health.num_bottlenecks = shannon.num_bottlenecks;
        health.total_capacity_bits_per_sec = shannon.total_capacity;
        health.actual_throughput_bits_per_sec = shannon.information_rate;

        if (shannon.total_capacity > 0) {
            health.capacity_utilization =
                shannon.information_rate / shannon.total_capacity;
        }
    }

    /* Measure layer connectivity (middleware to cognitive) */
    /* This would need specific implementation based on brain structure */
    health.layer_connectivity = 0.8f;  /* Placeholder - would compute from brain */

    /* Determine health status */
    health.is_healthy = (health.transfer_efficiency >= cfg.min_flow_efficiency) &&
                        (health.layer_connectivity >= cfg.min_layer_connectivity);

    return health;
}

//=============================================================================
// Main Assessment Function
//=============================================================================

/**
 * @brief Assess complete brain connectivity health
 */
brain_connectivity_health_t introspection_assess_connectivity_health(
    introspection_context_t introspection,
    const connectivity_health_config_t* config)
{
    brain_connectivity_health_t health = {0};
    uint64_t start_time = nimcp_get_time_ms();

    if (!introspection) {
        snprintf(health.primary_issue, sizeof(health.primary_issue),
                 "Invalid introspection context");
        health.num_critical = 1;
        return health;
    }

    connectivity_health_config_t cfg = config ? *config : connectivity_health_default_config();
    brain_t brain = introspection_get_brain(introspection);
    if (!brain) {
        snprintf(health.primary_issue, sizeof(health.primary_issue),
                 "No brain associated with introspection");
        health.num_critical = 1;
        return health;
    }

    /* Get network stats */
    health.total_neurons = brain_get_neuron_count(brain);
    health.total_synapses = brain_get_synapse_count(brain);

    /* Assess each component */
    health.community = introspection_assess_community_health(introspection, &cfg);
    health.hubs = introspection_assess_hub_health(introspection, &cfg);
    health.topology = introspection_assess_topology_health(introspection, &cfg);
    health.flow = introspection_assess_flow_health(introspection, &cfg);

    /* Calculate component scores for overall health */
    float modularity_score = health.community.modularity_q;
    if (modularity_score < 0) modularity_score = 0;
    if (modularity_score > 1) modularity_score = 1;

    float hub_score = health.hubs.is_healthy ? 1.0f : 0.5f;
    if (health.hubs.num_hubs == 0) hub_score = 0.0f;

    float topology_score = health.topology.topology_score;

    float flow_score = health.flow.transfer_efficiency;

    /* Weighted combination */
    health.overall_health =
        cfg.weight_modularity * modularity_score +
        cfg.weight_hubs * hub_score +
        cfg.weight_topology * topology_score +
        cfg.weight_flow * flow_score;

    /* Count issues */
    if (!health.community.is_healthy) {
        health.num_warnings++;
        if (health.community.modularity_q < 0.1f) {
            health.num_critical++;
        }
    }
    if (!health.hubs.is_healthy) {
        health.num_warnings++;
    }
    if (!health.topology.is_healthy) {
        health.num_warnings++;
        if (!health.topology.is_small_world) {
            health.num_critical++;
        }
    }
    if (!health.flow.is_healthy) {
        health.num_warnings++;
        if (health.flow.transfer_efficiency < 0.5f) {
            health.num_critical++;
        }
    }

    /* Determine primary issue */
    if (health.num_critical > 0 || health.num_warnings > 0) {
        if (!health.community.is_healthy) {
            snprintf(health.primary_issue, sizeof(health.primary_issue),
                     "Low modularity (Q=%.3f, need >%.3f)",
                     health.community.modularity_q, cfg.min_modularity);
        } else if (!health.topology.is_small_world) {
            snprintf(health.primary_issue, sizeof(health.primary_issue),
                     "Not small-world (sigma=%.3f, need >%.3f)",
                     health.topology.small_world_sigma, cfg.small_world_threshold);
        } else if (!health.hubs.is_healthy) {
            snprintf(health.primary_issue, sizeof(health.primary_issue),
                     "Missing hub neurons in key regions");
        } else if (!health.flow.is_healthy) {
            snprintf(health.primary_issue, sizeof(health.primary_issue),
                     "Low information flow efficiency (%.1f%%)",
                     health.flow.transfer_efficiency * 100.0f);
        }
    }

    /* Determine overall health status */
    health.is_healthy = health.community.is_healthy &&
                        health.hubs.is_healthy &&
                        health.topology.is_healthy &&
                        health.flow.is_healthy;

    /* Timing */
    health.assessment_timestamp_ms = start_time;
    health.assessment_duration_ms = (uint32_t)(nimcp_get_time_ms() - start_time);

    return health;
}

/**
 * @brief Quick connectivity health check
 */
float introspection_quick_connectivity_check(
    introspection_context_t introspection,
    bool* is_healthy)
{
    if (is_healthy) {
        *is_healthy = false;
    }

    if (!introspection) {
        return 0.0f;
    }

    brain_t brain = introspection_get_brain(introspection);
    if (!brain) {
        return 0.0f;
    }

    /* Use cached metrics if available */
    float score = 0.0f;
    uint32_t components = 0;

    /* Check cached modularity */
    if (brain->functional_modules) {
        float q = brain->functional_modules->modularity;
        score += (q > 0) ? q : 0.0f;
        components++;
    }

    /* Check cached topology */
    if (brain->topology_metrics) {
        if (brain->topology_metrics->is_valid) {
            score += 1.0f;
        }
        components++;
    }

    /* Check Shannon metrics */
    if (brain->enable_shannon_monitoring) {
        score += brain->last_shannon_metrics.average_efficiency;
        components++;
    }

    if (components > 0) {
        score /= (float)components;
    }

    if (is_healthy) {
        *is_healthy = (score >= 0.6f);
    }

    return score;
}

//=============================================================================
// Brain Integration Functions
//=============================================================================

/**
 * @brief Enable periodic connectivity health monitoring
 */
bool brain_enable_connectivity_monitoring(
    brain_t brain,
    const connectivity_health_config_t* config,
    void (*callback)(const brain_connectivity_health_t*, void*),
    void* callback_context)
{
    if (!brain) {
        return false;
    }

    /* Store configuration */
    connectivity_health_config_t cfg = config ? *config : connectivity_health_default_config();

    /* Enable monitoring flag in brain */
    brain->enable_connectivity_monitoring = true;
    brain->connectivity_health_callback = callback;
    brain->connectivity_health_callback_context = callback_context;

    /* Do initial assessment */
    if (brain->introspection) {
        brain->last_connectivity_health =
            introspection_assess_connectivity_health(brain->introspection, &cfg);
        brain->last_connectivity_assessment_time_ms = nimcp_get_time_ms();
    }

    return true;
}

/**
 * @brief Disable connectivity monitoring
 */
void brain_disable_connectivity_monitoring(brain_t brain)
{
    if (!brain) {
        return;
    }

    brain->enable_connectivity_monitoring = false;
    brain->connectivity_health_callback = NULL;
    brain->connectivity_health_callback_context = NULL;
}

/**
 * @brief Check if monitoring is enabled
 */
bool brain_is_connectivity_monitoring_enabled(brain_t brain)
{
    if (!brain) {
        return false;
    }
    return brain->enable_connectivity_monitoring;
}

/**
 * @brief Get last cached health assessment
 */
bool brain_get_connectivity_health(
    brain_t brain,
    brain_connectivity_health_t* health)
{
    if (!brain || !health) {
        return false;
    }

    if (brain->last_connectivity_assessment_time_ms == 0) {
        return false;  /* Never assessed */
    }

    *health = brain->last_connectivity_health;
    return true;
}

/**
 * @brief Force immediate assessment
 */
brain_connectivity_health_t brain_assess_connectivity_now(brain_t brain)
{
    brain_connectivity_health_t health = {0};

    if (!brain) {
        snprintf(health.primary_issue, sizeof(health.primary_issue),
                 "Invalid brain instance");
        return health;
    }

    if (!brain->introspection) {
        snprintf(health.primary_issue, sizeof(health.primary_issue),
                 "Introspection not initialized");
        return health;
    }

    health = introspection_assess_connectivity_health(brain->introspection, NULL);

    /* Cache result */
    brain->last_connectivity_health = health;
    brain->last_connectivity_assessment_time_ms = nimcp_get_time_ms();

    /* Invoke callback if registered */
    if (brain->connectivity_health_callback) {
        brain->connectivity_health_callback(
            &health,
            brain->connectivity_health_callback_context
        );
    }

    return health;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Get human-readable health status string
 */
const char* connectivity_health_to_string(
    const brain_connectivity_health_t* health,
    char* buffer,
    size_t buffer_size)
{
    if (!buffer || buffer_size == 0) {
        return NULL;  /* Invalid buffer - return NULL for safety */
    }

    if (!health) {
        snprintf(buffer, buffer_size, "Health assessment: N/A (null)");
        return buffer;
    }

    snprintf(buffer, buffer_size,
             "Connectivity Health: %.1f%% (%s)\n"
             "  Communities: %u (Q=%.3f, %s)\n"
             "  Hub Neurons: %u (%s)\n"
             "  Topology: C=%.3f, L=%.1f, sigma=%.2f (%s)\n"
             "  Info Flow: %.1f%% efficiency (%s)\n"
             "  Issues: %u warnings, %u critical\n"
             "%s%s",
             health->overall_health * 100.0f,
             health->is_healthy ? "HEALTHY" : "UNHEALTHY",
             health->community.num_communities,
             health->community.modularity_q,
             health->community.is_healthy ? "ok" : "ISSUE",
             health->hubs.num_hubs,
             health->hubs.is_healthy ? "ok" : "ISSUE",
             health->topology.clustering_coefficient,
             health->topology.avg_path_length,
             health->topology.small_world_sigma,
             health->topology.is_healthy ? "ok" : "ISSUE",
             health->flow.transfer_efficiency * 100.0f,
             health->flow.is_healthy ? "ok" : "ISSUE",
             health->num_warnings,
             health->num_critical,
             health->primary_issue[0] ? "  Primary Issue: " : "",
             health->primary_issue);

    return buffer;
}

/**
 * @brief Free dynamically allocated health data
 */
void connectivity_health_free(brain_connectivity_health_t* health)
{
    /* Currently no dynamic allocation in health struct */
    /* This function exists for future extensibility */
    (void)health;
}
