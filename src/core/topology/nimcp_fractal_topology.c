//=============================================================================
// nimcp_fractal_topology.c - Fractal Network Topology Implementation
//=============================================================================

#include "core/topology/nimcp_fractal_topology.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include <math.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

// Access to network internal structure for getting neuron count
// Note: Minimal struct definition to access num_neurons field
struct neural_network_struct {
    void* neurons;
    uint32_t num_neurons;
    uint32_t capacity;
    // Other fields not accessed from this module
};

//=============================================================================
// Error Handling - Thread-local storage
//=============================================================================

static _Thread_local char last_error[256] = {0};

/**
 * WHY: Thread-safe error reporting without global state
 * HOW: Thread-local storage ensures each thread has independent error string
 */
static void set_error(const char* message) {
    if (!message) return;
    strncpy(last_error, message, sizeof(last_error) - 1);
    last_error[sizeof(last_error) - 1] = '\0';
}

static void clear_error(void) {
    last_error[0] = '\0';
}

const char* topology_get_last_error(void) {
    return last_error[0] ? last_error : NULL;
}

//=============================================================================
// Default Configurations
//=============================================================================

/**
 * WHY: Provide sensible defaults based on empirical neuroscience data
 * HOW: Values from Sporns et al. (2004) and Van den Heuvel & Sporns (2011)
 */
scale_free_config_t topology_default_scale_free_config(void) {
    return (scale_free_config_t){
        .power_law_gamma = -2.1f,      // Typical cortical value
        .hub_ratio = 0.15f,            // 15% hubs matches cortical data
        .min_degree = 3,               // Minimum for connectivity
        .max_degree = 0,               // 0 = auto-compute as N/10
        .spatial_constraint = 0.5f,    // Moderate distance dependence
        .bidirectional = false         // Directed by default
    };
}

fractal_config_t topology_default_fractal_config(void) {
    return (fractal_config_t){
        .fractal_dimension = 2.5f,     // Cortical measured value
        .hierarchy_levels = 4,         // Matches cortical hierarchy
        .branching_factor = 3.0f,      // Average 3 branches per level
        .scale_factor = 0.7f,          // 70% size reduction per level
        .clustering_coeff = 0.4f       // Target local clustering
    };
}

//=============================================================================
// Configuration Validation
//=============================================================================

/**
 * WHY: Prevent invalid parameters that would produce broken networks
 * HOW: Guard clauses check each parameter against valid ranges
 */
bool topology_validate_config(const topology_config_t* config) {
    // Guard: NULL input
    if (!config) {
        set_error("Configuration is NULL");
        return false;
    }

    // Clear any previous errors on successful start
    clear_error();

    // Guard: Invalid topology type
    if (config->type < TOPOLOGY_RANDOM || config->type > TOPOLOGY_SPATIAL) {
        set_error("Invalid topology type");
        return false;
    }

    // Type-specific validation
    if (config->type == TOPOLOGY_SCALE_FREE) {
        const scale_free_config_t* sf = &config->params.scale_free;

        // Guard: Invalid power law exponent
        if (sf->power_law_gamma >= 0.0f || sf->power_law_gamma < -5.0f) {
            set_error("Power law gamma must be between -5.0 and 0.0");
            return false;
        }

        // Guard: Invalid hub ratio
        if (sf->hub_ratio < 0.0f || sf->hub_ratio > 0.5f) {
            set_error("Hub ratio must be between 0.0 and 0.5");
            return false;
        }

        // Guard: Invalid minimum degree
        if (sf->min_degree < 1) {
            set_error("Minimum degree must be at least 1");
            return false;
        }

        // Guard: Invalid spatial constraint
        if (sf->spatial_constraint < 0.0f || sf->spatial_constraint > 1.0f) {
            set_error("Spatial constraint must be between 0.0 and 1.0");
            return false;
        }
    } else if (config->type == TOPOLOGY_FRACTAL) {
        const fractal_config_t* frac = &config->params.fractal;

        // Guard: Invalid fractal dimension
        if (frac->fractal_dimension < 1.0f || frac->fractal_dimension > 3.0f) {
            set_error("Fractal dimension must be between 1.0 and 3.0");
            return false;
        }

        // Guard: Invalid hierarchy levels
        if (frac->hierarchy_levels < 2 || frac->hierarchy_levels > 10) {
            set_error("Hierarchy levels must be between 2 and 10");
            return false;
        }

        // Guard: Invalid branching factor
        if (frac->branching_factor < 1.5f || frac->branching_factor > 10.0f) {
            set_error("Branching factor must be between 1.5 and 10.0");
            return false;
        }

        // Guard: Invalid scale factor
        if (frac->scale_factor <= 0.0f || frac->scale_factor >= 1.0f) {
            set_error("Scale factor must be between 0.0 and 1.0");
            return false;
        }
    }

    return true;
}

//=============================================================================
// Power Law Distribution Sampling
//=============================================================================

/**
 * WHY: Generate degree values following power-law P(k) ~ k^γ
 * HOW: Inverse transform sampling from power-law CDF
 *
 * ALGORITHM:
 *   1. Generate uniform random u ∈ [0,1]
 *   2. Compute k = k_min * (1-u)^(-1/(γ+1))
 *   3. This gives k ~ k^γ for k ≥ k_min
 */
static uint32_t sample_power_law(float gamma, uint32_t k_min, uint32_t k_max) {
    // Guard: Invalid parameters
    if (gamma >= 0.0f || k_min < 1 || k_max < k_min) {
        return k_min;
    }

    float u = (float)rand() / (float)RAND_MAX;
    float exponent = -1.0f / (gamma + 1.0f);
    float k_float = (float)k_min * powf(1.0f - u, exponent);

    uint32_t k = (uint32_t)k_float;

    // Clamp to valid range
    if (k < k_min) k = k_min;
    if (k > k_max) k = k_max;

    return k;
}

//=============================================================================
// Degree Distribution Computation
//=============================================================================

/**
 * WHY: Compute degree for each neuron (needed for preferential attachment)
 * HOW: Count incoming + outgoing synapses per neuron
 */
static uint32_t* compute_degree_distribution(neural_network_t network, uint32_t num_neurons) {
    // Guard: NULL network
    if (!network) return NULL;

    // Guard: Zero neurons
    if (num_neurons == 0) return NULL;

    uint32_t* degrees = (uint32_t*)calloc(num_neurons, sizeof(uint32_t));

    // Guard: Allocation failure
    if (!degrees) {
        set_error("Failed to allocate degree array");
        return NULL;
    }

    // Count degrees for each neuron
    // NOTE: This requires access to network internals
    // For now, we'll use a placeholder - needs integration with neural_network API

    return degrees;
}

//=============================================================================
// Preferential Attachment Selection
//=============================================================================

/**
 * WHY: Select target neuron with probability proportional to degree
 * HOW: Weighted random sampling using cumulative degree distribution
 *
 * ALGORITHM:
 *   1. Build cumulative sum: cumsum[i] = Σ(degree[0..i])
 *   2. Generate random r ∈ [0, total_degree]
 *   3. Binary search to find i where cumsum[i-1] < r ≤ cumsum[i]
 *   4. Return neuron i
 */
static uint32_t select_preferential(const uint32_t* degrees, uint32_t num_neurons, uint32_t exclude_idx) {
    // Guard: NULL degrees
    if (!degrees) return 0;

    // Guard: No neurons
    if (num_neurons == 0) return 0;

    // Build cumulative distribution
    uint32_t total_degree = 0;
    for (uint32_t i = 0; i < num_neurons; i++) {
        if (i != exclude_idx) {
            total_degree += degrees[i] + 1;  // +1 to ensure non-zero probability
        }
    }

    // Guard: Zero total degree
    if (total_degree == 0) {
        // Fallback to uniform random
        uint32_t idx = rand() % num_neurons;
        return (idx == exclude_idx) ? (idx + 1) % num_neurons : idx;
    }

    // Sample from distribution
    uint32_t r = rand() % total_degree;
    uint32_t cumsum = 0;

    for (uint32_t i = 0; i < num_neurons; i++) {
        if (i == exclude_idx) continue;

        cumsum += degrees[i] + 1;
        if (r < cumsum) {
            return i;
        }
    }

    // Fallback (should never reach here)
    return num_neurons - 1;
}

//=============================================================================
// Scale-Free Topology Generation
//=============================================================================

/**
 * WHY: Generate biologically realistic scale-free connectivity
 * HOW: Barabási-Albert preferential attachment with spatial constraints
 *
 * ALGORITHM:
 *   1. Initialize small fully-connected seed network (3-5 neurons)
 *   2. For each remaining neuron:
 *      a. Determine degree from power-law distribution
 *      b. Select targets via preferential attachment (P ∝ degree)
 *      c. Apply spatial constraint if enabled
 *      d. Create synapses
 *   3. Update degree distribution after each neuron
 *
 * BIOLOGICAL JUSTIFICATION:
 * - Preferential attachment models "rich get richer" in brain development
 * - Spatial constraints model axonal growth cone guidance
 * - Results in hub neurons (high-degree) and sparse connections (efficiency)
 */
bool topology_generate_scale_free(
    neural_network_t network,
    const scale_free_config_t* config,
    topology_stats_t* stats
) {
    // Guard: NULL network
    if (!network) {
        set_error("Network is NULL");
        return false;
    }

    // Guard: NULL config
    if (!config) {
        set_error("Configuration is NULL");
        return false;
    }

    // Clear any previous errors
    clear_error();

    // Validate configuration
    topology_config_t temp_config = {
        .type = TOPOLOGY_SCALE_FREE,
        .params.scale_free = *config
    };

    // Guard: Invalid config
    if (!topology_validate_config(&temp_config)) {
        return false;  // Error message already set by validate
    }

    // Get network size
    uint32_t num_neurons = network->num_neurons;

    // Guard: Too few neurons
    if (num_neurons < 3) {
        set_error("Network must have at least 3 neurons");
        return false;
    }

    // Compute max degree if not specified
    uint32_t max_degree = config->max_degree;
    if (max_degree == 0) {
        max_degree = num_neurons / 10;
        if (max_degree < config->min_degree) {
            max_degree = config->min_degree;
        }
    }

    // Initialize degree tracking
    uint32_t* degrees = compute_degree_distribution(network, num_neurons);

    // Guard: Degree allocation failed
    if (!degrees) {
        return false;  // Error message already set
    }

    // Seed random number generator
    srand((unsigned int)time(NULL));

    // Create initial seed network (fully connected)
    // WHAT: Fully connect first 3 neurons to seed scale-free growth
    // WHY: Barabási-Albert model requires initial connected network
    // HOW: Create bidirectional connections between all pairs in seed
    uint32_t seed_size = 3;
    for (uint32_t i = 0; i < seed_size; i++) {
        for (uint32_t j = i + 1; j < seed_size; j++) {
            // WHAT: Create bidirectional synapse connection
            // WHY: Scale-free networks are typically undirected
            // HOW: Call neural_network_add_connection for both directions
            // FIX (Phase 5): Was TODO, now implemented
            float initial_weight = 0.5f;  // Start with moderate weight
            neural_network_add_connection(network, i, j, initial_weight);
            neural_network_add_connection(network, j, i, initial_weight);

            degrees[i]++;
            degrees[j]++;
        }
    }

    // Add remaining neurons with preferential attachment
    uint32_t total_synapses = seed_size * (seed_size - 1) / 2;

    for (uint32_t n = seed_size; n < num_neurons; n++) {
        // Sample degree from power-law
        uint32_t degree = sample_power_law(config->power_law_gamma,
                                           config->min_degree,
                                           max_degree);

        // Create connections via preferential attachment
        // WHAT: Connect new neuron to existing neurons based on their degree
        // WHY: Preferential attachment creates scale-free topology
        // HOW: Higher-degree neurons more likely to receive new connections
        for (uint32_t m = 0; m < degree; m++) {
            uint32_t target = select_preferential(degrees, n, n);  // Select from existing neurons

            // WHAT: Create bidirectional synapse between new neuron and target
            // WHY: Scale-free networks typically have undirected edges
            // HOW: Call neural_network_add_connection for both directions
            // FIX (Phase 5): Was TODO, now implemented
            float weight = 0.5f;  // Moderate initial weight
            neural_network_add_connection(network, n, target, weight);
            neural_network_add_connection(network, target, n, weight);

            degrees[n]++;
            degrees[target]++;
            total_synapses++;
        }
    }

    // Compute statistics if requested
    if (stats) {
        stats->num_neurons = num_neurons;
        stats->num_synapses = total_synapses;
        stats->avg_degree = (float)(2 * total_synapses) / (float)num_neurons;

        // Compute degree std dev
        float mean_degree = stats->avg_degree;
        float variance = 0.0f;
        for (uint32_t i = 0; i < num_neurons; i++) {
            float diff = (float)degrees[i] - mean_degree;
            variance += diff * diff;
        }
        stats->degree_std = sqrtf(variance / (float)num_neurons);

        // Count hubs (top hub_ratio by degree)
        uint32_t hub_threshold_degree = (uint32_t)(mean_degree + 2.0f * stats->degree_std);
        stats->num_hubs = 0;
        for (uint32_t i = 0; i < num_neurons; i++) {
            if (degrees[i] >= hub_threshold_degree) {
                stats->num_hubs++;
            }
        }

        // Other stats require full graph analysis
        stats->clustering_coefficient = 0.0f;  // TODO: Implement
        stats->characteristic_path = 0.0f;     // TODO: Implement
        stats->power_law_fit = 0.0f;           // TODO: Implement
        stats->hub_connectivity = 0.0f;        // TODO: Implement
        stats->small_world_sigma = 0.0f;       // TODO: Implement
    }

    free(degrees);
    return true;
}

//=============================================================================
// Fractal Topology Generation
//=============================================================================

/**
 * WHY: Generate hierarchical self-similar network structure
 * HOW: Recursive subdivision with scale-invariant patterns
 *
 * ALGORITHM:
 *   1. Partition neurons into hierarchy_levels groups
 *   2. At level 0 (bottom), create dense local clusters
 *   3. At each higher level:
 *      a. Create sparse inter-cluster connections
 *      b. Connection probability ~ distance^(-fractal_dimension)
 *   4. Ensure connectivity between all levels
 */
bool topology_generate_fractal(
    neural_network_t network,
    const fractal_config_t* config,
    topology_stats_t* stats
) {
    // Guard: NULL network
    if (!network) {
        set_error("Network is NULL");
        return false;
    }

    // Guard: NULL config
    if (!config) {
        set_error("Configuration is NULL");
        return false;
    }

    // Validate configuration
    topology_config_t temp_config = {
        .type = TOPOLOGY_FRACTAL,
        .params.fractal = *config
    };

    // Guard: Invalid config
    if (!topology_validate_config(&temp_config)) {
        return false;
    }

    // TODO: Implement fractal generation
    set_error("Fractal topology generation not yet implemented");
    return false;
}

//=============================================================================
// Unified Topology Generation
//=============================================================================

/**
 * WHY: Single entry point for all topology types
 * HOW: Strategy pattern - dispatch based on config.type
 */
bool topology_generate(
    neural_network_t network,
    const topology_config_t* config,
    topology_stats_t* stats
) {
    // Guard: NULL network
    if (!network) {
        set_error("Network is NULL");
        return false;
    }

    // Guard: NULL config
    if (!config) {
        set_error("Configuration is NULL");
        return false;
    }

    // Validate configuration first
    if (!topology_validate_config(config)) {
        return false;
    }

    // Strategy pattern: Dispatch to appropriate generator
    switch (config->type) {
        case TOPOLOGY_SCALE_FREE:
            return topology_generate_scale_free(network, &config->params.scale_free, stats);

        case TOPOLOGY_FRACTAL:
            return topology_generate_fractal(network, &config->params.fractal, stats);

        case TOPOLOGY_RANDOM:
        case TOPOLOGY_SMALL_WORLD:
        case TOPOLOGY_SPATIAL:
            set_error("Topology type not yet implemented");
            return false;

        default:
            set_error("Unknown topology type");
            return false;
    }
}

//=============================================================================
// Topology Analysis Functions
//=============================================================================

bool topology_compute_stats(
    neural_network_t network,
    topology_stats_t* stats
) {
    // Guard: NULL inputs
    if (!network || !stats) {
        set_error("NULL network or stats pointer");
        return false;
    }

    // TODO: Implement full graph analysis
    set_error("Topology statistics computation not yet implemented");
    return false;
}

bool topology_is_small_world(
    neural_network_t network,
    float* sigma
) {
    // Guard: NULL network
    if (!network) {
        set_error("Network is NULL");
        return false;
    }

    // TODO: Implement small-world test
    set_error("Small-world test not yet implemented");
    return false;
}

bool topology_fit_power_law(
    neural_network_t network,
    float* gamma,
    float* r_squared
) {
    // Guard: NULL network
    if (!network) {
        set_error("Network is NULL");
        return false;
    }

    // TODO: Implement power-law fitting
    set_error("Power-law fitting not yet implemented");
    return false;
}

//=============================================================================
// Hub Neuron Functions
//=============================================================================

bool topology_identify_hubs(
    neural_network_t network,
    float percentile,
    uint32_t** hub_indices,
    uint32_t* num_hubs
) {
    // Guard: NULL inputs
    if (!network || !hub_indices || !num_hubs) {
        set_error("NULL pointer in hub identification");
        return false;
    }

    // Guard: Invalid percentile
    if (percentile < 0.0f || percentile > 1.0f) {
        set_error("Percentile must be between 0.0 and 1.0");
        return false;
    }

    // TODO: Implement hub identification
    set_error("Hub identification not yet implemented");
    return false;
}

bool topology_compute_betweenness(
    neural_network_t network,
    float* centrality
) {
    // Guard: NULL inputs
    if (!network || !centrality) {
        set_error("NULL network or centrality pointer");
        return false;
    }

    // TODO: Implement Brandes' algorithm
    set_error("Betweenness centrality not yet implemented");
    return false;
}
