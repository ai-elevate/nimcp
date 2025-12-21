//=============================================================================
// nimcp_pink_noise_spatial.c - Spatially Correlated Pink Noise
//=============================================================================

#include "plasticity/noise/nimcp_pink_noise_spatial.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <math.h>

//=============================================================================
// Internal: Distance and Correlation
//=============================================================================

static float compute_distance(
    float x1, float y1, float z1,
    float x2, float y2, float z2
) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return sqrtf(dx * dx + dy * dy + dz * dz);
}

static float compute_correlation(
    float distance,
    const pink_spatial_config_t* config
) {
    float lambda = config->length_constant;
    float corr;

    switch (config->decay_type) {
        case PINK_SPATIAL_DECAY_EXPONENTIAL:
            corr = expf(-distance / lambda);
            break;
        case PINK_SPATIAL_DECAY_GAUSSIAN:
            corr = expf(-distance * distance / (2.0f * lambda * lambda));
            break;
        case PINK_SPATIAL_DECAY_POWER_LAW:
            corr = 1.0f / powf(1.0f + distance / lambda, config->power_law_exponent);
            break;
        default:
            corr = 1.0f;
    }

    // Apply floor
    if (corr < config->min_correlation) {
        corr = config->min_correlation;
    }

    return corr;
}

static bool cholesky_decomposition(
    const float* matrix,
    float* lower,
    uint32_t n
) {
    memset(lower, 0, n * n * sizeof(float));

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j <= i; j++) {
            float sum = 0.0f;

            if (j == i) {
                for (uint32_t k = 0; k < j; k++) {
                    sum += lower[j * n + k] * lower[j * n + k];
                }
                float diag = matrix[i * n + j] - sum;
                if (diag < 0.0f) {
                    return false;
                }
                lower[i * n + j] = sqrtf(diag);
            } else {
                for (uint32_t k = 0; k < j; k++) {
                    sum += lower[i * n + k] * lower[j * n + k];
                }
                float d = lower[j * n + j];
                if (fabsf(d) < 1e-10f) {
                    return false;
                }
                lower[i * n + j] = (matrix[i * n + j] - sum) / d;
            }
        }
    }

    return true;
}

//=============================================================================
// Configuration
//=============================================================================

pink_spatial_config_t pink_spatial_default_config(void) {
    pink_spatial_config_t config = {0};

    config.num_regions = 4;
    config.length_constant = 20.0f;  // 20mm
    config.min_correlation = 0.05f;
    config.decay_type = PINK_SPATIAL_DECAY_EXPONENTIAL;
    config.power_law_exponent = 2.0f;
    config.sample_rate = 1000.0f;
    config.seed = 0;

    // Default: 4 cortical regions
    config.regions[0] = (pink_spatial_region_t){"V1", 0.0f, -80.0f, 0.0f, 1.0f, 0.05f};
    config.regions[1] = (pink_spatial_region_t){"MT", 40.0f, -60.0f, 10.0f, 1.0f, 0.05f};
    config.regions[2] = (pink_spatial_region_t){"PPC", 30.0f, -40.0f, 40.0f, 1.0f, 0.05f};
    config.regions[3] = (pink_spatial_region_t){"PFC", 30.0f, 50.0f, 20.0f, 1.0f, 0.05f};

    return config;
}

pink_spatial_config_t pink_spatial_network_config(const char* network_type) {
    pink_spatial_config_t config = pink_spatial_default_config();

    if (strcmp(network_type, "visual") == 0) {
        config.num_regions = 5;
        config.regions[0] = (pink_spatial_region_t){"V1", 0, -90, 0, 1.0f, 0.04f};
        config.regions[1] = (pink_spatial_region_t){"V2", 10, -80, 5, 1.0f, 0.04f};
        config.regions[2] = (pink_spatial_region_t){"V4", 30, -70, 10, 1.0f, 0.05f};
        config.regions[3] = (pink_spatial_region_t){"MT", 40, -60, 15, 1.0f, 0.05f};
        config.regions[4] = (pink_spatial_region_t){"IT", 50, -30, -10, 1.0f, 0.06f};
    } else if (strcmp(network_type, "motor") == 0) {
        config.num_regions = 4;
        config.regions[0] = (pink_spatial_region_t){"M1", 35, -20, 60, 0.9f, 0.04f};
        config.regions[1] = (pink_spatial_region_t){"SMA", 5, 0, 70, 0.95f, 0.05f};
        config.regions[2] = (pink_spatial_region_t){"PMC", 40, 0, 50, 0.95f, 0.05f};
        config.regions[3] = (pink_spatial_region_t){"Cerebellum", 25, -60, -30, 1.1f, 0.06f};
    } else if (strcmp(network_type, "default_mode") == 0) {
        config.num_regions = 5;
        config.regions[0] = (pink_spatial_region_t){"mPFC", 5, 55, 5, 1.1f, 0.06f};
        config.regions[1] = (pink_spatial_region_t){"PCC", 5, -55, 25, 1.1f, 0.06f};
        config.regions[2] = (pink_spatial_region_t){"AG_L", -50, -60, 30, 1.0f, 0.05f};
        config.regions[3] = (pink_spatial_region_t){"AG_R", 50, -60, 30, 1.0f, 0.05f};
        config.regions[4] = (pink_spatial_region_t){"MTL", 30, -15, -25, 1.2f, 0.07f};
    } else if (strcmp(network_type, "salience") == 0) {
        config.num_regions = 4;
        config.regions[0] = (pink_spatial_region_t){"AI_L", -35, 20, 5, 0.9f, 0.05f};
        config.regions[1] = (pink_spatial_region_t){"AI_R", 35, 20, 5, 0.9f, 0.05f};
        config.regions[2] = (pink_spatial_region_t){"dACC", 5, 25, 35, 0.95f, 0.05f};
        config.regions[3] = (pink_spatial_region_t){"Amygdala", 25, 0, -20, 1.1f, 0.06f};
    }

    return config;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

pink_spatial_t* pink_spatial_create(const pink_spatial_config_t* config) {
    if (!config || config->num_regions == 0) return NULL;
    if (config->num_regions > PINK_SPATIAL_MAX_REGIONS) return NULL;

    pink_spatial_t* spatial = nimcp_calloc(1, sizeof(pink_spatial_t));
    if (!spatial) return NULL;

    memcpy(&spatial->config, config, sizeof(pink_spatial_config_t));

    uint32_t n = config->num_regions;
    spatial->distance_matrix = nimcp_calloc(n * n, sizeof(float));
    spatial->correlation_matrix = nimcp_calloc(n * n, sizeof(float));
    spatial->cholesky_matrix = nimcp_calloc(n * n, sizeof(float));

    if (!spatial->distance_matrix || !spatial->correlation_matrix ||
        !spatial->cholesky_matrix) {
        pink_spatial_destroy(spatial);
        return NULL;
    }

    // Create generators for each region
    for (uint32_t i = 0; i < n; i++) {
        pink_noise_config_t gen_config = pink_noise_default_config();
        gen_config.alpha = config->regions[i].alpha;
        gen_config.amplitude = config->regions[i].amplitude;
        gen_config.seed = config->seed + i;

        spatial->generators[i] = pink_noise_create(&gen_config);
        if (!spatial->generators[i]) {
            pink_spatial_destroy(spatial);
            return NULL;
        }
    }

    // Compute correlation matrices
    pink_spatial_compute_correlations(spatial);

    NIMCP_LOGGING_INFO("Created spatial pink noise with %u regions", n);
    return spatial;
}

void pink_spatial_destroy(pink_spatial_t* spatial) {
    if (!spatial) return;

    for (uint32_t i = 0; i < spatial->config.num_regions; i++) {
        if (spatial->generators[i]) {
            pink_noise_destroy(spatial->generators[i]);
        }
    }

    if (spatial->distance_matrix) nimcp_free(spatial->distance_matrix);
    if (spatial->correlation_matrix) nimcp_free(spatial->correlation_matrix);
    if (spatial->cholesky_matrix) nimcp_free(spatial->cholesky_matrix);

    nimcp_free(spatial);
}

//=============================================================================
// Correlation Functions
//=============================================================================

int pink_spatial_compute_correlations(pink_spatial_t* spatial) {
    if (!spatial) return -1;

    uint32_t n = spatial->config.num_regions;

    // Compute distance matrix
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            if (i == j) {
                spatial->distance_matrix[i * n + j] = 0.0f;
            } else {
                spatial->distance_matrix[i * n + j] = compute_distance(
                    spatial->config.regions[i].x,
                    spatial->config.regions[i].y,
                    spatial->config.regions[i].z,
                    spatial->config.regions[j].x,
                    spatial->config.regions[j].y,
                    spatial->config.regions[j].z
                );
            }
        }
    }

    // Compute correlation matrix
    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < n; j++) {
            if (i == j) {
                spatial->correlation_matrix[i * n + j] = 1.0f;
            } else {
                spatial->correlation_matrix[i * n + j] = compute_correlation(
                    spatial->distance_matrix[i * n + j],
                    &spatial->config
                );
            }
        }
    }

    // Compute Cholesky decomposition
    spatial->matrices_valid = cholesky_decomposition(
        spatial->correlation_matrix,
        spatial->cholesky_matrix,
        n
    );

    if (!spatial->matrices_valid) {
        NIMCP_LOGGING_WARN("Cholesky failed, using identity");
        for (uint32_t i = 0; i < n; i++) {
            spatial->cholesky_matrix[i * n + i] = 1.0f;
        }
        spatial->matrices_valid = true;
    }

    return 0;
}

int pink_spatial_set_correlation_matrix(
    pink_spatial_t* spatial,
    const float* matrix
) {
    if (!spatial || !matrix) return -1;

    uint32_t n = spatial->config.num_regions;
    memcpy(spatial->correlation_matrix, matrix, n * n * sizeof(float));

    spatial->matrices_valid = cholesky_decomposition(
        spatial->correlation_matrix,
        spatial->cholesky_matrix,
        n
    );

    return spatial->matrices_valid ? 0 : -1;
}

//=============================================================================
// Generation Functions
//=============================================================================

int pink_spatial_step(pink_spatial_t* spatial) {
    if (!spatial) return -1;

    uint32_t n = spatial->config.num_regions;

    // Generate independent pink noise
    for (uint32_t i = 0; i < n; i++) {
        pink_noise_generate_sample(spatial->generators[i],
                                   &spatial->independent_values[i]);
    }

    // Apply Cholesky transformation for spatial correlation
    for (uint32_t i = 0; i < n; i++) {
        float sum = 0.0f;
        for (uint32_t j = 0; j <= i; j++) {
            sum += spatial->cholesky_matrix[i * n + j] * spatial->independent_values[j];
        }
        spatial->current_values[i] = sum;
    }

    spatial->sample_count++;
    return 0;
}

float pink_spatial_get_region(
    const pink_spatial_t* spatial,
    uint32_t region_index
) {
    if (!spatial || region_index >= spatial->config.num_regions) return 0.0f;
    return spatial->current_values[region_index];
}

float pink_spatial_get_named(
    const pink_spatial_t* spatial,
    const char* name
) {
    if (!spatial || !name) return 0.0f;

    for (uint32_t i = 0; i < spatial->config.num_regions; i++) {
        if (spatial->config.regions[i].name &&
            strcmp(spatial->config.regions[i].name, name) == 0) {
            return spatial->current_values[i];
        }
    }

    return 0.0f;
}

int pink_spatial_get_all(
    const pink_spatial_t* spatial,
    float* values
) {
    if (!spatial || !values) return -1;
    memcpy(values, spatial->current_values,
           spatial->config.num_regions * sizeof(float));
    return 0;
}

float pink_spatial_get_correlation(
    const pink_spatial_t* spatial,
    uint32_t region_i,
    uint32_t region_j
) {
    if (!spatial) return 0.0f;
    if (region_i >= spatial->config.num_regions) return 0.0f;
    if (region_j >= spatial->config.num_regions) return 0.0f;

    uint32_t n = spatial->config.num_regions;
    return spatial->correlation_matrix[region_i * n + region_j];
}

float pink_spatial_get_distance(
    const pink_spatial_t* spatial,
    uint32_t region_i,
    uint32_t region_j
) {
    if (!spatial) return 0.0f;
    if (region_i >= spatial->config.num_regions) return 0.0f;
    if (region_j >= spatial->config.num_regions) return 0.0f;

    uint32_t n = spatial->config.num_regions;
    return spatial->distance_matrix[region_i * n + region_j];
}

int pink_spatial_add_region(
    pink_spatial_t* spatial,
    const char* name,
    float x, float y, float z,
    float alpha,
    float amplitude
) {
    if (!spatial) return -1;
    if (spatial->config.num_regions >= PINK_SPATIAL_MAX_REGIONS) return -1;

    uint32_t idx = spatial->config.num_regions;
    spatial->config.regions[idx].name = name;
    spatial->config.regions[idx].x = x;
    spatial->config.regions[idx].y = y;
    spatial->config.regions[idx].z = z;
    spatial->config.regions[idx].alpha = alpha;
    spatial->config.regions[idx].amplitude = amplitude;

    // Create generator for new region
    pink_noise_config_t gen_config = pink_noise_default_config();
    gen_config.alpha = alpha;
    gen_config.amplitude = amplitude;
    gen_config.seed = spatial->config.seed + idx;

    spatial->generators[idx] = pink_noise_create(&gen_config);
    if (!spatial->generators[idx]) return -1;

    spatial->config.num_regions++;

    // Recompute correlations
    pink_spatial_compute_correlations(spatial);

    return 0;
}

int pink_spatial_reset(pink_spatial_t* spatial, uint32_t new_seed) {
    if (!spatial) return -1;

    for (uint32_t i = 0; i < spatial->config.num_regions; i++) {
        uint32_t seed = (new_seed != 0) ? new_seed + i : spatial->config.seed + i;
        pink_noise_reset(spatial->generators[i], seed);
        spatial->current_values[i] = 0.0f;
        spatial->independent_values[i] = 0.0f;
    }

    spatial->sample_count = 0;
    return 0;
}
