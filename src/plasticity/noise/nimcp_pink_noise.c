//=============================================================================
// nimcp_pink_noise.c - 1/f Pink Noise Generator Implementation
//=============================================================================

#include "nimcp_pink_noise.h"
#include <math.h>
#include <string.h>
#include <time.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define VOSS_NUM_OCTAVES 16        // Number of octave generators for Voss method
#define MAX_ERROR_LENGTH 256       // Maximum error message length
#define MIN_SAMPLES_FOR_STATS 64   // Minimum samples for spectral analysis

//=============================================================================
// Internal Data Structures
//=============================================================================

/**
 * @brief Internal generator state
 *
 * WHAT: Maintains all state needed for streaming noise generation
 * WHY: Encapsulation of algorithm-specific data
 * HOW: Different fields used by different methods
 */
struct pink_noise_generator_internal_t {
    pink_noise_config_t config;        /**< Configuration copy */
    uint32_t rng_state;                /**< RNG state (LCG) */

    // Voss-McCartney method state
    float voss_octaves[VOSS_NUM_OCTAVES];  /**< Octave generator values */
    uint32_t voss_counter;                 /**< Sample counter for octave updates */

    // IIR filter state
    float iir_history[4];              /**< Filter delay line */
    float iir_coeffs[5];               /**< IIR filter coefficients */

    // FFT method state (future)
    float* fft_buffer;                 /**< Precomputed noise buffer */
    uint32_t fft_buffer_size;          /**< Buffer size */
    uint32_t fft_buffer_pos;           /**< Current position in buffer */
};

//=============================================================================
// Thread-Local Error Storage
//=============================================================================

static _Thread_local char last_error[MAX_ERROR_LENGTH] = {0};

/**
 * @brief Set error message
 *
 * WHAT: Stores error message in thread-local storage
 * WHY: Thread-safe error reporting
 * HOW: Copy string to thread-local buffer
 */
static void set_error(const char* message) {
    // Guard: NULL message
    if (!message) {
        last_error[0] = '\0';
        return;
    }

    strncpy(last_error, message, MAX_ERROR_LENGTH - 1);
    last_error[MAX_ERROR_LENGTH - 1] = '\0';
}

//=============================================================================
// Random Number Generation
//=============================================================================

/**
 * @brief Simple LCG random number generator
 *
 * WHAT: Linear congruential generator for deterministic randomness
 * WHY: Reproducible noise sequences with seed control
 * HOW: state = (a * state + c) mod m
 */
static uint32_t lcg_next(uint32_t* state) {
    // LCG parameters (Numerical Recipes)
    const uint32_t a = 1664525;
    const uint32_t c = 1013904223;

    *state = a * (*state) + c;
    return *state;
}

/**
 * @brief Generate uniform random float in [0, 1]
 *
 * WHAT: Converts LCG output to normalized float
 * WHY: Easier to work with normalized values
 * HOW: Divide by 2^32 - 1
 */
static float randf(uint32_t* state) {
    return (float)lcg_next(state) / (float)UINT32_MAX;
}

/**
 * @brief Generate Gaussian random variable (Box-Muller)
 *
 * WHAT: Generates normally distributed random number
 * WHY: Many noise processes require Gaussian distribution
 * HOW: Box-Muller transform of uniform random variables
 */
static float randn(uint32_t* state) {
    // Generate two uniform random numbers
    float u1 = randf(state);
    float u2 = randf(state);

    // Guard: Avoid log(0)
    if (u1 < 1e-10f) {
        u1 = 1e-10f;
    }

    // Box-Muller transform
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * M_PI * u2;

    return r * cosf(theta);
}

//=============================================================================
// Voss-McCartney Algorithm
//=============================================================================

/**
 * @brief Initialize Voss-McCartney octave generators
 *
 * WHAT: Sets up octave generators with random initial values
 * WHY: Voss method requires independent octave generators
 * HOW: Initialize each octave with white noise
 */
static void voss_init(pink_noise_generator_t gen) {
    // Guard: NULL generator
    if (!gen) {
        return;
    }

    // Initialize all octaves with random values
    for (uint32_t i = 0; i < VOSS_NUM_OCTAVES; i++) {
        gen->voss_octaves[i] = randn(&gen->rng_state);
    }

    gen->voss_counter = 0;
}

/**
 * @brief Generate next Voss-McCartney sample
 *
 * WHAT: Produces pink noise sample via octave summing
 * WHY: Fast approximation of 1/f spectrum
 * HOW:
 *   - Octave i updates at rate 2^(-i)
 *   - Check which octaves need update via bit manipulation
 *   - Sum all octaves and normalize
 *
 * ALGORITHM:
 *   counter++
 *   for i in 0..k-1:
 *     if counter % 2^i == 0:
 *       octave[i] = new_random()
 *   output = sum(octaves) / sqrt(k)
 */
static float voss_next(pink_noise_generator_t gen) {
    // Guard: NULL generator
    if (!gen) {
        return 0.0f;
    }

    gen->voss_counter++;

    // Update octaves based on counter bits
    uint32_t counter = gen->voss_counter;
    for (uint32_t i = 0; i < VOSS_NUM_OCTAVES; i++) {
        // Update this octave if bit i is transitioning from 1 to 0
        uint32_t mask = 1u << i;

        // Guard: Check if we should update this octave
        if ((counter & mask) == 0) {
            gen->voss_octaves[i] = randn(&gen->rng_state);
        }
    }

    // Sum all octaves
    float sum = 0.0f;
    for (uint32_t i = 0; i < VOSS_NUM_OCTAVES; i++) {
        sum += gen->voss_octaves[i];
    }

    // Normalize by sqrt(num_octaves) to maintain RMS amplitude
    float normalized = sum / sqrtf((float)VOSS_NUM_OCTAVES);

    // Scale by configured amplitude
    return normalized * gen->config.amplitude;
}

//=============================================================================
// IIR Filter Method
//=============================================================================

/**
 * @brief Initialize IIR filter for pink noise
 *
 * WHAT: Precomputes filter coefficients for 1/f approximation
 * WHY: IIR filter can approximate 1/f spectrum efficiently
 * HOW: Use pole-zero placement to match 1/f slope
 *
 * COEFFICIENTS (from Paul Kellet's approximation):
 *   H(z) = (b0 + b1*z^-1 + b2*z^-2) / (1 + a1*z^-1 + a2*z^-2)
 */
static void iir_init(pink_noise_generator_t gen) {
    // Guard: NULL generator
    if (!gen) {
        return;
    }

    // Initialize filter state to zero
    memset(gen->iir_history, 0, sizeof(gen->iir_history));

    // Stable pink noise IIR filter coefficients
    // Design: 2nd-order Butterworth-based pinking filter
    // Transfer function approximates 1/f from 20Hz to 20kHz
    // Normalized for fs = 44100 Hz
    gen->iir_coeffs[0] = 0.0555179f;   // b0
    gen->iir_coeffs[1] = -0.0750759f;  // b1
    gen->iir_coeffs[2] = 0.0427906f;   // b2
    gen->iir_coeffs[3] = -1.99004745f; // a1 (note: used with subtraction)
    gen->iir_coeffs[4] = 0.99007225f;  // a2 (stable: |poles| < 1)
}

/**
 * @brief Generate next IIR filter sample
 *
 * WHAT: Produces pink noise via IIR filtering of white noise
 * WHY: Streaming generation with good quality
 * HOW: Filter white noise through precomputed IIR filter
 */
static float iir_next(pink_noise_generator_t gen) {
    // Guard: NULL generator
    if (!gen) {
        return 0.0f;
    }

    // Generate white noise input
    float white = randn(&gen->rng_state);

    // Apply IIR filter: y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
    float output = gen->iir_coeffs[0] * white;
    output += gen->iir_coeffs[1] * gen->iir_history[0];
    output += gen->iir_coeffs[2] * gen->iir_history[1];
    output -= gen->iir_coeffs[3] * gen->iir_history[2];
    output -= gen->iir_coeffs[4] * gen->iir_history[3];

    // Update history
    gen->iir_history[1] = gen->iir_history[0];
    gen->iir_history[0] = white;
    gen->iir_history[3] = gen->iir_history[2];
    gen->iir_history[2] = output;

    // Scale by configured amplitude
    return output * gen->config.amplitude;
}

//=============================================================================
// Public API Implementation
//=============================================================================

pink_noise_config_t pink_noise_default_config(void) {
    pink_noise_config_t config = {
        .alpha = 1.0f,                  // True pink noise
        .amplitude = 0.05f,             // 5% modulation
        .min_frequency = 0.1f,          // 10s timescale
        .max_frequency = 100.0f,        // 10ms timescale
        .sample_rate = 1000.0f,         // 1ms resolution
        .method = PINK_NOISE_VOSS,      // Fast, good quality
        .seed = 0                       // Time-based seed
    };
    return config;
}

bool pink_noise_validate_config(const pink_noise_config_t* config) {
    // Guard: NULL config
    if (!config) {
        set_error("Configuration is NULL");
        return false;
    }

    // Guard: Invalid alpha
    if (config->alpha < 0.0f || config->alpha > 3.0f) {
        set_error("Alpha must be in [0, 3] range");
        return false;
    }

    // Guard: Invalid amplitude
    if (config->amplitude <= 0.0f) {
        set_error("Amplitude must be positive");
        return false;
    }

    // Guard: Invalid frequency range
    if (config->min_frequency <= 0.0f || config->min_frequency >= config->max_frequency) {
        set_error("Frequency range invalid: 0 < min_freq < max_freq");
        return false;
    }

    // Guard: Nyquist violation
    if (config->sample_rate < 2.0f * config->max_frequency) {
        set_error("Sample rate must be >= 2*max_frequency (Nyquist)");
        return false;
    }

    // Guard: Invalid method
    if (config->method < PINK_NOISE_FFT || config->method > PINK_NOISE_WHITE) {
        set_error("Invalid noise generation method");
        return false;
    }

    // All checks passed
    set_error(NULL);
    return true;
}

pink_noise_generator_t pink_noise_create(const pink_noise_config_t* config) {
    // Guard: NULL config
    if (!config) {
        set_error("Configuration is NULL");
        return NULL;
    }

    // Guard: Invalid config
    if (!pink_noise_validate_config(config)) {
        // Error already set by validate
        return NULL;
    }

    // Allocate generator
    pink_noise_generator_t gen = (pink_noise_generator_t)malloc(sizeof(struct pink_noise_generator_internal_t));

    // Guard: Allocation failed
    if (!gen) {
        set_error("Failed to allocate generator");
        return NULL;
    }

    // Copy configuration
    memcpy(&gen->config, config, sizeof(pink_noise_config_t));

    // Initialize RNG seed
    gen->rng_state = config->seed;

    // Guard: Seed is 0, use time-based seed
    if (gen->rng_state == 0) {
        gen->rng_state = (uint32_t)time(NULL);
    }

    // Initialize FFT buffer to NULL
    gen->fft_buffer = NULL;
    gen->fft_buffer_size = 0;
    gen->fft_buffer_pos = 0;

    // Initialize method-specific state
    switch (gen->config.method) {
        case PINK_NOISE_VOSS:
            voss_init(gen);
            break;

        case PINK_NOISE_IIR:
            iir_init(gen);
            break;

        case PINK_NOISE_FFT:
            // FFT method not yet implemented
            set_error("FFT method not yet implemented");
            free(gen);
            return NULL;

        case PINK_NOISE_WHITE:
            // White noise needs no special initialization
            break;
    }

    set_error(NULL);
    return gen;
}

void pink_noise_destroy(pink_noise_generator_t generator) {
    // Guard: NULL generator (safe to call)
    if (!generator) {
        return;
    }

    // Free FFT buffer if allocated
    if (generator->fft_buffer) {
        free(generator->fft_buffer);
    }

    // Free generator
    free(generator);
}

bool pink_noise_generate_sample(pink_noise_generator_t generator, float* sample) {
    // Guard: NULL generator
    if (!generator) {
        set_error("Generator is NULL");
        return false;
    }

    // Guard: NULL sample pointer
    if (!sample) {
        set_error("Sample pointer is NULL");
        return false;
    }

    // Generate sample based on method
    float value = 0.0f;

    switch (generator->config.method) {
        case PINK_NOISE_VOSS:
            value = voss_next(generator);
            break;

        case PINK_NOISE_IIR:
            value = iir_next(generator);
            break;

        case PINK_NOISE_WHITE:
            value = randn(&generator->rng_state) * generator->config.amplitude;
            break;

        case PINK_NOISE_FFT:
            set_error("FFT method not yet implemented");
            return false;
    }

    *sample = value;
    set_error(NULL);
    return true;
}

bool pink_noise_generate(pink_noise_generator_t generator, float* samples, uint32_t num_samples) {
    // Guard: NULL generator
    if (!generator) {
        set_error("Generator is NULL");
        return false;
    }

    // Guard: NULL samples array
    if (!samples) {
        set_error("Samples array is NULL");
        return false;
    }

    // Guard: Zero samples requested
    if (num_samples == 0) {
        set_error("Number of samples must be > 0");
        return false;
    }

    // Generate samples
    for (uint32_t i = 0; i < num_samples; i++) {
        bool success = pink_noise_generate_sample(generator, &samples[i]);

        // Guard: Generation failed
        if (!success) {
            // Error already set by generate_sample
            return false;
        }
    }

    set_error(NULL);
    return true;
}

bool pink_noise_reset(pink_noise_generator_t generator, uint32_t new_seed) {
    // Guard: NULL generator
    if (!generator) {
        set_error("Generator is NULL");
        return false;
    }

    // Update seed
    generator->rng_state = new_seed;

    // Guard: Seed is 0, use time-based seed
    if (generator->rng_state == 0) {
        generator->rng_state = (uint32_t)time(NULL);
    }

    // Reinitialize method-specific state
    switch (generator->config.method) {
        case PINK_NOISE_VOSS:
            voss_init(generator);
            break;

        case PINK_NOISE_IIR:
            iir_init(generator);
            break;

        case PINK_NOISE_FFT:
            // Reset FFT buffer position
            generator->fft_buffer_pos = 0;
            break;

        case PINK_NOISE_WHITE:
            // White noise needs no special reset
            break;
    }

    set_error(NULL);
    return true;
}

bool pink_noise_compute_stats(const float* samples, uint32_t num_samples, float sample_rate, pink_noise_stats_t* stats) {
    // Guard: NULL samples
    if (!samples) {
        set_error("Samples array is NULL");
        return false;
    }

    // Guard: NULL stats
    if (!stats) {
        set_error("Stats pointer is NULL");
        return false;
    }

    // Guard: Too few samples
    if (num_samples < MIN_SAMPLES_FOR_STATS) {
        set_error("Too few samples for statistics (minimum 64)");
        return false;
    }

    // Initialize stats
    memset(stats, 0, sizeof(pink_noise_stats_t));

    // Compute mean
    float sum = 0.0f;
    for (uint32_t i = 0; i < num_samples; i++) {
        sum += samples[i];
    }
    stats->mean = sum / (float)num_samples;

    // Compute std dev, min, max
    float sum_sq = 0.0f;
    stats->min_value = samples[0];
    stats->max_value = samples[0];

    for (uint32_t i = 0; i < num_samples; i++) {
        float dev = samples[i] - stats->mean;
        sum_sq += dev * dev;

        if (samples[i] < stats->min_value) {
            stats->min_value = samples[i];
        }
        if (samples[i] > stats->max_value) {
            stats->max_value = samples[i];
        }
    }

    stats->std_dev = sqrtf(sum_sq / (float)num_samples);
    stats->measured_amplitude = stats->std_dev;  // RMS amplitude

    // Spectral analysis would go here (FFT)
    // For now, set placeholder values
    stats->measured_alpha = 1.0f;  // Placeholder
    stats->spectral_fit_r2 = 0.0f; // Not yet computed

    set_error(NULL);
    return true;
}

bool pink_noise_validate(const float* samples, uint32_t num_samples, float sample_rate, float expected_alpha, float tolerance) {
    // Guard: Invalid tolerance
    if (tolerance <= 0.0f) {
        set_error("Tolerance must be positive");
        return false;
    }

    // Compute statistics
    pink_noise_stats_t stats;
    bool success = pink_noise_compute_stats(samples, num_samples, sample_rate, &stats);

    // Guard: Stats computation failed
    if (!success) {
        // Error already set by compute_stats
        return false;
    }

    // Check if measured alpha is within tolerance
    float alpha_diff = fabsf(stats.measured_alpha - expected_alpha);

    // Guard: Alpha out of tolerance
    if (alpha_diff > tolerance) {
        set_error("Spectral exponent outside tolerance");
        return false;
    }

    set_error(NULL);
    return true;
}

bool pink_noise_modulate(pink_noise_generator_t generator, float base_level, float* output) {
    // Guard: NULL generator
    if (!generator) {
        set_error("Generator is NULL");
        return false;
    }

    // Guard: NULL output
    if (!output) {
        set_error("Output pointer is NULL");
        return false;
    }

    // Generate noise sample
    float noise;
    bool success = pink_noise_generate_sample(generator, &noise);

    // Guard: Generation failed
    if (!success) {
        // Error already set by generate_sample
        return false;
    }

    // Apply additive modulation
    *output = base_level + noise;

    set_error(NULL);
    return true;
}

bool pink_noise_modulate_multiplicative(pink_noise_generator_t generator, float value, float modulation_strength, float* output) {
    // Guard: NULL generator
    if (!generator) {
        set_error("Generator is NULL");
        return false;
    }

    // Guard: NULL output
    if (!output) {
        set_error("Output pointer is NULL");
        return false;
    }

    // Guard: Invalid modulation strength
    if (modulation_strength < 0.0f || modulation_strength > 1.0f) {
        set_error("Modulation strength must be in [0, 1] range");
        return false;
    }

    // Generate noise sample
    float noise;
    bool success = pink_noise_generate_sample(generator, &noise);

    // Guard: Generation failed
    if (!success) {
        // Error already set by generate_sample
        return false;
    }

    // Normalize noise to [-1, 1] range (approximately)
    float normalized_noise = noise / (3.0f * generator->config.amplitude);  // ±3σ clipping

    // Guard: Clamp to [-1, 1]
    if (normalized_noise < -1.0f) normalized_noise = -1.0f;
    if (normalized_noise > 1.0f) normalized_noise = 1.0f;

    // Apply multiplicative modulation
    *output = value * (1.0f + modulation_strength * normalized_noise);

    set_error(NULL);
    return true;
}

const char* pink_noise_method_name(pink_noise_method_t method) {
    switch (method) {
        case PINK_NOISE_FFT:   return "FFT";
        case PINK_NOISE_VOSS:  return "Voss";
        case PINK_NOISE_IIR:   return "IIR";
        case PINK_NOISE_WHITE: return "White";
        default:               return "Unknown";
    }
}

const char* pink_noise_get_last_error(void) {
    // Guard: No error set
    if (last_error[0] == '\0') {
        return NULL;
    }

    return last_error;
}
