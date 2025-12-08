//=============================================================================
// nimcp_speech_motor.c - Speech Motor Planning Implementation
//=============================================================================
/**
 * @file nimcp_speech_motor.c
 * @brief Implementation of speech motor planning for Broca's region
 *
 * WHAT: Motor planning system converting phonemes to articulatory commands
 * WHY:  Enable biologically-inspired speech production
 * HOW:  Phoneme feature mapping, command generation, coarticulation smoothing
 *
 * ARCHITECTURE:
 * - Phoneme-to-feature mapping table (place, manner, voicing)
 * - Articulator state tracking (current position, velocity)
 * - Command queue with temporal ordering
 * - Coarticulation engine (blend adjacent phoneme targets)
 *
 * @author NIMCP Development Team
 * @version 2.7.0
 * @date 2025-11-22
 */

// Bio-async integration
#include "async/nimcp_bio_async.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"

// Logging integration
#include "utils/logging/nimcp_logging.h"

// Unified memory integration
#include "utils/memory/nimcp_unified_memory.h"

#include "core/brain/regions/broca/nimcp_speech_motor.h"
#include "utils/memory/nimcp_memory.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

#define LOG_MODULE "BROCA_SPEECH"

//=============================================================================
// Phoneme Feature Structures
//=============================================================================

/**
 * @brief Phonetic features for articulatory planning
 *
 * WHAT: Linguistic features determining articulator targets
 * WHY:  Map abstract phonemes to concrete motor positions
 * HOW:  Place, manner, voicing encode articulator configurations
 */
typedef struct {
    float lips_position;      ///< Lip position [0=closed, 1=open]
    float tongue_position;    ///< Tongue position [0=front, 1=back]
    float jaw_height;         ///< Jaw height [0=closed, 1=open]
    float larynx_tension;     ///< Vocal fold tension [0=slack, 1=tense]
    float velum_opening;      ///< Velopharyngeal port [0=closed, 1=open]
    float duration_ms;        ///< Typical phoneme duration
} phoneme_features_t;

//=============================================================================
// Articulator State
//=============================================================================

/**
 * @brief Current state of one articulator
 */
typedef struct {
    float position;           ///< Current position [0.0, 1.0]
    float velocity;           ///< Current velocity (units/sec)
    double last_update_time;  ///< Last update timestamp (ms)
} articulator_state_t;

//=============================================================================
// Internal Structure
//=============================================================================

/**
 * @brief Speech motor planner internal structure
 */
struct speech_motor_planner {
    // Configuration
    speech_motor_config_t config;

    // Articulator states
    articulator_state_t articulators[SPEECH_MOTOR_NUM_ARTICULATORS];

    // Command queue
    motor_command_t* command_queue;
    uint32_t queue_head;          ///< Next command to retrieve
    uint32_t queue_tail;          ///< Next position to write
    uint32_t queue_count;         ///< Number of queued commands

    // Timing
    double current_time_ms;       ///< Current planning time
    double next_phoneme_time_ms;  ///< When next phoneme starts

    // Statistics
    speech_motor_stats_t stats;

    // Coarticulation state
    uint8_t last_phoneme;         ///< Previous phoneme for coarticulation
    bool has_last_phoneme;        ///< Whether last_phoneme is valid

    // Trajectory interpolation settings
    bool enable_interpolation;    ///< Whether to generate interpolated waypoints
    uint32_t interpolation_points; ///< Number of intermediate points (2-10)
};

//=============================================================================
// Phoneme Feature Mapping Table
//=============================================================================

/**
 * WHAT: Comprehensive phoneme-to-articulator mapping
 * WHY:  Map IPA/SAMPA phoneme symbols to motor targets
 * HOW:  Lookup table indexed by ASCII phoneme character
 *
 * BIOLOGY: Based on articulatory phonetics
 * - Vowels: Open jaw, varying tongue position (front/back, high/low)
 * - Stops: Complete closures at bilabial, alveolar, velar, glottal places
 * - Fricatives: Narrow constrictions with turbulent airflow
 * - Affricates: Stop + fricative combinations
 * - Nasals: Velum open for nasal coupling
 * - Liquids: Lateral and rhotic approximants
 * - Glides: Rapid transitions (semivowels)
 *
 * FEATURES:
 * - lips_position:   [0=closed, 1=open/spread]
 * - tongue_position: [0=front, 1=back]
 * - jaw_height:      [0=closed, 1=open]
 * - larynx_tension:  [0=slack/voiceless, 1=tense/voiced]
 * - velum_opening:   [0=closed, 1=open for nasals]
 */
static const phoneme_features_t phoneme_table[256] = {
    /* Default/silence (all neutral) */
    [0]   = {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 50.0f},
    [' '] = {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 50.0f},
    ['_'] = {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 100.0f},  /* Pause */

    /*=========================================================================
     * VOWELS - Cardinal vowels and common allophones
     *=======================================================================*/

    /* Front vowels (low tongue_position) */
    ['i'] = {0.7f, 0.1f, 0.3f, 0.8f, 0.0f, 70.0f},   /* /i/ - close front unrounded */
    ['I'] = {0.6f, 0.2f, 0.4f, 0.8f, 0.0f, 65.0f},   /* /ɪ/ - near-close front */
    ['e'] = {0.7f, 0.2f, 0.5f, 0.8f, 0.0f, 75.0f},   /* /e/ - close-mid front */
    ['E'] = {0.7f, 0.3f, 0.6f, 0.8f, 0.0f, 80.0f},   /* /ɛ/ - open-mid front */
    ['&'] = {0.8f, 0.3f, 0.8f, 0.8f, 0.0f, 85.0f},   /* /æ/ - near-open front */

    /* Central vowels */
    ['@'] = {0.6f, 0.5f, 0.5f, 0.7f, 0.0f, 60.0f},   /* /ə/ - schwa */
    ['3'] = {0.6f, 0.5f, 0.6f, 0.8f, 0.0f, 90.0f},   /* /ɜ/ - open-mid central */
    ['V'] = {0.7f, 0.5f, 0.7f, 0.8f, 0.0f, 75.0f},   /* /ʌ/ - open-mid back unrounded */

    /* Back vowels (high tongue_position) */
    ['u'] = {0.2f, 0.9f, 0.3f, 0.8f, 0.0f, 70.0f},   /* /u/ - close back rounded */
    ['U'] = {0.3f, 0.8f, 0.4f, 0.8f, 0.0f, 65.0f},   /* /ʊ/ - near-close back */
    ['o'] = {0.4f, 0.8f, 0.5f, 0.8f, 0.0f, 75.0f},   /* /o/ - close-mid back rounded */
    ['O'] = {0.5f, 0.7f, 0.6f, 0.8f, 0.0f, 80.0f},   /* /ɔ/ - open-mid back rounded */
    ['a'] = {0.8f, 0.6f, 0.9f, 0.8f, 0.0f, 85.0f},   /* /a/ - open front */
    ['A'] = {0.8f, 0.8f, 0.9f, 0.8f, 0.0f, 90.0f},   /* /ɑ/ - open back unrounded */
    ['Q'] = {0.6f, 0.9f, 0.8f, 0.8f, 0.0f, 85.0f},   /* /ɒ/ - open back rounded */

    /* Front rounded vowels */
    ['y'] = {0.3f, 0.1f, 0.3f, 0.8f, 0.0f, 75.0f},   /* /y/ - close front rounded */
    ['Y'] = {0.4f, 0.2f, 0.4f, 0.8f, 0.0f, 70.0f},   /* /ʏ/ - near-close front rounded */
    ['2'] = {0.4f, 0.2f, 0.5f, 0.8f, 0.0f, 80.0f},   /* /ø/ - close-mid front rounded */
    ['9'] = {0.5f, 0.3f, 0.6f, 0.8f, 0.0f, 85.0f},   /* /œ/ - open-mid front rounded */

    /*=========================================================================
     * PLOSIVES/STOPS - Complete closures with release burst
     *=======================================================================*/

    /* Bilabial stops */
    ['p'] = {0.0f, 0.5f, 0.3f, 0.0f, 0.0f, 60.0f},   /* /p/ - voiceless bilabial */
    ['b'] = {0.0f, 0.5f, 0.3f, 0.9f, 0.0f, 65.0f},   /* /b/ - voiced bilabial */

    /* Alveolar stops */
    ['t'] = {0.5f, 0.2f, 0.3f, 0.0f, 0.0f, 55.0f},   /* /t/ - voiceless alveolar */
    ['d'] = {0.5f, 0.2f, 0.3f, 0.9f, 0.0f, 60.0f},   /* /d/ - voiced alveolar */

    /* Velar stops */
    ['k'] = {0.5f, 0.8f, 0.4f, 0.0f, 0.0f, 65.0f},   /* /k/ - voiceless velar */
    ['g'] = {0.5f, 0.8f, 0.4f, 0.9f, 0.0f, 70.0f},   /* /g/ - voiced velar */

    /* Glottal stop */
    ['?'] = {0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 40.0f},   /* /ʔ/ - glottal stop */

    /*=========================================================================
     * FRICATIVES - Narrow constrictions with turbulent airflow
     *=======================================================================*/

    /* Labiodental fricatives */
    ['f'] = {0.1f, 0.5f, 0.4f, 0.0f, 0.0f, 90.0f},   /* /f/ - voiceless labiodental */
    ['v'] = {0.1f, 0.5f, 0.4f, 0.9f, 0.0f, 85.0f},   /* /v/ - voiced labiodental */

    /* Dental fricatives */
    ['T'] = {0.4f, 0.15f, 0.4f, 0.0f, 0.0f, 95.0f},  /* /θ/ - voiceless dental */
    ['D'] = {0.4f, 0.15f, 0.4f, 0.9f, 0.0f, 90.0f},  /* /ð/ - voiced dental */

    /* Alveolar fricatives */
    ['s'] = {0.5f, 0.2f, 0.4f, 0.0f, 0.0f, 100.0f},  /* /s/ - voiceless alveolar */
    ['z'] = {0.5f, 0.2f, 0.4f, 0.9f, 0.0f, 95.0f},   /* /z/ - voiced alveolar */

    /* Post-alveolar fricatives */
    ['S'] = {0.4f, 0.35f, 0.4f, 0.0f, 0.0f, 100.0f}, /* /ʃ/ - voiceless post-alveolar */
    ['Z'] = {0.4f, 0.35f, 0.4f, 0.9f, 0.0f, 95.0f},  /* /ʒ/ - voiced post-alveolar */

    /* Palatal fricatives */
    ['C'] = {0.5f, 0.3f, 0.4f, 0.0f, 0.0f, 90.0f},   /* /ç/ - voiceless palatal */

    /* Velar fricatives */
    ['x'] = {0.5f, 0.8f, 0.5f, 0.0f, 0.0f, 85.0f},   /* /x/ - voiceless velar */
    ['G'] = {0.5f, 0.8f, 0.5f, 0.9f, 0.0f, 80.0f},   /* /ɣ/ - voiced velar */

    /* Glottal fricative */
    ['h'] = {0.7f, 0.5f, 0.6f, 0.0f, 0.0f, 80.0f},   /* /h/ - voiceless glottal */

    /*=========================================================================
     * AFFRICATES - Stop + fricative combinations
     *=======================================================================*/

    /* Using composite symbols for affricates */
    /* Note: In practice, affricates are usually decomposed into stop+fricative */

    /*=========================================================================
     * NASALS - Velum open for nasal coupling
     *=======================================================================*/

    ['m'] = {0.0f, 0.5f, 0.4f, 0.9f, 1.0f, 75.0f},   /* /m/ - bilabial nasal */
    ['n'] = {0.5f, 0.2f, 0.4f, 0.9f, 1.0f, 70.0f},   /* /n/ - alveolar nasal */
    ['N'] = {0.5f, 0.8f, 0.4f, 0.9f, 1.0f, 75.0f},   /* /ŋ/ - velar nasal */
    ['J'] = {0.5f, 0.3f, 0.4f, 0.9f, 1.0f, 70.0f},   /* /ɲ/ - palatal nasal */

    /*=========================================================================
     * LIQUIDS - Lateral and rhotic approximants
     *=======================================================================*/

    ['l'] = {0.6f, 0.2f, 0.5f, 0.9f, 0.0f, 65.0f},   /* /l/ - alveolar lateral */
    ['L'] = {0.6f, 0.8f, 0.5f, 0.9f, 0.0f, 70.0f},   /* /ɫ/ - velarized lateral (dark L) */
    ['r'] = {0.6f, 0.35f, 0.5f, 0.9f, 0.0f, 70.0f},  /* /ɹ/ - alveolar approximant */
    ['R'] = {0.6f, 0.9f, 0.5f, 0.9f, 0.0f, 75.0f},   /* /ʁ/ - uvular fricative */
    ['4'] = {0.6f, 0.2f, 0.5f, 0.9f, 0.0f, 40.0f},   /* /ɾ/ - alveolar tap */

    /*=========================================================================
     * GLIDES/SEMIVOWELS - Rapid transitions
     *=======================================================================*/

    ['w'] = {0.2f, 0.9f, 0.4f, 0.9f, 0.0f, 60.0f},   /* /w/ - labial-velar approximant */
    ['j'] = {0.7f, 0.1f, 0.4f, 0.9f, 0.0f, 55.0f},   /* /j/ - palatal approximant */
    ['H'] = {0.3f, 0.1f, 0.4f, 0.9f, 0.0f, 55.0f},   /* /ɥ/ - labial-palatal approximant */
};

//=============================================================================
// Internal Helper Functions
//=============================================================================

/* Forward declaration for velocity calculation */
static float calculate_velocity(float distance, float duration_ms);

/**
 * @brief Initialize articulator to neutral position
 */
static void init_articulator(articulator_state_t* art) {
    if (!art) return;
    art->position = 0.5f;  // Neutral/rest position
    art->velocity = 0.0f;
    art->last_update_time = 0.0;
}

/**
 * @brief Get phoneme features from table
 */
static phoneme_features_t get_phoneme_features(uint8_t phoneme) {
    return phoneme_table[phoneme & 0xFF];
}

/**
 * @brief Enqueue a motor command
 */
static bool enqueue_command(speech_motor_planner_t* planner, const motor_command_t* cmd) {
    if (!planner || !cmd) return false;

    // Check if queue is full
    if (planner->queue_count >= planner->config.max_commands) {
        return false;
    }

    // Add command to queue
    planner->command_queue[planner->queue_tail] = *cmd;
    planner->queue_tail = (planner->queue_tail + 1) % planner->config.max_commands;
    planner->queue_count++;

    return true;
}

/**
 * @brief Apply coarticulation blending
 */
static float apply_coarticulation(
    speech_motor_planner_t* planner,
    float current_target,
    float previous_target
) {
    if (!planner->config.enable_coarticulation || !planner->has_last_phoneme) {
        return current_target;
    }

    // Blend current and previous targets
    float strength = planner->config.coarticulation_strength;
    return current_target * (1.0f - strength) + previous_target * strength;
}

/**
 * @brief Generate motor commands for articulator positions
 */
static bool generate_commands_for_phoneme(
    speech_motor_planner_t* planner,
    const phoneme_features_t* features,
    uint8_t phoneme
) {
    if (!planner || !features) return false;

    motor_command_t cmd;
    cmd.phoneme = phoneme;
    cmd.timestamp = planner->next_phoneme_time_ms;
    cmd.velocity = planner->config.default_velocity;

    // Generate command for each articulator
    float targets[SPEECH_MOTOR_NUM_ARTICULATORS] = {
        features->lips_position,
        features->tongue_position,
        features->jaw_height,
        features->larynx_tension,
        features->velum_opening
    };

    // Apply coarticulation if enabled
    if (planner->has_last_phoneme) {
        phoneme_features_t last_features = get_phoneme_features(planner->last_phoneme);
        float last_targets[SPEECH_MOTOR_NUM_ARTICULATORS] = {
            last_features.lips_position,
            last_features.tongue_position,
            last_features.jaw_height,
            last_features.larynx_tension,
            last_features.velum_opening
        };

        for (int i = 0; i < SPEECH_MOTOR_NUM_ARTICULATORS; i++) {
            targets[i] = apply_coarticulation(planner, targets[i], last_targets[i]);
        }
    }

    // Create commands for each articulator
    for (int i = 0; i < SPEECH_MOTOR_NUM_ARTICULATORS; i++) {
        cmd.type = (articulator_type_t)i;
        cmd.position = targets[i];

        // Calculate distance-based velocity for more natural movement
        float current_pos = planner->articulators[i].position;
        float distance = targets[i] - current_pos;
        cmd.velocity = calculate_velocity(distance, features->duration_ms);

        if (!enqueue_command(planner, &cmd)) {
            return false;  // Queue full
        }

        // Update articulator state
        planner->articulators[i].position = targets[i];
        planner->articulators[i].velocity = cmd.velocity;
        planner->articulators[i].last_update_time = planner->next_phoneme_time_ms;
    }

    // Advance timing
    planner->next_phoneme_time_ms += features->duration_ms;

    // Update statistics
    planner->stats.commands_generated += SPEECH_MOTOR_NUM_ARTICULATORS;

    return true;
}

//=============================================================================
// Public API Implementation
//=============================================================================

speech_motor_config_t speech_motor_default_config(void) {
    speech_motor_config_t config;
    config.max_commands = SPEECH_MOTOR_MAX_COMMANDS;
    config.planning_window_ms = SPEECH_MOTOR_DEFAULT_PLANNING_WINDOW_MS;
    config.enable_coarticulation = true;
    config.coarticulation_strength = 0.7f;
    config.default_velocity = 5.0f;
    return config;
}

speech_motor_planner_t* speech_motor_create(const speech_motor_config_t* config) {
    // Use default config if none provided
    speech_motor_config_t default_config = speech_motor_default_config();
    if (!config) {
        config = &default_config;
    }

    // Validate configuration
    if (!speech_motor_validate_config(config)) {
        return NULL;
    }

    // Allocate planner structure
    speech_motor_planner_t* planner =
        (speech_motor_planner_t*)nimcp_calloc(1, sizeof(speech_motor_planner_t));
    if (!planner) {
        return NULL;
    }

    // Copy configuration
    planner->config = *config;

    // Allocate command queue
    planner->command_queue =
        (motor_command_t*)nimcp_calloc(config->max_commands, sizeof(motor_command_t));
    if (!planner->command_queue) {
        nimcp_free(planner);
        return NULL;
    }

    // Initialize articulators to neutral position
    for (int i = 0; i < SPEECH_MOTOR_NUM_ARTICULATORS; i++) {
        init_articulator(&planner->articulators[i]);
    }

    // Initialize queue state
    planner->queue_head = 0;
    planner->queue_tail = 0;
    planner->queue_count = 0;

    // Initialize timing - start with a small positive offset representing motor planning lead time
    // This reflects the ~200ms motor preparation window in biological speech production
    planner->current_time_ms = 0.0;
    planner->next_phoneme_time_ms = config->planning_window_ms / 2.0;

    // Initialize coarticulation state
    planner->last_phoneme = 0;
    planner->has_last_phoneme = false;

    // Initialize trajectory interpolation (disabled by default)
    planner->enable_interpolation = false;
    planner->interpolation_points = 0;

    // Clear statistics
    memset(&planner->stats, 0, sizeof(speech_motor_stats_t));

    return planner;
}

void speech_motor_destroy(speech_motor_planner_t* planner) {
    if (!planner) return;

    // Free command queue
    if (planner->command_queue) {
        nimcp_free(planner->command_queue);
        planner->command_queue = NULL;
    }

    // Free planner structure
    nimcp_free(planner);
}

bool speech_motor_plan_phoneme(speech_motor_planner_t* planner, uint8_t phoneme) {
    if (!planner) return false;

    // Get phoneme features
    phoneme_features_t features = get_phoneme_features(phoneme);

    // Generate motor commands
    if (!generate_commands_for_phoneme(planner, &features, phoneme)) {
        return false;
    }

    // Update coarticulation state
    planner->last_phoneme = phoneme;
    planner->has_last_phoneme = true;

    // Update statistics
    planner->stats.phonemes_planned++;

    return true;
}

bool speech_motor_get_commands(
    speech_motor_planner_t* planner,
    motor_command_t* commands,
    uint32_t* count
) {
    if (!planner || !commands || !count) return false;

    // Return false if queue is empty - no commands to retrieve
    if (planner->queue_count == 0) {
        *count = 0;
        return false;
    }

    uint32_t max_count = *count;
    uint32_t retrieved = 0;

    // Retrieve commands from queue
    while (retrieved < max_count && planner->queue_count > 0) {
        commands[retrieved] = planner->command_queue[planner->queue_head];
        planner->queue_head = (planner->queue_head + 1) % planner->config.max_commands;
        planner->queue_count--;
        retrieved++;
    }

    *count = retrieved;
    return true;
}

bool speech_motor_reset(speech_motor_planner_t* planner) {
    if (!planner) return false;

    // Clear command queue
    planner->queue_head = 0;
    planner->queue_tail = 0;
    planner->queue_count = 0;

    // Reset articulators to neutral
    for (int i = 0; i < SPEECH_MOTOR_NUM_ARTICULATORS; i++) {
        init_articulator(&planner->articulators[i]);
    }

    // Reset timing - restore motor planning lead time offset
    planner->current_time_ms = 0.0;
    planner->next_phoneme_time_ms = planner->config.planning_window_ms / 2.0;

    // Clear coarticulation state
    planner->last_phoneme = 0;
    planner->has_last_phoneme = false;

    // Initialize interpolation (disabled by default)
    planner->enable_interpolation = false;
    planner->interpolation_points = 0;

    return true;
}

bool speech_motor_plan_sequence(
    speech_motor_planner_t* planner,
    const uint8_t* phonemes,
    uint32_t num_phonemes
) {
    if (!planner || !phonemes || num_phonemes == 0) return false;

    // Plan each phoneme in sequence
    for (uint32_t i = 0; i < num_phonemes; i++) {
        if (!speech_motor_plan_phoneme(planner, phonemes[i])) {
            return false;  // Planning failed
        }
    }

    return true;
}

bool speech_motor_set_articulator(
    speech_motor_planner_t* planner,
    articulator_type_t articulator,
    float position
) {
    if (!planner) return false;
    if (articulator >= SPEECH_MOTOR_NUM_ARTICULATORS) return false;
    if (position < 0.0f || position > 1.0f) return false;

    planner->articulators[articulator].position = position;
    planner->articulators[articulator].last_update_time = planner->current_time_ms;

    return true;
}

bool speech_motor_get_articulator(
    const speech_motor_planner_t* planner,
    articulator_type_t articulator,
    float* position
) {
    if (!planner || !position) return false;
    if (articulator >= SPEECH_MOTOR_NUM_ARTICULATORS) return false;

    *position = planner->articulators[articulator].position;
    return true;
}

bool speech_motor_get_stats(
    const speech_motor_planner_t* planner,
    speech_motor_stats_t* stats
) {
    if (!planner || !stats) return false;

    *stats = planner->stats;
    stats->queue_size = planner->queue_count;

    return true;
}

const char* speech_motor_articulator_name(articulator_type_t articulator) {
    switch (articulator) {
        case ARTICULATOR_LIPS:   return "LIPS";
        case ARTICULATOR_TONGUE: return "TONGUE";
        case ARTICULATOR_JAW:    return "JAW";
        case ARTICULATOR_LARYNX: return "LARYNX";
        case ARTICULATOR_VELUM:  return "VELUM";
        default:                 return "UNKNOWN";
    }
}

bool speech_motor_validate_config(const speech_motor_config_t* config) {
    if (!config) return false;

    // Validate max_commands
    if (config->max_commands == 0 || config->max_commands > 10000) {
        return false;
    }

    // Validate planning_window_ms
    if (config->planning_window_ms < 0.0f || config->planning_window_ms > 1000.0f) {
        return false;
    }

    // Validate coarticulation_strength
    if (config->coarticulation_strength < 0.0f || config->coarticulation_strength > 1.0f) {
        return false;
    }

    // Validate default_velocity
    if (config->default_velocity < 0.0f || config->default_velocity > 100.0f) {
        return false;
    }

    return true;
}

//=============================================================================
// Trajectory Interpolation
//=============================================================================

float speech_motor_interpolate_position(
    float start_pos,
    float end_pos,
    float start_vel,
    float end_vel,
    float t
) {
    /*
     * WHAT: Cubic Hermite spline interpolation
     * WHY:  Smooth, C1-continuous trajectories between target positions
     * HOW:  H(t) = (2t³ - 3t² + 1)p0 + (t³ - 2t² + t)m0 + (-2t³ + 3t²)p1 + (t³ - t²)m1
     *
     * BIOLOGY: Motor cortex generates smooth movement trajectories
     * that minimize jerk (third derivative of position).
     * Hermite splines provide a good approximation.
     */

    /* Clamp t to [0, 1] */
    if (t < 0.0f) t = 0.0f;
    if (t > 1.0f) t = 1.0f;

    /* Hermite basis functions */
    float t2 = t * t;
    float t3 = t2 * t;

    float h00 = 2.0f * t3 - 3.0f * t2 + 1.0f;  /* Basis for p0 */
    float h10 = t3 - 2.0f * t2 + t;             /* Basis for m0 (tangent at p0) */
    float h01 = -2.0f * t3 + 3.0f * t2;         /* Basis for p1 */
    float h11 = t3 - t2;                         /* Basis for m1 (tangent at p1) */

    /* Interpolate position */
    float position = h00 * start_pos +
                     h10 * start_vel +
                     h01 * end_pos +
                     h11 * end_vel;

    /* Clamp to valid range [0, 1] */
    if (position < 0.0f) position = 0.0f;
    if (position > 1.0f) position = 1.0f;

    return position;
}

bool speech_motor_set_interpolation(
    speech_motor_planner_t* planner,
    bool enable,
    uint32_t num_interpolation_points
) {
    if (!planner) return false;

    /* Validate interpolation points (2-10 is reasonable) */
    if (enable && (num_interpolation_points < 2 || num_interpolation_points > 10)) {
        return false;
    }

    planner->enable_interpolation = enable;
    planner->interpolation_points = enable ? num_interpolation_points : 0;

    return true;
}

/**
 * @brief Calculate velocity based on distance and time
 *
 * WHAT: Compute articulator velocity for smooth movement
 * WHY:  Variable velocity based on movement distance feels more natural
 * HOW:  v = distance / time, with min/max clamping
 */
static float calculate_velocity(float distance, float duration_ms) {
    if (duration_ms <= 0.0f) return 0.0f;

    /* Convert to distance per ms, then scale */
    float velocity = fabsf(distance) / duration_ms * 1000.0f;

    /* Clamp to reasonable range (units per second) */
    if (velocity < 1.0f) velocity = 1.0f;
    if (velocity > 20.0f) velocity = 20.0f;

    return velocity;
}
