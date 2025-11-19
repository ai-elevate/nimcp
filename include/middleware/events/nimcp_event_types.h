//=============================================================================
// nimcp_event_types.h - Event Type Definitions for Middleware Event System
//=============================================================================

#ifndef NIMCP_EVENT_TYPES_H
#define NIMCP_EVENT_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file nimcp_event_types.h
 * @brief Event type definitions for NIMCP middleware event system
 *
 * WHAT: Type-safe event definitions for neural activity patterns
 * WHY:  Enable event-driven cognitive processing with clear semantics
 * HOW:  Tagged union pattern + metadata + lifecycle management
 *
 * DESIGN PATTERNS:
 * - Tagged Union: Type-safe polymorphism for event data
 * - Value Object: Immutable event semantics (copy on creation)
 * - Factory Pattern: Type-specific constructors
 *
 * PERFORMANCE:
 * - Event creation: O(1) - stack allocation preferred
 * - Event copying: O(1) - small fixed size
 * - No heap allocation for basic events
 */

//=============================================================================
// Event Type Enumeration
//=============================================================================

/**
 * @brief Event type enumeration
 *
 * WHAT: All event types the system can generate
 * WHY:  Type-safe dispatch and filtering
 * HOW:  Enum with semantic names matching cognitive events
 */
typedef enum {
    EVENT_TYPE_NONE = 0,

    // Neural activity events
    EVENT_TYPE_SPIKE_BURST,        /**< Multiple neurons firing together */
    EVENT_TYPE_PATTERN_DETECTED,   /**< Sequence/synchrony found */
    EVENT_TYPE_OSCILLATION_CHANGE, /**< Frequency shift detected */

    // Cognitive events
    EVENT_TYPE_ATTENTION_SHIFT,    /**< Attention moved to new focus */
    EVENT_TYPE_MEMORY_FORMED,      /**< Memory consolidation occurred */
    EVENT_TYPE_DECISION_MADE,      /**< Brain chose an action */
    EVENT_TYPE_ERROR_DETECTED,     /**< Prediction error occurred */

    // Salience events
    EVENT_TYPE_SALIENCE_PEAK,      /**< Important event detected */
    EVENT_TYPE_NOVELTY_DETECTED,   /**< Novel stimulus encountered */
    EVENT_TYPE_SURPRISE_DETECTED,  /**< Unexpected event occurred */

    // System events
    EVENT_TYPE_THRESHOLD_CROSSED,  /**< Value exceeded threshold */
    EVENT_TYPE_STATE_CHANGED,      /**< System state transition */
    EVENT_TYPE_CUSTOM,             /**< User-defined event */

    EVENT_TYPE_COUNT               /**< Total event types */
} event_type_t;

/**
 * @brief Event priority levels
 *
 * WHAT: Priority for event processing order
 * WHY:  Critical events processed before routine ones
 * HOW:  Lower number = higher priority (min-heap friendly)
 */
typedef enum {
    EVENT_PRIORITY_CRITICAL = 0,   /**< Process immediately */
    EVENT_PRIORITY_HIGH = 1,       /**< Process soon */
    EVENT_PRIORITY_NORMAL = 2,     /**< Standard processing */
    EVENT_PRIORITY_LOW = 3,        /**< Process when idle */
    EVENT_PRIORITY_BACKGROUND = 4  /**< Process opportunistically */
} event_priority_t;

/**
 * @brief Event source identifier
 *
 * WHAT: Which component generated the event
 * WHY:  Enable source-based filtering and routing
 * HOW:  Enum of all event-producing components
 */
typedef enum {
    EVENT_SOURCE_UNKNOWN = 0,
    EVENT_SOURCE_ENCODER,
    EVENT_SOURCE_FEATURE_EXTRACTOR,
    EVENT_SOURCE_PATTERN_DETECTOR,
    EVENT_SOURCE_ROUTER,
    EVENT_SOURCE_NORMALIZER,
    EVENT_SOURCE_BUFFER,
    EVENT_SOURCE_BRAIN,
    EVENT_SOURCE_ETHICS,
    EVENT_SOURCE_SALIENCE,
    EVENT_SOURCE_WORKING_MEMORY,
    EVENT_SOURCE_PREDICTIVE,
    EVENT_SOURCE_CUSTOM
} event_source_t;

//=============================================================================
// Event-Specific Data Structures
//=============================================================================

/**
 * @brief Spike burst event data
 *
 * WHAT: Neurons firing together in synchrony
 * WHY:  Indicates coordinated neural activity
 */
typedef struct {
    uint32_t* neuron_ids;      /**< IDs of firing neurons */
    uint32_t num_neurons;      /**< Count of neurons */
    float synchrony_score;     /**< Synchronization measure [0,1] */
    uint64_t burst_duration_us; /**< Duration in microseconds */
} spike_burst_data_t;

/**
 * @brief Pattern detected event data
 *
 * WHAT: Recognized sequence or synchrony pattern
 * WHY:  Indicates learned pattern activation
 */
typedef struct {
    uint32_t pattern_id;       /**< Pattern identifier */
    float match_confidence;    /**< Confidence in match [0,1] */
    uint32_t pattern_length;   /**< Pattern length (for sequences) */
    const char* pattern_name;  /**< Human-readable pattern name */
} pattern_detected_data_t;

/**
 * @brief Attention shift event data
 *
 * WHAT: Attention moved from one item to another
 * WHY:  Track focus of cognitive processing
 */
typedef struct {
    uint32_t previous_item;    /**< Previously attended item */
    uint32_t current_item;     /**< Currently attended item */
    float attention_strength;  /**< Attention intensity [0,1] */
    const char* shift_reason;  /**< Why attention shifted */
} attention_shift_data_t;

/**
 * @brief Memory formed event data
 *
 * WHAT: Memory consolidation completed
 * WHY:  Track learning and memory formation
 */
typedef struct {
    uint32_t memory_id;        /**< Unique memory identifier */
    float* memory_trace;       /**< Memory representation */
    uint32_t trace_size;       /**< Size of trace */
    float consolidation_strength; /**< How well consolidated [0,1] */
} memory_formed_data_t;

/**
 * @brief Salience peak event data
 *
 * WHAT: Highly salient stimulus detected
 * WHY:  Requires immediate attention allocation
 */
typedef struct {
    float salience_score;      /**< Overall salience [0,1] */
    float novelty_score;       /**< Novelty component [0,1] */
    float surprise_score;      /**< Surprise component [0,1] */
    float urgency_score;       /**< Urgency component [0,1] */
} salience_peak_data_t;

/**
 * @brief Oscillation change event data
 *
 * WHAT: Brain oscillation frequency shifted
 * WHY:  Indicates state transition (e.g., alert->drowsy)
 */
typedef struct {
    float previous_freq_hz;    /**< Previous frequency */
    float current_freq_hz;     /**< Current frequency */
    float power_change;        /**< Power change ratio */
    const char* band_name;     /**< Band name (alpha, beta, etc.) */
} oscillation_change_data_t;

/**
 * @brief Error detected event data
 *
 * WHAT: Prediction error or mismatch
 * WHY:  Drive learning and model updates
 */
typedef struct {
    float expected_value;      /**< What was predicted */
    float actual_value;        /**< What actually happened */
    float error_magnitude;     /**< Absolute error */
    uint32_t error_location;   /**< Where error occurred */
} error_detected_data_t;

/**
 * @brief Decision made event data
 *
 * WHAT: Brain selected an action
 * WHY:  Track decision-making for learning
 */
typedef struct {
    uint32_t decision_id;      /**< Decision identifier */
    float confidence;          /**< Decision confidence [0,1] */
    float* decision_vector;    /**< Decision representation */
    uint32_t vector_size;      /**< Size of decision vector */
} decision_made_data_t;

/**
 * @brief Generic custom event data
 *
 * WHAT: User-defined event payload
 * WHY:  Extensibility for application-specific events
 */
typedef struct {
    void* data;                /**< Custom data pointer */
    uint32_t data_size;        /**< Size of custom data */
    const char* description;   /**< Human-readable description */
} custom_event_data_t;

//=============================================================================
// Main Event Structure (Tagged Union)
//=============================================================================

/**
 * @brief Event structure (tagged union)
 *
 * WHAT: Complete event with metadata and type-specific data
 * WHY:  Type-safe, efficient event representation
 * HOW:  Tagged union with common metadata
 *
 * MEMORY: ~128 bytes per event (fits in 2 cache lines)
 */
typedef struct {
    // Common metadata (always present)
    event_type_t type;         /**< Event type tag */
    event_priority_t priority; /**< Processing priority */
    event_source_t source;     /**< Which component generated this */
    uint64_t timestamp_us;     /**< When event occurred (microseconds) */
    uint64_t sequence_number;  /**< Global sequence number */

    // Type-specific data (tagged union)
    union {
        spike_burst_data_t spike_burst;
        pattern_detected_data_t pattern_detected;
        attention_shift_data_t attention_shift;
        memory_formed_data_t memory_formed;
        salience_peak_data_t salience_peak;
        oscillation_change_data_t oscillation_change;
        error_detected_data_t error_detected;
        decision_made_data_t decision_made;
        custom_event_data_t custom;
    } data;

} event_t;

//=============================================================================
// Event Creation Functions (Factory Pattern)
//=============================================================================

/**
 * @brief Create spike burst event
 *
 * COMPLEXITY: O(1)
 * THREAD-SAFE: Yes
 */
event_t event_create_spike_burst(uint32_t* neuron_ids, uint32_t num_neurons,
                                  float synchrony, uint64_t duration_us,
                                  event_priority_t priority, event_source_t source);

/**
 * @brief Create pattern detected event
 */
event_t event_create_pattern_detected(uint32_t pattern_id, float confidence,
                                       uint32_t pattern_length, const char* pattern_name,
                                       event_priority_t priority, event_source_t source);

/**
 * @brief Create attention shift event
 */
event_t event_create_attention_shift(uint32_t prev_item, uint32_t curr_item,
                                      float attention_strength, const char* reason,
                                      event_priority_t priority, event_source_t source);

/**
 * @brief Create memory formed event
 */
event_t event_create_memory_formed(uint32_t memory_id, float* memory_trace,
                                    uint32_t trace_size, float consolidation,
                                    event_priority_t priority, event_source_t source);

/**
 * @brief Create salience peak event
 */
event_t event_create_salience_peak(float salience, float novelty,
                                    float surprise, float urgency,
                                    event_priority_t priority, event_source_t source);

/**
 * @brief Create oscillation change event
 */
event_t event_create_oscillation_change(float prev_freq, float curr_freq,
                                         float power_change, const char* band_name,
                                         event_priority_t priority, event_source_t source);

/**
 * @brief Create error detected event
 */
event_t event_create_error_detected(float expected, float actual,
                                     float magnitude, uint32_t location,
                                     event_priority_t priority, event_source_t source);

/**
 * @brief Create decision made event
 */
event_t event_create_decision_made(uint32_t decision_id, float confidence,
                                    float* decision_vector, uint32_t vector_size,
                                    event_priority_t priority, event_source_t source);

/**
 * @brief Create custom event
 */
event_t event_create_custom(void* data, uint32_t data_size, const char* description,
                             event_priority_t priority, event_source_t source);

//=============================================================================
// Event Utility Functions
//=============================================================================

/**
 * @brief Get event type name
 *
 * WHAT: Human-readable name for event type
 * WHY:  Debugging and logging
 * COMPLEXITY: O(1)
 */
const char* event_type_name(event_type_t type);

/**
 * @brief Get event source name
 */
const char* event_source_name(event_source_t source);

/**
 * @brief Get event priority name
 */
const char* event_priority_name(event_priority_t priority);

/**
 * @brief Copy event (deep copy if needed)
 *
 * WHAT: Create independent copy of event
 * WHY:  Thread-safe event passing
 * HOW:  Copies all data, allocates new pointers if needed
 *
 * COMPLEXITY: O(n) where n = size of embedded data
 * THREAD-SAFE: Yes
 *
 * @param dest Destination event
 * @param src Source event
 * @return true on success
 */
bool event_copy(event_t* dest, const event_t* src);

/**
 * @brief Free event resources
 *
 * WHAT: Free any dynamically allocated event data
 * WHY:  Prevent memory leaks
 * COMPLEXITY: O(1)
 *
 * @param event Event to free
 */
void event_free(event_t* event);

/**
 * @brief Print event (for debugging)
 *
 * WHAT: Human-readable event description
 * WHY:  Debugging and logging
 *
 * @param event Event to print
 */
void event_print(const event_t* event);

#ifdef __cplusplus
}
#endif

#endif // NIMCP_EVENT_TYPES_H
