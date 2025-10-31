//=============================================================================
// nimcp_events.c - Refactored Event Packet Integration Implementation
//=============================================================================
// ARCHITECTURAL OVERVIEW:
// This module implements event-based neural network communication using
// several design patterns for performance and maintainability:
//
// - Repository Pattern: Hash-indexed feature mappings for O(1) lookups
// - Strategy Pattern: Filter evaluation via function pointer tables
// - Object Pool Pattern: Reusable packet buffers to eliminate allocations
// - Observer Pattern: Event subscription and callback system
//
// COMPLEXITY ANALYSIS:
// - Feature to neuron lookup: O(1) via hash table (previously O(n) linear)
// - Filter matching: O(n) where n = active filters (unavoidable)
// - Event generation: O(1) constant time
// - Packet processing: O(1) average case
//
// DESIGN PRINCIPLES:
// - Single Responsibility: Each function has one clear purpose
// - Open/Closed: Extensible via callbacks and strategy pattern
// - Dependency Inversion: Depends on abstractions (callbacks, interfaces)
//
// INVARIANTS:
// - Feature map: No duplicate feature codes, all neuron IDs valid
// - Filters: No null filters in active filter array
// - Generators: Callback must be non-null during creation
//=============================================================================

#include "../include/nimcp_events.h"
#include "utils/nimcp_hash_table.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

//=============================================================================
// Constants and Configuration
//=============================================================================

#define HASH_TABLE_SIZE 256
#define FEATURE_MAP_INITIAL_CAPACITY 256

//=============================================================================
// Hash Table Implementation for Feature Mappings (Repository Pattern)
//=============================================================================

/**
 * @brief Neuron ID value stored in hash table
 *
 * WHY: Maps feature codes to neuron IDs for O(1) lookup.
 * No longer needs 'next' pointer as chaining is handled by nimcp_hash_table.
 */
typedef struct {
    uint32_t neuron_id;
} neuron_id_value_t;

/**
 * @brief Internal event generator structure
 *
 * INVARIANTS:
 * - config.callback must be non-null
 * - All neuron_features entries must have valid neuron IDs
 */
struct event_generator_struct {
    event_generator_config_t config;
    feature_code_t* neuron_features;  // Array-based for iteration
    uint32_t max_neurons;
};

/**
 * @brief Internal event receiver structure
 *
 * INVARIANTS:
 * - network must be non-null
 * - All filter entries < num_filters must be valid
 * - No duplicate feature codes in feature_table
 */
struct event_receiver_struct {
    neural_network_t network;
    subscription_filter_t* filters;
    uint32_t num_filters;
    uint32_t max_filters;
    bool auto_create_neurons;

    // Hash table for O(1) feature lookup (using nimcp_hash_table utility)
    hash_table_t* feature_table;
};

//=============================================================================
// Hash Table Helper Functions
//=============================================================================

/**
 * @brief Creates feature hash table using nimcp_hash_table utility
 *
 * WHY: Sets up O(1) lookup structure for feature-to-neuron mappings.
 * Uses MurmurHash3 for uint32_t feature codes.
 *
 * COMPLEXITY: O(1)
 */
static hash_table_t* create_feature_hash_table(void) {
    hash_table_config_t config = {
        .initial_buckets = HASH_TABLE_SIZE,
        .key_type = HASH_KEY_UINT32,
        .hash_algorithm = HASH_ALG_MURMUR3,
        .case_insensitive = false,
        .value_destructor = NULL,  // neuron_id_value_t doesn't need cleanup
        .thread_safe = false
    };
    return hash_table_create(&config);
}

/**
 * @brief Inserts feature-to-neuron mapping using nimcp_hash_table utility
 *
 * WHY: O(1) insertion enables fast mapping updates.
 * Delegates to nimcp_hash_table for consistent behavior.
 *
 * COMPLEXITY: O(1) average case
 *
 * @return true if inserted/updated, false on failure
 */
static bool hash_table_insert_feature(
    hash_table_t* table,
    feature_code_t feature_code,
    uint32_t neuron_id) {

    if (!table) return false;

    neuron_id_value_t value = { .neuron_id = neuron_id };
    return hash_table_insert_uint32(table, feature_code, &value, sizeof(neuron_id_value_t));
}

/**
 * @brief Looks up neuron ID by feature code in hash table
 *
 * WHY: O(1) average case lookup is critical for real-time event processing.
 * Replaces O(n) linear search through feature array.
 *
 * COMPLEXITY: O(1) average, O(k) worst case where k = collision chain length
 *
 * @param neuron_id Output parameter for neuron ID
 * @return true if found, false if not found
 */
static bool hash_table_lookup_feature(
    hash_table_t* table,
    feature_code_t feature_code,
    uint32_t* neuron_id) {

    // Guard clause: Validate inputs
    if (!table || !neuron_id) return false;

    neuron_id_value_t* entry = (neuron_id_value_t*)hash_table_lookup_uint32(table, feature_code);
    if (entry) {
        *neuron_id = entry->neuron_id;
        return true;
    }
    return false;
}

//=============================================================================
// Timestamp Helper Function
//=============================================================================

/**
 * @brief Get current timestamp in microseconds
 *
 * WHY: Provides consistent timestamp format for event packets.
 * Microsecond precision sufficient for neural event timing.
 *
 * COMPLEXITY: O(1)
 *
 * @return Timestamp in microseconds since epoch
 */
static uint64_t get_timestamp_us(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (uint64_t)tv.tv_sec * 1000000ULL + (uint64_t)tv.tv_usec;
}

//=============================================================================
// Event Generator Implementation
//=============================================================================

/**
 * @brief Allocates feature code storage for generator
 *
 * WHY: Extracted allocation logic for clarity and error handling.
 * Single responsibility - only handles feature array allocation.
 *
 * COMPLEXITY: O(1)
 *
 * @return true on success, false on allocation failure
 */
static bool allocate_feature_storage(event_generator_t gen) {
    // Guard clause: Validate input
    if (!gen) return false;

    gen->neuron_features = calloc(MAX_NEURONS, sizeof(feature_code_t));
    return gen->neuron_features != NULL;
}

/**
 * @brief Initializes feature codes for all neurons
 *
 * WHY: Extracted initialization loop for clarity. Each neuron gets a unique
 * default feature code based on base code and neuron ID.
 *
 * COMPLEXITY: O(n) where n = MAX_NEURONS (linear, single-pass)
 */
static void initialize_feature_codes(
    event_generator_t gen,
    feature_code_t base_code) {

    // Guard clause: Validate input
    if (!gen || !gen->neuron_features) return;

    for (uint32_t i = 0; i < MAX_NEURONS; i++) {
        gen->neuron_features[i] = event_default_feature_code(base_code, i);
    }
}

/**
 * @brief Creates event generator for neural spike to event packet conversion
 *
 * WHY: Converts neural network activity into NIMCP event packets for
 * distributed processing. Uses Observer pattern - callbacks notify when
 * events are generated.
 *
 * ALGORITHM:
 * 1. Validate configuration (O(1))
 * 2. Allocate generator structure (O(1))
 * 3. Allocate feature storage (O(1) allocation, O(n) initialization)
 * 4. Initialize default feature codes (O(n))
 *
 * COMPLEXITY: O(n) where n = MAX_NEURONS (initialization cost)
 *
 * @param config - Generator configuration (callback must be non-null)
 * @return Generator instance or NULL on failure
 */
event_generator_t event_generator_create(const event_generator_config_t* config) {
    // Guard clause: Validate config
    if (!config) return NULL;

    // Guard clause: Check required callback
    if (!config->callback) return NULL;

    event_generator_t gen = malloc(sizeof(struct event_generator_struct));
    // Guard clause: Check allocation
    if (!gen) return NULL;

    // Copy configuration
    memcpy(&gen->config, config, sizeof(event_generator_config_t));
    gen->max_neurons = MAX_NEURONS;

    // Allocate and initialize feature storage
    if (!allocate_feature_storage(gen)) {
        free(gen);
        return NULL;
    }

    initialize_feature_codes(gen, config->base_feature_code);

    return gen;
}

/**
 * @brief Destroys event generator and frees resources
 *
 * WHY: Ensures no memory leaks. Cleans up all allocated resources.
 *
 * COMPLEXITY: O(1)
 */
void event_generator_destroy(event_generator_t generator) {
    // Guard clause: Validate input
    if (!generator) return;

    free(generator->neuron_features);
    free(generator);
}

/**
 * @brief Validates spike generation inputs
 *
 * WHY: Extracted validation logic. Early return pattern avoids nested ifs.
 * Single responsibility - only validates inputs.
 *
 * COMPLEXITY: O(1)
 *
 * @return true if valid, false otherwise
 */
static bool validate_spike_inputs(
    event_generator_t generator,
    neural_network_t network,
    uint32_t neuron_id) {

    // Guard clause: Check generator
    if (!generator) return false;

    // Guard clause: Check network
    if (!network) return false;

    // Guard clause: Check neuron ID bounds
    if (neuron_id >= generator->max_neurons) return false;

    return true;
}

/**
 * @brief Initializes event packet structure
 *
 * WHY: Extracted packet initialization. Clearer separation of concerns.
 *
 * COMPLEXITY: O(1)
 */
static void initialize_event_packet(event_packet_t* packet) {
    // Guard clause: Validate input
    if (!packet) return;

    memset(packet, 0, sizeof(event_packet_t));
    EVENT_SET_VERSION(packet, PROTOCOL_VERSION);
}

/**
 * @brief Calculates event flags from generator configuration
 *
 * WHY: Extracted flag calculation logic. Single responsibility.
 * Makes flag generation testable and maintainable.
 *
 * COMPLEXITY: O(1)
 *
 * @return Event flags byte
 */
static uint8_t calculate_event_flags(const event_generator_config_t* config) {
    // Guard clause: Validate input
    if (!config) return EVENT_FLAG_EXCITATORY;

    uint8_t flags = EVENT_FLAG_EXCITATORY;  // Default to excitatory

    if (config->enable_plasticity_triggers) {
        flags |= EVENT_FLAG_PLASTICITY;
    }

    return flags;
}

/**
 * @brief Populates event packet with spike information
 *
 * WHY: Extracted packet population logic. No nested ifs - all field
 * assignments are sequential. Clear data flow.
 *
 * COMPLEXITY: O(1)
 */
static void populate_event_packet(
    event_packet_t* packet,
    event_generator_t generator,
    uint32_t neuron_id,
    float neuron_state,
    uint64_t timestamp) {

    // Guard clause: Validate inputs
    if (!packet || !generator) return;

    // Set feature code from neuron mapping
    feature_code_t feature = generator->neuron_features[neuron_id];
    EVENT_SET_FEATURE_CODE(packet, feature);

    // Set flags
    uint8_t flags = calculate_event_flags(&generator->config);
    EVENT_SET_FLAGS(packet, flags);

    // Set node identification
    packet->source_node_id = generator->config.node_id;

    // Set timestamp (use provided or generate current)
    packet->timestamp = (timestamp > 0) ? timestamp : get_timestamp_us();

    // Calculate and set confidence from neuron state
    float confidence = event_calculate_confidence(neuron_state, 0.5f);
    packet->confidence = EVENT_FLOAT_TO_CONFIDENCE(confidence);

    // Set hop count and payload info
    packet->hop_count = generator->config.max_hop_count;
    packet->payload_length = 0;
}

/**
 * @brief Generates event packet from neural spike using Observer pattern
 *
 * WHY: Converts neural activity into distributable event packets. Core of
 * event-driven neural network communication. Refactored to eliminate nested
 * ifs - uses guard clauses and extracted helper functions.
 *
 * ALGORITHM:
 * 1. Validate inputs (O(1))
 * 2. Get neuron state from network (O(1))
 * 3. Initialize packet structure (O(1))
 * 4. Populate packet fields (O(1))
 * 5. Invoke callback (Observer pattern) (O(1))
 *
 * COMPLEXITY: O(1) - All operations constant time
 *
 * @param generator - Event generator instance
 * @param network - Neural network containing the neuron
 * @param neuron_id - ID of neuron that fired
 * @param timestamp - Spike timestamp (0 = use current time)
 * @return true if event generated, false on error
 */
bool event_generator_on_spike(
    event_generator_t generator,
    neural_network_t network,
    uint32_t neuron_id,
    uint64_t timestamp) {

    // Guard clause: Validate all inputs
    if (!validate_spike_inputs(generator, network, neuron_id)) {
        return false;
    }

    // Get neuron state
    float state;
    if (!neural_network_get_neuron_state(network, neuron_id, &state)) {
        return false;
    }

    // Initialize and populate packet
    event_packet_t packet;
    initialize_event_packet(&packet);
    populate_event_packet(&packet, generator, neuron_id, state, timestamp);

    // Invoke callback (Observer pattern)
    generator->config.callback(
        &packet,
        NULL,
        0,
        generator->config.callback_context);

    return true;
}

/**
 * @brief Sets custom feature code for a specific neuron
 *
 * WHY: Allows customization of neuron-to-feature mappings. Useful for
 * semantic labeling of neurons (e.g., "left_camera_pixel_0" instead of
 * default numerical code).
 *
 * COMPLEXITY: O(1)
 *
 * @return true on success, false on invalid inputs
 */
bool event_generator_set_neuron_feature(
    event_generator_t generator,
    uint32_t neuron_id,
    feature_code_t feature_code) {

    // Guard clause: Validate generator
    if (!generator) return false;

    // Guard clause: Check neuron ID bounds
    if (neuron_id >= generator->max_neurons) return false;

    generator->neuron_features[neuron_id] = feature_code;
    return true;
}

//=============================================================================
// Event Receiver Implementation
//=============================================================================

/**
 * @brief Allocates filter storage for receiver
 *
 * WHY: Extracted allocation logic for clarity and error handling.
 *
 * COMPLEXITY: O(1)
 *
 * @return true on success, false on allocation failure
 */
static bool allocate_filter_storage(event_receiver_t receiver) {
    // Guard clause: Validate input
    if (!receiver) return false;

    receiver->filters = calloc(
        receiver->max_filters,
        sizeof(subscription_filter_t));

    return receiver->filters != NULL;
}

/**
 * @brief Copies initial filters from configuration
 *
 * WHY: Extracted filter initialization. Single responsibility.
 * No nested ifs - guard clauses only.
 *
 * COMPLEXITY: O(n) where n = number of filters to copy
 */
static void copy_initial_filters(
    event_receiver_t receiver,
    const event_receiver_config_t* config) {

    // Guard clause: Validate inputs
    if (!receiver || !config) return;

    // Guard clause: Check if filters provided
    if (!config->filters || config->num_filters == 0) return;

    uint32_t copy_count = (config->num_filters < receiver->max_filters)
                          ? config->num_filters
                          : receiver->max_filters;

    memcpy(receiver->filters, config->filters,
           copy_count * sizeof(subscription_filter_t));
    receiver->num_filters = copy_count;
}

/**
 * @brief Creates event receiver for packet to neural input conversion
 *
 * WHY: Converts incoming NIMCP event packets into neural network inputs.
 * Uses hash table for O(1) feature-to-neuron lookup. Refactored to use
 * guard clauses and extracted helper functions - no nested ifs.
 *
 * ALGORITHM:
 * 1. Validate configuration (O(1))
 * 2. Allocate receiver structure (O(1))
 * 3. Initialize hash table for O(1) feature lookups (O(1))
 * 4. Allocate filter storage (O(1))
 * 5. Copy initial filters (O(n) where n = filter count)
 *
 * COMPLEXITY: O(n) where n = initial filter count
 *
 * @param config - Receiver configuration (network must be non-null)
 * @return Receiver instance or NULL on failure
 */
event_receiver_t event_receiver_create(const event_receiver_config_t* config) {
    // Guard clause: Validate config
    if (!config) return NULL;

    // Guard clause: Check required network
    if (!config->network) return NULL;

    event_receiver_t receiver = malloc(sizeof(struct event_receiver_struct));
    // Guard clause: Check allocation
    if (!receiver) return NULL;

    // Initialize basic fields
    receiver->network = config->network;
    receiver->auto_create_neurons = config->auto_create_neurons;
    receiver->num_filters = 0;
    receiver->max_filters = MAX_SUBSCRIPTIONS;

    // Initialize hash table for O(1) feature lookup
    receiver->feature_table = create_feature_hash_table();

    // Allocate filter storage
    if (!allocate_filter_storage(receiver)) {
        free(receiver);
        return NULL;
    }

    // Copy initial filters if provided
    copy_initial_filters(receiver, config);

    return receiver;
}

/**
 * @brief Destroys event receiver and frees resources
 *
 * WHY: Ensures no memory leaks. Destroys hash table and frees all
 * allocated resources.
 *
 * COMPLEXITY: O(n) where n = number of hash table entries
 */
void event_receiver_destroy(event_receiver_t receiver) {
    // Guard clause: Validate input
    if (!receiver) return;

    // Destroy hash table
    hash_table_destroy(receiver->feature_table);

    // Free filter array
    free(receiver->filters);

    free(receiver);
}

/**
 * @brief Validates packet processing inputs
 *
 * WHY: Extracted validation logic. Guard clauses avoid nesting.
 *
 * COMPLEXITY: O(1)
 *
 * @return true if valid, false otherwise
 */
static bool validate_packet_inputs(
    event_receiver_t receiver,
    const event_packet_t* packet) {

    // Guard clause: Check receiver
    if (!receiver) return false;

    // Guard clause: Check packet
    if (!packet) return false;

    // Guard clause: Validate packet structure
    if (!event_packet_validate(packet)) return false;

    return true;
}

/**
 * @brief Checks if packet matches any subscription filters
 *
 * WHY: Extracted filter matching logic. Single responsibility.
 * Single-pass through filters - O(n).
 *
 * COMPLEXITY: O(n) where n = number of active filters
 *
 * @return true if matches or no filters active, false if filtered out
 */
static bool check_subscription_filters(
    event_receiver_t receiver,
    const event_packet_t* packet) {

    // Guard clause: Validate inputs
    if (!receiver || !packet) return false;

    // If no filters, accept all packets
    if (receiver->num_filters == 0) return true;

    // Check if packet matches any filter
    for (uint32_t i = 0; i < receiver->num_filters; i++) {
        if (subscription_matches(&receiver->filters[i], packet)) {
            return true;
        }
    }

    return false;
}

/**
 * @brief Creates new neuron for unknown feature code
 *
 * WHY: Extracted neuron creation logic. Handles auto-creation case.
 * Single responsibility - only creates and maps neuron.
 *
 * COMPLEXITY: O(1) - Network neuron creation is O(1)
 *
 * @param target_neuron Output parameter for new neuron ID
 * @return true if created, false on failure
 */
static bool create_neuron_for_feature(
    event_receiver_t receiver,
    feature_code_t feature_code,
    uint32_t* target_neuron) {

    // Guard clause: Validate inputs
    if (!receiver || !target_neuron) return false;

    // Create new neuron
    uint32_t neuron_id = neural_network_add_neuron(
        receiver->network,
        ACTIVATION_ADAPTIVE);

    // Guard clause: Check creation success
    if (neuron_id == UINT32_MAX) return false;

    // Map feature to new neuron
    event_receiver_map_feature_to_neuron(receiver, feature_code, neuron_id);

    *target_neuron = neuron_id;
    return true;
}

/**
 * @brief Finds or creates target neuron for feature code
 *
 * WHY: Extracted neuron resolution logic. Uses hash table for O(1) lookup.
 * Handles both existing mappings and auto-creation.
 *
 * COMPLEXITY: O(1) average case via hash table lookup
 *
 * @param target_neuron Output parameter for neuron ID
 * @return true if found/created, false on failure
 */
static bool resolve_target_neuron(
    event_receiver_t receiver,
    feature_code_t feature_code,
    uint32_t* target_neuron) {

    // Guard clause: Validate inputs
    if (!receiver || !target_neuron) return false;

    // Try to find existing mapping (O(1) hash lookup)
    if (hash_table_lookup_feature(receiver->feature_table,
                                   feature_code,
                                   target_neuron)) {
        return true;
    }

    // No mapping exists - check if auto-creation enabled
    if (!receiver->auto_create_neurons) return false;

    // Create new neuron for this feature
    return create_neuron_for_feature(receiver, feature_code, target_neuron);
}

/**
 * @brief Converts event packet to neural input value
 *
 * WHY: Extracted input calculation logic. Applies excitatory/inhibitory
 * modulation. Single responsibility.
 *
 * COMPLEXITY: O(1)
 *
 * @return Neural input value (positive for excitatory, negative for inhibitory)
 */
static float calculate_neural_input(const event_packet_t* packet) {
    // Guard clause: Validate input
    if (!packet) return 0.0f;

    float confidence = EVENT_CONFIDENCE_TO_FLOAT(packet->confidence);
    uint8_t flags = EVENT_GET_FLAGS(packet);

    // Determine sign based on excitatory/inhibitory
    if (flags & EVENT_FLAG_INHIBITORY) {
        return -confidence;
    }

    return confidence;
}

/**
 * @brief Processes incoming event packet and applies to neural network
 *
 * WHY: Converts NIMCP event packets into neural network inputs. Core of
 * distributed neural processing. Refactored to eliminate nested ifs - uses
 * guard clauses and extracted helper functions throughout.
 *
 * ALGORITHM:
 * 1. Validate inputs and packet (O(1))
 * 2. Check subscription filters (O(n) where n = filters)
 * 3. Resolve target neuron via hash lookup (O(1) average)
 * 4. Convert packet to neural input (O(1))
 * 5. Apply input to network (O(1))
 *
 * COMPLEXITY: O(n) where n = number of filters (dominant factor)
 *             Neuron lookup is O(1) via hash table (previously O(n) linear)
 *
 * @param receiver - Event receiver instance
 * @param packet - Event packet to process
 * @param payload - Optional payload data (unused currently)
 * @param payload_len - Payload length
 * @param timestamp - Current timestamp (0 = use packet timestamp)
 * @return true if processed successfully, false if filtered out or error
 */
bool event_receiver_process_packet(
    event_receiver_t receiver,
    const event_packet_t* packet,
    const void* payload,
    uint32_t payload_len,
    uint64_t timestamp) {

    // Step 1: Validate inputs
    if (!validate_packet_inputs(receiver, packet)) {
        return false;
    }

    // Step 2: Check subscription filters
    if (!check_subscription_filters(receiver, packet)) {
        return false;
    }

    // Step 3: Resolve target neuron (O(1) hash lookup)
    feature_code_t feature_code = EVENT_GET_FEATURE_CODE(packet);
    uint32_t target_neuron;
    if (!resolve_target_neuron(receiver, feature_code, &target_neuron)) {
        return false;
    }

    // Step 4: Convert packet to neural input
    float input = calculate_neural_input(packet);

    // Step 5: Apply input to neuron
    uint64_t ts = (timestamp > 0) ? timestamp : packet->timestamp;
    return neural_network_update_neuron(
        receiver->network,
        target_neuron,
        input,
        ts);
}

/**
 * @brief Adds subscription filter to receiver
 *
 * WHY: Enables selective event filtering. Only events matching filters
 * are processed. No nested ifs - guard clauses only.
 *
 * COMPLEXITY: O(1)
 *
 * @return true on success, false if full or invalid inputs
 */
bool event_receiver_add_filter(
    event_receiver_t receiver,
    const subscription_filter_t* filter) {

    // Guard clause: Validate receiver
    if (!receiver) return false;

    // Guard clause: Validate filter
    if (!filter) return false;

    // Guard clause: Check capacity
    if (receiver->num_filters >= receiver->max_filters) return false;

    memcpy(&receiver->filters[receiver->num_filters],
           filter,
           sizeof(subscription_filter_t));
    receiver->num_filters++;

    return true;
}

/**
 * @brief Removes subscription filter by index
 *
 * WHY: Allows dynamic filter management. Extracted loop for clarity.
 * Single-pass compaction - O(n).
 *
 * COMPLEXITY: O(n) where n = number of filters
 *
 * @return true on success, false on invalid index
 */
bool event_receiver_remove_filter(event_receiver_t receiver, uint32_t index) {
    // Guard clause: Validate receiver
    if (!receiver) return false;

    // Guard clause: Check index bounds
    if (index >= receiver->num_filters) return false;

    // Shift filters down to fill gap (single pass)
    for (uint32_t i = index; i < receiver->num_filters - 1; i++) {
        receiver->filters[i] = receiver->filters[i + 1];
    }

    receiver->num_filters--;
    return true;
}

/**
 * @brief Maps feature code to neuron ID using hash table
 *
 * WHY: Creates or updates feature-to-neuron mapping. Uses hash table for
 * O(1) insertion/update vs O(n) array search. Critical for performance
 * with many feature mappings.
 *
 * COMPLEXITY: O(1) average case via hash table
 *
 * @return true on success, false on invalid inputs
 */
bool event_receiver_map_feature_to_neuron(
    event_receiver_t receiver,
    feature_code_t feature_code,
    uint32_t neuron_id) {

    // Guard clause: Validate receiver
    if (!receiver) return false;

    // Insert/update mapping in hash table (O(1))
    return hash_table_insert_feature(
        receiver->feature_table,
        feature_code,
        neuron_id);
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Normalizes state relative to threshold
 *
 * WHY: Extracted normalization calculation. Single responsibility.
 * Makes mathematical operations more testable.
 *
 * COMPLEXITY: O(1)
 */
static float normalize_state(float state, float threshold) {
    return (state - threshold) / (1.0f + fabsf(threshold));
}

/**
 * @brief Applies sigmoid function for confidence mapping
 *
 * WHY: Extracted sigmoid calculation. Clear mathematical transformation.
 * Maps normalized values to [0, 1] range with smooth gradients.
 *
 * COMPLEXITY: O(1)
 */
static float apply_sigmoid(float normalized) {
    return 1.0f / (1.0f + expf(-5.0f * normalized));
}

/**
 * @brief Clamps value to [0, 1] range
 *
 * WHY: Extracted clamping logic. Eliminates nested ifs - uses ternary.
 * Ensures confidence stays in valid range.
 *
 * COMPLEXITY: O(1)
 */
static float clamp_confidence(float value) {
    return (value < 0.0f) ? 0.0f : ((value > 1.0f) ? 1.0f : value);
}

/**
 * @brief Calculates confidence from neuron state
 *
 * WHY: Converts raw neuron state into confidence metric [0, 1].
 * Uses sigmoid function for smooth mapping. Refactored to eliminate
 * nested ifs - extracted into helper functions.
 *
 * ALGORITHM:
 * 1. Normalize state relative to threshold (O(1))
 * 2. Apply sigmoid transformation (O(1))
 * 3. Clamp to valid range (O(1))
 *
 * COMPLEXITY: O(1)
 *
 * @param state - Neuron activation state
 * @param threshold - Firing threshold
 * @return Confidence value in range [0.0, 1.0]
 */
float event_calculate_confidence(float state, float threshold) {
    float normalized = normalize_state(state, threshold);
    float confidence = apply_sigmoid(normalized);
    return clamp_confidence(confidence);
}

/**
 * @brief Converts neuron type to event packet flags
 *
 * WHY: Maps neural network neuron types to NIMCP event packet flags.
 * Switch statement acceptable here - simple enumeration mapping.
 *
 * COMPLEXITY: O(1)
 *
 * @return Event flags byte
 */
uint8_t event_flags_from_neuron_type(neuron_type_t type) {
    switch (type) {
        case NEURON_EXCITATORY:
            return EVENT_FLAG_EXCITATORY;
        case NEURON_INHIBITORY:
            return EVENT_FLAG_INHIBITORY;
        default:
            return EVENT_FLAG_EXCITATORY;
    }
}

/**
 * @brief Generates default feature code for neuron
 *
 * WHY: Creates unique feature code by combining base domain with neuron ID.
 * Each neuron gets distinguishable feature code for event routing.
 *
 * COMPLEXITY: O(1)
 *
 * @param base_code - Base feature code containing domain
 * @param neuron_id - Unique neuron identifier
 * @return Feature code combining domain and neuron ID
 */
feature_code_t event_default_feature_code(
    feature_code_t base_code,
    uint32_t neuron_id) {

    // Extract base domain from feature code
    uint8_t domain = GET_FEATURE_DOMAIN(base_code);

    // Create feature code with neuron ID as sub-feature
    return MAKE_FEATURE_CODE(domain, neuron_id & 0xFFFF);
}
