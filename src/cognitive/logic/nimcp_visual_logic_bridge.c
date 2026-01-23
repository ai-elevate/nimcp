/**
 * @file nimcp_visual_logic_bridge.c
 * @brief Visual-Logic Bridge Implementation
 *
 * Connects visual perception to symbolic reasoning system.
 * Enables visual concept grounding and perception-based inference.
 */

#include "utils/bridge/nimcp_bridge_base.h"
#include "cognitive/logic/nimcp_visual_logic_bridge.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/time/nimcp_time.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <stdio.h>

//=============================================================================
// Internal Constants
//=============================================================================

#define MAX_GROUNDED_OBJECTS 256
#define MAX_PENDING_COMMANDS 32

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief Grounded object entry
 */
typedef struct {
    uint32_t object_id;
    char concept_name[64];
    float confidence;
    uint32_t location_x;
    uint32_t location_y;
    uint64_t grounded_time;
    bool active;
} grounded_object_t;

/**
 * @brief Bridge internal structure
 */
struct visual_logic_bridge {
    bridge_base_t base;              /**< MUST be first: base bridge infrastructure */
    void* visual;                           /**< Visual cortex handle */
    void* logic;                            /**< Logic module handle */
    visual_logic_config_t config;           /**< Configuration */
    visual_logic_stats_t stats;             /**< Statistics */

    /* Grounding state */
    grounded_object_t grounded[MAX_GROUNDED_OBJECTS];
    uint32_t grounded_count;

    /* Pending commands (logic -> visual) */
    logic_visual_command_t pending_commands[MAX_PENDING_COMMANDS];
    uint32_t pending_count;

    /* Running averages */
    float confidence_sum;
    float relation_confidence_sum;
    uint64_t confidence_count;
    uint64_t relation_count;
};

//=============================================================================
// Configuration API
//=============================================================================

visual_logic_config_t visual_logic_default_config(void) {
    return (visual_logic_config_t){
        .enable_object_grounding = true,
        .enable_relation_extraction = true,
        .enable_top_down_attention = true,
        .enable_verification = true,
        .min_confidence_threshold = 0.5f,
        .min_salience_threshold = 0.3f,
        .max_objects_per_frame = 32,
        .max_relations_per_frame = 64
    };
}

//=============================================================================
// Lifecycle API
//=============================================================================

visual_logic_bridge_t* visual_logic_bridge_create(
    void* visual,
    void* logic,
    const visual_logic_config_t* config
) {
    visual_logic_bridge_t* bridge = nimcp_calloc(1, sizeof(visual_logic_bridge_t));
    if (!bridge) return NULL;

    bridge->visual = visual;
    bridge->logic = logic;
    bridge->config = config ? *config : visual_logic_default_config();

    memset(&bridge->stats, 0, sizeof(bridge->stats));
    memset(bridge->grounded, 0, sizeof(bridge->grounded));
    bridge->grounded_count = 0;
    bridge->pending_count = 0;

    bridge->confidence_sum = 0.0f;
    bridge->relation_confidence_sum = 0.0f;
    bridge->confidence_count = 0;
    bridge->relation_count = 0;

    return bridge;
}

void visual_logic_bridge_destroy(visual_logic_bridge_t* bridge) {
    if (bridge) {
        nimcp_free(bridge);
    }
}

int visual_logic_bridge_reset(visual_logic_bridge_t* bridge) {
    if (!bridge) return -1;

    memset(bridge->grounded, 0, sizeof(bridge->grounded));
    bridge->grounded_count = 0;
    bridge->pending_count = 0;

    return 0;
}

//=============================================================================
// Helper Functions
//=============================================================================

static int find_grounded_slot(visual_logic_bridge_t* bridge, uint32_t object_id) {
    for (uint32_t i = 0; i < MAX_GROUNDED_OBJECTS; i++) {
        if (bridge->grounded[i].active && bridge->grounded[i].object_id == object_id) {
            return (int)i;
        }
    }
    return -1;
}

static int find_free_slot(visual_logic_bridge_t* bridge) {
    for (uint32_t i = 0; i < MAX_GROUNDED_OBJECTS; i++) {
        if (!bridge->grounded[i].active) {
            return (int)i;
        }
    }
    return -1;
}

//=============================================================================
// Visual → Logic API
//=============================================================================

int visual_logic_ground_observation(
    visual_logic_bridge_t* bridge,
    const visual_logic_observation_t* obs
) {
    if (!bridge || !obs) return -1;
    if (!bridge->config.enable_object_grounding) return 0;

    /* Filter by confidence and salience */
    if (obs->confidence < bridge->config.min_confidence_threshold) {
        return 0;
    }
    if (obs->salience < bridge->config.min_salience_threshold) {
        return 0;
    }

    /* Check if already grounded, update if so */
    int slot = find_grounded_slot(bridge, obs->object_id);
    if (slot < 0) {
        slot = find_free_slot(bridge);
        if (slot < 0) {
            /* No free slots - could evict oldest or lowest confidence */
            return -1;
        }
        bridge->grounded_count++;
    }

    /* Update grounded object entry */
    grounded_object_t* entry = &bridge->grounded[slot];
    entry->object_id = obs->object_id;
    strncpy(entry->concept_name, obs->concept_name, sizeof(entry->concept_name) - 1);
    entry->concept_name[sizeof(entry->concept_name) - 1] = '\0';
    entry->confidence = obs->confidence;
    entry->location_x = obs->location_x;
    entry->location_y = obs->location_y;
    entry->grounded_time = obs->timestamp_us ? obs->timestamp_us : nimcp_time_get_us();
    entry->active = true;

    /* Update statistics */
    bridge->stats.objects_grounded++;
    bridge->confidence_sum += obs->confidence;
    bridge->confidence_count++;
    bridge->stats.avg_grounding_confidence = bridge->confidence_sum / bridge->confidence_count;

    /*
     * TODO: Actually inject predicate into logic module
     * Would call something like:
     * symbolic_logic_assert_fact(bridge->logic, predicate_name, terms, arity, confidence);
     */

    return 0;
}

int visual_logic_report_relation(
    visual_logic_bridge_t* bridge,
    const visual_logic_relation_t* relation
) {
    if (!bridge || !relation) return -1;
    if (!bridge->config.enable_relation_extraction) return 0;

    /* Validate both objects are grounded */
    if (find_grounded_slot(bridge, relation->subject_id) < 0) {
        return 0; /* Subject not grounded */
    }
    if (find_grounded_slot(bridge, relation->object_id) < 0) {
        return 0; /* Object not grounded */
    }

    /* Update statistics */
    bridge->stats.relations_extracted++;
    bridge->relation_confidence_sum += relation->confidence;
    bridge->relation_count++;
    bridge->stats.avg_relation_confidence = bridge->relation_confidence_sum / bridge->relation_count;

    /*
     * TODO: Actually inject relation predicate into logic module
     * Would call something like:
     * symbolic_logic_assert_relation(bridge->logic, relation_name, subject, object, confidence);
     */

    return 0;
}

int visual_logic_process_frame(
    visual_logic_bridge_t* bridge,
    const visual_logic_observation_t* observations,
    uint32_t num_observations,
    const visual_logic_relation_t* relations,
    uint32_t num_relations
) {
    if (!bridge) return -1;

    int processed = 0;

    /* Process observations */
    uint32_t obs_limit = num_observations < bridge->config.max_objects_per_frame
                        ? num_observations : bridge->config.max_objects_per_frame;

    for (uint32_t i = 0; i < obs_limit; i++) {
        if (visual_logic_ground_observation(bridge, &observations[i]) == 0) {
            processed++;
        }
    }

    /* Process relations */
    if (relations && num_relations > 0) {
        uint32_t rel_limit = num_relations < bridge->config.max_relations_per_frame
                            ? num_relations : bridge->config.max_relations_per_frame;

        for (uint32_t i = 0; i < rel_limit; i++) {
            if (visual_logic_report_relation(bridge, &relations[i]) == 0) {
                processed++;
            }
        }
    }

    return processed;
}

//=============================================================================
// Logic → Visual API
//=============================================================================

int visual_logic_request_attention(
    visual_logic_bridge_t* bridge,
    const char* concept_name,
    float priority
) {
    if (!bridge || !concept_name) return -1;
    if (!bridge->config.enable_top_down_attention) return 0;

    if (bridge->pending_count >= MAX_PENDING_COMMANDS) {
        return -1; /* Command queue full */
    }

    logic_visual_command_t* cmd = &bridge->pending_commands[bridge->pending_count++];
    cmd->signal_type = LOGIC_VISUAL_ATTEND_OBJECT;
    strncpy(cmd->target_concept, concept_name, sizeof(cmd->target_concept) - 1);
    cmd->target_concept[sizeof(cmd->target_concept) - 1] = '\0';
    cmd->priority = priority < 0.0f ? 0.0f : (priority > 1.0f ? 1.0f : priority);
    cmd->spatial_hint = false;

    bridge->stats.attention_commands++;

    /*
     * TODO: Actually send attention command to visual cortex
     * Would call something like:
     * visual_cortex_set_attention_target(bridge->visual, concept_name, priority);
     */

    return 0;
}

int visual_logic_verify_predicate(
    visual_logic_bridge_t* bridge,
    const char* predicate_name,
    bool expected_value,
    bool* verified,
    float* confidence
) {
    if (!bridge || !predicate_name || !verified || !confidence) return -1;
    if (!bridge->config.enable_verification) {
        *verified = false;
        *confidence = 0.0f;
        return 0;
    }

    bridge->stats.verifications_requested++;

    /* Search grounded objects for matching concept */
    *verified = false;
    *confidence = 0.0f;

    for (uint32_t i = 0; i < MAX_GROUNDED_OBJECTS; i++) {
        if (bridge->grounded[i].active) {
            if (strncmp(bridge->grounded[i].concept_name, predicate_name,
                       sizeof(bridge->grounded[i].concept_name)) == 0) {
                *verified = expected_value; /* Found matching concept */
                *confidence = bridge->grounded[i].confidence;
                bridge->stats.verifications_confirmed++;
                return 0;
            }
        }
    }

    /* Not found - verification fails if expected true, succeeds if expected false */
    *verified = !expected_value;
    *confidence = 0.5f; /* Medium confidence for absence */

    return 0;
}

int visual_logic_send_command(
    visual_logic_bridge_t* bridge,
    const logic_visual_command_t* command
) {
    if (!bridge || !command) return -1;

    if (bridge->pending_count >= MAX_PENDING_COMMANDS) {
        return -1;
    }

    bridge->pending_commands[bridge->pending_count++] = *command;

    switch (command->signal_type) {
        case LOGIC_VISUAL_ATTEND_OBJECT:
        case LOGIC_VISUAL_EXPECT_FEATURE:
            bridge->stats.attention_commands++;
            break;
        case LOGIC_VISUAL_VERIFY_PREDICATE:
            bridge->stats.verifications_requested++;
            break;
        default:
            break;
    }

    return 0;
}

//=============================================================================
// Query API
//=============================================================================

int visual_logic_is_grounded(
    const visual_logic_bridge_t* bridge,
    uint32_t object_id,
    bool* grounded
) {
    if (!bridge || !grounded) return -1;

    for (uint32_t i = 0; i < MAX_GROUNDED_OBJECTS; i++) {
        if (bridge->grounded[i].active && bridge->grounded[i].object_id == object_id) {
            *grounded = true;
            return 0;
        }
    }

    *grounded = false;
    return 0;
}

int visual_logic_get_grounded_count(const visual_logic_bridge_t* bridge) {
    if (!bridge) return -1;
    return (int)bridge->grounded_count;
}

//=============================================================================
// Statistics API
//=============================================================================

int visual_logic_bridge_get_stats(
    const visual_logic_bridge_t* bridge,
    visual_logic_stats_t* stats
) {
    if (!bridge || !stats) return -1;
    *stats = bridge->stats;
    return 0;
}

void visual_logic_bridge_reset_stats(visual_logic_bridge_t* bridge) {
    if (bridge) {
        memset(&bridge->stats, 0, sizeof(bridge->stats));
        bridge->confidence_sum = 0.0f;
        bridge->relation_confidence_sum = 0.0f;
        bridge->confidence_count = 0;
        bridge->relation_count = 0;
    }
}

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int visual_logic_bridge_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;

    const kg_entity_t* self = kg_reader_get_entity(kg, "Visual_Logic_Bridge");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            (void)self->observations[i];
        }
    }

    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Visual_Logic_Bridge");
    if (connections) {
        kg_relation_list_destroy(connections);
    }

    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Visual_Logic_Bridge");
    if (incoming) {
        kg_relation_list_destroy(incoming);
    }

    return self ? 1 : 0;
}
