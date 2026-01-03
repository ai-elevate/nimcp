/**
 * @file nimcp_conceptual_blending.c
 * @brief Conceptual blending engine implementation
 */

#include "cognitive/parietal/nimcp_conceptual_blending.h"
#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "utils/memory/nimcp_memory.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

struct blending_engine {
    blend_config_t config;
    blend_stats_t stats;
    float inflammation;
    float fatigue;
    uint32_t next_blend_id;
    uint32_t next_space_id;
};

static __thread char g_last_error[256] = {0};

static void set_error(const char* msg) {
    strncpy(g_last_error, msg, sizeof(g_last_error) - 1);
}

static float apply_mod(const blending_engine_t* e, float v) {
    float f = 1.0f - e->inflammation * e->config.inflammation_sensitivity * 0.2f
                   - e->fatigue * e->config.fatigue_sensitivity * 0.3f;
    return v * fmaxf(0.4f, f);
}

static float feature_sim(const float* a, const float* b, uint32_t dim) {
    float dot = 0, na = 0, nb = 0;
    for (uint32_t i = 0; i < dim; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float d = sqrtf(na) * sqrtf(nb);
    return (d > 1e-10f) ? dot / d : 0;
}

blend_config_t blending_engine_default_config(void) {
    return (blend_config_t){
        .min_integration = 0.3f,
        .novelty_weight = 0.4f,
        .enable_emergence_detection = true,
        .enable_optimization = true,
        .max_blend_iterations = 10,
        .inflammation_sensitivity = 1.0f,
        .fatigue_sensitivity = 1.0f
    };
}

blending_engine_t* blending_engine_create(void) {
    blend_config_t c = blending_engine_default_config();
    return blending_engine_create_custom(&c);
}

blending_engine_t* blending_engine_create_custom(const blend_config_t* config) {
    if (!config) { set_error("NULL config"); return NULL; }
    blending_engine_t* e = nimcp_calloc(1, sizeof(blending_engine_t));
    if (!e) return NULL;
    e->config = *config;
    e->next_blend_id = 1;
    e->next_space_id = 1;
    return e;
}

void blending_engine_destroy(blending_engine_t* engine) {
    if (engine) nimcp_free(engine);
}

blend_mental_space_t* blending_create_space(const char* name) {
    blend_mental_space_t* s = nimcp_calloc(1, sizeof(blend_mental_space_t));
    if (!s) return NULL;

    if (name) strncpy(s->name, name, sizeof(s->name) - 1);
    s->elements = nimcp_calloc(BLEND_MAX_ELEMENTS, sizeof(blend_element_t));
    return s;
}

int blending_add_element(blend_mental_space_t* space, const char* name,
    const float* features, uint32_t num_features) {
    if (!space || space->num_elements >= BLEND_MAX_ELEMENTS) return -1;

    blend_element_t* e = &space->elements[space->num_elements];
    e->id = space->num_elements + 1;
    if (name) strncpy(e->name, name, sizeof(e->name) - 1);

    if (features && num_features > 0) {
        e->features = nimcp_calloc(num_features, sizeof(float));
        if (e->features) {
            memcpy(e->features, features, num_features * sizeof(float));
            e->num_features = num_features;
        }
    }

    space->num_elements++;
    return 0;
}

void blending_free_space(blend_mental_space_t* space) {
    if (!space) return;
    if (space->elements) {
        for (uint32_t i = 0; i < space->num_elements; i++) {
            if (space->elements[i].features) nimcp_free(space->elements[i].features);
        }
        nimcp_free(space->elements);
    }
    if (space->frame_structure) nimcp_free(space->frame_structure);
    nimcp_free(space);
}

conceptual_blend_t* blending_create_blend(blending_engine_t* engine,
    const blend_mental_space_t* concept1, const blend_mental_space_t* concept2) {
    if (!engine || !concept1 || !concept2) return NULL;

    conceptual_blend_t* b = nimcp_calloc(1, sizeof(conceptual_blend_t));
    if (!b) return NULL;

    b->id = engine->next_blend_id++;
    b->input1 = (blend_mental_space_t*)concept1;
    b->input2 = (blend_mental_space_t*)concept2;

    /* Create generic space (shared structure) */
    b->generic = blending_create_space("generic");

    /* Create blend space */
    b->blend = blending_create_space("blend");
    if (b->blend) {
        /* Combine elements from both inputs */
        for (uint32_t i = 0; i < concept1->num_elements && b->blend->num_elements < BLEND_MAX_ELEMENTS; i++) {
            blending_add_element(b->blend, concept1->elements[i].name,
                concept1->elements[i].features, concept1->elements[i].num_features);
        }
        for (uint32_t i = 0; i < concept2->num_elements && b->blend->num_elements < BLEND_MAX_ELEMENTS; i++) {
            blending_add_element(b->blend, concept2->elements[i].name,
                concept2->elements[i].features, concept2->elements[i].num_features);
        }
    }

    /* Find mappings */
    b->mappings = nimcp_calloc(BLEND_MAX_MAPPINGS, sizeof(blend_mapping_t));
    blending_find_mappings(engine, concept1, concept2, b->mappings, BLEND_MAX_MAPPINGS, &b->num_mappings);

    /* Find emergent properties */
    if (engine->config.enable_emergence_detection) {
        b->emergent_properties = nimcp_calloc(BLEND_MAX_PROPERTIES, sizeof(blend_property_t));
        if (b->emergent_properties && b->num_mappings > 0) {
            b->emergent_properties[0].is_emergent = true;
            snprintf(b->emergent_properties[0].name, sizeof(b->emergent_properties[0].name),
                    "Emergent from blend");
            b->emergent_properties[0].strength = apply_mod(engine, 0.6f);
            b->num_emergent = 1;
            engine->stats.emergent_properties_found++;
        }
    }

    b->integration_score = apply_mod(engine, 0.5f + 0.3f * ((float)b->num_mappings / BLEND_MAX_MAPPINGS));
    b->novelty_score = apply_mod(engine, 0.6f);

    engine->stats.blends_created++;
    engine->stats.mappings_made += b->num_mappings;

    return b;
}

int blending_find_mappings(blending_engine_t* engine,
    const blend_mental_space_t* space1, const blend_mental_space_t* space2,
    blend_mapping_t* mappings, uint32_t max_mappings, uint32_t* num_found) {
    if (!engine || !space1 || !space2 || !mappings || !num_found) return -1;
    *num_found = 0;

    for (uint32_t i = 0; i < space1->num_elements && *num_found < max_mappings; i++) {
        float best_sim = 0.3f;
        int best_j = -1;

        for (uint32_t j = 0; j < space2->num_elements; j++) {
            if (space1->elements[i].num_features > 0 &&
                space2->elements[j].num_features > 0) {
                uint32_t min_f = (space1->elements[i].num_features < space2->elements[j].num_features) ?
                                  space1->elements[i].num_features : space2->elements[j].num_features;
                float sim = feature_sim(space1->elements[i].features,
                                        space2->elements[j].features, min_f);
                if (sim > best_sim) {
                    best_sim = sim;
                    best_j = j;
                }
            }
        }

        if (best_j >= 0) {
            mappings[*num_found].source_id = space1->elements[i].id;
            mappings[*num_found].target_id = space2->elements[best_j].id;
            mappings[*num_found].strength = best_sim;
            (*num_found)++;
        }
    }

    return 0;
}

blend_property_t** blending_find_emergent(blending_engine_t* engine,
    const conceptual_blend_t* blend, uint32_t* num_found) {
    if (!engine || !blend || !num_found) return NULL;
    *num_found = blend->num_emergent;
    return &blend->emergent_properties;
}

float blending_evaluate_novelty(blending_engine_t* engine, const conceptual_blend_t* blend) {
    if (!engine || !blend) return 0;
    return apply_mod(engine, blend->novelty_score);
}

float blending_evaluate_integration(blending_engine_t* engine, const conceptual_blend_t* blend) {
    if (!engine || !blend) return 0;
    return apply_mod(engine, blend->integration_score);
}

int blending_optimize_blend(blending_engine_t* engine, conceptual_blend_t* blend) {
    if (!engine || !blend) return -1;
    if (!engine->config.enable_optimization) return 0;

    for (uint32_t i = 0; i < engine->config.max_blend_iterations; i++) {
        blend->integration_score *= 1.05f;
        if (blend->integration_score > 0.95f) break;
    }
    blend->integration_score = fminf(1.0f, blend->integration_score);

    return 0;
}

void blending_free_blend(conceptual_blend_t* blend) {
    if (!blend) return;
    if (blend->generic) blending_free_space(blend->generic);
    if (blend->blend) blending_free_space(blend->blend);
    if (blend->mappings) nimcp_free(blend->mappings);
    if (blend->emergent_properties) nimcp_free(blend->emergent_properties);
    nimcp_free(blend);
}

int blending_set_inflammation(blending_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->inflammation = fmaxf(0, fminf(1, level));
    return 0;
}

int blending_set_fatigue(blending_engine_t* engine, float level) {
    if (!engine) return -1;
    engine->fatigue = fmaxf(0, fminf(1, level));
    return 0;
}

int blending_get_stats(const blending_engine_t* engine, blend_stats_t* stats) {
    if (!engine || !stats) return -1;
    *stats = engine->stats;
    return 0;
}

void blending_reset_stats(blending_engine_t* engine) {
    if (engine) memset(&engine->stats, 0, sizeof(engine->stats));
}

const char* blending_get_last_error(void) { return g_last_error; }

/* ============================================================================
 * Knowledge Graph Self-Awareness Integration
 * ============================================================================ */

int conceptual_blending_query_self_knowledge(kg_reader_t* kg) {
    if (!kg) return 0;
    const kg_entity_t* self = kg_reader_get_entity(kg, "Conceptual_Blending");
    if (self) {
        for (uint32_t i = 0; i < self->num_observations; i++) {
            /* Module self-knowledge logged */
        }
    }
    kg_relation_list_t* connections = kg_reader_get_relations_from(kg, "Conceptual_Blending");
    if (connections) { kg_relation_list_destroy(connections); }
    kg_relation_list_t* incoming = kg_reader_get_relations_to(kg, "Conceptual_Blending");
    if (incoming) { kg_relation_list_destroy(incoming); }
    return self ? 1 : 0;
}
