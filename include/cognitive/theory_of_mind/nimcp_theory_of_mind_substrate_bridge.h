/**
 * @file nimcp_theory_of_mind_substrate_bridge.h
 * @brief Bridge between Theory of Mind system and neural substrate
 *
 * WHAT: Links Theory of Mind to metabolic/energy state
 * WHY: Mentalizing and perspective-taking require high cognitive resources
 * HOW: Monitors ATP/fatigue; modulates mental state inference, perspective-taking
 *
 * BIOLOGICAL BASIS:
 * - ToM involves medial prefrontal cortex and temporal-parietal junction
 * - ATP depletion leads to reduced mentalizing capacity
 * - Fatigue impairs perspective-taking and false belief reasoning
 * - Metabolic stress reduces recursive mental state inference depth
 *
 * @author NIMCP Team
 * @date 2024-12-30
 */

#ifndef NIMCP_THEORY_OF_MIND_SUBSTRATE_BRIDGE_H
#define NIMCP_THEORY_OF_MIND_SUBSTRATE_BRIDGE_H

#include "core/neural_substrate/nimcp_neural_substrate.h"
#include "async/nimcp_bio_router.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BIO_MODULE_SUBSTRATE_TOM 0x130A

typedef struct {
    float mentalizing_capacity;    /* Mental state inference capacity [0-1] */
    float perspective_taking;      /* Perspective-taking ability [0-1] */
    float recursive_depth;         /* Recursive inference depth [0-1] */
    float false_belief_reasoning;  /* False belief reasoning capacity [0-1] */
    float overall_capacity;        /* Combined modulation [0-1] */
} tom_substrate_effects_t;

typedef struct {
    bool enable_atp_modulation;
    bool enable_fatigue_modulation;
    bool enable_bio_async;
    float atp_sensitivity;
    float fatigue_sensitivity;
    float min_capacity;
} tom_substrate_config_t;

typedef struct tom_substrate_bridge tom_substrate_bridge_t;

tom_substrate_config_t tom_substrate_default_config(void);
tom_substrate_bridge_t* tom_substrate_bridge_create(void* tom, neural_substrate_t* substrate, const tom_substrate_config_t* config);
void tom_substrate_bridge_destroy(tom_substrate_bridge_t* bridge);
int tom_substrate_bridge_update(tom_substrate_bridge_t* bridge);
int tom_substrate_bridge_get_effects(const tom_substrate_bridge_t* bridge, tom_substrate_effects_t* effects);
int tom_substrate_bridge_apply_effects(tom_substrate_bridge_t* bridge);
int tom_substrate_bridge_register_bio_async(tom_substrate_bridge_t* bridge, bio_router_t* router);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_THEORY_OF_MIND_SUBSTRATE_BRIDGE_H */
