/**
 * @file nimcp_brain_init_region_cycles.c
 * @brief Wraps Wave 8B/8C tick drivers as driven coordinator cycles.
 *
 * Each of the 4 tick drivers (neuromod, sensorimotor, language,
 * physics_bridges) is already implemented as a standalone function and
 * called inline from brain_learn_vector + brain_decide. This file
 * additionally registers each as a driven cycle at 16ms so they advance
 * on the coordinator's own thread regardless of hot-path activity.
 *
 * Note on double-firing: brain_learn_vector, brain_decide, AND this
 * coordinator cycle all call the same tick functions. Because tick
 * functions are effectively idempotent for small dt (null-guarded +
 * integrate-forward), over-calling advances state faster but doesn't
 * corrupt it. The only observable cost is ~2x simulation time during
 * high-activity windows. Accepted trade for redundancy during idle
 * inference periods where hot-path hooks would otherwise be silent.
 */

#include "core/brain/factory/init/nimcp_brain_init_region_cycles.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/nimcp_brain_cycle_coordinator.h"
#include "core/brain/nimcp_brain_tick_neuromod.h"
#include "core/brain/nimcp_brain_tick_sensorimotor.h"
#include "core/brain/nimcp_brain_tick_language.h"
#include "core/brain/nimcp_brain_tick_physics_bridges.h"
#include "core/brain/nimcp_brain_tick_meta_learning.h"
#include "core/brain/nimcp_brain_tick_intuitive_physics.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/bridge/nimcp_bridge_boilerplate.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

BRIDGE_BOILERPLATE_MESH_ONLY(brain_init_region_cycles, MESH_ADAPTER_CATEGORY_SYSTEM)

#define REGION_CYCLES_LOG_MODULE "BRAIN_INIT_REGION_CYCLES"

/* Tick wrappers — adapt the brain_tick_*(brain, dt_ms) signatures to the
 * coordinator's (void*)->void callback type. dt_ms is fixed at the cycle's
 * registered interval (16ms). */

static void tick_neuromod_wrapper(void* ctx) {
    brain_tick_neuromod((brain_t)ctx, 16.0f);
}

static void tick_sensorimotor_wrapper(void* ctx) {
    brain_tick_sensorimotor((brain_t)ctx, 16.0f);
}

static void tick_language_wrapper(void* ctx) {
    brain_tick_language((brain_t)ctx, 16.0f);
}

static void tick_physics_bridges_wrapper(void* ctx) {
    brain_tick_physics_bridges((brain_t)ctx, 16.0f);
}

static void tick_meta_learning_wrapper(void* ctx) {
    /* 100ms cycle cadence — the tick driver ignores dt; see
     * brain_tick_meta_learning_impl.c for rationale. */
    brain_tick_meta_learning((brain_t)ctx, 100.0f);
}

static void tick_intuitive_physics_wrapper(void* ctx) {
    /* 16ms cycle cadence — matches stage_physics_task's fixed 16ms dt. */
    brain_tick_intuitive_physics((brain_t)ctx, 16.0f);
}

bool nimcp_brain_factory_init_region_cycles_subsystem(brain_t brain) {
    if (!brain) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
            "nimcp_brain_factory_init_region_cycles_subsystem: brain is NULL");
        return false;
    }

    brain_cycle_coordinator_t* coord =
        (brain_cycle_coordinator_t*)brain->cycle_coordinator;
    if (!coord || !brain->cycle_coordinator_enabled) {
        LOG_MODULE_INFO(REGION_CYCLES_LOG_MODULE,
            "cycle coordinator absent — skipping region cycle registration");
        return false;
    }

    LOG_MODULE_INFO(REGION_CYCLES_LOG_MODULE,
        "Registering Wave 8B/8C region adapters as driven cycles (16ms each)");

    int ok_count = 0;

    if (brain_cycle_coordinator_register_driven(
            coord, BRAIN_CYCLE_NEUROMOD, 16000ull,
            tick_neuromod_wrapper, brain, NULL) == 0) {
        ok_count++;
    } else {
        LOG_MODULE_WARN(REGION_CYCLES_LOG_MODULE,
            "register_driven(NEUROMOD) failed");
    }

    if (brain_cycle_coordinator_register_driven(
            coord, BRAIN_CYCLE_SENSORIMOTOR, 16000ull,
            tick_sensorimotor_wrapper, brain, NULL) == 0) {
        ok_count++;
    } else {
        LOG_MODULE_WARN(REGION_CYCLES_LOG_MODULE,
            "register_driven(SENSORIMOTOR) failed");
    }

    if (brain_cycle_coordinator_register_driven(
            coord, BRAIN_CYCLE_LANGUAGE, 16000ull,
            tick_language_wrapper, brain, NULL) == 0) {
        ok_count++;
    } else {
        LOG_MODULE_WARN(REGION_CYCLES_LOG_MODULE,
            "register_driven(LANGUAGE) failed");
    }

    if (brain_cycle_coordinator_register_driven(
            coord, BRAIN_CYCLE_PHYSICS_BRIDGES, 16000ull,
            tick_physics_bridges_wrapper, brain, NULL) == 0) {
        ok_count++;
    } else {
        LOG_MODULE_WARN(REGION_CYCLES_LOG_MODULE,
            "register_driven(PHYSICS_BRIDGES) failed");
    }

    /* Wave 4 meta_learning — 100ms cadence. meta_learner may be NULL if
     * enable_meta_learning was false; the tick driver null-guards. */
    if (brain_cycle_coordinator_register_driven(
            coord, BRAIN_CYCLE_META_LEARNING, 100000ull,
            tick_meta_learning_wrapper, brain, NULL) == 0) {
        ok_count++;
    } else {
        LOG_MODULE_WARN(REGION_CYCLES_LOG_MODULE,
            "register_driven(META_LEARNING) failed");
    }

    /* Wave 5 intuitive_physics — 16ms cadence. Engine may be NULL if
     * enable_intuitive_physics was false; the tick driver null-guards. */
    if (brain_cycle_coordinator_register_driven(
            coord, BRAIN_CYCLE_INTUITIVE_PHYSICS, 16000ull,
            tick_intuitive_physics_wrapper, brain, NULL) == 0) {
        ok_count++;
    } else {
        LOG_MODULE_WARN(REGION_CYCLES_LOG_MODULE,
            "register_driven(INTUITIVE_PHYSICS) failed");
    }

    LOG_MODULE_INFO(REGION_CYCLES_LOG_MODULE,
        "region cycles registered: %d/6", ok_count);
    return ok_count > 0;
}

void nimcp_brain_factory_destroy_region_cycles_subsystem(brain_t brain) {
    if (!brain) return;

    brain_cycle_coordinator_t* coord =
        (brain_cycle_coordinator_t*)brain->cycle_coordinator;
    if (!coord) return;

    /* Unregister in reverse order. Each unregister joins the driver
     * thread, so any in-flight tick completes before we return. */
    (void)brain_cycle_coordinator_unregister(coord, BRAIN_CYCLE_INTUITIVE_PHYSICS);
    (void)brain_cycle_coordinator_unregister(coord, BRAIN_CYCLE_META_LEARNING);
    (void)brain_cycle_coordinator_unregister(coord, BRAIN_CYCLE_PHYSICS_BRIDGES);
    (void)brain_cycle_coordinator_unregister(coord, BRAIN_CYCLE_LANGUAGE);
    (void)brain_cycle_coordinator_unregister(coord, BRAIN_CYCLE_SENSORIMOTOR);
    (void)brain_cycle_coordinator_unregister(coord, BRAIN_CYCLE_NEUROMOD);
}
