//=============================================================================
// nimcp_brain_tick_language.c - Language tick driver (broca + wernicke)
//=============================================================================
/**
 * @file nimcp_brain_tick_language.c
 * @brief Drives *_process_bio_messages() on broca + wernicke adapters.
 *
 * WHAT: Implements brain_tick_language() — a single call that drains the
 *       bio-router inboxes of the two language-region adapters.
 *
 * WHY:  The per-adapter primary ops (broca_process_bio_messages,
 *       wernicke_process_bio_messages) had zero external callers prior to
 *       Wave 8B-d. Without this driver, each adapter's inbox grows unbounded
 *       and the language loop (arcuate fasciculus: wernicke → broca → ...)
 *       never advances state in response to incoming messages.
 *
 * HOW:  For each adapter, check NULL + `*_enabled` flag, then invoke the
 *       primary op with `_LANG_BIO_BATCH` as the per-tick drain budget.
 *
 * DRAIN BUDGET: _LANG_BIO_BATCH = 32 messages per adapter per tick. Higher
 *       than the Round A _BIO_BATCH=16 because broca+wernicke talk
 *       back-and-forth via the arcuate fasciculus and accumulate in pairs.
 *
 * @author NIMCP Development Team
 * @date 2026-04-24
 * @version 1.0.0
 */

#include "core/brain/nimcp_brain_tick_language.h"

/* Include broca adapter before brain_internal.h (broca adapter is
 * self-contained and conflict-free). */
#include "core/brain/regions/broca/nimcp_broca_adapter.h"

/* NOTE: `nimcp_wernicke_adapter.h` forward-declares
 *   `typedef struct semantic_memory semantic_memory_t;`
 * while `cognitive/parietal/nimcp_intuition_integrations.h` (pulled in
 * transitively by brain_internal.h) declares it as
 *   `typedef struct semantic_memory_system semantic_memory_t;`
 * These conflict at compile time. Other consumers
 * (e.g. nimcp_brain_init_wernicke.c) work around this by forward-declaring
 * only the adapter type + the specific function they need. We do the same
 * here to keep the include surface minimal and the tick driver decoupled
 * from wernicke's internal type zoo.
 */
typedef struct wernicke_adapter wernicke_adapter_t;
uint32_t wernicke_process_bio_messages(wernicke_adapter_t* adapter,
                                        uint32_t max_messages);

#include "core/brain/nimcp_brain_internal.h"

#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_types.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/logging/nimcp_logging.h"

/* Maximum bio-router messages drained per language adapter per tick. */
#define _LANG_BIO_BATCH 32u

/* PA-3: drain Broca/Wernicke spike_output through the SNN language bridge.
 * Inert until the caller flips bridge->config.enable_snn_spike_routing on
 * via nimcp_brain_set_snn_language_bridge_spike_routing(). The bridge holds
 * the canonical list of attached pops; we just iterate it. */
static void brain_tick_lang_bridge_spike_routing(brain_t brain, float dt_ms)
{
    if (!brain || !brain->snn_lang_bridge || !brain->snn_network) return;

    /* Wallclock approximation for STDP timing — monotonic tick accumulator.
     * SNN step runs before brain_tick_language in the global coordinator,
     * so spike_output for this tick is fresh. The accumulator is per-call
     * (not per-brain), which is fine because tests run a single brain at
     * a time and the only requirement is that t_ms increases monotonically
     * across consecutive drain calls. */
    static float t_ms = 0.0f;
    t_ms += dt_ms;

    for (uint32_t i = 0; i < SNN_LANG_MAX_ATTACHED_POPS; i++) {
        int pop_id = -1;
        uint32_t n_neurons = 0;
        snn_lang_pop_role_t role = SNN_LANG_POP_ROLE_CONCEPT;
        if (snn_language_bridge_get_attached_pop(brain->snn_lang_bridge, i,
                                                   &pop_id, &n_neurons,
                                                   &role) != 0) {
            continue;
        }
        if (pop_id < 0) continue;

        snn_population_t* pop = snn_network_get_population(
            brain->snn_network, (uint32_t)pop_id);
        if (!pop || !pop->spike_output) continue;

        const float* spikes =
            (const float*)nimcp_tensor_data(pop->spike_output);
        if (!spikes) continue;

        (void)snn_language_bridge_drain_pop_spikes(brain->snn_lang_bridge,
                                                     pop_id, spikes,
                                                     pop->n_neurons, t_ms);
    }

    /* Always-on activation decay — cheap, idempotent, guards against runaway
     * even when other paths inject spikes through concept_spike/word_spike. */
    (void)snn_language_bridge_tick(brain->snn_lang_bridge, dt_ms);
}

void brain_tick_language(brain_t brain, float dt_ms)
{
    if (!brain) {
        return;
    }

    NIMCP_LOGGING_TRACE("brain_tick_language: dt_ms=%.3f", (double)dt_ms);

    /* Broca's area — drains inbox up to _LANG_BIO_BATCH messages. */
    if (brain->broca && brain->broca_enabled) {
        (void)broca_process_bio_messages(
            (broca_adapter_t*)brain->broca, _LANG_BIO_BATCH);
    }

    /* Wernicke's area — drains inbox up to _LANG_BIO_BATCH messages. */
    if (brain->wernicke && brain->wernicke_enabled) {
        (void)wernicke_process_bio_messages(
            (wernicke_adapter_t*)brain->wernicke, _LANG_BIO_BATCH);
    }

    /* PA-3: bridge spike routing + activation decay. */
    brain_tick_lang_bridge_spike_routing(brain, dt_ms);
}
