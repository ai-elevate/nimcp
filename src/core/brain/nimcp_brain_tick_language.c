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

#include "utils/logging/nimcp_logging.h"

/* Maximum bio-router messages drained per language adapter per tick. */
#define _LANG_BIO_BATCH 32u

void brain_tick_language(brain_t brain, float dt_ms)
{
    (void)dt_ms;  /* Unused — language adapters drain messages, not time. */

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
}
