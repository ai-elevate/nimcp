/**
 * @file nimcp_jepa_brain_bridges.c
 * @brief Brain-level wiring for the four JEPA bridges (Phase 4n-q).
 *
 * See header for purpose. Each create/destroy/tick entry no-ops
 * gracefully when the target subsystem is missing so the build stays
 * valid across brain configurations.
 */
#include "cognitive/jepa/nimcp_jepa_brain_bridges.h"
#include "core/brain/nimcp_brain_internal.h"

#include "cognitive/omni/nimcp_omni_jepa_bridge.h"
#include "plasticity/neuromodulators/nimcp_neuromod_jepa_bridge.h"
#include "perception/nimcp_audio_jepa_bridge.h"
#include "cognitive/memory/nimcp_engram_jepa_bridge.h"

#include "utils/logging/nimcp_logging.h"

/* Omni state dim — opaque to us; pick a reasonable small value. The
 * training pair is caller-provided vectors so bridge tolerates any dim
 * that matches between the pair, but since the unified tick doesn't
 * currently have natural per-call prev/cur vectors for omni, we just
 * set the size at init. If you later hook the bridge into a real
 * omni_wm_update wrapper, pass the actual state_dim in. */
#define _JEPA_OMNI_STATE_DIM 32u

/* Audio embedding dim (16 mel + 16 MFCC = 32). Matches the bridge's
 * packing logic inside audio_jepa_bridge_train_step. */
#define _JEPA_AUDIO_EMBED_DIM 32u

/* Engram embedding dim. Common engram_t embedding size. */
#define _JEPA_ENGRAM_EMBED_DIM 64u

void nimcp_jepa_brain_bridges_init(brain_t brain) {
    if (!brain) return;

    /* Omni world model bridge — allocates on any brain since omni is
     * typically present. The bridge is driver-agnostic; the tick just
     * does not train it until caller-supplied pairs are provided. */
    if (!brain->omni_jepa_bridge) {
        brain->omni_jepa_bridge = (void*)omni_jepa_bridge_create(_JEPA_OMNI_STATE_DIM);
        if (brain->omni_jepa_bridge) {
            NIMCP_LOGGING_INFO("jepa_brain_bridges: omni bridge live (dim=%u)",
                               _JEPA_OMNI_STATE_DIM);
        }
    }

    /* Neuromodulator bridge — requires an active neuromodulator_system. */
    if (!brain->neuromod_jepa_bridge && brain->neuromodulator_system) {
        brain->neuromod_jepa_bridge =
            (void*)neuromod_jepa_bridge_create(brain->neuromodulator_system);
        if (brain->neuromod_jepa_bridge) {
            NIMCP_LOGGING_INFO("jepa_brain_bridges: neuromod bridge live");
        }
    }

    /* Audio cortex bridge — requires an active audio_cortex. */
    if (!brain->audio_jepa_bridge && brain->audio_cortex) {
        brain->audio_jepa_bridge =
            (void*)audio_jepa_bridge_create(brain->audio_cortex,
                                             _JEPA_AUDIO_EMBED_DIM);
        if (brain->audio_jepa_bridge) {
            NIMCP_LOGGING_INFO("jepa_brain_bridges: audio bridge live (dim=%u)",
                               _JEPA_AUDIO_EMBED_DIM);
        }
    }

    /* Engram bridge — standalone, driven by caller record() calls. */
    if (!brain->engram_jepa_bridge) {
        brain->engram_jepa_bridge =
            (void*)engram_jepa_bridge_create(_JEPA_ENGRAM_EMBED_DIM);
        if (brain->engram_jepa_bridge) {
            NIMCP_LOGGING_INFO("jepa_brain_bridges: engram bridge live (dim=%u)",
                               _JEPA_ENGRAM_EMBED_DIM);
        }
    }
}

void nimcp_jepa_brain_bridges_destroy(brain_t brain) {
    if (!brain) return;
    if (brain->omni_jepa_bridge) {
        omni_jepa_bridge_destroy((omni_jepa_bridge_t*)brain->omni_jepa_bridge);
        brain->omni_jepa_bridge = NULL;
    }
    if (brain->neuromod_jepa_bridge) {
        neuromod_jepa_bridge_destroy(
            (neuromod_jepa_bridge_t*)brain->neuromod_jepa_bridge);
        brain->neuromod_jepa_bridge = NULL;
    }
    if (brain->audio_jepa_bridge) {
        audio_jepa_bridge_destroy((audio_jepa_bridge_t*)brain->audio_jepa_bridge);
        brain->audio_jepa_bridge = NULL;
    }
    if (brain->engram_jepa_bridge) {
        engram_jepa_bridge_destroy(
            (engram_jepa_bridge_t*)brain->engram_jepa_bridge);
        brain->engram_jepa_bridge = NULL;
    }
}

void nimcp_jepa_brain_bridges_tick(brain_t brain) {
    if (!brain) return;

    /* Only neuromod is self-driving (reads current levels against stored
     * prev). Omni + engram need caller-supplied vectors — they're
     * driven from specific code points (octopus hook, engram encode),
     * not from the generic tick. Audio self-drives against its cortex's
     * cached training state. Training is one-shot per call; cadence
     * gating lives upstream (in the brain-tick frequency). */
    if (brain->neuromod_jepa_bridge) {
        float loss = 0.0f;
        (void)neuromod_jepa_bridge_train_step(
            (neuromod_jepa_bridge_t*)brain->neuromod_jepa_bridge, &loss);
    }
    if (brain->audio_jepa_bridge) {
        float loss = 0.0f;
        (void)audio_jepa_bridge_train_step(
            (audio_jepa_bridge_t*)brain->audio_jepa_bridge, &loss);
    }
}
