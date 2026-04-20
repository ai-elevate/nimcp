/**
 * @file nimcp_octopus.c
 * @brief Octopus cognitive module — distributed peripheral cognition.
 *
 * See include/cognitive/octopus/nimcp_octopus.h for API + rationale.
 */
#include "cognitive/octopus/nimcp_octopus.h"

/* NIMCP infra: memory, logging, exceptions, math helpers. No raw
 * malloc/free or pthreads — all allocation flows through nimcp_memory. */
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"
#include "utils/math/nimcp_math_helpers.h"

#include <math.h>
#include <string.h>

/*============================================================================
 * Private struct
 *==========================================================================*/

struct octopus_system_s {
    uint32_t       n_arms;
    octopus_arm_t* arms;
    octopus_stats_t stats;

    /* Latest aggregated latent — kept so repeated reads don't re-aggregate. */
    float          last_aggregated[OCTOPUS_ARM_DIM];
    float          last_coherence;

    /* Integration hooks. */
    octopus_ethics_hook_t ethics_hook;   void* ethics_user;
    octopus_swarm_hook_t  swarm_hook;    void* swarm_user;
    octopus_world_hook_t  world_hook;    void* world_user;
    octopus_fep_hook_t    fep_hook;      void* fep_user;
    octopus_bio_hook_t    bio_hook;      void* bio_user;
    octopus_immune_hook_t immune_hook;   void* immune_user;

    /* Thresholds. */
    float    swarm_delegation_threshold; /**< arm.confidence > this → delegate */
    float    low_coherence_threshold;    /**< coherence < this → broadcast     */
};

/*============================================================================
 * Small helpers — all scalar math routes through utils/math/nimcp_math_helpers
 * (nimcp_clampf, nimcp_clamp01, nimcp_safe_expf, etc.). Do not re-define.
 *==========================================================================*/

/* Tanh-activated per-element transform on an input slice → latent.
 * This is the arm's local "processing" — a cheap nonlinearity plus
 * a contribution from the arm's recent history. Full ML weights aren't
 * the point; this is cheap distributed-reflex compute, biology-analogous. */
static void _arm_process_slice(octopus_arm_t* arm, const float* slice) {
    /* Store new input in history ring. */
    memcpy(arm->recent_inputs[arm->history_head], slice,
           sizeof(float) * OCTOPUS_ARM_DIM);
    arm->history_head = (arm->history_head + 1) % OCTOPUS_ARM_HISTORY;

    /* Compute new latent — 70% current slice, 30% smoothed history mean.
     * This gives arms continuity without being frozen to history. */
    float history_mean[OCTOPUS_ARM_DIM] = {0};
    for (uint32_t h = 0; h < OCTOPUS_ARM_HISTORY; h++) {
        for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
            history_mean[d] += arm->recent_inputs[h][d];
        }
    }
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
        history_mean[d] /= (float)OCTOPUS_ARM_HISTORY;
        float v = 0.7f * slice[d] + 0.3f * history_mean[d];
        arm->latent[d] = tanhf(v);
    }

    /* Confidence = L1 norm of latent, normalized. Arms whose latent is
     * all zeros or tiny have low confidence — literally "no opinion". */
    float l1 = 0.0f;
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) l1 += fabsf(arm->latent[d]);
    arm->confidence = nimcp_clampf(l1 / (float)OCTOPUS_ARM_DIM, 0.0f, 1.0f);

    /* Latent variance — how much this arm's latent deviates from the
     * mean of the full latent. High variance = arm is in a distinctive
     * local regime (exploration signal). */
    float mean = 0.0f;
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) mean += arm->latent[d];
    mean /= (float)OCTOPUS_ARM_DIM;
    float var = 0.0f;
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
        float diff = arm->latent[d] - mean;
        var += diff * diff;
    }
    arm->latent_variance = var / (float)OCTOPUS_ARM_DIM;

    /* Broadcast state — confidence × (1 + variance), squashed to [0,1].
     * An arm that's both confident AND distinctive broadcasts loudly. */
    arm->broadcast_state = nimcp_clampf(arm->confidence *
                                    (1.0f + arm->latent_variance), 0.0f, 1.0f);

    /* Chemosensory-tactile fusion: the first 32 elements of the slice
     * double as chemical fingerprint. Copy them directly so downstream
     * modules can read arm->chemo_id as a semantic-tactile sense. */
    for (uint32_t d = 0; d < 32; d++) arm->chemo_id[d] = slice[d];
}

/*============================================================================
 * Lifecycle
 *==========================================================================*/

octopus_system_t* octopus_create(uint32_t n_arms) {
    if (n_arms == 0) n_arms = OCTOPUS_DEFAULT_N_ARMS;
    if (n_arms < 2)  n_arms = 2;
    if (n_arms > 16) n_arms = 16;

    octopus_system_t* ctx = (octopus_system_t*)nimcp_calloc(
        1, sizeof(octopus_system_t));
    if (!ctx) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
                              "octopus_create: ctx alloc failed");
        return NULL;
    }
    ctx->n_arms = n_arms;
    ctx->arms = (octopus_arm_t*)nimcp_calloc(n_arms, sizeof(octopus_arm_t));
    if (!ctx->arms) {
        nimcp_free(ctx);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY,
                              "octopus_create: arms alloc failed");
        return NULL;
    }
    for (uint32_t i = 0; i < n_arms; i++) {
        ctx->arms[i].id = i;
    }
    ctx->stats.n_arms = n_arms;
    ctx->swarm_delegation_threshold = 0.75f;
    ctx->low_coherence_threshold    = 0.30f;

    NIMCP_LOGGING_INFO("octopus_create: %u arms initialized", n_arms);
    return ctx;
}

void octopus_destroy(octopus_system_t* ctx) {
    if (!ctx) return;
    nimcp_free(ctx->arms);
    nimcp_free(ctx);
}

/*============================================================================
 * Core operations
 *==========================================================================*/

int octopus_explore(octopus_system_t* ctx,
                     const float* input, uint32_t input_len) {
    if (!ctx || !input || input_len == 0) return -1;
    if (ctx->immune_hook) ctx->immune_hook(ctx->immune_user);

    float slice[OCTOPUS_ARM_DIM];
    for (uint32_t a = 0; a < ctx->n_arms; a++) {
        /* Per-arm slicing: strided windows with wraparound. This is
         * deliberately simple — each arm sees a different "part" of the
         * input and builds its own local opinion. */
        uint32_t start = (a * (OCTOPUS_ARM_DIM / 2)) % input_len;
        for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
            slice[d] = input[(start + d) % input_len];
        }
        _arm_process_slice(&ctx->arms[a], slice);
        ctx->arms[a].vetoed = false;

        /* Ethics gate — arm decisions are checked per-arm (biology: each
         * arm has local reflex-level "don't touch that" reactions). */
        if (ctx->ethics_hook) {
            if (!ctx->ethics_hook(&ctx->arms[a], ctx->ethics_user)) {
                ctx->arms[a].vetoed = true;
                ctx->stats.n_ethics_vetoes++;
                if (ctx->bio_hook) {
                    ctx->bio_hook("octopus_arm_vetoed",
                                  (float)a, ctx->bio_user);
                }
            }
        }

        /* Swarm delegation — highly confident arm findings get pushed to
         * an edge agent. Lower-confidence arm opinions stay local. */
        if (ctx->swarm_hook &&
            !ctx->arms[a].vetoed &&
            ctx->arms[a].confidence > ctx->swarm_delegation_threshold) {
            ctx->swarm_hook(&ctx->arms[a], ctx->swarm_user);
            ctx->stats.n_swarm_delegations++;
        }
    }

    ctx->stats.n_explorations++;
    return 0;
}

int octopus_integrate(octopus_system_t* ctx,
                       float* out, float* coherence) {
    if (!ctx || !out) return -1;

    /* Confidence-weighted mean of non-vetoed arms. */
    float weight_sum = 0.0f;
    float agg[OCTOPUS_ARM_DIM] = {0};
    uint32_t contributing = 0;
    float conf_sum = 0.0f;
    float var_sum = 0.0f;
    for (uint32_t a = 0; a < ctx->n_arms; a++) {
        if (ctx->arms[a].vetoed) continue;
        float w = ctx->arms[a].confidence;
        for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
            agg[d] += w * ctx->arms[a].latent[d];
        }
        weight_sum += w;
        conf_sum += ctx->arms[a].confidence;
        var_sum += ctx->arms[a].latent_variance;
        contributing++;
    }
    if (contributing == 0 || weight_sum < 1e-8f) {
        /* All arms vetoed or silent — zero output, low coherence. */
        memset(out, 0, sizeof(float) * OCTOPUS_ARM_DIM);
        if (coherence) *coherence = 0.0f;
        ctx->last_coherence = 0.0f;
        return -1;
    }
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
        agg[d] /= weight_sum;
        out[d] = agg[d];
    }
    memcpy(ctx->last_aggregated, out, sizeof(ctx->last_aggregated));

    /* Central coherence: 1 - mean-pairwise divergence of arm latents.
     * Cheap proxy: 1 - variance of element-wise means across arms. */
    float per_elem_var = 0.0f;
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
        float mean = 0.0f; uint32_t n = 0;
        for (uint32_t a = 0; a < ctx->n_arms; a++) {
            if (ctx->arms[a].vetoed) continue;
            mean += ctx->arms[a].latent[d]; n++;
        }
        if (n == 0) continue;
        mean /= (float)n;
        float v = 0.0f;
        for (uint32_t a = 0; a < ctx->n_arms; a++) {
            if (ctx->arms[a].vetoed) continue;
            float diff = ctx->arms[a].latent[d] - mean;
            v += diff * diff;
        }
        v /= (float)n;
        per_elem_var += v;
    }
    per_elem_var /= (float)OCTOPUS_ARM_DIM;
    float coh = nimcp_clampf(1.0f - per_elem_var, 0.0f, 1.0f);
    ctx->last_coherence = coh;
    if (coherence) *coherence = coh;

    /* Fire downstream hooks. */
    if (ctx->world_hook) {
        ctx->world_hook(out, OCTOPUS_ARM_DIM, ctx->world_user);
        ctx->stats.n_world_model_updates++;
    }
    if (ctx->fep_hook) {
        ctx->fep_hook(coh, ctx->fep_user);
    }
    if (ctx->bio_hook && coh < ctx->low_coherence_threshold) {
        ctx->bio_hook("octopus_low_coherence", coh, ctx->bio_user);
    }

    /* Update running stats. */
    ctx->stats.n_integrations++;
    ctx->stats.avg_arm_confidence = conf_sum / (float)contributing;
    ctx->stats.avg_arm_variance   = var_sum  / (float)contributing;
    ctx->stats.central_coherence  = coh;

    return 0;
}

void octopus_get_stats(const octopus_system_t* ctx, octopus_stats_t* out) {
    if (!ctx || !out) return;
    *out = ctx->stats;
}

/*============================================================================
 * Hook setters
 *==========================================================================*/

#define HOOK_SETTER(name, ftype)                                               \
    void octopus_set_##name##_hook(octopus_system_t* ctx,                      \
                                    ftype fn, void* user) {                    \
        if (!ctx) return;                                                      \
        ctx->name##_hook = fn;                                                 \
        ctx->name##_user = user;                                               \
    }

HOOK_SETTER(ethics, octopus_ethics_hook_t)
HOOK_SETTER(swarm,  octopus_swarm_hook_t)
HOOK_SETTER(world,  octopus_world_hook_t)
HOOK_SETTER(fep,    octopus_fep_hook_t)
HOOK_SETTER(bio,    octopus_bio_hook_t)
HOOK_SETTER(immune, octopus_immune_hook_t)

#undef HOOK_SETTER

/*============================================================================
 * Read-only accessors
 *==========================================================================*/

uint32_t octopus_get_n_arms(const octopus_system_t* ctx) {
    return ctx ? ctx->n_arms : 0;
}

const octopus_arm_t* octopus_get_arm(const octopus_system_t* ctx,
                                      uint32_t arm_id) {
    if (!ctx || arm_id >= ctx->n_arms) return NULL;
    return &ctx->arms[arm_id];
}

float octopus_get_broadcast_state(const octopus_system_t* ctx,
                                   uint32_t arm_id) {
    if (!ctx || arm_id >= ctx->n_arms) return 0.0f;
    return ctx->arms[arm_id].broadcast_state;
}
