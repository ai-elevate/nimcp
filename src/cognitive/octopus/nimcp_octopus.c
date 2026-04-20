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

/* Phase 3a integrations: pink noise (biologically-plausible exploration),
 * fractal DFA (arm health metric), Shannon entropy (arm diversity). */
#include "plasticity/noise/nimcp_pink_noise.h"
#include "cognitive/memory/core/nimcp_fractal.h"

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

    /* Phase 3a: shared pink-noise generator used to inject biologically
     * plausible 1/f stochastic exploration into arm latents. Shared across
     * arms (single temporal noise source) since biological neuromod noise
     * is a whole-system signal, not per-arm. Lazy-created on first use;
     * if the noise module is unavailable, arms just skip the bias. */
    pink_noise_generator_t pink_gen;

    /* DFA cadence: recompute fractal_dfa on each arm's history at most
     * once per N explorations, since DFA is O(N * num_scales). */
    uint32_t dfa_update_every;
    uint32_t dfa_update_counter;
};

/*============================================================================
 * Small helpers — all scalar math routes through utils/math/nimcp_math_helpers
 * (nimcp_clampf, nimcp_clamp01, nimcp_safe_expf, etc.). Do not re-define.
 *==========================================================================*/

/* Tanh-activated per-element transform on an input slice → latent.
 * This is the arm's local "processing" — a cheap nonlinearity plus
 * a contribution from the arm's recent history. Full ML weights aren't
 * the point; this is cheap distributed-reflex compute, biology-analogous.
 *
 * Phase 3a: injects a single pink-noise sample into the pre-activation
 * (before tanh) so arms explore in a 1/f correlated way. The sample is
 * shared across dims of one arm for one step (simulates a single neuro-
 * modulator affecting the whole arm), then regenerated per exploration.
 * Shannon entropy of the softmaxed latent is computed at the end as a
 * principled info-theoretic diversity measure. */
static void _arm_process_slice(octopus_arm_t* arm, const float* slice,
                                pink_noise_generator_t pink_gen,
                                octopus_stats_t* stats) {
    /* Store new input in history ring. */
    memcpy(arm->recent_inputs[arm->history_head], slice,
           sizeof(float) * OCTOPUS_ARM_DIM);
    arm->history_head = (arm->history_head + 1) % OCTOPUS_ARM_HISTORY;

    /* Pink-noise bias — one scalar sample for this arm this step. */
    float pn_bias = 0.0f;
    if (pink_gen && pink_noise_generate_sample(pink_gen, &pn_bias)) {
        if (stats) stats->n_pink_noise_injections++;
    }

    /* Compute new latent — 70% current slice, 30% smoothed history mean,
     * + pink-noise bias. Still tanh-squashed into [-1, 1]. */
    float history_mean[OCTOPUS_ARM_DIM] = {0};
    for (uint32_t h = 0; h < OCTOPUS_ARM_HISTORY; h++) {
        for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
            history_mean[d] += arm->recent_inputs[h][d];
        }
    }
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
        history_mean[d] /= (float)OCTOPUS_ARM_HISTORY;
        float v = 0.7f * slice[d] + 0.3f * history_mean[d] + pn_bias;
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

    /* Phase 3a: Shannon entropy over the arm's softmaxed latent. Gives
     * an info-theoretic "certainty" measure complementing latent_variance.
     * Values in [0, log2(N)] — normalized to [0, 1] for downstream use. */
    float probs[OCTOPUS_ARM_DIM];
    float max_v = arm->latent[0];
    for (uint32_t d = 1; d < OCTOPUS_ARM_DIM; d++) {
        if (arm->latent[d] > max_v) max_v = arm->latent[d];
    }
    float sum_exp = 0.0f;
    for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
        /* Subtract max for numerical stability. */
        probs[d] = expf(arm->latent[d] - max_v);
        sum_exp += probs[d];
    }
    if (sum_exp > 1e-8f) {
        for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) probs[d] /= sum_exp;
        float H = nimcp_entropy(probs, OCTOPUS_ARM_DIM);
        /* Normalize by log(N) = ln(64) ≈ 4.158 so arm->shannon_entropy is in [0,1]. */
        arm->shannon_entropy = nimcp_clamp01(H / 4.158883f);
    } else {
        arm->shannon_entropy = 0.0f;
    }
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
    ctx->dfa_update_every = 20;  /* once per 20 explorations — cheap cadence */

    /* Create the shared pink-noise generator. 1/f spectrum (alpha=1),
     * small amplitude so it biases exploration without overwhelming the
     * deterministic signal. Non-fatal if creation fails. */
    pink_noise_config_t pink_cfg = {0};
    pink_cfg.alpha          = 1.0f;
    pink_cfg.amplitude      = 0.05f;
    pink_cfg.min_frequency  = 0.1f;
    pink_cfg.max_frequency  = 100.0f;
    pink_cfg.sample_rate    = 100.0f;
    pink_cfg.method         = PINK_NOISE_VOSS;  /* fast streaming method */
    pink_cfg.seed           = 0;  /* time-based */
    ctx->pink_gen = pink_noise_create(&pink_cfg);
    if (!ctx->pink_gen) {
        NIMCP_LOGGING_WARN("octopus_create: pink noise generator unavailable; "
                           "arms will run without 1/f exploration bias");
    }

    NIMCP_LOGGING_INFO("octopus_create: %u arms initialized "
                       "(pink-noise=%s, DFA cadence=%u)",
                       n_arms, ctx->pink_gen ? "live" : "off",
                       ctx->dfa_update_every);
    return ctx;
}

void octopus_destroy(octopus_system_t* ctx) {
    if (!ctx) return;
    if (ctx->pink_gen) {
        pink_noise_destroy(ctx->pink_gen);
        ctx->pink_gen = NULL;
    }
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
        _arm_process_slice(&ctx->arms[a], slice, ctx->pink_gen, &ctx->stats);
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

    /* Phase 3a: periodically run DFA on each arm's history. DFA expects a
     * 1-D temporal series; we use L2 norm per history slot → 16 samples.
     * fractal_dfa requires >= FRACTAL_MIN_SAMPLES (64), so with the
     * current OCTOPUS_ARM_HISTORY=16 the call will fail and we skip —
     * arm->dfa_exponent stays 0.0 (meaning "not yet computed"). Phase 3b
     * should bump OCTOPUS_ARM_HISTORY to 64+ so this metric becomes live.
     * The call is kept here so the infra is ready when history grows. */
    ctx->dfa_update_counter++;
    if (ctx->dfa_update_counter >= ctx->dfa_update_every) {
        ctx->dfa_update_counter = 0;
        for (uint32_t a = 0; a < ctx->n_arms; a++) {
            octopus_arm_t* arm = &ctx->arms[a];
            /* L2 norm per history time-slot = one sample per time step.
             * This is the *temporal* signal DFA needs — not a flatten
             * of multi-dim data, which would measure the wrong thing. */
            float signal[OCTOPUS_ARM_HISTORY];
            for (uint32_t h = 0; h < OCTOPUS_ARM_HISTORY; h++) {
                float sq = 0.0f;
                for (uint32_t d = 0; d < OCTOPUS_ARM_DIM; d++) {
                    sq += arm->recent_inputs[h][d] * arm->recent_inputs[h][d];
                }
                signal[h] = sqrtf(sq);
            }
            fractal_result_t res = {0};
            if (fractal_dfa(signal, OCTOPUS_ARM_HISTORY, NULL, &res) == 0) {
                arm->dfa_exponent = res.dfa_exponent;
                ctx->stats.n_dfa_computations++;
            }
            /* Else: call fails because OCTOPUS_ARM_HISTORY < FRACTAL_MIN_SAMPLES;
             * dfa_exponent keeps its previous value (0.0 on fresh arm). */
        }
    }
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

    /* Phase 3a aggregations. */
    float ent_sum = 0.0f, dfa_sum = 0.0f;
    uint32_t dfa_contrib = 0;
    for (uint32_t a = 0; a < ctx->n_arms; a++) {
        if (ctx->arms[a].vetoed) continue;
        ent_sum += ctx->arms[a].shannon_entropy;
        if (ctx->arms[a].dfa_exponent > 0.0f) {
            dfa_sum += ctx->arms[a].dfa_exponent;
            dfa_contrib++;
        }
    }
    ctx->stats.avg_arm_entropy = ent_sum / (float)contributing;
    ctx->stats.avg_arm_dfa     = dfa_contrib > 0
                               ? (dfa_sum / (float)dfa_contrib) : 0.0f;

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

void octopus_set_bridge_stats(octopus_system_t* ctx,
                               uint64_t engram_encodings,
                               uint64_t kg_nodes_added,
                               uint64_t stress_broadcasts,
                               uint64_t fear_broadcasts,
                               uint64_t amygdala_steps,
                               float    last_cortisol,
                               float    last_fear) {
    if (!ctx) return;
    ctx->stats.bridge_engram_encodings  = engram_encodings;
    ctx->stats.bridge_kg_nodes_added    = kg_nodes_added;
    ctx->stats.bridge_stress_broadcasts = stress_broadcasts;
    ctx->stats.bridge_fear_broadcasts   = fear_broadcasts;
    ctx->stats.bridge_amygdala_steps    = amygdala_steps;
    ctx->stats.bridge_last_cortisol     = last_cortisol;
    ctx->stats.bridge_last_fear         = last_fear;
}
