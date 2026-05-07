//=============================================================================
// nimcp_brain_init_language_pops.c - Cold-init of language + sensorymotor pops
//=============================================================================

#include "core/brain/factory/init/nimcp_brain_init_language_pops.h"
#include "core/brain/nimcp_brain.h"
#include "core/brain/nimcp_brain_internal.h"
#include "core/brain/regions/broca/nimcp_broca_adapter.h"
#include "core/brain/regions/wernicke/nimcp_wernicke_adapter.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "snn/nimcp_snn_network.h"
#include "snn/nimcp_snn_synapse.h"
#include "snn/nimcp_snn_types.h"
#include "snn/bridges/nimcp_snn_language_bridge.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>

#define LOG_MODULE "BRAIN_INIT_LANG_POPS"

/* Pop sizes — 64K/64K/32K language + 40K sensorymotor = 200K total
 * (brain 2.0M → 2.2M, ~17 GB VRAM on 32 GB RTX 5090, ~3.2 s/step). */
#define LANG_BROCA_POP_NEURONS        64000u
#define LANG_WERNICKE_POP_NEURONS     64000u
#define LANG_ARCUATE_POP_NEURONS      32000u
#define SENSORYMOTOR_RING_POP_NEURONS 40000u

#define LANG_BROCA_POP_NAME        "broca_substrate"
#define LANG_WERNICKE_POP_NAME     "wernicke_substrate"
#define LANG_ARCUATE_POP_NAME      "arcuate_relay"
#define SENSORYMOTOR_RING_POP_NAME "sensorymotor_ring"

/* Wiring weights — conservative so they don't perturb the carefully-tuned
 * 1.8M hierarchy. Cold pops get drive but stay below saturation; homeostasis
 * adapts upward as needed. */
#define LANG_W_MEAN  0.04f
#define LANG_W_STD   0.012f

/* Connectivity ratios. Synapse counts at these levels:
 *   tier_tap × pop = ~14K syn each (hierarchy → wernicke / broca → hierarchy)
 *   internal loop  = ~20K syn each (wernicke ↔ arcuate ↔ broca)
 *   synfire ring   = SENSORYMOTOR_RING_FAN_IN × 40K = 480K syn (8 stages × 5K)
 * Total new synapses ≈ 1.1M, well under the 50M connection budget. */
#define CONN_HIERARCHY_TAP   0.005f   /* L3/L4/L5/L6 ↔ language pops */
#define CONN_INTERNAL_FWD    0.010f   /* feed-forward in language loop */
#define CONN_INTERNAL_FB     0.005f   /* feedback in language loop */
#define CONN_SENSORY_TO_EXEC 0.005f   /* sensorymotor → L5_exec */

/* Sensorymotor synfire ring (Stage 2 — replaces the Stage 1 random-recurrent
 * placeholder).
 *
 * Architecture: the 40K-neuron pop is split into N_STAGES contiguous stages
 * of N_PER_STAGE neurons each. Each dst neuron in stage (i+1)%N_STAGES
 * receives FAN_IN incoming synapses sampled deterministically from stage i.
 * This forms a closed delay-line: a wave of activity entering at stage 0
 * propagates s0→s1→…→sN-1→s0 in sync with the SNN tick. The ring length
 * (N_STAGES × dt) defines the temporal-integration window.
 *
 * Why a synfire chain (not random recurrence): a delay-line preserves
 * sequence order, so downstream consumers (cerebellum/basal-ganglia, L5_exec)
 * can read recent sensorimotor history at known offsets. Random recurrence
 * acts as a reservoir but loses sequence — wrong primitive for motor planning.
 *
 * Determinism: the LCG seed is fixed so the topology is reproducible across
 * runs, which keeps checkpoint reload bit-identical to fresh init. */
#define SENSORYMOTOR_RING_N_STAGES         8u
#define SENSORYMOTOR_RING_FAN_IN           12u
#define SENSORYMOTOR_RING_W_MEAN_SCALE     0.7f  /* dampened relative to lang_w */
#define SENSORYMOTOR_RING_W_STD_SCALE      0.7f
#define SENSORYMOTOR_RING_LCG_SEED         0xCAFEF00DULL
#define SENSORYMOTOR_RING_LCG_MUL          6364136223846793005ULL
#define SENSORYMOTOR_RING_LCG_INC          1442695040888963407ULL

static int find_pop_id_by_name(snn_network_t* snn, const char* name) {
    if (!snn || !name) return -1;
    for (uint32_t i = 0; i < snn->n_populations; i++) {
        snn_population_t* p = snn->populations[i];
        if (p && strncmp(p->name, name, sizeof(p->name)) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static bool pop_already_exists(snn_network_t* snn, const char* name) {
    return find_pop_id_by_name(snn, name) >= 0;
}

static bool add_one_pop(snn_network_t* snn, uint32_t n, const char* name,
                        int* out_id) {
    *out_id = find_pop_id_by_name(snn, name);
    if (*out_id >= 0) {
        LOG_DEBUG(LOG_MODULE, "pop '%s' already present id=%d — skipping", name, *out_id);
        return true;
    }
    int id = snn_network_add_population_lightweight(snn, n, NEURON_GENERIC_LIF, name);
    if (id < 0) {
        LOG_WARN(LOG_MODULE, "failed to add pop '%s' (n=%u) — capacity %u/%u",
                 name, n, snn->n_populations, snn->populations_capacity);
        *out_id = -1;
        return false;
    }
    *out_id = id;
    LOG_INFO(LOG_MODULE, "added pop '%s' id=%d n=%u (cold, lightweight CSR)",
             name, id, n);
    return true;
}

/* Connect every pop whose name starts with `src_prefix` to `dst_id`.
 * Returns total synapses created (for accounting). */
static uint64_t connect_prefix_to_pop(snn_network_t* snn, const char* src_prefix,
                                       int dst_id, float conn,
                                       float w_mean, float w_std) {
    if (!snn || !src_prefix || dst_id < 0) return 0;
    size_t plen = strlen(src_prefix);
    uint64_t total = 0;
    uint32_t matched = 0;
    for (uint32_t i = 0; i < snn->n_populations; i++) {
        snn_population_t* p = snn->populations[i];
        if (!p) continue;
        if (strncmp(p->name, src_prefix, plen) != 0) continue;
        int nc = snn_network_connect_populations(snn, i, (uint32_t)dst_id,
            SNN_TOPO_RANDOM, conn, SYNAPSE_AMPA, w_mean, w_std);
        if (nc > 0) {
            total += (uint64_t)nc;
            matched++;
        }
    }
    LOG_DEBUG(LOG_MODULE, "connect '%s*' → pop %d: %u src pops, %llu synapses",
              src_prefix, dst_id, matched, (unsigned long long)total);
    return total;
}

/* Connect a single src pop to every pop whose name starts with `dst_prefix`. */
static uint64_t connect_pop_to_prefix(snn_network_t* snn, int src_id,
                                       const char* dst_prefix, float conn,
                                       float w_mean, float w_std) {
    if (!snn || src_id < 0 || !dst_prefix) return 0;
    size_t plen = strlen(dst_prefix);
    uint64_t total = 0;
    uint32_t matched = 0;
    for (uint32_t i = 0; i < snn->n_populations; i++) {
        snn_population_t* p = snn->populations[i];
        if (!p) continue;
        if (strncmp(p->name, dst_prefix, plen) != 0) continue;
        int nc = snn_network_connect_populations(snn, (uint32_t)src_id, i,
            SNN_TOPO_RANDOM, conn, SYNAPSE_AMPA, w_mean, w_std);
        if (nc > 0) {
            total += (uint64_t)nc;
            matched++;
        }
    }
    LOG_DEBUG(LOG_MODULE, "connect pop %d → '%s*': %u dst pops, %llu synapses",
              src_id, dst_prefix, matched, (unsigned long long)total);
    return total;
}

static uint64_t connect_pop_pair(snn_network_t* snn, int src_id, int dst_id,
                                  float conn, float w_mean, float w_std) {
    if (!snn || src_id < 0 || dst_id < 0) return 0;
    int nc = snn_network_connect_populations(snn,
        (uint32_t)src_id, (uint32_t)dst_id,
        SNN_TOPO_RANDOM, conn, SYNAPSE_AMPA, w_mean, w_std);
    return (nc > 0) ? (uint64_t)nc : 0;
}

/* Deterministic 64-bit LCG (PCG-style constants) — used for synfire sampling
 * so the ring topology is reproducible. Updates *state in place, returns
 * the current state (caller takes the bits it needs). */
static inline uint64_t lcg_next(uint64_t* state) {
    *state = (*state) * SENSORYMOTOR_RING_LCG_MUL + SENSORYMOTOR_RING_LCG_INC;
    return *state;
}

/* Sample a Gaussian-ish weight: w_mean + w_std × N(0,1), clamped >= 0
 * so we don't accidentally route AMPA-positive weights as inhibitory.
 * Box-Muller would be cleaner but a 12-sample sum works fine for a
 * ring whose absolute weight is already noisy. */
static inline float lcg_weight(uint64_t* state, float w_mean, float w_std) {
    /* Sum of 12 uniforms ≈ N(6, 1), shift to N(0,1). */
    float acc = 0.0f;
    for (int i = 0; i < 12; i++) {
        acc += (float)((lcg_next(state) >> 32) & 0xFFFFu) / 65536.0f;
    }
    float z = acc - 6.0f;
    float w = w_mean + w_std * z;
    return (w < 0.0f) ? 0.0f : w;
}

/* Wire an intra-pop synfire ring: stage i → stage (i+1) mod N_STAGES.
 *
 * The pop is split into N_STAGES contiguous stages. For each dst neuron in
 * stage (i+1), FAN_IN distinct random src neurons from stage i are wired in
 * via direct CSR add. We bypass snn_network_connect_populations because that
 * API only supports pop-level topology; intra-pop sub-range wiring requires
 * the lower-level CSR add. We compensate by manually setting the per-pop
 * synapse_type so the receptor-routing in the deposit kernel still works.
 *
 * Returns total synapses created. */
static uint64_t wire_synfire_ring_intra_pop(snn_network_t* snn,
                                             int pop_id,
                                             uint32_t n_stages,
                                             uint32_t fan_in,
                                             float w_mean,
                                             float w_std,
                                             uint64_t seed) {
    if (!snn || pop_id < 0 || n_stages == 0 || fan_in == 0) return 0;
    snn_population_t* pop = snn->populations[pop_id];
    if (!pop || !pop->incoming_csr) return 0;

    const uint32_t total_n      = pop->n_neurons;
    const uint32_t n_per_stage  = total_n / n_stages;
    if (n_per_stage == 0) return 0;
    if (fan_in > n_per_stage) {
        LOG_WARN(LOG_MODULE,
                 "synfire ring: fan_in=%u > n_per_stage=%u — clamping",
                 fan_in, n_per_stage);
        fan_in = n_per_stage;
    }

    /* Set receptor type for self-pop pair. The CSR deposit kernel reads
     * synapse_type_per_src[src_pop] to route into g_ampa/g_nmda/g_gaba_a/b.
     * Without this, AMPA-positive weights still route correctly via sign
     * fallback, but explicit assignment is clearer + matches what
     * snn_network_connect_populations would have done. */
    pop->synapse_type_per_src[(uint32_t)pop_id] = (uint8_t)SYNAPSE_AMPA;

    uint64_t state = seed;
    uint64_t total_syn = 0;

    /* Sampling without replacement is overkill for a synfire chain — duplicate
     * sources are rare at fan_in << n_per_stage. We use a small per-dst
     * "seen" buffer to dedupe explicitly so the actual fan-in matches the
     * requested value exactly. */
    uint32_t seen[256];  /* compile-time check below */
    if (fan_in > sizeof(seen) / sizeof(seen[0])) {
        LOG_WARN(LOG_MODULE, "synfire ring: fan_in=%u exceeds seen buffer", fan_in);
        return 0;
    }

    for (uint32_t i = 0; i < n_stages; i++) {
        uint32_t src_stage = i;
        uint32_t dst_stage = (i + 1u) % n_stages;
        uint32_t src_base  = src_stage * n_per_stage;
        uint32_t dst_base  = dst_stage * n_per_stage;

        for (uint32_t d = 0; d < n_per_stage; d++) {
            uint32_t dst_neuron = dst_base + d;
            uint32_t picked = 0;
            /* Bounded retry: if all 16 picks duplicate, give up and accept
             * fewer connections — keeps init O(fan_in × n_per_stage). */
            uint32_t guard = fan_in * 16u;
            while (picked < fan_in && guard > 0) {
                guard--;
                uint32_t off = (uint32_t)(lcg_next(&state) % n_per_stage);
                uint32_t src_neuron = src_base + off;
                bool dup = false;
                for (uint32_t k = 0; k < picked; k++) {
                    if (seen[k] == src_neuron) { dup = true; break; }
                }
                if (dup) continue;
                seen[picked++] = src_neuron;
                float w = lcg_weight(&state, w_mean, w_std);
                if (snn_csr_add_entry(pop->incoming_csr,
                                      dst_neuron,
                                      (uint32_t)pop_id,
                                      src_neuron,
                                      w) == 0) {
                    total_syn++;
                }
            }
        }
        LOG_DEBUG(LOG_MODULE,
                  "synfire ring: stage %u → stage %u wired (fan_in=%u, dst_count=%u)",
                  src_stage, dst_stage, fan_in, n_per_stage);
    }
    return total_syn;
}

/* Bind broca + wernicke adapters to their substrate pops, so the adapter
 * production / comprehension paths can read pop spike rates and write back
 * excitatory bias. Idempotent: safe to call repeatedly. */
static void attach_lang_adapters_to_substrate(brain_t brain,
                                              snn_network_t* snn,
                                              int broca_pop_id,
                                              int wernicke_pop_id,
                                              int arcuate_pop_id) {
    if (!brain) return;
    if (brain->broca && broca_pop_id >= 0) {
        if (!broca_attach_snn_pop(brain->broca, snn, broca_pop_id)) {
            LOG_WARN(LOG_MODULE, "broca_attach_snn_pop failed (pop_id=%d)",
                     broca_pop_id);
        }
    } else if (!brain->broca) {
        LOG_DEBUG(LOG_MODULE,
                  "broca adapter not yet created — skipping attach "
                  "(retry once adapter is initialized)");
    }
    if (brain->wernicke && wernicke_pop_id >= 0) {
        if (!wernicke_attach_snn_pop(brain->wernicke, snn, wernicke_pop_id)) {
            LOG_WARN(LOG_MODULE, "wernicke_attach_snn_pop failed (pop_id=%d)",
                     wernicke_pop_id);
        }
    } else if (!brain->wernicke) {
        LOG_DEBUG(LOG_MODULE,
                  "wernicke adapter not yet created — skipping attach "
                  "(retry once adapter is initialized)");
    }

    /* PA-3: also register pops with the SNN-language bridge so that
     * brain_tick_language can drain spike_output → STDP. Inert until the
     * caller flips bridge->config.enable_snn_spike_routing on.
     *
     * Roles:
     *   Broca   → WORD     (production tier — spikes routed via word_spike)
     *   Wernicke→ CONCEPT  (comprehension tier — spikes via concept_spike)
     *   Arcuate → CONCEPT  (relay between Wernicke ↔ Broca; spike traffic
     *                       carries comprehended-concept content forward to
     *                       production, so concept-side routing matches the
     *                       biological path. Tier 3 phantom-wiring activation
     *                       — previously created + wired internally but never
     *                       registered with the bridge, so its spikes never
     *                       reached STDP / binding learning.) */
    if (brain->snn_lang_bridge) {
        if (broca_pop_id >= 0) {
            (void)snn_language_bridge_attach_snn_pop(brain->snn_lang_bridge,
                                                      broca_pop_id,
                                                      LANG_BROCA_POP_NEURONS,
                                                      SNN_LANG_POP_ROLE_WORD);
        }
        if (wernicke_pop_id >= 0) {
            (void)snn_language_bridge_attach_snn_pop(brain->snn_lang_bridge,
                                                      wernicke_pop_id,
                                                      LANG_WERNICKE_POP_NEURONS,
                                                      SNN_LANG_POP_ROLE_CONCEPT);
        }
        if (arcuate_pop_id >= 0) {
            (void)snn_language_bridge_attach_snn_pop(brain->snn_lang_bridge,
                                                      arcuate_pop_id,
                                                      LANG_ARCUATE_POP_NEURONS,
                                                      SNN_LANG_POP_ROLE_CONCEPT);
        }
    }
}

bool nimcp_brain_factory_init_language_pops(brain_t brain) {
    if (!brain) {
        LOG_WARN(LOG_MODULE, "brain is NULL — skipping");
        return true; /* non-fatal */
    }
    if (!brain->snn_network) {
        LOG_DEBUG(LOG_MODULE, "snn_network not available — skipping (non-fatal)");
        return true;
    }

    snn_network_t* snn = brain->snn_network;

    /* Resume guard. Pop creation (add_one_pop) is idempotent — it returns
     * the existing id if the named pop is already present. The wiring stage
     * (Stage 1.b) is NOT idempotent: snn_network_connect_populations adds
     * fresh synapses on every call. Re-running on a checkpoint that already
     * has the wiring duplicates every connection and trips a CSR write past
     * its allocated bucket → SIGSEGV in snn_csr_add_entry on the second pass.
     *
     * Use OR-of-any (not AND-of-all): the SNN load can truncate part-way
     * through the lightweight-CSR section (known MFS rename race), leaving
     * some pops loaded and the rest missing. AND-of-all would treat that as
     * "fresh" and re-wire — double-wiring the loaded ones → SIGSEGV. OR-of-any
     * treats partial load as resume; missing pops stay missing (graceful
     * degradation), but we never crash from re-wiring already-wired pops. */
    bool prebuilt = pop_already_exists(snn, LANG_WERNICKE_POP_NAME) ||
                    pop_already_exists(snn, LANG_BROCA_POP_NAME) ||
                    pop_already_exists(snn, LANG_ARCUATE_POP_NAME) ||
                    pop_already_exists(snn, SENSORYMOTOR_RING_POP_NAME);
    if (prebuilt) {
        int wid = find_pop_id_by_name(snn, LANG_WERNICKE_POP_NAME);
        int bid = find_pop_id_by_name(snn, LANG_BROCA_POP_NAME);
        int aid = find_pop_id_by_name(snn, LANG_ARCUATE_POP_NAME);
        attach_lang_adapters_to_substrate(brain, snn, bid, wid, aid);
        LOG_INFO(LOG_MODULE,
                 "language pops already present (resume path) — skipping "
                 "wiring; adapters rebound to broca=%d wernicke=%d arcuate=%d",
                 bid, wid, aid);
        return true;
    }

    /* Stage 1.a: create the four pops (idempotent — bails on duplicates). */
    int wernicke_id = -1, broca_id = -1, arcuate_id = -1, sensorymotor_id = -1;
    bool ok = true;
    ok &= add_one_pop(snn, LANG_WERNICKE_POP_NEURONS,     LANG_WERNICKE_POP_NAME,     &wernicke_id);
    ok &= add_one_pop(snn, LANG_BROCA_POP_NEURONS,        LANG_BROCA_POP_NAME,        &broca_id);
    ok &= add_one_pop(snn, LANG_ARCUATE_POP_NEURONS,      LANG_ARCUATE_POP_NAME,      &arcuate_id);
    ok &= add_one_pop(snn, SENSORYMOTOR_RING_POP_NEURONS, SENSORYMOTOR_RING_POP_NAME, &sensorymotor_id);

    if (!ok) {
        LOG_WARN(LOG_MODULE,
                 "one or more language/sensorymotor pops failed to register — "
                 "check SNN_MAX_POPULATIONS (capacity=%u) — skipping wiring",
                 snn->populations_capacity);
        return false;
    }

    /* Stage 1.b: wiring.
     *
     * Architecture (matches biological language network):
     *
     *   audio cortex (via bio_router)
     *           │
     *           ▼
     *   L3_concept_*  ───┐
     *                    │ feedforward (random sparse)
     *   L4_integr_*  ────┴───►  Wernicke pop  ◄──┐ feedback
     *                                ▲           │
     *                                │ ff        │ fb
     *                                ▼           │
     *                          Arcuate relay ────┘
     *                                ▲           │
     *                                │ fb        │ ff
     *                                │           ▼
     *                                └─────► Broca pop
     *                                              │
     *                                              ▼  (motor route)
     *                                        L5_exec_*
     *                                              │
     *                                              ▼
     *                                        L6_project_*
     *
     *   sensor_hub (via bio_router)
     *           │
     *           ▼
     *   sensorymotor_ring  ── synfire chain (8 stages × 5K, fan_in=12,
     *           │              wraps s0→s1→…→s7→s0; preserves sequence
     *           ▼              order across N_STAGES × dt window)
     *   L5_exec_*
     */
    uint64_t syn_total = 0;

    /* Hierarchy → Wernicke (L3 + L4 drive comprehension substrate). */
    syn_total += connect_prefix_to_pop(snn, "L3_concept_", wernicke_id,
                                       CONN_HIERARCHY_TAP, LANG_W_MEAN, LANG_W_STD);
    syn_total += connect_prefix_to_pop(snn, "L4_integr_",  wernicke_id,
                                       CONN_HIERARCHY_TAP, LANG_W_MEAN, LANG_W_STD);

    /* Internal language loop: Wernicke ↔ Arcuate ↔ Broca. */
    syn_total += connect_pop_pair(snn, wernicke_id, arcuate_id,
                                  CONN_INTERNAL_FWD, LANG_W_MEAN, LANG_W_STD);
    syn_total += connect_pop_pair(snn, arcuate_id, wernicke_id,
                                  CONN_INTERNAL_FB,  LANG_W_MEAN, LANG_W_STD);
    syn_total += connect_pop_pair(snn, arcuate_id, broca_id,
                                  CONN_INTERNAL_FWD, LANG_W_MEAN, LANG_W_STD);
    syn_total += connect_pop_pair(snn, broca_id, arcuate_id,
                                  CONN_INTERNAL_FB,  LANG_W_MEAN, LANG_W_STD);

    /* Broca → motor route (L5_exec → L6_project). */
    syn_total += connect_pop_to_prefix(snn, broca_id, "L5_exec_",
                                       CONN_HIERARCHY_TAP, LANG_W_MEAN, LANG_W_STD);
    syn_total += connect_pop_to_prefix(snn, broca_id, "L6_project_",
                                       CONN_HIERARCHY_TAP, LANG_W_MEAN, LANG_W_STD);

    /* Sensorymotor synfire ring (Stage 2): 8 stages × 5K neurons, each stage
     * forwards to the next with FAN_IN incoming per dst. Wraps s7 → s0 to
     * close the delay-line. See SENSORYMOTOR_RING_* constants for rationale. */
    uint64_t ring_syn = wire_synfire_ring_intra_pop(
        snn, sensorymotor_id,
        SENSORYMOTOR_RING_N_STAGES,
        SENSORYMOTOR_RING_FAN_IN,
        LANG_W_MEAN * SENSORYMOTOR_RING_W_MEAN_SCALE,
        LANG_W_STD  * SENSORYMOTOR_RING_W_STD_SCALE,
        SENSORYMOTOR_RING_LCG_SEED);
    syn_total += ring_syn;
    LOG_INFO(LOG_MODULE,
             "sensorymotor synfire ring wired: %u stages × %u neurons, "
             "fan_in=%u, %llu synapses",
             SENSORYMOTOR_RING_N_STAGES,
             SENSORYMOTOR_RING_POP_NEURONS / SENSORYMOTOR_RING_N_STAGES,
             SENSORYMOTOR_RING_FAN_IN,
             (unsigned long long)ring_syn);

    /* Sensorymotor → L5_exec (action selection / motor planning input). */
    syn_total += connect_pop_to_prefix(snn, sensorymotor_id, "L5_exec_",
                                       CONN_SENSORY_TO_EXEC, LANG_W_MEAN, LANG_W_STD);

    /* Build CSR row_ptr for the new lightweight-pop incoming synapses. */
    (void)snn_network_finalize_connections(snn);

    /* Stage 1.c: bind the cognitive adapters to their substrate pops.
     * MUST run AFTER finalize so the pop is wired and ready for use.
     * Safe to call even if the adapters aren't created yet (retried by
     * idempotency in subsequent init_language_pops invocations). */
    attach_lang_adapters_to_substrate(brain, snn, broca_id, wernicke_id, arcuate_id);

    LOG_INFO(LOG_MODULE,
             "language + sensorymotor pops created and wired: "
             "+%u neurons (broca=%u, wernicke=%u, arcuate=%u, sensorymotor=%u), "
             "+%llu synapses (incl. synfire ring); adapters: broca=%s, wernicke=%s",
             LANG_BROCA_POP_NEURONS + LANG_WERNICKE_POP_NEURONS
                 + LANG_ARCUATE_POP_NEURONS + SENSORYMOTOR_RING_POP_NEURONS,
             LANG_BROCA_POP_NEURONS, LANG_WERNICKE_POP_NEURONS,
             LANG_ARCUATE_POP_NEURONS, SENSORYMOTOR_RING_POP_NEURONS,
             (unsigned long long)syn_total,
             brain->broca ? "attached" : "deferred",
             brain->wernicke ? "attached" : "deferred");
    return true;
}
