/**
 * @file nimcp_snn_hierarchical.c
 * @brief Hierarchical SNN creation for 1.8M-neuron primary brain
 *
 * WHAT: Creates a cortical-inspired population hierarchy with 46 populations
 *       across 8 tiers, from sensory input through conceptual processing to
 *       motor output.
 * WHY:  The SNN is the primary brain — 1.8M spiking neurons with local
 *       learning rules (R-STDP + dopamine modulation). This replaces the
 *       previous 768-neuron toy SNN.
 * HOW:  Builds populations tier-by-tier, then wires feedforward, recurrent,
 *       and skip connections between them using efficient direct-sampling
 *       connectivity (not O(n×m) all-pairs).
 *
 * Population Hierarchy:
 *   Input (4 pops, 80K)  → L1 (6 pops, 132K) → L2 (8 pops, 240K)
 *   → L3 (8 pops, 360K)  → L4 (6 pops, 348K) → L5 (6 pops, 264K)
 *   → L6 (4 pops, 120K)  → Output (4 pops, 256K)
 *   Total: 46 populations, 1,800,000 neurons
 *
 * @version 1.0
 * @date 2026-04-12
 */

#include "snn/nimcp_snn_network.h"
#include <sys/statvfs.h>  /* for disk-space guard before .snn cache write */
#include "snn/nimcp_snn_types.h"
#include "core/neuron_types/nimcp_neuron_types.h"
#include "core/synapse_types/nimcp_synapse_types.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>

#define LOG_MODULE "SNN_HIERARCHICAL"

/* Tier definitions: { n_populations, neurons_per_pop, name } */
typedef struct {
    uint32_t n_pops;
    uint32_t neurons_per_pop;
    const char* name;
    float ff_connectivity;    /* feedforward to next tier */
    float recurrent_connectivity; /* within-tier recurrent */
} snn_tier_def_t;

static const snn_tier_def_t TIER_DEFS[] = {
    /* Tier 0: Input — sensory encoding
     * Connectivity scaled for 125 GB RAM budget (~1B synapse cap).
     * Each inter-tier connection creates n_src × n_dst × connectivity synapses
     * PER population pair, summed across all src×dst pop combinations. */
    { 4, 20000, "input",      0.005f, 0.0f    },
    /* Tier 1: L1 — feature extraction */
    { 6, 22000, "L1_feature",  0.004f, 0.0f    },
    /* Tier 2: L2 — pattern binding */
    { 8, 30000, "L2_pattern",  0.003f, 0.0005f },
    /* Tier 3: L3 — conceptual (recurrent for working memory) */
    { 8, 45000, "L3_concept",  0.002f, 0.001f  },
    /* Tier 4: L4 — integration peak (recurrent for attractor dynamics) */
    { 6, 58000, "L4_integr",   0.002f, 0.001f  },
    /* Tier 5: L5 — executive/planning (recurrent) */
    { 6, 44000, "L5_exec",     0.003f, 0.001f  },
    /* Tier 6: L6 — output projection */
    { 4, 30000, "L6_project",  0.004f, 0.0f    },
    /* Tier 7: Output — motor/response */
    { 4, 64000, "output",      0.0f,  0.0f     },
};
#define NUM_TIERS (sizeof(TIER_DEFS) / sizeof(TIER_DEFS[0]))

/* Skip connection definitions: { src_tier, dst_tier, connectivity } */
typedef struct {
    uint32_t src_tier;
    uint32_t dst_tier;
    float connectivity;
} snn_skip_def_t;

static const snn_skip_def_t SKIP_DEFS[] = {
    { 1, 5, 0.0005f },  /* L1 → L5: fast sensory-to-executive shortcut */
    { 2, 6, 0.0005f },  /* L2 → L6: pattern-to-output projection */
};
#define NUM_SKIPS (sizeof(SKIP_DEFS) / sizeof(SKIP_DEFS[0]))

/* Input fan-out and output convergence wiring parameters */
/* Weight tuning history (for a 1.8M-neuron lightweight CSR SNN):
 *  0.3  (original): L1 fired 1-2/22K, L2+ all dead (V stuck at -65)
 *  1.2  (4x):       1.75M spikes/step (97% saturation) — runaway
 *  0.6  (2x):       input_0 saturates 20K/20K, L2+ 30K/30K (all firing)
 *  Input fanout dominates: input_pop fires 1023/1024 every step, so any
 *  non-tiny weight from input→tier0 causes downstream saturation.
 *  Tier-to-tier needs moderate weight because pre-spike activity is sparse.
 *  Split: small input fanout, moderate tier FF. */
#define SNN_INPUT_FANOUT_CONNECTIVITY   0.01f  /**< input_pop → tier 0 connectivity */
#define SNN_INPUT_FANOUT_WEIGHT_MEAN    0.08f  /**< Small: input_pop has 100% activity */
#define SNN_INPUT_FANOUT_WEIGHT_STD     0.03f
#define SNN_OUTPUT_CONVERGE_CONNECTIVITY 0.005f /**< tier 7 → output_pop connectivity */
#define SNN_OUTPUT_CONVERGE_WEIGHT_MEAN  0.25f  /**< Moderate — tier 7 sparser */
#define SNN_OUTPUT_CONVERGE_WEIGHT_STD   0.08f


/* Well-known SNN sidecar path from the checkpoint system.
 * brain.save(path) writes the SNN to path.snn — this is the canonical
 * location for the trained hierarchical SNN with CSR weights.
 * Also check the legacy separate cache path for backward compat. */
#define SNN_SIDECAR_PATH "checkpoints/athena/athena_immersive.bin.snn"
#define SNN_LEGACY_CACHE_PATH "checkpoints/athena/snn_hierarchical.bin"

snn_network_t* snn_create_hierarchical_network(
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint32_t target_total_neurons)
{
    /* Try to load cached SNN from the checkpoint sidecar (saves ~8 min of wiring).
     * The sidecar is updated every checkpoint cycle with trained weights. */
    {
        extern snn_network_t* snn_network_load(const char* path);
        const char* cache_paths[] = { SNN_SIDECAR_PATH, SNN_LEGACY_CACHE_PATH };
        for (int ci = 0; ci < 2; ci++) {
            snn_network_t* cached = snn_network_load(cache_paths[ci]);
            if (!cached) continue;
            if (cached->config.n_inputs == n_inputs &&
                cached->config.n_outputs == n_outputs) {
                LOG_INFO("Loaded cached hierarchical SNN from %s "
                         "(%u populations, skipped wiring)",
                         cache_paths[ci], cached->n_populations);
                return cached;
            }
            LOG_WARN("Cached SNN at %s mismatch (inputs=%u/%u outputs=%u/%u)",
                     cache_paths[ci], cached->config.n_inputs, n_inputs,
                     cached->config.n_outputs, n_outputs);
            extern void snn_network_destroy(snn_network_t*);
            snn_network_destroy(cached);
        }
    }

    /* Compute actual total from tier definitions */
    uint32_t actual_total = 0;
    for (uint32_t t = 0; t < NUM_TIERS; t++) {
        actual_total += TIER_DEFS[t].n_pops * TIER_DEFS[t].neurons_per_pop;
    }
    LOG_INFO("Creating hierarchical SNN: %u neurons across %zu tiers "
             "(target=%u, actual=%u)",
             actual_total, NUM_TIERS, target_total_neurons, actual_total);

    /* Count total populations */
    uint32_t total_pops = 0;
    for (uint32_t t = 0; t < NUM_TIERS; t++) {
        total_pops += TIER_DEFS[t].n_pops;
    }

    /* Create SNN with base config */
    snn_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.n_inputs = n_inputs;
    cfg.n_outputs = n_outputs;
    /* Lightweight mode: tier populations use CSR synapse storage, NOT
     * neural_network_t neurons. Set n_hidden = 0 so the base network only
     * allocates input_pop + output_pop (tiny: ~3K neurons × 14KB = 43KB).
     * The 1.8M tier neurons use ~1 GB CSR instead of ~25 GB neuron_t. */
    cfg.n_hidden = 0;
    cfg.dt = 1.0f;  /* 1ms timestep */
    cfg.v_rest = -65.0f;
    cfg.v_reset = -70.0f;
    cfg.v_thresh = -50.0f;
    cfg.tau_mem = 20.0f;
    cfg.tau_syn = 5.0f;
    cfg.t_ref = 2.0f;
    cfg.input_current_scale = 45.0f;
    cfg.learning_rate = 0.001f;
    cfg.train_mode = SNN_TRAIN_R_STDP;
    cfg.enable_stdp = true;
    cfg.encoder.time_window = 100.0f;  /* 100ms encoding window */
    cfg.decoder.time_window = 100.0f;  /* 100ms decoding window (rate-based) */

    snn_network_t* net = snn_network_create(&cfg);
    if (!net) {
        LOG_ERROR("Failed to create base SNN network");
        return NULL;
    }

    /* Track population indices per tier for wiring.
     * Initialize all entries to 0 so early goto wire_connections
     * leaves unset tiers with safe (empty) ranges. */
    uint32_t tier_start_pop[NUM_TIERS + 1];
    memset(tier_start_pop, 0, sizeof(tier_start_pop));

    /* Note: snn_network_create already created input/hidden/output populations.
     * We need to add the additional populations. The initial 3 populations
     * (from snn_network_create) serve as tier 0 input + tier 7 output stubs.
     * We'll add tiers 1-6 as additional hidden populations. */

    /* Actually, let's just use the initial populations as-is and add more.
     * The key is wiring them correctly. Record all population indices. */
    uint32_t pop_idx = net->n_populations;  /* next free population slot */
    uint32_t pop_map[128];  /* pop_map[flat_idx] = actual population index in network */
    uint32_t flat_idx = 0;

    /* Map initial populations (created by snn_network_create) */
    /* Pop 0 = input, Pop 1 = hidden, Pop 2 = output from default create */
    /* We'll re-use pop 0 as tier 0 pop 0, and add the rest */

    /* For simplicity: add ALL tier populations as new hidden populations,
     * ignoring the default 3. The default input/output pops are tiny stubs
     * that we can leave unused. The real work happens in our tier pops. */
    for (uint32_t t = 0; t < NUM_TIERS; t++) {
        tier_start_pop[t] = flat_idx;
        for (uint32_t p = 0; p < TIER_DEFS[t].n_pops; p++) {
            /* Add a new population with the tier's neuron count */
            neuron_type_t ntype = NEURON_GENERIC_LIF;
            char pop_name[64];
            snprintf(pop_name, sizeof(pop_name), "%s_%u",
                     TIER_DEFS[t].name, p);
            int rc = snn_network_add_population_lightweight(net,
                TIER_DEFS[t].neurons_per_pop,
                ntype, pop_name);

            if (rc < 0) {
                LOG_ERROR("Failed to add population %u (tier %s, pop %u/%u): rc=%d",
                          flat_idx, TIER_DEFS[t].name, p, TIER_DEFS[t].n_pops, rc);
                /* Continue with what we have — partial hierarchy is better than none */
                goto wire_connections;
            }
            pop_map[flat_idx] = (uint32_t)rc;  /* rc is the population index */
            flat_idx++;
        }
    }
wire_connections:
    tier_start_pop[NUM_TIERS] = flat_idx;  /* Always set, even on partial creation */
    LOG_INFO("Created %u populations (target %u). Wiring connections...", flat_idx, total_pops);

    /* Wire feedforward connections between adjacent tiers */
    uint32_t total_connections = 0;
    for (uint32_t t = 0; t + 1 < NUM_TIERS; t++) {
        float ff_conn = TIER_DEFS[t].ff_connectivity;
        if (ff_conn <= 0.0f) continue;

        /* Connect each pop in tier t to each pop in tier t+1 */
        for (uint32_t sp = tier_start_pop[t]; sp < tier_start_pop[t + 1]; sp++) {
            for (uint32_t dp = tier_start_pop[t + 1]; dp < tier_start_pop[t + 2]; dp++) {
                if (sp >= flat_idx || dp >= flat_idx) continue;
                /* Tier FF weight 0.4 — between dead (0.3) and saturated (0.6).
                 * Input fanout is separately tuned much lower (0.08) because
                 * input_pop has 100% activity. */
                int nc = snn_network_connect_populations(net,
                    pop_map[sp], pop_map[dp],
                    SNN_TOPO_RANDOM, ff_conn,
                    SYNAPSE_AMPA, 0.4f, 0.13f);
                if (nc > 0) total_connections += (uint32_t)nc;
            }
        }
    }
    LOG_INFO("Feedforward: %u connections", total_connections);

    /* Wire within-tier recurrent connections (L3-L5 for working memory) */
    uint32_t recurrent_connections = 0;
    for (uint32_t t = 0; t < NUM_TIERS; t++) {
        float rec_conn = TIER_DEFS[t].recurrent_connectivity;
        if (rec_conn <= 0.0f) continue;

        for (uint32_t sp = tier_start_pop[t]; sp < tier_start_pop[t + 1]; sp++) {
            for (uint32_t dp = tier_start_pop[t]; dp < tier_start_pop[t + 1]; dp++) {
                if (sp == dp) continue;  /* no self-connections */
                if (sp >= flat_idx || dp >= flat_idx) continue;
                /* Mix excitatory (80%) and inhibitory (20%) for balance */
                synapse_type_t type = ((sp + dp) % 5 == 0) ?
                    SYNAPSE_GABA_A : SYNAPSE_AMPA;
                float w_mean = (type == SYNAPSE_GABA_A) ? -0.2f : 0.2f;
                int nc = snn_network_connect_populations(net,
                    pop_map[sp], pop_map[dp],
                    SNN_TOPO_RANDOM, rec_conn,
                    type, w_mean, 0.05f);
                if (nc > 0) {
                    recurrent_connections += (uint32_t)nc;
                } else if (nc < 0) {
                    LOG_ERROR("Recurrent connect failed: pop %u→%u, tier %u, rc=%d",
                              pop_map[sp], pop_map[dp], t, nc);
                } else {
                    LOG_WARN("Recurrent connect returned 0: pop %u→%u (src=%u dst=%u neurons), "
                             "connectivity=%.6f, tier %u",
                             pop_map[sp], pop_map[dp],
                             net->populations[pop_map[sp]] ? net->populations[pop_map[sp]]->n_neurons : 0,
                             net->populations[pop_map[dp]] ? net->populations[pop_map[dp]]->n_neurons : 0,
                             rec_conn, t);
                }
            }
        }
    }
    LOG_INFO("Recurrent: %u connections", recurrent_connections);

    /* Wire skip connections */
    uint32_t skip_connections = 0;
    for (uint32_t s = 0; s < NUM_SKIPS; s++) {
        uint32_t st = SKIP_DEFS[s].src_tier;
        uint32_t dt = SKIP_DEFS[s].dst_tier;
        float sc = SKIP_DEFS[s].connectivity;
        if (st >= NUM_TIERS || dt >= NUM_TIERS) continue;

        for (uint32_t sp = tier_start_pop[st]; sp < tier_start_pop[st + 1]; sp++) {
            for (uint32_t dp = tier_start_pop[dt]; dp < tier_start_pop[dt + 1]; dp++) {
                if (sp >= flat_idx || dp >= flat_idx) continue;
                /* Skip weights 2x (was 0.15 → 0.3) */
                int nc = snn_network_connect_populations(net,
                    pop_map[sp], pop_map[dp],
                    SNN_TOPO_RANDOM, sc,
                    SYNAPSE_AMPA, 0.3f, 0.1f);
                if (nc > 0) skip_connections += (uint32_t)nc;
            }
        }
    }
    LOG_INFO("Skip: %u connections", skip_connections);

    /* Wire input_pop (pop 0) → tier 0 populations.
     * The base network's input_pop has n_inputs neurons (e.g. 1024).
     * Tier 0 has 4 × 20K = 80K neurons. We fan out from input_pop to
     * all tier 0 populations so each input drives multiple tier-0 neurons.
     * Connectivity 10% = each input neuron drives ~2000 tier-0 neurons. */
    uint32_t input_fanout_conn = 0;
    if (tier_start_pop[0] < flat_idx) {
        for (uint32_t dp = tier_start_pop[0]; dp < tier_start_pop[1] && dp < flat_idx; dp++) {
            int nc = snn_network_connect_populations(net,
                0,  /* pop 0 = input_pop */
                pop_map[dp],
                SNN_TOPO_RANDOM, SNN_INPUT_FANOUT_CONNECTIVITY,
                SYNAPSE_AMPA, SNN_INPUT_FANOUT_WEIGHT_MEAN,
                SNN_INPUT_FANOUT_WEIGHT_STD);
            if (nc > 0) input_fanout_conn += (uint32_t)nc;
        }
    }
    LOG_INFO("Input fan-out: input_pop → tier 0: %u connections", input_fanout_conn);

    /* Wire tier 7 populations → output_pop (pop 2).
     * Tier 7 has 4 × 64K = 256K neurons. The output_pop has n_outputs
     * neurons (e.g. 2048). We converge tier 7 activity into output_pop
     * so snn_network_get_outputs() captures the hierarchy's final state.
     * Connectivity 5% = each output neuron receives from ~12800 tier-7 neurons. */
    uint32_t output_converge_conn = 0;
    if (NUM_TIERS > 0 && tier_start_pop[NUM_TIERS - 1] < flat_idx) {
        for (uint32_t sp = tier_start_pop[NUM_TIERS - 1];
             sp < tier_start_pop[NUM_TIERS] && sp < flat_idx; sp++) {
            int nc = snn_network_connect_populations(net,
                pop_map[sp],
                2,  /* pop 2 = output_pop */
                SNN_TOPO_RANDOM, SNN_OUTPUT_CONVERGE_CONNECTIVITY,
                SYNAPSE_AMPA, SNN_OUTPUT_CONVERGE_WEIGHT_MEAN,
                SNN_OUTPUT_CONVERGE_WEIGHT_STD);
            if (nc > 0) {
                output_converge_conn += (uint32_t)nc;
            } else {
                LOG_WARN("Output convergence returned %d: pop %u→2 (src=%u, dst=%u neurons)",
                         nc, pop_map[sp],
                         net->populations[pop_map[sp]] ? net->populations[pop_map[sp]]->n_neurons : 0,
                         net->output_pop ? net->output_pop->n_neurons : 0);
            }
        }
    }
    LOG_INFO("Output convergence: tier 7 → output_pop: %u connections",
             output_converge_conn);

    /* Finalize CSR storage on all lightweight populations.
     * This sorts the COO entries by destination neuron and builds
     * the row_ptr index for O(1) per-neuron lookup during stepping. */
    extern int snn_network_finalize_connections(snn_network_t* network);
    snn_network_finalize_connections(net);

    LOG_INFO("Hierarchical SNN complete: %u neurons, %u connections "
             "(ff=%u, rec=%u, skip=%u, in_fan=%u, out_conv=%u)",
             actual_total,
             total_connections + recurrent_connections + skip_connections
                 + input_fanout_conn + output_converge_conn,
             total_connections, recurrent_connections, skip_connections,
             input_fanout_conn, output_converge_conn);

    /* Save initial CSR to sidecar path so the first checkpoint has it.
     * Subsequent saves happen during brain.save() checkpoint cycle.
     * Disk guard: refuse if < 15 GB free (gzipped .snn is ~10-12 GB).
     * Without this check, a low-disk pod fills up trying to write
     * the cache and crashes the brain init. */
    {
        struct statvfs _st;
        if (statvfs("checkpoints/athena", &_st) == 0) {
            double free_gb = (double)_st.f_bavail * _st.f_frsize / (1024.0 * 1024.0 * 1024.0);
            if (free_gb < 15.0) {
                LOG_WARN("Skipping initial SNN sidecar save: only %.1f GB free "
                         "(need 15+). Brain will work but next restart re-wires SNN.",
                         free_gb);
            } else {
                extern int snn_network_save(snn_network_t* network, const char* path);
                if (snn_network_save(net, SNN_SIDECAR_PATH) == 0) {
                    LOG_INFO("Saved initial SNN to %s (subsequent saves via checkpoint cycle)",
                             SNN_SIDECAR_PATH);
                }
            }
        }
    }

    return net;
}
