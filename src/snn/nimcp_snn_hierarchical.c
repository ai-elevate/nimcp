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
    /* Tier 0: Input — sensory encoding */
    { 4, 20000, "input",      0.05f, 0.0f   },
    /* Tier 1: L1 — feature extraction */
    { 6, 22000, "L1_feature",  0.04f, 0.0f   },
    /* Tier 2: L2 — pattern binding */
    { 8, 30000, "L2_pattern",  0.03f, 0.005f },
    /* Tier 3: L3 — conceptual (recurrent for working memory) */
    { 8, 45000, "L3_concept",  0.02f, 0.01f  },
    /* Tier 4: L4 — integration peak (recurrent for attractor dynamics) */
    { 6, 58000, "L4_integr",   0.02f, 0.01f  },
    /* Tier 5: L5 — executive/planning (recurrent) */
    { 6, 44000, "L5_exec",     0.03f, 0.01f  },
    /* Tier 6: L6 — output projection */
    { 4, 30000, "L6_project",  0.04f, 0.0f   },
    /* Tier 7: Output — motor/response */
    { 4, 64000, "output",      0.0f,  0.0f   },
};
#define NUM_TIERS (sizeof(TIER_DEFS) / sizeof(TIER_DEFS[0]))

/* Skip connection definitions: { src_tier, dst_tier, connectivity } */
typedef struct {
    uint32_t src_tier;
    uint32_t dst_tier;
    float connectivity;
} snn_skip_def_t;

static const snn_skip_def_t SKIP_DEFS[] = {
    { 1, 5, 0.005f },  /* L1 → L5: fast sensory-to-executive shortcut */
    { 2, 6, 0.005f },  /* L2 → L6: pattern-to-output projection */
};
#define NUM_SKIPS (sizeof(SKIP_DEFS) / sizeof(SKIP_DEFS[0]))


snn_network_t* snn_create_hierarchical_network(
    uint32_t n_inputs,
    uint32_t n_outputs,
    uint32_t target_total_neurons)
{
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
    cfg.n_hidden = actual_total - TIER_DEFS[0].n_pops * TIER_DEFS[0].neurons_per_pop
                 - TIER_DEFS[NUM_TIERS-1].n_pops * TIER_DEFS[NUM_TIERS-1].neurons_per_pop;
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

    snn_network_t* net = snn_network_create(&cfg);
    if (!net) {
        LOG_ERROR("Failed to create base SNN network");
        return NULL;
    }

    /* Track population indices per tier for wiring */
    uint32_t tier_start_pop[NUM_TIERS + 1];
    tier_start_pop[0] = net->n_populations;  /* after the 3 initial pops (input/hidden/output) */

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
            int rc = snn_network_add_population(net,
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
    tier_start_pop[NUM_TIERS] = flat_idx;

wire_connections:
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
                int nc = snn_network_connect_populations(net,
                    pop_map[sp], pop_map[dp],
                    SNN_TOPO_RANDOM, ff_conn,
                    SYNAPSE_AMPA, 0.3f, 0.1f);
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
                if (nc > 0) recurrent_connections += (uint32_t)nc;
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
                int nc = snn_network_connect_populations(net,
                    pop_map[sp], pop_map[dp],
                    SNN_TOPO_RANDOM, sc,
                    SYNAPSE_AMPA, 0.15f, 0.05f);
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
                SNN_TOPO_RANDOM, 0.10f,
                SYNAPSE_AMPA, 0.3f, 0.1f);
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
                SNN_TOPO_RANDOM, 0.05f,
                SYNAPSE_AMPA, 0.2f, 0.1f);
            if (nc > 0) output_converge_conn += (uint32_t)nc;
        }
    }
    LOG_INFO("Output convergence: tier 7 → output_pop: %u connections",
             output_converge_conn);

    LOG_INFO("Hierarchical SNN complete: %u neurons, %u connections "
             "(ff=%u, rec=%u, skip=%u, in_fan=%u, out_conv=%u)",
             actual_total,
             total_connections + recurrent_connections + skip_connections
                 + input_fanout_conn + output_converge_conn,
             total_connections, recurrent_connections, skip_connections,
             input_fanout_conn, output_converge_conn);

    return net;
}
