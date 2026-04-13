/**
 * @file nimcp_snn_synapse.c
 * @brief CSR synapse storage implementation for lightweight SNN populations
 *
 * Build flow:
 * 1. snn_csr_create() — allocate storage
 * 2. snn_csr_add_entry() × N — append entries in any order (COO mode)
 * 3. snn_csr_finalize() — sort by dst neuron, build row_ptr (CSR mode)
 * 4. snn_csr_get_incoming() — O(1) lookup per neuron
 */

#include "snn/nimcp_snn_synapse.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/logging/nimcp_logging.h"
#include <string.h>
#include <stdlib.h>

#define LOG_MODULE "SNN_CSR"

/* COO entry with destination tag for sorting */
typedef struct {
    uint32_t dst_neuron;
    snn_csr_synapse_t syn;
} coo_tagged_entry_t;

/* ========================================================================= */
/* Lifecycle                                                                  */
/* ========================================================================= */

snn_csr_storage_t* snn_csr_create(uint32_t n_neurons, uint32_t estimated_synapses)
{
    if (n_neurons == 0) return NULL;
    if (estimated_synapses == 0) estimated_synapses = n_neurons * 10;

    snn_csr_storage_t* csr = nimcp_calloc(1, sizeof(snn_csr_storage_t));
    if (!csr) return NULL;

    csr->n_neurons = n_neurons;
    csr->n_synapses = 0;
    csr->capacity = estimated_synapses;
    csr->finalized = false;

    csr->row_ptr = nimcp_calloc(n_neurons + 1, sizeof(uint32_t));

    /* During COO phase, entries stores coo_tagged_entry_t (16 bytes each).
     * After finalize, entries stores snn_csr_synapse_t (12 bytes each).
     * Allocate for the larger of the two. */
    size_t entry_size = sizeof(coo_tagged_entry_t);  /* 16 bytes */
    csr->entries = nimcp_malloc(estimated_synapses * entry_size);

    if (!csr->row_ptr || !csr->entries) {
        snn_csr_destroy(csr);
        return NULL;
    }

    return csr;
}

void snn_csr_destroy(snn_csr_storage_t* csr)
{
    if (!csr) return;
    nimcp_free(csr->row_ptr);
    nimcp_free(csr->entries);
    nimcp_free(csr->weights);
    nimcp_free(csr->flat_col_idx);
    nimcp_free(csr);
}

/* ========================================================================= */
/* Build phase (COO mode — entries are coo_tagged_entry_t)                    */
/* ========================================================================= */

int snn_csr_add_entry(snn_csr_storage_t* csr,
                      uint32_t dst_neuron,
                      uint32_t src_pop,
                      uint32_t src_neuron,
                      float weight)
{
    if (!csr || csr->finalized) return -1;
    if (dst_neuron >= csr->n_neurons) return -1;

    /* Grow if needed */
    if (csr->n_synapses >= csr->capacity) {
        uint32_t new_cap = csr->capacity * 2;
        if (new_cap < csr->capacity + 4096) new_cap = csr->capacity + 4096;
        void* new_entries = nimcp_realloc(
            csr->entries, (size_t)new_cap * sizeof(coo_tagged_entry_t));
        if (!new_entries) return -1;
        csr->entries = new_entries;
        csr->capacity = new_cap;
    }

    /* Store as tagged COO entry */
    coo_tagged_entry_t* tagged = (coo_tagged_entry_t*)csr->entries;
    tagged[csr->n_synapses].dst_neuron = dst_neuron;
    tagged[csr->n_synapses].syn.src_pop = src_pop;
    tagged[csr->n_synapses].syn.src_neuron = src_neuron;
    tagged[csr->n_synapses].syn.weight = weight;
    csr->n_synapses++;

    return 0;
}

/* qsort comparator: sort by dst_neuron */
static int coo_compare(const void* a, const void* b)
{
    const coo_tagged_entry_t* ea = (const coo_tagged_entry_t*)a;
    const coo_tagged_entry_t* eb = (const coo_tagged_entry_t*)b;
    if (ea->dst_neuron < eb->dst_neuron) return -1;
    if (ea->dst_neuron > eb->dst_neuron) return 1;
    return 0;
}

int snn_csr_finalize(snn_csr_storage_t* csr)
{
    if (!csr || csr->finalized) return -1;

    uint32_t nnz = csr->n_synapses;

    if (nnz == 0) {
        /* Empty — just zero row_ptr */
        memset(csr->row_ptr, 0, (csr->n_neurons + 1) * sizeof(uint32_t));
        csr->finalized = true;
        return 0;
    }

    /* Step 1: Sort COO entries by dst_neuron */
    coo_tagged_entry_t* tagged = (coo_tagged_entry_t*)csr->entries;
    qsort(tagged, nnz, sizeof(coo_tagged_entry_t), coo_compare);

    /* Step 2: Build row_ptr from sorted entries */
    memset(csr->row_ptr, 0, (csr->n_neurons + 1) * sizeof(uint32_t));
    for (uint32_t i = 0; i < nnz; i++) {
        csr->row_ptr[tagged[i].dst_neuron + 1]++;
    }
    for (uint32_t i = 0; i < csr->n_neurons; i++) {
        csr->row_ptr[i + 1] += csr->row_ptr[i];
    }

    /* Step 3: Extract synapse data from tagged entries into compact format.
     * We can do this in-place since snn_csr_synapse_t (12 bytes) is smaller
     * than coo_tagged_entry_t (16 bytes), and entries are already sorted. */
    snn_csr_synapse_t* compact = (snn_csr_synapse_t*)csr->entries;
    for (uint32_t i = 0; i < nnz; i++) {
        compact[i] = tagged[i].syn;  /* Copy 12 bytes from 16-byte slot */
    }

    /* Step 4: Shrink allocation if significantly oversized */
    if (csr->capacity > nnz * 2 && nnz > 1024) {
        snn_csr_synapse_t* shrunk = nimcp_realloc(
            csr->entries, nnz * sizeof(snn_csr_synapse_t));
        if (shrunk) {
            csr->entries = shrunk;
            csr->capacity = nnz;
        }
    }

    csr->finalized = true;

    LOG_INFO("CSR finalized: %u neurons, %u synapses (%.1f avg/neuron)",
             csr->n_neurons, nnz, (float)nnz / csr->n_neurons);

    return 0;
}

int snn_csr_prepare_gpu(snn_csr_storage_t* csr,
                        const uint32_t* pop_offsets,
                        uint32_t n_populations)
{
    if (!csr || !csr->finalized || !pop_offsets) return -1;
    if (csr->gpu_ready) return 0;  /* Already prepared */

    uint32_t nnz = csr->n_synapses;
    if (nnz == 0) {
        csr->gpu_ready = true;
        return 0;
    }

    csr->weights = nimcp_malloc(nnz * sizeof(float));
    csr->flat_col_idx = nimcp_malloc(nnz * sizeof(uint32_t));
    if (!csr->weights || !csr->flat_col_idx) {
        nimcp_free(csr->weights);
        nimcp_free(csr->flat_col_idx);
        csr->weights = NULL;
        csr->flat_col_idx = NULL;
        return -1;
    }

    for (uint32_t i = 0; i < nnz; i++) {
        csr->weights[i] = csr->entries[i].weight;
        uint32_t sp = csr->entries[i].src_pop;
        uint32_t sn = csr->entries[i].src_neuron;
        csr->flat_col_idx[i] = (sp < n_populations) ?
            pop_offsets[sp] + sn : 0;
    }

    csr->gpu_ready = true;
    LOG_INFO("CSR GPU prepared: %u synapses, flat column indices built", nnz);
    return 0;
}
