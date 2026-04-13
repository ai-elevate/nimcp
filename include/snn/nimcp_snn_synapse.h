/**
 * @file nimcp_snn_synapse.h
 * @brief Lightweight CSR (Compressed Sparse Row) synapse storage for SNN
 *
 * WHAT: Cache-friendly synapse storage that replaces the heavy neuron_t
 *       sparse_synapse_storage_t (12KB per neuron) with a compact CSR
 *       format (~12 bytes per synapse).
 * WHY:  neuron_t costs 14KB each; 1.8M SNN neurons = 25 GB. CSR reduces
 *       this to ~1 GB for the same connectivity.
 * HOW:  Per-population CSR stores incoming synapses. For neuron i,
 *       synapses are at entries[row_ptr[i]..row_ptr[i+1]).
 *       Built in COO (coordinate) mode, then finalized to CSR.
 */

#ifndef NIMCP_SNN_SYNAPSE_H
#define NIMCP_SNN_SYNAPSE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Single incoming synapse in CSR format (12 bytes)
 */
typedef struct {
    uint32_t src_pop;       /**< Source population index in network */
    uint32_t src_neuron;    /**< Neuron index within source population */
    float    weight;        /**< Synaptic weight (mutable for learning) */
} snn_csr_synapse_t;

/**
 * @brief CSR incoming synapse storage for one population.
 *
 * Two modes:
 * - COO (build) mode: entries appended unsorted, row_ptr unused.
 *   Call snn_csr_finalize() to sort and build row_ptr.
 * - CSR (query) mode: entries sorted by dst neuron, row_ptr valid.
 *   snn_csr_get_incoming() returns O(1) per neuron.
 */
typedef struct snn_csr_storage_s {
    uint32_t*          row_ptr;    /**< [n_neurons + 1] row start indices */
    snn_csr_synapse_t* entries;    /**< [n_synapses] flat synapse array */
    uint32_t           n_neurons;  /**< Number of rows (neurons in this pop) */
    uint32_t           n_synapses; /**< Number of entries used */
    uint32_t           capacity;   /**< Allocated capacity of entries[] */
    bool               finalized;  /**< true after snn_csr_finalize() */

    /* GPU-ready flat arrays (populated by snn_csr_prepare_gpu) */
    float*             weights;         /**< [n_synapses] synapse weights (flat) */
    uint32_t*          flat_col_idx;    /**< [n_synapses] flat global neuron index */
    bool               gpu_ready;       /**< true after snn_csr_prepare_gpu() */
} snn_csr_storage_t;

/* ========================================================================= */
/* Lifecycle                                                                  */
/* ========================================================================= */

/**
 * @brief Create CSR storage for a population
 * @param n_neurons Number of neurons (rows)
 * @param estimated_synapses Initial capacity hint (grows if needed)
 * @return Storage handle, or NULL on error
 */
snn_csr_storage_t* snn_csr_create(uint32_t n_neurons, uint32_t estimated_synapses);

/**
 * @brief Destroy CSR storage and free all memory
 * @param csr Storage to destroy (NULL-safe)
 */
void snn_csr_destroy(snn_csr_storage_t* csr);

/* ========================================================================= */
/* Build phase (COO mode)                                                     */
/* ========================================================================= */

/**
 * @brief Append a synapse entry (COO mode — unsorted)
 * @param csr Storage (must not be finalized)
 * @param dst_neuron Destination neuron index within this population
 * @param src_pop Source population index
 * @param src_neuron Source neuron index within source population
 * @param weight Synaptic weight
 * @return 0 on success, -1 on error
 */
int snn_csr_add_entry(snn_csr_storage_t* csr,
                      uint32_t dst_neuron,
                      uint32_t src_pop,
                      uint32_t src_neuron,
                      float weight);

/**
 * @brief Sort entries by destination neuron and build row_ptr
 *
 * After this call, snn_csr_get_incoming() is valid.
 * No more entries can be added.
 *
 * @param csr Storage to finalize
 * @return 0 on success, -1 on error
 */
int snn_csr_finalize(snn_csr_storage_t* csr);

/* ========================================================================= */
/* Query phase (CSR mode — after finalize)                                    */
/* ========================================================================= */

/**
 * @brief Get incoming synapses for a neuron (O(1) lookup)
 * @param csr Finalized CSR storage
 * @param neuron_idx Neuron index within population
 * @param[out] count Number of incoming synapses
 * @return Pointer to first synapse entry, or NULL if none
 */
static inline snn_csr_synapse_t* snn_csr_get_incoming(
    const snn_csr_storage_t* csr,
    uint32_t neuron_idx,
    uint32_t* count)
{
    if (!csr || !csr->finalized || neuron_idx >= csr->n_neurons) {
        if (count) *count = 0;
        return NULL;
    }
    uint32_t start = csr->row_ptr[neuron_idx];
    uint32_t end   = csr->row_ptr[neuron_idx + 1];
    if (count) *count = end - start;
    return (end > start) ? &csr->entries[start] : NULL;
}

/* ========================================================================= */
/* GPU preparation (extract flat weight/index arrays for SpMV)                */
/* ========================================================================= */

/**
 * @brief Prepare GPU-friendly flat arrays from CSR entries.
 *
 * Extracts separate weights[] and flat_col_idx[] arrays from the
 * snn_csr_synapse_t entries. flat_col_idx maps (src_pop, src_neuron)
 * to a global flat index using the provided population offsets.
 *
 * @param csr Finalized CSR storage
 * @param pop_offsets [n_populations] flat offset for each population
 * @param n_populations Number of populations
 * @return 0 on success, -1 on error
 */
int snn_csr_prepare_gpu(snn_csr_storage_t* csr,
                        const uint32_t* pop_offsets,
                        uint32_t n_populations);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SNN_SYNAPSE_H */
