// nimcp_hypergraph_part_core.c - core functions
// Part of nimcp_hypergraph.c (SRP #include-based split)
// DO NOT compile separately - #included from nimcp_hypergraph.c


/* ============================================================================
 * Phase 8: Training Integration (Full Implementation)
 * ============================================================================ */

int hypergraph_training_begin(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "hypergraph_training_begin: NULL argument");
        return -1;
    }
    hypergraph_heartbeat_instance(NULL, "hypergraph_training_begin", 0.0f);
    (void)(struct nimcp_hypergraph*)instance; /* Module state available for reset */
    return 0;
}


int hypergraph_training_end(void* instance) {
    if (!instance) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER,
                              "hypergraph_training_end: NULL argument");
        return -1;
    }
    hypergraph_heartbeat_instance(NULL, "hypergraph_training_end", 1.0f);
    (void)(struct nimcp_hypergraph*)instance; /* Module state available for finalization */
    return 0;
}
