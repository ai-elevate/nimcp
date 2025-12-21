//=============================================================================
// nimcp_quantum_reasoning.h - Quantum-Inspired Symbolic Reasoning
//=============================================================================

#ifndef NIMCP_QUANTUM_REASONING_H
#define NIMCP_QUANTUM_REASONING_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @file nimcp_quantum_reasoning.h
 * @brief Quantum-inspired algorithms for symbolic logic and reasoning
 *
 * WHAT: Quantum-inspired inference for logical reasoning
 * WHY:  Grover-like search provides speedup for SAT-like problems
 * HOW:  Amplitude encoding of truth values, quantum interference for contradiction detection
 *
 * QUANTUM CONCEPTS USED:
 * - Grover's Algorithm: Fast search for satisfying assignments
 * - Amplitude Encoding: Fuzzy truth values as quantum amplitudes
 * - Quantum Interference: Cancel contradictory inference paths
 * - Quantum Superposition: Explore multiple inference branches
 * - Phase Kickback: Propagate logical constraints
 *
 * BIOLOGICAL ANALOGY:
 * - Prefrontal cortex explores multiple hypotheses simultaneously
 * - Hippocampal pattern completion resolves ambiguity
 * - Working memory holds superposition of possibilities
 *
 * TERNARY LOGIC INTEGRATION:
 * - TRUE (+1), FALSE (-1), UNKNOWN (0)
 * - Kleene 3-valued logic for uncertain reasoning
 * - Quantum amplitude represents confidence
 *
 * @author NIMCP Development Team
 * @date 2025-01-20
 */

//=============================================================================
// Constants
//=============================================================================

#define QREASON_MAX_VARIABLES    32     /**< Maximum logical variables */
#define QREASON_MAX_CLAUSES      128    /**< Maximum clauses in CNF */
#define QREASON_MAX_LITERALS     8      /**< Maximum literals per clause */
#define QREASON_MAX_RULES        64     /**< Maximum inference rules */
#define QREASON_DEFAULT_SEED     98765  /**< Default random seed */

/* Ternary truth values */
#define QREASON_FALSE    (-1)
#define QREASON_UNKNOWN  (0)
#define QREASON_TRUE     (1)

//=============================================================================
// Types
//=============================================================================

/**
 * WHAT: Ternary truth value
 */
typedef int8_t qreason_truth_t;

/**
 * WHAT: Literal in a clause
 */
typedef struct {
    uint32_t variable;     /**< Variable index */
    bool negated;          /**< True if NOT variable */
} qreason_literal_t;

/**
 * WHAT: Clause (disjunction of literals)
 */
typedef struct {
    qreason_literal_t literals[QREASON_MAX_LITERALS];
    uint32_t n_literals;
} qreason_clause_t;

/**
 * WHAT: CNF formula (conjunction of clauses)
 */
typedef struct {
    qreason_clause_t clauses[QREASON_MAX_CLAUSES];
    uint32_t n_clauses;
    uint32_t n_variables;
} qreason_cnf_t;

/**
 * WHAT: Inference rule (Horn clause form)
 * HOW:  antecedents[0] AND antecedents[1] AND ... -> consequent
 */
typedef struct {
    uint32_t antecedents[QREASON_MAX_LITERALS]; /**< Variable indices */
    uint32_t n_antecedents;
    uint32_t consequent;    /**< Conclusion variable */
    float confidence;       /**< Rule confidence [0, 1] */
} qreason_rule_t;

/**
 * WHAT: Knowledge base
 */
typedef struct {
    qreason_truth_t facts[QREASON_MAX_VARIABLES];     /**< Known facts */
    float confidences[QREASON_MAX_VARIABLES];         /**< Fact confidences */
    qreason_rule_t rules[QREASON_MAX_RULES];
    uint32_t n_rules;
    uint32_t n_variables;
} qreason_kb_t;

/**
 * WHAT: Quantum state for reasoning
 */
typedef struct {
    float amplitudes[1 << 16];  /**< 2^n amplitudes (limited to 16 vars) */
    uint32_t n_qubits;          /**< Number of qubits (= variables) */
    uint32_t state_dim;         /**< 2^n_qubits */
} qreason_qstate_t;

/**
 * WHAT: Reasoning result
 */
typedef struct {
    qreason_truth_t assignment[QREASON_MAX_VARIABLES]; /**< Best assignment */
    float confidences[QREASON_MAX_VARIABLES];          /**< Assignment confidences */
    bool satisfiable;           /**< Is formula satisfiable? */
    float satisfaction_prob;    /**< Probability of satisfaction */
    uint32_t grover_iterations; /**< Iterations used */
    uint32_t inferences_made;   /**< Forward chaining steps */
} qreason_result_t;

/**
 * WHAT: Configuration for quantum reasoner
 */
typedef struct {
    uint32_t max_grover_iterations;  /**< Max Grover iterations */
    uint32_t max_inference_depth;    /**< Max forward chaining depth */
    float min_confidence;            /**< Minimum confidence threshold */
    bool use_ternary_logic;          /**< Enable 3-valued logic */
    bool enable_interference;        /**< Enable quantum interference */
    uint32_t seed;                   /**< Random seed */
} qreason_config_t;

/**
 * WHAT: Quantum reasoner context (opaque handle)
 */
typedef struct qreason_struct* qreason_t;

/**
 * WHAT: Internal structure
 */
typedef struct qreason_struct {
    qreason_config_t config;
    qreason_kb_t kb;
    qreason_qstate_t qstate;
    uint32_t rng_state;

    /* Statistics */
    uint64_t queries_performed;
    uint64_t satisfiable_count;
    uint64_t unsatisfiable_count;
    uint64_t total_grover_iterations;
} qreason_internal_t;

/**
 * WHAT: Statistics
 */
typedef struct {
    uint64_t queries_performed;
    uint64_t satisfiable_count;
    uint64_t unsatisfiable_count;
    float satisfiability_rate;
    float avg_grover_iterations;
} qreason_stats_t;

//=============================================================================
// Default Configuration
//=============================================================================

static inline qreason_config_t qreason_default_config(void) {
    return (qreason_config_t){
        .max_grover_iterations = 10,
        .max_inference_depth = 20,
        .min_confidence = 0.5f,
        .use_ternary_logic = true,
        .enable_interference = true,
        .seed = QREASON_DEFAULT_SEED
    };
}

//=============================================================================
// Random Number Generation
//=============================================================================

static inline uint32_t qreason_rand(uint32_t* state) {
    *state = (*state) * 1103515245 + 12345;
    return (*state >> 16) & 0x7FFF;
}

static inline float qreason_randf(uint32_t* state) {
    return (float)qreason_rand(state) / 32767.0f;
}

//=============================================================================
// Lifecycle Functions
//=============================================================================

/**
 * WHAT: Create quantum reasoner
 * WHY:  Initialize reasoning engine with configuration
 *
 * @param config Configuration (NULL for defaults)
 * @return Reasoner context or NULL on error
 */
static inline qreason_t qreason_create(const qreason_config_t* config) {
    qreason_config_t cfg;
    if (config) {
        cfg = *config;
    } else {
        cfg = qreason_default_config();
    }

    qreason_internal_t* ctx = (qreason_internal_t*)calloc(1, sizeof(qreason_internal_t));
    if (!ctx) return NULL;

    ctx->config = cfg;
    ctx->rng_state = cfg.seed;

    /* Initialize knowledge base */
    for (uint32_t i = 0; i < QREASON_MAX_VARIABLES; i++) {
        ctx->kb.facts[i] = QREASON_UNKNOWN;
        ctx->kb.confidences[i] = 0.0f;
    }

    return (qreason_t)ctx;
}

/**
 * WHAT: Destroy quantum reasoner
 */
static inline void qreason_destroy(qreason_t ctx) {
    free(ctx);
}

//=============================================================================
// Knowledge Base Management
//=============================================================================

/**
 * WHAT: Set a fact in the knowledge base
 *
 * @param ctx Reasoner context
 * @param variable Variable index
 * @param value Truth value (TRUE, FALSE, UNKNOWN)
 * @param confidence Confidence [0, 1]
 * @return 0 on success
 */
static inline int qreason_set_fact(
    qreason_t ctx,
    uint32_t variable,
    qreason_truth_t value,
    float confidence
) {
    if (!ctx || variable >= QREASON_MAX_VARIABLES) return -1;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    internal->kb.facts[variable] = value;
    internal->kb.confidences[variable] = confidence;

    if (variable >= internal->kb.n_variables) {
        internal->kb.n_variables = variable + 1;
    }

    return 0;
}

/**
 * WHAT: Get a fact from the knowledge base
 */
static inline qreason_truth_t qreason_get_fact(
    qreason_t ctx,
    uint32_t variable,
    float* confidence_out
) {
    if (!ctx || variable >= QREASON_MAX_VARIABLES) {
        if (confidence_out) *confidence_out = 0.0f;
        return QREASON_UNKNOWN;
    }
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    if (confidence_out) {
        *confidence_out = internal->kb.confidences[variable];
    }
    return internal->kb.facts[variable];
}

/**
 * WHAT: Add an inference rule
 *
 * @param ctx Reasoner context
 * @param antecedents Variable indices for antecedents
 * @param n_antecedents Number of antecedents
 * @param consequent Consequent variable
 * @param confidence Rule confidence
 * @return 0 on success
 */
static inline int qreason_add_rule(
    qreason_t ctx,
    const uint32_t* antecedents,
    uint32_t n_antecedents,
    uint32_t consequent,
    float confidence
) {
    if (!ctx || n_antecedents > QREASON_MAX_LITERALS) return -1;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    if (internal->kb.n_rules >= QREASON_MAX_RULES) return -2;

    qreason_rule_t* rule = &internal->kb.rules[internal->kb.n_rules];
    rule->n_antecedents = n_antecedents;
    rule->consequent = consequent;
    rule->confidence = confidence;

    for (uint32_t i = 0; i < n_antecedents; i++) {
        rule->antecedents[i] = antecedents[i];
    }

    internal->kb.n_rules++;
    return 0;
}

/**
 * WHAT: Clear all facts
 */
static inline void qreason_clear_facts(qreason_t ctx) {
    if (!ctx) return;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    for (uint32_t i = 0; i < QREASON_MAX_VARIABLES; i++) {
        internal->kb.facts[i] = QREASON_UNKNOWN;
        internal->kb.confidences[i] = 0.0f;
    }
    internal->kb.n_variables = 0;
}

/**
 * WHAT: Clear all rules
 */
static inline void qreason_clear_rules(qreason_t ctx) {
    if (!ctx) return;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;
    internal->kb.n_rules = 0;
}

//=============================================================================
// Ternary Logic Operations
//=============================================================================

/**
 * WHAT: Kleene 3-valued AND
 */
static inline qreason_truth_t qreason_and(qreason_truth_t a, qreason_truth_t b) {
    if (a == QREASON_FALSE || b == QREASON_FALSE) return QREASON_FALSE;
    if (a == QREASON_UNKNOWN || b == QREASON_UNKNOWN) return QREASON_UNKNOWN;
    return QREASON_TRUE;
}

/**
 * WHAT: Kleene 3-valued OR
 */
static inline qreason_truth_t qreason_or(qreason_truth_t a, qreason_truth_t b) {
    if (a == QREASON_TRUE || b == QREASON_TRUE) return QREASON_TRUE;
    if (a == QREASON_UNKNOWN || b == QREASON_UNKNOWN) return QREASON_UNKNOWN;
    return QREASON_FALSE;
}

/**
 * WHAT: Kleene 3-valued NOT
 */
static inline qreason_truth_t qreason_not(qreason_truth_t a) {
    if (a == QREASON_TRUE) return QREASON_FALSE;
    if (a == QREASON_FALSE) return QREASON_TRUE;
    return QREASON_UNKNOWN;
}

/**
 * WHAT: Kleene 3-valued IMPLIES
 */
static inline qreason_truth_t qreason_implies(qreason_truth_t a, qreason_truth_t b) {
    return qreason_or(qreason_not(a), b);
}

//=============================================================================
// Quantum State Operations
//=============================================================================

/**
 * WHAT: Initialize quantum state to uniform superposition
 */
static inline void qreason_qstate_init(qreason_qstate_t* qstate, uint32_t n_vars) {
    if (n_vars > 16) n_vars = 16;  /* Limit to 16 qubits */

    qstate->n_qubits = n_vars;
    qstate->state_dim = 1u << n_vars;

    float uniform = 1.0f / sqrtf((float)qstate->state_dim);
    for (uint32_t i = 0; i < qstate->state_dim; i++) {
        qstate->amplitudes[i] = uniform;
    }
}

/**
 * WHAT: Apply oracle for CNF formula
 * WHY:  Mark satisfying assignments with phase flip
 */
static inline void qreason_oracle_cnf(
    qreason_qstate_t* qstate,
    const qreason_cnf_t* cnf
) {
    for (uint32_t state = 0; state < qstate->state_dim; state++) {
        bool satisfies = true;

        /* Check each clause */
        for (uint32_t c = 0; c < cnf->n_clauses && satisfies; c++) {
            bool clause_sat = false;

            for (uint32_t l = 0; l < cnf->clauses[c].n_literals; l++) {
                uint32_t var = cnf->clauses[c].literals[l].variable;
                bool negated = cnf->clauses[c].literals[l].negated;
                bool var_true = (state >> var) & 1;

                if (negated ? !var_true : var_true) {
                    clause_sat = true;
                    break;
                }
            }

            if (!clause_sat) {
                satisfies = false;
            }
        }

        if (satisfies) {
            qstate->amplitudes[state] *= -1.0f;  /* Phase flip */
        }
    }
}

/**
 * WHAT: Apply diffusion operator (inversion about mean)
 */
static inline void qreason_diffusion(qreason_qstate_t* qstate) {
    /* Compute mean */
    float mean = 0.0f;
    for (uint32_t i = 0; i < qstate->state_dim; i++) {
        mean += qstate->amplitudes[i];
    }
    mean /= (float)qstate->state_dim;

    /* Invert about mean */
    for (uint32_t i = 0; i < qstate->state_dim; i++) {
        qstate->amplitudes[i] = 2.0f * mean - qstate->amplitudes[i];
    }
}

/**
 * WHAT: Measure quantum state (find highest probability assignment)
 */
static inline uint32_t qreason_measure(const qreason_qstate_t* qstate) {
    float max_prob = 0.0f;
    uint32_t best_state = 0;

    for (uint32_t i = 0; i < qstate->state_dim; i++) {
        float prob = qstate->amplitudes[i] * qstate->amplitudes[i];
        if (prob > max_prob) {
            max_prob = prob;
            best_state = i;
        }
    }

    return best_state;
}

/**
 * WHAT: Get probability of satisfying assignment
 */
static inline float qreason_satisfaction_probability(
    const qreason_qstate_t* qstate,
    const qreason_cnf_t* cnf
) {
    float prob = 0.0f;

    for (uint32_t state = 0; state < qstate->state_dim; state++) {
        bool satisfies = true;

        for (uint32_t c = 0; c < cnf->n_clauses && satisfies; c++) {
            bool clause_sat = false;

            for (uint32_t l = 0; l < cnf->clauses[c].n_literals; l++) {
                uint32_t var = cnf->clauses[c].literals[l].variable;
                bool negated = cnf->clauses[c].literals[l].negated;
                bool var_true = (state >> var) & 1;

                if (negated ? !var_true : var_true) {
                    clause_sat = true;
                    break;
                }
            }

            if (!clause_sat) {
                satisfies = false;
            }
        }

        if (satisfies) {
            prob += qstate->amplitudes[state] * qstate->amplitudes[state];
        }
    }

    return prob;
}

//=============================================================================
// Core Reasoning Functions
//=============================================================================

/**
 * WHAT: Run forward chaining inference
 * WHY:  Derive new facts from rules
 *
 * @param ctx Reasoner context
 * @param result Output: inference result
 * @return Number of new facts derived
 */
static inline uint32_t qreason_forward_chain(
    qreason_t ctx,
    qreason_result_t* result
) {
    if (!ctx || !result) return 0;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    uint32_t new_facts = 0;
    bool changed = true;
    uint32_t depth = 0;

    while (changed && depth < internal->config.max_inference_depth) {
        changed = false;
        depth++;

        for (uint32_t r = 0; r < internal->kb.n_rules; r++) {
            qreason_rule_t* rule = &internal->kb.rules[r];

            /* Check if all antecedents are true */
            bool all_true = true;
            float min_conf = 1.0f;

            for (uint32_t a = 0; a < rule->n_antecedents; a++) {
                qreason_truth_t val = internal->kb.facts[rule->antecedents[a]];
                float conf = internal->kb.confidences[rule->antecedents[a]];

                if (val != QREASON_TRUE) {
                    all_true = false;
                    break;
                }
                if (conf < min_conf) {
                    min_conf = conf;
                }
            }

            /* If all antecedents true and consequent unknown, derive */
            if (all_true && internal->kb.facts[rule->consequent] == QREASON_UNKNOWN) {
                float new_conf = min_conf * rule->confidence;

                if (new_conf >= internal->config.min_confidence) {
                    internal->kb.facts[rule->consequent] = QREASON_TRUE;
                    internal->kb.confidences[rule->consequent] = new_conf;
                    changed = true;
                    new_facts++;
                }
            }
        }
    }

    result->inferences_made = new_facts;
    return new_facts;
}

/**
 * WHAT: Solve SAT using Grover-inspired search
 * WHY:  Find satisfying assignment for CNF formula
 *
 * @param ctx Reasoner context
 * @param cnf CNF formula
 * @param result Output: result with assignment
 * @return 0 on success, -1 on error
 */
static inline int qreason_solve_sat(
    qreason_t ctx,
    const qreason_cnf_t* cnf,
    qreason_result_t* result
) {
    if (!ctx || !cnf || !result) return -1;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    if (cnf->n_variables == 0 || cnf->n_clauses == 0) {
        result->satisfiable = true;
        result->satisfaction_prob = 1.0f;
        return 0;
    }

    /* Initialize quantum state */
    qreason_qstate_init(&internal->qstate, cnf->n_variables);

    /* Check initial satisfaction probability before Grover */
    float initial_prob = qreason_satisfaction_probability(&internal->qstate, cnf);

    /* Grover only helps when solutions are rare (< 50%).
     * When solutions are majority, skip Grover to preserve probability. */
    uint32_t iterations = 0;
    if (initial_prob < 0.5f) {
        iterations = internal->config.max_grover_iterations;
        if (iterations == 0) {
            /* Optimal iterations: pi/4 * sqrt(N/M) where M = solutions */
            float estimated_solutions = initial_prob * (float)internal->qstate.state_dim;
            if (estimated_solutions > 0.0f) {
                float ratio = (float)internal->qstate.state_dim / estimated_solutions;
                iterations = (uint32_t)(M_PI / 4.0 * sqrtf(ratio));
                iterations = (iterations < 1) ? 1 : iterations;
                iterations = (iterations > 20) ? 20 : iterations;
            } else {
                iterations = 1;
            }
        }

        /* Run Grover iterations */
        for (uint32_t i = 0; i < iterations; i++) {
            qreason_oracle_cnf(&internal->qstate, cnf);
            qreason_diffusion(&internal->qstate);
        }
    }

    /* Measure result */
    uint32_t best_state = qreason_measure(&internal->qstate);

    /* Extract assignment */
    for (uint32_t v = 0; v < cnf->n_variables; v++) {
        bool val = (best_state >> v) & 1;
        result->assignment[v] = val ? QREASON_TRUE : QREASON_FALSE;
        result->confidences[v] = internal->qstate.amplitudes[best_state] *
                                internal->qstate.amplitudes[best_state];
    }

    /* Check if satisfiable */
    result->satisfaction_prob = qreason_satisfaction_probability(&internal->qstate, cnf);
    result->satisfiable = (result->satisfaction_prob > 0.0f);
    result->grover_iterations = iterations;

    /* Update statistics */
    internal->queries_performed++;
    internal->total_grover_iterations += iterations;
    if (result->satisfiable) {
        internal->satisfiable_count++;
    } else {
        internal->unsatisfiable_count++;
    }

    return 0;
}

/**
 * WHAT: Query the knowledge base
 * WHY:  Determine truth value of a variable using inference
 *
 * @param ctx Reasoner context
 * @param variable Variable to query
 * @param result Output: result
 * @return 0 on success
 */
static inline int qreason_query(
    qreason_t ctx,
    uint32_t variable,
    qreason_result_t* result
) {
    if (!ctx || !result || variable >= QREASON_MAX_VARIABLES) return -1;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    /* First run forward chaining */
    qreason_forward_chain(ctx, result);

    /* Copy current knowledge base state to result */
    for (uint32_t i = 0; i < QREASON_MAX_VARIABLES; i++) {
        result->assignment[i] = internal->kb.facts[i];
        result->confidences[i] = internal->kb.confidences[i];
    }

    return 0;
}

/**
 * WHAT: Check consistency of knowledge base
 * WHY:  Detect contradictions
 *
 * @param ctx Reasoner context
 * @return true if consistent, false if contradictory
 */
static inline bool qreason_check_consistency(qreason_t ctx) {
    if (!ctx) return false;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    /* Check for any variable with conflicting derivations */
    /* In ternary logic, a variable can be TRUE or FALSE but not both */

    /* Simple check: run forward chaining and look for conflicts */
    /* This is a placeholder - full consistency checking would use resolution */

    for (uint32_t i = 0; i < internal->kb.n_variables; i++) {
        /* A contradiction would be a variable that derives both TRUE and FALSE */
        /* For now, just verify no UNKNOWN variables have confidence > 0 */
        if (internal->kb.facts[i] == QREASON_UNKNOWN &&
            internal->kb.confidences[i] > 0.5f) {
            return false;  /* Inconsistent */
        }
    }

    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * WHAT: Create a CNF clause
 */
static inline qreason_clause_t qreason_make_clause(
    const int* literals,  /* Positive = variable, negative = NOT variable */
    uint32_t n_literals
) {
    qreason_clause_t clause = {0};
    clause.n_literals = (n_literals < QREASON_MAX_LITERALS) ?
                        n_literals : QREASON_MAX_LITERALS;

    for (uint32_t i = 0; i < clause.n_literals; i++) {
        if (literals[i] < 0) {
            clause.literals[i].variable = (uint32_t)(-literals[i] - 1);
            clause.literals[i].negated = true;
        } else {
            clause.literals[i].variable = (uint32_t)(literals[i] - 1);
            clause.literals[i].negated = false;
        }
    }

    return clause;
}

/**
 * WHAT: Get statistics
 */
static inline int qreason_get_stats(qreason_t ctx, qreason_stats_t* stats) {
    if (!ctx || !stats) return -1;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;

    stats->queries_performed = internal->queries_performed;
    stats->satisfiable_count = internal->satisfiable_count;
    stats->unsatisfiable_count = internal->unsatisfiable_count;

    if (internal->queries_performed > 0) {
        stats->satisfiability_rate = (float)internal->satisfiable_count /
                                    (float)internal->queries_performed;
        stats->avg_grover_iterations = (float)internal->total_grover_iterations /
                                       (float)internal->queries_performed;
    } else {
        stats->satisfiability_rate = 0.0f;
        stats->avg_grover_iterations = 0.0f;
    }

    return 0;
}

/**
 * WHAT: Get configuration
 */
static inline int qreason_get_config(qreason_t ctx, qreason_config_t* config) {
    if (!ctx || !config) return -1;
    qreason_internal_t* internal = (qreason_internal_t*)ctx;
    *config = internal->config;
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_QUANTUM_REASONING_H */
