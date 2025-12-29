/**
 * @file nimcp_software_engineering.h
 * @brief Software engineering reasoning module for parietal lobe
 *
 * Implements software engineering reasoning capabilities:
 * - Algorithm complexity analysis (Big-O notation)
 * - Design pattern recognition
 * - Code structure analysis
 * - Dependency graph analysis
 * - Software metrics calculation
 *
 * BIOLOGICAL BASIS:
 * Software engineering reasoning combines mathematical intuition
 * (for complexity analysis), pattern detection (for design patterns),
 * and spatial reasoning (for code architecture visualization).
 *
 * USAGE:
 * ```c
 * software_eng_t* se = software_eng_create();
 *
 * // Analyze algorithm complexity
 * complexity_t complexity = software_eng_analyze_complexity(se, algorithm);
 *
 * // Detect design pattern
 * design_pattern_t pattern = software_eng_detect_pattern(se, code_structure);
 *
 * software_eng_destroy(se);
 * ```
 */

#ifndef NIMCP_SOFTWARE_ENGINEERING_H
#define NIMCP_SOFTWARE_ENGINEERING_H

#include "utils/validation/nimcp_common.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * CONSTANTS
 * ============================================================================ */

/** Maximum identifier length */
#define SWENG_MAX_IDENTIFIER            128

/** Maximum number of nodes in dependency graph */
#define SWENG_MAX_GRAPH_NODES           256

/** Maximum number of metrics */
#define SWENG_MAX_METRICS               32

/** Bio-async module ID for software engineering */
#define BIO_MODULE_SOFTWARE_ENG         0x0387

/* ============================================================================
 * TYPES
 * ============================================================================ */

/** Opaque handle for software engineering processor */
typedef struct software_eng software_eng_t;

/**
 * @brief Big-O complexity classes
 */
typedef enum {
    COMPLEXITY_O_1,             /**< O(1) - Constant */
    COMPLEXITY_O_LOG_N,         /**< O(log n) - Logarithmic */
    COMPLEXITY_O_N,             /**< O(n) - Linear */
    COMPLEXITY_O_N_LOG_N,       /**< O(n log n) - Linearithmic */
    COMPLEXITY_O_N_SQUARED,     /**< O(n^2) - Quadratic */
    COMPLEXITY_O_N_CUBED,       /**< O(n^3) - Cubic */
    COMPLEXITY_O_2_N,           /**< O(2^n) - Exponential */
    COMPLEXITY_O_N_FACTORIAL,   /**< O(n!) - Factorial */
    COMPLEXITY_UNKNOWN          /**< Unknown complexity */
} complexity_class_t;

/**
 * @brief Design pattern categories
 */
typedef enum {
    PATTERN_CATEGORY_CREATIONAL,
    PATTERN_CATEGORY_STRUCTURAL,
    PATTERN_CATEGORY_BEHAVIORAL,
    PATTERN_CATEGORY_UNKNOWN
} pattern_category_t;

/**
 * @brief Design pattern types
 */
typedef enum {
    /* Creational */
    DESIGN_PATTERN_SINGLETON,
    DESIGN_PATTERN_FACTORY,
    DESIGN_PATTERN_ABSTRACT_FACTORY,
    DESIGN_PATTERN_BUILDER,
    DESIGN_PATTERN_PROTOTYPE,
    /* Structural */
    DESIGN_PATTERN_ADAPTER,
    DESIGN_PATTERN_BRIDGE,
    DESIGN_PATTERN_COMPOSITE,
    DESIGN_PATTERN_DECORATOR,
    DESIGN_PATTERN_FACADE,
    DESIGN_PATTERN_FLYWEIGHT,
    DESIGN_PATTERN_PROXY,
    /* Behavioral */
    DESIGN_PATTERN_CHAIN_OF_RESPONSIBILITY,
    DESIGN_PATTERN_COMMAND,
    DESIGN_PATTERN_ITERATOR,
    DESIGN_PATTERN_MEDIATOR,
    DESIGN_PATTERN_MEMENTO,
    DESIGN_PATTERN_OBSERVER,
    DESIGN_PATTERN_STATE,
    DESIGN_PATTERN_STRATEGY,
    DESIGN_PATTERN_TEMPLATE_METHOD,
    DESIGN_PATTERN_VISITOR,
    /* Unknown */
    DESIGN_PATTERN_NONE
} design_pattern_t;

/**
 * @brief Code smell types
 */
typedef enum {
    SMELL_LONG_METHOD,
    SMELL_LARGE_CLASS,
    SMELL_FEATURE_ENVY,
    SMELL_DATA_CLUMPS,
    SMELL_PRIMITIVE_OBSESSION,
    SMELL_SWITCH_STATEMENTS,
    SMELL_PARALLEL_INHERITANCE,
    SMELL_LAZY_CLASS,
    SMELL_SPECULATIVE_GENERALITY,
    SMELL_DEAD_CODE,
    SMELL_DUPLICATE_CODE,
    SMELL_GOD_CLASS,
    SMELL_NONE
} code_smell_t;

/**
 * @brief Complexity analysis result
 */
typedef struct {
    complexity_class_t time_complexity;     /**< Time complexity class */
    complexity_class_t space_complexity;    /**< Space complexity class */
    float time_coefficient;                 /**< Estimated coefficient */
    float space_coefficient;                /**< Space coefficient */
    float confidence;                       /**< Confidence in analysis [0,1] */
    char description[256];                  /**< Human-readable description */
} complexity_result_t;

/**
 * @brief Algorithm characteristics for complexity analysis
 */
typedef struct {
    bool has_loops;                         /**< Contains loops */
    uint32_t loop_depth;                    /**< Maximum nested loop depth */
    bool has_recursion;                     /**< Uses recursion */
    uint32_t recursive_calls;               /**< Number of recursive calls */
    bool has_divide_conquer;                /**< Uses divide and conquer */
    bool has_dynamic_programming;           /**< Uses dynamic programming */
    bool has_sorting;                       /**< Contains sorting */
    bool has_searching;                     /**< Contains searching */
    uint32_t input_size_n;                  /**< Primary input size parameter */
    uint32_t auxiliary_space;               /**< Auxiliary space used */
} algorithm_traits_t;

/**
 * @brief Code structure for pattern detection
 */
typedef struct {
    uint32_t num_classes;                   /**< Number of classes/modules */
    uint32_t num_interfaces;                /**< Number of interfaces */
    uint32_t num_methods;                   /**< Total methods */
    bool has_single_instance;               /**< Singleton-like behavior */
    bool has_factory_method;                /**< Factory method present */
    bool has_abstract_base;                 /**< Abstract base class */
    bool has_composition;                   /**< Uses composition */
    bool has_inheritance;                   /**< Uses inheritance */
    bool has_callbacks;                     /**< Uses callbacks/observers */
    bool has_strategy_interface;            /**< Swappable strategies */
    float coupling_score;                   /**< Coupling metric [0,1] */
    float cohesion_score;                   /**< Cohesion metric [0,1] */
} code_structure_t;

/**
 * @brief Pattern detection result
 */
typedef struct {
    design_pattern_t pattern;               /**< Detected pattern */
    pattern_category_t category;            /**< Pattern category */
    float confidence;                       /**< Detection confidence [0,1] */
    char description[256];                  /**< Pattern description */
    char recommendation[256];               /**< Usage recommendation */
} pattern_result_t;

/**
 * @brief Dependency graph node
 */
typedef struct {
    uint32_t id;                            /**< Node ID */
    char name[SWENG_MAX_IDENTIFIER];        /**< Module/class name */
    uint32_t dependencies[SWENG_MAX_GRAPH_NODES]; /**< Dependency IDs */
    uint32_t num_dependencies;              /**< Number of dependencies */
    uint32_t dependents_count;              /**< How many depend on this */
} dep_node_t;

/**
 * @brief Dependency graph
 */
typedef struct {
    dep_node_t nodes[SWENG_MAX_GRAPH_NODES]; /**< Graph nodes */
    uint32_t num_nodes;                     /**< Number of nodes */
    bool has_cycles;                        /**< Contains circular dependencies */
    uint32_t max_depth;                     /**< Maximum dependency depth */
    float stability_avg;                    /**< Average stability metric */
} dep_graph_t;

/**
 * @brief Software metrics
 */
typedef struct {
    /* Size metrics */
    uint32_t lines_of_code;                 /**< Total LOC */
    uint32_t num_functions;                 /**< Number of functions */
    uint32_t num_classes;                   /**< Number of classes */
    uint32_t num_modules;                   /**< Number of modules */

    /* Complexity metrics */
    float cyclomatic_complexity;            /**< McCabe's cyclomatic complexity */
    float cognitive_complexity;             /**< Cognitive complexity */
    float halstead_difficulty;              /**< Halstead difficulty */
    float halstead_effort;                  /**< Halstead effort */

    /* Coupling/Cohesion */
    float afferent_coupling;                /**< Ca - incoming dependencies */
    float efferent_coupling;                /**< Ce - outgoing dependencies */
    float instability;                      /**< I = Ce / (Ca + Ce) */
    float abstractness;                     /**< A = abstract / total */

    /* Quality metrics */
    float maintainability_index;            /**< Maintainability index */
    float test_coverage;                    /**< Test coverage percentage */
    float documentation_ratio;              /**< Documentation ratio */
} software_metrics_t;

/**
 * @brief Code smell detection result
 */
typedef struct {
    code_smell_t smell;                     /**< Detected smell */
    float severity;                         /**< Severity [0,1] */
    char location[SWENG_MAX_IDENTIFIER];    /**< Location in code */
    char suggestion[256];                   /**< Refactoring suggestion */
} smell_result_t;

/**
 * @brief Software engineering configuration
 */
typedef struct {
    float complexity_threshold;             /**< Threshold for high complexity */
    float coupling_threshold;               /**< Threshold for high coupling */
    uint32_t method_length_limit;           /**< Max method length */
    uint32_t class_size_limit;              /**< Max class size */
    bool strict_mode;                       /**< Strict analysis mode */
    bool enable_bio_async;                  /**< Enable bio-async messaging */
    float inflammation_sensitivity;         /**< Immune modulation sensitivity */
    float sleep_deprivation_factor;         /**< Sleep modulation factor */
} software_eng_config_t;

/**
 * @brief Software engineering statistics
 */
typedef struct {
    uint64_t complexity_analyses;           /**< Complexity analyses performed */
    uint64_t pattern_detections;            /**< Pattern detections */
    uint64_t smell_detections;              /**< Smell detections */
    uint64_t metric_calculations;           /**< Metric calculations */
    float avg_processing_time_us;           /**< Average processing time */
} software_eng_stats_t;

/* ============================================================================
 * LIFECYCLE API
 * ============================================================================ */

/**
 * @brief Create software engineering processor with default configuration
 *
 * @return Handle or NULL on error
 */
software_eng_t* software_eng_create(void);

/**
 * @brief Create software engineering processor with custom configuration
 *
 * @param config Configuration (NULL for defaults)
 * @return Handle or NULL on error
 */
software_eng_t* software_eng_create_custom(const software_eng_config_t* config);

/**
 * @brief Destroy software engineering processor
 *
 * @param se Handle (NULL safe)
 */
void software_eng_destroy(software_eng_t* se);

/**
 * @brief Get default configuration
 *
 * @return Default configuration struct
 */
software_eng_config_t software_eng_default_config(void);

/**
 * @brief Validate configuration
 *
 * @param config Configuration to validate
 * @return true if valid
 */
bool software_eng_validate_config(const software_eng_config_t* config);

/* ============================================================================
 * COMPLEXITY ANALYSIS API
 * ============================================================================ */

/**
 * @brief Analyze algorithm complexity from traits
 *
 * @param se Software engineering handle
 * @param traits Algorithm characteristics
 * @param result Output complexity result
 * @return 0 on success
 */
int software_eng_analyze_complexity(
    software_eng_t* se,
    const algorithm_traits_t* traits,
    complexity_result_t* result
);

/**
 * @brief Get complexity class description
 *
 * @param complexity Complexity class
 * @return Human-readable description
 */
const char* software_eng_complexity_to_string(complexity_class_t complexity);

/**
 * @brief Compare two complexity classes
 *
 * @param a First complexity
 * @param b Second complexity
 * @return -1 if a < b, 0 if equal, 1 if a > b
 */
int software_eng_compare_complexity(
    complexity_class_t a,
    complexity_class_t b
);

/**
 * @brief Estimate runtime for given input size
 *
 * @param se Software engineering handle
 * @param complexity Complexity class
 * @param coefficient Base coefficient
 * @param input_size Input size n
 * @return Estimated operations count
 */
double software_eng_estimate_runtime(
    const software_eng_t* se,
    complexity_class_t complexity,
    double coefficient,
    uint64_t input_size
);

/* ============================================================================
 * PATTERN DETECTION API
 * ============================================================================ */

/**
 * @brief Detect design pattern from code structure
 *
 * @param se Software engineering handle
 * @param structure Code structure description
 * @param result Output pattern result
 * @return 0 on success
 */
int software_eng_detect_pattern(
    software_eng_t* se,
    const code_structure_t* structure,
    pattern_result_t* result
);

/**
 * @brief Get pattern type name
 *
 * @param pattern Pattern type
 * @return Pattern name string
 */
const char* software_eng_pattern_to_string(design_pattern_t pattern);

/**
 * @brief Get pattern category name
 *
 * @param category Pattern category
 * @return Category name string
 */
const char* software_eng_category_to_string(pattern_category_t category);

/**
 * @brief Check if pattern is applicable to structure
 *
 * @param se Software engineering handle
 * @param pattern Pattern to check
 * @param structure Code structure
 * @return Applicability score [0,1]
 */
float software_eng_pattern_applicability(
    const software_eng_t* se,
    design_pattern_t pattern,
    const code_structure_t* structure
);

/* ============================================================================
 * CODE SMELL DETECTION API
 * ============================================================================ */

/**
 * @brief Detect code smells in metrics
 *
 * @param se Software engineering handle
 * @param metrics Software metrics
 * @param results Output smell results (array)
 * @param max_results Maximum results to return
 * @return Number of smells detected
 */
uint32_t software_eng_detect_smells(
    software_eng_t* se,
    const software_metrics_t* metrics,
    smell_result_t* results,
    uint32_t max_results
);

/**
 * @brief Get smell name
 *
 * @param smell Code smell type
 * @return Smell name string
 */
const char* software_eng_smell_to_string(code_smell_t smell);

/* ============================================================================
 * DEPENDENCY ANALYSIS API
 * ============================================================================ */

/**
 * @brief Initialize dependency graph
 *
 * @param graph Graph to initialize
 */
void software_eng_init_graph(dep_graph_t* graph);

/**
 * @brief Add node to dependency graph
 *
 * @param graph Dependency graph
 * @param name Node name
 * @return Node ID or -1 on error
 */
int software_eng_add_node(dep_graph_t* graph, const char* name);

/**
 * @brief Add dependency edge
 *
 * @param graph Dependency graph
 * @param from_id Source node ID
 * @param to_id Target node ID
 * @return 0 on success
 */
int software_eng_add_dependency(dep_graph_t* graph, uint32_t from_id, uint32_t to_id);

/**
 * @brief Detect cycles in dependency graph
 *
 * @param se Software engineering handle
 * @param graph Dependency graph
 * @return true if cycles exist
 */
bool software_eng_detect_cycles(software_eng_t* se, dep_graph_t* graph);

/**
 * @brief Calculate dependency metrics
 *
 * @param se Software engineering handle
 * @param graph Dependency graph
 * @return 0 on success
 */
int software_eng_analyze_dependencies(software_eng_t* se, dep_graph_t* graph);

/**
 * @brief Get topological sort order
 *
 * @param se Software engineering handle
 * @param graph Dependency graph
 * @param order Output order array (node IDs)
 * @param max_order Maximum order entries
 * @return Number of nodes in order, or -1 if cycles
 */
int software_eng_topological_sort(
    software_eng_t* se,
    const dep_graph_t* graph,
    uint32_t* order,
    uint32_t max_order
);

/* ============================================================================
 * METRICS API
 * ============================================================================ */

/**
 * @brief Calculate cyclomatic complexity
 *
 * @param edges Number of edges in control flow graph
 * @param nodes Number of nodes
 * @param components Number of connected components
 * @return Cyclomatic complexity
 */
float software_eng_cyclomatic_complexity(
    uint32_t edges,
    uint32_t nodes,
    uint32_t components
);

/**
 * @brief Calculate maintainability index
 *
 * @param halstead_volume Halstead volume
 * @param cyclomatic_complexity Cyclomatic complexity
 * @param loc Lines of code
 * @return Maintainability index [0, 100]
 */
float software_eng_maintainability_index(
    float halstead_volume,
    float cyclomatic_complexity,
    uint32_t loc
);

/**
 * @brief Calculate instability metric
 *
 * @param afferent_coupling Ca (incoming dependencies)
 * @param efferent_coupling Ce (outgoing dependencies)
 * @return Instability [0, 1]
 */
float software_eng_instability(float afferent_coupling, float efferent_coupling);

/* ============================================================================
 * MODULATION API
 * ============================================================================ */

/**
 * @brief Set inflammation level
 *
 * @param se Software engineering handle
 * @param level Inflammation level [0,1]
 * @return 0 on success
 */
int software_eng_set_inflammation(software_eng_t* se, float level);

/**
 * @brief Set sleep deprivation level
 *
 * @param se Software engineering handle
 * @param level Deprivation level [0,1]
 * @return 0 on success
 */
int software_eng_set_sleep_deprivation(software_eng_t* se, float level);

/* ============================================================================
 * STATISTICS API
 * ============================================================================ */

/**
 * @brief Get statistics
 *
 * @param se Software engineering handle
 * @param stats Output statistics
 * @return 0 on success
 */
int software_eng_get_stats(const software_eng_t* se, software_eng_stats_t* stats);

/**
 * @brief Reset statistics
 *
 * @param se Software engineering handle
 */
void software_eng_reset_stats(software_eng_t* se);

/**
 * @brief Get last error message
 *
 * @return Thread-local error message
 */
const char* software_eng_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* NIMCP_SOFTWARE_ENGINEERING_H */
