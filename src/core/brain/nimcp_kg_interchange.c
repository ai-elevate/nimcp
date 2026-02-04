/**
 * @file nimcp_kg_interchange.c
 * @brief Import/Export & Interoperability for Brain Knowledge Graph
 * @version 1.0.0
 * @date 2026-01-16
 *
 * Implementation of KG import/export to various formats (JSON, RDF, GraphML, etc.)
 * and integration with external ontologies (WordNet, ConceptNet, Wikidata, etc.).
 */

#include "core/brain/nimcp_kg_interchange.h"
#include "utils/memory/nimcp_memory.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>
#include <time.h>
#include <stdio.h>
#include "utils/fault_tolerance/nimcp_health_agent_macros.h"
#include "mesh/nimcp_mesh_participant.h"
#include "mesh/nimcp_mesh_adapter.h"

NIMCP_DECLARE_HEALTH_AGENT_ATOMIC(kg_interchange)
//=============================================================================
// Mesh Participant Registration
//=============================================================================

static mesh_participant_id_t g_kg_interchange_mesh_id = 0;
static mesh_participant_registry_t* g_kg_interchange_mesh_registry = NULL;

nimcp_error_t kg_interchange_mesh_register(mesh_participant_registry_t* registry) {
    if (!registry) return NIMCP_ERROR_NULL_POINTER;
    if (g_kg_interchange_mesh_id != 0) return NIMCP_SUCCESS;
    mesh_participant_interface_t iface;
    mesh_participant_interface_init(&iface);
    strncpy(iface.module_name, "kg_interchange", MESH_MAX_NAME_LEN - 1);
    iface.type = MESH_PARTICIPANT_MODULE;
    iface.home_channel = mesh_adapter_get_default_channel(MESH_ADAPTER_CATEGORY_SYSTEM);
    mesh_participant_config_t config;
    mesh_participant_config_init(&config);
    config.module_name = "kg_interchange";
    config.type = MESH_PARTICIPANT_MODULE;
    config.home_channel = iface.home_channel;
    nimcp_error_t err = mesh_participant_register(registry, &iface, &config, &g_kg_interchange_mesh_id);
    if (err == NIMCP_SUCCESS) g_kg_interchange_mesh_registry = registry;
    return err;
}

void kg_interchange_mesh_unregister(void) {
    if (g_kg_interchange_mesh_registry && g_kg_interchange_mesh_id != 0) {
        mesh_participant_unregister(g_kg_interchange_mesh_registry, g_kg_interchange_mesh_id);
        g_kg_interchange_mesh_id = 0;
        g_kg_interchange_mesh_registry = NULL;
    }
}


/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * @brief Get current timestamp in milliseconds
 */
static uint64_t get_current_timestamp_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)ts.tv_nsec / 1000000ULL;
}

/**
 * @brief Get file extension from path
 */
static const char* get_extension(const char* path) {
    const char* dot = strrchr(path, '.');
    if (!dot || dot == path) {
        return "";
    }
    return dot + 1;
}

/* ============================================================================
 * Default Configuration Functions
 * ============================================================================ */

int kg_export_options_default(kg_export_options_t* options) {
    if (!options) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "options is NULL");

        return -1;
    }

    memset(options, 0, sizeof(*options));
    options->format = KG_FORMAT_JSON;
    options->include_metadata = true;
    options->include_weights = false;
    options->include_history = false;
    options->compress_output = false;
    options->encrypt_output = false;
    options->node_filter = NULL;
    options->edge_filter = NULL;
    options->max_depth = 0; /* Unlimited */

    return 0;
}

int kg_import_options_default(kg_import_options_t* options) {
    if (!options) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "options is NULL");

        return -1;
    }

    memset(options, 0, sizeof(*options));
    options->format = KG_FORMAT_JSON;
    options->merge_existing = true;
    options->overwrite_conflicts = false;
    options->validate_schema = true;
    options->dry_run = false;
    options->id_mapping_file = NULL;
    options->default_classification = NULL;

    return 0;
}

void kg_import_result_free(kg_import_result_t* result) {
    if (!result) {
        return;
    }

    if (result->error_log_path) {
        nimcp_free(result->error_log_path);
        result->error_log_path = NULL;
    }
}

/* ============================================================================
 * Export Operations API
 * ============================================================================ */

int kg_export_full(
    const brain_kg_t* kg,
    const char* output_path,
    const kg_export_options_t* options
) {
    if (!kg || !output_path) {
        return -1;
    }

    kg_export_options_t local_options;
    if (options) {
        memcpy(&local_options, options, sizeof(kg_export_options_t));
    } else {
        kg_export_options_default(&local_options);
    }

    /* Infer format from path if needed */
    if (local_options.format == KG_FORMAT_JSON) {
        local_options.format = kg_infer_format_from_path(output_path);
    }

    /* In a real implementation:
     * 1. Open output file
     * 2. Iterate all nodes (apply filter if set)
     * 3. Iterate all edges (apply filter if set)
     * 4. Serialize according to format
     * 5. Optionally compress/encrypt
     * 6. Write to file
     */

    FILE* file = fopen(output_path, "w");
    if (!file) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "file is NULL");

        return -1;
    }

    /* Placeholder: write minimal JSON */
    if (local_options.format == KG_FORMAT_JSON) {
        fprintf(file, "{\n  \"nodes\": [],\n  \"edges\": []\n}\n");
    }

    fclose(file);

    return 0;
}

int kg_export_subgraph(
    const brain_kg_t* kg,
    brain_kg_node_id_t root,
    uint32_t depth,
    const char* output_path,
    const kg_export_options_t* options
) {
    if (!kg || !output_path) {
        return -1;
    }

    (void)root;
    (void)depth;

    /* In a real implementation:
     * 1. BFS/DFS from root to depth
     * 2. Collect reachable nodes and edges
     * 3. Export subset
     */

    return kg_export_full(kg, output_path, options);
}

int kg_export_to_buffer(
    const brain_kg_t* kg,
    const kg_export_options_t* options,
    void** buffer,
    size_t* size
) {
    if (!kg || !buffer || !size) {
        return -1;
    }

    kg_export_options_t local_options;
    if (options) {
        memcpy(&local_options, options, sizeof(kg_export_options_t));
    } else {
        kg_export_options_default(&local_options);
    }

    /* Placeholder: allocate minimal JSON buffer */
    const char* placeholder = "{\"nodes\":[],\"edges\":[]}";
    size_t len = strlen(placeholder) + 1;

    *buffer = nimcp_malloc(len);
    if (!*buffer) {
        return -1;
    }

    memcpy(*buffer, placeholder, len);
    *size = len;

    return 0;
}

/* ============================================================================
 * Import Operations API
 * ============================================================================ */

int kg_import_from_file(
    brain_kg_t* kg,
    const char* input_path,
    const kg_import_options_t* options,
    kg_import_result_t* result
) {
    if (!kg || !input_path) {
        return -1;
    }

    uint64_t start_time = get_current_timestamp_ms();

    kg_import_options_t local_options;
    if (options) {
        memcpy(&local_options, options, sizeof(kg_import_options_t));
    } else {
        kg_import_options_default(&local_options);
    }

    /* Initialize result */
    kg_import_result_t local_result;
    memset(&local_result, 0, sizeof(local_result));

    /* In a real implementation:
     * 1. Open input file
     * 2. Parse according to format
     * 3. Validate schema if enabled
     * 4. Resolve ID conflicts
     * 5. Merge or replace existing
     * 6. Report statistics
     */

    FILE* file = fopen(input_path, "r");
    if (!file) {
        if (result) {
            result->errors = 1;
            result->duration_ms = get_current_timestamp_ms() - start_time;
        }
        return -1;
    }

    fclose(file);

    local_result.duration_ms = get_current_timestamp_ms() - start_time;

    if (result) {
        memcpy(result, &local_result, sizeof(kg_import_result_t));
    }

    return 0;
}

int kg_import_from_buffer(
    brain_kg_t* kg,
    const void* buffer,
    size_t size,
    const kg_import_options_t* options,
    kg_import_result_t* result
) {
    if (!kg || !buffer || size == 0) {
        return -1;
    }

    uint64_t start_time = get_current_timestamp_ms();

    kg_import_result_t local_result;
    memset(&local_result, 0, sizeof(local_result));

    /* In a real implementation, parse buffer and import */
    (void)options;

    local_result.duration_ms = get_current_timestamp_ms() - start_time;

    if (result) {
        memcpy(result, &local_result, sizeof(kg_import_result_t));
    }

    return 0;
}

int kg_import_validate(
    const char* input_path,
    const kg_import_options_t* options,
    kg_import_result_t* preview
) {
    if (!input_path) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "input_path is NULL");

        return -1;
    }

    (void)options;

    kg_import_result_t local_preview;
    memset(&local_preview, 0, sizeof(local_preview));

    /* In a real implementation, parse without applying changes */

    if (preview) {
        memcpy(preview, &local_preview, sizeof(kg_import_result_t));
    }

    return 0;
}

/* ============================================================================
 * Incremental Sync API
 * ============================================================================ */

int kg_sync_export_changes(
    const brain_kg_t* kg,
    uint64_t since_timestamp,
    const char* output_path,
    const kg_export_options_t* options
) {
    if (!kg || !output_path) {
        return -1;
    }

    (void)since_timestamp;

    /* In a real implementation:
     * 1. Query nodes/edges modified since timestamp
     * 2. Export delta
     */

    return kg_export_full(kg, output_path, options);
}

int kg_sync_import_changes(
    brain_kg_t* kg,
    const char* input_path,
    const kg_import_options_t* options,
    kg_import_result_t* result
) {
    if (!kg || !input_path) {
        return -1;
    }

    /* Delta import is same as regular import with merge mode */
    return kg_import_from_file(kg, input_path, options, result);
}

/* ============================================================================
 * External Ontology Integration API
 * ============================================================================ */

int kg_ontology_import(
    brain_kg_t* kg,
    kg_ontology_type_t type,
    const char* path_or_url
) {
    if (!kg || !path_or_url) {
        return -1;
    }

    (void)type;

    /* In a real implementation:
     * 1. Download or open ontology file
     * 2. Parse ontology format (WordNet, OWL, etc.)
     * 3. Create reference nodes for ontology entries
     * 4. Store external IDs for linking
     */

    return 0;
}

int kg_ontology_link(
    brain_kg_t* kg,
    brain_kg_node_id_t node_id,
    kg_ontology_type_t type,
    const char* external_id
) {
    if (!kg || !external_id) {
        return -1;
    }

    (void)node_id;
    (void)type;

    /* In a real implementation:
     * 1. Find ontology reference node by external_id
     * 2. Create edge from node to ontology entry
     * 3. Store link metadata
     */

    return 0;
}

int kg_ontology_lookup(
    const brain_kg_t* kg,
    kg_ontology_type_t type,
    const char* term,
    brain_kg_node_id_t* matches,
    uint32_t max,
    uint32_t* count
) {
    if (!kg || !term || !matches || max == 0 || !count) {
        return -1;
    }

    (void)type;

    /* In a real implementation:
     * 1. Search ontology index for matching terms
     * 2. Return corresponding node IDs
     */

    *count = 0;

    return 0;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

static const char* export_format_strings[] = {
    "JSON",
    "JSON_LD",
    "RDF_XML",
    "RDF_TURTLE",
    "RDF_NTRIPLES",
    "GRAPHML",
    "GEXF",
    "CSV",
    "PARQUET",
    "NIMCP_BINARY"
};

const char* kg_export_format_to_string(kg_export_format_t format) {
    if (format >= 0 && format < KG_FORMAT_COUNT) {
        return export_format_strings[format];
    }
    return "UNKNOWN";
}

static const char* ontology_type_strings[] = {
    "WordNet",
    "ConceptNet",
    "Wikidata",
    "DBpedia",
    "Schema.org",
    "Custom"
};

const char* kg_ontology_type_to_string(kg_ontology_type_t type) {
    if (type >= 0 && type < KG_ONTOLOGY_COUNT) {
        return ontology_type_strings[type];
    }
    return "UNKNOWN";
}

kg_export_format_t kg_infer_format_from_path(const char* path) {
    if (!path) {
        return KG_FORMAT_JSON;
    }

    const char* ext = get_extension(path);

    if (strcmp(ext, "json") == 0) return KG_FORMAT_JSON;
    if (strcmp(ext, "jsonld") == 0) return KG_FORMAT_JSON_LD;
    if (strcmp(ext, "rdf") == 0 || strcmp(ext, "xml") == 0) return KG_FORMAT_RDF_XML;
    if (strcmp(ext, "ttl") == 0) return KG_FORMAT_RDF_TURTLE;
    if (strcmp(ext, "nt") == 0) return KG_FORMAT_RDF_NTRIPLES;
    if (strcmp(ext, "graphml") == 0) return KG_FORMAT_GRAPHML;
    if (strcmp(ext, "gexf") == 0) return KG_FORMAT_GEXF;
    if (strcmp(ext, "csv") == 0) return KG_FORMAT_CSV;
    if (strcmp(ext, "parquet") == 0) return KG_FORMAT_PARQUET;
    if (strcmp(ext, "nimcp") == 0 || strcmp(ext, "bin") == 0) return KG_FORMAT_NIMCP_BINARY;

    return KG_FORMAT_JSON; /* Default */
}

static const char* format_extensions[] = {
    ".json",
    ".jsonld",
    ".rdf",
    ".ttl",
    ".nt",
    ".graphml",
    ".gexf",
    ".csv",
    ".parquet",
    ".nimcp"
};

const char* kg_format_extension(kg_export_format_t format) {
    if (format >= 0 && format < KG_FORMAT_COUNT) {
        return format_extensions[format];
    }
    return ".json";
}
