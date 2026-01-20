/**
 * @file nimcp_kg_reader.c
 * @brief Knowledge Graph Reader Implementation for NIMCP Self-Awareness
 *
 * Parses .aim/memory-nimcp.jsonl and provides runtime access to system's
 * structural self-knowledge.
 *
 * JSONL FORMAT:
 * Each line is a JSON object, either:
 * - Entity: {"type":"entity","name":"...","entityType":"...","observations":[...]}
 * - Relation: {"type":"relation","from":"...","to":"...","relationType":"..."}
 *
 * @version 1.0.0
 * @author NIMCP Development Team
 * @date 2025-12-29
 */

#include "cognitive/knowledge/nimcp_kg_reader.h"
#include "api/nimcp_api_exception.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <time.h>

/* ============================================================================
 * INTERNAL STRUCTURES
 * ============================================================================ */

/**
 * @brief Internal knowledge graph reader structure
 */
struct kg_reader {
    kg_entity_t* entities[KG_MAX_ENTITIES];     /**< Entity storage */
    uint32_t num_entities;                       /**< Number of entities */

    kg_relation_t* relations[KG_MAX_RELATIONS]; /**< Relation storage */
    uint32_t num_relations;                      /**< Number of relations */

    char file_path[512];                         /**< Loaded file path */
    time_t file_mtime;                           /**< File modification time at load */
    uint64_t load_time_us;                       /**< Load time in microseconds */
};

/* Thread-local error message */
static __thread char kg_error_msg[256] = {0};

/* ============================================================================
 * HELPER FUNCTIONS
 * ============================================================================ */

/**
 * @brief Set error message
 */
static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(kg_error_msg, sizeof(kg_error_msg), fmt, args);
    va_end(args);
}

/**
 * @brief Skip whitespace in string
 */
static const char* skip_ws(const char* s) {
    while (*s && isspace((unsigned char)*s)) s++;
    return s;
}

/**
 * @brief Parse JSON string value (handles escape sequences)
 */
static char* parse_json_string(const char** pos) {
    const char* s = *pos;
    s = skip_ws(s);

    if (*s != '"') return NULL;
    s++;

    /* Find end of string */
    const char* start = s;
    size_t len = 0;
    while (*s && *s != '"') {
        if (*s == '\\' && *(s+1)) {
            s += 2;
            len++;
        } else {
            s++;
            len++;
        }
    }

    if (*s != '"') return NULL;

    /* Allocate and copy with escape handling */
    char* result = malloc(len + 1);
    if (!result) return NULL;

    const char* src = start;
    char* dst = result;
    while (src < s) {
        if (*src == '\\' && *(src+1)) {
            src++;
            switch (*src) {
                case 'n': *dst++ = '\n'; break;
                case 't': *dst++ = '\t'; break;
                case 'r': *dst++ = '\r'; break;
                case '"': *dst++ = '"'; break;
                case '\\': *dst++ = '\\'; break;
                default: *dst++ = *src; break;
            }
            src++;
        } else {
            *dst++ = *src++;
        }
    }
    *dst = '\0';

    *pos = s + 1;  /* Skip closing quote */
    return result;
}

/**
 * @brief Find key in JSON object and return position after colon
 */
static const char* find_json_key(const char* json, const char* key) {
    char search[KG_MAX_NAME_LENGTH + 4];
    snprintf(search, sizeof(search), "\"%s\"", key);

    const char* pos = strstr(json, search);
    if (!pos) return NULL;

    pos += strlen(search);
    pos = skip_ws(pos);
    if (*pos != ':') return NULL;
    pos++;
    return skip_ws(pos);
}

/**
 * @brief Parse JSON string array
 */
static int parse_json_string_array(const char* json, char** out_strings, uint32_t max_strings, uint32_t* out_count) {
    const char* pos = skip_ws(json);
    if (*pos != '[') return -1;
    pos++;

    uint32_t count = 0;
    while (count < max_strings) {
        pos = skip_ws(pos);
        if (*pos == ']') break;
        if (*pos == ',') { pos++; continue; }

        char* str = parse_json_string(&pos);
        if (!str) break;

        out_strings[count++] = str;
    }

    *out_count = count;
    return 0;
}

/**
 * @brief Parse entity from JSON line
 */
static kg_entity_t* parse_entity(const char* json) {
    kg_entity_t* entity = calloc(1, sizeof(kg_entity_t));
    if (!entity) return NULL;

    /* Parse name */
    const char* name_pos = find_json_key(json, "name");
    if (name_pos) {
        char* name = parse_json_string(&name_pos);
        if (name) {
            strncpy(entity->name, name, KG_MAX_NAME_LENGTH - 1);
            free(name);
        }
    }

    /* Parse entityType */
    const char* type_pos = find_json_key(json, "entityType");
    if (type_pos) {
        char* type = parse_json_string(&type_pos);
        if (type) {
            strncpy(entity->entity_type, type, KG_MAX_TYPE_LENGTH - 1);
            free(type);
        }
    }

    /* Parse observations array */
    const char* obs_pos = find_json_key(json, "observations");
    if (obs_pos) {
        parse_json_string_array(obs_pos, entity->observations, KG_MAX_OBSERVATIONS, &entity->num_observations);
    }

    if (entity->name[0] == '\0') {
        /* Invalid entity - no name */
        for (uint32_t i = 0; i < entity->num_observations; i++) {
            free(entity->observations[i]);
        }
        free(entity);
        return NULL;
    }

    return entity;
}

/**
 * @brief Parse relation from JSON line
 */
static kg_relation_t* parse_relation(const char* json) {
    kg_relation_t* relation = calloc(1, sizeof(kg_relation_t));
    if (!relation) return NULL;

    /* Parse from */
    const char* from_pos = find_json_key(json, "from");
    if (from_pos) {
        char* from = parse_json_string(&from_pos);
        if (from) {
            strncpy(relation->from, from, KG_MAX_NAME_LENGTH - 1);
            free(from);
        }
    }

    /* Parse to */
    const char* to_pos = find_json_key(json, "to");
    if (to_pos) {
        char* to = parse_json_string(&to_pos);
        if (to) {
            strncpy(relation->to, to, KG_MAX_NAME_LENGTH - 1);
            free(to);
        }
    }

    /* Parse relationType */
    const char* type_pos = find_json_key(json, "relationType");
    if (type_pos) {
        char* type = parse_json_string(&type_pos);
        if (type) {
            strncpy(relation->relation_type, type, KG_MAX_TYPE_LENGTH - 1);
            free(type);
        }
    }

    if (relation->from[0] == '\0' || relation->to[0] == '\0') {
        free(relation);
        return NULL;
    }

    return relation;
}

/**
 * @brief Determine line type from JSON
 */
static const char* get_line_type(const char* json) {
    /* Check for entity markers */
    if (strstr(json, "\"entityType\"") || strstr(json, "\"observations\"")) {
        return "entity";
    }
    /* Check for relation markers */
    if (strstr(json, "\"relationType\"") && strstr(json, "\"from\"") && strstr(json, "\"to\"")) {
        return "relation";
    }
    return NULL;
}

/**
 * @brief Case-insensitive string search
 */
static const char* strcasestr_local(const char* haystack, const char* needle) {
    if (!*needle) return haystack;

    for (; *haystack; haystack++) {
        const char* h = haystack;
        const char* n = needle;

        while (*h && *n && tolower((unsigned char)*h) == tolower((unsigned char)*n)) {
            h++;
            n++;
        }

        if (!*n) return haystack;
    }

    return NULL;
}

/* ============================================================================
 * LIFECYCLE API
 * ============================================================================ */

kg_reader_t* kg_reader_create(void) {
    kg_reader_t* reader = calloc(1, sizeof(kg_reader_t));
    NIMCP_API_CHECK_ALLOC(reader, "Failed to allocate KG reader");
    return reader;
}

void kg_reader_destroy(kg_reader_t* reader) {
    if (!reader) return;

    /* Free entities */
    for (uint32_t i = 0; i < reader->num_entities; i++) {
        if (reader->entities[i]) {
            for (uint32_t j = 0; j < reader->entities[i]->num_observations; j++) {
                free(reader->entities[i]->observations[j]);
            }
            free(reader->entities[i]);
        }
    }

    /* Free relations */
    for (uint32_t i = 0; i < reader->num_relations; i++) {
        free(reader->relations[i]);
    }

    free(reader);
}

int kg_reader_load(kg_reader_t* reader, const char* file_path) {
    NIMCP_API_CHECK_NULL(reader, -1, "NULL reader in kg_reader_load");

    const char* path = file_path ? file_path : KG_DEFAULT_PATH;

    FILE* fp = fopen(path, "r");
    if (!fp) {
        set_error("Failed to open KG file: %s", path);
        return -1;
    }

    /* Get file modification time */
    struct stat st;
    if (stat(path, &st) == 0) {
        reader->file_mtime = st.st_mtime;
    }

    strncpy(reader->file_path, path, sizeof(reader->file_path) - 1);

    /* Clear existing data */
    for (uint32_t i = 0; i < reader->num_entities; i++) {
        if (reader->entities[i]) {
            for (uint32_t j = 0; j < reader->entities[i]->num_observations; j++) {
                free(reader->entities[i]->observations[j]);
            }
            free(reader->entities[i]);
            reader->entities[i] = NULL;
        }
    }
    for (uint32_t i = 0; i < reader->num_relations; i++) {
        free(reader->relations[i]);
        reader->relations[i] = NULL;
    }
    reader->num_entities = 0;
    reader->num_relations = 0;

    /* Parse JSONL file */
    char line[8192];
    uint64_t start_time = (uint64_t)clock();

    while (fgets(line, sizeof(line), fp)) {
        /* Skip empty lines */
        const char* trimmed = skip_ws(line);
        if (*trimmed == '\0' || *trimmed == '\n') continue;

        const char* line_type = get_line_type(line);

        if (line_type && strcmp(line_type, "entity") == 0) {
            if (reader->num_entities < KG_MAX_ENTITIES) {
                kg_entity_t* entity = parse_entity(line);
                if (entity) {
                    reader->entities[reader->num_entities++] = entity;
                }
            }
        } else if (line_type && strcmp(line_type, "relation") == 0) {
            if (reader->num_relations < KG_MAX_RELATIONS) {
                kg_relation_t* relation = parse_relation(line);
                if (relation) {
                    reader->relations[reader->num_relations++] = relation;
                }
            }
        }
    }

    uint64_t end_time = (uint64_t)clock();
    reader->load_time_us = (end_time - start_time) * 1000000 / CLOCKS_PER_SEC;

    fclose(fp);
    return 0;
}

int kg_reader_reload(kg_reader_t* reader) {
    NIMCP_API_CHECK_NULL(reader, -1, "NULL reader in kg_reader_reload");
    NIMCP_API_CHECK(reader->file_path[0] != '\0', -1, "No file loaded to reload");
    return kg_reader_load(reader, reader->file_path);
}

bool kg_reader_is_modified(const kg_reader_t* reader) {
    if (!reader || reader->file_path[0] == '\0') return false;

    struct stat st;
    if (stat(reader->file_path, &st) != 0) return false;

    return st.st_mtime > reader->file_mtime;
}

/* ============================================================================
 * ENTITY QUERY API
 * ============================================================================ */

const kg_entity_t* kg_reader_get_entity(const kg_reader_t* reader, const char* name) {
    if (!reader || !name) return NULL;

    for (uint32_t i = 0; i < reader->num_entities; i++) {
        if (reader->entities[i] && strcmp(reader->entities[i]->name, name) == 0) {
            return reader->entities[i];
        }
    }
    return NULL;
}

kg_entity_list_t* kg_reader_get_entities_by_type(const kg_reader_t* reader, const char* entity_type) {
    if (!reader || !entity_type) return NULL;

    kg_entity_list_t* list = calloc(1, sizeof(kg_entity_list_t));
    if (!list) return NULL;

    list->capacity = 64;
    list->entities = calloc(list->capacity, sizeof(kg_entity_t*));
    if (!list->entities) {
        free(list);
        return NULL;
    }

    for (uint32_t i = 0; i < reader->num_entities; i++) {
        if (reader->entities[i] && strcmp(reader->entities[i]->entity_type, entity_type) == 0) {
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                kg_entity_t** new_entities = realloc(list->entities, list->capacity * sizeof(kg_entity_t*));
                if (!new_entities) break;
                list->entities = new_entities;
            }
            list->entities[list->count++] = reader->entities[i];
        }
    }

    return list;
}

kg_entity_list_t* kg_reader_get_all_entities(const kg_reader_t* reader) {
    if (!reader) return NULL;

    kg_entity_list_t* list = calloc(1, sizeof(kg_entity_list_t));
    if (!list) return NULL;

    list->capacity = reader->num_entities;
    list->entities = calloc(list->capacity, sizeof(kg_entity_t*));
    if (!list->entities) {
        free(list);
        return NULL;
    }

    for (uint32_t i = 0; i < reader->num_entities; i++) {
        if (reader->entities[i]) {
            list->entities[list->count++] = reader->entities[i];
        }
    }

    return list;
}

kg_entity_list_t* kg_reader_search_entities(const kg_reader_t* reader, const char* search_text) {
    if (!reader || !search_text) return NULL;

    kg_entity_list_t* list = calloc(1, sizeof(kg_entity_list_t));
    if (!list) return NULL;

    list->capacity = 64;
    list->entities = calloc(list->capacity, sizeof(kg_entity_t*));
    if (!list->entities) {
        free(list);
        return NULL;
    }

    for (uint32_t i = 0; i < reader->num_entities; i++) {
        kg_entity_t* e = reader->entities[i];
        if (!e) continue;

        bool found = false;

        /* Search in name */
        if (strcasestr_local(e->name, search_text)) found = true;

        /* Search in type */
        if (!found && strcasestr_local(e->entity_type, search_text)) found = true;

        /* Search in observations */
        if (!found) {
            for (uint32_t j = 0; j < e->num_observations; j++) {
                if (e->observations[j] && strcasestr_local(e->observations[j], search_text)) {
                    found = true;
                    break;
                }
            }
        }

        if (found) {
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                kg_entity_t** new_entities = realloc(list->entities, list->capacity * sizeof(kg_entity_t*));
                if (!new_entities) break;
                list->entities = new_entities;
            }
            list->entities[list->count++] = e;
        }
    }

    return list;
}

void kg_entity_list_destroy(kg_entity_list_t* list) {
    if (!list) return;
    free(list->entities);
    free(list);
}

/* ============================================================================
 * RELATION QUERY API
 * ============================================================================ */

kg_relation_list_t* kg_reader_get_relations_from(const kg_reader_t* reader, const char* from_entity) {
    if (!reader || !from_entity) return NULL;

    kg_relation_list_t* list = calloc(1, sizeof(kg_relation_list_t));
    if (!list) return NULL;

    list->capacity = 32;
    list->relations = calloc(list->capacity, sizeof(kg_relation_t*));
    if (!list->relations) {
        free(list);
        return NULL;
    }

    for (uint32_t i = 0; i < reader->num_relations; i++) {
        if (reader->relations[i] && strcmp(reader->relations[i]->from, from_entity) == 0) {
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                kg_relation_t** new_rels = realloc(list->relations, list->capacity * sizeof(kg_relation_t*));
                if (!new_rels) break;
                list->relations = new_rels;
            }
            list->relations[list->count++] = reader->relations[i];
        }
    }

    return list;
}

kg_relation_list_t* kg_reader_get_relations_to(const kg_reader_t* reader, const char* to_entity) {
    if (!reader || !to_entity) return NULL;

    kg_relation_list_t* list = calloc(1, sizeof(kg_relation_list_t));
    if (!list) return NULL;

    list->capacity = 32;
    list->relations = calloc(list->capacity, sizeof(kg_relation_t*));
    if (!list->relations) {
        free(list);
        return NULL;
    }

    for (uint32_t i = 0; i < reader->num_relations; i++) {
        if (reader->relations[i] && strcmp(reader->relations[i]->to, to_entity) == 0) {
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                kg_relation_t** new_rels = realloc(list->relations, list->capacity * sizeof(kg_relation_t*));
                if (!new_rels) break;
                list->relations = new_rels;
            }
            list->relations[list->count++] = reader->relations[i];
        }
    }

    return list;
}

kg_relation_list_t* kg_reader_get_relations_by_type(const kg_reader_t* reader, const char* relation_type) {
    if (!reader || !relation_type) return NULL;

    kg_relation_list_t* list = calloc(1, sizeof(kg_relation_list_t));
    if (!list) return NULL;

    list->capacity = 32;
    list->relations = calloc(list->capacity, sizeof(kg_relation_t*));
    if (!list->relations) {
        free(list);
        return NULL;
    }

    for (uint32_t i = 0; i < reader->num_relations; i++) {
        if (reader->relations[i] && strcmp(reader->relations[i]->relation_type, relation_type) == 0) {
            if (list->count >= list->capacity) {
                list->capacity *= 2;
                kg_relation_t** new_rels = realloc(list->relations, list->capacity * sizeof(kg_relation_t*));
                if (!new_rels) break;
                list->relations = new_rels;
            }
            list->relations[list->count++] = reader->relations[i];
        }
    }

    return list;
}

const char* kg_reader_are_connected(const kg_reader_t* reader, const char* from_entity, const char* to_entity) {
    if (!reader || !from_entity || !to_entity) return NULL;

    for (uint32_t i = 0; i < reader->num_relations; i++) {
        kg_relation_t* r = reader->relations[i];
        if (r && strcmp(r->from, from_entity) == 0 && strcmp(r->to, to_entity) == 0) {
            return r->relation_type;
        }
    }
    return NULL;
}

void kg_relation_list_destroy(kg_relation_list_t* list) {
    if (!list) return;
    free(list->relations);
    free(list);
}

/* ============================================================================
 * OBSERVATION QUERY API
 * ============================================================================ */

const char* kg_reader_get_observation(const kg_reader_t* reader, const char* entity_name, const char* keyword) {
    const kg_entity_t* entity = kg_reader_get_entity(reader, entity_name);
    if (!entity || !keyword) return NULL;

    for (uint32_t i = 0; i < entity->num_observations; i++) {
        if (entity->observations[i] && strcasestr_local(entity->observations[i], keyword)) {
            return entity->observations[i];
        }
    }
    return NULL;
}

const char* const* kg_reader_get_observations(const kg_reader_t* reader, const char* entity_name, uint32_t* out_count) {
    const kg_entity_t* entity = kg_reader_get_entity(reader, entity_name);
    if (!entity) {
        if (out_count) *out_count = 0;
        return NULL;
    }

    if (out_count) *out_count = entity->num_observations;
    return (const char* const*)entity->observations;
}

/* ============================================================================
 * INTROSPECTION API
 * ============================================================================ */

const char** kg_reader_get_module_names(const kg_reader_t* reader, uint32_t* out_count) {
    kg_entity_list_t* modules = kg_reader_get_entities_by_type(reader, "Module");
    if (!modules) {
        if (out_count) *out_count = 0;
        return NULL;
    }

    const char** names = calloc(modules->count, sizeof(char*));
    if (!names) {
        kg_entity_list_destroy(modules);
        if (out_count) *out_count = 0;
        return NULL;
    }

    for (uint32_t i = 0; i < modules->count; i++) {
        names[i] = modules->entities[i]->name;
    }

    if (out_count) *out_count = modules->count;
    kg_entity_list_destroy(modules);
    return names;
}

const char* kg_reader_get_module_capabilities(const kg_reader_t* reader, const char* module_name) {
    const char* cap = kg_reader_get_observation(reader, module_name, "capabilit");
    if (cap) return cap;

    cap = kg_reader_get_observation(reader, module_name, "provides");
    if (cap) return cap;

    cap = kg_reader_get_observation(reader, module_name, "Core");
    return cap;
}

const char* kg_reader_get_module_location(const kg_reader_t* reader, const char* module_name) {
    const char* loc = kg_reader_get_observation(reader, module_name, "Located at");
    if (loc) return loc;

    loc = kg_reader_get_observation(reader, module_name, "include/");
    if (loc) return loc;

    loc = kg_reader_get_observation(reader, module_name, "src/");
    return loc;
}

kg_relation_list_t* kg_reader_get_module_integrations(const kg_reader_t* reader, const char* module_name) {
    return kg_reader_get_relations_from(reader, module_name);
}

int kg_reader_generate_self_description(const kg_reader_t* reader, char* buffer, size_t buffer_size) {
    if (!reader || !buffer || buffer_size == 0) return 0;

    int written = 0;
    int n;

    n = snprintf(buffer + written, buffer_size - written,
                 "I am NIMCP (Neuro-Inspired Model of Cognitive Processing).\n\n");
    if (n > 0) written += n;

    /* List modules */
    uint32_t mod_count = 0;
    const char** modules = kg_reader_get_module_names(reader, &mod_count);

    if (modules && mod_count > 0) {
        n = snprintf(buffer + written, buffer_size - written,
                     "I have %u modules:\n", mod_count);
        if (n > 0) written += n;

        for (uint32_t i = 0; i < mod_count && (size_t)written < buffer_size - 100; i++) {
            const char* cap = kg_reader_get_module_capabilities(reader, modules[i]);
            n = snprintf(buffer + written, buffer_size - written,
                         "  - %s: %s\n", modules[i], cap ? cap : "(no description)");
            if (n > 0) written += n;
        }

        free((void*)modules);
    }

    /* Count integrations */
    n = snprintf(buffer + written, buffer_size - written,
                 "\nI have %u integration relationships connecting my subsystems.\n",
                 reader->num_relations);
    if (n > 0) written += n;

    return written;
}

/* ============================================================================
 * STATISTICS API
 * ============================================================================ */

int kg_reader_get_stats(const kg_reader_t* reader, kg_reader_stats_t* stats) {
    if (!reader || !stats) return -1;

    memset(stats, 0, sizeof(kg_reader_stats_t));
    stats->total_entities = reader->num_entities;
    stats->total_relations = reader->num_relations;
    stats->load_time_us = reader->load_time_us;
    stats->last_reload_time = reader->file_mtime;
    strncpy(stats->file_path, reader->file_path, sizeof(stats->file_path) - 1);

    /* Count total observations */
    for (uint32_t i = 0; i < reader->num_entities; i++) {
        if (reader->entities[i]) {
            stats->total_observations += reader->entities[i]->num_observations;
        }
    }

    return 0;
}

const char* kg_reader_get_last_error(void) {
    return kg_error_msg;
}
