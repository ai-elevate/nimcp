/**
 * @file nimcp_sbom.c
 * @brief Software Bill of Materials (SBOM) Generation and Management
 *
 * WHAT: SPDX and CycloneDX SBOM format support
 * WHY: Document software composition for security audits and compliance
 * HOW: Parse/generate SBOM documents, track dependencies
 */

#include "security/nimcp_supply_chain.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "utils/validation/nimcp_common.h"

/* Error code aliases for this file */
#ifndef NIMCP_OK
#define NIMCP_OK NIMCP_SUCCESS
#endif
#ifndef NIMCP_ERROR_INVALID_ARGUMENT
#define NIMCP_ERROR_INVALID_ARGUMENT NIMCP_ERROR_INVALID_PARAM
#endif
#ifndef NIMCP_ERROR_IO
#define NIMCP_ERROR_IO (-121)
#endif
#ifndef NIMCP_ERROR_CRYPTO
#define NIMCP_ERROR_CRYPTO (-130)
#endif
#include "utils/error/nimcp_error_codes.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

/* External context accessor */
extern struct nimcp_supply_chain {
    uint32_t magic;
    nimcp_supply_chain_config_t config;
    nimcp_supply_chain_stats_t stats;
    nimcp_dependency_t* dependencies;
    size_t dependency_count;
    size_t dependency_capacity;
    nimcp_trusted_source_t* sources;
    size_t source_count;
    size_t source_capacity;
    pthread_t monitor_thread;
    bool monitoring_active;
    pthread_mutex_t lock;
    bio_module_context_t bio_ctx;
    bool bio_registered;
};

/* ========================================================================
 * SBOM Loading
 * ======================================================================== */

nimcp_error_t nimcp_sbom_load(nimcp_supply_chain_t sc,
                               const char* filepath,
                               nimcp_sbom_format_t format) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !filepath) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    FILE* file = fopen(filepath, "r");
    if (!file) {
        LOG_ERROR("nimcp_sbom_load: Cannot open file %s", filepath);
        return NIMCP_ERROR_IO;
    }

    LOG_INFO("Loading SBOM from %s (format=%d)", filepath, format);

    /* Read file content */
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* content = (char*)malloc(file_size + 1);
    if (!content) {
        fclose(file);
        return NIMCP_ERROR_NO_MEMORY;
    }

    size_t bytes_read = fread(content, 1, file_size, file);
    content[bytes_read] = '\0';
    fclose(file);

    /*
     * Parse SBOM (simplified - production would use full SPDX/CycloneDX parser)
     * For now, we'll look for dependency patterns
     */

    pthread_mutex_lock(&sc->lock);

    /* Simple pattern matching for demonstration */
    char* line = strtok(content, "\n");
    while (line != NULL) {
        /* Look for dependency entries (very simplified) */
        if (strstr(line, "PackageName:") || strstr(line, "component")) {
            LOG_DEBUG("Found dependency entry: %s", line);
            /* Would parse full dependency information here */
        }
        line = strtok(NULL, "\n");
    }

    sc->stats.sbom_loads++;
    sc->stats.last_sbom_update = time(NULL);

    pthread_mutex_unlock(&sc->lock);

    free(content);

    LOG_INFO("SBOM loaded successfully");

    return NIMCP_OK;
}

/* ========================================================================
 * SBOM Generation
 * ======================================================================== */

static nimcp_error_t generate_spdx_sbom(nimcp_supply_chain_t sc, char** output) {
    /* Estimate size */
    size_t estimated_size = 4096 + (sc->dependency_count * 1024);
    char* sbom = (char*)malloc(estimated_size);
    if (!sbom) {
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* SPDX 2.3 header */
    char* ptr = sbom;
    ptr += sprintf(ptr, "SPDXVersion: SPDX-2.3\n");
    ptr += sprintf(ptr, "DataLicense: CC0-1.0\n");
    ptr += sprintf(ptr, "SPDXID: SPDXRef-DOCUMENT\n");
    ptr += sprintf(ptr, "DocumentName: NIMCP-SBOM\n");
    ptr += sprintf(ptr, "DocumentNamespace: https://nimcp.ai/sbom/%ld\n", time(NULL));
    ptr += sprintf(ptr, "Creator: Tool: NIMCP Supply Chain Security\n");
    ptr += sprintf(ptr, "Created: %s", ctime(&sc->stats.last_sbom_update));
    ptr += sprintf(ptr, "\n");

    /* Package information */
    ptr += sprintf(ptr, "PackageName: NIMCP\n");
    ptr += sprintf(ptr, "SPDXID: SPDXRef-Package-NIMCP\n");
    ptr += sprintf(ptr, "PackageVersion: 1.0.0\n");
    ptr += sprintf(ptr, "PackageSupplier: Organization: NIMCP Project\n");
    ptr += sprintf(ptr, "\n");

    /* Dependencies */
    for (size_t i = 0; i < sc->dependency_count; i++) {
        nimcp_dependency_t* dep = &sc->dependencies[i];
        ptr += sprintf(ptr, "PackageName: %s\n", dep->name);
        ptr += sprintf(ptr, "SPDXID: SPDXRef-Package-%zu\n", i);
        ptr += sprintf(ptr, "PackageVersion: %s\n", dep->version);
        ptr += sprintf(ptr, "PackageSupplier: %s\n", dep->supplier);
        if (dep->license[0] != '\0') {
            ptr += sprintf(ptr, "PackageLicenseConcluded: %s\n", dep->license);
        }
        if (dep->hash_sha256[0] != '\0') {
            ptr += sprintf(ptr, "PackageChecksum: SHA256: %s\n", dep->hash_sha256);
        }
        ptr += sprintf(ptr, "\n");
    }

    *output = sbom;
    return NIMCP_OK;
}

static nimcp_error_t generate_cyclonedx_sbom(nimcp_supply_chain_t sc, char** output) {
    /* Estimate size */
    size_t estimated_size = 4096 + (sc->dependency_count * 1024);
    char* sbom = (char*)malloc(estimated_size);
    if (!sbom) {
        return NIMCP_ERROR_NO_MEMORY;
    }

    /* CycloneDX 1.5 JSON format */
    char* ptr = sbom;
    ptr += sprintf(ptr, "{\n");
    ptr += sprintf(ptr, "  \"bomFormat\": \"CycloneDX\",\n");
    ptr += sprintf(ptr, "  \"specVersion\": \"1.5\",\n");
    ptr += sprintf(ptr, "  \"version\": 1,\n");
    ptr += sprintf(ptr, "  \"metadata\": {\n");
    ptr += sprintf(ptr, "    \"timestamp\": \"%ld\",\n", time(NULL));
    ptr += sprintf(ptr, "    \"tools\": [\n");
    ptr += sprintf(ptr, "      {\"name\": \"NIMCP Supply Chain Security\"}\n");
    ptr += sprintf(ptr, "    ]\n");
    ptr += sprintf(ptr, "  },\n");
    ptr += sprintf(ptr, "  \"components\": [\n");

    /* Components */
    for (size_t i = 0; i < sc->dependency_count; i++) {
        nimcp_dependency_t* dep = &sc->dependencies[i];
        ptr += sprintf(ptr, "    {\n");
        ptr += sprintf(ptr, "      \"type\": \"library\",\n");
        ptr += sprintf(ptr, "      \"name\": \"%s\",\n", dep->name);
        ptr += sprintf(ptr, "      \"version\": \"%s\",\n", dep->version);
        if (dep->supplier[0] != '\0') {
            ptr += sprintf(ptr, "      \"supplier\": {\"name\": \"%s\"},\n", dep->supplier);
        }
        if (dep->hash_sha256[0] != '\0') {
            ptr += sprintf(ptr, "      \"hashes\": [{\"alg\": \"SHA-256\", \"content\": \"%s\"}],\n",
                          dep->hash_sha256);
        }
        ptr += sprintf(ptr, "      \"purl\": \"pkg:generic/%s@%s\"\n", dep->name, dep->version);
        ptr += sprintf(ptr, "    }%s\n", (i < sc->dependency_count - 1) ? "," : "");
    }

    ptr += sprintf(ptr, "  ]\n");
    ptr += sprintf(ptr, "}\n");

    *output = sbom;
    return NIMCP_OK;
}

nimcp_error_t nimcp_sbom_generate(nimcp_supply_chain_t sc,
                                   nimcp_sbom_format_t format,
                                   char** output) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !output) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&sc->lock);

    nimcp_error_t err;
    if (format == NIMCP_SBOM_FORMAT_SPDX) {
        err = generate_spdx_sbom(sc, output);
    } else if (format == NIMCP_SBOM_FORMAT_CYCLONEDX) {
        err = generate_cyclonedx_sbom(sc, output);
    } else {
        pthread_mutex_unlock(&sc->lock);
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    if (err == NIMCP_OK) {
        sc->stats.sbom_generations++;
        sc->stats.last_sbom_update = time(NULL);
    }

    pthread_mutex_unlock(&sc->lock);

    LOG_INFO("SBOM generated successfully (format=%d, size=%zu)",
                   format, *output ? strlen(*output) : 0);

    return err;
}

nimcp_error_t nimcp_sbom_save(nimcp_supply_chain_t sc,
                               const char* filepath,
                               nimcp_sbom_format_t format) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !filepath) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    char* sbom_content = NULL;
    nimcp_error_t err = nimcp_sbom_generate(sc, format, &sbom_content);
    if (err != NIMCP_OK || !sbom_content) {
        return err;
    }

    FILE* file = fopen(filepath, "w");
    if (!file) {
        free(sbom_content);
        LOG_ERROR("nimcp_sbom_save: Cannot open file %s for writing", filepath);
        return NIMCP_ERROR_IO;
    }

    size_t bytes_written = fwrite(sbom_content, 1, strlen(sbom_content), file);
    fclose(file);
    free(sbom_content);

    if (bytes_written == 0) {
        LOG_ERROR("nimcp_sbom_save: Write failed");
        return NIMCP_ERROR_IO;
    }

    LOG_INFO("SBOM saved to %s", filepath);

    return NIMCP_OK;
}

/* ========================================================================
 * Dependency Management
 * ======================================================================== */

nimcp_error_t nimcp_sbom_get_dependencies(nimcp_supply_chain_t sc,
                                           nimcp_dependency_t** deps,
                                           size_t* count) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !deps || !count) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&sc->lock);

    if (sc->dependency_count == 0) {
        *deps = NULL;
        *count = 0;
        pthread_mutex_unlock(&sc->lock);
        return NIMCP_OK;
    }

    /* Allocate copy of dependencies */
    *deps = (nimcp_dependency_t*)malloc(sc->dependency_count * sizeof(nimcp_dependency_t));
    if (!*deps) {
        pthread_mutex_unlock(&sc->lock);
        return NIMCP_ERROR_NO_MEMORY;
    }

    memcpy(*deps, sc->dependencies, sc->dependency_count * sizeof(nimcp_dependency_t));
    *count = sc->dependency_count;

    pthread_mutex_unlock(&sc->lock);

    LOG_DEBUG("Retrieved %zu dependencies", *count);

    return NIMCP_OK;
}

nimcp_error_t nimcp_sbom_add_dependency(nimcp_supply_chain_t sc,
                                         const nimcp_dependency_t* dep) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !dep) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&sc->lock);

    /* Check capacity */
    if (sc->dependency_count >= sc->dependency_capacity) {
        size_t new_capacity = sc->dependency_capacity * 2;
        nimcp_dependency_t* new_deps = (nimcp_dependency_t*)realloc(
            sc->dependencies,
            new_capacity * sizeof(nimcp_dependency_t)
        );
        if (!new_deps) {
            pthread_mutex_unlock(&sc->lock);
            return NIMCP_ERROR_NO_MEMORY;
        }
        sc->dependencies = new_deps;
        sc->dependency_capacity = new_capacity;
    }

    /* Add dependency */
    memcpy(&sc->dependencies[sc->dependency_count], dep, sizeof(nimcp_dependency_t));
    sc->dependencies[sc->dependency_count].magic = NIMCP_DEPENDENCY_MAGIC;
    sc->dependency_count++;
    sc->stats.total_dependencies++;

    pthread_mutex_unlock(&sc->lock);

    LOG_INFO("Added dependency: %s version %s", dep->name, dep->version);

    return NIMCP_OK;
}

nimcp_error_t nimcp_sbom_remove_dependency(nimcp_supply_chain_t sc,
                                            const char* name) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !name) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&sc->lock);

    /* Find and remove dependency */
    for (size_t i = 0; i < sc->dependency_count; i++) {
        if (strcmp(sc->dependencies[i].name, name) == 0) {
            /* Shift remaining dependencies */
            memmove(&sc->dependencies[i], &sc->dependencies[i + 1],
                    (sc->dependency_count - i - 1) * sizeof(nimcp_dependency_t));
            sc->dependency_count--;
            pthread_mutex_unlock(&sc->lock);
            LOG_INFO("Removed dependency: %s", name);
            return NIMCP_OK;
        }
    }

    pthread_mutex_unlock(&sc->lock);

    LOG_WARN("Dependency not found: %s", name);
    return NIMCP_ERROR_NOT_FOUND;
}

nimcp_error_t nimcp_sbom_query_dependency(nimcp_supply_chain_t sc,
                                           const char* name,
                                           nimcp_dependency_t* dep) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !name || !dep) {
        return NIMCP_ERROR_INVALID_ARGUMENT;
    }

    pthread_mutex_lock(&sc->lock);

    /* Find dependency */
    for (size_t i = 0; i < sc->dependency_count; i++) {
        if (strcmp(sc->dependencies[i].name, name) == 0) {
            memcpy(dep, &sc->dependencies[i], sizeof(nimcp_dependency_t));
            pthread_mutex_unlock(&sc->lock);
            return NIMCP_OK;
        }
    }

    pthread_mutex_unlock(&sc->lock);

    return NIMCP_ERROR_NOT_FOUND;
}
