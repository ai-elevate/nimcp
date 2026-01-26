/**
 * @file nimcp_supply_chain.c
 * @brief Supply Chain Security Main Implementation
 *
 * WHAT: Core supply chain security context and artifact verification
 * WHY: Protect against supply chain attacks and tampering
 * HOW: Hash verification, signature checking, runtime monitoring
 */

#include "security/nimcp_supply_chain.h"
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "utils/validation/nimcp_common.h"
#include "api/nimcp_api_exception.h"

/* Error code aliases for this file */
#ifndef NIMCP_OK
#define NIMCP_OK NIMCP_SUCCESS
#endif
#ifndef NIMCP_ERROR_INVALID_PARAM
#define NIMCP_ERROR_INVALID_PARAM NIMCP_ERROR_INVALID_PARAM
#endif
#ifndef NIMCP_ERROR_IO
#define NIMCP_ERROR_IO (-121)
#endif
#ifndef NIMCP_ERROR_CRYPTO
#define NIMCP_ERROR_CRYPTO (-130)
#endif
#include "utils/error/nimcp_error_codes.h"
#include "utils/logging/nimcp_logging.h"
#include "async/nimcp_bio_router.h"
#include "async/nimcp_bio_messages.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdio.h>
#include <openssl/sha.h>
#include <openssl/evp.h>

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for supply_chain module */
static nimcp_health_agent_t* g_supply_chain_health_agent = NULL;

/**
 * @brief Set health agent for supply_chain heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void supply_chain_set_health_agent(nimcp_health_agent_t* agent) {
    g_supply_chain_health_agent = agent;
}

/** @brief Send heartbeat from supply_chain module */
static inline void supply_chain_heartbeat(const char* operation, float progress) {
    if (g_supply_chain_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_supply_chain_health_agent, operation, progress);
    }
}


/* Supply chain context structure */
struct nimcp_supply_chain {
    uint32_t magic;
    nimcp_supply_chain_config_t config;
    nimcp_supply_chain_stats_t stats;

    /* Dependencies */
    nimcp_dependency_t* dependencies;
    size_t dependency_count;
    size_t dependency_capacity;

    /* Trusted sources */
    nimcp_trusted_source_t* sources;
    size_t source_count;
    size_t source_capacity;

    /* Runtime monitoring */
    pthread_t monitor_thread;
    bool monitoring_active;

    pthread_mutex_t lock;
    bio_module_context_t bio_ctx;
    bool bio_registered;
};

/* Bio-async message types */
#define NIMCP_SC_MSG_VERIFY         1
#define NIMCP_SC_MSG_VIOLATION      2
#define NIMCP_SC_MSG_UPDATE         3

/* ========================================================================
 * SHA-256/512 Hash Computation
 * ======================================================================== */

static nimcp_error_t compute_file_hash(const char* filepath,
                                        nimcp_hash_algorithm_t algo,
                                        char* hash_output) {
    if (!filepath || !hash_output) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    FILE* file = fopen(filepath, "rb");
    if (!file) {
        LOG_ERROR("compute_file_hash: Cannot open file %s", filepath);
        return NIMCP_ERROR_IO;
    }

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) {
        fclose(file);
        return NIMCP_ERROR_NO_MEMORY;
    }

    const EVP_MD* md = NULL;
    size_t hash_len = 0;

    switch (algo) {
    case NIMCP_HASH_SHA256:
        md = EVP_sha256();
        hash_len = 32;
        break;
    case NIMCP_HASH_SHA512:
        md = EVP_sha512();
        hash_len = 64;
        break;
    case NIMCP_HASH_SHA3_256:
        md = EVP_sha3_256();
        hash_len = 32;
        break;
    case NIMCP_HASH_SHA3_512:
        md = EVP_sha3_512();
        hash_len = 64;
        break;
    default:
        EVP_MD_CTX_free(ctx);
        fclose(file);
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (EVP_DigestInit_ex(ctx, md, NULL) != 1) {
        EVP_MD_CTX_free(ctx);
        fclose(file);
        return NIMCP_ERROR_CRYPTO;
    }

    /* Read and hash file in chunks */
    uint8_t buffer[8192];
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, sizeof(buffer), file)) > 0) {
        if (EVP_DigestUpdate(ctx, buffer, bytes_read) != 1) {
            EVP_MD_CTX_free(ctx);
            fclose(file);
            return NIMCP_ERROR_CRYPTO;
        }
    }

    fclose(file);

    /* Finalize hash */
    uint8_t hash[64];
    unsigned int actual_len = 0;
    if (EVP_DigestFinal_ex(ctx, hash, &actual_len) != 1) {
        EVP_MD_CTX_free(ctx);
        return NIMCP_ERROR_CRYPTO;
    }

    EVP_MD_CTX_free(ctx);

    /* Convert to hex string */
    for (size_t i = 0; i < hash_len; i++) {
        sprintf(&hash_output[i * 2], "%02x", hash[i]);
    }
    hash_output[hash_len * 2] = '\0';

    return NIMCP_OK;
}

/* ========================================================================
 * Bio-Async Integration
 * ======================================================================== */

static nimcp_error_t sc_inbox_handler(
    const void* msg,
    size_t msg_size,
    nimcp_bio_promise_t response_promise,
    void* user_data)
{
    (void)response_promise;
    nimcp_supply_chain_t sc = (nimcp_supply_chain_t)user_data;

    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!msg || msg_size < sizeof(bio_message_header_t)) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    const bio_message_header_t* header = (const bio_message_header_t*)msg;
    LOG_DEBUG("Supply chain inbox received message type %u", header->type);

    switch (header->type) {
    case NIMCP_SC_MSG_VERIFY:
        LOG_INFO("Supply chain verification request");
        break;
    case NIMCP_SC_MSG_VIOLATION:
        LOG_ERROR("Supply chain integrity violation reported!");
        break;
    case NIMCP_SC_MSG_UPDATE:
        LOG_INFO("Supply chain update notification");
        break;
    default:
        LOG_WARN("Unknown supply chain message type %u", header->type);
        break;
    }

    return NIMCP_OK;
}

/* ========================================================================
 * Context Management
 * ======================================================================== */

nimcp_supply_chain_t nimcp_supply_chain_create(const nimcp_supply_chain_config_t* config) {
    /* Allocate context */
    nimcp_supply_chain_t sc = (nimcp_supply_chain_t)calloc(1, sizeof(struct nimcp_supply_chain));
    NIMCP_API_CHECK_ALLOC(sc, "Failed to allocate supply chain context");

    /* Set magic number */
    sc->magic = NIMCP_SUPPLY_CHAIN_MAGIC;

    /* Copy configuration or use defaults */
    if (config) {
        memcpy(&sc->config, config, sizeof(nimcp_supply_chain_config_t));
    } else {
        /* Default configuration */
        sc->config.enable_logging = true;
        sc->config.strict_mode = false;
        sc->config.default_hash_algo = NIMCP_HASH_SHA256;
        sc->config.default_sig_algo = NIMCP_SIG_ED25519;
        sc->config.monitor_config.enable_periodic_checks = false;
        sc->config.monitor_config.check_interval_seconds = 300;  /* 5 minutes */
        sc->config.monitor_config.verify_on_load = true;
        sc->config.sbom_cache_dir = "/tmp/nimcp_sbom";
        sc->config.artifact_cache_dir = "/tmp/nimcp_artifacts";
        sc->config.bio_ctx = NULL;
    }

    /* Initialize statistics */
    memset(&sc->stats, 0, sizeof(nimcp_supply_chain_stats_t));

    /* Initialize dependency list */
    sc->dependency_capacity = 64;
    sc->dependencies = (nimcp_dependency_t*)calloc(sc->dependency_capacity,
                                                    sizeof(nimcp_dependency_t));
    if (!sc->dependencies) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "Failed to allocate supply chain dependencies");
        free(sc);
        return NULL;
    }

    /* Initialize trusted sources list */
    sc->source_capacity = 16;
    sc->sources = (nimcp_trusted_source_t*)calloc(sc->source_capacity,
                                                   sizeof(nimcp_trusted_source_t));
    if (!sc->sources) {
        free(sc->dependencies);
        free(sc);
        LOG_ERROR("nimcp_supply_chain_create: Source allocation failed");
        return NULL;
    }

    /* Initialize mutex */
    if (pthread_mutex_init(&sc->lock, NULL) != 0) {
        free(sc->sources);
        free(sc->dependencies);
        free(sc);
        LOG_ERROR("nimcp_supply_chain_create: Mutex initialization failed");
        return NULL;
    }

    /* Register with bio-async if available */
    bio_module_info_t module_info = {
        .module_id = BIO_MODULE_SECURITY,
        .module_name = "supply_chain",
        .inbox_capacity = 64,
        .user_data = sc
    };

    sc->bio_ctx = bio_router_register_module(&module_info);
    if (sc->bio_ctx) {
        sc->bio_registered = true;
        LOG_INFO("Supply chain security registered with bio-async");
    } else {
        LOG_WARN("Failed to register supply chain with bio-async");
    }

    LOG_INFO("Supply chain context created");

    return sc;
}

void nimcp_supply_chain_destroy(nimcp_supply_chain_t sc) {
    if (!sc) {
        return;
    }

    if (sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC) {
        LOG_WARN("nimcp_supply_chain_destroy: Invalid magic number");
        return;
    }

    /* Stop monitoring if active */
    if (sc->monitoring_active) {
        nimcp_runtime_disable_monitoring(sc);
    }

    /* Unregister from bio-async */
    if (sc->bio_registered && sc->bio_ctx) {
        bio_router_unregister_module(sc->bio_ctx);
        LOG_INFO("Supply chain unregistered from bio-async");
    }

    /* Free dependencies */
    free(sc->dependencies);

    /* Free sources */
    free(sc->sources);

    /* Destroy mutex */
    pthread_mutex_destroy(&sc->lock);

    /* Clear magic */
    sc->magic = 0;

    /* Free context */
    free(sc);

    LOG_INFO("Supply chain context destroyed");
}

nimcp_error_t nimcp_supply_chain_get_stats(nimcp_supply_chain_t sc,
                                             nimcp_supply_chain_stats_t* stats) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !stats) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&sc->lock);
    memcpy(stats, &sc->stats, sizeof(nimcp_supply_chain_stats_t));
    pthread_mutex_unlock(&sc->lock);

    return NIMCP_OK;
}

/* ========================================================================
 * Artifact Verification
 * ======================================================================== */

nimcp_error_t nimcp_artifact_verify_hash(
    nimcp_supply_chain_t sc,
    const char* filepath,
    const char* expected_hash,
    nimcp_hash_algorithm_t algo)
{
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC ||
        !filepath || !expected_hash) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    char computed_hash[129];
    nimcp_error_t err = compute_file_hash(filepath, algo, computed_hash);
    if (err != NIMCP_OK) {
        pthread_mutex_lock(&sc->lock);
        sc->stats.failed_verifications++;
        pthread_mutex_unlock(&sc->lock);
        LOG_ERROR("Hash computation failed for %s", filepath);
        return err;
    }

    /* Compare hashes (case-insensitive) */
    if (strcasecmp(computed_hash, expected_hash) != 0) {
        pthread_mutex_lock(&sc->lock);
        sc->stats.failed_verifications++;
        sc->stats.integrity_violations++;
        pthread_mutex_unlock(&sc->lock);

        LOG_ERROR("Hash mismatch for %s: expected %s, got %s",
                       filepath, expected_hash, computed_hash);

        /* Bio-async alert would be sent here if needed */

        return NIMCP_ERROR_VERIFICATION_FAILED;
    }

    pthread_mutex_lock(&sc->lock);
    sc->stats.verified_dependencies++;
    sc->stats.last_verification = time(NULL);
    pthread_mutex_unlock(&sc->lock);

    LOG_INFO("Hash verified successfully for %s", filepath);

    return NIMCP_OK;
}

nimcp_error_t nimcp_artifact_compute_hash(
    nimcp_supply_chain_t sc,
    const char* filepath,
    nimcp_hash_algorithm_t algo,
    char* hash_output)
{
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC ||
        !filepath || !hash_output) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    return compute_file_hash(filepath, algo, hash_output);
}

/* ========================================================================
 * Runtime Verification (Stubs)
 * ======================================================================== */

nimcp_error_t nimcp_runtime_verify_library(nimcp_supply_chain_t sc,
                                             const char* library_path) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !library_path) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&sc->lock);
    sc->stats.runtime_checks++;
    pthread_mutex_unlock(&sc->lock);

    LOG_INFO("Runtime library verification for %s", library_path);

    return NIMCP_OK;
}

nimcp_error_t nimcp_runtime_verify_all(nimcp_supply_chain_t sc) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    LOG_INFO("Verifying all loaded libraries...");

    /* Would iterate through all loaded libraries */
    pthread_mutex_lock(&sc->lock);
    sc->stats.runtime_checks++;
    pthread_mutex_unlock(&sc->lock);

    return NIMCP_OK;
}

nimcp_error_t nimcp_runtime_enable_monitoring(nimcp_supply_chain_t sc) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (sc->monitoring_active) {
        LOG_WARN("Runtime monitoring already active");
        return NIMCP_OK;
    }

    sc->monitoring_active = true;
    LOG_INFO("Runtime monitoring enabled");

    return NIMCP_OK;
}

nimcp_error_t nimcp_runtime_disable_monitoring(nimcp_supply_chain_t sc) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    if (!sc->monitoring_active) {
        return NIMCP_OK;
    }

    sc->monitoring_active = false;
    LOG_INFO("Runtime monitoring disabled");

    return NIMCP_OK;
}

nimcp_error_t nimcp_runtime_verify_binary(nimcp_supply_chain_t sc,
                                            const char* binary_path) {
    if (!sc || sc->magic != NIMCP_SUPPLY_CHAIN_MAGIC || !binary_path) {
        return NIMCP_ERROR_INVALID_PARAM;
    }

    pthread_mutex_lock(&sc->lock);
    sc->stats.runtime_checks++;
    pthread_mutex_unlock(&sc->lock);

    LOG_INFO("Binary integrity verification for %s", binary_path);

    return NIMCP_OK;
}
