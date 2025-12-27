/**
 * @file nimcp_jepa_weights.c
 * @brief JEPA Weight Loading Implementation
 * @version 1.0.0
 * @date 2025-12-26
 *
 * WHAT: Implementation of weight file loading and management
 * WHY:  Enable loading pretrained V-JEPA weights
 * HOW:  Binary format parsing with dimension validation
 */

#include "cognitive/jepa/nimcp_jepa_weights.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * Module Constants
 * ============================================================================ */

#define LOG_MODULE "[JEPA_WEIGHTS]"

/* CRC32 polynomial */
#define CRC32_POLYNOMIAL 0xEDB88320

/* ============================================================================
 * Internal Helpers - CRC32
 * ============================================================================ */

static uint32_t crc32_table[256];
static bool crc32_initialized = false;

static void init_crc32_table(void) {
    if (crc32_initialized) return;

    for (uint32_t i = 0; i < 256; i++) {
        uint32_t crc = i;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    crc32_initialized = true;
}

uint32_t jepa_weights_crc32(const void* data, size_t size) {
    /* Handle null data or zero size */
    if (!data || size == 0) {
        return 0;
    }

    init_crc32_table();

    const uint8_t* bytes = (const uint8_t*)data;
    uint32_t crc = 0xFFFFFFFF;

    for (size_t i = 0; i < size; i++) {
        uint8_t index = (uint8_t)((crc ^ bytes[i]) & 0xFF);
        crc = (crc >> 8) ^ crc32_table[index];
    }

    return crc ^ 0xFFFFFFFF;
}

/* ============================================================================
 * Internal Helpers - File I/O
 * ============================================================================ */

static int read_header(FILE* fp, jepa_weights_header_t* header) {
    if (fread(header, sizeof(jepa_weights_header_t), 1, fp) != 1) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Failed to read header");
        return NIMCP_ERROR_FILE_READ;
    }

    /* Validate magic */
    if (header->magic != JEPA_WEIGHTS_MAGIC) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Invalid magic: expected 0x%08X, got 0x%08X",
                          JEPA_WEIGHTS_MAGIC, header->magic);
        return NIMCP_ERROR_FILE_CORRUPT;
    }

    /* Validate version */
    if (header->version > JEPA_WEIGHTS_VERSION) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Unsupported version: %u", header->version);
        return NIMCP_ERROR_FILE_CORRUPT;
    }

    return NIMCP_SUCCESS;
}

static int read_tensor_desc(FILE* fp, jepa_tensor_desc_t* tensor) {
    uint16_t name_len;
    if (fread(&name_len, sizeof(uint16_t), 1, fp) != 1) {
        return NIMCP_ERROR_FILE_READ;
    }

    if (name_len >= JEPA_WEIGHTS_MAX_NAME_LEN) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Tensor name too long: %u", name_len);
        return NIMCP_ERROR_FILE_CORRUPT;
    }

    if (fread(tensor->name, 1, name_len, fp) != name_len) {
        return NIMCP_ERROR_FILE_READ;
    }
    tensor->name[name_len] = '\0';

    uint8_t ndims;
    if (fread(&ndims, sizeof(uint8_t), 1, fp) != 1) {
        return NIMCP_ERROR_FILE_READ;
    }
    tensor->ndims = ndims;

    if (ndims > JEPA_WEIGHTS_MAX_DIMS) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Too many dimensions: %u", ndims);
        return NIMCP_ERROR_FILE_CORRUPT;
    }

    tensor->num_elements = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        if (fread(&tensor->dims[i], sizeof(uint32_t), 1, fp) != 1) {
            return NIMCP_ERROR_FILE_READ;
        }
        tensor->num_elements *= tensor->dims[i];
    }

    uint8_t dtype;
    if (fread(&dtype, sizeof(uint8_t), 1, fp) != 1) {
        return NIMCP_ERROR_FILE_READ;
    }
    tensor->dtype = (jepa_weight_dtype_t)dtype;

    tensor->data = NULL;  /* Data loaded separately */

    return NIMCP_SUCCESS;
}

static int read_tensor_data(FILE* fp, jepa_tensor_desc_t* tensor) {
    if (tensor->dtype != JEPA_DTYPE_F32) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Only F32 dtype supported currently");
        return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    tensor->data = nimcp_malloc(tensor->num_elements * sizeof(float));
    if (!tensor->data) {
        return NIMCP_ERROR_MEMORY;
    }

    if (fread(tensor->data, sizeof(float), tensor->num_elements, fp) != tensor->num_elements) {
        nimcp_free(tensor->data);
        tensor->data = NULL;
        return NIMCP_ERROR_FILE_READ;
    }

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * File Loading API
 * ============================================================================ */

jepa_weights_t* jepa_weights_open(const char* path) {
    if (!path) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " NULL path");
        return NULL;
    }

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Failed to open: %s", path);
        return NULL;
    }

    jepa_weights_t* weights = nimcp_malloc(sizeof(jepa_weights_t));
    if (!weights) {
        fclose(fp);
        return NULL;
    }
    memset(weights, 0, sizeof(jepa_weights_t));

    /* Read header */
    if (read_header(fp, &weights->header) != NIMCP_SUCCESS) {
        fclose(fp);
        nimcp_free(weights);
        return NULL;
    }

    /* Store path */
    strncpy(weights->filepath, path, sizeof(weights->filepath) - 1);

    /* Allocate tensor descriptors */
    weights->tensors = nimcp_malloc(weights->header.num_tensors * sizeof(jepa_tensor_desc_t));
    if (!weights->tensors) {
        fclose(fp);
        nimcp_free(weights);
        return NULL;
    }
    memset(weights->tensors, 0, weights->header.num_tensors * sizeof(jepa_tensor_desc_t));

    /* Read tensor descriptors */
    for (uint32_t i = 0; i < weights->header.num_tensors; i++) {
        if (read_tensor_desc(fp, &weights->tensors[i]) != NIMCP_SUCCESS) {
            jepa_weights_close(weights);
            fclose(fp);
            return NULL;
        }
    }

    /* Read tensor data */
    for (uint32_t i = 0; i < weights->header.num_tensors; i++) {
        if (read_tensor_data(fp, &weights->tensors[i]) != NIMCP_SUCCESS) {
            jepa_weights_close(weights);
            fclose(fp);
            return NULL;
        }
    }

    fclose(fp);
    weights->is_loaded = true;

    NIMCP_LOGGING_INFO(LOG_MODULE " Loaded %s: %u tensors, %lu params",
                      path, weights->header.num_tensors,
                      (unsigned long)weights->header.total_params);

    return weights;
}

void jepa_weights_close(jepa_weights_t* weights) {
    if (!weights) return;

    if (weights->tensors) {
        for (uint32_t i = 0; i < weights->header.num_tensors; i++) {
            if (weights->tensors[i].data) {
                nimcp_free(weights->tensors[i].data);
            }
        }
        nimcp_free(weights->tensors);
    }

    nimcp_free(weights);
}

int jepa_weights_validate(const char* path, uint32_t expected_latent_dim) {
    jepa_weights_header_t header;
    int result = jepa_weights_info(path, &header);

    if (result != NIMCP_SUCCESS) {
        return result;
    }

    if (expected_latent_dim > 0 && header.latent_dim != expected_latent_dim) {
        NIMCP_LOGGING_WARN(LOG_MODULE " Latent dim mismatch: expected %u, got %u",
                          expected_latent_dim, header.latent_dim);
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    return NIMCP_SUCCESS;
}

int jepa_weights_info(const char* path, jepa_weights_header_t* header) {
    if (!path) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        return NIMCP_ERROR_FILE_READ;
    }

    jepa_weights_header_t local_header;
    int result = read_header(fp, &local_header);
    fclose(fp);

    if (result == NIMCP_SUCCESS && header) {
        *header = local_header;
    }

    return result;
}

/* ============================================================================
 * Weight Loading API
 * ============================================================================ */

jepa_load_result_t jepa_weights_load(const char* path,
                                      jepa_predictor_t* predictor) {
    jepa_load_result_t result = {0};
    result.status = JEPA_LOAD_FAILED;

    if (!path || !predictor) {
        snprintf(result.message, sizeof(result.message), "NULL parameter");
        return result;
    }

    /* Open weight file */
    jepa_weights_t* weights = jepa_weights_open(path);
    if (!weights) {
        snprintf(result.message, sizeof(result.message), "Failed to open weight file");
        return result;
    }

    /* Check architecture compatibility */
    if (predictor->type != JEPA_PREDICTOR_MLP &&
        predictor->type != JEPA_PREDICTOR_LINEAR) {
        snprintf(result.message, sizeof(result.message),
                "Only MLP/Linear predictors supported");
        result.status = JEPA_LOAD_INCOMPATIBLE;
        jepa_weights_close(weights);
        return result;
    }

    /* Load weights for each layer */
    jepa_mlp_t* mlp = &predictor->network.mlp;

    for (uint32_t layer_idx = 0; layer_idx < mlp->num_layers; layer_idx++) {
        jepa_mlp_layer_t* layer = &mlp->layers[layer_idx];

        /* Try to find matching weight tensor */
        char weight_name[128];
        snprintf(weight_name, sizeof(weight_name), "predictor.layer%u.weight", layer_idx);

        const jepa_tensor_desc_t* weight_tensor = jepa_weights_get_tensor(weights, weight_name);
        if (weight_tensor) {
            /* Validate dimensions */
            if (weight_tensor->ndims == 2 &&
                weight_tensor->dims[0] == layer->out_dim &&
                weight_tensor->dims[1] == layer->in_dim) {
                memcpy(layer->weights, weight_tensor->data,
                       layer->out_dim * layer->in_dim * sizeof(float));
                result.tensors_loaded++;
                result.params_loaded += layer->out_dim * layer->in_dim;
            } else {
                NIMCP_LOGGING_WARN(LOG_MODULE " Dimension mismatch for %s: "
                                  "expected [%u,%u], got [%u,%u]",
                                  weight_name, layer->out_dim, layer->in_dim,
                                  weight_tensor->dims[0], weight_tensor->dims[1]);
                result.tensors_skipped++;
            }
        } else {
            /* Try alternative naming conventions */
            snprintf(weight_name, sizeof(weight_name), "layers.%u.weight", layer_idx);
            weight_tensor = jepa_weights_get_tensor(weights, weight_name);
            if (weight_tensor && weight_tensor->ndims == 2 &&
                weight_tensor->dims[0] == layer->out_dim &&
                weight_tensor->dims[1] == layer->in_dim) {
                memcpy(layer->weights, weight_tensor->data,
                       layer->out_dim * layer->in_dim * sizeof(float));
                result.tensors_loaded++;
                result.params_loaded += layer->out_dim * layer->in_dim;
            } else {
                result.tensors_skipped++;
            }
        }

        /* Try to find matching bias tensor */
        char bias_name[128];
        snprintf(bias_name, sizeof(bias_name), "predictor.layer%u.bias", layer_idx);

        const jepa_tensor_desc_t* bias_tensor = jepa_weights_get_tensor(weights, bias_name);
        if (bias_tensor) {
            if (bias_tensor->ndims == 1 && bias_tensor->dims[0] == layer->out_dim) {
                memcpy(layer->bias, bias_tensor->data, layer->out_dim * sizeof(float));
                result.tensors_loaded++;
                result.params_loaded += layer->out_dim;
            } else {
                result.tensors_skipped++;
            }
        } else {
            /* Try alternative naming */
            snprintf(bias_name, sizeof(bias_name), "layers.%u.bias", layer_idx);
            bias_tensor = jepa_weights_get_tensor(weights, bias_name);
            if (bias_tensor && bias_tensor->ndims == 1 &&
                bias_tensor->dims[0] == layer->out_dim) {
                memcpy(layer->bias, bias_tensor->data, layer->out_dim * sizeof(float));
                result.tensors_loaded++;
                result.params_loaded += layer->out_dim;
            }
        }
    }

    /* Determine overall status */
    if (result.tensors_loaded == 0) {
        result.status = JEPA_LOAD_FAILED;
        snprintf(result.message, sizeof(result.message),
                "No matching tensors found");
    } else if (result.tensors_skipped > 0) {
        result.status = JEPA_LOAD_PARTIAL;
        snprintf(result.message, sizeof(result.message),
                "Loaded %u tensors, skipped %u",
                result.tensors_loaded, result.tensors_skipped);
    } else {
        result.status = JEPA_LOAD_SUCCESS;
        snprintf(result.message, sizeof(result.message),
                "Successfully loaded %u tensors (%lu params)",
                result.tensors_loaded, (unsigned long)result.params_loaded);
    }

    NIMCP_LOGGING_INFO(LOG_MODULE " %s", result.message);

    jepa_weights_close(weights);
    return result;
}

int jepa_weights_load_tensor(jepa_weights_t* weights,
                              const char* layer_name,
                              float* output,
                              uint64_t expected_size) {
    if (!weights || !layer_name || !output) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    const jepa_tensor_desc_t* tensor = jepa_weights_get_tensor(weights, layer_name);
    if (!tensor) {
        return NIMCP_ERROR_NOT_FOUND;
    }

    if (tensor->num_elements != expected_size) {
        NIMCP_LOGGING_WARN(LOG_MODULE " Size mismatch for %s: expected %lu, got %lu",
                          layer_name, (unsigned long)expected_size,
                          (unsigned long)tensor->num_elements);
        return NIMCP_ERROR_INVALID_PARAMETER;
    }

    memcpy(output, tensor->data, expected_size * sizeof(float));
    return NIMCP_SUCCESS;
}

jepa_load_result_t jepa_weights_load_adaptive(const char* path,
                                               jepa_predictor_t* predictor,
                                               bool allow_resize) {
    /* For now, just use regular load */
    /* Future: implement dimension adaptation (truncation/padding) */
    (void)allow_resize;
    return jepa_weights_load(path, predictor);
}

/* ============================================================================
 * Weight Saving API
 * ============================================================================ */

int jepa_weights_save(const char* path, const jepa_predictor_t* predictor) {
    return jepa_weights_save_with_meta(path, predictor, JEPA_MODEL_CUSTOM, NULL);
}

int jepa_weights_save_with_meta(const char* path,
                                 const jepa_predictor_t* predictor,
                                 jepa_model_type_t model_type,
                                 const char* extra_metadata) {
    (void)extra_metadata;  /* Reserved for future use */

    if (!path || !predictor) {
        return NIMCP_ERROR_NULL_POINTER;
    }

    if (predictor->type != JEPA_PREDICTOR_MLP &&
        predictor->type != JEPA_PREDICTOR_LINEAR) {
        return NIMCP_ERROR_NOT_IMPLEMENTED;
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        NIMCP_LOGGING_ERROR(LOG_MODULE " Failed to create: %s", path);
        return NIMCP_ERROR_FILE_READ;
    }

    const jepa_mlp_t* mlp = &predictor->network.mlp;

    /* Prepare header */
    jepa_weights_header_t header = {0};
    header.magic = JEPA_WEIGHTS_MAGIC;
    header.version = JEPA_WEIGHTS_VERSION;
    header.num_tensors = mlp->num_layers * 2;  /* weights + biases */
    header.model_type = model_type;
    header.latent_dim = predictor->config.output_dim;
    header.hidden_dim = predictor->config.hidden_dim;
    header.num_layers = mlp->num_layers;

    /* Count total params */
    header.total_params = 0;
    for (uint32_t i = 0; i < mlp->num_layers; i++) {
        const jepa_mlp_layer_t* layer = &mlp->layers[i];
        header.total_params += layer->in_dim * layer->out_dim + layer->out_dim;
    }

    /* Write header */
    if (fwrite(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return NIMCP_ERROR_FILE_READ;
    }

    /* Write tensors */
    for (uint32_t layer_idx = 0; layer_idx < mlp->num_layers; layer_idx++) {
        const jepa_mlp_layer_t* layer = &mlp->layers[layer_idx];

        /* Write weight tensor */
        char name[128];
        snprintf(name, sizeof(name), "predictor.layer%u.weight", layer_idx);
        uint16_t name_len = (uint16_t)strlen(name);
        fwrite(&name_len, sizeof(uint16_t), 1, fp);
        fwrite(name, 1, name_len, fp);

        uint8_t ndims = 2;
        fwrite(&ndims, sizeof(uint8_t), 1, fp);
        fwrite(&layer->out_dim, sizeof(uint32_t), 1, fp);
        fwrite(&layer->in_dim, sizeof(uint32_t), 1, fp);

        uint8_t dtype = JEPA_DTYPE_F32;
        fwrite(&dtype, sizeof(uint8_t), 1, fp);

        fwrite(layer->weights, sizeof(float), layer->out_dim * layer->in_dim, fp);

        /* Write bias tensor */
        snprintf(name, sizeof(name), "predictor.layer%u.bias", layer_idx);
        name_len = (uint16_t)strlen(name);
        fwrite(&name_len, sizeof(uint16_t), 1, fp);
        fwrite(name, 1, name_len, fp);

        ndims = 1;
        fwrite(&ndims, sizeof(uint8_t), 1, fp);
        fwrite(&layer->out_dim, sizeof(uint32_t), 1, fp);

        fwrite(&dtype, sizeof(uint8_t), 1, fp);
        fwrite(layer->bias, sizeof(float), layer->out_dim, fp);
    }

    fclose(fp);

    NIMCP_LOGGING_INFO(LOG_MODULE " Saved %lu params to %s",
                      (unsigned long)header.total_params, path);

    return NIMCP_SUCCESS;
}

/* ============================================================================
 * Utility API
 * ============================================================================ */

void jepa_weights_list_tensors(const jepa_weights_t* weights) {
    if (!weights) return;

    NIMCP_LOGGING_INFO(LOG_MODULE " Weight file: %s", weights->filepath);
    NIMCP_LOGGING_INFO(LOG_MODULE " Model: %s, Version: %u",
                      jepa_model_type_to_string(weights->header.model_type),
                      weights->header.version);
    NIMCP_LOGGING_INFO(LOG_MODULE " Tensors: %u, Total params: %lu",
                      weights->header.num_tensors,
                      (unsigned long)weights->header.total_params);

    for (uint32_t i = 0; i < weights->header.num_tensors; i++) {
        const jepa_tensor_desc_t* t = &weights->tensors[i];
        char dims_str[128] = "";
        int offset = 0;

        for (uint32_t d = 0; d < t->ndims; d++) {
            if (d > 0) {
                offset += snprintf(dims_str + offset, sizeof(dims_str) - offset, "×");
            }
            offset += snprintf(dims_str + offset, sizeof(dims_str) - offset, "%u", t->dims[d]);
        }

        NIMCP_LOGGING_INFO(LOG_MODULE "   [%u] %s: [%s] %s (%lu elements)",
                          i, t->name, dims_str,
                          jepa_weight_dtype_to_string(t->dtype),
                          (unsigned long)t->num_elements);
    }
}

const jepa_tensor_desc_t* jepa_weights_get_tensor(const jepa_weights_t* weights,
                                                    const char* name) {
    if (!weights || !name) return NULL;

    for (uint32_t i = 0; i < weights->header.num_tensors; i++) {
        if (strcmp(weights->tensors[i].name, name) == 0) {
            return &weights->tensors[i];
        }
    }

    return NULL;
}

/* ============================================================================
 * String Conversion API
 * ============================================================================ */

const char* jepa_model_type_to_string(jepa_model_type_t type) {
    switch (type) {
        case JEPA_MODEL_CUSTOM:      return "custom";
        case JEPA_MODEL_VJEPA2_VITL: return "vjepa2-vitl";
        case JEPA_MODEL_VJEPA2_VITH: return "vjepa2-vith";
        case JEPA_MODEL_VJEPA2_VITG: return "vjepa2-vitg";
        case JEPA_MODEL_IJEPA_VITL:  return "ijepa-vitl";
        case JEPA_MODEL_IJEPA_VITH:  return "ijepa-vith";
        default:                     return "unknown";
    }
}

const char* jepa_weight_dtype_to_string(jepa_weight_dtype_t dtype) {
    switch (dtype) {
        case JEPA_DTYPE_F32:  return "float32";
        case JEPA_DTYPE_F16:  return "float16";
        case JEPA_DTYPE_BF16: return "bfloat16";
        case JEPA_DTYPE_INT8: return "int8";
        default:             return "unknown";
    }
}

const char* jepa_load_status_to_string(jepa_load_status_t status) {
    switch (status) {
        case JEPA_LOAD_SUCCESS:      return "success";
        case JEPA_LOAD_PARTIAL:      return "partial";
        case JEPA_LOAD_FAILED:       return "failed";
        case JEPA_LOAD_INCOMPATIBLE: return "incompatible";
        default:                     return "unknown";
    }
}
