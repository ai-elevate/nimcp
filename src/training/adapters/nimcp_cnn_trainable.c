/**
 * @file nimcp_cnn_trainable.c
 * @brief Adapter: wraps cnn_trainer_t behind the unified trainable network interface
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "training/nimcp_cnn_training.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>

/* CNN trainer internal struct is opaque outside nimcp_cnn_training.c.
 * We access it through the public API only. */

typedef struct {
    cnn_trainer_t* trainer;     /* NOT owned — must outlive adapter */
    float* last_output;         /* Cached forward output */
    uint32_t output_dim;
    uint32_t input_dim;
} cnn_adapter_ctx_t;

/* --- vtable implementations --- */

static int cnn_adapter_forward(void* ctx, const float* input, uint32_t input_dim,
                               float* output, uint32_t output_dim) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer) return -1;

    /* CNN trainer uses tensor-based forward; for the unified interface we'd
     * need to wrap raw floats into tensors. For Phase 1, this is a stub
     * that the adapter exposes but isn't called directly — the CNN trainer
     * continues to use its own forward path until Phase 3 wiring. */
    (void)input; (void)input_dim; (void)output; (void)output_dim;

    NIMCP_LOGGING_DEBUG("cnn_adapter_forward: stub (CNN uses tensor-based forward)");
    return 0;
}

static int cnn_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                float* dl_dinput, uint32_t input_dim) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer) return -1;

    /* Phase 1 stub — CNN backward uses tensor-based API */
    (void)dl_doutput; (void)output_dim; (void)dl_dinput; (void)input_dim;

    NIMCP_LOGGING_DEBUG("cnn_adapter_backward: stub (CNN uses tensor-based backward)");
    return 0;
}

static int cnn_adapter_get_param_groups(void* ctx,
                                        nimcp_utm_param_group_t** groups,
                                        uint32_t* num_groups) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer || !groups || !num_groups) return -1;

    /* Phase 1 stub: returns empty param groups.
     * Phase 3 will iterate CNN layers and expose weights/gradients. */
    *groups = NULL;
    *num_groups = 0;
    return 0;
}

static int cnn_adapter_zero_grad(void* ctx) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer) return -1;

    cnn_trainer_zero_grad(a->trainer);
    return 0;
}

static uint32_t cnn_adapter_get_output_dim(void* ctx) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    return a ? a->output_dim : 0;
}

static uint32_t cnn_adapter_get_input_dim(void* ctx) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    return a ? a->input_dim : 0;
}

static float cnn_adapter_auxiliary_loss(void* ctx) {
    (void)ctx;
    return 0.0f;
}

static void cnn_adapter_destroy(void* ctx) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a) return;
    nimcp_free(a->last_output);
    nimcp_free(a);
}

/* --- static vtable --- */

static const nimcp_trainable_network_ops_t cnn_trainable_ops = {
    .name = "CNN",
    .type = NIMCP_TRAINABLE_CNN,
    .forward = cnn_adapter_forward,
    .backward = cnn_adapter_backward,
    .get_param_groups = cnn_adapter_get_param_groups,
    .zero_grad = cnn_adapter_zero_grad,
    .get_output_dim = cnn_adapter_get_output_dim,
    .get_input_dim = cnn_adapter_get_input_dim,
    .compute_auxiliary_loss = cnn_adapter_auxiliary_loss,
    .destroy = cnn_adapter_destroy,
};

/* --- public creation --- */

int nimcp_trainable_cnn_create(struct cnn_trainer_s* trainer,
                               const nimcp_trainable_network_ops_t** ops,
                               void** ctx) {
    if (!trainer || !ops || !ctx) return -1;

    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)nimcp_calloc(1, sizeof(cnn_adapter_ctx_t));
    if (!a) return -1;

    a->trainer = trainer;
    /* Dimensions will be set when layers are configured.
     * For now, default to 0 (unknown until forward pass). */
    a->output_dim = 0;
    a->input_dim = 0;

    *ops = &cnn_trainable_ops;
    *ctx = a;

    NIMCP_LOGGING_INFO("Created CNN trainable adapter");
    return 0;
}
