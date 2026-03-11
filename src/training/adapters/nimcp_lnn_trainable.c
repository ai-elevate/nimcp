/**
 * @file nimcp_lnn_trainable.c
 * @brief Adapter: wraps lnn_training_ctx_t behind the unified trainable network interface
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "lnn/nimcp_lnn_training.h"
#include "lnn/nimcp_lnn.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>

typedef struct {
    lnn_training_ctx_t* lnn_ctx;    /* NOT owned */
    uint32_t output_dim;
    uint32_t input_dim;
} lnn_adapter_ctx_t;

/* --- vtable implementations --- */

static int lnn_adapter_forward(void* ctx, const float* input, uint32_t input_dim,
                               float* output, uint32_t output_dim) {
    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)ctx;
    if (!a || !a->lnn_ctx || !a->lnn_ctx->network || !input || !output) return -1;

    /* Wrap raw floats into tensors for LNN API */
    uint32_t in_dims[1] = { input_dim };
    uint32_t out_dims[1] = { output_dim };
    nimcp_tensor_t* in_t = nimcp_tensor_create(in_dims, 1, NIMCP_DTYPE_F32);
    nimcp_tensor_t* out_t = nimcp_tensor_create(out_dims, 1, NIMCP_DTYPE_F32);
    if (!in_t || !out_t) {
        nimcp_tensor_destroy(in_t);
        nimcp_tensor_destroy(out_t);
        return -1;
    }

    memcpy(nimcp_tensor_data(in_t), input, input_dim * sizeof(float));

    int rc = lnn_forward_step(a->lnn_ctx->network, in_t, out_t, 1.0f);

    if (rc == 0) {
        const float* out_data = (const float*)nimcp_tensor_data_const(out_t);
        uint32_t copy_dim = (output_dim < (uint32_t)nimcp_tensor_numel(out_t)) ?
                             output_dim : (uint32_t)nimcp_tensor_numel(out_t);
        memcpy(output, out_data, copy_dim * sizeof(float));
    }

    nimcp_tensor_destroy(in_t);
    nimcp_tensor_destroy(out_t);
    return (rc == 0) ? 0 : -1;
}

static int lnn_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                float* dl_dinput, uint32_t input_dim) {
    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)ctx;
    if (!a || !a->lnn_ctx || !a->lnn_ctx->network || !dl_doutput) return -1;

    /* Wrap loss gradient into tensor for LNN adjoint backward */
    uint32_t dims[1] = { output_dim };
    nimcp_tensor_t* grad_t = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!grad_t) return -1;

    memcpy(nimcp_tensor_data(grad_t), dl_doutput, output_dim * sizeof(float));

    int rc = lnn_backward(a->lnn_ctx->network, grad_t);

    /* Propagate input gradient for bridge flow */
    if (dl_dinput && rc == 0) {
        uint32_t dim = (input_dim < output_dim) ? input_dim : output_dim;
        memcpy(dl_dinput, dl_doutput, dim * sizeof(float));
    }

    nimcp_tensor_destroy(grad_t);
    return (rc == 0) ? 0 : -1;
}

static int lnn_adapter_get_param_groups(void* ctx,
                                        nimcp_utm_param_group_t** groups,
                                        uint32_t* num_groups) {
    (void)ctx;
    if (!groups || !num_groups) return -1;
    /* LNN parameters managed via adjoint method + internal optimizer */
    *groups = NULL;
    *num_groups = 0;
    return 0;
}

static int lnn_adapter_zero_grad(void* ctx) {
    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)ctx;
    if (!a || !a->lnn_ctx || !a->lnn_ctx->network) return -1;
    lnn_network_zero_gradients(a->lnn_ctx->network);
    return 0;
}

static uint32_t lnn_adapter_get_output_dim(void* ctx) {
    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)ctx;
    return a ? a->output_dim : 0;
}

static uint32_t lnn_adapter_get_input_dim(void* ctx) {
    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)ctx;
    return a ? a->input_dim : 0;
}

static float lnn_adapter_auxiliary_loss(void* ctx) {
    (void)ctx;
    return 0.0f;
}

static void lnn_adapter_destroy(void* ctx) {
    nimcp_free(ctx);
}

/* --- static vtable --- */

static const nimcp_trainable_network_ops_t lnn_trainable_ops = {
    .name = "LNN",
    .type = NIMCP_TRAINABLE_LNN,
    .forward = lnn_adapter_forward,
    .backward = lnn_adapter_backward,
    .get_param_groups = lnn_adapter_get_param_groups,
    .zero_grad = lnn_adapter_zero_grad,
    .get_output_dim = lnn_adapter_get_output_dim,
    .get_input_dim = lnn_adapter_get_input_dim,
    .compute_auxiliary_loss = lnn_adapter_auxiliary_loss,
    .destroy = lnn_adapter_destroy,
};

/* --- public creation --- */

int nimcp_trainable_lnn_create(struct lnn_training_ctx_s* lnn_ctx,
                               const nimcp_trainable_network_ops_t** ops,
                               void** ctx) {
    if (!lnn_ctx || !ops || !ctx) return -1;

    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)nimcp_calloc(1, sizeof(lnn_adapter_ctx_t));
    if (!a) return -1;

    a->lnn_ctx = (lnn_training_ctx_t*)lnn_ctx;

    /* Get dims from LNN network if available */
    if (a->lnn_ctx->network) {
        a->input_dim = a->lnn_ctx->network->n_inputs;
        a->output_dim = a->lnn_ctx->network->n_outputs;
    }

    *ops = &lnn_trainable_ops;
    *ctx = a;

    NIMCP_LOGGING_INFO("Created LNN trainable adapter (%ux → %ux)",
                       a->input_dim, a->output_dim);
    return 0;
}
