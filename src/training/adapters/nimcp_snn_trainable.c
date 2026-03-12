/**
 * @file nimcp_snn_trainable.c
 * @brief Adapter: wraps snn_backprop_ctx_t behind the unified trainable network interface
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "training/nimcp_snn_backprop.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>

typedef struct {
    snn_backprop_ctx_t* snn_ctx;    /* NOT owned */
    uint32_t output_dim;
    uint32_t input_dim;
    bool managed_by_utm;            /* When true, skip internal optimizer step */
} snn_adapter_ctx_t;

/* --- vtable implementations --- */

static int snn_adapter_forward(void* ctx, const float* input, uint32_t input_dim,
                               float* output, uint32_t output_dim) {
    snn_adapter_ctx_t* a = (snn_adapter_ctx_t*)ctx;
    if (!a || !a->snn_ctx || !input || !output) return -1;

    /* SNN forward: single sample, 1ms duration */
    int rc = snn_backprop_forward(a->snn_ctx, input, 1, 1.0f, output);
    return (rc == 0) ? 0 : -1;
}

static int snn_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                float* dl_dinput, uint32_t input_dim) {
    snn_adapter_ctx_t* a = (snn_adapter_ctx_t*)ctx;
    if (!a || !a->snn_ctx || !dl_doutput) return -1;

    /* SNN backward: accumulates gradients into ctx->gradients */
    int rc = snn_backprop_backward(a->snn_ctx, dl_doutput, 1);

    /* Run SNN's internal optimizer step after backward — unless UTM manages params */
    if (rc == 0 && !a->managed_by_utm) {
        snn_backprop_step(a->snn_ctx, 0.0f);  /* 0 = use config LR */
    }

    /* Copy real input gradients from SNN surrogate gradient backprop */
    if (dl_dinput) {
        uint32_t grad_size = 0;
        const float* in_grad = snn_backprop_get_input_grad(a->snn_ctx, &grad_size);
        if (in_grad && grad_size > 0) {
            uint32_t copy_dim = (input_dim < grad_size) ? input_dim : grad_size;
            memcpy(dl_dinput, in_grad, copy_dim * sizeof(float));
            if (input_dim > copy_dim) {
                memset(dl_dinput + copy_dim, 0, (input_dim - copy_dim) * sizeof(float));
            }
        } else {
            memset(dl_dinput, 0, input_dim * sizeof(float));
        }
    }
    return (rc == 0) ? 0 : -1;
}

static int snn_adapter_get_param_groups(void* ctx,
                                        nimcp_utm_param_group_t** groups,
                                        uint32_t* num_groups) {
    (void)ctx;
    if (!groups || !num_groups) return -1;
    /* SNN parameters managed internally via surrogate gradient / e-prop */
    *groups = NULL;
    *num_groups = 0;
    return 0;
}

static int snn_adapter_zero_grad(void* ctx) {
    snn_adapter_ctx_t* a = (snn_adapter_ctx_t*)ctx;
    if (!a || !a->snn_ctx) return -1;
    snn_backprop_zero_grad(a->snn_ctx);
    return 0;
}

static uint32_t snn_adapter_get_output_dim(void* ctx) {
    snn_adapter_ctx_t* a = (snn_adapter_ctx_t*)ctx;
    return a ? a->output_dim : 0;
}

static uint32_t snn_adapter_get_input_dim(void* ctx) {
    snn_adapter_ctx_t* a = (snn_adapter_ctx_t*)ctx;
    return a ? a->input_dim : 0;
}

static float snn_adapter_auxiliary_loss(void* ctx) {
    /* Could add spike regularization loss here */
    (void)ctx;
    return 0.0f;
}

static void snn_adapter_destroy(void* ctx) {
    nimcp_free(ctx);
}

/* --- static vtable --- */

static const nimcp_trainable_network_ops_t snn_trainable_ops = {
    .name = "SNN",
    .type = NIMCP_TRAINABLE_SNN,
    .forward = snn_adapter_forward,
    .backward = snn_adapter_backward,
    .get_param_groups = snn_adapter_get_param_groups,
    .zero_grad = snn_adapter_zero_grad,
    .get_output_dim = snn_adapter_get_output_dim,
    .get_input_dim = snn_adapter_get_input_dim,
    .compute_auxiliary_loss = snn_adapter_auxiliary_loss,
    .destroy = snn_adapter_destroy,
};

/* --- public creation --- */

int nimcp_trainable_snn_create(struct snn_backprop_ctx_s* snn_ctx,
                               const nimcp_trainable_network_ops_t** ops,
                               void** ctx) {
    if (!snn_ctx || !ops || !ctx) return -1;

    snn_adapter_ctx_t* a = (snn_adapter_ctx_t*)nimcp_calloc(1, sizeof(snn_adapter_ctx_t));
    if (!a) return -1;

    a->snn_ctx = (snn_backprop_ctx_t*)snn_ctx;
    /* Dims set from brain config at registration time.
     * SNN doesn't expose dims via public API — these are set by
     * the caller after nimcp_trainable_snn_create(). */
    a->output_dim = 0;
    a->input_dim = 0;

    *ops = &snn_trainable_ops;
    *ctx = a;

    NIMCP_LOGGING_INFO("Created SNN trainable adapter");
    return 0;
}
