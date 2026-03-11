/**
 * @file nimcp_lnn_trainable.c
 * @brief Adapter: wraps lnn_training_ctx_t behind the unified trainable network interface
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "lnn/nimcp_lnn_training.h"
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
    (void)ctx; (void)input; (void)input_dim; (void)output; (void)output_dim;
    NIMCP_LOGGING_DEBUG("lnn_adapter_forward: stub (LNN uses ODE-based forward)");
    return 0;
}

static int lnn_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                float* dl_dinput, uint32_t input_dim) {
    (void)ctx; (void)dl_doutput; (void)output_dim; (void)dl_dinput; (void)input_dim;
    NIMCP_LOGGING_DEBUG("lnn_adapter_backward: stub (LNN uses adjoint method backward)");
    return 0;
}

static int lnn_adapter_get_param_groups(void* ctx,
                                        nimcp_utm_param_group_t** groups,
                                        uint32_t* num_groups) {
    (void)ctx;
    if (!groups || !num_groups) return -1;
    /* Phase 3: will expose W_in, W_rec, tau_base, b_in, b_tau per layer
     * with appropriate lr_scale (tau=0.1x, bias=2.0x) */
    *groups = NULL;
    *num_groups = 0;
    return 0;
}

static int lnn_adapter_zero_grad(void* ctx) {
    lnn_adapter_ctx_t* a = (lnn_adapter_ctx_t*)ctx;
    if (!a || !a->lnn_ctx) return -1;
    /* LNN gradient reset is done via lnn_gradient_reset() in training loop */
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
    a->output_dim = 0;
    a->input_dim = 0;

    *ops = &lnn_trainable_ops;
    *ctx = a;

    NIMCP_LOGGING_INFO("Created LNN trainable adapter");
    return 0;
}
