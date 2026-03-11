/**
 * @file nimcp_adaptive_trainable.c
 * @brief Adapter: wraps neural_network_struct behind the unified trainable network interface
 *
 * The adaptive (backbone) network uses GPU-accelerated forward/backward via
 * nimcp_training_bridge.c. This adapter exposes it to the unified manager.
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>

typedef struct {
    neural_network_t network;       /* NOT owned — pointer typedef */
    uint32_t output_dim;
    uint32_t input_dim;
} adaptive_adapter_ctx_t;

/* --- vtable implementations --- */

static int adaptive_adapter_forward(void* ctx, const float* input, uint32_t input_dim,
                                    float* output, uint32_t output_dim) {
    (void)ctx; (void)input; (void)input_dim; (void)output; (void)output_dim;
    /* Phase 1 stub: adaptive network uses GPU bridge forward via brain_decide/brain_learn */
    NIMCP_LOGGING_DEBUG("adaptive_adapter_forward: stub (uses GPU training bridge)");
    return 0;
}

static int adaptive_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                     float* dl_dinput, uint32_t input_dim) {
    (void)ctx; (void)dl_doutput; (void)output_dim; (void)dl_dinput; (void)input_dim;
    NIMCP_LOGGING_DEBUG("adaptive_adapter_backward: stub (uses GPU training bridge)");
    return 0;
}

static int adaptive_adapter_get_param_groups(void* ctx,
                                             nimcp_utm_param_group_t** groups,
                                             uint32_t* num_groups) {
    (void)ctx;
    if (!groups || !num_groups) return -1;
    /* Phase 3: will expose GPU weight cache as param groups */
    *groups = NULL;
    *num_groups = 0;
    return 0;
}

static int adaptive_adapter_zero_grad(void* ctx) {
    (void)ctx;
    /* GPU bridge handles its own gradient zeroing */
    return 0;
}

static uint32_t adaptive_adapter_get_output_dim(void* ctx) {
    adaptive_adapter_ctx_t* a = (adaptive_adapter_ctx_t*)ctx;
    return a ? a->output_dim : 0;
}

static uint32_t adaptive_adapter_get_input_dim(void* ctx) {
    adaptive_adapter_ctx_t* a = (adaptive_adapter_ctx_t*)ctx;
    return a ? a->input_dim : 0;
}

static float adaptive_adapter_auxiliary_loss(void* ctx) {
    (void)ctx;
    return 0.0f;
}

static void adaptive_adapter_destroy(void* ctx) {
    nimcp_free(ctx);
}

/* --- static vtable --- */

static const nimcp_trainable_network_ops_t adaptive_trainable_ops = {
    .name = "Adaptive",
    .type = NIMCP_TRAINABLE_ADAPTIVE,
    .forward = adaptive_adapter_forward,
    .backward = adaptive_adapter_backward,
    .get_param_groups = adaptive_adapter_get_param_groups,
    .zero_grad = adaptive_adapter_zero_grad,
    .get_output_dim = adaptive_adapter_get_output_dim,
    .get_input_dim = adaptive_adapter_get_input_dim,
    .compute_auxiliary_loss = adaptive_adapter_auxiliary_loss,
    .destroy = adaptive_adapter_destroy,
};

/* --- public creation --- */

int nimcp_trainable_adaptive_create(struct neural_network_struct* network,
                                    const nimcp_trainable_network_ops_t** ops,
                                    void** ctx) {
    if (!network || !ops || !ctx) return -1;

    adaptive_adapter_ctx_t* a = (adaptive_adapter_ctx_t*)nimcp_calloc(
        1, sizeof(adaptive_adapter_ctx_t));
    if (!a) return -1;

    a->network = network;  /* neural_network_t is already a pointer */
    a->output_dim = 0;
    a->input_dim = 0;

    *ops = &adaptive_trainable_ops;
    *ctx = a;

    NIMCP_LOGGING_INFO("Created Adaptive trainable adapter");
    return 0;
}
