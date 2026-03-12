/**
 * @file nimcp_cnn_trainable.c
 * @brief Adapter: wraps cnn_trainer_t behind the unified trainable network interface
 *
 * @author NIMCP Development Team
 * @date 2026-03-11
 */

#include "training/nimcp_unified_training.h"
#include "training/nimcp_cnn_training.h"
#include "utils/tensor/nimcp_tensor.h"
#include "utils/memory/nimcp_unified_memory.h"
#include "utils/logging/nimcp_logging.h"

#include <string.h>

typedef struct {
    cnn_trainer_t* trainer;     /* NOT owned — must outlive adapter */
    cnn_forward_result_t last_result; /* Cached forward result for backward */
    bool has_result;
    uint32_t output_dim;
    uint32_t input_dim;
    bool managed_by_utm;        /* When true, skip internal optimizer step */
} cnn_adapter_ctx_t;

/* --- vtable implementations --- */

static int cnn_adapter_forward(void* ctx, const float* input, uint32_t input_dim,
                               float* output, uint32_t output_dim) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer || !input || !output) return -1;

    /* Wrap raw floats into 1D tensor for CNN forward */
    uint32_t dims[1] = { input_dim };
    nimcp_tensor_t* in_t = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!in_t) return -1;

    memcpy(nimcp_tensor_data(in_t), input, input_dim * sizeof(float));

    memset(&a->last_result, 0, sizeof(a->last_result));
    nimcp_error_t err = cnn_trainer_forward(a->trainer, in_t, &a->last_result);

    if (err == NIMCP_SUCCESS && a->last_result.output) {
        const float* out_data = (const float*)nimcp_tensor_data_const(a->last_result.output);
        size_t out_numel = nimcp_tensor_numel(a->last_result.output);
        uint32_t copy_dim = (output_dim < (uint32_t)out_numel) ?
                             output_dim : (uint32_t)out_numel;
        memcpy(output, out_data, copy_dim * sizeof(float));
        a->has_result = true;
    }

    nimcp_tensor_destroy(in_t);
    return (err == NIMCP_SUCCESS) ? 0 : -1;
}

static int cnn_adapter_backward(void* ctx, const float* dl_doutput, uint32_t output_dim,
                                float* dl_dinput, uint32_t input_dim) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer || !dl_doutput || !a->has_result) return -1;

    /* CNN backward expects target tensor (MSE gradient computed internally).
     * We wrap dl_doutput as the "target" — backward computes gradients. */
    uint32_t dims[1] = { output_dim };
    nimcp_tensor_t* target_t = nimcp_tensor_create(dims, 1, NIMCP_DTYPE_F32);
    if (!target_t) return -1;

    memcpy(nimcp_tensor_data(target_t), dl_doutput, output_dim * sizeof(float));

    nimcp_error_t err = cnn_trainer_backward(a->trainer, target_t, &a->last_result);

    /* Run CNN's internal optimizer step after backward — unless UTM manages params */
    if (err == NIMCP_SUCCESS && !a->managed_by_utm) {
        cnn_trainer_step(a->trainer);
    }

    /* Copy real input gradients from CNN backward pass if available */
    if (dl_dinput) {
        const nimcp_tensor_t* in_grad = cnn_trainer_get_input_grad(a->trainer);
        if (in_grad) {
            const float* grad_data = (const float*)nimcp_tensor_data_const(in_grad);
            size_t grad_numel = nimcp_tensor_numel(in_grad);
            uint32_t copy_dim = (input_dim < (uint32_t)grad_numel) ?
                                 input_dim : (uint32_t)grad_numel;
            memcpy(dl_dinput, grad_data, copy_dim * sizeof(float));
            /* Zero any remaining elements if input_dim > grad tensor size */
            if (input_dim > copy_dim) {
                memset(dl_dinput + copy_dim, 0, (input_dim - copy_dim) * sizeof(float));
            }
        } else {
            memset(dl_dinput, 0, input_dim * sizeof(float));
        }
    }

    nimcp_tensor_destroy(target_t);
    return (err == NIMCP_SUCCESS) ? 0 : -1;
}

static int cnn_adapter_get_param_groups(void* ctx,
                                        nimcp_utm_param_group_t** groups,
                                        uint32_t* num_groups) {
    cnn_adapter_ctx_t* a = (cnn_adapter_ctx_t*)ctx;
    if (!a || !a->trainer || !groups || !num_groups) return -1;
    /* CNN parameters managed by cnn_trainer_step() / internal optimizer */
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
    /* Don't free last_result — CNN trainer owns the activation tensors */
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
    a->output_dim = 0;  /* Set at registration time from brain config */
    a->input_dim = 0;

    *ops = &cnn_trainable_ops;
    *ctx = a;

    NIMCP_LOGGING_INFO("Created CNN trainable adapter");
    return 0;
}
