/**
 * @file nimcp_gpu_stubs_neuron.c
 * @brief CPU fallback stubs for AWS Neuron/Inferentia functions
 *
 * WHAT: Stub implementations when NIMCP_ENABLE_NEURON is not defined
 * WHY:  Clean builds on non-Inferentia systems (no NRT dependency)
 * HOW:  All functions return NULL/false/-1 indicating Neuron unavailable
 *
 * NOTE: These stubs are only compiled when NRT is not detected.
 * When NRT IS available, the real implementations in
 * src/gpu/neuron/nimcp_neuron_context.c and nimcp_neuron_bridge.c are used.
 *
 * @author NIMCP Development Team
 * @date 2026
 */

#ifndef NIMCP_ENABLE_NEURON

#include "gpu/neuron/nimcp_neuron_context.h"
#include "gpu/neuron/nimcp_neuron_bridge.h"

#include <stddef.h>

//=============================================================================
// Context Stubs
//=============================================================================

nimcp_neuron_context_t* nimcp_neuron_context_create(int device_id)
{
    (void)device_id;
    return NULL;
}

void nimcp_neuron_context_destroy(nimcp_neuron_context_t* ctx)
{
    (void)ctx;
}

bool nimcp_neuron_context_is_valid(const nimcp_neuron_context_t* ctx)
{
    (void)ctx;
    return false;
}

int nimcp_neuron_load_neff(nimcp_neuron_context_t* ctx, const char* neff_path)
{
    (void)ctx;
    (void)neff_path;
    return -1;
}

int nimcp_neuron_unload_model(nimcp_neuron_context_t* ctx)
{
    (void)ctx;
    return -1;
}

int nimcp_neuron_execute(nimcp_neuron_context_t* ctx,
                          const void* input, size_t input_size,
                          void* output, size_t output_size)
{
    (void)ctx;
    (void)input;
    (void)input_size;
    (void)output;
    (void)output_size;
    return -1;
}

//=============================================================================
// Bridge Stubs
//=============================================================================

nimcp_neuron_inference_cache_t* nimcp_neuron_cache_create(
    nimcp_neuron_context_t* ctx,
    const uint32_t* layer_sizes,
    uint32_t num_layers)
{
    (void)ctx;
    (void)layer_sizes;
    (void)num_layers;
    return NULL;
}

void nimcp_neuron_cache_destroy(nimcp_neuron_inference_cache_t* cache)
{
    (void)cache;
}

int nimcp_neuron_export_onnx(nimcp_neuron_inference_cache_t* cache,
                              neural_network_t net,
                              const char* onnx_path)
{
    (void)cache;
    (void)net;
    (void)onnx_path;
    return -1;
}

int nimcp_neuron_compile_neff(nimcp_neuron_inference_cache_t* cache,
                               neural_network_t net,
                               const char* output_path)
{
    (void)cache;
    (void)net;
    (void)output_path;
    return -1;
}

int nimcp_neuron_load_neff_model(nimcp_neuron_inference_cache_t* cache,
                                  const char* neff_path)
{
    (void)cache;
    (void)neff_path;
    return -1;
}

int nimcp_neuron_forward_pass(nimcp_neuron_inference_cache_t* cache,
                               const float* input, uint32_t input_size,
                               float* output, uint32_t output_size)
{
    (void)cache;
    (void)input;
    (void)input_size;
    (void)output;
    (void)output_size;
    return -1;
}

int nimcp_neuron_forward_pass_batch(nimcp_neuron_inference_cache_t* cache,
                                     const float* inputs, uint32_t batch_size,
                                     uint32_t input_size,
                                     float* outputs, uint32_t output_size)
{
    (void)cache;
    (void)inputs;
    (void)batch_size;
    (void)input_size;
    (void)outputs;
    (void)output_size;
    return -1;
}

void nimcp_neuron_invalidate_weights(nimcp_neuron_inference_cache_t* cache)
{
    (void)cache;
}

bool nimcp_neuron_is_ready(const nimcp_neuron_inference_cache_t* cache)
{
    (void)cache;
    return false;
}

#endif // !NIMCP_ENABLE_NEURON
