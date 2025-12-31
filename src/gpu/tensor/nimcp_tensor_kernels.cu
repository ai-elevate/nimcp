/**
 * @file nimcp_tensor_kernels.cu
 * @brief GPU Tensor CUDA Kernels Implementation
 *
 * WHAT: CUDA kernels for tensor operations
 * WHY:  GPU acceleration for neural network computations
 * HOW:  cuBLAS for GEMM, custom kernels for element-wise and reductions
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "utils/logging/nimcp_logging.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math.h>
#include <float.h>

#define LOG_MODULE "TENSOR_GPU"

//=============================================================================
// CUDA Error Checking
//=============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        LOG_ERROR("cuBLAS error at %s:%d: %d", __FILE__, __LINE__, status); \
        return false; \
    } \
} while(0)

//=============================================================================
// Kernel Configuration
//=============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Calculate grid size for N elements
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

//=============================================================================
// Helper Functions
//=============================================================================

static size_t get_elem_size(nimcp_gpu_precision_t precision)
{
    switch (precision) {
        case NIMCP_GPU_PRECISION_FP32: return sizeof(float);
        case NIMCP_GPU_PRECISION_FP16: return sizeof(__half);
        case NIMCP_GPU_PRECISION_BF16: return sizeof(__nv_bfloat16);
        case NIMCP_GPU_PRECISION_INT8: return sizeof(int8_t);
        case NIMCP_GPU_PRECISION_TF32: return sizeof(float);
        default: return sizeof(float);
    }
}

static size_t compute_numel(const size_t* dims, uint32_t ndim)
{
    if (!dims || ndim == 0) return 0;
    size_t numel = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        numel *= dims[i];
    }
    return numel;
}

//=============================================================================
// Tensor Lifecycle
//=============================================================================

nimcp_gpu_tensor_t* nimcp_gpu_tensor_create(
    nimcp_gpu_context_t* ctx,
    const size_t* dims,
    uint32_t ndim,
    nimcp_gpu_precision_t precision)
{
    if (!ctx || !dims || ndim == 0) {
        LOG_ERROR("Invalid parameters for tensor creation");
        return NULL;
    }

    nimcp_gpu_tensor_t* tensor = (nimcp_gpu_tensor_t*)calloc(1, sizeof(nimcp_gpu_tensor_t));
    if (!tensor) {
        LOG_ERROR("Failed to allocate tensor structure");
        return NULL;
    }

    tensor->ndim = ndim;
    tensor->precision = precision;
    tensor->layout = NIMCP_TENSOR_LAYOUT_ROW_MAJOR;
    tensor->ctx = ctx;
    tensor->owns_data = true;
    tensor->elem_size = get_elem_size(precision);

    // Allocate dimensions array on host
    tensor->dims = (size_t*)malloc(ndim * sizeof(size_t));
    tensor->strides = (size_t*)malloc(ndim * sizeof(size_t));
    if (!tensor->dims || !tensor->strides) {
        LOG_ERROR("Failed to allocate dimension arrays");
        free(tensor->dims);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    // Copy dimensions and compute strides (row-major)
    tensor->numel = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        tensor->dims[i] = dims[i];
        tensor->strides[i] = tensor->numel;
        tensor->numel *= dims[i];
    }

    // Allocate device memory
    size_t data_size = tensor->numel * tensor->elem_size;
    cudaError_t err = cudaMalloc(&tensor->data, data_size);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to allocate %zu bytes on GPU: %s", data_size, cudaGetErrorString(err));
        free(tensor->dims);
        free(tensor->strides);
        free(tensor);
        return NULL;
    }

    // Zero-initialize
    cudaMemset(tensor->data, 0, data_size);

    LOG_DEBUG("Created GPU tensor: %u dims, %zu elements, %zu bytes", ndim, tensor->numel, data_size);
    return tensor;
}

nimcp_gpu_tensor_t* nimcp_gpu_tensor_from_host(
    nimcp_gpu_context_t* ctx,
    const void* host_data,
    const size_t* dims,
    uint32_t ndim,
    nimcp_gpu_precision_t precision)
{
    if (!host_data) {
        LOG_ERROR("NULL host data provided");
        return NULL;
    }

    nimcp_gpu_tensor_t* tensor = nimcp_gpu_tensor_create(ctx, dims, ndim, precision);
    if (!tensor) {
        return NULL;
    }

    size_t data_size = tensor->numel * tensor->elem_size;
    cudaError_t err = cudaMemcpy(tensor->data, host_data, data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to copy data to GPU: %s", cudaGetErrorString(err));
        nimcp_gpu_tensor_destroy(tensor);
        return NULL;
    }

    return tensor;
}

bool nimcp_gpu_tensor_to_host(const nimcp_gpu_tensor_t* tensor, void* host_data)
{
    if (!tensor || !host_data) {
        LOG_ERROR("Invalid parameters for tensor to host copy");
        return false;
    }

    size_t data_size = tensor->numel * tensor->elem_size;
    CUDA_CHECK(cudaMemcpy(host_data, tensor->data, data_size, cudaMemcpyDeviceToHost));
    return true;
}

void nimcp_gpu_tensor_destroy(nimcp_gpu_tensor_t* tensor)
{
    if (!tensor) return;

    if (tensor->owns_data && tensor->data) {
        cudaFree(tensor->data);
    }

    free(tensor->dims);
    free(tensor->strides);
    free(tensor);
}

nimcp_gpu_tensor_t* nimcp_gpu_tensor_clone(const nimcp_gpu_tensor_t* tensor)
{
    if (!tensor) return NULL;

    nimcp_gpu_tensor_t* clone = nimcp_gpu_tensor_create(
        tensor->ctx, tensor->dims, tensor->ndim, tensor->precision);
    if (!clone) return NULL;

    size_t data_size = tensor->numel * tensor->elem_size;
    cudaError_t err = cudaMemcpy(clone->data, tensor->data, data_size, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        nimcp_gpu_tensor_destroy(clone);
        return NULL;
    }

    return clone;
}

//=============================================================================
// GEMM Operations (using cuBLAS)
//=============================================================================

bool nimcp_gpu_gemm(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* A,
    const nimcp_gpu_tensor_t* B,
    nimcp_gpu_tensor_t* C,
    float alpha,
    float beta,
    bool trans_a,
    bool trans_b)
{
    if (!ctx || !A || !B || !C) {
        LOG_ERROR("Invalid parameters for GEMM");
        return false;
    }

    if (A->ndim < 2 || B->ndim < 2 || C->ndim < 2) {
        LOG_ERROR("GEMM requires at least 2D tensors");
        return false;
    }

    // Get dimensions (assuming last two dims are matrix dims)
    int M = trans_a ? A->dims[A->ndim - 1] : A->dims[A->ndim - 2];
    int K_a = trans_a ? A->dims[A->ndim - 2] : A->dims[A->ndim - 1];
    int K_b = trans_b ? B->dims[B->ndim - 1] : B->dims[B->ndim - 2];
    int N = trans_b ? B->dims[B->ndim - 2] : B->dims[B->ndim - 1];

    if (K_a != K_b) {
        LOG_ERROR("GEMM dimension mismatch: K_a=%d, K_b=%d", K_a, K_b);
        return false;
    }

    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;
    cublasOperation_t opA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Leading dimensions (column-major for cuBLAS)
    int lda = trans_a ? K_a : M;
    int ldb = trans_b ? N : K_a;
    int ldc = M;

    // cuBLAS uses column-major, so we compute B^T @ A^T = (A @ B)^T
    // which gives us C in row-major format
    CUBLAS_CHECK(cublasSgemm(handle,
        opB, opA,
        N, M, K_a,
        &alpha,
        (const float*)B->data, trans_b ? K_a : N,
        (const float*)A->data, trans_a ? M : K_a,
        &beta,
        (float*)C->data, N));

    return true;
}

bool nimcp_gpu_gemv(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* A,
    const nimcp_gpu_tensor_t* x,
    nimcp_gpu_tensor_t* y,
    float alpha,
    float beta,
    bool trans_a)
{
    if (!ctx || !A || !x || !y) {
        LOG_ERROR("Invalid parameters for GEMV");
        return false;
    }

    int M = A->dims[A->ndim - 2];
    int N = A->dims[A->ndim - 1];

    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;
    cublasOperation_t op = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    // For row-major A, we use transposed operation
    CUBLAS_CHECK(cublasSgemv(handle,
        trans_a ? CUBLAS_OP_N : CUBLAS_OP_T,
        N, M,
        &alpha,
        (const float*)A->data, N,
        (const float*)x->data, 1,
        &beta,
        (float*)y->data, 1));

    return true;
}

bool nimcp_gpu_gemm_batched(
    nimcp_gpu_context_t* ctx,
    const nimcp_gpu_tensor_t* A,
    const nimcp_gpu_tensor_t* B,
    nimcp_gpu_tensor_t* C,
    float alpha,
    float beta,
    bool trans_a,
    bool trans_b)
{
    if (!ctx || !A || !B || !C) {
        LOG_ERROR("Invalid parameters for batched GEMM");
        return false;
    }

    if (A->ndim < 3 || B->ndim < 3 || C->ndim < 3) {
        LOG_ERROR("Batched GEMM requires at least 3D tensors");
        return false;
    }

    int batch_size = A->dims[0];
    int M = trans_a ? A->dims[2] : A->dims[1];
    int K = trans_a ? A->dims[1] : A->dims[2];
    int N = trans_b ? B->dims[1] : B->dims[2];

    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;
    cublasOperation_t opA = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t opB = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    long long strideA = M * K;
    long long strideB = K * N;
    long long strideC = M * N;

    CUBLAS_CHECK(cublasSgemmStridedBatched(handle,
        opB, opA,
        N, M, K,
        &alpha,
        (const float*)B->data, trans_b ? K : N, strideB,
        (const float*)A->data, trans_a ? M : K, strideA,
        &beta,
        (float*)C->data, N, strideC,
        batch_size));

    return true;
}

//=============================================================================
// Element-wise Kernels
//=============================================================================

__global__ void kernel_add(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void kernel_sub(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

__global__ void kernel_mul(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

__global__ void kernel_div(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] / b[idx];
    }
}

__global__ void kernel_add_scalar(const float* a, float scalar, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + scalar;
    }
}

__global__ void kernel_mul_scalar(const float* a, float scalar, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

bool nimcp_gpu_add(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    if (a->numel != b->numel || a->numel != out->numel) {
        LOG_ERROR("Tensor size mismatch for add");
        return false;
    }
    kernel_add<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_sub(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    if (a->numel != b->numel || a->numel != out->numel) return false;
    kernel_sub<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_mul(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    if (a->numel != b->numel || a->numel != out->numel) return false;
    kernel_mul<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_div(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    if (a->numel != b->numel || a->numel != out->numel) return false;
    kernel_div<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_add_scalar(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                          float scalar, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !out) return false;
    kernel_add_scalar<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, scalar, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_mul_scalar(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                          float scalar, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !out) return false;
    kernel_mul_scalar<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, scalar, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Activation Kernels
//=============================================================================

__global__ void kernel_relu(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fmaxf(0.0f, x[idx]);
    }
}

__global__ void kernel_leaky_relu(const float* x, float* out, float alpha, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = val > 0.0f ? val : alpha * val;
    }
}

__global__ void kernel_sigmoid(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

__global__ void kernel_tanh_activation(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = tanhf(x[idx]);
    }
}

__global__ void kernel_gelu(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        float val = x[idx];
        float val_cubed = val * val * val;
        float inner = 0.7978845608f * (val + 0.044715f * val_cubed);  // sqrt(2/pi) ~ 0.7978845608
        out[idx] = 0.5f * val * (1.0f + tanhf(inner));
    }
}

__global__ void kernel_silu(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));  // x * sigmoid(x)
    }
}

bool nimcp_gpu_relu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_relu<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_leaky_relu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                          nimcp_gpu_tensor_t* out, float alpha)
{
    if (!ctx || !x || !out) return false;
    kernel_leaky_relu<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>(
        (const float*)x->data, (float*)out->data, alpha, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_sigmoid(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_sigmoid<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_tanh(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_tanh_activation<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_gelu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_gelu<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_silu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_silu<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Softmax Kernel (Numerically Stable)
//=============================================================================

__global__ void kernel_softmax_1d(const float* x, float* out, size_t batch_size, size_t dim_size)
{
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x_row = x + batch_idx * dim_size;
    float* out_row = out + batch_idx * dim_size;

    // Step 1: Find max (for numerical stability)
    __shared__ float shared_max[BLOCK_SIZE];
    float thread_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];

    // Step 2: Compute exp(x - max) and sum
    __shared__ float shared_sum[BLOCK_SIZE];
    float thread_sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float exp_val = expf(x_row[i] - max_val);
        out_row[i] = exp_val;  // Store temporarily
        thread_sum += exp_val;
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce to find sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float sum_val = shared_sum[0];

    // Step 3: Normalize
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        out_row[i] = out_row[i] / sum_val;
    }
}

bool nimcp_gpu_softmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    if (x->ndim < 1) return false;

    size_t dim_size = x->dims[x->ndim - 1];
    size_t batch_size = x->numel / dim_size;

    kernel_softmax_1d<<<batch_size, BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, batch_size, dim_size);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

__global__ void kernel_log_softmax_1d(const float* x, float* out, size_t batch_size, size_t dim_size)
{
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    const float* x_row = x + batch_idx * dim_size;
    float* out_row = out + batch_idx * dim_size;

    // Step 1: Find max
    __shared__ float shared_max[BLOCK_SIZE];
    float thread_max = -FLT_MAX;
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, x_row[i]);
    }
    shared_max[threadIdx.x] = thread_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_max[threadIdx.x] = fmaxf(shared_max[threadIdx.x], shared_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    float max_val = shared_max[0];

    // Step 2: Compute sum of exp(x - max)
    __shared__ float shared_sum[BLOCK_SIZE];
    float thread_sum = 0.0f;
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_sum += expf(x_row[i] - max_val);
    }
    shared_sum[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    float log_sum = logf(shared_sum[0]);

    // Step 3: Compute log_softmax = x - max - log(sum)
    for (size_t i = threadIdx.x; i < dim_size; i += blockDim.x) {
        out_row[i] = x_row[i] - max_val - log_sum;
    }
}

bool nimcp_gpu_log_softmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    if (x->ndim < 1) return false;

    size_t dim_size = x->dims[x->ndim - 1];
    size_t batch_size = x->numel / dim_size;

    kernel_log_softmax_1d<<<batch_size, BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, batch_size, dim_size);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Math Kernels
//=============================================================================

__global__ void kernel_exp(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = expf(x[idx]);
    }
}

__global__ void kernel_log(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = logf(x[idx]);
    }
}

__global__ void kernel_sqrt(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = sqrtf(x[idx]);
    }
}

__global__ void kernel_pow(const float* x, float exponent, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = powf(x[idx], exponent);
    }
}

__global__ void kernel_abs(const float* x, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fabsf(x[idx]);
    }
}

__global__ void kernel_clamp(const float* x, float min_val, float max_val, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = fminf(fmaxf(x[idx], min_val), max_val);
    }
}

bool nimcp_gpu_exp(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_exp<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_log(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_log<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_sqrt(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_sqrt<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_pow(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float exponent, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_pow<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, exponent, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_abs(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_abs<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_clamp(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                     float min_val, float max_val, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    kernel_clamp<<<GRID_SIZE(x->numel), BLOCK_SIZE>>>(
        (const float*)x->data, min_val, max_val, (float*)out->data, x->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Reduction Kernels with Warp Shuffle
//=============================================================================

__device__ inline float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ inline float warp_reduce_max(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ inline float warp_reduce_min(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void kernel_reduce_sum(const float* x, float* out, size_t n)
{
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    float sum = 0.0f;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        sum += x[i];
    }

    // Warp-level reduction
    sum = warp_reduce_sum(sum);

    // First thread in each warp writes to shared memory
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces shared memory
    if (warp_id == 0) {
        sum = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(out, sum);
        }
    }
}

__global__ void kernel_reduce_max(const float* x, float* out, size_t n)
{
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    float max_val = -FLT_MAX;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        max_val = fmaxf(max_val, x[i]);
    }

    max_val = warp_reduce_max(max_val);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : -FLT_MAX;
        max_val = warp_reduce_max(max_val);
        if (lane == 0) {
            // Atomic max for float (using atomicCAS)
            int* out_int = (int*)out;
            int old = *out_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(out_int, assumed,
                    __float_as_int(fmaxf(max_val, __int_as_float(assumed))));
            } while (assumed != old);
        }
    }
}

__global__ void kernel_reduce_min(const float* x, float* out, size_t n)
{
    __shared__ float shared[BLOCK_SIZE / WARP_SIZE];

    float min_val = FLT_MAX;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        min_val = fminf(min_val, x[i]);
    }

    min_val = warp_reduce_min(min_val);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (lane < blockDim.x / WARP_SIZE) ? shared[lane] : FLT_MAX;
        min_val = warp_reduce_min(min_val);
        if (lane == 0) {
            int* out_int = (int*)out;
            int old = *out_int, assumed;
            do {
                assumed = old;
                old = atomicCAS(out_int, assumed,
                    __float_as_int(fminf(min_val, __int_as_float(assumed))));
            } while (assumed != old);
        }
    }
}

bool nimcp_gpu_sum(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims)
{
    if (!ctx || !x || !out) return false;

    // Full reduction (all elements)
    if (axis < 0) {
        CUDA_CHECK(cudaMemset(out->data, 0, sizeof(float)));
        int grid = (x->numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid = grid > 256 ? 256 : grid;  // Limit grid size
        kernel_reduce_sum<<<grid, BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    // TODO: Implement axis-specific reduction
    LOG_WARN("Axis-specific reduction not yet implemented, using full reduction");
    return nimcp_gpu_sum(ctx, x, out, -1, keepdims);
}

bool nimcp_gpu_mean(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                    nimcp_gpu_tensor_t* out, int axis, bool keepdims)
{
    if (!nimcp_gpu_sum(ctx, x, out, axis, keepdims)) return false;

    size_t n = (axis < 0) ? x->numel : x->dims[axis];
    float scale = 1.0f / (float)n;

    kernel_mul_scalar<<<1, 1>>>((const float*)out->data, scale, (float*)out->data, 1);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_max(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims)
{
    if (!ctx || !x || !out) return false;

    if (axis < 0) {
        float init = -FLT_MAX;
        CUDA_CHECK(cudaMemcpy(out->data, &init, sizeof(float), cudaMemcpyHostToDevice));
        int grid = (x->numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid = grid > 256 ? 256 : grid;
        kernel_reduce_max<<<grid, BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    LOG_WARN("Axis-specific max reduction not yet implemented");
    return nimcp_gpu_max(ctx, x, out, -1, keepdims);
}

bool nimcp_gpu_min(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims)
{
    if (!ctx || !x || !out) return false;

    if (axis < 0) {
        float init = FLT_MAX;
        CUDA_CHECK(cudaMemcpy(out->data, &init, sizeof(float), cudaMemcpyHostToDevice));
        int grid = (x->numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
        grid = grid > 256 ? 256 : grid;
        kernel_reduce_min<<<grid, BLOCK_SIZE>>>((const float*)x->data, (float*)out->data, x->numel);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    LOG_WARN("Axis-specific min reduction not yet implemented");
    return nimcp_gpu_min(ctx, x, out, -1, keepdims);
}

//=============================================================================
// Argmax/Argmin Kernels
//=============================================================================

__global__ void kernel_argmax(const float* x, int64_t* out, size_t n)
{
    __shared__ float shared_val[BLOCK_SIZE];
    __shared__ int shared_idx[BLOCK_SIZE];

    float max_val = -FLT_MAX;
    int max_idx = 0;

    for (size_t i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > max_val) {
            max_val = x[i];
            max_idx = i;
        }
    }

    shared_val[threadIdx.x] = max_val;
    shared_idx[threadIdx.x] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shared_val[threadIdx.x + stride] > shared_val[threadIdx.x]) {
                shared_val[threadIdx.x] = shared_val[threadIdx.x + stride];
                shared_idx[threadIdx.x] = shared_idx[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *out = shared_idx[0];
    }
}

bool nimcp_gpu_argmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                      nimcp_gpu_tensor_t* out, int axis)
{
    if (!ctx || !x || !out) return false;

    if (axis < 0) {
        kernel_argmax<<<1, BLOCK_SIZE>>>((const float*)x->data, (int64_t*)out->data, x->numel);
        CUDA_CHECK(cudaGetLastError());
        return true;
    }

    LOG_WARN("Axis-specific argmax not yet implemented");
    return nimcp_gpu_argmax(ctx, x, out, -1);
}

bool nimcp_gpu_argmin(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                      nimcp_gpu_tensor_t* out, int axis)
{
    // Similar to argmax but with min comparison
    // TODO: Implement properly
    LOG_WARN("argmin not fully implemented");
    return false;
}

//=============================================================================
// Variance and Standard Deviation
//=============================================================================

bool nimcp_gpu_var(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims, bool unbiased)
{
    // Var = E[X^2] - E[X]^2
    // For now, compute on CPU side with GPU data
    LOG_WARN("Variance not fully optimized for GPU");

    // Get mean
    float mean_val = 0.0f;
    nimcp_gpu_tensor_t* mean_tensor = nimcp_gpu_tensor_create(ctx, (size_t[]){1}, 1, x->precision);
    if (!mean_tensor) return false;

    nimcp_gpu_mean(ctx, x, mean_tensor, -1, false);
    nimcp_gpu_tensor_to_host(mean_tensor, &mean_val);

    // Compute variance on CPU (temporary solution)
    float* host_data = (float*)malloc(x->numel * sizeof(float));
    nimcp_gpu_tensor_to_host(x, host_data);

    float var = 0.0f;
    for (size_t i = 0; i < x->numel; i++) {
        float diff = host_data[i] - mean_val;
        var += diff * diff;
    }
    var /= unbiased ? (x->numel - 1) : x->numel;

    cudaMemcpy(out->data, &var, sizeof(float), cudaMemcpyHostToDevice);

    free(host_data);
    nimcp_gpu_tensor_destroy(mean_tensor);
    return true;
}

bool nimcp_gpu_std(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims, bool unbiased)
{
    if (!nimcp_gpu_var(ctx, x, out, axis, keepdims, unbiased)) return false;
    kernel_sqrt<<<1, 1>>>((const float*)out->data, (float*)out->data, 1);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Norm Operations
//=============================================================================

bool nimcp_gpu_norm_l1(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result)
{
    if (!ctx || !x || !result) return false;

    // Create temp tensor for abs values
    nimcp_gpu_tensor_t* abs_tensor = nimcp_gpu_tensor_clone(x);
    if (!abs_tensor) return false;

    nimcp_gpu_abs(ctx, x, abs_tensor);

    nimcp_gpu_tensor_t* sum_tensor = nimcp_gpu_tensor_create(ctx, (size_t[]){1}, 1, x->precision);
    nimcp_gpu_sum(ctx, abs_tensor, sum_tensor, -1, false);
    nimcp_gpu_tensor_to_host(sum_tensor, result);

    nimcp_gpu_tensor_destroy(abs_tensor);
    nimcp_gpu_tensor_destroy(sum_tensor);
    return true;
}

bool nimcp_gpu_norm_l2(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result)
{
    if (!ctx || !x || !result) return false;

    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;
    CUBLAS_CHECK(cublasSnrm2(handle, x->numel, (const float*)x->data, 1, result));
    return true;
}

bool nimcp_gpu_norm_linf(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result)
{
    if (!ctx || !x || !result) return false;

    nimcp_gpu_tensor_t* abs_tensor = nimcp_gpu_tensor_clone(x);
    if (!abs_tensor) return false;

    nimcp_gpu_abs(ctx, x, abs_tensor);

    nimcp_gpu_tensor_t* max_tensor = nimcp_gpu_tensor_create(ctx, (size_t[]){1}, 1, x->precision);
    nimcp_gpu_max(ctx, abs_tensor, max_tensor, -1, false);
    nimcp_gpu_tensor_to_host(max_tensor, result);

    nimcp_gpu_tensor_destroy(abs_tensor);
    nimcp_gpu_tensor_destroy(max_tensor);
    return true;
}

bool nimcp_gpu_norm_frobenius(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result)
{
    // Frobenius norm is same as L2 norm for flattened tensor
    return nimcp_gpu_norm_l2(ctx, x, result);
}

//=============================================================================
// Comparison Operations
//=============================================================================

__global__ void kernel_eq(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_gt(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_lt(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_ge(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_le(const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
    }
}

__global__ void kernel_where(const float* cond, const float* a, const float* b, float* out, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (cond[idx] != 0.0f) ? a[idx] : b[idx];
    }
}

bool nimcp_gpu_eq(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    kernel_eq<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_gt(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    kernel_gt<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_lt(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    kernel_lt<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_ge(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    kernel_ge<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_le(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !a || !b || !out) return false;
    kernel_le<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_where(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* cond,
                     const nimcp_gpu_tensor_t* a, const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !cond || !a || !b || !out) return false;
    kernel_where<<<GRID_SIZE(a->numel), BLOCK_SIZE>>>(
        (const float*)cond->data, (const float*)a->data, (const float*)b->data, (float*)out->data, a->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

//=============================================================================
// Memory Operations
//=============================================================================

__global__ void kernel_fill(float* data, float value, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

bool nimcp_gpu_fill(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* tensor, float value)
{
    if (!ctx || !tensor) return false;
    kernel_fill<<<GRID_SIZE(tensor->numel), BLOCK_SIZE>>>((float*)tensor->data, value, tensor->numel);
    CUDA_CHECK(cudaGetLastError());
    return true;
}

bool nimcp_gpu_zeros(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* tensor)
{
    if (!ctx || !tensor) return false;
    CUDA_CHECK(cudaMemset(tensor->data, 0, tensor->numel * tensor->elem_size));
    return true;
}

bool nimcp_gpu_ones(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* tensor)
{
    return nimcp_gpu_fill(ctx, tensor, 1.0f);
}

bool nimcp_gpu_copy(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* src, nimcp_gpu_tensor_t* dst)
{
    if (!ctx || !src || !dst) return false;
    if (src->numel != dst->numel) return false;
    CUDA_CHECK(cudaMemcpy(dst->data, src->data, src->numel * src->elem_size, cudaMemcpyDeviceToDevice));
    return true;
}

bool nimcp_gpu_transpose(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out)
{
    if (!ctx || !x || !out) return false;
    if (x->ndim < 2) return false;

    // Use cuBLAS for 2D transpose
    size_t rows = x->dims[x->ndim - 2];
    size_t cols = x->dims[x->ndim - 1];

    cublasHandle_t handle = (cublasHandle_t)ctx->cublas_handle;
    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgeam(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        rows, cols,
        &alpha, (const float*)x->data, cols,
        &beta, NULL, rows,
        (float*)out->data, rows));

    return true;
}

bool nimcp_gpu_reshape(nimcp_gpu_tensor_t* tensor, const size_t* new_dims, uint32_t new_ndim)
{
    if (!tensor || !new_dims || new_ndim == 0) return false;

    size_t new_numel = compute_numel(new_dims, new_ndim);
    if (new_numel != tensor->numel) {
        LOG_ERROR("Cannot reshape: numel mismatch (%zu vs %zu)", tensor->numel, new_numel);
        return false;
    }

    // Reallocate dimension arrays
    size_t* new_dims_array = (size_t*)realloc(tensor->dims, new_ndim * sizeof(size_t));
    size_t* new_strides = (size_t*)realloc(tensor->strides, new_ndim * sizeof(size_t));
    if (!new_dims_array || !new_strides) return false;

    tensor->dims = new_dims_array;
    tensor->strides = new_strides;
    tensor->ndim = new_ndim;

    // Recompute strides
    size_t stride = 1;
    for (int i = new_ndim - 1; i >= 0; i--) {
        tensor->dims[i] = new_dims[i];
        tensor->strides[i] = stride;
        stride *= new_dims[i];
    }

    return true;
}

#else // !NIMCP_ENABLE_CUDA

//=============================================================================
// CPU Fallback Stubs
//=============================================================================

#include "gpu/tensor/nimcp_tensor_gpu.h"
#include "utils/logging/nimcp_logging.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LOG_MODULE "TENSOR_GPU"

nimcp_gpu_tensor_t* nimcp_gpu_tensor_create(nimcp_gpu_context_t* ctx, const size_t* dims,
                                             uint32_t ndim, nimcp_gpu_precision_t precision)
{
    LOG_WARN("CUDA not available - tensor operations will use CPU fallback");
    return NULL;
}

nimcp_gpu_tensor_t* nimcp_gpu_tensor_from_host(nimcp_gpu_context_t* ctx, const void* host_data,
                                                const size_t* dims, uint32_t ndim,
                                                nimcp_gpu_precision_t precision)
{
    return NULL;
}

bool nimcp_gpu_tensor_to_host(const nimcp_gpu_tensor_t* tensor, void* host_data) { return false; }
void nimcp_gpu_tensor_destroy(nimcp_gpu_tensor_t* tensor) {}
nimcp_gpu_tensor_t* nimcp_gpu_tensor_clone(const nimcp_gpu_tensor_t* tensor) { return NULL; }

bool nimcp_gpu_gemm(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* A,
                    const nimcp_gpu_tensor_t* B, nimcp_gpu_tensor_t* C,
                    float alpha, float beta, bool trans_a, bool trans_b) { return false; }

bool nimcp_gpu_gemv(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* A,
                    const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* y,
                    float alpha, float beta, bool trans_a) { return false; }

bool nimcp_gpu_gemm_batched(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* A,
                            const nimcp_gpu_tensor_t* B, nimcp_gpu_tensor_t* C,
                            float alpha, float beta, bool trans_a, bool trans_b) { return false; }

bool nimcp_gpu_add(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_sub(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_mul(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_div(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                   const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_add_scalar(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                          float scalar, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_mul_scalar(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                          float scalar, nimcp_gpu_tensor_t* out) { return false; }

bool nimcp_gpu_relu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_leaky_relu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                          nimcp_gpu_tensor_t* out, float alpha) { return false; }
bool nimcp_gpu_sigmoid(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_tanh(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_gelu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_silu(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_softmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_log_softmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }

bool nimcp_gpu_exp(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_log(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_sqrt(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_pow(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float exp, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_abs(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_clamp(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                     float min_val, float max_val, nimcp_gpu_tensor_t* out) { return false; }

bool nimcp_gpu_sum(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims) { return false; }
bool nimcp_gpu_mean(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                    nimcp_gpu_tensor_t* out, int axis, bool keepdims) { return false; }
bool nimcp_gpu_max(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims) { return false; }
bool nimcp_gpu_min(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims) { return false; }
bool nimcp_gpu_argmax(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                      nimcp_gpu_tensor_t* out, int axis) { return false; }
bool nimcp_gpu_argmin(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                      nimcp_gpu_tensor_t* out, int axis) { return false; }
bool nimcp_gpu_var(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims, bool unbiased) { return false; }
bool nimcp_gpu_std(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                   nimcp_gpu_tensor_t* out, int axis, bool keepdims, bool unbiased) { return false; }

bool nimcp_gpu_norm_l1(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result) { return false; }
bool nimcp_gpu_norm_l2(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result) { return false; }
bool nimcp_gpu_norm_linf(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result) { return false; }
bool nimcp_gpu_norm_frobenius(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, float* result) { return false; }

bool nimcp_gpu_eq(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_gt(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_lt(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_ge(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_le(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* a,
                  const nimcp_gpu_tensor_t* b, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_where(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* cond,
                     const nimcp_gpu_tensor_t* a, const nimcp_gpu_tensor_t* b,
                     nimcp_gpu_tensor_t* out) { return false; }

bool nimcp_gpu_fill(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* tensor, float value) { return false; }
bool nimcp_gpu_zeros(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* tensor) { return false; }
bool nimcp_gpu_ones(nimcp_gpu_context_t* ctx, nimcp_gpu_tensor_t* tensor) { return false; }
bool nimcp_gpu_copy(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* src,
                    nimcp_gpu_tensor_t* dst) { return false; }
bool nimcp_gpu_transpose(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                         nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_reshape(nimcp_gpu_tensor_t* tensor, const size_t* new_dims, uint32_t new_ndim) { return false; }

bool nimcp_gpu_fft_1d(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                      nimcp_gpu_tensor_t* out, bool inverse) { return false; }
bool nimcp_gpu_fft_2d(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x,
                      nimcp_gpu_tensor_t* out, bool inverse) { return false; }
bool nimcp_gpu_rfft(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }
bool nimcp_gpu_irfft(nimcp_gpu_context_t* ctx, const nimcp_gpu_tensor_t* x, nimcp_gpu_tensor_t* out) { return false; }

#endif // NIMCP_ENABLE_CUDA
