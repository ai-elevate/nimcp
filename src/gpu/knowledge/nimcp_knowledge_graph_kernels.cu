/**
 * @file nimcp_knowledge_graph_kernels.cu
 * @brief GPU Knowledge Graph CUDA Kernels Implementation
 *
 * WHAT: CUDA kernels for knowledge graph operations
 * WHY:  GPU acceleration for large-scale graph traversal and similarity
 * HOW:  Custom kernels for BFS/DFS, embeddings, and hyperbolic operations
 *
 * ARCHITECTURE:
 * - Parallel graph traversal (BFS/DFS across frontiers)
 * - Semantic similarity via cosine similarity on embeddings
 * - Hyperbolic space operations for hierarchical knowledge
 * - Node/edge feature aggregation with shared memory
 * - Subgraph matching with parallel pattern detection
 *
 * @version 1.0
 * @author NIMCP Development Team
 * @date 2025
 */

#ifdef NIMCP_ENABLE_CUDA

// Include CUDA headers FIRST (before any extern "C" blocks from our headers)
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <float.h>

// Now include our headers (which have extern "C" blocks)
#include "gpu/knowledge/nimcp_knowledge_graph_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "KNOWLEDGE_GPU"

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

#define CUDA_CHECK_NULL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return NULL; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        LOG_ERROR("CUDA error at %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

//=============================================================================
// Kernel Configuration
//=============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_SHARED_MEM 48000  // 48KB shared memory limit

// Calculate grid size for N elements
#define GRID_SIZE(n) (((n) + BLOCK_SIZE - 1) / BLOCK_SIZE)

// Invalid node marker for BFS/DFS
#define INVALID_NODE 0xFFFFFFFF

//=============================================================================
// Device Utility Functions
//=============================================================================

/**
 * @brief Warp-level reduction sum
 */
__device__ inline float warp_reduce_sum(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/**
 * @brief Warp-level reduction max
 */
__device__ inline float warp_reduce_max(float val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

/**
 * @brief Block-level reduction sum using shared memory
 */
__device__ inline float block_reduce_sum(float val, float* shared_mem)
{
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    val = warp_reduce_sum(val);

    // Write reduced value to shared memory
    if (lane == 0) {
        shared_mem[wid] = val;
    }
    __syncthreads();

    // Read from shared memory only if that warp existed
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared_mem[threadIdx.x] : 0.0f;

    // Final warp reduction
    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

/**
 * @brief Compute L2 norm of a vector segment
 */
__device__ inline float compute_l2_norm(const float* vec, uint32_t dim)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += vec[i] * vec[i];
    }
    return sqrtf(sum);
}

/**
 * @brief Compute dot product of two vectors
 */
__device__ inline float compute_dot_product(const float* a, const float* b, uint32_t dim)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

//=============================================================================
// Graph Creation and Destruction
//=============================================================================

nimcp_gpu_knowledge_graph_t* nimcp_gpu_knowledge_graph_create(
    nimcp_gpu_context_t* ctx,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const float* edge_weights,
    uint32_t num_nodes,
    uint32_t num_edges,
    uint32_t embed_dim)
{
    if (!ctx || !row_offsets || !col_indices) {
        LOG_ERROR("Invalid parameters for graph creation");
        return NULL;
    }

    if (num_nodes == 0) {
        LOG_ERROR("Graph must have at least one node");
        return NULL;
    }

    nimcp_gpu_knowledge_graph_t* graph = (nimcp_gpu_knowledge_graph_t*)calloc(1, sizeof(nimcp_gpu_knowledge_graph_t));
    if (!graph) {
        LOG_ERROR("Failed to allocate graph structure");
        return NULL;
    }

    graph->num_nodes = num_nodes;
    graph->num_edges = num_edges;
    graph->embed_dim = embed_dim;
    graph->is_hyperbolic = false;
    graph->ctx = ctx;

    // Create row_offsets tensor [num_nodes + 1]
    size_t row_dims[] = {num_nodes + 1};
    graph->row_offsets = nimcp_gpu_tensor_from_host(ctx, row_offsets, row_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    if (!graph->row_offsets) {
        LOG_ERROR("Failed to create row_offsets tensor");
        nimcp_gpu_knowledge_graph_destroy(graph);
        return NULL;
    }

    // Create col_indices tensor [num_edges]
    if (num_edges > 0) {
        size_t col_dims[] = {num_edges};
        graph->col_indices = nimcp_gpu_tensor_from_host(ctx, col_indices, col_dims, 1, NIMCP_GPU_PRECISION_UINT32);
        if (!graph->col_indices) {
            LOG_ERROR("Failed to create col_indices tensor");
            nimcp_gpu_knowledge_graph_destroy(graph);
            return NULL;
        }

        // Create edge_weights tensor if provided
        if (edge_weights) {
            graph->edge_weights = nimcp_gpu_tensor_from_host(ctx, edge_weights, col_dims, 1, NIMCP_GPU_PRECISION_FP32);
            if (!graph->edge_weights) {
                LOG_ERROR("Failed to create edge_weights tensor");
                nimcp_gpu_knowledge_graph_destroy(graph);
                return NULL;
            }
        }
    }

    // Create node_embeddings tensor [num_nodes x embed_dim]
    if (embed_dim > 0) {
        size_t embed_dims[] = {num_nodes, embed_dim};
        graph->node_embeddings = nimcp_gpu_tensor_create(ctx, embed_dims, 2, NIMCP_GPU_PRECISION_FP32);
        if (!graph->node_embeddings) {
            LOG_ERROR("Failed to create node_embeddings tensor");
            nimcp_gpu_knowledge_graph_destroy(graph);
            return NULL;
        }
    }

    LOG_DEBUG("Created GPU knowledge graph: %u nodes, %u edges, embed_dim=%u",
              num_nodes, num_edges, embed_dim);
    return graph;
}

void nimcp_gpu_knowledge_graph_destroy(nimcp_gpu_knowledge_graph_t* graph)
{
    if (!graph) return;

    nimcp_gpu_tensor_destroy(graph->row_offsets);
    nimcp_gpu_tensor_destroy(graph->col_indices);
    nimcp_gpu_tensor_destroy(graph->edge_weights);
    nimcp_gpu_tensor_destroy(graph->node_embeddings);
    nimcp_gpu_tensor_destroy(graph->edge_embeddings);

    free(graph);
}

bool nimcp_gpu_knowledge_graph_set_embeddings(
    nimcp_gpu_knowledge_graph_t* graph,
    const float* embeddings)
{
    if (!graph || !embeddings) {
        LOG_ERROR("Invalid parameters for set_embeddings");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embedding tensor allocated");
        return false;
    }

    size_t data_size = graph->num_nodes * graph->embed_dim * sizeof(float);
    CUDA_CHECK(cudaMemcpy(graph->node_embeddings->data, embeddings, data_size, cudaMemcpyHostToDevice));

    graph->is_hyperbolic = false;
    return true;
}

bool nimcp_gpu_knowledge_graph_set_hyperbolic_embeddings(
    nimcp_gpu_knowledge_graph_t* graph,
    const float* embeddings)
{
    if (!nimcp_gpu_knowledge_graph_set_embeddings(graph, embeddings)) {
        return false;
    }
    graph->is_hyperbolic = true;
    return true;
}

//=============================================================================
// BFS Kernels
//=============================================================================

/**
 * @brief Initialize BFS distances and visited arrays
 */
__global__ void kernel_bfs_init(
    uint32_t* distances,
    uint32_t* parents,
    uint32_t* visited,
    uint32_t source,
    uint32_t num_nodes)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (idx == source) {
        distances[idx] = 0;
        parents[idx] = source;
        visited[idx] = 1;
    } else {
        distances[idx] = INVALID_NODE;
        parents[idx] = INVALID_NODE;
        visited[idx] = 0;
    }
}

/**
 * @brief BFS frontier expansion kernel
 *
 * Each thread processes one node in the current frontier and marks neighbors.
 */
__global__ void kernel_bfs_expand(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const uint32_t* current_frontier,
    uint32_t frontier_size,
    uint32_t* distances,
    uint32_t* parents,
    uint32_t* visited,
    uint32_t* next_frontier,
    uint32_t* next_frontier_count,
    uint32_t current_depth,
    uint32_t num_nodes)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontier_size) return;

    uint32_t node = current_frontier[idx];
    if (node >= num_nodes) return;

    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];

    for (uint32_t e = row_start; e < row_end; e++) {
        uint32_t neighbor = col_indices[e];
        if (neighbor >= num_nodes) continue;

        // Try to mark neighbor as visited
        uint32_t old = atomicCAS(&visited[neighbor], 0, 1);
        if (old == 0) {
            // First visit - update distance and parent
            distances[neighbor] = current_depth + 1;
            parents[neighbor] = node;

            // Add to next frontier
            uint32_t pos = atomicAdd(next_frontier_count, 1);
            next_frontier[pos] = neighbor;
        }
    }
}

/**
 * @brief Multi-source BFS initialization
 */
__global__ void kernel_bfs_init_multi_source(
    uint32_t* distances,
    uint32_t* parents,
    uint32_t* visited,
    const uint32_t* source_nodes,
    uint32_t num_sources,
    uint32_t num_nodes)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    distances[idx] = INVALID_NODE;
    parents[idx] = INVALID_NODE;
    visited[idx] = 0;

    // Check if this node is a source
    for (uint32_t s = 0; s < num_sources; s++) {
        if (source_nodes[s] == idx) {
            distances[idx] = 0;
            parents[idx] = idx;
            visited[idx] = 1;
            break;
        }
    }
}

bool nimcp_gpu_bfs(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t source_node,
    int32_t max_depth,
    nimcp_graph_traversal_result_t* result)
{
    if (!graph || !result) {
        LOG_ERROR("Invalid parameters for BFS");
        return false;
    }

    if (source_node >= graph->num_nodes) {
        LOG_ERROR("Source node %u out of range (max %u)", source_node, graph->num_nodes - 1);
        return false;
    }

    uint32_t num_nodes = graph->num_nodes;
    size_t node_dims[] = {num_nodes};

    // Create result tensors
    result->distances = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->parents = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->visited = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);

    if (!result->distances || !result->parents || !result->visited) {
        LOG_ERROR("Failed to allocate BFS result tensors");
        nimcp_graph_traversal_result_destroy(result);
        return false;
    }

    // Allocate frontier buffers
    uint32_t* d_current_frontier = NULL;
    uint32_t* d_next_frontier = NULL;
    uint32_t* d_frontier_count = NULL;

    CUDA_CHECK(cudaMalloc(&d_current_frontier, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_count, sizeof(uint32_t)));

    // Initialize distances and visited
    kernel_bfs_init<<<GRID_SIZE(num_nodes), BLOCK_SIZE>>>(
        (uint32_t*)result->distances->data,
        (uint32_t*)result->parents->data,
        (uint32_t*)result->visited->data,
        source_node,
        num_nodes);
    CUDA_CHECK(cudaGetLastError());

    // Initialize frontier with source
    CUDA_CHECK(cudaMemcpy(d_current_frontier, &source_node, sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t frontier_size = 1;
    uint32_t current_depth = 0;
    uint32_t total_visited = 1;

    // BFS loop
    while (frontier_size > 0) {
        if (max_depth >= 0 && (int32_t)current_depth >= max_depth) {
            break;
        }

        // Reset next frontier count
        CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(uint32_t)));

        // Expand frontier
        kernel_bfs_expand<<<GRID_SIZE(frontier_size), BLOCK_SIZE>>>(
            (const uint32_t*)graph->row_offsets->data,
            (const uint32_t*)graph->col_indices->data,
            d_current_frontier,
            frontier_size,
            (uint32_t*)result->distances->data,
            (uint32_t*)result->parents->data,
            (uint32_t*)result->visited->data,
            d_next_frontier,
            d_frontier_count,
            current_depth,
            num_nodes);
        CUDA_CHECK(cudaGetLastError());

        // Get next frontier size
        uint32_t next_size = 0;
        CUDA_CHECK(cudaMemcpy(&next_size, d_frontier_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        // Swap frontiers
        uint32_t* temp = d_current_frontier;
        d_current_frontier = d_next_frontier;
        d_next_frontier = temp;

        frontier_size = next_size;
        total_visited += next_size;
        current_depth++;
    }

    result->num_visited = total_visited;
    result->max_depth = current_depth;

    // Cleanup
    cudaFree(d_current_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_count);

    return true;
}

bool nimcp_gpu_bfs_multi_source(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* source_nodes,
    uint32_t num_sources,
    int32_t max_depth,
    nimcp_graph_traversal_result_t* result)
{
    if (!graph || !source_nodes || !result || num_sources == 0) {
        LOG_ERROR("Invalid parameters for multi-source BFS");
        return false;
    }

    uint32_t num_nodes = graph->num_nodes;
    size_t node_dims[] = {num_nodes};

    // Create result tensors
    result->distances = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->parents = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->visited = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);

    if (!result->distances || !result->parents || !result->visited) {
        nimcp_graph_traversal_result_destroy(result);
        return false;
    }

    // Copy source nodes to device
    uint32_t* d_sources = NULL;
    CUDA_CHECK(cudaMalloc(&d_sources, num_sources * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_sources, source_nodes, num_sources * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Allocate frontier buffers
    uint32_t* d_current_frontier = NULL;
    uint32_t* d_next_frontier = NULL;
    uint32_t* d_frontier_count = NULL;

    CUDA_CHECK(cudaMalloc(&d_current_frontier, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_next_frontier, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_frontier_count, sizeof(uint32_t)));

    // Initialize with multi-source
    kernel_bfs_init_multi_source<<<GRID_SIZE(num_nodes), BLOCK_SIZE>>>(
        (uint32_t*)result->distances->data,
        (uint32_t*)result->parents->data,
        (uint32_t*)result->visited->data,
        d_sources,
        num_sources,
        num_nodes);
    CUDA_CHECK(cudaGetLastError());

    // Initialize frontier with all sources
    CUDA_CHECK(cudaMemcpy(d_current_frontier, source_nodes, num_sources * sizeof(uint32_t), cudaMemcpyHostToDevice));

    uint32_t frontier_size = num_sources;
    uint32_t current_depth = 0;
    uint32_t total_visited = num_sources;

    // BFS loop
    while (frontier_size > 0) {
        if (max_depth >= 0 && (int32_t)current_depth >= max_depth) {
            break;
        }

        CUDA_CHECK(cudaMemset(d_frontier_count, 0, sizeof(uint32_t)));

        kernel_bfs_expand<<<GRID_SIZE(frontier_size), BLOCK_SIZE>>>(
            (const uint32_t*)graph->row_offsets->data,
            (const uint32_t*)graph->col_indices->data,
            d_current_frontier,
            frontier_size,
            (uint32_t*)result->distances->data,
            (uint32_t*)result->parents->data,
            (uint32_t*)result->visited->data,
            d_next_frontier,
            d_frontier_count,
            current_depth,
            num_nodes);
        CUDA_CHECK(cudaGetLastError());

        uint32_t next_size = 0;
        CUDA_CHECK(cudaMemcpy(&next_size, d_frontier_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t* temp = d_current_frontier;
        d_current_frontier = d_next_frontier;
        d_next_frontier = temp;

        frontier_size = next_size;
        total_visited += next_size;
        current_depth++;
    }

    result->num_visited = total_visited;
    result->max_depth = current_depth;

    cudaFree(d_sources);
    cudaFree(d_current_frontier);
    cudaFree(d_next_frontier);
    cudaFree(d_frontier_count);

    return true;
}

//=============================================================================
// DFS Kernels
//=============================================================================

/**
 * @brief DFS kernel using iterative approach with work-stealing
 *
 * Uses a parallel stack-based approach where each thread block processes
 * a portion of the graph.
 */
__global__ void kernel_dfs_explore(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    uint32_t* stack,
    uint32_t* stack_ptr,
    uint32_t* distances,
    uint32_t* parents,
    uint32_t* visited,
    uint32_t* visit_order,
    uint32_t* visit_count,
    int32_t max_depth,
    uint32_t num_nodes)
{
    __shared__ uint32_t shared_stack[256];
    __shared__ uint32_t shared_stack_ptr;

    if (threadIdx.x == 0) {
        shared_stack_ptr = 0;
    }
    __syncthreads();

    while (true) {
        // Try to pop from global stack
        uint32_t current = INVALID_NODE;
        uint32_t current_depth = 0;

        if (threadIdx.x == 0) {
            uint32_t old_ptr = atomicSub(stack_ptr, 1);
            if (old_ptr > 0) {
                current = stack[old_ptr - 1];
                current_depth = distances[current];
            } else {
                atomicAdd(stack_ptr, 1);  // Restore if empty
            }
        }

        current = __shfl_sync(0xFFFFFFFF, current, 0);
        current_depth = __shfl_sync(0xFFFFFFFF, current_depth, 0);

        if (current == INVALID_NODE) {
            break;  // Stack is empty
        }

        // Check depth limit
        if (max_depth >= 0 && (int32_t)current_depth >= max_depth) {
            continue;
        }

        // Explore neighbors
        uint32_t row_start = row_offsets[current];
        uint32_t row_end = row_offsets[current + 1];

        for (uint32_t e = row_start + threadIdx.x; e < row_end; e += blockDim.x) {
            uint32_t neighbor = col_indices[e];
            if (neighbor >= num_nodes) continue;

            uint32_t old = atomicCAS(&visited[neighbor], 0, 1);
            if (old == 0) {
                distances[neighbor] = current_depth + 1;
                parents[neighbor] = current;

                // Record visit order
                uint32_t order = atomicAdd(visit_count, 1);
                visit_order[order] = neighbor;

                // Add to stack
                uint32_t pos = atomicAdd(stack_ptr, 1);
                if (pos < num_nodes) {
                    stack[pos] = neighbor;
                }
            }
        }

        __syncthreads();
    }
}

bool nimcp_gpu_dfs(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t source_node,
    int32_t max_depth,
    nimcp_graph_traversal_result_t* result)
{
    if (!graph || !result) {
        LOG_ERROR("Invalid parameters for DFS");
        return false;
    }

    if (source_node >= graph->num_nodes) {
        LOG_ERROR("Source node out of range");
        return false;
    }

    uint32_t num_nodes = graph->num_nodes;
    size_t node_dims[] = {num_nodes};

    // Create result tensors
    result->distances = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->parents = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->visited = nimcp_gpu_tensor_create(graph->ctx, node_dims, 1, NIMCP_GPU_PRECISION_UINT32);

    if (!result->distances || !result->parents || !result->visited) {
        nimcp_graph_traversal_result_destroy(result);
        return false;
    }

    // Allocate DFS stack and auxiliary arrays
    uint32_t* d_stack = NULL;
    uint32_t* d_stack_ptr = NULL;
    uint32_t* d_visit_order = NULL;
    uint32_t* d_visit_count = NULL;

    CUDA_CHECK(cudaMalloc(&d_stack, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_stack_ptr, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_visit_order, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_visit_count, sizeof(uint32_t)));

    // Initialize
    kernel_bfs_init<<<GRID_SIZE(num_nodes), BLOCK_SIZE>>>(
        (uint32_t*)result->distances->data,
        (uint32_t*)result->parents->data,
        (uint32_t*)result->visited->data,
        source_node,
        num_nodes);
    CUDA_CHECK(cudaGetLastError());

    // Initialize stack with source
    CUDA_CHECK(cudaMemcpy(d_stack, &source_node, sizeof(uint32_t), cudaMemcpyHostToDevice));
    uint32_t one = 1;
    CUDA_CHECK(cudaMemcpy(d_stack_ptr, &one, sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_visit_count, 0, sizeof(uint32_t)));

    // Run DFS exploration
    kernel_dfs_explore<<<1, BLOCK_SIZE>>>(
        (const uint32_t*)graph->row_offsets->data,
        (const uint32_t*)graph->col_indices->data,
        d_stack,
        d_stack_ptr,
        (uint32_t*)result->distances->data,
        (uint32_t*)result->parents->data,
        (uint32_t*)result->visited->data,
        d_visit_order,
        d_visit_count,
        max_depth,
        num_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get visit count
    uint32_t visit_count = 0;
    CUDA_CHECK(cudaMemcpy(&visit_count, d_visit_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    result->num_visited = visit_count + 1;  // +1 for source
    result->max_depth = max_depth >= 0 ? max_depth : num_nodes;

    cudaFree(d_stack);
    cudaFree(d_stack_ptr);
    cudaFree(d_visit_order);
    cudaFree(d_visit_count);

    return true;
}

//=============================================================================
// Shortest Path (Bidirectional BFS)
//=============================================================================

/**
 * @brief Check if frontiers from forward and backward BFS have met
 */
__global__ void kernel_check_frontier_intersection(
    const uint32_t* forward_visited,
    const uint32_t* backward_visited,
    uint32_t* meeting_point,
    uint32_t num_nodes)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    if (forward_visited[idx] && backward_visited[idx]) {
        atomicMin(meeting_point, idx);
    }
}

bool nimcp_gpu_shortest_path(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t source,
    uint32_t target,
    uint32_t* path_out,
    uint32_t* path_length)
{
    if (!graph || !path_out || !path_length) {
        LOG_ERROR("Invalid parameters for shortest_path");
        return false;
    }

    if (source >= graph->num_nodes || target >= graph->num_nodes) {
        LOG_ERROR("Source or target node out of range");
        return false;
    }

    // Special case: source == target
    if (source == target) {
        path_out[0] = source;
        *path_length = 1;
        return true;
    }

    // Run BFS from source
    nimcp_graph_traversal_result_t result;
    memset(&result, 0, sizeof(result));

    if (!nimcp_gpu_bfs(graph, source, -1, &result)) {
        return false;
    }

    // Check if target was reached
    uint32_t* h_distances = (uint32_t*)malloc(graph->num_nodes * sizeof(uint32_t));
    uint32_t* h_parents = (uint32_t*)malloc(graph->num_nodes * sizeof(uint32_t));

    if (!h_distances || !h_parents) {
        free(h_distances);
        free(h_parents);
        nimcp_graph_traversal_result_destroy(&result);
        return false;
    }

    cudaMemcpy(h_distances, result.distances->data, graph->num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_parents, result.parents->data, graph->num_nodes * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    if (h_distances[target] == INVALID_NODE) {
        // No path exists
        free(h_distances);
        free(h_parents);
        nimcp_graph_traversal_result_destroy(&result);
        *path_length = 0;
        return false;
    }

    // Reconstruct path
    uint32_t len = h_distances[target] + 1;
    *path_length = len;

    // Build path backwards
    uint32_t current = target;
    for (int i = len - 1; i >= 0; i--) {
        path_out[i] = current;
        current = h_parents[current];
    }

    free(h_distances);
    free(h_parents);
    nimcp_graph_traversal_result_destroy(&result);

    return true;
}

void nimcp_graph_traversal_result_destroy(nimcp_graph_traversal_result_t* result)
{
    if (!result) return;

    nimcp_gpu_tensor_destroy(result->distances);
    nimcp_gpu_tensor_destroy(result->parents);
    nimcp_gpu_tensor_destroy(result->visited);

    result->distances = NULL;
    result->parents = NULL;
    result->visited = NULL;
    result->num_visited = 0;
    result->max_depth = 0;
}

//=============================================================================
// Cosine Similarity Kernels
//=============================================================================

/**
 * @brief Compute cosine similarity between two embedding vectors
 */
__global__ void kernel_cosine_similarity_single(
    const float* embeddings,
    uint32_t node_a,
    uint32_t node_b,
    uint32_t embed_dim,
    float* similarity)
{
    __shared__ float dot_prod[WARP_SIZE];
    __shared__ float norm_a[WARP_SIZE];
    __shared__ float norm_b[WARP_SIZE];

    float local_dot = 0.0f;
    float local_norm_a = 0.0f;
    float local_norm_b = 0.0f;

    const float* vec_a = embeddings + node_a * embed_dim;
    const float* vec_b = embeddings + node_b * embed_dim;

    for (uint32_t i = threadIdx.x; i < embed_dim; i += blockDim.x) {
        float a = vec_a[i];
        float b = vec_b[i];
        local_dot += a * b;
        local_norm_a += a * a;
        local_norm_b += b * b;
    }

    // Warp reduction
    local_dot = warp_reduce_sum(local_dot);
    local_norm_a = warp_reduce_sum(local_norm_a);
    local_norm_b = warp_reduce_sum(local_norm_b);

    if (threadIdx.x % WARP_SIZE == 0) {
        dot_prod[threadIdx.x / WARP_SIZE] = local_dot;
        norm_a[threadIdx.x / WARP_SIZE] = local_norm_a;
        norm_b[threadIdx.x / WARP_SIZE] = local_norm_b;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total_dot = 0.0f;
        float total_norm_a = 0.0f;
        float total_norm_b = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

        for (int i = 0; i < num_warps; i++) {
            total_dot += dot_prod[i];
            total_norm_a += norm_a[i];
            total_norm_b += norm_b[i];
        }

        float denom = sqrtf(total_norm_a) * sqrtf(total_norm_b);
        *similarity = (denom > 1e-8f) ? total_dot / denom : 0.0f;
    }
}

/**
 * @brief Compute pairwise cosine similarity matrix
 */
__global__ void kernel_pairwise_similarity(
    const float* embeddings,
    const uint32_t* node_indices,
    uint32_t num_nodes,
    uint32_t embed_dim,
    float* similarity_matrix)
{
    uint32_t i = blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_nodes || j >= num_nodes) return;

    uint32_t node_a = node_indices[i];
    uint32_t node_b = node_indices[j];

    const float* vec_a = embeddings + node_a * embed_dim;
    const float* vec_b = embeddings + node_b * embed_dim;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (uint32_t d = 0; d < embed_dim; d++) {
        float a = vec_a[d];
        float b = vec_b[d];
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    float denom = sqrtf(norm_a) * sqrtf(norm_b);
    similarity_matrix[i * num_nodes + j] = (denom > 1e-8f) ? dot / denom : 0.0f;
}

bool nimcp_gpu_node_similarity(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t node_a,
    uint32_t node_b,
    float* similarity)
{
    if (!graph || !similarity) {
        LOG_ERROR("Invalid parameters for node_similarity");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    if (node_a >= graph->num_nodes || node_b >= graph->num_nodes) {
        LOG_ERROR("Node index out of range");
        return false;
    }

    float* d_similarity = NULL;
    CUDA_CHECK(cudaMalloc(&d_similarity, sizeof(float)));

    kernel_cosine_similarity_single<<<1, BLOCK_SIZE>>>(
        (const float*)graph->node_embeddings->data,
        node_a,
        node_b,
        graph->embed_dim,
        d_similarity);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(similarity, d_similarity, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_similarity);

    return true;
}

bool nimcp_gpu_pairwise_similarity(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* node_indices,
    uint32_t num_nodes,
    nimcp_gpu_tensor_t* similarity_matrix)
{
    if (!graph || !node_indices || !similarity_matrix) {
        LOG_ERROR("Invalid parameters for pairwise_similarity");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    // Copy node indices to device
    uint32_t* d_indices = NULL;
    CUDA_CHECK(cudaMalloc(&d_indices, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_indices, node_indices, num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid((num_nodes + 31) / 32, num_nodes);

    kernel_pairwise_similarity<<<grid, block>>>(
        (const float*)graph->node_embeddings->data,
        d_indices,
        num_nodes,
        graph->embed_dim,
        (float*)similarity_matrix->data);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_indices);
    return true;
}

//=============================================================================
// KNN Similarity Search
//=============================================================================

/**
 * @brief Compute similarities for all nodes to query
 */
__global__ void kernel_compute_all_similarities(
    const float* embeddings,
    const float* query_embedding,
    uint32_t num_nodes,
    uint32_t embed_dim,
    float* similarities)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    const float* node_emb = embeddings + idx * embed_dim;

    float dot = 0.0f;
    float norm_node = 0.0f;
    float norm_query = 0.0f;

    for (uint32_t d = 0; d < embed_dim; d++) {
        float n = node_emb[d];
        float q = query_embedding[d];
        dot += n * q;
        norm_node += n * n;
        norm_query += q * q;
    }

    float denom = sqrtf(norm_node) * sqrtf(norm_query);
    similarities[idx] = (denom > 1e-8f) ? dot / denom : 0.0f;
}

/**
 * @brief Simple top-k selection (for small k)
 */
__global__ void kernel_top_k_select(
    const float* similarities,
    uint32_t num_nodes,
    uint32_t k,
    uint32_t* top_indices,
    float* top_scores)
{
    // Initialize with worst values
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) {
        top_indices[i] = 0;
        top_scores[i] = -FLT_MAX;
    }
    __syncthreads();

    if (threadIdx.x != 0) return;

    // Simple sequential top-k for now
    for (uint32_t n = 0; n < num_nodes; n++) {
        float score = similarities[n];

        // Find insertion point
        uint32_t insert_pos = k;
        for (uint32_t i = 0; i < k; i++) {
            if (score > top_scores[i]) {
                insert_pos = i;
                break;
            }
        }

        if (insert_pos < k) {
            // Shift elements down
            for (uint32_t i = k - 1; i > insert_pos; i--) {
                top_indices[i] = top_indices[i - 1];
                top_scores[i] = top_scores[i - 1];
            }
            top_indices[insert_pos] = n;
            top_scores[insert_pos] = score;
        }
    }
}

bool nimcp_gpu_knn_similarity(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t query_node,
    uint32_t k,
    nimcp_similarity_result_t* result)
{
    if (!graph || !result) {
        LOG_ERROR("Invalid parameters for knn_similarity");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    if (query_node >= graph->num_nodes) {
        LOG_ERROR("Query node out of range");
        return false;
    }

    k = (k > graph->num_nodes) ? graph->num_nodes : k;

    // Get query embedding
    const float* query_emb = (const float*)graph->node_embeddings->data + query_node * graph->embed_dim;

    // Allocate similarity buffer
    float* d_similarities = NULL;
    CUDA_CHECK(cudaMalloc(&d_similarities, graph->num_nodes * sizeof(float)));

    // Compute similarities
    kernel_compute_all_similarities<<<GRID_SIZE(graph->num_nodes), BLOCK_SIZE>>>(
        (const float*)graph->node_embeddings->data,
        query_emb,
        graph->num_nodes,
        graph->embed_dim,
        d_similarities);
    CUDA_CHECK(cudaGetLastError());

    // Create result tensors
    size_t k_dims[] = {k};
    result->indices = nimcp_gpu_tensor_create(graph->ctx, k_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->scores = nimcp_gpu_tensor_create(graph->ctx, k_dims, 1, NIMCP_GPU_PRECISION_FP32);
    result->k = k;

    if (!result->indices || !result->scores) {
        cudaFree(d_similarities);
        nimcp_similarity_result_destroy(result);
        return false;
    }

    // Top-k selection
    kernel_top_k_select<<<1, 1>>>(
        d_similarities,
        graph->num_nodes,
        k,
        (uint32_t*)result->indices->data,
        (float*)result->scores->data);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_similarities);
    return true;
}

bool nimcp_gpu_knn_similarity_embedding(
    nimcp_gpu_knowledge_graph_t* graph,
    const float* query_embedding,
    uint32_t k,
    nimcp_similarity_result_t* result)
{
    if (!graph || !query_embedding || !result) {
        LOG_ERROR("Invalid parameters for knn_similarity_embedding");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    k = (k > graph->num_nodes) ? graph->num_nodes : k;

    // Copy query embedding to device
    float* d_query = NULL;
    CUDA_CHECK(cudaMalloc(&d_query, graph->embed_dim * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_query, query_embedding, graph->embed_dim * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate similarity buffer
    float* d_similarities = NULL;
    CUDA_CHECK(cudaMalloc(&d_similarities, graph->num_nodes * sizeof(float)));

    // Compute similarities
    kernel_compute_all_similarities<<<GRID_SIZE(graph->num_nodes), BLOCK_SIZE>>>(
        (const float*)graph->node_embeddings->data,
        d_query,
        graph->num_nodes,
        graph->embed_dim,
        d_similarities);
    CUDA_CHECK(cudaGetLastError());

    // Create result tensors
    size_t k_dims[] = {k};
    result->indices = nimcp_gpu_tensor_create(graph->ctx, k_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->scores = nimcp_gpu_tensor_create(graph->ctx, k_dims, 1, NIMCP_GPU_PRECISION_FP32);
    result->k = k;

    if (!result->indices || !result->scores) {
        cudaFree(d_query);
        cudaFree(d_similarities);
        nimcp_similarity_result_destroy(result);
        return false;
    }

    // Top-k selection
    kernel_top_k_select<<<1, 1>>>(
        d_similarities,
        graph->num_nodes,
        k,
        (uint32_t*)result->indices->data,
        (float*)result->scores->data);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_query);
    cudaFree(d_similarities);
    return true;
}

void nimcp_similarity_result_destroy(nimcp_similarity_result_t* result)
{
    if (!result) return;

    nimcp_gpu_tensor_destroy(result->indices);
    nimcp_gpu_tensor_destroy(result->scores);

    result->indices = NULL;
    result->scores = NULL;
    result->k = 0;
}

//=============================================================================
// Embedding Update Operations
//=============================================================================

/**
 * @brief Apply gradient descent update to embeddings
 */
__global__ void kernel_embedding_update(
    float* embeddings,
    const float* gradients,
    float learning_rate,
    size_t numel)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    embeddings[idx] -= learning_rate * gradients[idx];
}

bool nimcp_gpu_embedding_update(
    nimcp_gpu_knowledge_graph_t* graph,
    const nimcp_gpu_tensor_t* gradients,
    float learning_rate)
{
    if (!graph || !gradients) {
        LOG_ERROR("Invalid parameters for embedding_update");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    size_t numel = graph->num_nodes * graph->embed_dim;

    kernel_embedding_update<<<GRID_SIZE(numel), BLOCK_SIZE>>>(
        (float*)graph->node_embeddings->data,
        (const float*)gradients->data,
        learning_rate,
        numel);
    CUDA_CHECK(cudaGetLastError());

    return true;
}

/**
 * @brief Compute triplet loss for knowledge graph embeddings
 */
__global__ void kernel_triplet_loss(
    const float* embeddings,
    const uint32_t* anchors,
    const uint32_t* positives,
    const uint32_t* negatives,
    uint32_t batch_size,
    uint32_t embed_dim,
    float margin,
    float* losses)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    uint32_t anchor = anchors[idx];
    uint32_t positive = positives[idx];
    uint32_t negative = negatives[idx];

    const float* a = embeddings + anchor * embed_dim;
    const float* p = embeddings + positive * embed_dim;
    const float* n = embeddings + negative * embed_dim;

    // Compute distances
    float dist_pos = 0.0f;
    float dist_neg = 0.0f;

    for (uint32_t d = 0; d < embed_dim; d++) {
        float diff_p = a[d] - p[d];
        float diff_n = a[d] - n[d];
        dist_pos += diff_p * diff_p;
        dist_neg += diff_n * diff_n;
    }

    dist_pos = sqrtf(dist_pos);
    dist_neg = sqrtf(dist_neg);

    // Triplet loss: max(0, margin + d_pos - d_neg)
    losses[idx] = fmaxf(0.0f, margin + dist_pos - dist_neg);
}

bool nimcp_gpu_triplet_loss(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* anchors,
    const uint32_t* positives,
    const uint32_t* negatives,
    uint32_t batch_size,
    float margin,
    float* loss_out)
{
    if (!graph || !anchors || !positives || !negatives || !loss_out) {
        LOG_ERROR("Invalid parameters for triplet_loss");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    // Copy indices to device
    uint32_t* d_anchors = NULL;
    uint32_t* d_positives = NULL;
    uint32_t* d_negatives = NULL;
    float* d_losses = NULL;

    CUDA_CHECK(cudaMalloc(&d_anchors, batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_positives, batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_negatives, batch_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_losses, batch_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_anchors, anchors, batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positives, positives, batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_negatives, negatives, batch_size * sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_triplet_loss<<<GRID_SIZE(batch_size), BLOCK_SIZE>>>(
        (const float*)graph->node_embeddings->data,
        d_anchors,
        d_positives,
        d_negatives,
        batch_size,
        graph->embed_dim,
        margin,
        d_losses);
    CUDA_CHECK(cudaGetLastError());

    // Sum losses on host
    float* h_losses = (float*)malloc(batch_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_losses, d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (uint32_t i = 0; i < batch_size; i++) {
        total_loss += h_losses[i];
    }
    *loss_out = total_loss / batch_size;

    free(h_losses);
    cudaFree(d_anchors);
    cudaFree(d_positives);
    cudaFree(d_negatives);
    cudaFree(d_losses);

    return true;
}

/**
 * @brief L2 normalize embeddings
 */
__global__ void kernel_normalize_embeddings(
    float* embeddings,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* emb = embeddings + node * embed_dim;

    // Compute norm
    __shared__ float norm_shared[8];
    float local_norm = 0.0f;

    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        local_norm += emb[d] * emb[d];
    }

    local_norm = warp_reduce_sum(local_norm);
    if (threadIdx.x % WARP_SIZE == 0) {
        norm_shared[threadIdx.x / WARP_SIZE] = local_norm;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        for (int i = 0; i < num_warps; i++) {
            total += norm_shared[i];
        }
        norm_shared[0] = sqrtf(total);
    }
    __syncthreads();

    float norm = norm_shared[0];
    if (norm > 1e-8f) {
        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            emb[d] /= norm;
        }
    }
}

bool nimcp_gpu_normalize_embeddings(nimcp_gpu_knowledge_graph_t* graph)
{
    if (!graph) {
        LOG_ERROR("Invalid graph parameter");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    kernel_normalize_embeddings<<<graph->num_nodes, BLOCK_SIZE>>>(
        (float*)graph->node_embeddings->data,
        graph->num_nodes,
        graph->embed_dim);
    CUDA_CHECK(cudaGetLastError());

    return true;
}

//=============================================================================
// Feature Aggregation
//=============================================================================

/**
 * @brief Aggregate neighbor features (sum mode)
 */
__global__ void kernel_aggregate_sum(
    const float* embeddings,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    float* output,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* out = output + node * embed_dim;

    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];

    // Initialize output to zero
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        out[d] = 0.0f;
    }
    __syncthreads();

    // Sum neighbor embeddings
    for (uint32_t e = row_start; e < row_end; e++) {
        uint32_t neighbor = col_indices[e];
        const float* neigh_emb = embeddings + neighbor * embed_dim;

        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            atomicAdd(&out[d], neigh_emb[d]);
        }
    }
}

/**
 * @brief Aggregate neighbor features (mean mode)
 */
__global__ void kernel_aggregate_mean(
    const float* embeddings,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    float* output,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* out = output + node * embed_dim;

    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];
    uint32_t degree = row_end - row_start;

    if (degree == 0) {
        // No neighbors - copy self embedding
        const float* self_emb = embeddings + node * embed_dim;
        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            out[d] = self_emb[d];
        }
        return;
    }

    // Initialize output to zero
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        out[d] = 0.0f;
    }
    __syncthreads();

    // Sum neighbor embeddings
    for (uint32_t e = row_start; e < row_end; e++) {
        uint32_t neighbor = col_indices[e];
        const float* neigh_emb = embeddings + neighbor * embed_dim;

        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            atomicAdd(&out[d], neigh_emb[d]);
        }
    }
    __syncthreads();

    // Divide by degree
    float inv_degree = 1.0f / degree;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        out[d] *= inv_degree;
    }
}

/**
 * @brief Aggregate neighbor features (max mode)
 */
__global__ void kernel_aggregate_max(
    const float* embeddings,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    float* output,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* out = output + node * embed_dim;

    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];

    if (row_end == row_start) {
        const float* self_emb = embeddings + node * embed_dim;
        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            out[d] = self_emb[d];
        }
        return;
    }

    // Initialize with first neighbor
    const float* first_neigh = embeddings + col_indices[row_start] * embed_dim;
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        out[d] = first_neigh[d];
    }
    __syncthreads();

    // Max over remaining neighbors
    for (uint32_t e = row_start + 1; e < row_end; e++) {
        uint32_t neighbor = col_indices[e];
        const float* neigh_emb = embeddings + neighbor * embed_dim;

        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            out[d] = fmaxf(out[d], neigh_emb[d]);
        }
        __syncthreads();
    }
}

bool nimcp_gpu_aggregate_neighbors(
    nimcp_gpu_knowledge_graph_t* graph,
    nimcp_aggregate_mode_t mode,
    nimcp_gpu_tensor_t* output)
{
    if (!graph || !output) {
        LOG_ERROR("Invalid parameters for aggregate_neighbors");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    switch (mode) {
        case NIMCP_AGGREGATE_SUM:
            kernel_aggregate_sum<<<graph->num_nodes, BLOCK_SIZE>>>(
                (const float*)graph->node_embeddings->data,
                (const uint32_t*)graph->row_offsets->data,
                (const uint32_t*)graph->col_indices->data,
                (float*)output->data,
                graph->num_nodes,
                graph->embed_dim);
            break;

        case NIMCP_AGGREGATE_MEAN:
            kernel_aggregate_mean<<<graph->num_nodes, BLOCK_SIZE>>>(
                (const float*)graph->node_embeddings->data,
                (const uint32_t*)graph->row_offsets->data,
                (const uint32_t*)graph->col_indices->data,
                (float*)output->data,
                graph->num_nodes,
                graph->embed_dim);
            break;

        case NIMCP_AGGREGATE_MAX:
            kernel_aggregate_max<<<graph->num_nodes, BLOCK_SIZE>>>(
                (const float*)graph->node_embeddings->data,
                (const uint32_t*)graph->row_offsets->data,
                (const uint32_t*)graph->col_indices->data,
                (float*)output->data,
                graph->num_nodes,
                graph->embed_dim);
            break;

        case NIMCP_AGGREGATE_ATTENTION:
            // Fall through to attention function
            LOG_WARN("Use nimcp_gpu_aggregate_attention for attention aggregation");
            return false;

        default:
            LOG_ERROR("Unknown aggregation mode: %d", mode);
            return false;
    }

    CUDA_CHECK(cudaGetLastError());
    return true;
}

/**
 * @brief Attention-weighted aggregation kernel
 */
__global__ void kernel_aggregate_attention(
    const float* embeddings,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const float* query_weights,
    const float* key_weights,
    float* output,
    uint32_t num_nodes,
    uint32_t embed_dim,
    uint32_t attn_dim)
{
    extern __shared__ float shared_mem[];

    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* out = output + node * embed_dim;
    const float* node_emb = embeddings + node * embed_dim;

    uint32_t row_start = row_offsets[node];
    uint32_t row_end = row_offsets[node + 1];
    uint32_t degree = row_end - row_start;

    if (degree == 0) {
        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            out[d] = node_emb[d];
        }
        return;
    }

    // Compute query: q = W_q * node_emb
    float* query = shared_mem;
    float* attn_scores = shared_mem + attn_dim;

    // Simple query projection (thread 0 only for simplicity)
    if (threadIdx.x == 0) {
        for (uint32_t a = 0; a < attn_dim; a++) {
            float sum = 0.0f;
            for (uint32_t d = 0; d < embed_dim; d++) {
                sum += query_weights[a * embed_dim + d] * node_emb[d];
            }
            query[a] = sum;
        }
    }
    __syncthreads();

    // Compute attention scores
    if (threadIdx.x < degree) {
        uint32_t neighbor = col_indices[row_start + threadIdx.x];
        const float* neigh_emb = embeddings + neighbor * embed_dim;

        // Compute key: k = W_k * neigh_emb
        float score = 0.0f;
        for (uint32_t a = 0; a < attn_dim; a++) {
            float key_val = 0.0f;
            for (uint32_t d = 0; d < embed_dim; d++) {
                key_val += key_weights[a * embed_dim + d] * neigh_emb[d];
            }
            score += query[a] * key_val;
        }
        attn_scores[threadIdx.x] = score / sqrtf((float)attn_dim);
    }
    __syncthreads();

    // Softmax over attention scores
    if (threadIdx.x == 0) {
        float max_score = -FLT_MAX;
        for (uint32_t i = 0; i < degree; i++) {
            max_score = fmaxf(max_score, attn_scores[i]);
        }
        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < degree; i++) {
            attn_scores[i] = expf(attn_scores[i] - max_score);
            sum_exp += attn_scores[i];
        }
        for (uint32_t i = 0; i < degree; i++) {
            attn_scores[i] /= sum_exp;
        }
    }
    __syncthreads();

    // Weighted aggregation
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < degree; i++) {
            uint32_t neighbor = col_indices[row_start + i];
            sum += attn_scores[i] * embeddings[neighbor * embed_dim + d];
        }
        out[d] = sum;
    }
}

bool nimcp_gpu_aggregate_attention(
    nimcp_gpu_knowledge_graph_t* graph,
    const nimcp_gpu_tensor_t* query_weights,
    const nimcp_gpu_tensor_t* key_weights,
    nimcp_gpu_tensor_t* output)
{
    if (!graph || !query_weights || !key_weights || !output) {
        LOG_ERROR("Invalid parameters for aggregate_attention");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    uint32_t attn_dim = query_weights->dims[0];
    size_t shared_size = (attn_dim + graph->num_nodes) * sizeof(float);

    if (shared_size > MAX_SHARED_MEM) {
        LOG_ERROR("Attention dimension too large for shared memory");
        return false;
    }

    kernel_aggregate_attention<<<graph->num_nodes, BLOCK_SIZE, shared_size>>>(
        (const float*)graph->node_embeddings->data,
        (const uint32_t*)graph->row_offsets->data,
        (const uint32_t*)graph->col_indices->data,
        (const float*)query_weights->data,
        (const float*)key_weights->data,
        (float*)output->data,
        graph->num_nodes,
        graph->embed_dim,
        attn_dim);
    CUDA_CHECK(cudaGetLastError());

    return true;
}

bool nimcp_gpu_multi_hop_aggregate(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t num_hops,
    nimcp_aggregate_mode_t mode,
    nimcp_gpu_tensor_t* output)
{
    if (!graph || !output || num_hops == 0) {
        LOG_ERROR("Invalid parameters for multi_hop_aggregate");
        return false;
    }

    size_t embed_dims[] = {graph->num_nodes, graph->embed_dim};

    // Create temporary buffer
    nimcp_gpu_tensor_t* temp = nimcp_gpu_tensor_create(graph->ctx, embed_dims, 2, NIMCP_GPU_PRECISION_FP32);
    if (!temp) {
        return false;
    }

    // Copy initial embeddings to output
    size_t data_size = graph->num_nodes * graph->embed_dim * sizeof(float);
    cudaMemcpy(output->data, graph->node_embeddings->data, data_size, cudaMemcpyDeviceToDevice);

    // Iteratively aggregate
    for (uint32_t hop = 0; hop < num_hops; hop++) {
        // Swap output and temp
        nimcp_gpu_tensor_t* current_input = (hop % 2 == 0) ? output : temp;
        nimcp_gpu_tensor_t* current_output = (hop % 2 == 0) ? temp : output;

        // Temporarily swap graph embeddings
        nimcp_gpu_tensor_t* saved_embeddings = graph->node_embeddings;
        graph->node_embeddings = current_input;

        if (!nimcp_gpu_aggregate_neighbors(graph, mode, current_output)) {
            graph->node_embeddings = saved_embeddings;
            nimcp_gpu_tensor_destroy(temp);
            return false;
        }

        graph->node_embeddings = saved_embeddings;
    }

    // Ensure result is in output
    if (num_hops % 2 == 1) {
        cudaMemcpy(output->data, temp->data, data_size, cudaMemcpyDeviceToDevice);
    }

    nimcp_gpu_tensor_destroy(temp);
    return true;
}

//=============================================================================
// Subgraph Matching
//=============================================================================

bool nimcp_gpu_subgraph_match(
    nimcp_gpu_knowledge_graph_t* graph,
    nimcp_gpu_knowledge_graph_t* pattern,
    uint32_t max_matches,
    nimcp_subgraph_match_result_t* result)
{
    if (!graph || !pattern || !result) {
        LOG_ERROR("Invalid parameters for subgraph_match");
        return false;
    }

    // Simple brute-force matching for small patterns
    // TODO: Implement VF2 or GraphQL algorithm for better performance

    size_t mapping_dims[] = {max_matches, pattern->num_nodes};
    size_t score_dims[] = {max_matches};

    result->mappings = nimcp_gpu_tensor_create(graph->ctx, mapping_dims, 2, NIMCP_GPU_PRECISION_UINT32);
    result->scores = nimcp_gpu_tensor_create(graph->ctx, score_dims, 1, NIMCP_GPU_PRECISION_FP32);
    result->num_matches = 0;

    if (!result->mappings || !result->scores) {
        nimcp_subgraph_match_result_destroy(result);
        return false;
    }

    // For now, return empty result - full implementation would require
    // complex parallel subgraph isomorphism algorithm
    LOG_WARN("Subgraph matching not fully implemented - returning empty result");

    return true;
}

bool nimcp_gpu_subgraph_match_approximate(
    nimcp_gpu_knowledge_graph_t* graph,
    nimcp_gpu_knowledge_graph_t* pattern,
    float similarity_threshold,
    uint32_t max_matches,
    nimcp_subgraph_match_result_t* result)
{
    if (!graph || !pattern || !result) {
        LOG_ERROR("Invalid parameters for approximate subgraph_match");
        return false;
    }

    if (!graph->node_embeddings || !pattern->node_embeddings) {
        LOG_ERROR("Both graphs must have embeddings for approximate matching");
        return false;
    }

    size_t mapping_dims[] = {max_matches, pattern->num_nodes};
    size_t score_dims[] = {max_matches};

    result->mappings = nimcp_gpu_tensor_create(graph->ctx, mapping_dims, 2, NIMCP_GPU_PRECISION_UINT32);
    result->scores = nimcp_gpu_tensor_create(graph->ctx, score_dims, 1, NIMCP_GPU_PRECISION_FP32);
    result->num_matches = 0;

    if (!result->mappings || !result->scores) {
        nimcp_subgraph_match_result_destroy(result);
        return false;
    }

    // TODO: Implement embedding-based approximate matching
    LOG_WARN("Approximate subgraph matching not fully implemented");

    return true;
}

void nimcp_subgraph_match_result_destroy(nimcp_subgraph_match_result_t* result)
{
    if (!result) return;

    nimcp_gpu_tensor_destroy(result->mappings);
    nimcp_gpu_tensor_destroy(result->scores);

    result->mappings = NULL;
    result->scores = NULL;
    result->num_matches = 0;
}

//=============================================================================
// Hyperbolic Space Operations
//=============================================================================

/**
 * @brief Compute hyperbolic distance in Poincare ball model
 *
 * d(u,v) = arcosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))
 */
__device__ inline float device_hyperbolic_distance(
    const float* u,
    const float* v,
    uint32_t dim)
{
    float diff_sq = 0.0f;
    float norm_u_sq = 0.0f;
    float norm_v_sq = 0.0f;

    for (uint32_t d = 0; d < dim; d++) {
        float diff = u[d] - v[d];
        diff_sq += diff * diff;
        norm_u_sq += u[d] * u[d];
        norm_v_sq += v[d] * v[d];
    }

    // Clamp norms to stay inside Poincare ball
    norm_u_sq = fminf(norm_u_sq, 1.0f - 1e-5f);
    norm_v_sq = fminf(norm_v_sq, 1.0f - 1e-5f);

    float numerator = 2.0f * diff_sq;
    float denominator = (1.0f - norm_u_sq) * (1.0f - norm_v_sq);

    float arg = 1.0f + numerator / denominator;
    return acoshf(fmaxf(arg, 1.0f));
}

__global__ void kernel_hyperbolic_distance_single(
    const float* embeddings,
    uint32_t node_a,
    uint32_t node_b,
    uint32_t embed_dim,
    float* distance)
{
    if (threadIdx.x != 0) return;

    const float* u = embeddings + node_a * embed_dim;
    const float* v = embeddings + node_b * embed_dim;

    *distance = device_hyperbolic_distance(u, v, embed_dim);
}

bool nimcp_gpu_hyperbolic_distance(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t node_a,
    uint32_t node_b,
    float* distance)
{
    if (!graph || !distance) {
        LOG_ERROR("Invalid parameters for hyperbolic_distance");
        return false;
    }

    if (!graph->is_hyperbolic) {
        LOG_ERROR("Graph embeddings are not in hyperbolic space");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    if (node_a >= graph->num_nodes || node_b >= graph->num_nodes) {
        LOG_ERROR("Node index out of range");
        return false;
    }

    float* d_distance = NULL;
    CUDA_CHECK(cudaMalloc(&d_distance, sizeof(float)));

    kernel_hyperbolic_distance_single<<<1, 1>>>(
        (const float*)graph->node_embeddings->data,
        node_a,
        node_b,
        graph->embed_dim,
        d_distance);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(distance, d_distance, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_distance);

    return true;
}

/**
 * @brief Compute pairwise hyperbolic distances
 */
__global__ void kernel_hyperbolic_pairwise(
    const float* embeddings,
    const uint32_t* node_indices,
    uint32_t num_nodes,
    uint32_t embed_dim,
    float* distance_matrix)
{
    uint32_t i = blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= num_nodes || j >= num_nodes) return;

    uint32_t node_a = node_indices[i];
    uint32_t node_b = node_indices[j];

    const float* u = embeddings + node_a * embed_dim;
    const float* v = embeddings + node_b * embed_dim;

    distance_matrix[i * num_nodes + j] = device_hyperbolic_distance(u, v, embed_dim);
}

bool nimcp_gpu_hyperbolic_pairwise_distance(
    nimcp_gpu_knowledge_graph_t* graph,
    const uint32_t* node_indices,
    uint32_t num_nodes,
    nimcp_gpu_tensor_t* distance_matrix)
{
    if (!graph || !node_indices || !distance_matrix) {
        LOG_ERROR("Invalid parameters for hyperbolic_pairwise_distance");
        return false;
    }

    if (!graph->is_hyperbolic) {
        LOG_ERROR("Graph embeddings are not in hyperbolic space");
        return false;
    }

    uint32_t* d_indices = NULL;
    CUDA_CHECK(cudaMalloc(&d_indices, num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_indices, node_indices, num_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid((num_nodes + 31) / 32, num_nodes);

    kernel_hyperbolic_pairwise<<<grid, block>>>(
        (const float*)graph->node_embeddings->data,
        d_indices,
        num_nodes,
        graph->embed_dim,
        (float*)distance_matrix->data);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_indices);
    return true;
}

/**
 * @brief Riemannian gradient for Poincare ball
 *
 * Scales Euclidean gradient by (1 - ||x||^2)^2 / 4
 */
__global__ void kernel_riemannian_gradient(
    const float* embeddings,
    const float* euclidean_grad,
    float* riemannian_grad,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes * embed_dim) return;

    uint32_t node = idx / embed_dim;
    uint32_t d = idx % embed_dim;

    const float* emb = embeddings + node * embed_dim;

    // Compute norm squared
    float norm_sq = 0.0f;
    for (uint32_t i = 0; i < embed_dim; i++) {
        norm_sq += emb[i] * emb[i];
    }
    norm_sq = fminf(norm_sq, 1.0f - 1e-5f);

    // Scale factor
    float scale = (1.0f - norm_sq) * (1.0f - norm_sq) / 4.0f;

    riemannian_grad[idx] = scale * euclidean_grad[idx];
}

/**
 * @brief Exponential map for Poincare ball (retraction)
 *
 * exp_x(v) = x + v (approximately, for small v)
 * Then project back into the ball
 */
__global__ void kernel_hyperbolic_exp_map(
    float* embeddings,
    const float* tangent_vectors,
    float learning_rate,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* emb = embeddings + node * embed_dim;
    const float* tangent = tangent_vectors + node * embed_dim;

    // Update embedding
    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        emb[d] -= learning_rate * tangent[d];
    }
    __syncthreads();

    // Project back into Poincare ball
    if (threadIdx.x == 0) {
        float norm_sq = 0.0f;
        for (uint32_t d = 0; d < embed_dim; d++) {
            norm_sq += emb[d] * emb[d];
        }

        if (norm_sq >= 1.0f) {
            float scale = 0.99f / sqrtf(norm_sq);
            for (uint32_t d = 0; d < embed_dim; d++) {
                emb[d] *= scale;
            }
        }
    }
}

bool nimcp_gpu_hyperbolic_sgd_step(
    nimcp_gpu_knowledge_graph_t* graph,
    const nimcp_gpu_tensor_t* euclidean_gradients,
    float learning_rate)
{
    if (!graph || !euclidean_gradients) {
        LOG_ERROR("Invalid parameters for hyperbolic_sgd_step");
        return false;
    }

    if (!graph->is_hyperbolic) {
        LOG_ERROR("Graph embeddings are not in hyperbolic space");
        return false;
    }

    size_t numel = graph->num_nodes * graph->embed_dim;
    size_t dims[] = {graph->num_nodes, graph->embed_dim};

    // Allocate Riemannian gradient buffer
    nimcp_gpu_tensor_t* riem_grad = nimcp_gpu_tensor_create(graph->ctx, dims, 2, NIMCP_GPU_PRECISION_FP32);
    if (!riem_grad) {
        return false;
    }

    // Convert Euclidean to Riemannian gradient
    kernel_riemannian_gradient<<<GRID_SIZE(numel), BLOCK_SIZE>>>(
        (const float*)graph->node_embeddings->data,
        (const float*)euclidean_gradients->data,
        (float*)riem_grad->data,
        graph->num_nodes,
        graph->embed_dim);
    CUDA_CHECK(cudaGetLastError());

    // Apply exponential map update
    kernel_hyperbolic_exp_map<<<graph->num_nodes, BLOCK_SIZE>>>(
        (float*)graph->node_embeddings->data,
        (const float*)riem_grad->data,
        learning_rate,
        graph->num_nodes,
        graph->embed_dim);
    CUDA_CHECK(cudaGetLastError());

    nimcp_gpu_tensor_destroy(riem_grad);
    return true;
}

/**
 * @brief Convert Euclidean embeddings to hyperbolic via exponential map at origin
 */
__global__ void kernel_euclidean_to_hyperbolic(
    float* embeddings,
    uint32_t num_nodes,
    uint32_t embed_dim)
{
    uint32_t node = blockIdx.x;
    if (node >= num_nodes) return;

    float* emb = embeddings + node * embed_dim;

    // Compute norm
    __shared__ float norm_sq_shared;
    float local_norm_sq = 0.0f;

    for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
        local_norm_sq += emb[d] * emb[d];
    }
    local_norm_sq = warp_reduce_sum(local_norm_sq);

    if (threadIdx.x == 0) {
        norm_sq_shared = local_norm_sq;
    }
    __syncthreads();

    float norm = sqrtf(norm_sq_shared);

    // Exponential map at origin: tanh(||v||/2) * v / ||v||
    if (norm > 1e-8f) {
        float scale = tanhf(norm / 2.0f) / norm;
        for (uint32_t d = threadIdx.x; d < embed_dim; d += blockDim.x) {
            emb[d] *= scale;
        }
    }
}

bool nimcp_gpu_euclidean_to_hyperbolic(nimcp_gpu_knowledge_graph_t* graph)
{
    if (!graph) {
        LOG_ERROR("Invalid graph parameter");
        return false;
    }

    if (!graph->node_embeddings) {
        LOG_ERROR("Graph has no embeddings");
        return false;
    }

    kernel_euclidean_to_hyperbolic<<<graph->num_nodes, BLOCK_SIZE>>>(
        (float*)graph->node_embeddings->data,
        graph->num_nodes,
        graph->embed_dim);
    CUDA_CHECK(cudaGetLastError());

    graph->is_hyperbolic = true;
    return true;
}

/**
 * @brief Hyperbolic k-NN search
 */
__global__ void kernel_hyperbolic_all_distances(
    const float* embeddings,
    uint32_t query_node,
    uint32_t num_nodes,
    uint32_t embed_dim,
    float* distances)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    const float* query = embeddings + query_node * embed_dim;
    const float* target = embeddings + idx * embed_dim;

    distances[idx] = device_hyperbolic_distance(query, target, embed_dim);
}

bool nimcp_gpu_hyperbolic_knn(
    nimcp_gpu_knowledge_graph_t* graph,
    uint32_t query_node,
    uint32_t k,
    nimcp_similarity_result_t* result)
{
    if (!graph || !result) {
        LOG_ERROR("Invalid parameters for hyperbolic_knn");
        return false;
    }

    if (!graph->is_hyperbolic) {
        LOG_ERROR("Graph embeddings are not in hyperbolic space");
        return false;
    }

    if (query_node >= graph->num_nodes) {
        LOG_ERROR("Query node out of range");
        return false;
    }

    k = (k > graph->num_nodes) ? graph->num_nodes : k;

    // Allocate distance buffer
    float* d_distances = NULL;
    CUDA_CHECK(cudaMalloc(&d_distances, graph->num_nodes * sizeof(float)));

    // Compute all hyperbolic distances
    kernel_hyperbolic_all_distances<<<GRID_SIZE(graph->num_nodes), BLOCK_SIZE>>>(
        (const float*)graph->node_embeddings->data,
        query_node,
        graph->num_nodes,
        graph->embed_dim,
        d_distances);
    CUDA_CHECK(cudaGetLastError());

    // Create result tensors
    size_t k_dims[] = {k};
    result->indices = nimcp_gpu_tensor_create(graph->ctx, k_dims, 1, NIMCP_GPU_PRECISION_UINT32);
    result->scores = nimcp_gpu_tensor_create(graph->ctx, k_dims, 1, NIMCP_GPU_PRECISION_FP32);
    result->k = k;

    if (!result->indices || !result->scores) {
        cudaFree(d_distances);
        nimcp_similarity_result_destroy(result);
        return false;
    }

    // Top-k selection (smallest distances)
    // Negate distances so we can reuse the max-based top_k kernel
    // Or use a separate min-based kernel

    // For now, copy to host and do selection there
    float* h_distances = (float*)malloc(graph->num_nodes * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_distances, d_distances, graph->num_nodes * sizeof(float), cudaMemcpyDeviceToHost));

    uint32_t* h_indices = (uint32_t*)malloc(k * sizeof(uint32_t));
    float* h_scores = (float*)malloc(k * sizeof(float));

    // Initialize with worst values
    for (uint32_t i = 0; i < k; i++) {
        h_indices[i] = 0;
        h_scores[i] = FLT_MAX;
    }

    // Find k smallest distances
    for (uint32_t n = 0; n < graph->num_nodes; n++) {
        float dist = h_distances[n];

        uint32_t insert_pos = k;
        for (uint32_t i = 0; i < k; i++) {
            if (dist < h_scores[i]) {
                insert_pos = i;
                break;
            }
        }

        if (insert_pos < k) {
            for (uint32_t i = k - 1; i > insert_pos; i--) {
                h_indices[i] = h_indices[i - 1];
                h_scores[i] = h_scores[i - 1];
            }
            h_indices[insert_pos] = n;
            h_scores[insert_pos] = dist;
        }
    }

    CUDA_CHECK(cudaMemcpy(result->indices->data, h_indices, k * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result->scores->data, h_scores, k * sizeof(float), cudaMemcpyHostToDevice));

    free(h_distances);
    free(h_indices);
    free(h_scores);
    cudaFree(d_distances);

    return true;
}

//=============================================================================
// Utility Functions
//=============================================================================

/**
 * @brief Compute degree statistics
 */
__global__ void kernel_compute_degrees(
    const uint32_t* row_offsets,
    uint32_t num_nodes,
    uint32_t* degrees,
    uint32_t* max_degree)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;

    uint32_t degree = row_offsets[idx + 1] - row_offsets[idx];
    degrees[idx] = degree;
    atomicMax(max_degree, degree);
}

bool nimcp_gpu_knowledge_graph_stats(
    nimcp_gpu_knowledge_graph_t* graph,
    float* avg_degree,
    uint32_t* max_degree,
    float* embedding_norm)
{
    if (!graph) {
        LOG_ERROR("Invalid graph parameter");
        return false;
    }

    // Compute degree statistics
    uint32_t* d_degrees = NULL;
    uint32_t* d_max_degree = NULL;
    CUDA_CHECK(cudaMalloc(&d_degrees, graph->num_nodes * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_max_degree, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_max_degree, 0, sizeof(uint32_t)));

    kernel_compute_degrees<<<GRID_SIZE(graph->num_nodes), BLOCK_SIZE>>>(
        (const uint32_t*)graph->row_offsets->data,
        graph->num_nodes,
        d_degrees,
        d_max_degree);
    CUDA_CHECK(cudaGetLastError());

    // Get max degree
    CUDA_CHECK(cudaMemcpy(max_degree, d_max_degree, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Compute average degree
    *avg_degree = (float)graph->num_edges / (float)graph->num_nodes;

    // Compute average embedding norm if embeddings exist
    if (graph->node_embeddings && embedding_norm) {
        float* h_embeddings = (float*)malloc(graph->num_nodes * graph->embed_dim * sizeof(float));
        CUDA_CHECK(cudaMemcpy(h_embeddings, graph->node_embeddings->data,
                              graph->num_nodes * graph->embed_dim * sizeof(float),
                              cudaMemcpyDeviceToHost));

        float total_norm = 0.0f;
        for (uint32_t n = 0; n < graph->num_nodes; n++) {
            float norm = 0.0f;
            for (uint32_t d = 0; d < graph->embed_dim; d++) {
                float val = h_embeddings[n * graph->embed_dim + d];
                norm += val * val;
            }
            total_norm += sqrtf(norm);
        }
        *embedding_norm = total_norm / graph->num_nodes;

        free(h_embeddings);
    } else if (embedding_norm) {
        *embedding_norm = 0.0f;
    }

    cudaFree(d_degrees);
    cudaFree(d_max_degree);

    return true;
}

bool nimcp_gpu_knowledge_graph_is_valid(const nimcp_gpu_knowledge_graph_t* graph)
{
    if (!graph) {
        return false;
    }

    if (!graph->ctx) {
        LOG_ERROR("Graph has no GPU context");
        return false;
    }

    if (!graph->row_offsets) {
        LOG_ERROR("Graph has no row_offsets");
        return false;
    }

    if (graph->num_edges > 0 && !graph->col_indices) {
        LOG_ERROR("Graph has edges but no col_indices");
        return false;
    }

    if (graph->num_nodes == 0) {
        LOG_ERROR("Graph has no nodes");
        return false;
    }

    return true;
}

#else  // !NIMCP_ENABLE_CUDA

// Stub implementations when CUDA is not enabled
#include "gpu/knowledge/nimcp_knowledge_graph_gpu.h"
#include "utils/logging/nimcp_logging.h"

#define LOG_MODULE "KNOWLEDGE_GPU"

nimcp_gpu_knowledge_graph_t* nimcp_gpu_knowledge_graph_create(
    nimcp_gpu_context_t* ctx,
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const float* edge_weights,
    uint32_t num_nodes,
    uint32_t num_edges,
    uint32_t embed_dim)
{
    (void)ctx; (void)row_offsets; (void)col_indices; (void)edge_weights;
    (void)num_nodes; (void)num_edges; (void)embed_dim;
    LOG_ERROR("CUDA not enabled - cannot create GPU knowledge graph");
    return NULL;
}

void nimcp_gpu_knowledge_graph_destroy(nimcp_gpu_knowledge_graph_t* graph)
{
    (void)graph;
}

bool nimcp_gpu_knowledge_graph_set_embeddings(nimcp_gpu_knowledge_graph_t* graph, const float* embeddings)
{
    (void)graph; (void)embeddings;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_knowledge_graph_set_hyperbolic_embeddings(nimcp_gpu_knowledge_graph_t* graph, const float* embeddings)
{
    (void)graph; (void)embeddings;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_bfs(nimcp_gpu_knowledge_graph_t* graph, uint32_t source, int32_t max_depth, nimcp_graph_traversal_result_t* result)
{
    (void)graph; (void)source; (void)max_depth; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_bfs_multi_source(nimcp_gpu_knowledge_graph_t* graph, const uint32_t* sources, uint32_t num_sources, int32_t max_depth, nimcp_graph_traversal_result_t* result)
{
    (void)graph; (void)sources; (void)num_sources; (void)max_depth; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_dfs(nimcp_gpu_knowledge_graph_t* graph, uint32_t source, int32_t max_depth, nimcp_graph_traversal_result_t* result)
{
    (void)graph; (void)source; (void)max_depth; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_shortest_path(nimcp_gpu_knowledge_graph_t* graph, uint32_t source, uint32_t target, uint32_t* path, uint32_t* len)
{
    (void)graph; (void)source; (void)target; (void)path; (void)len;
    LOG_ERROR("CUDA not enabled");
    return false;
}

void nimcp_graph_traversal_result_destroy(nimcp_graph_traversal_result_t* result)
{
    (void)result;
}

bool nimcp_gpu_node_similarity(nimcp_gpu_knowledge_graph_t* graph, uint32_t a, uint32_t b, float* sim)
{
    (void)graph; (void)a; (void)b; (void)sim;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_pairwise_similarity(nimcp_gpu_knowledge_graph_t* graph, const uint32_t* indices, uint32_t n, nimcp_gpu_tensor_t* out)
{
    (void)graph; (void)indices; (void)n; (void)out;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_knn_similarity(nimcp_gpu_knowledge_graph_t* graph, uint32_t query, uint32_t k, nimcp_similarity_result_t* result)
{
    (void)graph; (void)query; (void)k; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_knn_similarity_embedding(nimcp_gpu_knowledge_graph_t* graph, const float* emb, uint32_t k, nimcp_similarity_result_t* result)
{
    (void)graph; (void)emb; (void)k; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

void nimcp_similarity_result_destroy(nimcp_similarity_result_t* result)
{
    (void)result;
}

bool nimcp_gpu_embedding_update(nimcp_gpu_knowledge_graph_t* graph, const nimcp_gpu_tensor_t* grad, float lr)
{
    (void)graph; (void)grad; (void)lr;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_triplet_loss(nimcp_gpu_knowledge_graph_t* graph, const uint32_t* a, const uint32_t* p, const uint32_t* n, uint32_t batch, float margin, float* loss)
{
    (void)graph; (void)a; (void)p; (void)n; (void)batch; (void)margin; (void)loss;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_normalize_embeddings(nimcp_gpu_knowledge_graph_t* graph)
{
    (void)graph;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_aggregate_neighbors(nimcp_gpu_knowledge_graph_t* graph, nimcp_aggregate_mode_t mode, nimcp_gpu_tensor_t* out)
{
    (void)graph; (void)mode; (void)out;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_aggregate_attention(nimcp_gpu_knowledge_graph_t* graph, const nimcp_gpu_tensor_t* q, const nimcp_gpu_tensor_t* k, nimcp_gpu_tensor_t* out)
{
    (void)graph; (void)q; (void)k; (void)out;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_multi_hop_aggregate(nimcp_gpu_knowledge_graph_t* graph, uint32_t hops, nimcp_aggregate_mode_t mode, nimcp_gpu_tensor_t* out)
{
    (void)graph; (void)hops; (void)mode; (void)out;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_subgraph_match(nimcp_gpu_knowledge_graph_t* graph, nimcp_gpu_knowledge_graph_t* pattern, uint32_t max, nimcp_subgraph_match_result_t* result)
{
    (void)graph; (void)pattern; (void)max; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_subgraph_match_approximate(nimcp_gpu_knowledge_graph_t* graph, nimcp_gpu_knowledge_graph_t* pattern, float thresh, uint32_t max, nimcp_subgraph_match_result_t* result)
{
    (void)graph; (void)pattern; (void)thresh; (void)max; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

void nimcp_subgraph_match_result_destroy(nimcp_subgraph_match_result_t* result)
{
    (void)result;
}

bool nimcp_gpu_hyperbolic_distance(nimcp_gpu_knowledge_graph_t* graph, uint32_t a, uint32_t b, float* dist)
{
    (void)graph; (void)a; (void)b; (void)dist;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_hyperbolic_pairwise_distance(nimcp_gpu_knowledge_graph_t* graph, const uint32_t* indices, uint32_t n, nimcp_gpu_tensor_t* out)
{
    (void)graph; (void)indices; (void)n; (void)out;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_hyperbolic_sgd_step(nimcp_gpu_knowledge_graph_t* graph, const nimcp_gpu_tensor_t* grad, float lr)
{
    (void)graph; (void)grad; (void)lr;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_euclidean_to_hyperbolic(nimcp_gpu_knowledge_graph_t* graph)
{
    (void)graph;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_hyperbolic_knn(nimcp_gpu_knowledge_graph_t* graph, uint32_t query, uint32_t k, nimcp_similarity_result_t* result)
{
    (void)graph; (void)query; (void)k; (void)result;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_knowledge_graph_stats(nimcp_gpu_knowledge_graph_t* graph, float* avg_deg, uint32_t* max_deg, float* norm)
{
    (void)graph; (void)avg_deg; (void)max_deg; (void)norm;
    LOG_ERROR("CUDA not enabled");
    return false;
}

bool nimcp_gpu_knowledge_graph_is_valid(const nimcp_gpu_knowledge_graph_t* graph)
{
    (void)graph;
    return false;
}

#endif  // NIMCP_ENABLE_CUDA
