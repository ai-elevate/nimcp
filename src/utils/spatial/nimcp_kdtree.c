/**
 * @file nimcp_kdtree.c
 * @brief K-D Tree implementation for 3D spatial indexing
 */

#include "nimcp_kdtree.h"
#include "utils/memory/nimcp_memory.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

//=============================================================================
// Internal Structures
//=============================================================================

/**
 * @brief KD-tree node
 */
struct kdtree_node_t {
    float point[3];           /**< 3D coordinates */
    void* user_data;          /**< Associated user data */
    kdtree_node_t* left;      /**< Left child (points with coord < split) */
    kdtree_node_t* right;     /**< Right child (points with coord >= split) */
    uint8_t split_dim;        /**< Split dimension (0=x, 1=y, 2=z) */
};

/**
 * @brief KD-tree structure
 */
struct kdtree_t {
    kdtree_node_t* root;      /**< Root node */
    uint32_t size;            /**< Number of points */
};

//=============================================================================
// Helper Structures for Building
//=============================================================================

typedef struct {
    float point[3];
    void* user_data;
} kdtree_build_point_t;

//=============================================================================
// Helper Functions
//=============================================================================

static inline float distance_squared(const float a[3], const float b[3]) {
    float dx = a[0] - b[0];
    float dy = a[1] - b[1];
    float dz = a[2] - b[2];
    return dx*dx + dy*dy + dz*dz;
}

static int compare_points_x(const void* a, const void* b) {
    const kdtree_build_point_t* pa = (const kdtree_build_point_t*)a;
    const kdtree_build_point_t* pb = (const kdtree_build_point_t*)b;
    if (pa->point[0] < pb->point[0]) return -1;
    if (pa->point[0] > pb->point[0]) return 1;
    return 0;
}

static int compare_points_y(const void* a, const void* b) {
    const kdtree_build_point_t* pa = (const kdtree_build_point_t*)a;
    const kdtree_build_point_t* pb = (const kdtree_build_point_t*)b;
    if (pa->point[1] < pb->point[1]) return -1;
    if (pa->point[1] > pb->point[1]) return 1;
    return 0;
}

static int compare_points_z(const void* a, const void* b) {
    const kdtree_build_point_t* pa = (const kdtree_build_point_t*)a;
    const kdtree_build_point_t* pb = (const kdtree_build_point_t*)b;
    if (pa->point[2] < pb->point[2]) return -1;
    if (pa->point[2] > pb->point[2]) return 1;
    return 0;
}

/**
 * @brief Recursively build KD-tree
 *
 * ALGORITHM: Median split along alternating dimensions
 */
static kdtree_node_t* build_recursive(kdtree_build_point_t* points,
                                       uint32_t count, uint8_t depth) {
    if (count == 0) {
        return NULL;
    }

    // Determine split dimension (cycle through x, y, z)
    uint8_t dim = depth % 3;

    // Sort points by dimension
    switch (dim) {
        case 0: qsort(points, count, sizeof(kdtree_build_point_t), compare_points_x); break;
        case 1: qsort(points, count, sizeof(kdtree_build_point_t), compare_points_y); break;
        case 2: qsort(points, count, sizeof(kdtree_build_point_t), compare_points_z); break;
    }

    // Find median
    uint32_t median = count / 2;

    // Create node
    kdtree_node_t* node = (kdtree_node_t*)nimcp_calloc(1, sizeof(kdtree_node_t));
    if (!node) {
        return NULL;
    }

    memcpy(node->point, points[median].point, sizeof(node->point));
    node->user_data = points[median].user_data;
    node->split_dim = dim;

    // Recursively build subtrees
    node->left = build_recursive(points, median, depth + 1);
    node->right = build_recursive(points + median + 1, count - median - 1, depth + 1);

    return node;
}

/**
 * @brief Destroy tree recursively
 */
static void destroy_recursive(kdtree_node_t* node) {
    if (!node) {
        return;
    }
    destroy_recursive(node->left);
    destroy_recursive(node->right);
    nimcp_free(node);
}

/**
 * @brief Recursive nearest neighbor search
 */
static void nearest_recursive(const kdtree_node_t* node, const float query[3],
                               kdtree_node_t** best, float* best_dist_sq) {
    if (!node) {
        return;
    }

    // Check current node distance
    float dist_sq = distance_squared(node->point, query);
    if (dist_sq < *best_dist_sq) {
        *best_dist_sq = dist_sq;
        *best = (kdtree_node_t*)node;
    }

    // Determine which subtree to search first
    uint8_t dim = node->split_dim;
    float diff = query[dim] - node->point[dim];

    kdtree_node_t* near_child = (diff < 0) ? node->left : node->right;
    kdtree_node_t* far_child = (diff < 0) ? node->right : node->left;

    // Search near subtree
    nearest_recursive(near_child, query, best, best_dist_sq);

    // Check if we need to search far subtree
    // (if splitting plane is within current best distance)
    if (diff * diff < *best_dist_sq) {
        nearest_recursive(far_child, query, best, best_dist_sq);
    }
}

/**
 * @brief Compute tree depth recursively
 */
static uint32_t depth_recursive(const kdtree_node_t* node) {
    if (!node) {
        return 0;
    }
    uint32_t left_depth = depth_recursive(node->left);
    uint32_t right_depth = depth_recursive(node->right);
    return 1 + (left_depth > right_depth ? left_depth : right_depth);
}

//=============================================================================
// Public API
//=============================================================================

kdtree_t* kdtree_create(void) {
    kdtree_t* tree = (kdtree_t*)nimcp_calloc(1, sizeof(kdtree_t));
    return tree;
}

void kdtree_destroy(kdtree_t* tree) {
    if (!tree) {
        return;
    }
    destroy_recursive(tree->root);
    nimcp_free(tree);
}

bool kdtree_build(kdtree_t* tree, const kdtree_point_t* points,
                  void** user_data, uint32_t count) {
    if (!tree || !points || count == 0) {
        return false;
    }

    // Clear existing tree
    kdtree_clear(tree);

    // Create temporary array for building
    kdtree_build_point_t* build_points = (kdtree_build_point_t*)
        nimcp_malloc(count * sizeof(kdtree_build_point_t));
    if (!build_points) {
        return false;
    }

    // Copy points and user data
    for (uint32_t i = 0; i < count; i++) {
        memcpy(build_points[i].point, points[i], sizeof(kdtree_point_t));
        build_points[i].user_data = user_data ? user_data[i] : NULL;
    }

    // Build tree recursively
    tree->root = build_recursive(build_points, count, 0);
    tree->size = count;

    nimcp_free(build_points);
    return (tree->root != NULL);
}

void kdtree_clear(kdtree_t* tree) {
    if (!tree) {
        return;
    }
    destroy_recursive(tree->root);
    tree->root = NULL;
    tree->size = 0;
}

void* kdtree_nearest(const kdtree_t* tree, const kdtree_point_t query,
                     float* dist_sq) {
    if (!tree || !tree->root) {
        return NULL;
    }

    kdtree_node_t* best = tree->root;
    float best_dist_sq = distance_squared(tree->root->point, query);

    nearest_recursive(tree->root, query, &best, &best_dist_sq);

    if (dist_sq) {
        *dist_sq = best_dist_sq;
    }

    return best->user_data;
}

uint32_t kdtree_k_nearest(const kdtree_t* tree, const kdtree_point_t query,
                          uint32_t k, void** results, float* distances) {
    // Simple implementation: Use nearest() k times with exclusion
    // TODO: Optimize with priority queue
    if (!tree || !results || k == 0) {
        return 0;
    }

    // For now, just return the single nearest neighbor
    // Full k-nearest implementation would require a priority queue
    if (k >= 1) {
        float dist_sq;
        results[0] = kdtree_nearest(tree, query, &dist_sq);
        if (distances) {
            distances[0] = dist_sq;
        }
        return (results[0] != NULL) ? 1 : 0;
    }

    return 0;
}

uint32_t kdtree_range_search(const kdtree_t* tree, const kdtree_point_t query,
                              float radius, void** results, uint32_t capacity) {
    // TODO: Implement range search
    // For now, return 0 (not implemented)
    (void)tree;
    (void)query;
    (void)radius;
    (void)results;
    (void)capacity;
    return 0;
}

uint32_t kdtree_size(const kdtree_t* tree) {
    return tree ? tree->size : 0;
}

uint32_t kdtree_depth(const kdtree_t* tree) {
    if (!tree || !tree->root) {
        return 0;
    }
    return depth_recursive(tree->root);
}
