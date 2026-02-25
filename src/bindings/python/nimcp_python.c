/**
 * @file nimcp_python.c
 * @brief Comprehensive Python bindings for NIMCP v2.6.3 with all features
 * @version 2.6.3
 * @date 2025-11-05
 *
 * WHAT: Complete Python interface to NIMCP unified API
 * WHY:  Enable Python developers to leverage NIMCP's cognitive capabilities
 * HOW:  Python C extension using CPython API, wrapping nimcp.h public interface
 *
 * FEATURES IMPLEMENTED:
 * - Brain API: create, learn, predict, save/load
 * - Neural Network API: create, forward, backward, save/load
 * - v2.6.3 Enhancements:
 *   * Batch Processing: predict_batch, learn_batch
 *   * Async Inference: async_predict with futures
 *   * Checkpointing: enable_checkpointing, checkpoint, list_checkpoints
 *   * SIMD Operations: dot_product, vector_add
 *   * Network Statistics: get_stats
 * - Memory management: Proper Python reference counting
 * - Error handling: Python exceptions with NIMCP error messages
 *
 * CODING STANDARDS:
 * - WHAT/WHY/HOW documentation for all functions
 * - Guard clauses for input validation
 * - Consistent error handling patterns
 * - Memory safety (no leaks, proper cleanup)
 * - Clear API that matches NIMCP conventions
 *
 * EXAMPLE USAGE:
 * ```python
 * import nimcp
 *
 * # Create and train a brain
 * brain = nimcp.Brain("classifier", nimcp.BRAIN_SMALL, nimcp.TASK_CLASSIFICATION, 10, 5)
 * brain.learn([0.1, 0.2, ...], "class_a", 0.9)
 * label, confidence = brain.predict([0.15, 0.25, ...])
 *
 * # Enable checkpointing
 * brain.enable_checkpointing("/tmp/checkpoints")
 * brain.checkpoint("manual_checkpoint")
 *
 * # Batch processing
 * labels, confidences = brain.predict_batch([[0.1, 0.2, ...], [0.3, 0.4, ...]])
 * ```
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "nimcp.h"

/* Socratic active learning: cognitive module + LGSS bindings */
#include "api/nimcp_api_internal.h"
#include "core/brain/nimcp_brain_internal.h"  /* brain_struct access for cortex fields */
#include "core/brain/accessors/nimcp_brain_accessors.h"
#include "cognitive/curiosity/nimcp_curiosity.h"
#include "cognitive/consolidation/nimcp_consolidation.h"
#include "cognitive/introspection/nimcp_introspection.h"
#include "cognitive/nimcp_self_model.h"
#include "security/lgss/perception/nimcp_lgss_content_filter.h"
#include "perception/nimcp_audio_cortex.h"
#include "perception/nimcp_visual_cortex.h"
#include "perception/nimcp_speech_cortex.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#include <stddef.h>  /* for NULL */
#include "utils/memory/nimcp_memory.h"
#include "security/nimcp_bbb_helpers.h"
#include "constants/nimcp_buffer_constants.h"
#include "core/brain/learning/nimcp_brain_learning.h"
#include "cognitive/training/nimcp_training_integration.h"
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Shutdown guard — set by nimcp.shutdown(), prevents Brain_dealloc crash */
static int g_py_shutdown_called = 0;

/** Global health agent for python module */
static nimcp_health_agent_t* g_python_health_agent = NULL;

/**
 * @brief Set health agent for python heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void python_set_health_agent(nimcp_health_agent_t* agent) {
    g_python_health_agent = agent;
}

/** @brief Send heartbeat from python module */
static inline void python_heartbeat(const char* operation, float progress) {
    if (g_python_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_python_health_agent, operation, progress);
    }
}


// Forward declaration for signal filter module
extern int init_signal_filter_module(PyObject* module);

//=============================================================================
// Utility: Convert Python list to float array
//=============================================================================

/**
 * WHAT: Convert Python list of floats to C float array
 * WHY:  Bridge between Python and C data representations
 * HOW:  Iterate list, extract floats, allocate array
 *
 * @param list Python list object
 * @param size Output: size of array
 * @return float array (caller must free) or NULL on error
 */
static float* py_list_to_float_array(PyObject* list, Py_ssize_t* size) {
    // Guard: Validate input
    if (!PyList_Check(list)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "py_list_to_float_array: Expected list of floats");
        PyErr_SetString(PyExc_TypeError, "Expected list of floats");
        return NULL;
    }

    *size = PyList_Size(list);
    if (*size == 0) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "py_list_to_float_array: List cannot be empty");
        PyErr_SetString(PyExc_ValueError, "List cannot be empty");
        return NULL;
    }

    // Allocate array
    float* array = (float*)nimcp_malloc((*size) * sizeof(float));
    if (!array) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, (*size) * sizeof(float),
                          "py_list_to_float_array: Failed to allocate float array");
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate float array");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "array is NULL");

        return NULL;
    }

    // Convert elements
    for (Py_ssize_t i = 0; i < *size; i++) {
        PyObject* item = PyList_GetItem(list, i);
        if (!PyFloat_Check(item) && !PyLong_Check(item)) {
            nimcp_free(array);
            NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM,
                       "py_list_to_float_array: List element %zd is not a number", i);
            PyErr_SetString(PyExc_TypeError, "List must contain only numbers");
            return NULL;
        }
        array[i] = (float)PyFloat_AsDouble(item);
    }

    return array;
}

//=============================================================================
// Brain Object Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    nimcp_brain_t brain;
    consolidation_community_cache_t* community_cache;  /* Cached community topology */
} BrainObject;

/**
 * WHAT: Deallocate Brain object
 * WHY:  Clean up resources when Python object is garbage collected
 * HOW:  Call nimcp_brain_destroy, free Python object memory
 */
static void Brain_dealloc(BrainObject* self) {
    if (self->community_cache) {
        consolidation_community_cache_destroy(self->community_cache);
        self->community_cache = NULL;
    }
    if (self->brain && !g_py_shutdown_called) {
        nimcp_brain_destroy(self->brain);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

/**
 * WHAT: Allocate new Brain object
 * WHY:  Create Python object to hold brain handle
 * HOW:  Allocate memory, initialize to NULL
 */
static PyObject* Brain_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    BrainObject* self = (BrainObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->brain = NULL;
        self->community_cache = NULL;
    }
    return (PyObject*)self;
}

/**
 * WHAT: Initialize Brain object
 * WHY:  Create NIMCP brain with specified configuration
 * HOW:  Parse arguments, call nimcp_brain_create
 *
 * @param name Brain name (for logging/identification)
 * @param size Brain size (BRAIN_TINY/SMALL/MEDIUM/LARGE)
 * @param task Task type (TASK_CLASSIFICATION/REGRESSION/etc)
 * @param num_inputs Number of input features
 * @param num_outputs Number of output classes/values
 * @return 0 on success, -1 on error
 */
static int Brain_init(BrainObject* self, PyObject* args, PyObject* kwds) {
    const char* name;
    int size = NIMCP_BRAIN_SMALL;
    int task = NIMCP_TASK_CLASSIFICATION;
    unsigned int num_inputs = 10;
    unsigned int num_outputs = 10;
    unsigned int neuron_count = 0;

    static char* kwlist[] = {"name", "size", "task", "num_inputs", "num_outputs", "neuron_count", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|iiIII", kwlist,
                                     &name, &size, &task, &num_inputs, &num_outputs, &neuron_count)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_init: Invalid arguments");
        return -1;
    }

    if (neuron_count > 0) {
        self->brain = nimcp_brain_create_with_neurons(name, (nimcp_brain_task_t)task,
                                                       num_inputs, num_outputs, neuron_count);
    } else {
        self->brain = nimcp_brain_create(name, (nimcp_brain_size_t)size,
                                         (nimcp_brain_task_t)task, num_inputs, num_outputs);
    }

    if (!self->brain) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_NOT_INITIALIZED, 0, "python_binding",
                         "Brain_init: Failed to create brain '%s'", name);
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return -1;
    }

    return 0;
}

/**
 * WHAT: Train brain on single example
 * WHY:  Supervised learning from labeled data
 * HOW:  Convert Python list to array, call nimcp_brain_learn_example
 *
 * @param features List of feature values
 * @param label Class label (string)
 * @param confidence Training confidence (0-1, default 1.0)
 * @return float loss value
 */
static PyObject* Brain_learn(BrainObject* self, PyObject* args) {
    PyObject* features_list;
    const char* label;
    float confidence = 1.0F;

    if (!PyArg_ParseTuple(args, "Os|f", &features_list, &label, &confidence)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_learn: Invalid arguments");
        return NULL;
    }

    Py_ssize_t num_features;
    float* features = py_list_to_float_array(features_list, &num_features);
    if (!features) {
        // Exception already thrown in py_list_to_float_array
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "features is NULL");

        return NULL;
    }

    /* P1-49: nimcp_brain_learn_example returns nimcp_status_t, NOT float.
     * Error codes are POSITIVE (not negative), so checking < 0.0F was wrong. */
    nimcp_status_t status;
    nimcp_brain_t brain_ref = self->brain;
    uint32_t nf = (uint32_t)num_features;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_learn_example(brain_ref, features, nf, label, confidence);
    nimcp_free(features);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "python_binding",
                         "Brain_learn: Failed to learn example with label '%s'", label);
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    /* Return actual loss from the C learning engine */
    float loss = nimcp_brain_get_last_loss(self->brain);
    return PyFloat_FromDouble((double)loss);
}

/**
 * WHAT: Make prediction on single example
 * WHY:  Inference after training
 * HOW:  Convert features, call nimcp_brain_predict
 *
 * @param features List of feature values
 * @return Tuple (label, confidence)
 */
static PyObject* Brain_predict(BrainObject* self, PyObject* args) {
    PyObject* features_list;

    if (!PyArg_ParseTuple(args, "O", &features_list)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_predict: Invalid arguments");
        return NULL;
    }

    Py_ssize_t num_features;
    float* features = py_list_to_float_array(features_list, &num_features);
    if (!features) {
        // Exception already thrown in py_list_to_float_array
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "features is NULL");

        return NULL;
    }

    char label[NIMCP_NAME_BUFFER_SIZE];
    float confidence;
    nimcp_status_t status;
    nimcp_brain_t brain_ref = self->brain;
    uint32_t nf = (uint32_t)num_features;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_predict(brain_ref, features, nf, label, &confidence);
    nimcp_free(features);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "python_binding",
                         "Brain_predict: Prediction failed");
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    return Py_BuildValue("(sf)", label, confidence);
}

/**
 * WHAT: Fast prediction — forward pass only, no cognitive stages
 * WHY:  brain.predict() runs 28 cognitive stages per call (sleep, curiosity,
 *        theory of mind, etc.), making training loops 10-100x slower than needed.
 *        predict_fast() does a pure neural network forward pass.
 * HOW:  Calls nimcp_brain_predict_fast() which uses adaptive_network_forward() directly
 */
static PyObject* Brain_predict_fast(BrainObject* self, PyObject* args) {
    PyObject* features_list;

    if (!PyArg_ParseTuple(args, "O", &features_list)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_predict_fast: Invalid arguments");
        return NULL;
    }

    Py_ssize_t num_features;
    float* features = py_list_to_float_array(features_list, &num_features);
    if (!features) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "features is NULL");
        return NULL;
    }

    char label[NIMCP_NAME_BUFFER_SIZE];
    float confidence;
    nimcp_status_t status;
    nimcp_brain_t brain_ref = self->brain;
    uint32_t nf = (uint32_t)num_features;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_predict_fast(brain_ref, features, nf, label, &confidence);
    nimcp_free(features);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    return Py_BuildValue("(sf)", label, confidence);
}

/**
 * WHAT: Batch prediction (v2.7.0 enhancement)
 * WHY:  Efficient processing of multiple samples
 * HOW:  Convert list of lists to 2D array, call nimcp_brain_predict_batch
 *
 * @param features_batch List of feature lists
 * @return Tuple (labels_list, confidences_list)
 */
static PyObject* Brain_predict_batch(BrainObject* self, PyObject* args) {
    PyObject* features_batch;

    if (!PyArg_ParseTuple(args, "O", &features_batch)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_predict_batch: Invalid arguments");
        return NULL;
    }

    if (!PyList_Check(features_batch)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_predict_batch: features_batch must be a list");
        PyErr_SetString(PyExc_TypeError, "features_batch must be a list of lists");
        return NULL;
    }

    Py_ssize_t batch_size = PyList_Size(features_batch);
    if (batch_size == 0) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_predict_batch: Batch cannot be empty");
        PyErr_SetString(PyExc_ValueError, "Batch cannot be empty");
        return NULL;
    }

    // Get number of features from first example
    PyObject* first_example = PyList_GetItem(features_batch, 0);
    Py_ssize_t num_features;
    float* first_features = py_list_to_float_array(first_example, &num_features);
    if (!first_features) {
        // Exception already thrown in py_list_to_float_array
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "first_features is NULL");

        return NULL;
    }

    // Allocate batch arrays
    const float** features_ptrs = (const float**)nimcp_malloc(batch_size * sizeof(float*));
    float** feature_arrays = (float**)nimcp_malloc(batch_size * sizeof(float*));
    char** labels = (char**)nimcp_malloc(batch_size * sizeof(char*));
    float* confidences = (float*)nimcp_malloc(batch_size * sizeof(float));

    if (!features_ptrs || !feature_arrays || !labels || !confidences) {
        nimcp_free(first_features);
        nimcp_free(features_ptrs);
        nimcp_free(feature_arrays);
        nimcp_free(labels);
        nimcp_free(confidences);
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, batch_size * sizeof(float*),
                          "Brain_predict_batch: Failed to allocate batch arrays");
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate batch arrays");
        return NULL;
    }

    // Convert all feature lists
    feature_arrays[0] = first_features;
    features_ptrs[0] = first_features;

    for (Py_ssize_t i = 1; i < batch_size; i++) {
        PyObject* example = PyList_GetItem(features_batch, i);
        Py_ssize_t size;
        float* features = py_list_to_float_array(example, &size);
        if (!features || size != num_features) {
            // Clean up
            for (Py_ssize_t j = 0; j < i; j++) {
                nimcp_free(feature_arrays[j]);
            }
            nimcp_free(features_ptrs);
            nimcp_free(feature_arrays);
            nimcp_free(labels);
            nimcp_free(confidences);
            PyErr_SetString(PyExc_ValueError, "All feature lists must have same length");
            NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain_predict_batch: operation failed");
            return NULL;
        }
        feature_arrays[i] = features;
        features_ptrs[i] = features;
    }

    // Allocate label buffers
    for (Py_ssize_t i = 0; i < batch_size; i++) {
        labels[i] = (char*)nimcp_malloc(256);
        if (!labels[i]) {
            for (Py_ssize_t j = 0; j < i; j++) {
                nimcp_free(labels[j]);
            }
            for (Py_ssize_t j = 0; j < batch_size; j++) {
                nimcp_free(feature_arrays[j]);
            }
            nimcp_free(features_ptrs);
            nimcp_free(feature_arrays);
            nimcp_free(labels);
            nimcp_free(confidences);
            NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, 256,
                              "Brain_predict_batch: Failed to allocate label buffer %zd", i);
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate label buffers");
            return NULL;
        }
    }

    // Call batch prediction
    // NOTE: nimcp_brain_predict_batch not yet implemented in C library
    // Fallback to individual predictions for now
    nimcp_status_t status = NIMCP_OK;
    nimcp_brain_t brain_ref = self->brain;
    uint32_t nf = (uint32_t)num_features;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < batch_size; i++) {
        status = nimcp_brain_predict(brain_ref, features_ptrs[i], nf,
                                      labels[i], &confidences[i]);
        if (status != NIMCP_OK) break;
    }
    Py_END_ALLOW_THREADS

    // Build result
    PyObject* labels_list = NULL;
    PyObject* confidences_list = NULL;

    if (status == NIMCP_OK) {
        labels_list = PyList_New(batch_size);
        confidences_list = PyList_New(batch_size);

        for (Py_ssize_t i = 0; i < batch_size; i++) {
            PyList_SetItem(labels_list, i, PyUnicode_FromString(labels[i]));
            PyList_SetItem(confidences_list, i, PyFloat_FromDouble(confidences[i]));
        }
    }

    // Clean up
    for (Py_ssize_t i = 0; i < batch_size; i++) {
        nimcp_free(feature_arrays[i]);
        nimcp_free(labels[i]);
    }
    nimcp_free(features_ptrs);
    nimcp_free(feature_arrays);
    nimcp_free(labels);
    nimcp_free(confidences);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain_predict_batch: validation failed");
        return NULL;
    }

    /* P2-4: Use "NN" instead of "OO" to transfer ownership and avoid ref leak.
     * "N" steals the reference, "O" increments it - since we don't Py_DECREF
     * labels_list/confidences_list after Py_BuildValue, we must use "N". */
    return Py_BuildValue("(NN)", labels_list, confidences_list);
}

// Note: File-based checkpointing not yet in public API
// COW snapshots are available via nimcp_brain_snapshot_cow()
// Will add wrapper in future version

/**
 * WHAT: Save brain to file
 * WHY:  Persist trained brain for later use
 * HOW:  Call nimcp_brain_save
 */
static PyObject* Brain_save(BrainObject* self, PyObject* args) {
    const char* filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_save: Invalid arguments");
        return NULL;
    }

    nimcp_status_t status = nimcp_brain_save(self->brain, filepath);

    if (status != NIMCP_OK) {
        NIMCP_THROW_IO(NIMCP_ERROR_FILE_WRITE, filepath,
                      "Brain_save: Failed to save brain to '%s'", filepath);
        PyErr_SetString(PyExc_IOError, nimcp_get_error());
        return NULL;
    }

    Py_RETURN_NONE;
}

/**
 * WHAT: Load brain from file (class method)
 * WHY:  Restore previously saved brain
 * HOW:  Call nimcp_brain_load, create Python object
 */
static PyObject* Brain_load(PyTypeObject* type, PyObject* args) {
    const char* filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_load: Invalid arguments");
        return NULL;
    }

    nimcp_brain_t brain = nimcp_brain_load(filepath);

    if (!brain) {
        NIMCP_THROW_IO(NIMCP_ERROR_FILE_READ, filepath,
                      "Brain_load: Failed to load brain from '%s'", filepath);
        PyErr_SetString(PyExc_IOError, nimcp_get_error());
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "brain is NULL");


        return NULL;
    }

    BrainObject* self = (BrainObject*)type->tp_alloc(type, 0);
    if (!self) {
        NIMCP_THROW_MEMORY(NIMCP_ERROR_NO_MEMORY, sizeof(BrainObject),
                          "Brain_load: Failed to allocate BrainObject");
        nimcp_brain_destroy(brain);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate Brain object");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "self is NULL");

        return NULL;
    }
    self->brain = brain;

    return (PyObject*)self;
}

//=============================================================================
// Phase 2.8: Dynamic Brain Resizing (Python Bindings)
//=============================================================================

/**
 * WHAT: Resize brain to new neuron count
 * WHY:  Allow dynamic brain growth during training
 * HOW:  Call brain_resize from nimcp_brain_resize.h
 *
 * @param new_neuron_count Target neuron count (must be > current)
 * @return True on success, raises RuntimeError on failure
 */
static PyObject* Brain_resize(BrainObject* self, PyObject* args) {
    uint32_t new_neuron_count;

    if (!PyArg_ParseTuple(args, "I", &new_neuron_count)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_resize: Invalid arguments");
        return NULL;
    }

    // CRITICAL FIX: Pass internal brain, not handle wrapper
    // self->brain is nimcp_brain_t (struct nimcp_brain_handle*)
    // brain_resize needs brain_t (self->brain->internal_brain)
    bool success = nimcp_brain_resize(self->brain, new_neuron_count);

    if (!success) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "python_binding",
                         "Brain_resize: Failed to resize brain to %u neurons", new_neuron_count);
        PyErr_SetString(PyExc_RuntimeError, "Failed to resize brain. Check that new size > current size and sufficient memory available.");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "success is NULL");


        return NULL;
    }

    Py_RETURN_TRUE;
}

/**
 * WHAT: Auto-resize brain based on utilization
 * WHY:  Enable automatic capacity management
 * HOW:  Call brain_auto_resize on internal brain
 *
 * CRITICAL FIX: self->brain is nimcp_brain_t (handle wrapper)
 * WHY: brain_auto_resize expects brain_t, not nimcp_brain_t
 * WHAT: Dereference handle to access internal_brain field
 *
 * @return True if resize occurred, False if no resize needed
 */
static PyObject* Brain_auto_resize(BrainObject* self, PyObject* args) {
    // CRITICAL FIX: Pass internal brain, not handle wrapper
    // self->brain is nimcp_brain_t (struct nimcp_brain_handle*)
    // brain_auto_resize needs brain_t (self->brain->internal_brain)
    bool resized = nimcp_brain_auto_resize(self->brain);

    if (resized) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

/**
 * WHAT: Get current neuron count
 * WHY:  Monitor brain size for metrics/logging
 * HOW:  Call brain_get_neuron_count on internal brain
 *
 * CRITICAL FIX: Pass internal brain, not handle wrapper
 *
 * @return Integer neuron count
 */
static PyObject* Brain_get_neuron_count(BrainObject* self, PyObject* args) {
    // CRITICAL FIX: Pass internal brain, not handle wrapper
    uint32_t count = nimcp_brain_get_neuron_count(self->brain);
    return PyLong_FromUnsignedLong(count);
}

/**
 * WHAT: Get brain utilization metrics
 * WHY:  Monitor capacity and decide when to resize
 * HOW:  Call brain_get_utilization_metrics on internal brain
 *
 * CRITICAL FIX: Pass internal brain, not handle wrapper
 *
 * @return Tuple (utilization, saturation) where both are floats [0.0, 1.0]
 */
static PyObject* Brain_get_utilization_metrics(BrainObject* self, PyObject* args) {
    float utilization = 0.0F;
    float saturation = 0.0F;

    // CRITICAL FIX: Pass internal brain, not handle wrapper
    bool success = nimcp_brain_get_utilization_metrics(self->brain, &utilization, &saturation);

    if (!success) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "python_binding",
                         "Brain_get_utilization_metrics: Failed to get metrics");
        PyErr_SetString(PyExc_RuntimeError, "Failed to get utilization metrics");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "success is NULL");


        return NULL;
    }

    return Py_BuildValue("(ff)", utilization, saturation);
}

//=============================================================================
// COW Snapshot API (Python Bindings)
//=============================================================================

static void Brain_snapshot_capsule_destructor(PyObject* capsule)
{
    nimcp_brain_snapshot_t snapshot =
        (nimcp_brain_snapshot_t)PyCapsule_GetPointer(capsule, "nimcp.brain_snapshot");
    if (snapshot) {
        nimcp_brain_snapshot_destroy(snapshot);
    }
}

/**
 * WHAT: Create instant COW snapshot of brain state
 * WHY:  Enable efficient rollback during training
 * HOW:  Call nimcp_brain_snapshot_cow, wrap handle in PyCapsule
 */
static PyObject* Brain_snapshot_cow(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain_snapshot_cow: self->brain is NULL");
        return NULL;
    }

    nimcp_brain_snapshot_t snapshot;

    Py_BEGIN_ALLOW_THREADS
    snapshot = nimcp_brain_snapshot_cow(self->brain);
    Py_END_ALLOW_THREADS

    if (!snapshot) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create COW snapshot");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Brain_snapshot_cow: snapshot is NULL");
        return NULL;
    }

    return PyCapsule_New(snapshot, "nimcp.brain_snapshot",
                         Brain_snapshot_capsule_destructor);
}

/**
 * WHAT: Restore brain state from COW snapshot
 * WHY:  Instant rollback to previous state via pointer swapping
 * HOW:  Extract snapshot handle from PyCapsule, call nimcp_brain_restore_cow
 */
static PyObject* Brain_restore_cow(BrainObject* self, PyObject* args) {
    PyObject* capsule;

    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_restore_cow: Invalid arguments");
        return NULL;
    }

    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain_restore_cow: self->brain is NULL");
        return NULL;
    }

    if (!PyCapsule_IsValid(capsule, "nimcp.brain_snapshot")) {
        PyErr_SetString(PyExc_TypeError, "Expected a brain snapshot capsule from snapshot_cow()");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Brain_restore_cow: invalid capsule");
        return NULL;
    }

    nimcp_brain_snapshot_t snapshot =
        (nimcp_brain_snapshot_t)PyCapsule_GetPointer(capsule, "nimcp.brain_snapshot");

    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_restore_cow(self->brain, snapshot);
    Py_END_ALLOW_THREADS

    return PyBool_FromLong(status == NIMCP_OK);
}

/**
 * WHAT: Destroy a COW snapshot and release resources
 * WHY:  Explicit cleanup (snapshots also auto-destroy on GC)
 * HOW:  Call nimcp_brain_snapshot_destroy, invalidate capsule
 */
static PyObject* Brain_destroy_cow_snapshot(BrainObject* self, PyObject* args) {
    PyObject* capsule;

    if (!PyArg_ParseTuple(args, "O", &capsule)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_destroy_cow_snapshot: Invalid arguments");
        return NULL;
    }

    if (!PyCapsule_IsValid(capsule, "nimcp.brain_snapshot")) {
        PyErr_SetString(PyExc_TypeError, "Expected a brain snapshot capsule from snapshot_cow()");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_INVALID_PARAM, "Brain_destroy_cow_snapshot: invalid capsule");
        return NULL;
    }

    nimcp_brain_snapshot_t snapshot =
        (nimcp_brain_snapshot_t)PyCapsule_GetPointer(capsule, "nimcp.brain_snapshot");

    if (snapshot) {
        nimcp_brain_snapshot_destroy(snapshot);
        PyCapsule_SetPointer(capsule, NULL);
        PyCapsule_SetDestructor(capsule, NULL);
    }

    Py_RETURN_NONE;
}

//=============================================================================
// Brain Probe Broadcasting (Python Bindings)
//=============================================================================

/**
 * WHAT: Probe brain metrics and broadcast via bio-async
 * WHY:  Enable monitoring of brain health across multiple instances
 * HOW:  Call nimcp_brain_broadcast_probe
 */
static PyObject* Brain_broadcast_probe(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Brain_broadcast_probe: self->brain is NULL");
        return NULL;
    }

    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_broadcast_probe(self->brain);
    Py_END_ALLOW_THREADS

    return PyBool_FromLong(status == NIMCP_OK);
}

/**
 * WHAT: Get comprehensive brain metrics as a Python dict
 * WHY:  Dashboard needs real-time metrics (synapses, memory, learning rate)
 * HOW:  Call nimcp_brain_probe and build dict from the result
 */
static PyObject* Brain_probe(BrainObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    nimcp_brain_probe_t probe;
    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_probe(self->brain, &probe);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to probe brain");
        return NULL;
    }

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    SET("task_name",              PyUnicode_FromString(probe.task_name));
    SET("size",                   PyLong_FromLong(probe.size));
    SET("task",                   PyLong_FromLong(probe.task));
    SET("num_neurons",            PyLong_FromUnsignedLong(probe.num_neurons));
    SET("num_synapses",           PyLong_FromUnsignedLong(probe.num_synapses));
    SET("num_active_synapses",    PyLong_FromUnsignedLong(probe.num_active_synapses));
    SET("total_inferences",       PyLong_FromUnsignedLongLong(probe.total_inferences));
    SET("total_learning_steps",   PyLong_FromUnsignedLongLong(probe.total_learning_steps));
    SET("avg_sparsity",           PyFloat_FromDouble(probe.avg_sparsity));
    SET("avg_inference_time_us",  PyFloat_FromDouble(probe.avg_inference_time_us));
    SET("current_learning_rate",  PyFloat_FromDouble(probe.current_learning_rate));
    SET("accuracy",               PyFloat_FromDouble(probe.accuracy));
    SET("memory_bytes",           PyLong_FromSize_t(probe.memory_bytes));
    SET("num_inputs",             PyLong_FromUnsignedLong(probe.num_inputs));
    SET("num_outputs",            PyLong_FromUnsignedLong(probe.num_outputs));
    SET("is_cow_clone",           PyBool_FromLong(probe.is_cow_clone));
    SET("cow_ref_count",          PyLong_FromUnsignedLong(probe.cow_ref_count));
    SET("cow_shared_bytes",       PyLong_FromSize_t(probe.cow_shared_bytes));
    SET("cow_private_bytes",      PyLong_FromSize_t(probe.cow_private_bytes));
    SET("gpu_available",          PyBool_FromLong(probe.gpu_available));

#undef SET

    return dict;
}

/**
 * WHAT: Evaluate quality of last brain decision using two-tier rubric
 * WHY:  Human-style grading (A+ through F) of cognitive output quality
 * HOW:  Call nimcp_brain_rubric and build dict from all 15 scores + grade
 */
static PyObject* Brain_rubric(BrainObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    nimcp_rubric_t rubric;
    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_rubric(self->brain, &rubric);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError,
                        status == NIMCP_ERROR_INVALID
                            ? "No decision to evaluate — call predict() first"
                            : "Failed to evaluate rubric");
        return NULL;
    }

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    /* Tier 1 */
    SET("internal_consistency",   PyFloat_FromDouble(rubric.internal_consistency));
    SET("confidence_calibration", PyFloat_FromDouble(rubric.confidence_calibration));
    SET("completeness",           PyFloat_FromDouble(rubric.completeness));
    SET("reasoning_chain_quality", PyFloat_FromDouble(rubric.reasoning_chain_quality));
    SET("epistemic_quality",      PyFloat_FromDouble(rubric.epistemic_quality));
    SET("ethical_alignment",      PyFloat_FromDouble(rubric.ethical_alignment));
    SET("tier1_score",            PyFloat_FromDouble(rubric.tier1_score));

    /* Tier 2 */
    SET("originality",            PyFloat_FromDouble(rubric.originality));
    SET("integration_depth",      PyFloat_FromDouble(rubric.integration_depth));
    SET("communication_clarity",  PyFloat_FromDouble(rubric.communication_clarity));
    SET("engagement_quality",     PyFloat_FromDouble(rubric.engagement_quality));
    SET("empathetic_accuracy",    PyFloat_FromDouble(rubric.empathetic_accuracy));
    SET("information_density",    PyFloat_FromDouble(rubric.information_density));
    SET("tier2_score",            PyFloat_FromDouble(rubric.tier2_score));

    /* Overall */
    SET("overall_score",          PyFloat_FromDouble(rubric.overall_score));

    /* Grade as string (e.g., "A+", "B-", "C") */
    char grade_str[3] = { rubric.grade, rubric.grade_modifier == ' ' ? '\0' : rubric.grade_modifier, '\0' };
    SET("grade",                  PyUnicode_FromString(grade_str));

    SET("subsystems_available",   PyLong_FromUnsignedLong(rubric.subsystems_available));
    SET("evaluation_time_us",     PyLong_FromUnsignedLongLong(rubric.evaluation_time_us));

#undef SET

    return dict;
}

static PyObject* Brain_set_rubric_validation(BrainObject* self, PyObject* args) {
    PyObject* features_list;
    if (!PyArg_ParseTuple(args, "O", &features_list))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    Py_ssize_t num_features;
    float* features = py_list_to_float_array(features_list, &num_features);
    if (!features)
        return NULL;

    nimcp_status_t status;
    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_set_rubric_validation(self->brain, features, (uint32_t)num_features);
    Py_END_ALLOW_THREADS

    free(features);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to set rubric validation features");
        return NULL;
    }

    Py_RETURN_TRUE;
}

static PyObject* Brain_training_rubric(BrainObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    uint64_t eval_count = 0;
    float min_score = 0.0f, max_score = 0.0f, avg_score = 0.0f;
    nimcp_rubric_t last_rubric;
    memset(&last_rubric, 0, sizeof(last_rubric));

    nimcp_status_t status;
    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_get_rubric_training_stats(
        self->brain, &eval_count, &min_score, &max_score, &avg_score, &last_rubric);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get rubric training stats");
        return NULL;
    }

    if (eval_count == 0) {
        Py_RETURN_NONE;
    }

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    SET("eval_count", PyLong_FromUnsignedLongLong(eval_count));
    SET("min_score",  PyFloat_FromDouble(min_score));
    SET("max_score",  PyFloat_FromDouble(max_score));
    SET("avg_score",  PyFloat_FromDouble(avg_score));
    SET("last_score", PyFloat_FromDouble(last_rubric.overall_score));

    char grade_str[3] = { last_rubric.grade,
                          last_rubric.grade_modifier == ' ' ? '\0' : last_rubric.grade_modifier,
                          '\0' };
    SET("last_grade", PyUnicode_FromString(grade_str));

#undef SET

    return dict;
}

static PyObject* Brain_decide_full(BrainObject* self, PyObject* args) {
    PyObject* features_list;
    if (!PyArg_ParseTuple(args, "O", &features_list))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    Py_ssize_t num_features;
    float* features = py_list_to_float_array(features_list, &num_features);
    if (!features)
        return NULL;

    char label[64];
    float confidence;
    char explanation[256];
    float output_vector[1024];
    uint32_t output_size = 1024;
    uint32_t num_active_neurons = 0;
    float sparsity = 0.0f;
    uint64_t inference_time_us = 0;

    nimcp_status_t status;
    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_decide_full(
        self->brain, features, (uint32_t)num_features,
        label, &confidence, explanation,
        output_vector, &output_size,
        &num_active_neurons, &sparsity, &inference_time_us);
    Py_END_ALLOW_THREADS

    nimcp_free(features);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, "decide_full failed");
        return NULL;
    }

    uint32_t vec_len = (output_size < 1024) ? output_size : 1024;
    PyObject* vec_list = PyList_New(vec_len);
    for (uint32_t i = 0; i < vec_len; i++)
        PyList_SetItem(vec_list, i, PyFloat_FromDouble(output_vector[i]));

    PyObject* result = PyDict_New();
    PyObject* tmp;
    tmp = PyUnicode_FromString(label);
    PyDict_SetItemString(result, "label", tmp); Py_DECREF(tmp);
    tmp = PyFloat_FromDouble(confidence);
    PyDict_SetItemString(result, "confidence", tmp); Py_DECREF(tmp);
    tmp = PyUnicode_FromString(explanation);
    PyDict_SetItemString(result, "explanation", tmp); Py_DECREF(tmp);
    PyDict_SetItemString(result, "output_vector", vec_list); Py_DECREF(vec_list);
    tmp = PyLong_FromUnsignedLong(num_active_neurons);
    PyDict_SetItemString(result, "num_active_neurons", tmp); Py_DECREF(tmp);
    tmp = PyFloat_FromDouble(sparsity);
    PyDict_SetItemString(result, "sparsity", tmp); Py_DECREF(tmp);
    tmp = PyLong_FromUnsignedLongLong(inference_time_us);
    PyDict_SetItemString(result, "inference_time_us", tmp); Py_DECREF(tmp);

    return result;
}

/**
 * WHAT: Get running label-match accuracy (EMA)
 * WHY:  Monitor training progress with a meaningful metric
 * HOW:  Call nimcp_brain_get_accuracy
 */
static PyObject* Brain_get_accuracy(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float accuracy = nimcp_brain_get_accuracy(self->brain);
    return PyFloat_FromDouble((double)accuracy);
}

/**
 * @brief Freeze brain for inference-only mode
 */
static PyObject* Brain_freeze(BrainObject* self, PyObject* args) {
    (void)args;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    nimcp_status_t status = nimcp_brain_freeze(self->brain);
    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to freeze brain");
        return NULL;
    }
    Py_RETURN_TRUE;
}

//=============================================================================
// Socratic Active Learning: Cognitive Module Bindings
//=============================================================================

/**
 * WHAT: Detect knowledge gaps via curiosity module
 * WHY:  Drive active learning by finding what the brain doesn't know
 * HOW:  Call curiosity_detect_knowledge_gap() through brain accessor
 */
static PyObject* Brain_curiosity_detect_gaps(BrainObject* self, PyObject* args) {
    const char* topic;
    if (!PyArg_ParseTuple(args, "s", &topic))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    curiosity_engine_t curiosity = brain_get_curiosity(self->brain->internal_brain);
    if (!curiosity) {
        /* No curiosity engine — return empty dict rather than error */
        PyObject* dict = PyDict_New();
        return dict;
    }

    knowledge_gap_t gap;
    Py_BEGIN_ALLOW_THREADS
    gap = curiosity_detect_knowledge_gap(curiosity, topic);
    Py_END_ALLOW_THREADS

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    SET("topic", PyUnicode_FromString(gap.topic));
    SET("gap_size", PyFloat_FromDouble(gap.gap_size));
    SET("curiosity_intensity", PyFloat_FromDouble(gap.curiosity_intensity));
    SET("learning_potential", PyFloat_FromDouble(gap.learning_potential));
    SET("related_concepts", PyLong_FromUnsignedLong(gap.related_concepts));

    /* Generate questions from the gap */
    generated_question_t questions[8];
    uint32_t num_q;
    Py_BEGIN_ALLOW_THREADS
    num_q = curiosity_generate_questions(curiosity, &gap, questions, 8);
    Py_END_ALLOW_THREADS

    PyObject* q_list = PyList_New(num_q);
    if (!q_list) { Py_DECREF(dict); return NULL; }
    for (uint32_t i = 0; i < num_q; i++) {
        PyObject* q_str = PyUnicode_FromString(questions[i].question);
        if (!q_str) { Py_DECREF(q_list); Py_DECREF(dict); return NULL; }
        PyList_SET_ITEM(q_list, i, q_str);
    }
    SET("questions", q_list);

#undef SET
    return dict;
}

/**
 * WHAT: Trigger synchronous memory consolidation with configurable mode
 * WHY:  "Sleep" between training phases — replay, prune, integrate
 * HOW:  Select config preset by mode: "auto" (scale-aware), "light" (replay only), "full" (original)
 *
 * Python usage:
 *   brain.consolidate()                    # auto mode (scale-aware)
 *   brain.consolidate(mode="light")        # replay only (~ms)
 *   brain.consolidate(mode="full")         # original 10-cycle
 *   brain.consolidate(mode="auto", cycles=3)  # override cycles
 */
static PyObject* Brain_consolidate(BrainObject* self, PyObject* args, PyObject* kwds) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    const char* mode = "auto";
    unsigned int cycles = 0;
    unsigned int prune_passes = 0;

    static char* kwlist[] = {"mode", "cycles", "prune_passes", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sII", kwlist,
                                     &mode, &cycles, &prune_passes)) {
        return NULL;
    }

    /* Select config based on mode */
    consolidation_config_t config;
    if (strcmp(mode, "light") == 0) {
        config = consolidation_light_config();
    } else if (strcmp(mode, "full") == 0) {
        config = consolidation_default_config();
    } else {
        /* "auto" or unrecognized — scale-aware */
        uint32_t num_neurons = nimcp_brain_get_neuron_count(self->brain);
        config = consolidation_auto_config(num_neurons);
    }

    /* Apply overrides if specified */
    if (cycles > 0) {
        config.consolidation_cycles = cycles;
    }
    if (prune_passes > 0) {
        config.max_prune_passes = prune_passes;
    }

    bool success;

    Py_BEGIN_ALLOW_THREADS
    if (self->community_cache && self->community_cache->is_valid &&
        config.use_community_cache) {
        success = brain_consolidate_memory_community_aware(
            self->brain->internal_brain, &config, self->community_cache);
    } else {
        success = brain_consolidate_memory(self->brain->internal_brain, &config);
    }
    Py_END_ALLOW_THREADS

    if (success) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

//=============================================================================
// Training Integration API (basal ganglia, medulla, reasoning, adaptive LR)
//=============================================================================

/**
 * WHAT: Update basal ganglia reward signal
 * WHY:  Reinforcement learning needs reward/expected pairs for RPE computation
 * HOW:  Calls brain_ti_update_reward with (reward, expected) floats
 */
static PyObject* Brain_bg_update_reward(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float reward, expected;
    if (!PyArg_ParseTuple(args, "ff", &reward, &expected)) return NULL;

    int result = brain_ti_update_reward(self->brain->internal_brain, reward, expected);
    return PyLong_FromLong(result);
}

/**
 * WHAT: Get basal ganglia conflict level
 * WHY:  Conflict detection indicates competing action selections
 * HOW:  Calls brain_ti_get_conflict, returns float
 */
static PyObject* Brain_bg_get_conflict(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float conflict = brain_ti_get_conflict(self->brain->internal_brain);
    return PyFloat_FromDouble((double)conflict);
}

/**
 * WHAT: Get basal ganglia operating mode
 * WHY:  Distinguish between exploration, exploitation, habit modes
 * HOW:  Calls brain_ti_get_mode, returns int enum value
 */
static PyObject* Brain_bg_get_mode(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int mode = brain_ti_get_mode(self->brain->internal_brain);
    return PyLong_FromLong(mode);
}

/**
 * WHAT: Get basal ganglia dopamine level
 * WHY:  Dopamine modulates learning rate and motivation
 * HOW:  Calls brain_ti_get_dopamine, returns float
 */
static PyObject* Brain_bg_get_dopamine(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float dopamine = brain_ti_get_dopamine(self->brain->internal_brain);
    return PyFloat_FromDouble((double)dopamine);
}

/**
 * WHAT: Get reward prediction error from basal ganglia
 * WHY:  RPE drives learning — positive RPE strengthens, negative weakens
 * HOW:  Calls brain_ti_get_rpe, returns float
 */
static PyObject* Brain_bg_get_rpe(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float rpe = brain_ti_get_rpe(self->brain->internal_brain);
    return PyFloat_FromDouble((double)rpe);
}

/**
 * WHAT: Register a new habit in the basal ganglia
 * WHY:  Habit formation automates frequently-used action sequences
 * HOW:  Calls brain_ti_register_habit with domain string and action ID
 */
static PyObject* Brain_bg_register_habit(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* domain;
    int action_id;
    if (!PyArg_ParseTuple(args, "si", &domain, &action_id)) return NULL;

    int result = brain_ti_register_habit(self->brain->internal_brain, domain, action_id);
    return PyLong_FromLong(result);
}

/**
 * WHAT: Check if a habit exists for the given domain
 * WHY:  Allows training loop to skip exploration when habits are formed
 * HOW:  Calls brain_ti_check_habit with domain string, returns habit ID or -1
 */
static PyObject* Brain_bg_check_habit(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* domain;
    if (!PyArg_ParseTuple(args, "s", &domain)) return NULL;

    int result = brain_ti_check_habit(self->brain->internal_brain, domain);
    return PyLong_FromLong(result);
}

/**
 * WHAT: Strengthen an existing habit after successful use
 * WHY:  Repeated success solidifies habits, reducing cognitive load
 * HOW:  Calls brain_ti_strengthen_habit with habit ID and success flag
 */
static PyObject* Brain_bg_strengthen_habit(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int habit_id;
    int success;
    if (!PyArg_ParseTuple(args, "ip", &habit_id, &success)) return NULL;

    int result = brain_ti_strengthen_habit(self->brain->internal_brain, habit_id, success);
    return PyLong_FromLong(result);
}

/**
 * WHAT: Get medulla arousal level
 * WHY:  Arousal modulates attention and processing intensity
 * HOW:  Calls brain_ti_get_arousal, returns float [0.0, 1.0]
 */
static PyObject* Brain_medulla_get_arousal(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float arousal = brain_ti_get_arousal(self->brain->internal_brain);
    return PyFloat_FromDouble((double)arousal);
}

/**
 * WHAT: Get medulla circadian phase
 * WHY:  Circadian phase affects consolidation timing and learning efficiency
 * HOW:  Calls brain_ti_get_circadian_phase, returns int phase enum
 */
static PyObject* Brain_medulla_get_circadian_phase(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int phase = brain_ti_get_circadian_phase(self->brain->internal_brain);
    return PyLong_FromLong(phase);
}

/**
 * WHAT: Boost medulla arousal by a delta
 * WHY:  Training may need to artificially increase arousal for difficult examples
 * HOW:  Calls brain_ti_boost_arousal with delta float
 */
static PyObject* Brain_medulla_boost_arousal(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float delta;
    if (!PyArg_ParseTuple(args, "f", &delta)) return NULL;

    brain_ti_boost_arousal(self->brain->internal_brain, delta);
    Py_RETURN_NONE;
}

/**
 * WHAT: Get circadian efficiency factor from medulla
 * WHY:  Allows training to scale learning rate by time-of-day efficiency
 * HOW:  Calls brain_ti_get_circadian_efficiency, returns float [0.0, 1.0]
 */
static PyObject* Brain_medulla_get_circadian_efficiency(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float efficiency = brain_ti_get_circadian_efficiency(self->brain->internal_brain);
    return PyFloat_FromDouble((double)efficiency);
}

/**
 * WHAT: Add a fact to the reasoning knowledge base
 * WHY:  Training integration can inject domain knowledge as facts
 * HOW:  Calls brain_ti_add_fact with fact string and salience float
 */
static PyObject* Brain_ti_add_fact(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* fact;
    float salience;
    if (!PyArg_ParseTuple(args, "sf", &fact, &salience)) return NULL;

    int result = brain_ti_add_fact(self->brain->internal_brain, fact, salience);
    if (result == 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * WHAT: Add a rule to the reasoning engine
 * WHY:  Rules enable forward/backward chaining over accumulated facts
 * HOW:  Calls brain_ti_add_rule with rule string and priority float
 */
static PyObject* Brain_ti_add_rule(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* rule;
    float priority;
    if (!PyArg_ParseTuple(args, "sf", &rule, &priority)) return NULL;

    int result = brain_ti_add_rule(self->brain->internal_brain, rule, priority);
    if (result == 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * WHAT: Run forward chaining on the reasoning knowledge base
 * WHY:  Derives new facts from existing facts + rules, bounded by max_iterations
 * HOW:  Calls brain_ti_forward_chain, returns number of new facts derived
 */
static PyObject* Brain_ti_forward_chain(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int max_iterations;
    if (!PyArg_ParseTuple(args, "i", &max_iterations)) return NULL;

    int derived = brain_ti_forward_chain(self->brain->internal_brain, max_iterations);
    return PyLong_FromLong(derived);
}

/**
 * WHAT: Run backward chaining to prove a goal
 * WHY:  Goal-directed reasoning checks if a conclusion can be derived
 * HOW:  Calls brain_ti_backward_chain with goal string, returns confidence float
 */
static PyObject* Brain_ti_backward_chain(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* goal;
    if (!PyArg_ParseTuple(args, "s", &goal)) return NULL;

    float confidence = brain_ti_backward_chain(self->brain->internal_brain, goal);
    return PyFloat_FromDouble((double)confidence);
}

/**
 * WHAT: Query knowledge base for matching facts
 * WHY:  Allows training to check what the brain already knows
 * HOW:  Calls brain_ti_query_knowledge with query string, returns match count
 */
static PyObject* Brain_ti_query_knowledge(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* query;
    if (!PyArg_ParseTuple(args, "s", &query)) return NULL;

    int count = brain_ti_query_knowledge(self->brain->internal_brain, query);
    return PyLong_FromLong(count);
}

/**
 * WHAT: Initialize the reasoning subsystem
 * WHY:  Must be called before add_fact/add_rule/forward_chain/backward_chain
 * HOW:  Calls brain_ti_init_reasoning, returns True on success
 */
static PyObject* Brain_ti_init_reasoning(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int result = brain_ti_init_reasoning(self->brain->internal_brain);
    if (result == 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * WHAT: Run a reasoning query and get confidence
 * WHY:  High-level reasoning interface for training feedback loops
 * HOW:  Calls brain_ti_reason with query string, returns confidence float
 */
static PyObject* Brain_ti_reason(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    const char* query;
    if (!PyArg_ParseTuple(args, "s", &query)) return NULL;

    float confidence = brain_ti_reason(self->brain->internal_brain, query);
    return PyFloat_FromDouble((double)confidence);
}

/**
 * WHAT: Compute adaptive learning rate based on brain state
 * WHY:  Basal ganglia dopamine + medulla arousal modulate optimal LR
 * HOW:  Calls brain_ti_compute_adaptive_lr with base_lr, returns adjusted float
 */
static PyObject* Brain_ti_compute_adaptive_lr(BrainObject* self, PyObject* args) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float base_lr;
    if (!PyArg_ParseTuple(args, "f", &base_lr)) return NULL;

    float adaptive_lr = brain_ti_compute_adaptive_lr(self->brain->internal_brain, base_lr);
    return PyFloat_FromDouble((double)adaptive_lr);
}

/**
 * WHAT: Post-batch update — feed accuracy/domain back to training integration
 * WHY:  Closes the loop: training results inform basal ganglia, medulla, reasoning
 * HOW:  Calls brain_ti_post_batch_update with accuracy, expected, domain
 */
static PyObject* Brain_ti_post_batch_update(BrainObject* self, PyObject* args, PyObject* kwds) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    float accuracy, expected;
    const char* domain;

    static char* kwlist[] = {"accuracy", "expected", "domain", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ffs", kwlist,
                                     &accuracy, &expected, &domain)) {
        return NULL;
    }

    int result = brain_ti_post_batch_update(self->brain->internal_brain,
                                            accuracy, expected, domain);
    if (result == 0) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * WHAT: Check if Portia recommends skipping reasoning
 */
static PyObject* Brain_ti_should_skip_reasoning(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    bool skip = brain_ti_should_skip_reasoning();
    return PyBool_FromLong(skip);
}

/**
 * WHAT: Get Portia degradation level affecting reasoning
 */
static PyObject* Brain_ti_get_reasoning_degradation(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int level = brain_ti_get_reasoning_degradation();
    return PyLong_FromLong(level);
}

/**
 * WHAT: Get number of reasoning phases disabled by Portia
 */
static PyObject* Brain_ti_get_reasoning_phases_disabled(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    int disabled = brain_ti_get_reasoning_phases_disabled();
    return PyLong_FromLong(disabled);
}

/**
 * WHAT: Pre-compute and cache community structure for consolidation replay
 * WHY:  Louvain community detection is O(N log N), takes hours on 2M neurons.
 *       Caching allows consolidation to use community-aware prioritization
 *       without re-triggering the expensive algorithm.
 * HOW:  Runs community_detect() + community_detect_hubs() on the brain's
 *       neural network and stores results in the BrainObject's cache.
 *
 * Python usage:
 *   brain.cache_communities()  # Pre-compute (blocking)
 *   brain.consolidate()        # Uses cache automatically
 */
static PyObject* Brain_cache_communities(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    /* Create cache if needed */
    if (!self->community_cache) {
        self->community_cache = consolidation_community_cache_create();
        if (!self->community_cache) {
            PyErr_SetString(PyExc_MemoryError, "Failed to allocate community cache");
            return NULL;
        }
    }

    bool success;
    brain_t internal = self->brain->internal_brain;
    consolidation_community_cache_t* cache = self->community_cache;

    Py_BEGIN_ALLOW_THREADS
    success = consolidation_cache_communities(cache, internal);
    Py_END_ALLOW_THREADS

    if (!success) {
        PyErr_SetString(PyExc_RuntimeError, "Community detection failed");
        return NULL;
    }

    /* Return dict with cache info */
    PyObject* result = PyDict_New();
    if (!result) return NULL;

    PyDict_SetItemString(result, "num_communities",
                         PyLong_FromUnsignedLong(self->community_cache->num_communities));
    PyDict_SetItemString(result, "num_hubs",
                         PyLong_FromUnsignedLong(self->community_cache->num_hubs));
    PyDict_SetItemString(result, "modularity",
                         PyFloat_FromDouble(self->community_cache->modularity));
    PyDict_SetItemString(result, "num_neurons",
                         PyLong_FromUnsignedLong(self->community_cache->num_neurons));

    return result;
}

/**
 * WHAT: Invalidate the cached community structure
 * WHY:  Force re-computation on next cache_communities() call
 * HOW:  Mark cache as stale without freeing memory
 *
 * Python usage:
 *   brain.invalidate_community_cache()
 */
static PyObject* Brain_invalidate_community_cache(BrainObject* self, PyObject* Py_UNUSED(args)) {
    if (self->community_cache) {
        consolidation_community_cache_invalidate(self->community_cache);
    }
    Py_RETURN_NONE;
}

/**
 * WHAT: Get epistemic + aleatoric uncertainty for input features
 * WHY:  Metacognitive monitoring — "how confident is the brain?"
 * HOW:  Call brain_get_uncertainty() through introspection accessor
 */
static PyObject* Brain_get_uncertainty(BrainObject* self, PyObject* args) {
    PyObject* features_list = NULL;
    /* Features are optional — if not provided, return overall uncertainty */
    if (!PyArg_ParseTuple(args, "|O", &features_list))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    introspection_context_t introspection = brain_get_introspection(self->brain->internal_brain);
    if (!introspection) {
        /* No introspection — return zeros */
        PyObject* dict = PyDict_New();
        if (!dict) return NULL;
        PyObject* zero = PyFloat_FromDouble(0.0);
        PyDict_SetItemString(dict, "epistemic", zero);
        PyDict_SetItemString(dict, "aleatoric", zero);
        PyDict_SetItemString(dict, "total", zero);
        PyDict_SetItemString(dict, "confidence", zero);
        Py_DECREF(zero);
        return dict;
    }

    float* features = NULL;
    Py_ssize_t num_features = 0;
    if (features_list && features_list != Py_None) {
        features = py_list_to_float_array(features_list, &num_features);
        if (!features) return NULL;
    }

    brain_uncertainty_t unc;
    Py_BEGIN_ALLOW_THREADS
    if (features) {
        unc = brain_get_uncertainty(introspection, features, (uint32_t)num_features);
    } else {
        /* No features — use a zero vector for general uncertainty */
        float dummy[1] = {0.0f};
        unc = brain_get_uncertainty(introspection, dummy, 1);
    }
    Py_END_ALLOW_THREADS

    if (features) nimcp_free(features);

    PyObject* dict = PyDict_New();
    if (!dict) { brain_uncertainty_free(&unc); return NULL; }

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); brain_uncertainty_free(&unc); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    SET("epistemic", PyFloat_FromDouble(unc.epistemic));
    SET("aleatoric", PyFloat_FromDouble(unc.aleatoric));
    SET("total", PyFloat_FromDouble(unc.total));
    SET("confidence", PyFloat_FromDouble(unc.confidence));
    SET("ensemble_size", PyLong_FromUnsignedLong(unc.ensemble_size));

#undef SET

    brain_uncertainty_free(&unc);
    return dict;
}

/**
 * WHAT: Self-model assessment for a domain
 * WHY:  Track brain's perceived capabilities per domain
 * HOW:  Call self_model_get() through brain internal access
 */
static PyObject* Brain_self_assess(BrainObject* self, PyObject* args) {
    const char* domain;
    if (!PyArg_ParseTuple(args, "s", &domain))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    /* Return probe-based assessment since self_model doesn't have a
       per-domain getter — use probe metrics as a proxy */
    nimcp_brain_probe_t probe;
    nimcp_status_t status;
    Py_BEGIN_ALLOW_THREADS
    status = nimcp_brain_probe(self->brain, &probe);
    Py_END_ALLOW_THREADS

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    SET("domain", PyUnicode_FromString(domain));
    if (status == NIMCP_OK) {
        SET("accuracy", PyFloat_FromDouble(probe.accuracy));
        SET("learning_steps", PyLong_FromUnsignedLongLong(probe.total_learning_steps));
        SET("sparsity", PyFloat_FromDouble(probe.avg_sparsity));
        SET("learning_rate", PyFloat_FromDouble(probe.current_learning_rate));
        SET("num_neurons", PyLong_FromUnsignedLong(probe.num_neurons));
        SET("assessment", PyUnicode_FromString("active"));
    } else {
        SET("assessment", PyUnicode_FromString("unavailable"));
    }

#undef SET
    return dict;
}

/**
 * WHAT: Process audio through the brain's audio cortex
 * WHY:  Extract MFCC + mel features from raw audio for multimodal training
 * HOW:  Call audio_cortex_process() on the brain's audio cortex (GIL released)
 *
 * @param samples Python list of float audio samples
 * @return Python list of float features, or empty list if cortex unavailable
 */
static PyObject* Brain_audio_cortex_process(BrainObject* self, PyObject* args) {
    PyObject* samples_list;
    if (!PyArg_ParseTuple(args, "O", &samples_list))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    Py_ssize_t num_samples;
    float* samples = py_list_to_float_array(samples_list, &num_samples);
    if (!samples) return NULL;

    /* Access audio cortex from internal brain */
    brain_t ib = self->brain->internal_brain;
    if (!ib || !ib->audio_cortex) {
        nimcp_free(samples);
        /* Return empty list — cortex not initialized */
        return PyList_New(0);
    }

    /* Query feature dimension */
    uint32_t feat_dim = audio_cortex_get_feature_dim(ib->audio_cortex);
    if (feat_dim == 0) feat_dim = 128;

    float* features = (float*)nimcp_calloc(feat_dim, sizeof(float));
    if (!features) {
        nimcp_free(samples);
        return PyErr_NoMemory();
    }

    bool success;
    Py_BEGIN_ALLOW_THREADS
    success = audio_cortex_process(ib->audio_cortex, samples,
                                    (uint32_t)num_samples, 1, features);
    Py_END_ALLOW_THREADS

    nimcp_free(samples);

    if (!success) {
        nimcp_free(features);
        return PyList_New(0);
    }

    /* Build Python list */
    PyObject* result = PyList_New(feat_dim);
    if (!result) { nimcp_free(features); return NULL; }
    for (uint32_t i = 0; i < feat_dim; i++) {
        PyObject* val = PyFloat_FromDouble((double)features[i]);
        if (!val) { Py_DECREF(result); nimcp_free(features); return NULL; }
        PyList_SET_ITEM(result, i, val);
    }
    nimcp_free(features);
    return result;
}

/**
 * WHAT: Process image through the brain's visual cortex
 * WHY:  Extract V1 Gabor filter features from raw pixels for visual training
 * HOW:  Call visual_cortex_process() on the brain's visual cortex (GIL released)
 *
 * @param pixels Python list of float pixel values [0-1] or [0-255]
 * @param width Image width
 * @param height Image height
 * @param channels Number of channels (1=grayscale, 3=RGB)
 * @return Python list of float features, or empty list if cortex unavailable
 */
static PyObject* Brain_visual_cortex_process(BrainObject* self, PyObject* args) {
    PyObject* pixels_list;
    unsigned int width, height, channels;
    if (!PyArg_ParseTuple(args, "OIII", &pixels_list, &width, &height, &channels))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    Py_ssize_t num_pixels;
    float* pixels_float = py_list_to_float_array(pixels_list, &num_pixels);
    if (!pixels_float) return NULL;

    /* Visual cortex expects uint8_t* image — convert float [0,1] to [0,255] */
    uint8_t* pixels = (uint8_t*)nimcp_malloc((size_t)num_pixels);
    if (!pixels) {
        nimcp_free(pixels_float);
        return PyErr_NoMemory();
    }
    for (Py_ssize_t i = 0; i < num_pixels; i++) {
        float v = pixels_float[i];
        if (v <= 1.0f && v >= 0.0f) v *= 255.0f;
        pixels[i] = (uint8_t)(v > 255.0f ? 255 : (v < 0.0f ? 0 : v));
    }
    nimcp_free(pixels_float);

    /* Access visual cortex from internal brain */
    brain_t ib = self->brain->internal_brain;
    if (!ib || !ib->visual_cortex) {
        nimcp_free(pixels);
        return PyList_New(0);
    }

    uint32_t feat_dim = visual_cortex_get_feature_dim(ib->visual_cortex);
    if (feat_dim == 0) feat_dim = 128;

    float* features = (float*)nimcp_calloc(feat_dim, sizeof(float));
    if (!features) {
        nimcp_free(pixels);
        return PyErr_NoMemory();
    }

    bool success;
    Py_BEGIN_ALLOW_THREADS
    success = visual_cortex_process(ib->visual_cortex, pixels,
                                     width, height, channels, features);
    Py_END_ALLOW_THREADS

    nimcp_free(pixels);

    if (!success) {
        nimcp_free(features);
        return PyList_New(0);
    }

    PyObject* result = PyList_New(feat_dim);
    if (!result) { nimcp_free(features); return NULL; }
    for (uint32_t i = 0; i < feat_dim; i++) {
        PyObject* val = PyFloat_FromDouble((double)features[i]);
        if (!val) { Py_DECREF(result); nimcp_free(features); return NULL; }
        PyList_SET_ITEM(result, i, val);
    }
    nimcp_free(features);
    return result;
}

/**
 * WHAT: Process audio through the brain's speech cortex
 * WHY:  Extract phoneme + prosody features from speech for speech training
 * HOW:  Call speech_cortex_process() on the brain's speech cortex (GIL released)
 *
 * @param samples Python list of float audio samples
 * @return Python list of float features, or empty list if cortex unavailable
 */
static PyObject* Brain_speech_cortex_process(BrainObject* self, PyObject* args) {
    PyObject* samples_list;
    if (!PyArg_ParseTuple(args, "O", &samples_list))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    Py_ssize_t num_samples;
    float* samples = py_list_to_float_array(samples_list, &num_samples);
    if (!samples) return NULL;

    brain_t ib = self->brain->internal_brain;
    if (!ib || !ib->speech_cortex) {
        nimcp_free(samples);
        return PyList_New(0);
    }

    /* Speech cortex feature dim from brain config, fallback to 128 */
    uint32_t feat_dim = ib->config.speech_feature_dim;
    if (feat_dim == 0) feat_dim = 128;

    float* features = (float*)nimcp_calloc(feat_dim, sizeof(float));
    if (!features) {
        nimcp_free(samples);
        return PyErr_NoMemory();
    }

    bool success;
    Py_BEGIN_ALLOW_THREADS
    success = speech_cortex_process(ib->speech_cortex, samples,
                                     (uint32_t)num_samples, features);
    Py_END_ALLOW_THREADS

    nimcp_free(samples);

    if (!success) {
        nimcp_free(features);
        return PyList_New(0);
    }

    PyObject* result = PyList_New(feat_dim);
    if (!result) { nimcp_free(features); return NULL; }
    for (uint32_t i = 0; i < feat_dim; i++) {
        PyObject* val = PyFloat_FromDouble((double)features[i]);
        if (!val) { Py_DECREF(result); nimcp_free(features); return NULL; }
        PyList_SET_ITEM(result, i, val);
    }
    nimcp_free(features);
    return result;
}

/**
 * WHAT: LGSS content safety check
 * WHY:  Gate web-fetched and generated content before learning
 * HOW:  Create temporary LGSS filter, run is_safe check, destroy
 */
static PyObject* Brain_lgss_check_content(BrainObject* self, PyObject* args) {
    const char* text;
    if (!PyArg_ParseTuple(args, "s", &text))
        return NULL;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }

    size_t text_len = strlen(text);

    /* Create a temporary content filter with default config */
    lgss_content_filter_config_t config;
    lgss_content_filter_default_config(&config);

    lgss_content_filter_t* filter = lgss_content_filter_create(NULL, &config);
    if (!filter) {
        /* No filter available — assume safe (defense in depth: Python layer
           also checks) */
        PyObject* dict = PyDict_New();
        if (!dict) return NULL;
        PyObject* v = Py_True; Py_INCREF(v);
        PyDict_SetItemString(dict, "is_safe", v); Py_DECREF(v);
        v = PyUnicode_FromString("no lgss filter available");
        PyDict_SetItemString(dict, "reason", v); Py_DECREF(v);
        return dict;
    }

    lgss_content_filter_result_t result;
    memset(&result, 0, sizeof(result));
    nimcp_error_t err;

    Py_BEGIN_ALLOW_THREADS
    err = lgss_content_filter_is_safe(filter, text, text_len, &result);
    Py_END_ALLOW_THREADS

    lgss_content_filter_destroy(filter);

    PyObject* dict = PyDict_New();
    if (!dict) return NULL;

#define SET(key, val) do { \
    PyObject* v = (val); \
    if (!v) { Py_DECREF(dict); return NULL; } \
    PyDict_SetItemString(dict, (key), v); \
    Py_DECREF(v); \
} while (0)

    bool is_safe = (err == NIMCP_OK && result.status == LGSS_CONTENT_SAFE);
    SET("is_safe", PyBool_FromLong(is_safe));
    SET("status", PyLong_FromLong(result.status));
    SET("confidence", PyFloat_FromDouble(result.confidence));
    SET("explanation", PyUnicode_FromString(result.explanation));
    if (result.pattern_matched) {
        SET("matched_pattern", PyUnicode_FromString(result.matched_pattern));
    }
    SET("reason", PyUnicode_FromString(
        is_safe ? "content is safe" : "content flagged by lgss"));

#undef SET
    return dict;
}

/**
 * @brief Enable multi-network ensemble training (LNN + CNN + Adaptive)
 *
 * WHAT: Python binding for brain_enable_multi_network_training()
 * WHY:  Allow Python scripts to enable ensemble training from all architectures
 * HOW:  Calls C API, raises RuntimeError on failure
 */
static PyObject* Brain_enable_multi_network(BrainObject* self, PyObject* args) {
    (void)args;
    if (!self->brain || !self->brain->internal_brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    // self->brain is nimcp_brain_t (handle wrapper)
    // brain_enable_multi_network_training expects brain_t (internal)
    int rc = brain_enable_multi_network_training(self->brain->internal_brain);
    if (rc < 0) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to enable multi-network training");
        return NULL;
    }
    Py_RETURN_NONE;
}

/**
 * @brief Check if brain is frozen
 */
static PyObject* Brain_get_is_frozen(BrainObject* self, void* closure) {
    (void)closure;
    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, "Brain not initialized");
        return NULL;
    }
    if (nimcp_brain_is_frozen(self->brain)) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

static PyMethodDef Brain_methods[] = {
    {"learn", (PyCFunction)Brain_learn, METH_VARARGS,
     "Learn from example: learn(features, label, confidence=1.0) -> loss"},
    {"predict", (PyCFunction)Brain_predict, METH_VARARGS,
     "Make prediction: predict(features) -> (label, confidence)"},
    {"predict_batch", (PyCFunction)Brain_predict_batch, METH_VARARGS,
     "Batch prediction: predict_batch(features_list) -> (labels, confidences)"},
    {"predict_fast", (PyCFunction)Brain_predict_fast, METH_VARARGS,
     "Fast prediction — forward pass only, no cognitive stages: predict_fast(features) -> (label, confidence)"},
    {"save", (PyCFunction)Brain_save, METH_VARARGS,
     "Save to file: save(filepath)"},
    {"load", (PyCFunction)Brain_load, METH_VARARGS | METH_CLASS,
     "Load from file: Brain.load(filepath) -> Brain"},

    // Phase 2.8: Dynamic Brain Resizing
    {"resize", (PyCFunction)Brain_resize, METH_VARARGS,
     "Resize brain to new neuron count: resize(new_neuron_count) -> True"},
    {"auto_resize", (PyCFunction)Brain_auto_resize, METH_NOARGS,
     "Auto-resize based on utilization: auto_resize() -> True if resized, False otherwise"},
    {"get_neuron_count", (PyCFunction)Brain_get_neuron_count, METH_NOARGS,
     "Get current neuron count: get_neuron_count() -> int"},
    {"get_utilization_metrics", (PyCFunction)Brain_get_utilization_metrics, METH_NOARGS,
     "Get utilization metrics: get_utilization_metrics() -> (utilization, saturation)"},

    // COW Snapshot API
    {"snapshot_cow", (PyCFunction)Brain_snapshot_cow, METH_NOARGS,
     "Create instant COW snapshot: snapshot_cow() -> capsule"},
    {"restore_cow", (PyCFunction)Brain_restore_cow, METH_VARARGS,
     "Restore from COW snapshot: restore_cow(snapshot) -> bool"},
    {"destroy_cow_snapshot", (PyCFunction)Brain_destroy_cow_snapshot, METH_VARARGS,
     "Destroy COW snapshot: destroy_cow_snapshot(snapshot)"},

    // Multi-network ensemble training
    {"enable_multi_network", (PyCFunction)Brain_enable_multi_network, METH_NOARGS,
     "Enable LNN + CNN ensemble training alongside adaptive SNN"},

    // Training metrics
    {"get_accuracy", (PyCFunction)Brain_get_accuracy, METH_NOARGS,
     "Get running label-match accuracy (EMA): get_accuracy() -> float"},

    // Probe
    {"probe", (PyCFunction)Brain_probe, METH_NOARGS,
     "Get brain metrics as dict: probe() -> dict"},
    {"broadcast_probe", (PyCFunction)Brain_broadcast_probe, METH_NOARGS,
     "Probe brain metrics and broadcast via bio-async: broadcast_probe() -> bool"},

    // Rubric (cognitive output quality evaluation)
    {"rubric", (PyCFunction)Brain_rubric, METH_NOARGS,
     "Evaluate quality of last decision: rubric() -> dict with 15 scores + grade"},
    {"set_rubric_validation", (PyCFunction)Brain_set_rubric_validation, METH_VARARGS,
     "Set validation features for rubric during training: set_rubric_validation(features) -> True"},
    {"training_rubric", (PyCFunction)Brain_training_rubric, METH_NOARGS,
     "Get rubric stats from training: training_rubric() -> dict or None"},

    // Full cognitive decision
    {"decide_full", (PyCFunction)Brain_decide_full, METH_VARARGS,
     "Run full cognitive pipeline: decide_full(features) -> dict"},

    // Frozen inference
    {"freeze", (PyCFunction)Brain_freeze, METH_NOARGS,
     "Freeze brain for inference-only mode: freeze() -> True"},

    // Socratic active learning: cognitive module bindings
    {"curiosity_detect_gaps", (PyCFunction)Brain_curiosity_detect_gaps, METH_VARARGS,
     "Detect knowledge gaps: curiosity_detect_gaps(topic) -> dict"},
    {"consolidate", (PyCFunction)Brain_consolidate, METH_VARARGS | METH_KEYWORDS,
     "Trigger memory consolidation: consolidate(mode='auto', cycles=0, prune_passes=0) -> bool"},
    {"cache_communities", (PyCFunction)Brain_cache_communities, METH_NOARGS,
     "Pre-compute community structure for consolidation: cache_communities() -> dict"},
    {"invalidate_community_cache", (PyCFunction)Brain_invalidate_community_cache, METH_NOARGS,
     "Invalidate cached community structure: invalidate_community_cache() -> None"},
    {"get_uncertainty", (PyCFunction)Brain_get_uncertainty, METH_VARARGS,
     "Get uncertainty: get_uncertainty([features]) -> dict"},
    {"self_assess", (PyCFunction)Brain_self_assess, METH_VARARGS,
     "Self-model assessment: self_assess(domain) -> dict"},
    {"lgss_check_content", (PyCFunction)Brain_lgss_check_content, METH_VARARGS,
     "LGSS content filter: lgss_check_content(text) -> dict"},

    // Sensory cortex bindings for multimodal training
    {"audio_cortex_process", (PyCFunction)Brain_audio_cortex_process, METH_VARARGS,
     "Process audio through audio cortex: audio_cortex_process(samples) -> list of features"},
    {"visual_cortex_process", (PyCFunction)Brain_visual_cortex_process, METH_VARARGS,
     "Process image through visual cortex: visual_cortex_process(pixels, width, height, channels) -> list of features"},
    {"speech_cortex_process", (PyCFunction)Brain_speech_cortex_process, METH_VARARGS,
     "Process audio through speech cortex: speech_cortex_process(samples) -> list of features"},

    // Training Integration API: basal ganglia
    {"bg_update_reward", (PyCFunction)Brain_bg_update_reward, METH_VARARGS,
     "Update basal ganglia reward: bg_update_reward(reward, expected) -> int"},
    {"bg_get_conflict", (PyCFunction)Brain_bg_get_conflict, METH_NOARGS,
     "Get basal ganglia conflict level: bg_get_conflict() -> float"},
    {"bg_get_mode", (PyCFunction)Brain_bg_get_mode, METH_NOARGS,
     "Get basal ganglia operating mode: bg_get_mode() -> int"},
    {"bg_get_dopamine", (PyCFunction)Brain_bg_get_dopamine, METH_NOARGS,
     "Get basal ganglia dopamine level: bg_get_dopamine() -> float"},
    {"bg_get_rpe", (PyCFunction)Brain_bg_get_rpe, METH_NOARGS,
     "Get reward prediction error: bg_get_rpe() -> float"},
    {"bg_register_habit", (PyCFunction)Brain_bg_register_habit, METH_VARARGS,
     "Register a habit: bg_register_habit(domain, action_id) -> int"},
    {"bg_check_habit", (PyCFunction)Brain_bg_check_habit, METH_VARARGS,
     "Check if habit exists: bg_check_habit(domain) -> int"},
    {"bg_strengthen_habit", (PyCFunction)Brain_bg_strengthen_habit, METH_VARARGS,
     "Strengthen a habit: bg_strengthen_habit(habit_id, success) -> int"},

    // Training Integration API: medulla
    {"medulla_get_arousal", (PyCFunction)Brain_medulla_get_arousal, METH_NOARGS,
     "Get medulla arousal level: medulla_get_arousal() -> float"},
    {"medulla_get_circadian_phase", (PyCFunction)Brain_medulla_get_circadian_phase, METH_NOARGS,
     "Get circadian phase: medulla_get_circadian_phase() -> int"},
    {"medulla_boost_arousal", (PyCFunction)Brain_medulla_boost_arousal, METH_VARARGS,
     "Boost arousal: medulla_boost_arousal(delta) -> None"},
    {"medulla_get_circadian_efficiency", (PyCFunction)Brain_medulla_get_circadian_efficiency, METH_NOARGS,
     "Get circadian efficiency: medulla_get_circadian_efficiency() -> float"},

    // Training Integration API: reasoning
    {"ti_add_fact", (PyCFunction)Brain_ti_add_fact, METH_VARARGS,
     "Add fact to knowledge base: ti_add_fact(fact, salience) -> bool"},
    {"ti_add_rule", (PyCFunction)Brain_ti_add_rule, METH_VARARGS,
     "Add rule to reasoning engine: ti_add_rule(rule, priority) -> bool"},
    {"ti_forward_chain", (PyCFunction)Brain_ti_forward_chain, METH_VARARGS,
     "Run forward chaining: ti_forward_chain(max_iterations) -> int"},
    {"ti_backward_chain", (PyCFunction)Brain_ti_backward_chain, METH_VARARGS,
     "Run backward chaining: ti_backward_chain(goal) -> float"},
    {"ti_query_knowledge", (PyCFunction)Brain_ti_query_knowledge, METH_VARARGS,
     "Query knowledge base: ti_query_knowledge(query) -> int"},
    {"ti_init_reasoning", (PyCFunction)Brain_ti_init_reasoning, METH_NOARGS,
     "Initialize reasoning subsystem: ti_init_reasoning() -> bool"},
    {"ti_reason", (PyCFunction)Brain_ti_reason, METH_VARARGS,
     "Run reasoning query: ti_reason(query) -> float"},

    // Training Integration API: adaptive learning rate + post-batch
    {"ti_compute_adaptive_lr", (PyCFunction)Brain_ti_compute_adaptive_lr, METH_VARARGS,
     "Compute adaptive learning rate: ti_compute_adaptive_lr(base_lr) -> float"},
    {"ti_post_batch_update", (PyCFunction)Brain_ti_post_batch_update, METH_VARARGS | METH_KEYWORDS,
     "Post-batch update: ti_post_batch_update(accuracy, expected, domain) -> bool"},
    {"ti_should_skip_reasoning", (PyCFunction)Brain_ti_should_skip_reasoning, METH_NOARGS,
     "Check if Portia recommends skipping reasoning due to resource pressure"},
    {"ti_get_reasoning_degradation", (PyCFunction)Brain_ti_get_reasoning_degradation, METH_NOARGS,
     "Get Portia degradation level affecting reasoning (0-4)"},
    {"ti_get_reasoning_phases_disabled", (PyCFunction)Brain_ti_get_reasoning_phases_disabled, METH_NOARGS,
     "Get number of reasoning phases currently disabled by Portia"},

    {NULL}
};

static PyGetSetDef Brain_getsetters[] = {
    {"is_frozen", (getter)Brain_get_is_frozen, NULL,
     "True if brain is frozen for inference-only mode", NULL},
    {NULL}
};

static PyTypeObject BrainType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.Brain",
    .tp_doc = "NIMCP Brain - High-level cognitive learning system",
    .tp_basicsize = sizeof(BrainObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Brain_new,
    .tp_init = (initproc)Brain_init,
    .tp_dealloc = (destructor)Brain_dealloc,
    .tp_methods = Brain_methods,
    .tp_getset = Brain_getsetters,
};

//=============================================================================
// NeuralNetwork Object Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    nimcp_network_t network;
} NetworkObject;

static void Network_dealloc(NetworkObject* self) {
    if (self->network && !g_py_shutdown_called) {
        nimcp_network_destroy(self->network);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Network_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    NetworkObject* self = (NetworkObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->network = NULL;
    }
    return (PyObject*)self;
}

/**
 * WHAT: Initialize NeuralNetwork with architecture params
 * WHY:  Create low-level neural network for custom topologies
 * HOW:  Call nimcp_network_create
 */
static int Network_init(NetworkObject* self, PyObject* args, PyObject* kwds) {
    unsigned int num_inputs, num_outputs, num_hidden;
    float learning_rate = 0.01F;

    static char* kwlist[] = {"num_inputs", "num_outputs", "num_hidden", "learning_rate", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "III|f", kwlist,
                                     &num_inputs, &num_outputs, &num_hidden, &learning_rate)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Network_init: Invalid arguments");
        return -1;
    }

    self->network = nimcp_network_create(num_inputs, num_outputs, num_hidden, learning_rate);

    if (!self->network) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Network_init: network is NULL");
        return -1;
    }

    return 0;
}

/**
 * WHAT: Forward pass through network
 * WHY:  Inference / activation propagation
 * HOW:  Convert Python list to C array, call nimcp_network_forward
 */
static PyObject* Network_forward(NetworkObject* self, PyObject* args) {
    PyObject* input_list;
    unsigned int num_outputs;

    if (!PyArg_ParseTuple(args, "OI", &input_list, &num_outputs)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Network_forward: Invalid arguments");
        return NULL;
    }

    Py_ssize_t num_inputs;
    float* inputs = py_list_to_float_array(input_list, &num_inputs);
    if (!inputs) return NULL;

    float* outputs = (float*)nimcp_malloc(num_outputs * sizeof(float));
    if (!outputs) {
        nimcp_free(inputs);
        return PyErr_NoMemory();
    }

    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_network_forward(self->network, inputs, (uint32_t)num_inputs,
                                    outputs, num_outputs);
    Py_END_ALLOW_THREADS

    nimcp_free(inputs);

    if (status != NIMCP_OK) {
        nimcp_free(outputs);
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_OPERATION_FAILED, "Network_forward: forward failed");
        return NULL;
    }

    PyObject* result = PyList_New(num_outputs);
    for (unsigned int i = 0; i < num_outputs; i++) {
        PyList_SET_ITEM(result, i, PyFloat_FromDouble((double)outputs[i]));
    }
    nimcp_free(outputs);
    return result;
}

/**
 * WHAT: Train network on a single example
 * WHY:  Supervised learning with input/target pairs
 * HOW:  Call nimcp_network_train
 *
 * NOTE: Currently a stub in the C API - will be implemented in a future version
 */
static PyObject* Network_train(NetworkObject* self, PyObject* args) {
    PyObject* input_list;
    PyObject* target_list;

    if (!PyArg_ParseTuple(args, "OO", &input_list, &target_list)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Network_train: Invalid arguments");
        return NULL;
    }

    Py_ssize_t num_inputs;
    float* inputs = py_list_to_float_array(input_list, &num_inputs);
    if (!inputs) return NULL;

    Py_ssize_t num_targets;
    float* targets = py_list_to_float_array(target_list, &num_targets);
    if (!targets) {
        nimcp_free(inputs);
        return NULL;
    }

    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_network_train(self->network, inputs, (uint32_t)num_inputs,
                                  targets, (uint32_t)num_targets);
    Py_END_ALLOW_THREADS

    nimcp_free(inputs);
    nimcp_free(targets);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError,
            "Network-level training not yet implemented in the C API. "
            "Use Brain.learn() for supervised learning.");
        return NULL;
    }

    Py_RETURN_TRUE;
}

static PyMethodDef Network_methods[] = {
    {"forward", (PyCFunction)Network_forward, METH_VARARGS,
     "Forward pass: forward(inputs, num_outputs) -> list[float]"},
    {"train", (PyCFunction)Network_train, METH_VARARGS,
     "Train on single example: train(inputs, targets) -> bool\n"
     "Note: Currently a stub - use Brain.learn() for training"},
    {NULL}
};

static PyTypeObject NetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.NeuralNetwork",
    .tp_doc = "NIMCP Neural Network - Low-level network operations",
    .tp_basicsize = sizeof(NetworkObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Network_new,
    .tp_init = (initproc)Network_init,
    .tp_dealloc = (destructor)Network_dealloc,
    .tp_methods = Network_methods,
};

//=============================================================================
// KnowledgeSystem Object Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    nimcp_knowledge_t knowledge;
} KnowledgeObject;

static void Knowledge_dealloc(KnowledgeObject* self) {
    if (self->knowledge && !g_py_shutdown_called) {
        nimcp_knowledge_destroy(self->knowledge);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Knowledge_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    KnowledgeObject* self = (KnowledgeObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->knowledge = NULL;
    }
    return (PyObject*)self;
}

static int Knowledge_init(KnowledgeObject* self, PyObject* args, PyObject* kwds) {
    self->knowledge = nimcp_knowledge_create();

    if (!self->knowledge) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "Knowledge_init: knowledge is NULL");
        return -1;
    }

    return 0;
}

/**
 * WHAT: Add a fact to the knowledge graph
 * WHY:  Build knowledge base with subject-predicate-object triples
 * HOW:  Call nimcp_knowledge_add_fact
 */
static PyObject* Knowledge_add_fact(KnowledgeObject* self, PyObject* args) {
    const char* subject;
    const char* predicate;
    const char* object;

    if (!PyArg_ParseTuple(args, "sss", &subject, &predicate, &object)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Knowledge_add_fact: Invalid arguments");
        return NULL;
    }

    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_knowledge_add_fact(self->knowledge, subject, predicate, object);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    Py_RETURN_TRUE;
}

/**
 * WHAT: Query the knowledge graph
 * WHY:  Retrieve information about concepts
 * HOW:  Call nimcp_knowledge_query with pre-allocated buffer
 */
static PyObject* Knowledge_query(KnowledgeObject* self, PyObject* args) {
    const char* query_str;

    if (!PyArg_ParseTuple(args, "s", &query_str)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Knowledge_query: Invalid arguments");
        return NULL;
    }

    char result[NIMCP_CMD_BUFFER_SIZE];
    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_knowledge_query(self->knowledge, query_str,
                                    result, NIMCP_CMD_BUFFER_SIZE);
    Py_END_ALLOW_THREADS

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    return PyUnicode_FromString(result);
}

static PyMethodDef Knowledge_methods[] = {
    {"add_fact", (PyCFunction)Knowledge_add_fact, METH_VARARGS,
     "Add fact: add_fact(subject, predicate, object) -> bool"},
    {"query", (PyCFunction)Knowledge_query, METH_VARARGS,
     "Query knowledge: query(concept) -> str"},
    {NULL}
};

static PyTypeObject KnowledgeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.KnowledgeSystem",
    .tp_doc = "NIMCP Knowledge System - Knowledge graph with fact storage and querying",
    .tp_basicsize = sizeof(KnowledgeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Knowledge_new,
    .tp_init = (initproc)Knowledge_init,
    .tp_dealloc = (destructor)Knowledge_dealloc,
    .tp_methods = Knowledge_methods,
};

//=============================================================================
// Module Functions
//=============================================================================

/**
 * WHAT: Get NIMCP version
 * WHY:  Check library version for compatibility
 * HOW:  Call nimcp_version()
 */
static PyObject* nimcp_get_version(PyObject* self, PyObject* args) {
    return Py_BuildValue("s", nimcp_version());
}

/**
 * WHAT: Initialize NIMCP library
 * WHY:  Required before using any NIMCP functions
 * HOW:  Call nimcp_init()
 */
static PyObject* nimcp_initialize(PyObject* self, PyObject* args) {
    nimcp_status_t status = nimcp_init();
    if (status != NIMCP_OK) {
        NIMCP_THROW_CRITICAL(NIMCP_ERROR_NOT_INITIALIZED,
                            "nimcp_initialize: Failed to initialize NIMCP library");
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize NIMCP");
        return NULL;
    }
    Py_RETURN_NONE;
}

/**
 * WHAT: Shutdown NIMCP library
 * WHY:  Clean up resources before program exit
 * HOW:  Call nimcp_shutdown()
 */
static PyObject* nimcp_shutdown_lib(PyObject* self, PyObject* args) {
    g_py_shutdown_called = 1;
    nimcp_shutdown();
    Py_RETURN_NONE;
}

// Note: SIMD operations not yet exposed in public nimcp.h API
// Will be added in future version

static PyMethodDef module_methods[] = {
    {"version", nimcp_get_version, METH_NOARGS, "Get NIMCP version string"},
    {"init", nimcp_initialize, METH_NOARGS, "Initialize NIMCP library"},
    {"shutdown", nimcp_shutdown_lib, METH_NOARGS, "Shutdown NIMCP library"},
    {NULL}
};

static struct PyModuleDef nimcp_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "nimcp",
    /* P2-5: Use correct version matching library */
    .m_doc = "NIMCP - Neural Interface Message Communication Protocol (Python Bindings v" NIMCP_VERSION_STRING ")",
    .m_size = -1,
    .m_methods = module_methods
};

/**
 * WHAT: Python module initialization
 * WHY:  Entry point for Python interpreter to load module
 * HOW:  Register types, add constants, create module object
 */
PyMODINIT_FUNC PyInit_nimcp(void) {
    PyObject* m;

    LOG_MODULE_INFO("bindings.python", "Initializing Python bindings for NIMCP v" NIMCP_VERSION_STRING);

    // Initialize NIMCP library
    if (nimcp_init() != NIMCP_OK) {
        LOG_MODULE_ERROR("bindings.python", "Failed to initialize NIMCP core library");
        NIMCP_THROW_CRITICAL(NIMCP_ERROR_NOT_INITIALIZED,
                            "PyInit_nimcp: Failed to initialize NIMCP core library");
        PyErr_SetString(PyExc_RuntimeError, "Failed to initialize NIMCP");
        return NULL;
    }

    // Prepare types
    if (PyType_Ready(&BrainType) < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "PyInit_nimcp: BrainType ready failed");
        return NULL;
    }
    if (PyType_Ready(&NetworkType) < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "PyInit_nimcp: NetworkType ready failed");
        return NULL;
    }
    if (PyType_Ready(&KnowledgeType) < 0) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "PyInit_nimcp: KnowledgeType ready failed");
        return NULL;
    }

    // Create module
    m = PyModule_Create(&nimcp_module);
    if (m == NULL) {
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NO_MEMORY, "PyInit_nimcp: validation failed");
        return NULL;
    }

    // Add Brain type
    Py_INCREF(&BrainType);
    if (PyModule_AddObject(m, "Brain", (PyObject*)&BrainType) < 0) {
        Py_DECREF(&BrainType);
        Py_DECREF(m);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "PyInit_nimcp: validation failed");
        return NULL;
    }

    // Add NeuralNetwork type
    Py_INCREF(&NetworkType);
    if (PyModule_AddObject(m, "NeuralNetwork", (PyObject*)&NetworkType) < 0) {
        Py_DECREF(&NetworkType);
        Py_DECREF(m);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "PyInit_nimcp: validation failed");
        return NULL;
    }

    // Add KnowledgeSystem type
    Py_INCREF(&KnowledgeType);
    if (PyModule_AddObject(m, "KnowledgeSystem", (PyObject*)&KnowledgeType) < 0) {
        Py_DECREF(&KnowledgeType);
        Py_DECREF(m);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "PyInit_nimcp: validation failed");
        return NULL;
    }

    // Add constants for brain sizes
    PyModule_AddIntConstant(m, "BRAIN_TINY", NIMCP_BRAIN_TINY);
    PyModule_AddIntConstant(m, "BRAIN_SMALL", NIMCP_BRAIN_SMALL);
    PyModule_AddIntConstant(m, "BRAIN_MEDIUM", NIMCP_BRAIN_MEDIUM);
    PyModule_AddIntConstant(m, "BRAIN_LARGE", NIMCP_BRAIN_LARGE);

    // Add constants for task types
    PyModule_AddIntConstant(m, "TASK_CLASSIFICATION", NIMCP_TASK_CLASSIFICATION);
    PyModule_AddIntConstant(m, "TASK_REGRESSION", NIMCP_TASK_REGRESSION);
    PyModule_AddIntConstant(m, "TASK_PATTERN_MATCHING", NIMCP_TASK_PATTERN_MATCHING);
    PyModule_AddIntConstant(m, "TASK_SEQUENCE", NIMCP_TASK_SEQUENCE);
    PyModule_AddIntConstant(m, "TASK_ASSOCIATION", NIMCP_TASK_ASSOCIATION);

    // Add status constants
    PyModule_AddIntConstant(m, "OK", NIMCP_OK);
    PyModule_AddIntConstant(m, "ERROR", NIMCP_ERROR);

    // Initialize signal filter module
    if (init_signal_filter_module(m) < 0) {
        LOG_MODULE_ERROR("bindings.python", "Failed to initialize signal filter module");
        Py_DECREF(m);
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NOT_INITIALIZED, "PyInit_nimcp: validation failed");
        return NULL;
    }

    LOG_MODULE_INFO("bindings.python", "Successfully initialized Python bindings");
    return m;
}
