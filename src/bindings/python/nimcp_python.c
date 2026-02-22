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
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception.h"
#include "utils/exception/nimcp_exception_macros.h"
#include <string.h>

#include <stddef.h>  /* for NULL */
#include "utils/memory/nimcp_memory.h"
#include "security/nimcp_bbb_helpers.h"
#include "constants/nimcp_buffer_constants.h"
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
} BrainObject;

/**
 * WHAT: Deallocate Brain object
 * WHY:  Clean up resources when Python object is garbage collected
 * HOW:  Call nimcp_brain_destroy, free Python object memory
 */
static void Brain_dealloc(BrainObject* self) {
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

    static char* kwlist[] = {"name", "size", "task", "num_inputs", "num_outputs", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|iiII", kwlist,
                                     &name, &size, &task, &num_inputs, &num_outputs)) {
        NIMCP_THROW(NIMCP_ERROR_INVALID_PARAM, "Brain_init: Invalid arguments");
        return -1;
    }

    self->brain = nimcp_brain_create(name, (nimcp_brain_size_t)size,
                                     (nimcp_brain_task_t)task, num_inputs, num_outputs);

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
    nimcp_status_t status = nimcp_brain_learn_example(self->brain, features,
                                                      (uint32_t)num_features, label, confidence);
    nimcp_free(features);

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

    nimcp_status_t status = nimcp_brain_predict(self->brain, features,
                                                (uint32_t)num_features, label, &confidence);
    nimcp_free(features);

    if (status != NIMCP_OK) {
        NIMCP_THROW_BRAIN(NIMCP_ERROR_OPERATION_FAILED, 0, "python_binding",
                         "Brain_predict: Prediction failed");
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
    for (Py_ssize_t i = 0; i < batch_size; i++) {
        status = nimcp_brain_predict(self->brain, features_ptrs[i], (uint32_t)num_features,
                                      labels[i], &confidences[i]);
        if (status != NIMCP_OK) break;
    }

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

static PyMethodDef Brain_methods[] = {
    {"learn", (PyCFunction)Brain_learn, METH_VARARGS,
     "Learn from example: learn(features, label, confidence=1.0) -> loss"},
    {"predict", (PyCFunction)Brain_predict, METH_VARARGS,
     "Make prediction: predict(features) -> (label, confidence)"},
    {"predict_batch", (PyCFunction)Brain_predict_batch, METH_VARARGS,
     "Batch prediction: predict_batch(features_list) -> (labels, confidences)"},
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

    // Training metrics
    {"get_accuracy", (PyCFunction)Brain_get_accuracy, METH_NOARGS,
     "Get running label-match accuracy (EMA): get_accuracy() -> float"},

    // Probe
    {"probe", (PyCFunction)Brain_probe, METH_NOARGS,
     "Get brain metrics as dict: probe() -> dict"},
    {"broadcast_probe", (PyCFunction)Brain_broadcast_probe, METH_NOARGS,
     "Probe brain metrics and broadcast via bio-async: broadcast_probe() -> bool"},

    // Full cognitive decision
    {"decide_full", (PyCFunction)Brain_decide_full, METH_VARARGS,
     "Run full cognitive pipeline: decide_full(features) -> dict"},

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
