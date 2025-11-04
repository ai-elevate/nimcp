/**
 * @file nimcp_py.c
 * @brief Python bindings for NIMCP using unified nimcp.h API
 *
 * This is a clean, standalone Python extension that ONLY uses the public
 * nimcp.h API. It does not depend on internal NIMCP headers.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "../../include/nimcp.h"

//=============================================================================
// Brain Object Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    nimcp_brain_t brain;
} BrainObject;

static void Brain_dealloc(BrainObject* self) {
    if (self->brain) {
        nimcp_brain_destroy(self->brain);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Brain_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    BrainObject* self;
    self = (BrainObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->brain = NULL;
    }
    return (PyObject*)self;
}

static int Brain_init(BrainObject* self, PyObject* args, PyObject* kwds) {
    const char* name;
    int size = NIMCP_BRAIN_SMALL;
    int task = NIMCP_TASK_CLASSIFICATION;
    unsigned int num_inputs = 10;
    unsigned int num_outputs = 10;

    static char* kwlist[] = {"name", "size", "task", "num_inputs", "num_outputs", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|iiII", kwlist,
                                     &name, &size, &task, &num_inputs, &num_outputs)) {
        return -1;
    }

    self->brain = nimcp_brain_create(name, (nimcp_brain_size_t)size,
                                     (nimcp_brain_task_t)task, num_inputs, num_outputs);

    if (!self->brain) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return -1;
    }

    return 0;
}

static PyObject* Brain_learn(BrainObject* self, PyObject* args) {
    PyObject* features_list;
    const char* label;
    float confidence = 1.0f;

    if (!PyArg_ParseTuple(args, "Os|f", &features_list, &label, &confidence)) {
        return NULL;
    }

    if (!PyList_Check(features_list)) {
        PyErr_SetString(PyExc_TypeError, "features must be a list");
        return NULL;
    }

    Py_ssize_t num_features = PyList_Size(features_list);
    float* features = (float*)malloc(num_features * sizeof(float));

    for (Py_ssize_t i = 0; i < num_features; i++) {
        PyObject* item = PyList_GetItem(features_list, i);
        features[i] = (float)PyFloat_AsDouble(item);
    }

    nimcp_status_t status = nimcp_brain_learn_example(self->brain, features,
                                                       (uint32_t)num_features, label, confidence);
    free(features);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* Brain_predict(BrainObject* self, PyObject* args) {
    PyObject* features_list;

    if (!PyArg_ParseTuple(args, "O", &features_list)) {
        return NULL;
    }

    if (!PyList_Check(features_list)) {
        PyErr_SetString(PyExc_TypeError, "features must be a list");
        return NULL;
    }

    Py_ssize_t num_features = PyList_Size(features_list);
    float* features = (float*)malloc(num_features * sizeof(float));

    for (Py_ssize_t i = 0; i < num_features; i++) {
        PyObject* item = PyList_GetItem(features_list, i);
        features[i] = (float)PyFloat_AsDouble(item);
    }

    char label[64];
    float confidence;

    nimcp_status_t status = nimcp_brain_predict(self->brain, features,
                                                (uint32_t)num_features, label, &confidence);
    free(features);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    return Py_BuildValue("(sf)", label, confidence);
}

static PyObject* Brain_save(BrainObject* self, PyObject* args) {
    const char* filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }

    nimcp_status_t status = nimcp_brain_save(self->brain, filepath);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_IOError, nimcp_get_error());
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* Brain_load(PyTypeObject* type, PyObject* args) {
    const char* filepath;

    if (!PyArg_ParseTuple(args, "s", &filepath)) {
        return NULL;
    }

    nimcp_brain_t brain = nimcp_brain_load(filepath);

    if (!brain) {
        PyErr_SetString(PyExc_IOError, nimcp_get_error());
        return NULL;
    }

    BrainObject* self = (BrainObject*)type->tp_alloc(type, 0);
    self->brain = brain;

    return (PyObject*)self;
}

static PyMethodDef Brain_methods[] = {
    {"learn", (PyCFunction)Brain_learn, METH_VARARGS, "Learn from example"},
    {"predict", (PyCFunction)Brain_predict, METH_VARARGS, "Make prediction"},
    {"save", (PyCFunction)Brain_save, METH_VARARGS, "Save brain to file"},
    {"load", (PyCFunction)Brain_load, METH_VARARGS | METH_CLASS, "Load brain from file"},
    {NULL}
};

static PyTypeObject BrainType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.Brain",
    .tp_doc = "NIMCP Brain - High-level learning system",
    .tp_basicsize = sizeof(BrainObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Brain_new,
    .tp_init = (initproc)Brain_init,
    .tp_dealloc = (destructor)Brain_dealloc,
    .tp_methods = Brain_methods,
};

//=============================================================================
// Network Object Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    nimcp_network_t network;
} NetworkObject;

static void Network_dealloc(NetworkObject* self) {
    if (self->network) {
        nimcp_network_destroy(self->network);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Network_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    NetworkObject* self;
    self = (NetworkObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->network = NULL;
    }
    return (PyObject*)self;
}

static int Network_init(NetworkObject* self, PyObject* args, PyObject* kwds) {
    unsigned int num_inputs, num_outputs, num_hidden = 100;
    float learning_rate = 0.01f;

    static char* kwlist[] = {"num_inputs", "num_outputs", "num_hidden", "learning_rate", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "II|If", kwlist,
                                     &num_inputs, &num_outputs, &num_hidden, &learning_rate)) {
        return -1;
    }

    self->network = nimcp_network_create(num_inputs, num_outputs, num_hidden, learning_rate);

    if (!self->network) {
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return -1;
    }

    return 0;
}

static PyObject* Network_forward(NetworkObject* self, PyObject* args) {
    PyObject* inputs_list;

    if (!PyArg_ParseTuple(args, "O", &inputs_list)) {
        return NULL;
    }

    if (!PyList_Check(inputs_list)) {
        PyErr_SetString(PyExc_TypeError, "inputs must be a list");
        return NULL;
    }

    Py_ssize_t num_inputs = PyList_Size(inputs_list);
    float* inputs = (float*)malloc(num_inputs * sizeof(float));

    for (Py_ssize_t i = 0; i < num_inputs; i++) {
        PyObject* item = PyList_GetItem(inputs_list, i);
        inputs[i] = (float)PyFloat_AsDouble(item);
    }

    // Assume outputs same size as inputs for now
    float* outputs = (float*)malloc(num_inputs * sizeof(float));

    nimcp_status_t status = nimcp_network_forward(self->network, inputs,
                                                  (uint32_t)num_inputs, outputs, (uint32_t)num_inputs);
    free(inputs);

    if (status != NIMCP_OK) {
        free(outputs);
        PyErr_SetString(PyExc_RuntimeError, nimcp_get_error());
        return NULL;
    }

    PyObject* result = PyList_New(num_inputs);
    for (Py_ssize_t i = 0; i < num_inputs; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(outputs[i]));
    }

    free(outputs);
    return result;
}

static PyMethodDef Network_methods[] = {
    {"forward", (PyCFunction)Network_forward, METH_VARARGS, "Forward pass through network"},
    {NULL}
};

static PyTypeObject NetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.Network",
    .tp_doc = "NIMCP Neural Network - Low-level control",
    .tp_basicsize = sizeof(NetworkObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Network_new,
    .tp_init = (initproc)Network_init,
    .tp_dealloc = (destructor)Network_dealloc,
    .tp_methods = Network_methods,
};

//=============================================================================
// Module Definition
//=============================================================================

static PyObject* nimcp_version(PyObject* self, PyObject* args) {
    return Py_BuildValue("s", nimcp_version());
}

static PyMethodDef module_methods[] = {
    {"version", nimcp_version, METH_NOARGS, "Get NIMCP version"},
    {NULL}
};

static struct PyModuleDef nimcp_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "nimcp",
    .m_doc = "NIMCP - Neural Interface Message Communication Protocol",
    .m_size = -1,
    .m_methods = module_methods
};

PyMODINIT_FUNC PyInit_nimcp(void) {
    PyObject* m;

    // Initialize NIMCP library
    nimcp_init();

    if (PyType_Ready(&BrainType) < 0)
        return NULL;

    if (PyType_Ready(&NetworkType) < 0)
        return NULL;

    m = PyModule_Create(&nimcp_module);
    if (m == NULL)
        return NULL;

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

    // Add types to module
    Py_INCREF(&BrainType);
    if (PyModule_AddObject(m, "Brain", (PyObject*)&BrainType) < 0) {
        Py_DECREF(&BrainType);
        Py_DECREF(m);
        return NULL;
    }

    Py_INCREF(&NetworkType);
    if (PyModule_AddObject(m, "Network", (PyObject*)&NetworkType) < 0) {
        Py_DECREF(&NetworkType);
        Py_DECREF(&BrainType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
