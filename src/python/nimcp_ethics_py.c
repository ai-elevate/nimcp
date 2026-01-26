/**
 * @file nimcp_ethics_py.c
 * @brief Python bindings for NIMCP Ethics Module
 *
 * Provides Python types for:
 * - Ethics: Ethical reasoning and decision evaluation
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>
#include "nimcp.h"
#include "common/nimcp_module.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

#include <stddef.h>  /* for NULL */
//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for ethics_py module */
static nimcp_health_agent_t* g_ethics_py_health_agent = NULL;

/**
 * @brief Set health agent for ethics_py heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void ethics_py_set_health_agent(nimcp_health_agent_t* agent) {
    g_ethics_py_health_agent = agent;
}

/** @brief Send heartbeat from ethics_py module */
static inline void ethics_py_heartbeat(const char* operation, float progress) {
    if (g_ethics_py_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_ethics_py_health_agent, operation, progress);
    }
}


/* ============================================================================
 * Ethics Type
 * ============================================================================ */

typedef struct {
    PyObject_HEAD
    nimcp_ethics_t ethics;
} EthicsObject;

static void Ethics_dealloc(EthicsObject* self)
{
    if (self->ethics != NULL) {
        nimcp_ethics_destroy(self->ethics);
        self->ethics = NULL;
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* Ethics_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    EthicsObject* self = (EthicsObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->ethics = NULL;
    }
    return (PyObject*)self;
}

static int Ethics_init(EthicsObject* self, PyObject* args, PyObject* kwds)
{
    /* Create ethics module */
    self->ethics = nimcp_ethics_create();
    if (self->ethics == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create ethics module");
        return -1;
    }

    return 0;
}

/**
 * @brief Ethics.check(situation) -> float
 *
 * Check if an action/situation is ethically acceptable.
 * Returns ethical score: -1.0 (harmful) to 1.0 (beneficial)
 */
static PyObject* Ethics_check(EthicsObject* self, PyObject* args)
{
    PyObject* situation_list;

    if (!PyArg_ParseTuple(args, "O", &situation_list)) {
        return NULL;
    }

    if (!PyList_Check(situation_list)) {
        PyErr_SetString(PyExc_TypeError, "situation must be a list of floats");
        return NULL;
    }

    Py_ssize_t num_features = PyList_Size(situation_list);
    if (num_features <= 0) {
        PyErr_SetString(PyExc_ValueError, "situation must be non-empty");
        return NULL;
    }

    /* Convert to C array */
    float* situation = (float*)malloc(sizeof(float) * (size_t)num_features);
    if (situation == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t i = 0; i < num_features; i++) {
        PyObject* item = PyList_GetItem(situation_list, i);
        if (!PyFloat_Check(item) && !PyLong_Check(item)) {
            free(situation);
            PyErr_SetString(PyExc_TypeError, "All situation values must be numbers");
            return NULL;
        }
        situation[i] = (float)PyFloat_AsDouble(item);
    }

    float score;
    nimcp_status_t status;

    Py_BEGIN_ALLOW_THREADS
    status = nimcp_ethics_check(self->ethics, situation, (uint32_t)num_features, &score);
    Py_END_ALLOW_THREADS

    free(situation);

    if (status != NIMCP_OK) {
        PyErr_SetString(PyExc_RuntimeError, "Ethics check failed");
        return NULL;
    }

    return PyFloat_FromDouble((double)score);
}

static PyObject* Ethics_repr(EthicsObject* self)
{
    return PyUnicode_FromString("Ethics()");
}

static PyMethodDef Ethics_methods[] = {
    {"check", (PyCFunction)Ethics_check, METH_VARARGS,
     "Check if a situation is ethically acceptable\n\n"
     "Evaluates the ethical implications of a situation or action.\n\n"
     "Args:\n"
     "    situation (list): Feature vector describing the situation\n"
     "Returns:\n"
     "    float: Ethical score from -1.0 (harmful) to 1.0 (beneficial)\n"
     "        - Negative: Potentially harmful action\n"
     "        - Zero: Neutral action\n"
     "        - Positive: Beneficial action\n\n"
     "Example:\n"
     "    ethics = nimcp.Ethics()\n"
     "    score = ethics.check([0.5, 0.3, 0.8, 0.2])\n"
     "    if score > 0:\n"
     "        print('Action appears beneficial')"},
    {NULL}
};

PyTypeObject EthicsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.Ethics",
    .tp_doc = PyDoc_STR(
        "NIMCP Ethics Module for ethical reasoning and decision evaluation.\n\n"
        "The Ethics module provides AI safety mechanisms by evaluating\n"
        "the ethical implications of actions and situations.\n\n"
        "Example:\n"
        "    >>> ethics = nimcp.Ethics()\n"
        "    >>> score = ethics.check([0.5, 0.3, 0.8])\n"
        "    >>> print(f'Ethical score: {score:.2f}')"
    ),
    .tp_basicsize = sizeof(EthicsObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Ethics_new,
    .tp_init = (initproc)Ethics_init,
    .tp_dealloc = (destructor)Ethics_dealloc,
    .tp_repr = (reprfunc)Ethics_repr,
    .tp_methods = Ethics_methods,
};

/* ============================================================================
 * Module initialization
 * ============================================================================ */

int init_ethics_module(PyObject* module)
{
    LOG_MODULE_INFO("bindings.python.ethics", "Initializing ethics module");

    if (PyType_Ready(&EthicsType) < 0)
        return -1;

    Py_INCREF(&EthicsType);
    if (PyModule_AddObject(module, "Ethics", (PyObject*)&EthicsType) < 0) {
        Py_DECREF(&EthicsType);
        return -1;
    }

    LOG_MODULE_INFO("bindings.python.ethics", "Ethics module initialized successfully");
    return 0;
}
