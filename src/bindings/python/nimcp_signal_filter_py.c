//=============================================================================
// nimcp_signal_filter_py.c - Python Bindings for Signal Filter Module
//=============================================================================
/**
 * @file nimcp_signal_filter_py.c
 * @brief Python interface for signal filtering and band-pass operations
 *
 * WHAT: Exposes signal filtering functions to Python for PAC detection
 * WHY: Enable Python scripts to perform band-pass filtering for oscillation analysis
 * HOW: Python Type objects with methods wrapping C API
 *
 * @author NIMCP Development Team
 * @date 2025-11-22
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include "../../include/utils/signal/nimcp_signal_filter.h"

//=============================================================================
// Signal Filter Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    signal_filter_t* filter;
} SignalFilterObject;

static void SignalFilter_dealloc(SignalFilterObject* self) {
    if (self->filter) {
        signal_filter_destroy(self->filter);
    }
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* SignalFilter_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    SignalFilterObject* self = (SignalFilterObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->filter = NULL;
    }
    return (PyObject*)self;
}

static int SignalFilter_init(SignalFilterObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"filter_type", "low_freq", "high_freq", "cutoff_freq",
                              "sample_rate", "order", "window", NULL};

    const char* filter_type_str = "bandpass";
    float low_freq = 4.0f;
    float high_freq = 8.0f;
    float cutoff_freq = 0.0f;
    float sample_rate = 1000.0f;
    int order = 64;
    const char* window_str = "hamming";

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sffffis", kwlist,
                                      &filter_type_str, &low_freq, &high_freq,
                                      &cutoff_freq, &sample_rate, &order, &window_str)) {
        return -1;
    }

    // Create configuration
    signal_filter_config_t config;

    // Parse filter type
    if (strcmp(filter_type_str, "bandpass") == 0) {
        config = signal_filter_bandpass_config(low_freq, high_freq, sample_rate);
    } else if (strcmp(filter_type_str, "lowpass") == 0) {
        config = signal_filter_lowpass_config(cutoff_freq, sample_rate);
    } else if (strcmp(filter_type_str, "highpass") == 0) {
        config = signal_filter_highpass_config(cutoff_freq, sample_rate);
    } else {
        config = signal_filter_default_config();
        if (strcmp(filter_type_str, "bandstop") == 0) {
            config.type = FILTER_BANDSTOP;
            config.low_freq = low_freq;
            config.high_freq = high_freq;
        }
    }

    config.order = (uint32_t)order;

    // Parse window type
    if (strcmp(window_str, "hamming") == 0) {
        config.window = WINDOW_HAMMING;
    } else if (strcmp(window_str, "hann") == 0) {
        config.window = WINDOW_HANN;
    } else if (strcmp(window_str, "blackman") == 0) {
        config.window = WINDOW_BLACKMAN;
    } else if (strcmp(window_str, "rectangular") == 0) {
        config.window = WINDOW_RECTANGULAR;
    } else {
        PyErr_SetString(PyExc_ValueError, "Invalid window type");
        return -1;
    }

    // Validate and create filter
    if (!signal_filter_validate_config(&config)) {
        PyErr_SetString(PyExc_ValueError, "Invalid filter configuration");
        return -1;
    }

    self->filter = signal_filter_create(&config);
    if (!self->filter) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create signal filter");
        return -1;
    }

    return 0;
}

//=============================================================================
// SignalFilter Methods
//=============================================================================

static PyObject* SignalFilter_apply(SignalFilterObject* self, PyObject* args) {
    PyArrayObject* input_array;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &input_array)) {
        return NULL;
    }

    // Guard clause
    if (!self->filter) {
        PyErr_SetString(PyExc_RuntimeError, "Filter not initialized");
        return NULL;
    }

    // Validate input array
    if (PyArray_NDIM(input_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Input must be 1D array");
        return NULL;
    }

    if (PyArray_TYPE(input_array) != NPY_FLOAT32 && PyArray_TYPE(input_array) != NPY_FLOAT64) {
        PyErr_SetString(PyExc_ValueError, "Input must be float32 or float64 array");
        return NULL;
    }

    npy_intp n = PyArray_DIM(input_array, 0);

    // Allocate output array
    npy_intp dims[1] = {n};
    PyArrayObject* output_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!output_array) {
        return NULL;
    }

    // Get data pointers
    float* input_data = (float*)PyArray_DATA(input_array);
    float* output_data = (float*)PyArray_DATA(output_array);

    // Apply filter
    if (!signal_filter_apply(self->filter, input_data, output_data, (uint32_t)n)) {
        Py_DECREF(output_array);
        PyErr_SetString(PyExc_RuntimeError, "Filter operation failed");
        return NULL;
    }

    return (PyObject*)output_array;
}

static PyObject* SignalFilter_reset(SignalFilterObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->filter) {
        PyErr_SetString(PyExc_RuntimeError, "Filter not initialized");
        return NULL;
    }

    if (!signal_filter_reset(self->filter)) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to reset filter");
        return NULL;
    }

    Py_RETURN_NONE;
}

static PyObject* SignalFilter_get_delay(SignalFilterObject* self, PyObject* Py_UNUSED(ignored)) {
    if (!self->filter) {
        PyErr_SetString(PyExc_RuntimeError, "Filter not initialized");
        return NULL;
    }

    uint32_t delay = signal_filter_get_delay(self->filter);
    return PyLong_FromUnsignedLong(delay);
}

static PyObject* SignalFilter_get_response(SignalFilterObject* self, PyObject* args) {
    PyArrayObject* freq_array;

    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &freq_array)) {
        return NULL;
    }

    if (!self->filter) {
        PyErr_SetString(PyExc_RuntimeError, "Filter not initialized");
        return NULL;
    }

    // Validate input
    if (PyArray_NDIM(freq_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Frequencies must be 1D array");
        return NULL;
    }

    npy_intp n = PyArray_DIM(freq_array, 0);

    // Allocate output
    npy_intp dims[1] = {n};
    PyArrayObject* response_array = (PyArrayObject*)PyArray_SimpleNew(1, dims, NPY_FLOAT32);
    if (!response_array) {
        return NULL;
    }

    float* freqs = (float*)PyArray_DATA(freq_array);
    float* response = (float*)PyArray_DATA(response_array);

    if (!signal_filter_get_response(self->filter, freqs, response, (uint32_t)n)) {
        Py_DECREF(response_array);
        PyErr_SetString(PyExc_RuntimeError, "Failed to compute frequency response");
        return NULL;
    }

    return (PyObject*)response_array;
}

static PyMethodDef SignalFilter_methods[] = {
    {"apply", (PyCFunction)SignalFilter_apply, METH_VARARGS,
     "Apply filter to input signal and return filtered output"},
    {"reset", (PyCFunction)SignalFilter_reset, METH_NOARGS,
     "Reset filter state (clear circular buffer)"},
    {"get_delay", (PyCFunction)SignalFilter_get_delay, METH_NOARGS,
     "Get group delay in samples (order/2 for linear phase FIR)"},
    {"get_response", (PyCFunction)SignalFilter_get_response, METH_VARARGS,
     "Compute frequency response at specified frequencies"},
    {NULL}
};

static PyTypeObject SignalFilterType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.SignalFilter",
    .tp_doc = "Signal filter for band-pass, lowpass, highpass, and bandstop filtering",
    .tp_basicsize = sizeof(SignalFilterObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = SignalFilter_new,
    .tp_init = (initproc)SignalFilter_init,
    .tp_dealloc = (destructor)SignalFilter_dealloc,
    .tp_methods = SignalFilter_methods,
};

//=============================================================================
// Module Initialization
//=============================================================================

int init_signal_filter_module(PyObject* module) {
    // Import NumPy array API
    import_array();

    // Initialize SignalFilter type
    if (PyType_Ready(&SignalFilterType) < 0) {
        return -1;
    }

    Py_INCREF(&SignalFilterType);
    if (PyModule_AddObject(module, "SignalFilter", (PyObject*)&SignalFilterType) < 0) {
        Py_DECREF(&SignalFilterType);
        return -1;
    }

    // Add filter type constants
    PyModule_AddIntConstant(module, "FILTER_LOWPASS", FILTER_LOWPASS);
    PyModule_AddIntConstant(module, "FILTER_HIGHPASS", FILTER_HIGHPASS);
    PyModule_AddIntConstant(module, "FILTER_BANDPASS", FILTER_BANDPASS);
    PyModule_AddIntConstant(module, "FILTER_BANDSTOP", FILTER_BANDSTOP);

    // Add window type constants
    PyModule_AddIntConstant(module, "WINDOW_RECTANGULAR", WINDOW_RECTANGULAR);
    PyModule_AddIntConstant(module, "WINDOW_HAMMING", WINDOW_HAMMING);
    PyModule_AddIntConstant(module, "WINDOW_HANN", WINDOW_HANN);
    PyModule_AddIntConstant(module, "WINDOW_BLACKMAN", WINDOW_BLACKMAN);

    return 0;
}
