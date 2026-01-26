#include <stddef.h>  /* for NULL */
//=============================================================================
// nimcp_topology_py.c - Python Bindings for Fractal Topology Module
//=============================================================================
/**
 * @file nimcp_topology_py.c
 * @brief Python interface for scale-free network topology generation
 *
 * WHAT: Exposes topology generation functions to Python
 * WHY: Enable Python scripts to create biologically realistic network topologies
 * HOW: Python Type objects with methods wrapping C API
 *
 * @author NIMCP Development Team
 * @date 2025-11-08
 */

#include <Python.h>
#include "security/nimcp_security.h"
#include "security/nimcp_blood_brain_barrier.h"

#include "async/nimcp_bio_async.h"
#include "async/nimcp_bio_router.h"

#include "core/topology/nimcp_fractal_topology.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "common/nimcp_module.h"
#include "utils/logging/nimcp_logging.h"
#include "utils/exception/nimcp_exception_macros.h"

//=============================================================================
// Health Agent Integration (Phase 8: System-Wide Health Integration)
//=============================================================================
struct nimcp_health_agent;
typedef struct nimcp_health_agent nimcp_health_agent_t;
extern void nimcp_health_agent_heartbeat_ex(nimcp_health_agent_t* agent,
                                             const char* operation,
                                             float progress);

/** Global health agent for topology_py module */
static nimcp_health_agent_t* g_topology_py_health_agent = NULL;

/**
 * @brief Set health agent for topology_py heartbeats
 * @param agent Health agent (can be NULL to disable)
 */
static void topology_py_set_health_agent(nimcp_health_agent_t* agent) {
    g_topology_py_health_agent = agent;
}

/** @brief Send heartbeat from topology_py module */
static inline void topology_py_heartbeat(const char* operation, float progress) {
    if (g_topology_py_health_agent) {
        nimcp_health_agent_heartbeat_ex(g_topology_py_health_agent, operation, progress);
    }
}


//=============================================================================
// Topology Configuration Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    topology_config_t config;
} TopologyConfigObject;

static void TopologyConfig_dealloc(TopologyConfigObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TopologyConfig_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    TopologyConfigObject* self = (TopologyConfigObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        // Initialize with default scale-free config
        self->config.type = TOPOLOGY_SCALE_FREE;
        self->config.params.scale_free = topology_default_scale_free_config();
    }
    return (PyObject*)self;
}

static int TopologyConfig_init(TopologyConfigObject* self, PyObject* args, PyObject* kwds) {
    static char* kwlist[] = {"topology_type", "power_law_gamma", "hub_ratio",
                              "min_degree", "max_degree", "spatial_constraint",
                              "bidirectional", NULL};

    const char* topology_type_str = "scale_free";
    float power_law_gamma = -2.1F;
    float hub_ratio = 0.15F;
    int min_degree = 3;
    int max_degree = 50;
    float spatial_constraint = 0.0F;
    int bidirectional = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sffiiifp", kwlist,
                                      &topology_type_str, &power_law_gamma,
                                      &hub_ratio, &min_degree, &max_degree,
                                      &spatial_constraint, &bidirectional)) {
        return -1;
    }

    // Parse topology type
    if (strcmp(topology_type_str, "scale_free") == 0) {
        self->config.type = TOPOLOGY_SCALE_FREE;
        self->config.params.scale_free.power_law_gamma = power_law_gamma;
        self->config.params.scale_free.hub_ratio = hub_ratio;
        self->config.params.scale_free.min_degree = (uint32_t)min_degree;
        self->config.params.scale_free.max_degree = (uint32_t)max_degree;
        self->config.params.scale_free.spatial_constraint = spatial_constraint;
        self->config.params.scale_free.bidirectional = (bool)bidirectional;
    } else if (strcmp(topology_type_str, "fractal") == 0) {
        self->config.type = TOPOLOGY_FRACTAL;
        self->config.params.fractal = topology_default_fractal_config();
    } else {
        PyErr_SetString(PyExc_ValueError, "Invalid topology type");
        return -1;
    }

    // Validate configuration
    if (!topology_validate_config(&self->config)) {
        const char* error = topology_get_last_error();
        PyErr_SetString(PyExc_ValueError, error ? error : "Invalid configuration");
        return -1;
    }

    return 0;
}

static PyTypeObject TopologyConfigType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.TopologyConfig",
    .tp_doc = "Topology generation configuration",
    .tp_basicsize = sizeof(TopologyConfigObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = TopologyConfig_new,
    .tp_init = (initproc)TopologyConfig_init,
    .tp_dealloc = (destructor)TopologyConfig_dealloc,
};

//=============================================================================
// Topology Stats Type
//=============================================================================

typedef struct {
    PyObject_HEAD
    topology_stats_t stats;
} TopologyStatsObject;

static void TopologyStats_dealloc(TopologyStatsObject* self) {
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* TopologyStats_new(PyTypeObject* type, PyObject* args, PyObject* kwds) {
    TopologyStatsObject* self = (TopologyStatsObject*)type->tp_alloc(type, 0);
    if (self != NULL) {
        memset(&self->stats, 0, sizeof(topology_stats_t));
    }
    return (PyObject*)self;
}

// TopologyStats.num_neurons (read-only property)
static PyObject* TopologyStats_get_num_neurons(TopologyStatsObject* self, void* closure) {
    return PyLong_FromUnsignedLong(self->stats.num_neurons);
}

// TopologyStats.num_synapses (read-only property)
static PyObject* TopologyStats_get_num_synapses(TopologyStatsObject* self, void* closure) {
    return PyLong_FromUnsignedLong(self->stats.num_synapses);
}

// TopologyStats.avg_degree (read-only property)
static PyObject* TopologyStats_get_avg_degree(TopologyStatsObject* self, void* closure) {
    return PyFloat_FromDouble(self->stats.avg_degree);
}

// TopologyStats.clustering_coefficient (read-only property)
static PyObject* TopologyStats_get_clustering_coefficient(TopologyStatsObject* self, void* closure) {
    return PyFloat_FromDouble(self->stats.clustering_coefficient);
}

// TopologyStats.num_hubs (read-only property)
static PyObject* TopologyStats_get_num_hubs(TopologyStatsObject* self, void* closure) {
    return PyLong_FromUnsignedLong(self->stats.num_hubs);
}

static PyGetSetDef TopologyStats_getsetters[] = {
    {"num_neurons", (getter)TopologyStats_get_num_neurons, NULL, "Number of neurons", NULL},
    {"num_synapses", (getter)TopologyStats_get_num_synapses, NULL, "Number of synapses", NULL},
    {"avg_degree", (getter)TopologyStats_get_avg_degree, NULL, "Average degree", NULL},
    {"clustering_coefficient", (getter)TopologyStats_get_clustering_coefficient, NULL, "Clustering coefficient", NULL},
    {"num_hubs", (getter)TopologyStats_get_num_hubs, NULL, "Number of hub neurons", NULL},
    {NULL}
};

static PyTypeObject TopologyStatsType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "nimcp.TopologyStats",
    .tp_doc = "Topology statistics",
    .tp_basicsize = sizeof(TopologyStatsObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = TopologyStats_new,
    .tp_dealloc = (destructor)TopologyStats_dealloc,
    .tp_getset = TopologyStats_getsetters,
};

//=============================================================================
// Module-Level Functions
//=============================================================================

// topology_generate(network, config) -> TopologyStats
static PyObject* py_topology_generate(PyObject* self, PyObject* args) {
    PyObject* network_obj;
    PyObject* config_obj;

    if (!PyArg_ParseTuple(args, "OO", &network_obj, &config_obj)) {
        return NULL;
    }

    // Extract neural network
    if (!PyObject_TypeCheck(network_obj, &NeuralNetworkType)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be NeuralNetwork");
        return NULL;
    }
    NeuralNetworkObject* network_py = (NeuralNetworkObject*)network_obj;

    // Extract topology config
    if (!PyObject_TypeCheck(config_obj, &TopologyConfigType)) {
        PyErr_SetString(PyExc_TypeError, "Second argument must be TopologyConfig");
        return NULL;
    }
    TopologyConfigObject* config_py = (TopologyConfigObject*)config_obj;

    // Create stats object
    TopologyStatsObject* stats_obj = (TopologyStatsObject*)TopologyStats_new(&TopologyStatsType, NULL, NULL);
    if (!stats_obj) {
        return PyErr_NoMemory();
    }

    // Release GIL during potentially long topology generation
    bool success;
    Py_BEGIN_ALLOW_THREADS
    success = topology_generate(network_py->network, &config_py->config, &stats_obj->stats);
    Py_END_ALLOW_THREADS

    if (!success) {
        Py_DECREF(stats_obj);
        const char* error = topology_get_last_error();
        PyErr_SetString(NetworkError, error ? error : "Topology generation failed");
        NIMCP_THROW_TO_IMMUNE(NIMCP_ERROR_NULL_POINTER, "success is NULL");


        return NULL;
    }

    return (PyObject*)stats_obj;
}

// topology_default_scale_free_config() -> TopologyConfig
static PyObject* py_topology_default_scale_free_config(PyObject* self, PyObject* args) {
    TopologyConfigObject* config_obj = (TopologyConfigObject*)TopologyConfig_new(&TopologyConfigType, NULL, NULL);
    if (!config_obj) {
        return PyErr_NoMemory();
    }

    config_obj->config.type = TOPOLOGY_SCALE_FREE;
    config_obj->config.params.scale_free = topology_default_scale_free_config();

    return (PyObject*)config_obj;
}

//=============================================================================
// Module Method Table
//=============================================================================

static PyMethodDef topology_methods[] = {
    {"generate_topology", py_topology_generate, METH_VARARGS,
     "Generate network topology\n\n"
     "Args:\n"
     "    network (NeuralNetwork): Neural network to add topology to\n"
     "    config (TopologyConfig): Topology configuration\n\n"
     "Returns:\n"
     "    TopologyStats: Statistics about generated topology\n"},

    {"default_scale_free_config", py_topology_default_scale_free_config, METH_NOARGS,
     "Get default scale-free topology configuration\n\n"
     "Returns:\n"
     "    TopologyConfig: Default configuration\n"},

    {NULL, NULL, 0, NULL}
};

//=============================================================================
// Module Initialization
//=============================================================================

int init_topology_module(PyObject* module) {
    // Prepare types
    if (PyType_Ready(&TopologyConfigType) < 0)
        return -1;
    if (PyType_Ready(&TopologyStatsType) < 0)
        return -1;

    // Add types to module
    Py_INCREF(&TopologyConfigType);
    if (PyModule_AddObject(module, "TopologyConfig", (PyObject*)&TopologyConfigType) < 0) {
        Py_DECREF(&TopologyConfigType);
        return -1;
    }
    Py_INCREF(&TopologyStatsType);
    if (PyModule_AddObject(module, "TopologyStats", (PyObject*)&TopologyStatsType) < 0) {
        Py_DECREF(&TopologyStatsType);
        return -1;
    }

    // Add methods to module
    for (int i = 0; topology_methods[i].ml_name != NULL; i++) {
        PyObject* func = PyCFunction_New(&topology_methods[i], NULL);
        if (func == NULL)
            return -1;
        if (PyModule_AddObject(module, topology_methods[i].ml_name, func) < 0) {
            Py_DECREF(func);
            return -1;
        }
    }

    return 0;
}
