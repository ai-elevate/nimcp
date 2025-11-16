#include "common/nimcp_module.h"

// Forward declaration for metrics module
extern int init_metrics_module(PyObject* module);

// Define exception types
// These are initialized in PyInit_nimcp()
PyObject* NIMCPError = NULL;
PyObject* NetworkError = NULL;
PyObject* ProtocolError = NULL;
PyObject* NodeError = NULL;

// Module method definitions
static PyMethodDef nimcp_methods[] = {
    // Protocol methods will be added here
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef nimcp_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "nimcp",
    .m_doc = "Neural Interface Message Communication Protocol",
    .m_size = -1,
    .m_methods = nimcp_methods,
    .m_slots = NULL
};

// Module initialization
PyMODINIT_FUNC PyInit_nimcp(void)
{
    PyObject* m;

    // Initialize types
    if (PyType_Ready(&BrainType) < 0)
        return NULL;
    if (PyType_Ready(&NeuralNetworkType) < 0)
        return NULL;
    if (PyType_Ready(&P2PNodeType) < 0)
        return NULL;
    if (PyType_Ready(&NetworkConfigType) < 0)
        return NULL;
    if (PyType_Ready(&NodeConfigType) < 0)
        return NULL;
    if (PyType_Ready(&GlialIntegrationType) < 0)
        return NULL;

    m = PyModule_Create(&nimcp_module);
    if (m == NULL)
        return NULL;

    // Create exception hierarchy
    NIMCPError = PyErr_NewException("nimcp.NIMCPError", NULL, NULL);
    NetworkError = PyErr_NewException("nimcp.NetworkError", NIMCPError, NULL);
    ProtocolError = PyErr_NewException("nimcp.ProtocolError", NIMCPError, NULL);
    NodeError = PyErr_NewException("nimcp.NodeError", NIMCPError, NULL);

    // Add types to module
    Py_INCREF(&BrainType);
    PyModule_AddObject(m, "Brain", (PyObject*) &BrainType);
    Py_INCREF(&NeuralNetworkType);
    PyModule_AddObject(m, "NeuralNetwork", (PyObject*) &NeuralNetworkType);
    Py_INCREF(&P2PNodeType);
    PyModule_AddObject(m, "P2PNode", (PyObject*) &P2PNodeType);
    Py_INCREF(&NetworkConfigType);
    PyModule_AddObject(m, "NetworkConfig", (PyObject*) &NetworkConfigType);
    Py_INCREF(&NodeConfigType);
    PyModule_AddObject(m, "NodeConfig", (PyObject*) &NodeConfigType);
    Py_INCREF(&GlialIntegrationType);
    PyModule_AddObject(m, "GlialIntegration", (PyObject*) &GlialIntegrationType);

    // Add exceptions to module
    PyModule_AddObject(m, "NIMCPError", NIMCPError);
    PyModule_AddObject(m, "NetworkError", NetworkError);
    PyModule_AddObject(m, "ProtocolError", ProtocolError);
    PyModule_AddObject(m, "NodeError", NodeError);

    // Initialize metrics module
    if (init_metrics_module(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    // Initialize topology module
    if (init_topology_module(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    // Initialize pink noise module
    if (init_pink_noise_module(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    // Initialize community detection module
    if (init_community_module(m) < 0) {
        Py_DECREF(m);
        return NULL;
    }

    return m;
}
