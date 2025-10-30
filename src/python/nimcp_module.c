#include "../include/nimcp_module.h"

// Initialize exception types
PyObject* NIMCPError;
PyObject* NetworkError;
PyObject* ProtocolError;
PyObject* NodeError;

// Module method definitions
static PyMethodDef nimcp_methods[] = {
    // Protocol methods will be added here
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef nimcp_module = {
    PyModuleDef_HEAD_INIT,
    "nimcp",
    "Neural Interface Message Communication Protocol",
    -1,
    nimcp_methods
};

// Module initialization
PyMODINIT_FUNC PyInit_nimcp(void) {
    PyObject *m;
    
    // Initialize types
    if (PyType_Ready(&NeuralNetworkType) < 0)
        return NULL;
    if (PyType_Ready(&P2PNodeType) < 0)
        return NULL;
    if (PyType_Ready(&NetworkConfigType) < 0)
        return NULL;
    if (PyType_Ready(&NodeConfigType) < 0)
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
    Py_INCREF(&NeuralNetworkType);
    PyModule_AddObject(m, "NeuralNetwork", (PyObject *)&NeuralNetworkType);
    Py_INCREF(&P2PNodeType);
    PyModule_AddObject(m, "P2PNode", (PyObject *)&P2PNodeType);
    Py_INCREF(&NetworkConfigType);
    PyModule_AddObject(m, "NetworkConfig", (PyObject *)&NetworkConfigType);
    Py_INCREF(&NodeConfigType);
    PyModule_AddObject(m, "NodeConfig", (PyObject *)&NodeConfigType);

    // Add exceptions to module
    PyModule_AddObject(m, "NIMCPError", NIMCPError);
    PyModule_AddObject(m, "NetworkError", NetworkError);
    PyModule_AddObject(m, "ProtocolError", ProtocolError);
    PyModule_AddObject(m, "NodeError", NodeError);

    return m;
}
