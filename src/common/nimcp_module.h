#ifndef NIMCP_MODULE_H
#define NIMCP_MODULE_H

#define PY_SSIZE_T_CLEAN
#include "/usr/include/python3.10/Python.h"
#include "core/neuralnet/nimcp_neuralnet.h"
#include "networking/p2p/nimcp_p2pnode.h"
#include "networking/protocol/nimcp_protocol.h"

// Exception types
extern PyObject* NIMCPError;
extern PyObject* NetworkError;
extern PyObject* ProtocolError;
extern PyObject* NodeError;

// Type objects
extern PyTypeObject NeuralNetworkType;
extern PyTypeObject P2PNodeType;
extern PyTypeObject NetworkConfigType;
extern PyTypeObject NodeConfigType;

// Custom types definitions
typedef struct {
    PyObject_HEAD neural_network_t network;
} NeuralNetworkObject;

typedef struct {
    PyObject_HEAD p2p_node_t node;
} P2PNodeObject;

typedef struct {
    PyObject_HEAD network_config_t config;
} NetworkConfigObject;

typedef struct {
    PyObject_HEAD node_config_t config;
} NodeConfigObject;

// Module initialization function
PyMODINIT_FUNC PyInit_nimcp(void);

#endif  // NIMCP_MODULE_H
