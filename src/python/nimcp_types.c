/**
 * @file nimcp_types.c
 * @brief Python type definitions for NIMCP module
 */

#include "common/nimcp_module.h"

//=============================================================================
// NeuralNetwork Type
//=============================================================================

/**
 * @brief Initializes NeuralNetwork object from constructor arguments
 *
 * WHY: Python's type system calls tp_new to allocate, then tp_init to initialize.
 * This function creates the actual C neural network from Python arguments.
 *
 * SIGNATURE: NeuralNetwork(num_neurons)
 */
static int NeuralNetwork_init(NeuralNetworkObject* self, PyObject* args, PyObject* kwds)
{
    int num_neurons;

    // Parse constructor arguments
    if (!PyArg_ParseTuple(args, "i", &num_neurons)) {
        return -1;
    }

    // Create default configuration
    network_config_t config = {.num_neurons = num_neurons,
                               .ei_ratio = 0.8f,
                               .learning_rate = 0.01f,
                               .hebbian_rate = 0.1f,
                               .stdp_window = 20.0f,
                               .homeostatic_rate = 0.001f,
                               .target_activity = 0.1f,
                               .adaptation_rate = 0.1f,
                               .refractory_period = 5.0f,
                               .min_weight = -1.0f,
                               .max_weight = 1.0f,
                               .update_interval = 1000,
                               .input_size = 0,
                               .output_size = 0,
                               .num_layers = 0,
                               .layer_sizes = NULL,
                               .enable_stdp = true,
                               .enable_hebbian = true,
                               .enable_oja = true,
                               .enable_homeostasis = true};

    // Create the neural network
    self->network = neural_network_create(&config);
    if (!self->network) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create neural network");
        return -1;
    }

    return 0;
}

static void NeuralNetwork_dealloc(NeuralNetworkObject* self)
{
    if (self->network) {
        neural_network_destroy(self->network);
    }
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* NeuralNetwork_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    NeuralNetworkObject* self = (NeuralNetworkObject*) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->network = NULL;
    }
    return (PyObject*) self;
}

PyTypeObject NeuralNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "nimcp.NeuralNetwork",
    .tp_doc = "NIMCP Neural Network",
    .tp_basicsize = sizeof(NeuralNetworkObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NeuralNetwork_new,
    .tp_init = (initproc) NeuralNetwork_init,
    .tp_dealloc = (destructor) NeuralNetwork_dealloc,
};

//=============================================================================
// P2PNode Type
//=============================================================================

/**
 * @brief Initializes P2PNode object from constructor arguments
 *
 * WHY: Creates the actual C P2P node from Python arguments.
 *
 * SIGNATURE: P2PNode(listen_port)
 */
static int P2PNode_init(P2PNodeObject* self, PyObject* args, PyObject* kwds)
{
    unsigned short listen_port;

    // Parse constructor arguments
    if (!PyArg_ParseTuple(args, "H", &listen_port)) {
        return -1;
    }

    // Create default node configuration
    node_config_t config = {.listen_port = listen_port,
                            .max_peers = 10,
                            .keepalive_interval = 1000,
                            .discovery_interval = 5000,
                            .reconnect_interval = 3000,
                            .max_retries = 3,
                            .ping_interval = 2000};

    // Create the P2P node
    self->node = p2p_node_create(&config);
    if (!self->node) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create P2P node");
        return -1;
    }

    return 0;
}

static void P2PNode_dealloc(P2PNodeObject* self)
{
    if (self->node) {
        p2p_node_destroy(self->node);
    }
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* P2PNode_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    P2PNodeObject* self = (P2PNodeObject*) type->tp_alloc(type, 0);
    if (self != NULL) {
        self->node = NULL;
    }
    return (PyObject*) self;
}

PyTypeObject P2PNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "nimcp.P2PNode",
    .tp_doc = "NIMCP P2P Node",
    .tp_basicsize = sizeof(P2PNodeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = P2PNode_new,
    .tp_init = (initproc) P2PNode_init,
    .tp_dealloc = (destructor) P2PNode_dealloc,
};

//=============================================================================
// NetworkConfig Type
//=============================================================================

/**
 * @brief Initializes NetworkConfig object from constructor arguments
 *
 * WHY: Properly initializes config structure from Python constructor arguments.
 *
 * SIGNATURE: NetworkConfig(num_neurons, ei_ratio, learning_rate, hebbian_rate,
 *                          stdp_window, homeostatic_rate, target_activity, adaptation_rate)
 */
static int NetworkConfig_init(NetworkConfigObject* self, PyObject* args, PyObject* kwds)
{
    int num_neurons;
    float ei_ratio, learning_rate, hebbian_rate, stdp_window;
    float homeostatic_rate, target_activity, adaptation_rate;

    // Parse constructor arguments
    if (!PyArg_ParseTuple(args, "ifffffff", &num_neurons, &ei_ratio, &learning_rate, &hebbian_rate,
                          &stdp_window, &homeostatic_rate, &target_activity, &adaptation_rate)) {
        return -1;
    }

    // Initialize configuration
    self->config.num_neurons = num_neurons;
    self->config.ei_ratio = ei_ratio;
    self->config.learning_rate = learning_rate;
    self->config.hebbian_rate = hebbian_rate;
    self->config.stdp_window = stdp_window;
    self->config.homeostatic_rate = homeostatic_rate;
    self->config.target_activity = target_activity;
    self->config.adaptation_rate = adaptation_rate;

    // Set defaults for other fields
    self->config.refractory_period = 5.0f;
    self->config.min_weight = -1.0f;
    self->config.max_weight = 1.0f;
    self->config.update_interval = 1000;
    self->config.input_size = 0;
    self->config.output_size = 0;
    self->config.num_layers = 0;
    self->config.layer_sizes = NULL;
    self->config.enable_stdp = true;
    self->config.enable_hebbian = true;
    self->config.enable_oja = true;
    self->config.enable_homeostasis = true;

    return 0;
}

static void NetworkConfig_dealloc(NetworkConfigObject* self)
{
    // Config cleanup if needed
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* NetworkConfig_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    NetworkConfigObject* self = (NetworkConfigObject*) type->tp_alloc(type, 0);
    if (self != NULL) {
        memset(&self->config, 0, sizeof(network_config_t));
    }
    return (PyObject*) self;
}

PyTypeObject NetworkConfigType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "nimcp.NetworkConfig",
    .tp_doc = "NIMCP Network Configuration",
    .tp_basicsize = sizeof(NetworkConfigObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NetworkConfig_new,
    .tp_init = (initproc) NetworkConfig_init,
    .tp_dealloc = (destructor) NetworkConfig_dealloc,
};

//=============================================================================
// NodeConfig Type
//=============================================================================

/**
 * @brief Initializes NodeConfig object from constructor arguments
 *
 * WHY: Properly initializes node config structure from Python constructor arguments.
 *
 * SIGNATURE: NodeConfig(listen_port, max_peers, keepalive_interval,
 *                       discovery_interval, reconnect_interval, max_retries)
 */
static int NodeConfig_init(NodeConfigObject* self, PyObject* args, PyObject* kwds)
{
    unsigned short listen_port;
    int max_peers, keepalive_interval, discovery_interval;
    int reconnect_interval, max_retries;

    // Parse constructor arguments
    if (!PyArg_ParseTuple(args, "Hiiiii", &listen_port, &max_peers, &keepalive_interval,
                          &discovery_interval, &reconnect_interval, &max_retries)) {
        return -1;
    }

    // Initialize configuration
    self->config.listen_port = listen_port;
    self->config.max_peers = max_peers;
    self->config.keepalive_interval = keepalive_interval;
    self->config.discovery_interval = discovery_interval;
    self->config.reconnect_interval = reconnect_interval;
    self->config.max_retries = max_retries;

    // Set default for ping_interval
    self->config.ping_interval = 2000;

    return 0;
}

static void NodeConfig_dealloc(NodeConfigObject* self)
{
    // Config cleanup if needed
    Py_TYPE(self)->tp_free((PyObject*) self);
}

static PyObject* NodeConfig_new(PyTypeObject* type, PyObject* args, PyObject* kwds)
{
    NodeConfigObject* self = (NodeConfigObject*) type->tp_alloc(type, 0);
    if (self != NULL) {
        memset(&self->config, 0, sizeof(node_config_t));
    }
    return (PyObject*) self;
}

PyTypeObject NodeConfigType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "nimcp.NodeConfig",
    .tp_doc = "NIMCP Node Configuration",
    .tp_basicsize = sizeof(NodeConfigObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NodeConfig_new,
    .tp_init = (initproc) NodeConfig_init,
    .tp_dealloc = (destructor) NodeConfig_dealloc,
};
