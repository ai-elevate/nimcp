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

// NeuralNetwork.forward(inputs, num_outputs) -> list
static PyObject* NeuralNetwork_forward(NeuralNetworkObject* self, PyObject* args)
{
    PyObject* input_list;
    int num_outputs;

    if (!PyArg_ParseTuple(args, "Oi", &input_list, &num_outputs)) {
        return NULL;
    }

    // Convert Python list to C array
    if (!PyList_Check(input_list)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be a list");
        return NULL;
    }

    Py_ssize_t input_size = PyList_Size(input_list);
    float* inputs = (float*)malloc(sizeof(float) * input_size);
    if (!inputs) {
        return PyErr_NoMemory();
    }

    for (Py_ssize_t i = 0; i < input_size; i++) {
        PyObject* item = PyList_GetItem(input_list, i);
        inputs[i] = (float)PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            free(inputs);
            return NULL;
        }
    }

    // Allocate output array
    float* outputs = (float*)malloc(sizeof(float) * num_outputs);
    if (!outputs) {
        free(inputs);
        return PyErr_NoMemory();
    }

    // Call C function
    bool success = neural_network_forward(self->network, inputs, input_size, outputs, num_outputs);

    if (!success) {
        free(inputs);
        free(outputs);
        PyErr_SetString(NetworkError, "Forward pass failed");
        return NULL;
    }

    // Convert output array to Python list
    PyObject* result = PyList_New(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        PyList_SetItem(result, i, PyFloat_FromDouble(outputs[i]));
    }

    free(inputs);
    free(outputs);

    return result;
}

// NeuralNetwork.add_connection(from_id, to_id, weight) -> bool
static PyObject* NeuralNetwork_add_connection(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int from_id, to_id;
    float weight;

    if (!PyArg_ParseTuple(args, "IIf", &from_id, &to_id, &weight)) {
        return NULL;
    }

    bool success = neural_network_add_connection(self->network, from_id, to_id, weight);

    if (!success) {
        PyErr_SetString(NetworkError, "Failed to add connection");
        return NULL;
    }

    Py_RETURN_TRUE;
}

// NeuralNetwork.update_neuron(neuron_id, new_state, timestamp) -> bool
static PyObject* NeuralNetwork_update_neuron(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    float new_state;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IfK", &neuron_id, &new_state, &timestamp)) {
        return NULL;
    }

    bool success = neural_network_update_neuron(self->network, neuron_id, new_state, timestamp);

    if (!success) {
        PyErr_SetString(NetworkError, "Failed to update neuron");
        return NULL;
    }

    Py_RETURN_TRUE;
}

// NeuralNetwork.get_neuron_state(neuron_id) -> float
static PyObject* NeuralNetwork_get_neuron_state(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    float state;

    if (!PyArg_ParseTuple(args, "I", &neuron_id)) {
        return NULL;
    }

    bool success = neural_network_get_neuron_state(self->network, neuron_id, &state);

    if (!success) {
        PyErr_SetString(NetworkError, "Failed to get neuron state");
        return NULL;
    }

    return PyFloat_FromDouble(state);
}

// NeuralNetwork.compute_step(timestamp) -> int
static PyObject* NeuralNetwork_compute_step(NeuralNetworkObject* self, PyObject* args)
{
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "K", &timestamp)) {
        return NULL;
    }

    uint32_t active_neurons = neural_network_compute_step(self->network, timestamp);

    return PyLong_FromUnsignedLong(active_neurons);
}

// NeuralNetwork.reset() -> None
static PyObject* NeuralNetwork_reset(NeuralNetworkObject* self, PyObject* Py_UNUSED(ignored))
{
    neural_network_reset(self->network);
    Py_RETURN_NONE;
}

// NeuralNetwork.add_neuron(activation_type) -> int
static PyObject* NeuralNetwork_add_neuron(NeuralNetworkObject* self, PyObject* args)
{
    int activation_type;

    if (!PyArg_ParseTuple(args, "i", &activation_type)) {
        return NULL;
    }

    uint32_t neuron_id = neural_network_add_neuron(self->network, (activation_type_t)activation_type);
    return PyLong_FromUnsignedLong(neuron_id);
}

// NeuralNetwork.apply_stdp(neuron_id, timestamp) -> int
static PyObject* NeuralNetwork_apply_stdp(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IK", &neuron_id, &timestamp)) {
        return NULL;
    }

    uint32_t count = neural_network_apply_stdp(self->network, neuron_id, timestamp);
    return PyLong_FromUnsignedLong(count);
}

// NeuralNetwork.apply_oja(neuron_id, timestamp) -> int
static PyObject* NeuralNetwork_apply_oja(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IK", &neuron_id, &timestamp)) {
        return NULL;
    }

    uint32_t count = neural_network_apply_oja(self->network, neuron_id, timestamp);
    return PyLong_FromUnsignedLong(count);
}

// NeuralNetwork.apply_homeostasis(neuron_id, timestamp) -> bool
static PyObject* NeuralNetwork_apply_homeostasis(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IK", &neuron_id, &timestamp)) {
        return NULL;
    }

    bool success = neural_network_apply_homeostasis(self->network, neuron_id, timestamp);
    if (success) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// NeuralNetwork.normalize_weights(neuron_id) -> bool
static PyObject* NeuralNetwork_normalize_weights(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;

    if (!PyArg_ParseTuple(args, "I", &neuron_id)) {
        return NULL;
    }

    bool success = neural_network_normalize_weights(self->network, neuron_id);
    if (success) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// NeuralNetwork.record_spike(neuron_id, magnitude, timestamp) -> bool
static PyObject* NeuralNetwork_record_spike(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    float magnitude;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IfK", &neuron_id, &magnitude, &timestamp)) {
        return NULL;
    }

    bool success = neural_network_record_spike(self->network, neuron_id, magnitude, timestamp);
    if (success) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// NeuralNetwork.get_average_activity(neuron_id) -> float
static PyObject* NeuralNetwork_get_average_activity(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;

    if (!PyArg_ParseTuple(args, "I", &neuron_id)) {
        return NULL;
    }

    float activity = neural_network_get_average_activity(self->network, neuron_id);
    return PyFloat_FromDouble(activity);
}

// NeuralNetwork.get_weight_norm(neuron_id) -> float
static PyObject* NeuralNetwork_get_weight_norm(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;

    if (!PyArg_ParseTuple(args, "I", &neuron_id)) {
        return NULL;
    }

    float norm = neural_network_get_weight_norm(self->network, neuron_id);
    return PyFloat_FromDouble(norm);
}

// NeuralNetwork.get_weight_statistics(neuron_id) -> (mean, std_dev)
static PyObject* NeuralNetwork_get_weight_statistics(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    float mean, std_dev;

    if (!PyArg_ParseTuple(args, "I", &neuron_id)) {
        return NULL;
    }

    neural_network_get_weight_statistics(self->network, neuron_id, &mean, &std_dev);

    return Py_BuildValue("(ff)", mean, std_dev);
}

// NeuralNetwork.maintain_homeostasis(timestamp) -> None
static PyObject* NeuralNetwork_maintain_homeostasis(NeuralNetworkObject* self, PyObject* args)
{
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "K", &timestamp)) {
        return NULL;
    }

    neural_network_maintain_homeostasis(self->network, timestamp);
    Py_RETURN_NONE;
}

// NeuralNetwork.prune_synapses(threshold) -> int
static PyObject* NeuralNetwork_prune_synapses(NeuralNetworkObject* self, PyObject* args)
{
    float threshold;

    if (!PyArg_ParseTuple(args, "f", &threshold)) {
        return NULL;
    }

    uint32_t pruned = neural_network_prune_synapses(self->network, threshold);
    return PyLong_FromUnsignedLong(pruned);
}

// NeuralNetwork.get_incoming_synapse_count(neuron_id) -> int
static PyObject* NeuralNetwork_get_incoming_synapse_count(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;

    if (!PyArg_ParseTuple(args, "I", &neuron_id)) {
        return NULL;
    }

    uint32_t count = neural_network_get_incoming_synapse_count(self->network, neuron_id);
    return PyLong_FromUnsignedLong(count);
}

// NeuralNetwork.update_plasticity(neuron_id, timestamp) -> int
static PyObject* NeuralNetwork_update_plasticity(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IK", &neuron_id, &timestamp)) {
        return NULL;
    }

    uint32_t updates = neural_network_update_plasticity(self->network, neuron_id, timestamp);
    return PyLong_FromUnsignedLong(updates);
}

// NeuralNetwork.adapt_threshold(neuron_id, timestamp) -> bool
static PyObject* NeuralNetwork_adapt_threshold(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    unsigned long long timestamp;

    if (!PyArg_ParseTuple(args, "IK", &neuron_id, &timestamp)) {
        return NULL;
    }

    bool success = neural_network_adapt_threshold(self->network, neuron_id, timestamp);
    if (success) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// NeuralNetwork.set_neuron_model(neuron_id, model_type) -> bool
static PyObject* NeuralNetwork_set_neuron_model(NeuralNetworkObject* self, PyObject* args)
{
    unsigned int neuron_id;
    int model_type;

    // Parse arguments: neuron_id (required), model_type (required)
    // Note: params parameter not yet supported, defaults used
    if (!PyArg_ParseTuple(args, "Ii", &neuron_id, &model_type)) {
        return NULL;
    }

    // Validate model_type range
    if (model_type < 0 || model_type > 3) {
        PyErr_SetString(PyExc_ValueError, "Invalid model_type: must be 0-3 (LIF, Izhikevich, AdEx, Hodgkin-Huxley)");
        return NULL;
    }

    // Call C function with NULL params (use defaults)
    bool success = neural_network_set_neuron_model(
        self->network,
        neuron_id,
        (neuron_model_type_t)model_type,
        NULL  // Use default parameters for the model
    );

    if (success) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyMethodDef NeuralNetwork_methods[] = {
    {"forward", (PyCFunction)NeuralNetwork_forward, METH_VARARGS,
     "Perform forward pass through the network\n\n"
     "Args:\n"
     "    inputs (list): List of input values\n"
     "    num_outputs (int): Number of output neurons\n"
     "Returns:\n"
     "    list: Output values"},

    {"add_connection", (PyCFunction)NeuralNetwork_add_connection, METH_VARARGS,
     "Add a synaptic connection between neurons\n\n"
     "Args:\n"
     "    from_id (int): Source neuron ID\n"
     "    to_id (int): Target neuron ID\n"
     "    weight (float): Connection weight\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"update_neuron", (PyCFunction)NeuralNetwork_update_neuron, METH_VARARGS,
     "Update a neuron's state\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    new_state (float): New state value\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"get_neuron_state", (PyCFunction)NeuralNetwork_get_neuron_state, METH_VARARGS,
     "Get a neuron's current state\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "Returns:\n"
     "    float: Neuron state value"},

    {"compute_step", (PyCFunction)NeuralNetwork_compute_step, METH_VARARGS,
     "Compute one simulation step\n\n"
     "Args:\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    int: Number of active neurons"},

    {"reset", (PyCFunction)NeuralNetwork_reset, METH_NOARGS,
     "Reset the neural network to initial state"},

    {"add_neuron", (PyCFunction)NeuralNetwork_add_neuron, METH_VARARGS,
     "Add a new neuron to the network\n\n"
     "Args:\n"
     "    activation_type (int): Activation function type\n"
     "Returns:\n"
     "    int: New neuron ID"},

    {"apply_stdp", (PyCFunction)NeuralNetwork_apply_stdp, METH_VARARGS,
     "Apply Spike-Timing-Dependent Plasticity\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    int: Number of synapses updated"},

    {"apply_oja", (PyCFunction)NeuralNetwork_apply_oja, METH_VARARGS,
     "Apply Oja's learning rule\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    int: Number of synapses updated"},

    {"apply_homeostasis", (PyCFunction)NeuralNetwork_apply_homeostasis, METH_VARARGS,
     "Apply homeostatic plasticity\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"normalize_weights", (PyCFunction)NeuralNetwork_normalize_weights, METH_VARARGS,
     "Normalize synaptic weights for a neuron\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"record_spike", (PyCFunction)NeuralNetwork_record_spike, METH_VARARGS,
     "Record a spike event\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    magnitude (float): Spike magnitude\n"
     "    timestamp (int): Spike timestamp\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"get_average_activity", (PyCFunction)NeuralNetwork_get_average_activity, METH_VARARGS,
     "Get average activity level of a neuron\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "Returns:\n"
     "    float: Average activity"},

    {"get_weight_norm", (PyCFunction)NeuralNetwork_get_weight_norm, METH_VARARGS,
     "Get L2 norm of weights for a neuron\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "Returns:\n"
     "    float: Weight norm"},

    {"get_weight_statistics", (PyCFunction)NeuralNetwork_get_weight_statistics, METH_VARARGS,
     "Get weight statistics (mean and std dev)\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "Returns:\n"
     "    tuple: (mean, std_dev)"},

    {"maintain_homeostasis", (PyCFunction)NeuralNetwork_maintain_homeostasis, METH_VARARGS,
     "Maintain homeostatic balance across network\n\n"
     "Args:\n"
     "    timestamp (int): Current timestamp"},

    {"prune_synapses", (PyCFunction)NeuralNetwork_prune_synapses, METH_VARARGS,
     "Prune weak synapses below threshold\n\n"
     "Args:\n"
     "    threshold (float): Weight threshold\n"
     "Returns:\n"
     "    int: Number of synapses pruned"},

    {"get_incoming_synapse_count", (PyCFunction)NeuralNetwork_get_incoming_synapse_count, METH_VARARGS,
     "Get number of incoming synapses\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "Returns:\n"
     "    int: Number of incoming connections"},

    {"update_plasticity", (PyCFunction)NeuralNetwork_update_plasticity, METH_VARARGS,
     "Update plasticity mechanisms for a neuron\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    int: Number of updates applied"},

    {"adapt_threshold", (PyCFunction)NeuralNetwork_adapt_threshold, METH_VARARGS,
     "Adapt firing threshold for a neuron\n\n"
     "Args:\n"
     "    neuron_id (int): Neuron ID\n"
     "    timestamp (int): Current timestamp\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"set_neuron_model", (PyCFunction)NeuralNetwork_set_neuron_model, METH_VARARGS,
     "Set neuron model type for a specific neuron\n\n"
     "Args:\n"
     "    neuron_id (int): ID of neuron to modify\n"
     "    model_type (int): Model type (0=LIF, 1=Izhikevich, 2=AdEx, 3=Hodgkin-Huxley)\n"
     "Returns:\n"
     "    bool: True if successful, False otherwise\n\n"
     "Example:\n"
     "    # Switch neuron 5 to Izhikevich model\n"
     "    network.set_neuron_model(5, 1)"},

    {NULL, NULL, 0, NULL}
};

PyTypeObject NeuralNetworkType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "nimcp.NeuralNetwork",
    .tp_doc = "NIMCP Neural Network",
    .tp_basicsize = sizeof(NeuralNetworkObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = NeuralNetwork_new,
    .tp_init = (initproc) NeuralNetwork_init,
    .tp_dealloc = (destructor) NeuralNetwork_dealloc,
    .tp_methods = NeuralNetwork_methods,
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

// P2PNode.start() -> bool
static PyObject* P2PNode_start(P2PNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    bool success = p2p_node_start(self->node);

    if (!success) {
        PyErr_SetString(NodeError, "Failed to start P2P node");
        return NULL;
    }

    Py_RETURN_TRUE;
}

// P2PNode.stop() -> bool
static PyObject* P2PNode_stop(P2PNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    bool success = p2p_node_stop(self->node);

    if (!success) {
        PyErr_SetString(NodeError, "Failed to stop P2P node");
        return NULL;
    }

    Py_RETURN_TRUE;
}

// P2PNode.connect_peer(ip, port) -> bool
static PyObject* P2PNode_connect_peer(P2PNodeObject* self, PyObject* args)
{
    const char* peer_ip;
    unsigned short peer_port;

    if (!PyArg_ParseTuple(args, "sH", &peer_ip, &peer_port)) {
        return NULL;
    }

    bool success = p2p_node_connect_peer(self->node, peer_ip, peer_port);

    if (!success) {
        PyErr_SetString(NodeError, "Failed to connect to peer");
        return NULL;
    }

    Py_RETURN_TRUE;
}

// P2PNode.disconnect_peer(ip, port) -> bool
static PyObject* P2PNode_disconnect_peer(P2PNodeObject* self, PyObject* args)
{
    const char* peer_ip;
    unsigned short peer_port;

    if (!PyArg_ParseTuple(args, "sH", &peer_ip, &peer_port)) {
        return NULL;
    }

    bool success = p2p_node_disconnect_peer(self->node, peer_ip, peer_port);

    if (!success) {
        PyErr_SetString(NodeError, "Failed to disconnect from peer");
        return NULL;
    }

    Py_RETURN_TRUE;
}

// P2PNode.is_peer_connected(ip, port) -> bool
static PyObject* P2PNode_is_peer_connected(P2PNodeObject* self, PyObject* args)
{
    const char* peer_ip;
    unsigned short peer_port;

    if (!PyArg_ParseTuple(args, "sH", &peer_ip, &peer_port)) {
        return NULL;
    }

    bool connected = p2p_node_is_peer_connected(self->node, peer_ip, peer_port);

    if (connected) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

// P2PNode.send_heartbeats() -> int
static PyObject* P2PNode_send_heartbeats(P2PNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    uint32_t sent = p2p_node_send_heartbeats(self->node);
    return PyLong_FromUnsignedLong(sent);
}

// P2PNode.check_peer_health(timeout_ms) -> int
static PyObject* P2PNode_check_peer_health(P2PNodeObject* self, PyObject* args)
{
    unsigned int timeout_ms;

    if (!PyArg_ParseTuple(args, "I", &timeout_ms)) {
        return NULL;
    }

    uint32_t unhealthy = p2p_node_check_peer_health(self->node, timeout_ms);
    return PyLong_FromUnsignedLong(unhealthy);
}

// P2PNode.reconnect_unhealthy() -> int
static PyObject* P2PNode_reconnect_unhealthy(P2PNodeObject* self, PyObject* Py_UNUSED(ignored))
{
    uint32_t reconnected = p2p_node_reconnect_unhealthy(self->node);
    return PyLong_FromUnsignedLong(reconnected);
}

// P2PNode.process_pong(ip, port) -> bool
static PyObject* P2PNode_process_pong(P2PNodeObject* self, PyObject* args)
{
    const char* peer_ip;
    unsigned short peer_port;

    if (!PyArg_ParseTuple(args, "sH", &peer_ip, &peer_port)) {
        return NULL;
    }

    bool success = p2p_node_process_pong(self->node, peer_ip, peer_port);

    if (success) {
        Py_RETURN_TRUE;
    } else {
        Py_RETURN_FALSE;
    }
}

static PyMethodDef P2PNode_methods[] = {
    {"start", (PyCFunction)P2PNode_start, METH_NOARGS,
     "Start the P2P node\n\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"stop", (PyCFunction)P2PNode_stop, METH_NOARGS,
     "Stop the P2P node\n\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"connect_peer", (PyCFunction)P2PNode_connect_peer, METH_VARARGS,
     "Connect to a peer node\n\n"
     "Args:\n"
     "    ip (str): Peer IP address\n"
     "    port (int): Peer port number\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"disconnect_peer", (PyCFunction)P2PNode_disconnect_peer, METH_VARARGS,
     "Disconnect from a peer node\n\n"
     "Args:\n"
     "    ip (str): Peer IP address\n"
     "    port (int): Peer port number\n"
     "Returns:\n"
     "    bool: True if successful"},

    {"is_peer_connected", (PyCFunction)P2PNode_is_peer_connected, METH_VARARGS,
     "Check if a peer is connected\n\n"
     "Args:\n"
     "    ip (str): Peer IP address\n"
     "    port (int): Peer port number\n"
     "Returns:\n"
     "    bool: True if connected"},

    {"send_heartbeats", (PyCFunction)P2PNode_send_heartbeats, METH_NOARGS,
     "Send heartbeats to all connected peers\n\n"
     "Returns:\n"
     "    int: Number of heartbeats sent"},

    {"check_peer_health", (PyCFunction)P2PNode_check_peer_health, METH_VARARGS,
     "Check health of all peers\n\n"
     "Args:\n"
     "    timeout_ms (int): Timeout in milliseconds\n"
     "Returns:\n"
     "    int: Number of unhealthy peers"},

    {"reconnect_unhealthy", (PyCFunction)P2PNode_reconnect_unhealthy, METH_NOARGS,
     "Reconnect to unhealthy peers\n\n"
     "Returns:\n"
     "    int: Number of peers reconnected"},

    {"process_pong", (PyCFunction)P2PNode_process_pong, METH_VARARGS,
     "Process pong message from peer\n\n"
     "Args:\n"
     "    ip (str): Peer IP address\n"
     "    port (int): Peer port number\n"
     "Returns:\n"
     "    bool: True if successful"},

    {NULL, NULL, 0, NULL}
};

PyTypeObject P2PNodeType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "nimcp.P2PNode",
    .tp_doc = "NIMCP P2P Node",
    .tp_basicsize = sizeof(P2PNodeObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = P2PNode_new,
    .tp_init = (initproc) P2PNode_init,
    .tp_dealloc = (destructor) P2PNode_dealloc,
    .tp_methods = P2PNode_methods,
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
