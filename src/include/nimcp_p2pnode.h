// nimcp_p2pnode.h

#ifndef NIMCP_P2PNODE_H
#define NIMCP_P2PNODE_H

#include "/usr/include/python3.10/Python.h"
#include "nimcp_export.h"

// Your type declarations
extern NIMCP_EXPORT PyTypeObject NeuralNetworkType;


#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Enumeration of possible node states
 */
typedef enum {
    NODE_STATUS_INIT,    /**< Node is initializing */
    NODE_STATUS_RUNNING, /**< Node is operational */
    NODE_STATUS_ERROR,   /**< Node has encountered an error */
    NODE_STATUS_SHUTDOWN /**< Node is shutting down */
} node_status_t;

/**
 * @brief Structure containing peer connection information
 */
typedef struct {
    char ip[16];    /**< IPv4 address as string */
    uint16_t port;  /**< Port number */
    int socket_fd;  /** Stores the socket file descrptor */
    bool connected; /**< Connection status */
} peer_info_t;

/**
 * @brief Configuration structure for node initialization
 */
typedef struct {
    uint16_t listen_port;         /**< Port to listen on */
    uint32_t max_peers;           /**< Maximum number of peer connections */
    uint32_t keepalive_interval;  // Add this field
    uint32_t discovery_interval;
    uint32_t reconnect_interval;
    uint32_t max_retries;
    uint32_t ping_interval; /**< Health check interval in milliseconds */
} node_config_t;

/**
 * @brief Opaque pointer type for node instance
 */
typedef struct p2p_node_struct* p2p_node_t;

/**
 * @brief Creates a new P2P node instance
 * @param config Pointer to configuration structure
 * @return Handle to the created node, NULL if creation fails
 */
p2p_node_t p2p_node_create(const node_config_t* config);

/**
 * @brief Destroys a P2P node instance and frees resources
 * @param node Handle to the node to destroy
 */
void p2p_node_destroy(p2p_node_t node);

/**
 * @brief Gets current status of the node
 * @param node Handle to the node
 * @return Current node status
 */
node_status_t p2p_node_get_status(p2p_node_t node);

/**
 * @brief Attempts to connect to a peer
 * @param node Handle to the node
 * @param peer_ip IP address of peer
 * @param peer_port Port number of peer
 * @return true if connection initiated successfully, false otherwise
 */
bool p2p_node_connect_peer(p2p_node_t node, const char* peer_ip, uint16_t peer_port);

/**
 * @brief Checks if a peer is currently connected
 * @param node Handle to the node
 * @param peer_ip IP address of peer
 * @param peer_port Port number of peer
 * @return true if peer is connected, false otherwise
 */
bool p2p_node_is_peer_connected(p2p_node_t node, const char* peer_ip, uint16_t peer_port);

/**
 * @brief Disconnects from a peer
 * @param node Handle to the node
 * @param peer_ip IP address of peer
 * @param peer_port Port number of peer
 * @return true if disconnection successful, false otherwise
 */
bool p2p_node_disconnect_peer(p2p_node_t node, const char* peer_ip, uint16_t peer_port);

/**
 * @brief Starts the node's main operation
 * @param node Handle to the node
 * @return true if startup successful, false otherwise
 */
bool p2p_node_start(p2p_node_t node);

/**
 * @brief Stops the node's operation
 * @param node Handle to the node
 * @return true if shutdown successful, false otherwise
 */
bool p2p_node_stop(p2p_node_t node);

/**
 * @brief Get topology graph for network analysis
 *
 * Returns the internal topology graph that tracks peer connections.
 * Useful for introspection, pathfinding, and network analysis.
 *
 * @param node Handle to the node
 * @return Pointer to topology graph, or NULL if node is invalid
 *
 * @note The graph is owned by the node - do not destroy it
 * @note Graph operations are thread-safe (protected by internal mutex)
 * @note Use nimcp_graph.h API functions to query the graph
 */
struct NimcpGraph* p2p_node_get_topology_graph(p2p_node_t node);

#endif  // NIMCP_P2PNODE_H
