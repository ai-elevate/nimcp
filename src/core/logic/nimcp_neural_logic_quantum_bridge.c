/**
 * @file nimcp_neural_logic_quantum_bridge.c
 * @brief Implementation of quantum-accelerated neural logic bridge
 *
 * WHAT: Quantum SAT solving for neural logic circuits
 * WHY:  √N speedup for satisfiability search
 * HOW:  Grover's algorithm on circuit CNF encoding
 *
 * @author NIMCP Development Team
 * @date 2025-01-20
 */

#define NIMCP_NEURAL_LOGIC_QUANTUM_BRIDGE_IMPLEMENTATION
#include "core/logic/nimcp_neural_logic_quantum_bridge.h"
#include "utils/exception/nimcp_exception_macros.h"

/**
 * WHAT: Export default config function for C linkage
 * WHY:  Allow neural_logic.c to call without including full bridge header
 * NOTE: Different name from inline function to avoid conflict
 */
void neural_logic_quantum_default_config_opaque(void* config_out)
{
    if (!config_out) return;
    neural_logic_quantum_config_t cfg = neural_logic_quantum_default_config();
    memcpy(config_out, &cfg, sizeof(cfg));
}
