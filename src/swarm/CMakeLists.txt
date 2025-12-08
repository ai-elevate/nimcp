# =============================================================================
# NIMCP Swarm Module - CMakeLists.txt
# =============================================================================
# Swarm brain coordination for drone/IoT collective intelligence
# =============================================================================

set(SWARM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_swarm_protocol.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_swarm_signal.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_collective_workspace.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_swarm_emergence.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_swarm_consensus.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_swarm_brain.c
    ${CMAKE_CURRENT_SOURCE_DIR}/nimcp_swarm_gateway.c
)

# Export to parent scope for inclusion in main library
set(SWARM_SOURCES ${SWARM_SOURCES} PARENT_SCOPE)

# Include directories
target_include_directories(nimcp PUBLIC
    ${CMAKE_SOURCE_DIR}/include/swarm
)
