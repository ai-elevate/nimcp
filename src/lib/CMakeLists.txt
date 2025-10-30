# Core library sources
set(NIMCP_CORE_SOURCES
    nimcp_neuralnet.c
    nimcp_p2pnode.c
    nimcp_protocol.c
    nimcp_events.c
    nimcp_adaptive.c
    nimcp_brain.c
    nimcp_ethics.c
    nimcp_curiosity.c
    nimcp_knowledge.c
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_module.c   # Add the Python module source
    ${CMAKE_SOURCE_DIR}/src/python/nimcp_types.c    # Add the Python types
)

# Core library
add_library(nimcp_core SHARED
    ${NIMCP_CORE_SOURCES}
)

target_include_directories(nimcp_core
    PUBLIC
        ${NIMCP_INCLUDE_DIR}
    PRIVATE
        ${NIMCP_LIB_DIR}
        ${Python3_INCLUDE_DIRS}
)

target_link_libraries(nimcp_core
    PUBLIC
        Python3::Python
)

# Enable Position Independent Code
set_target_properties(nimcp_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ENABLE_EXPORTS ON
)

# Installation rules
install(TARGETS nimcp_core
    EXPORT nimcp-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
