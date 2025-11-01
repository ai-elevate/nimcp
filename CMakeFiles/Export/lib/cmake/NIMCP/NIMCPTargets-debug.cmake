#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "NIMCP::nimcp_core" for configuration "Debug"
set_property(TARGET NIMCP::nimcp_core APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(NIMCP::nimcp_core PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libnimcp_core.so.2.5.0"
  IMPORTED_SONAME_DEBUG "libnimcp_core.so.2"
  )

list(APPEND _IMPORT_CHECK_TARGETS NIMCP::nimcp_core )
list(APPEND _IMPORT_CHECK_FILES_FOR_NIMCP::nimcp_core "${_IMPORT_PREFIX}/lib/libnimcp_core.so.2.5.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
