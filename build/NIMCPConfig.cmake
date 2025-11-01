# NIMCPConfig.cmake - CMake configuration file for NIMCP


####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was NIMCPConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

set(NIMCP_VERSION "2.5.0")

# Import targets
include("${CMAKE_CURRENT_LIST_DIR}/NIMCPTargets.cmake")

# Provide targets
if(NOT TARGET NIMCP::core)
    message(FATAL_ERROR "Expected target NIMCP::core not found")
endif()

# Set variables for backward compatibility
set(NIMCP_LIBRARIES NIMCP::core)
set(NIMCP_INCLUDE_DIRS "/usr/local/include")

check_required_components(NIMCP)
