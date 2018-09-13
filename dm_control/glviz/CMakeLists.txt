cmake_minimum_required( VERSION 3.0.0 )

project( glviz )

message( "CONFIGURING GLVIZ BINDINGS" )

add_subdirectory( pybind11 )

# Add external libraries ----------------------------------------------
set( EXT_INCLUDE_DIRECTORIES
	 "${CMAKE_SOURCE_DIR}/ext/glfw/include"
     "${CMAKE_BINARY_DIR}/ext/glad/include"
     "${CMAKE_SOURCE_DIR}/ext/glm"
     "${CMAKE_SOURCE_DIR}/glengine/include" )

# Avoid building the examples in the libraries
set( GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build the GLFW example programs" )
set( GLFW_BUILD_TESTS OFF CACHE BOOL "Build the GLFW test programs" )
set( GLM_TEST_ENABLE OFF CACHE BOOL "Build unit tests" )

add_subdirectory( ext/glfw )
add_subdirectory( ext/glad )
add_subdirectory( ext/glm )
# ---------------------------------------------------------------------

# glengine library ----------------------------------------------------
set( CAT1_BUILD_EXT_LIBS OFF CACHE BOOL "Build external dependencies along with the project" )
set( CAT1_INCLUDE_DIRECTORIES
     "${CMAKE_SOURCE_DIR}/ext/glfw/include"
     "${CMAKE_SOURCE_DIR}/ext/glad/include"
     "${CMAKE_SOURCE_DIR}/ext/glm" )
set( CAT1_ENGINE_RESOURCES_PATH
     "${CMAKE_SOURCE_DIR}/glengine/res/" )
add_subdirectory( glengine )
# ---------------------------------------------------------------------

# wrappers ------------------------------------------------------------
add_subdirectory( wrappers )
# ---------------------------------------------------------------------