
find_package( OpenGL REQUIRED )

include_directories( ${EXT_INCLUDE_DIRECTORIES} )

pybind11_add_module( glviz
                     src/glengine_bindings.cpp )
target_link_libraries( glviz PRIVATE 
                       gengine )