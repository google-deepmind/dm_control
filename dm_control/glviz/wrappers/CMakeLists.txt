
find_package( OpenGL REQUIRED )

include_directories( ${EXT_INCLUDE_DIRECTORIES} )

pybind11_add_module( enginewrapper
                     src/glengine_mesh_bindings.cpp
                     src/glengine_meshbuilder_bindings.cpp
                     src/glengine_debugsystem_bindings.cpp
                     src/glengine_input_bindings.cpp
                     src/glengine_bindings.cpp )
target_link_libraries( enginewrapper PRIVATE 
                       gengine )