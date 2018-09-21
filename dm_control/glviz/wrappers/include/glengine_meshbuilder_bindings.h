
#pragma once

// pybind functions
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// glengine functionality
#include <LApp.h>
#include <LMeshBuilder.h>
// wrapped functionality
#include <glengine_mesh_bindings.h>

namespace py = pybind11;

namespace glwrapper
{

    Mesh* createSphere( float radius );
    Mesh* createBox( float width, float height, float depth );
    Mesh* createCapsule( float radius, float height );
    Mesh* createPlane( float width, float depth, float texRangeWidth = 10.0f, float texRangeDepth = 10.0f );
}

#define DefDefaultParamsPlane py::arg( "width" ), py::arg( "height" ), \
                               py::arg( "texRangeWidth" ) = 10.0f, py::arg( "texRangeDepth" ) = 10.0f

#define GLENGINE_MESHBUILDER_BINDINGS(m) m.def( "createSphere", &glwrapper::createSphere );\
                                         m.def( "createBox", &glwrapper::createBox );\
                                         m.def( "createCapsule", &glwrapper::createCapsule );\
                                         m.def( "createPlane", &glwrapper::createPlane, "Creates a mesh-plane wrapper", DefDefaultParamsPlane );