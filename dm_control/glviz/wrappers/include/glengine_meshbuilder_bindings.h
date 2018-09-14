
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
    Mesh* createPlane( float width, float depth );
}