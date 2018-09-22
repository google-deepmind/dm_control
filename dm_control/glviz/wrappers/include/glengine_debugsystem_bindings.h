
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <debug/LDebugSystem.h>

namespace py = pybind11;


namespace glwrapper
{

    void drawLine( float x1, float y1, float z1, 
                   float x2, float y2, float z2,
                   float r, float g, float b );

    void drawLine( py::array_t<float> p1, 
                   py::array_t<float> p2,
                   py::array_t<float> color );

}

#define GLENGINE_DEBUGSYSTEM_BINDINGS(m) m.def( "drawLine", (void (*)(float,float,float,float,float,float,float,float,float)) &glwrapper::drawLine );\
                                         m.def( "drawLine", (void (*)(py::array_t<float>,py::array_t<float>,py::array_t<float>)) &glwrapper::drawLine );