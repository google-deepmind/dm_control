
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <input/LInputSystem.h>

namespace py = pybind11;

namespace glwrapper
{

    bool isKeyDown( int key );
    bool isMouseDown( int button );

    py::array_t<float> getCursorPosition();

}

#define GLENGINE_INPUT_BINDINGS(m) m.def( "isKeyDown", &glwrapper::isKeyDown );\
                                   m.def( "isMouseDown", &glwrapper::isMouseDown );\
                                   m.def( "getCursorPosition", &glwrapper::getCursorPosition );