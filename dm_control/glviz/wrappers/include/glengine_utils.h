
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <LCommon.h>

namespace py = pybind11;

namespace glwrapper
{

    engine::LVec3 numpyToVec3( py::array_t<float> npArray );
    py::array_t<float> vec3ToNumpy( const engine::LVec3& vec3 );


}