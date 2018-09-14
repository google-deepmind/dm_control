
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <LMesh.h>

namespace py = pybind11;

namespace glwrapper
{

    class Mesh
    {

        private :

        engine::LMesh* m_glMeshRef;

        public :

        Mesh();
        ~Mesh();

        void setMeshReference( engine::LMesh* pMesh );

        // Python exposed-API **************************************

        void setX( float x );
        void setY( float y );
        void setZ( float z );
        void setPosition( float x, float y, float z );
        void setPosition( py::array_t<float> posArray );

        float getX();
        float getY();
        float getZ();
        py::array_t<float> getPosition();

        // void setRotation();
        // void getRotation();

        // *********************************************************
    };


}