
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <LAssetsManager.h>
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
        Mesh( engine::LMesh* pMesh );
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

        void setRotation( py::array_t<float> mat );
        // void getRotation();

        void setBuiltInTexture( const string& textureId );
        void setColor( float r, float g, float b );

        engine::LMesh* ptr() { return m_glMeshRef; }

        // *********************************************************
    };

}

#define GLENGINE_MESH_BINDINGS(m) py::class_<glwrapper::Mesh>( m, "Mesh" ) \
                                        .def( py::init<>() ) \
                                        .def( "setX", &glwrapper::Mesh::setX ) \
                                        .def( "setY", &glwrapper::Mesh::setY ) \
                                        .def( "setZ", &glwrapper::Mesh::setZ ) \
                                        .def( "setPosition", (void (glwrapper::Mesh::*)(float,float,float)) &glwrapper::Mesh::setPosition ) \
                                        .def( "setPosition", (void (glwrapper::Mesh::*)(py::array_t<float>)) &glwrapper::Mesh::setPosition ) \
                                        .def( "getX", &glwrapper::Mesh::getX ) \
                                        .def( "getY", &glwrapper::Mesh::getY ) \
                                        .def( "getZ", &glwrapper::Mesh::getZ ) \
                                        .def( "getPosition", &glwrapper::Mesh::getPosition ) \
                                        .def( "setRotation", &glwrapper::Mesh::setRotation ) \
                                        .def( "setBuiltInTexture", &glwrapper::Mesh::setBuiltInTexture ) \
                                        .def( "setColor", &glwrapper::Mesh::setColor );