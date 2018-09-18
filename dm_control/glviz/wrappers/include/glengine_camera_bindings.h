
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <LICamera.h>
#include <LFixedCamera3d.h>
#include <LFpsCamera.h>

#include <LApp.h>

#include <glengine_utils.h>
#include <glengine_mesh_bindings.h>

using namespace std;

namespace py = pybind11;

#define CAMERA_TYPE_FIXED 0
#define CAMERA_TYPE_FPS 1
#define CAMERA_TYPE_ORBIT 2
#define CAMERA_TYPE_FOLLOW 3

namespace glwrapper
{

    class Camera
    {
        protected :

        engine::LICamera* m_cameraPtr;

        public :

        Camera( const string& name,
                py::array_t<float> position,
                py::array_t<float> targetDir,
                int worldUpId,
                float fov,
                float aspectRatio,
                float zNear, float zFar );
        Camera( engine::LICamera* ptrCamera );
        ~Camera();

        void setPosition( py::array_t<float> position );
        void setTargetDir( py::array_t<float> targetDir );

        py::array_t<float> getPosition();
        py::array_t<float> getTargetDir();

        string name();
        string type();

        engine::LICamera* ptr() { return m_cameraPtr; }
    };

    class CameraFixed : public Camera
    {
        public :

        CameraFixed( const string& name,
                     py::array_t<float> position,
                     py::array_t<float> targetDir,
                     int worldUpId,
                     float fov,
                     float aspectRatio,
                     float zNear, float zFar );
        ~CameraFixed(){}
    };

    class CameraFps : public Camera
    {
        public :

        CameraFps( const string& name,
                   py::array_t<float> position,
                   py::array_t<float> targetDir,
                   int worldUpId,
                   float fov,
                   float aspectRatio,
                   float zNear, float zFar );
        ~CameraFps(){}
    };
    
    class CameraOrbit : public Camera
    {
        public :

        CameraOrbit( const string& name,
                     py::array_t<float> position,
                     py::array_t<float> targetDir,
                     int worldUpId,
                     float fov,
                     float aspectRatio,
                     float zNear, float zFar );
        ~CameraOrbit(){}
    };

    class CameraFollow : public Camera
    {
        private :

        Mesh* m_meshPtr;

        public :

        CameraFollow( const string& name,
                      py::array_t<float> position,
                      py::array_t<float> targetDir,
                      int worldUpId,
                      float fov,
                      float aspectRatio,
                      float zNear, float zFar );
        ~CameraFollow(){}

        void setFollowReference( Mesh* pMeshWrapperObj );
    };

    Camera* createCamera( int type, const string& name,
                          py::array_t<float> position,
                          py::array_t<float> targetDir,
                          int worldUpId,
                          float fov, float zNear, float zFar );
    Camera* getCurrentCamera();
}

#define DefCameraConstructorParams const string& , py::array_t<float>,\
                                    py::array_t<float> , int ,\
                                    float , float , float , float 
#define GLENGINE_CAMERA_BINDINGS(m) py::class_<glwrapper::Camera>( m, "Camera" ) \
                                        .def( py::init<DefCameraConstructorParams>() ) \
                                        .def( "setPosition", &glwrapper::Camera::setPosition ) \
                                        .def( "setTargetDir", &glwrapper::Camera::setTargetDir ) \
                                        .def( "getPosition", &glwrapper::Camera::getPosition ) \
                                        .def( "getTargetDir", &glwrapper::Camera::getTargetDir ) \
                                        .def( "name", &glwrapper::Camera::name ) \
                                        .def( "type", &glwrapper::Camera::type );\
                                    py::class_<glwrapper::CameraFixed, glwrapper::Camera>( m, "CameraFixed" ) \
                                        .def( py::init<DefCameraConstructorParams>() );\
                                    py::class_<glwrapper::CameraFps, glwrapper::Camera>( m, "CameraFps" ) \
                                        .def( py::init<DefCameraConstructorParams>() );\
                                    py::class_<glwrapper::CameraOrbit, glwrapper::Camera>( m, "CameraOrbit" )\
                                        .def( py::init<DefCameraConstructorParams>() );\
                                    py::class_<glwrapper::CameraFollow, glwrapper::Camera>( m, "CameraFollow" )\
                                        .def( py::init<DefCameraConstructorParams>() ) \
                                        .def( "setFollowReference", &glwrapper::CameraFollow::setFollowReference );\
                                    m.def( "createCamera", &glwrapper::createCamera );\
                                    m.def( "getCurrentCamera", &glwrapper::getCurrentCamera );