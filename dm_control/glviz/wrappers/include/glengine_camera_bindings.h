
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <LICamera.h>
#include <LFixedCamera3d.h>
#include <LFpsCamera.h>
#include <LFollowCamera.h>

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

        py::array_t<float> getPosition();
        py::array_t<float> getTargetDir();
        py::array_t<float> getTargetPoint();

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

    Camera* _createCamera( int type, const string& name,
                           py::array_t<float> position,
                           py::array_t<float> targetPoint,
                           int worldUpId = engine::LICamera::UP_Z,
                           float fov = 45.0f, 
                           float zNear = 1.0f, float zFar = 40.0f );
                           
    CameraFixed* createFixedCamera( const string& name,
                                    py::array_t<float> position,
                                    py::array_t<float> targetPoint,
                                    int worldUpId = engine::LICamera::UP_Z,
                                    float fov = 45.0f, 
                                    float zNear = 1.0f, float zFar = 40.0f );

    CameraFps* createFpsCamera( const string& name,
                                py::array_t<float> position,
                                py::array_t<float> targetPoint,
                                int worldUpId = engine::LICamera::UP_Z,
                                float fov = 45.0f, 
                                float zNear = 1.0f, float zFar = 40.0f );

    CameraOrbit* createOrbitCamera( const string& name,
                                    py::array_t<float> position,
                                    py::array_t<float> targetPoint,
                                    int worldUpId = engine::LICamera::UP_Z,
                                    float fov = 45.0f, 
                                    float zNear = 1.0f, float zFar = 40.0f );

    CameraFollow* createFollowCamera( const string& name,
                                      py::array_t<float> position,
                                      py::array_t<float> targetPoint,
                                      int worldUpId = engine::LICamera::UP_Z,
                                      float fov = 45.0f, 
                                      float zNear = 1.0f, float zFar = 40.0f );
    Camera* getCurrentCamera();

    void changeToCameraByName( const string& name );
}

#define DefCameraConstructorParams const string& , py::array_t<float>,\
                                    py::array_t<float> , int ,\
                                    float , float , float , float 
#define DefDefaultCameraParams py::arg( "name" ), py::arg( "position" ), py::arg( "targetPoint" ), \
                               py::arg( "worldUpId" ) = 2, py::arg( "fov" ) = 45.0f, \
                               py::arg( "zNear" ) = 1.0f, py::arg( "zFar" ) = 40.0f
#define GLENGINE_CAMERA_BINDINGS(m) py::class_<glwrapper::Camera>( m, "Camera" ) \
                                        .def( py::init<DefCameraConstructorParams>() ) \
                                        .def( "setPosition", &glwrapper::Camera::setPosition ) \
                                        .def( "getPosition", &glwrapper::Camera::getPosition ) \
                                        .def( "getTargetDir", &glwrapper::Camera::getTargetDir ) \
                                        .def( "getTargetPoint", &glwrapper::Camera::getTargetPoint ) \
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
                                    m.def( "createFixedCamera", &glwrapper::createFixedCamera, "Creates a fixed-type camera", DefDefaultCameraParams );\
                                    m.def( "createFpsCamera", &glwrapper::createFpsCamera, "Creates a fps-type camera", DefDefaultCameraParams );\
                                    m.def( "createOrbitCamera", &glwrapper::createOrbitCamera, "Creates an orbit-type camera", DefDefaultCameraParams );\
                                    m.def( "createFollowCamera", &glwrapper::createFollowCamera, "Creates a follow-type camera", DefDefaultCameraParams );\
                                    m.def( "getCurrentCamera", &glwrapper::getCurrentCamera );\
                                    m.def( "changeToCameraByName", &glwrapper::changeToCameraByName );