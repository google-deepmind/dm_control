
#include <glengine_camera_bindings.h>

namespace py = pybind11;

using namespace std;

namespace glwrapper
{
    // Camera-wrapper base class ************************************************************

    Camera::Camera( const string& name,
                    py::array_t<float> position,
                    py::array_t<float> targetPoint,
                    int worldUpId,
                    float fov,
                    float aspectRatio,
                    float zNear, float zFar )
    {
        m_cameraPtr = NULL;
    }

    Camera::Camera( engine::LICamera* ptrCamera )
    {
        m_cameraPtr = ptrCamera;
    }

    Camera::~Camera()
    {
        // @TODO: Check if the camera is in the scene ...
        // so that we can remove it from there
        m_cameraPtr = NULL;
    }

    void Camera::setPosition( py::array_t<float> position )
    {
        if ( m_cameraPtr == NULL )
        {
            std::cout << "WARNING> This camera has no engine-camera associated" << std::endl;
            return;
        }

        auto _vec3 = numpyToVec3( position );
        m_cameraPtr->setPosition( _vec3 );
    }

    py::array_t<float> Camera::getPosition()
    {
        if ( m_cameraPtr == NULL )
        {
            std::cout << "WARNING> This camera has no engine-camera associated" << std::endl;
            return vec3ToNumpy( engine::LVec3() );
        }

        auto _vec3 = m_cameraPtr->getPosition();
        auto _npArray = vec3ToNumpy( _vec3 );

        return _npArray;
    }

    py::array_t<float> Camera::getTargetDir()
    {
        if ( m_cameraPtr == NULL )
        {
            std::cout << "WARNING> This camera has no engine-camera associated" << std::endl;
            return vec3ToNumpy( engine::LVec3() );
        }

        auto _vec3 = m_cameraPtr->getTargetDir();
        auto _npArray = vec3ToNumpy( _vec3 );

        return _npArray;
    }

    py::array_t<float> Camera::getTargetPoint()
    {
        if ( m_cameraPtr == NULL )
        {
            std::cout << "WARNING> This camera has no engine-camera associated" << std::endl;
            return vec3ToNumpy( engine::LVec3() );
        }

        auto _vec3 = m_cameraPtr->getTargetPoint();
        auto _npArray = vec3ToNumpy( _vec3 );

        return _npArray;
    }

    string Camera::name()
    {
        if ( m_cameraPtr == NULL )
        {
            std::cout << "WARNING> This camera has no engine-camera associated" << std::endl;
            return "none";
        }

        return m_cameraPtr->name();
    }

    string Camera::type()
    {
        if ( m_cameraPtr == NULL )
        {
            std::cout << "WARNING> This camera has no engine-camera associated" << std::endl;
            return "none";
        }

        return m_cameraPtr->type();
    }
    // **************************************************************************************

    // Fixed camera class *******************************************************************

    CameraFixed::CameraFixed( const string& name,
                              py::array_t<float> position,
                              py::array_t<float> targetPoint,
                              int worldUpId,
                              float fov,
                              float aspectRatio,
                              float zNear, float zFar )
        : Camera( name, position, targetPoint, worldUpId, fov, aspectRatio, zNear, zFar )
    {
        m_cameraPtr = new engine::LFixedCamera3d( name,
                                                  numpyToVec3( position ),
                                                  numpyToVec3( targetPoint ),
                                                  worldUpId,
                                                  fov, aspectRatio,
                                                  zNear, zFar );
    }

    // Fps-camera class *********************************************************************

    CameraFps::CameraFps( const string& name,
                          py::array_t<float> position,
                          py::array_t<float> targetPoint,
                          int worldUpId,
                          float fov,
                          float aspectRatio,
                          float zNear, float zFar )
        : Camera( name, position, targetPoint, worldUpId, fov, aspectRatio, zNear, zFar )
    {
        m_cameraPtr = new engine::LFpsCamera( name,
                                              numpyToVec3( position ),
                                              numpyToVec3( targetPoint ),
                                              worldUpId,
                                              fov, aspectRatio,
                                              zNear, zFar );
    }

    // **************************************************************************************

    // Orbit-camera class *******************************************************************

    CameraOrbit::CameraOrbit( const string& name,
                              py::array_t<float> position,
                              py::array_t<float> targetPoint,
                              int worldUpId,
                              float fov,
                              float aspectRatio,
                              float zNear, float zFar )
        : Camera( name, position, targetPoint, worldUpId, fov, aspectRatio, zNear, zFar )
    {
        m_cameraPtr = new engine::LFixedCamera3d( name,
                                        numpyToVec3( position ),
                                        numpyToVec3( targetPoint ),
                                        worldUpId,
                                        fov, aspectRatio,
                                        zNear, zFar );
    }

    // **************************************************************************************

    // Follow-camera class ******************************************************************

    CameraFollow::CameraFollow( const string& name,
                                py::array_t<float> position,
                                py::array_t<float> targetPoint,
                                int worldUpId,
                                float fov,
                                float aspectRatio,
                                float zNear, float zFar )
        : Camera( name, position, targetPoint, worldUpId, fov, aspectRatio, zNear, zFar )
    {
        m_cameraPtr = new engine::LFollowCamera( name,
                                                 numpyToVec3( position ),
                                                 numpyToVec3( targetPoint ),
                                                 worldUpId,
                                                 fov, aspectRatio,
                                                 zNear, zFar );
    }

    void CameraFollow::setFollowReference( Mesh* pMeshWrapperObj )
    {
        reinterpret_cast<engine::LFollowCamera*>( m_cameraPtr )->setMeshReference( pMeshWrapperObj->ptr() );
    }

    // **************************************************************************************

    Camera* _createCamera( int type, const string& name,
                           py::array_t<float> position,
                           py::array_t<float> targetPoint,
                           int worldUpId,
                           float fov, float zNear, float zFar )
    {
        auto _app = engine::LApp::GetInstance();
        auto _scene = _app->scene();
        auto _window = _app->window();

        float _aspectRatio = ( (float)_window->width() ) / _window->height();

        Camera* _camera = NULL;

        if ( type == CAMERA_TYPE_FIXED )
        {
            // std::cout << "INFO> created fixed camera" << std::endl;
            _camera = new CameraFixed( name, position, targetPoint, worldUpId,
                                       fov, _aspectRatio,
                                       zNear, zFar );
        }
        else if ( type == CAMERA_TYPE_FPS )
        {
            // std::cout << "INFO> created fps camera" << std::endl;
            _camera = new CameraFps( name, position, targetPoint, worldUpId,
                                     fov, _aspectRatio,
                                     zNear, zFar );
        }
        else if ( type == CAMERA_TYPE_ORBIT )
        {
            // std::cout << "INFO> created orbit camera" << std::endl;
            _camera = new CameraOrbit( name, position, targetPoint, worldUpId,
                                       fov, _aspectRatio,
                                       zNear, zFar );
        }
        else if ( type == CAMERA_TYPE_FOLLOW )
        {
            // std::cout << "INFO> created follow camera" << std::endl;
            _camera = new CameraFollow( name, position, targetPoint, worldUpId,
                                        fov, _aspectRatio,
                                        zNear, zFar );
        }
        else
        {
            // If non givem, make just a fixed camera
            std::cout << "WARNING> Wrong type of camera selected" << std::endl;
            _camera = new CameraFixed( name, position, targetPoint, worldUpId,
                                       fov, _aspectRatio,
                                       zNear, zFar );
        }

        _scene->addCamera( _camera->ptr() );

        return _camera;
    }

    CameraFixed* createFixedCamera( const string& name,
                                    py::array_t<float> position,
                                    py::array_t<float> targetPoint,
                                    int worldUpId,
                                    float fov, float zNear, float zFar )
    {
        auto _camera = _createCamera( CAMERA_TYPE_FIXED, name, 
                                      position, targetPoint, worldUpId,
                                      fov, zNear, zFar );

        return reinterpret_cast< CameraFixed* >( _camera );
    }

    CameraFps* createFpsCamera( const string& name,
                                py::array_t<float> position,
                                py::array_t<float> targetPoint,
                                int worldUpId,
                                float fov, float zNear, float zFar )
    {
        auto _camera = _createCamera( CAMERA_TYPE_FPS, name, 
                                      position, targetPoint, worldUpId,
                                      fov, zNear, zFar );

        return reinterpret_cast< CameraFps* >( _camera );
    }

    CameraOrbit* createOrbitCamera( const string& name,
                                    py::array_t<float> position,
                                    py::array_t<float> targetPoint,
                                    int worldUpId,
                                    float fov, float zNear, float zFar )
    {
        auto _camera = _createCamera( CAMERA_TYPE_ORBIT, name, 
                                      position, targetPoint, worldUpId,
                                      fov, zNear, zFar );
        
        return reinterpret_cast< CameraOrbit* >( _camera );
    }

    CameraFollow* createFollowCamera( const string& name,
                                      py::array_t<float> position,
                                      py::array_t<float> targetPoint,
                                      int worldUpId,
                                      float fov, float zNear, float zFar )
    {
        auto _camera = _createCamera( CAMERA_TYPE_FOLLOW, name, 
                                      position, targetPoint, worldUpId,
                                      fov, zNear, zFar );

        return reinterpret_cast< CameraFollow* >( _camera );
    }

    Camera* getCurrentCamera()
    {
        auto _app = engine::LApp::GetInstance();
        auto _scene = _app->scene();
        auto _cameraPtr = _scene->getCurrentCamera();

        return new Camera( _cameraPtr );
    }

    void changeToCameraByName( const string& name )
    {
        auto _app = engine::LApp::GetInstance();
        auto _scene = _app->scene();

        _scene->changeToCameraByName( name );
    }

}