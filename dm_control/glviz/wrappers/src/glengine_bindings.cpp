
#include <pybind11/pybind11.h>

#include <glengine_bindings.h>

#include <LApp.h>
#include <LMeshBuilder.h>

namespace py = pybind11;

void init()
{
    auto _app = engine::LApp::GetInstance();
    auto _scene = _app->scene();

    // make a sample camera
    auto _camera = new engine::LFpsCamera( "main", 
                                           engine::LVec3( 1.0f, 2.0f, 1.0f ),
                                           engine::LVec3( 0.0f, 0.0f, 0.0f ),
                                           engine::LICamera::UP_Z );
    // auto _camera = new engine::LFixedCamera3d( "main",
    //                                            engine::LVec3( 2.0f, 4.0f, 2.0f ),
    //                                            engine::LVec3( 0.0f, 0.0f, 0.0f ),
    //                                            engine::LICamera::UP_Z );
    // make a sample light source
    auto _light = new engine::LLightDirectional( engine::LVec3( 0.2, 0.2, 0.2 ), 
                                                 engine::LVec3( 0.8, 0.8, 0.8 ),
                                                 engine::LVec3( 0.15, 0.15, 0.15 ), 
                                                 0, 
                                                 engine::LVec3( -1, -1, 0 ) );

    // add these components to the scene
    _scene->addCamera( _camera );
    _scene->addLight( _light );
}

void update()
{
    auto _app = engine::LApp::GetInstance();
    _app->update();
}

bool isActive()
{
    auto _app = engine::LApp::GetInstance();
    return _app->isActive();
}

glwrapper::Mesh* createMesh()
{
    auto _app = engine::LApp::GetInstance();
    auto _scene = _app->scene();

    auto _mesh = engine::LMeshBuilder::createBox( 0.5f, 0.5f, 0.5f );
    _scene->addRenderable( _mesh );

    auto _meshWrapper = new glwrapper::Mesh();
    _meshWrapper->setMeshReference( _mesh );

    return _meshWrapper;
}

PYBIND11_MODULE( enginewrapper, m )
{
    // wrapper control methods
    m.def( "init", &init );
    m.def( "update", &update );
    m.def( "isActive", &isActive );
    m.def( "createMesh", &createMesh, py::return_value_policy::automatic );

    // mesh bindings
    GLENGINE_MESH_BINDINGS(m)
    // meshbuilder bindings
    GLENGINE_MESHBUILDER_BINDINGS(m)
    // debug system bindings
    GLENGINE_DEBUGSYSTEM_BINDINGS(m)
    // input system bindings
    GLENGINE_INPUT_BINDINGS(m)
    // camera bindings
    GLENGINE_CAMERA_BINDINGS(m)

    m.attr( "__version__" ) = "dev";
}