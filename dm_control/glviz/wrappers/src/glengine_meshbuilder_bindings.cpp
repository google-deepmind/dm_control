
#include <glengine_meshbuilder_bindings.h>

namespace py = pybind11;

namespace glwrapper
{

    Mesh* createSphere( float radius )
    {
        auto _mesh = engine::LMeshBuilder::createSphere( radius );
        auto _scene = engine::LApp::GetInstance()->scene();
        _scene->addRenderable( _mesh );
        return new Mesh( _mesh );
    }

    Mesh* createBox( float width, float height, float depth )
    {
        auto _mesh = engine::LMeshBuilder::createBox( width, height, depth );
        auto _scene = engine::LApp::GetInstance()->scene();
        _scene->addRenderable( _mesh );
        return new Mesh( _mesh );
    }

    Mesh* createCapsule( float radius, float height )
    {
        auto _mesh = engine::LMeshBuilder::createCapsule( radius, height );
        auto _scene = engine::LApp::GetInstance()->scene();
        _scene->addRenderable( _mesh );
        return new Mesh( _mesh );
    }

    Mesh* createPlane( float width, float depth, float texRangeWidth, float texRangeDepth )
    {
        auto _mesh = engine::LMeshBuilder::createPlane( width, depth, texRangeWidth, texRangeDepth );
        auto _scene = engine::LApp::GetInstance()->scene();
        _scene->addRenderable( _mesh );
        return new Mesh( _mesh );
    }

}