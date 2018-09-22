
#include <glengine_mesh_bindings.h>

namespace py = pybind11;

namespace glwrapper
{
    Mesh::Mesh( engine::LMesh* pMesh )
    {
        m_glMeshRef = pMesh;
    }

    Mesh::Mesh()
    {
        m_glMeshRef = NULL;
    }

    Mesh::~Mesh()
    {
        // @TODO: Enable deletion request to the engine
        m_glMeshRef = NULL;
    }

    void Mesh::setMeshReference( engine::LMesh* pMesh )
    {
        m_glMeshRef = pMesh;
    }

    // Python exposed-API **************************************

    void Mesh::setX( float x ) 
    { 
        if ( m_glMeshRef == NULL ) { return; }

        m_glMeshRef->pos.x = x; 
    }
    void Mesh::setY( float y ) 
    { 
        if ( m_glMeshRef == NULL ) { return; }

        m_glMeshRef->pos.y = y; 
    }
    void Mesh::setZ( float z ) 
    { 
        if ( m_glMeshRef == NULL ) { return; }

        m_glMeshRef->pos.z = z; 
    }
    void Mesh::setPosition( float x, float y, float z )
    {
        if ( m_glMeshRef == NULL ) { return; }

        m_glMeshRef->pos.x = x;
        m_glMeshRef->pos.y = y;
        m_glMeshRef->pos.z = z;
    }
    void Mesh::setPosition( py::array_t<float> posArray )
    {
        if ( m_glMeshRef == NULL ) { return; }

        auto _buff = posArray.request();
        if ( _buff.size != 3 )
        {
            std::cout << "WARNING> Should pass an array with 3 elements" << std::endl;
            return;
        }

        auto _ptr = ( float* ) _buff.ptr;
        m_glMeshRef->pos.x = _ptr[0];
        m_glMeshRef->pos.y = _ptr[1];
        m_glMeshRef->pos.z = _ptr[2];
    }

    float Mesh::getX() 
    {
        if ( m_glMeshRef == NULL ) { return 0.0f; }

        return m_glMeshRef->pos.x; 
    }
    float Mesh::getY() 
    { 
        if ( m_glMeshRef == NULL ) { return 0.0f; }

        return m_glMeshRef->pos.y; 
    }
    float Mesh::getZ() 
    { 
        if ( m_glMeshRef == NULL ) { return 0.0f; }

        return m_glMeshRef->pos.z; 
    }
    py::array_t<float> Mesh::getPosition()
    {
        auto _result = py::array_t<float>( 3 );
        auto _buffer = _result.request();

        if ( m_glMeshRef != NULL )
        {
            auto _ptr = ( float* ) _buffer.ptr;
            _ptr[0] = m_glMeshRef->pos.x;
            _ptr[1] = m_glMeshRef->pos.y;
            _ptr[2] = m_glMeshRef->pos.z;
        }

        return _result;
    }

    void Mesh::setBuiltInTexture( const string& textureId )
    {
        if ( m_glMeshRef == NULL ) { return; }

        auto _texture = engine::LAssetsManager::getBuiltInTexture( textureId );
        if ( _texture == NULL )
        {
            std::cout << "WARNING> Tried to set non-existent built in texture" << std::endl;
            return;
        }

        m_glMeshRef->addTexture( _texture );
    }

    void Mesh::setColor( float r, float g, float b )
    {
        if ( m_glMeshRef == NULL ) { return; }

        auto _material = m_glMeshRef->getMaterial();
        _material->ambient.x = r;
        _material->ambient.y = g;
        _material->ambient.z = b;

        _material->diffuse.x = r;
        _material->diffuse.y = g;
        _material->diffuse.z = b;

        _material->specular.x = r;
        _material->specular.y = g;
        _material->specular.z = b;
    }

    void Mesh::setRotation( py::array_t<float> mat )
    {
        if ( m_glMeshRef == NULL ) { return; }

        auto _buffer = mat.request();
        auto _ptr = ( float* ) _buffer.ptr;

        m_glMeshRef->rotation[0][0] = _ptr[0];//_ptr[0];
        m_glMeshRef->rotation[1][0] = _ptr[1];//_ptr[3];
        m_glMeshRef->rotation[2][0] = _ptr[2];//_ptr[6];

        m_glMeshRef->rotation[0][1] = _ptr[3];//_ptr[1];
        m_glMeshRef->rotation[1][1] = _ptr[4];//_ptr[4];
        m_glMeshRef->rotation[2][1] = _ptr[5];//_ptr[7];

        m_glMeshRef->rotation[0][2] = _ptr[6];//_ptr[2];
        m_glMeshRef->rotation[1][2] = _ptr[7];//_ptr[5];
        m_glMeshRef->rotation[2][2] = _ptr[8];//_ptr[8];
    }


    // *********************************************************
}