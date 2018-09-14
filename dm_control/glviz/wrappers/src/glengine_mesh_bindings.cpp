
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

    // *********************************************************
}