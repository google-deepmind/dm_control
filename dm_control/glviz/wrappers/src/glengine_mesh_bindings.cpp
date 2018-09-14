
#include <glengine_mesh_bindings.h>

namespace py = pybind11;

namespace glwrapper
{

    Mesh::Mesh()
    {
        m_glMeshRef = NULL;
    }

    Mesh::~Mesh()
    {
        m_glMeshRef = NULL;
    }

    void Mesh::setMeshReference( engine::LMesh* pMesh )
    {
        m_glMeshRef = pMesh;
    }

    // Python exposed-API **************************************

    void Mesh::setX( float x ) { m_glMeshRef->pos.x = x; }
    void Mesh::setY( float y ) { m_glMeshRef->pos.y = y; }
    void Mesh::setZ( float z ) { m_glMeshRef->pos.z = z; }

    void Mesh::setPosition( float x, float y, float z )
    {
        m_glMeshRef->pos.x = x;
        m_glMeshRef->pos.y = y;
        m_glMeshRef->pos.z = z;
    }

    void Mesh::setPosition( py::array_t<float> posArray )
    {
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

    float Mesh::getX() { return m_glMeshRef->pos.x; }
    float Mesh::getY() { return m_glMeshRef->pos.y; }
    float Mesh::getZ() { return m_glMeshRef->pos.z; }

    py::array_t<float> Mesh::getPosition()
    {
        auto _result = py::array_t<float>( 3 );
        auto _buffer = _result.request();

        auto _ptr = ( float* ) _buffer.ptr;
        _ptr[0] = m_glMeshRef->pos.x;
        _ptr[1] = m_glMeshRef->pos.y;
        _ptr[2] = m_glMeshRef->pos.z;

        return _result;
    }

    // *********************************************************
}