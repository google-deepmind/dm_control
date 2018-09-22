
#include <glengine_debugsystem_bindings.h>


namespace py = pybind11;


namespace glwrapper
{

    void drawLine( float x1, float y1, float z1, 
                   float x2, float y2, float z2,
                   float r, float g, float b )
    {
        auto _p1 = engine::LVec3( x1, y1, z1 );
        auto _p2 = engine::LVec3( x2, y2, z2 );
        auto _color = engine::LVec3( r, g, b );

        engine::DebugSystem::drawLine( _p1, _p2, _color );
    }

    void drawLine( py::array_t<float> p1, 
                   py::array_t<float> p2,
                   py::array_t<float> color )
    {
        auto _buff1 = p1.request();
        auto _buff2 = p2.request();
        auto _buffc = color.request();

        if ( _buff1.size != 3 || _buff2.size != 3 || _buffc.size != 3 )
        {
            std::cout << "WARNING> drawline requires arrays with "
                      << "3 elements as arguments. Check usage" << std::endl;
            return;
        }

        auto _ptr1 = ( float* ) _buff1.ptr;
        auto _ptr2 = ( float* ) _buff2.ptr;
        auto _ptrc = ( float* ) _buffc.ptr;

        auto _p1 = engine::LVec3( _ptr1[0], _ptr1[1], _ptr1[2] );
        auto _p2 = engine::LVec3( _ptr2[0], _ptr2[1], _ptr2[2] );
        auto _color = engine::LVec3( _ptrc[0], _ptrc[1], _ptrc[2] );

        engine::DebugSystem::drawLine( _p1, _p2, _color );
    }
}