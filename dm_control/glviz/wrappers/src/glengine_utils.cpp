
#include <glengine_utils.h>

namespace glwrapper
{

    engine::LVec3 numpyToVec3( py::array_t<float> npArray )
    {
        // check if the size is the appropiate
        auto _buffer = npArray.request();
        auto _ptr = ( float* ) _buffer.ptr;
        if ( _buffer.size != 3 )
        {
            std::cout << "WARNING> Must pass an array with 3 elements" << std::endl;
            return engine::LVec3();
        }

        // extract the data
        engine::LVec3 _result;
        _result.x = _ptr[0];
        _result.y = _ptr[1];
        _result.z = _ptr[2];

        return _result;
    }

    py::array_t<float> vec3ToNumpy( const engine::LVec3& vec3 )
    {
        auto _result = py::array_t<float>( 3 );
        auto _buffer = _result.request();
        auto _ptr = ( float* ) _buffer.ptr;

        _ptr[0] = vec3.x;
        _ptr[1] = vec3.y;
        _ptr[2] = vec3.z;

        return _result;
    }

}