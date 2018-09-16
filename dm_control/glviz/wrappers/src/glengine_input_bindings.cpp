
#include <glengine_input_bindings.h>

namespace py = pybind11;

namespace glwrapper
{

    bool isKeyDown( int key )
    {
        return engine::InputSystem::isKeyDown( key );
    }

    bool isMouseDown( int button )
    {
        return engine::InputSystem::isMouseDown( button );
    }

    py::array_t<float> getCursorPosition()
    {
        auto _result = py::array_t<float>( 2 );
        auto _buffer = _result.request();

        auto _pos = engine::InputSystem::getCursorPosition();
        auto _ptr = ( float* ) _buffer.ptr;

        _ptr[0] = _pos.x;
        _ptr[1] = _pos.y;

        return _result;
    }

}