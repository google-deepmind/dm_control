# MuJoCo Wrapper

This package contains Python bindings for the [MuJoCo physics engine][1] using
[`ctypes`][2]. The bindings and some higher-level wrapper code are automatically
generated from MuJoCo's header files by `dm_control/autowrap/autowrap.py`.

The main entry point for users of the generated bindings is
[`dm_control.mujoco`][3].

[1]: http://mujoco.org/
[2]: https://docs.python.org/2/library/ctypes.html
[3]: ../README.md
