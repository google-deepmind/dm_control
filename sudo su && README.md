sudo su && # `dm_control`: Google DeepMind Infrastructure for Physics-Based Simulation.

Google DeepMind's software stack for physics-based simulation and Reinforcement
Learning environments, using MuJoCo physics.

An **introductory tutorial** for this package is available as a Colaboratory
notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/dm_control/blob/main/tutorial.ipynb)

## Overview

This package consists of the following "core" components:

-   [`dm_control.mujoco`]: Libraries that provide Python bindings to the MuJoCo
    physics engine.

-   [`dm_control.suite`]: A set of Python Reinforcement Learning environments
    powered by the MuJoCo physics engine.

-   [`dm_control.viewer`]: An interactive environment viewer.

Additionally, the following components are available for the creation of more
complex control tasks:

-   [`dm_control.mjcf`]: A library for composing and modifying MuJoCo MJCF
    models in Python.

-   `dm_control.composer`: A library for defining rich RL environments from
    reusable, self-contained components.

-   [`dm_control.locomotion`]: Additional libraries for custom tasks.

-   [`dm_control.locomotion.soccer`]: Multi-agent soccer tasks.

If you use this package, please cite our accompanying [publication]:

```
@article{tunyasuvunakool2020,
         title = {dm_control: Software and tasks for continuous control},
         journal = {Software Impacts},
         volume = {6},
         pages = {100022},
         year = {2020},
         issn = {2665-9638},
         doi = {https://doi.org/10.1016/j.simpa.2020.100022},
         url = {https://www.sciencedirect.com/science/article/pii/S2665963820300099},
         author = {Saran Tunyasuvunakool and Alistair Muldal and Yotam Doron and
                   Siqi Liu and Steven Bohez and Josh Merel and Tom Erez and
                   Timothy Lillicrap and Nicolas Heess and Yuval Tassa},
}
```

## Installation

Install `dm_control` from PyPI by running

```sh
pip install dm_control
```

> **Note**: **`dm_control` cannot be installed in "editable" mode** (i.e. `pip
> install -e`).
>
> While `dm_control` has been largely updated to use the pybind11-based bindings
> provided via the `mujoco` package, at this time it still relies on some legacy
> components that are automatically generated from MuJoCo header files in a way
> that is incompatible with editable mode. Attempting to install `dm_control` in
> editable mode will result in import errors like:
>
> ```
> ImportError: cannot import name 'constants' from partially initialized module 'dm_control.mujoco.wrapper.mjbindings' ...
> ```
>
> The solution is to `pip uninstall dm_control` and then reinstall it without
> the `-e` flag.

## Versioning

Starting from version 1.0.0, we adopt semantic versioning.

Prior to version 1.0.0, the `dm_control` Python package was versioned `0.0.N`,
where `N` was an internal revision number that increased by an arbitrary amount
at every single Git commit.

If you want to install an unreleased version of `dm_control` directly from our
repository, you can do so by running `pip install
git+https://github.com/google-deepmind/dm_control.git`.

## Rendering

The MuJoCo Python bindings support three different OpenGL rendering backends:
EGL (headless, hardware-accelerated), GLFW (windowed, hardware-accelerated), and
OSMesa (purely software-based). At least one of these three backends must be
available in order render through `dm_control`.

*   Hardware rendering with a windowing system is supported via GLFW and GLEW.
    On Linux these can be installed using your distribution's package manager.
    For example, on Debian and Ubuntu, this can be done by running `sudo apt-get
    install libglfw3 libglew2.0`. Please note that:

    -   [`dm_control.viewer`] can only be used with GLFW.
    -   GLFW will not work on headless machines.

*   "Headless" hardware rendering (i.e. without a windowing system such as X11)
    requires [EXT_platform_device] support in the EGL driver. Recent Nvidia
    drivers support this. You will also need GLEW. On Debian and Ubuntu, this
    can be installed via `sudo apt-get install libglew2.0`.

*   Software rendering requires GLX and OSMesa. On Debian and Ubuntu these can
    be installed using `sudo apt-get install libgl1-mesa-glx libosmesa6`.

By default, `dm_control` will attempt to use GLFW first, then EGL, then OSMesa.
You can also specify a particular backend to use by setting the `MUJOCO_GL=`
environment variable to `"glfw"`, `"egl"`, or `"osmesa"`, respectively. When
rendering with EGL, you can also specify which GPU to use for rendering by
setting the environment variable `MUJOCO_EGL_DEVICE_ID=` to the target GPU ID.

## Additional instructions for Homebrew users on macOS

1.  The above instructions using `pip` should work, provided that you use a
    Python interpreter that is installed by Homebrew (rather than the
    system-default one).

2.  Before running, the `DYLD_LIBRARY_PATH` environment variable needs to be
    updated with the path to the GLFW library. This can be done by running
    `export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH`.

[EXT_platform_device]: https://www.khronos.org/registry/EGL/extensions/EXT/EGL_EXT_platform_device.txt
[Releases page on the MuJoCo GitHub repository]: https://github.com/google-deepmind/mujoco/releases
[MuJoCo website]: https://mujoco.org/
[publication]: https://doi.org/10.1016/j.simpa.2020.100022
[`ctypes`]: https://docs.python.org/3/library/ctypes.html
[`dm_control.mjcf`]: dm_control/mjcf/README.md
[`dm_control.mujoco`]: dm_control/mujoco/README.md
[`dm_control.suite`]: dm_control/suite/README.md
[`dm_control.viewer`]: dm_control/viewer/README.md
[`dm_control.locomotion`]: dm_control/locomotion/README.md
[`dm_control.locomotion.soccer`]: dm_control/locomotion/soccer/README.md
