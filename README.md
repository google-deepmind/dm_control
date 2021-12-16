# `dm_control`: DeepMind Infrastructure for Physics-Based Simulation.

DeepMind's software stack for physics-based simulation and Reinforcement
Learning environments, using MuJoCo physics.

An **introductory tutorial** for this package is available as a Colaboratory
notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/deepmind/dm_control/blob/master/tutorial.ipynb)

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

If you use this package, please cite our accompanying [tech report]:

```
@misc{tassa2020dmcontrol,
    title={dm_control: Software and Tasks for Continuous Control},
    author={Yuval Tassa and Saran Tunyasuvunakool and Alistair Muldal and
            Yotam Doron and Siqi Liu and Steven Bohez and Josh Merel and
            Tom Erez and Timothy Lillicrap and Nicolas Heess},
    year={2020},
    eprint={2006.12983},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```

## Requirements and Installation

`dm_control` is regularly tested on Ubuntu 16.04 against the following Python
versions:

*   3.7
*   3.8
*   3.9

Various people have been successful in getting `dm_control` to work on other
Linux distros, macOS, and Windows. We do not provide active support for these,
but will endeavour to answer questions on a best-effort basis.

Follow these steps to install `dm_control`:

1.  Download MuJoCo 2.1.1 from the
    [Releases page on the MuJoCo GitHub repository]. MuJoCo must be installed
    before `dm_control`, since `dm_control`'s install script generates Python
    [`ctypes`] bindings based on MuJoCo's header files. By default, `dm_control`
    assumes that MuJoCo is installed via the following instructions:

    -   On Linux, extract the tarball into `~/.mujoco`.
    -   On Windows, extract the zip archive into either `%HOMEPATH%\MuJoCo` or
        `%PUBLIC%\MuJoCo`.
    -   On macOS, either place `MuJoCo.app` into `/Applications`, or place
        `MuJoCo.Framework` into `~/.mujoco`.

2.  Install the `dm_control` Python package by running `pip install dm_control`.
    We recommend `pip install`ing into a `virtualenv`, or with the `--user` flag
    to avoid interfering with system packages. At installation time,
    `dm_control` looks for the MuJoCo headers at the paths described in Step 1
    by default, however this path can be configured with the `headers-dir`
    command line argument.

3.  If the shared library provided by MuJoCo (i.e. `libmujoco.so.2.1.1` or
    `libmujoco.2.1.1.dylib` or `mujoco.dll`) is installed at a non-default path,
    specify its location using the `MJLIB_PATH` environment variable. This
    environment variable should be set to the full path to the library file
    itself, e.g. `export MJLIB_PATH=/path/to/libmujoco.so.2.1.1`.

## Versioning

`dm_control` is released on a rolling basis: the latest commit on the `master`
branch of our GitHub repository represents our latest release. Our Python
package is versioned `0.0.N`, where `N` is the number that appears in the
`PiperOrigin-RevId` field of the commit message. We always ensure that `N`
strictly increases between a parent commit and its children. We do not upload
all versions to PyPI, and occasionally the latest version on PyPI may lag behind
the latest commit on GitHub. Should this happen, you can still install the
newest version available by running `pip install
git+git://github.com/deepmind/dm_control.git`.

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
setting the environment variable `EGL_DEVICE_ID=` to the target GPU ID.

## Additional instructions for Homebrew users on macOS

1.  The above instructions using `pip` should work, provided that you use a
    Python interpreter that is installed by Homebrew (rather than the
    system-default one).

2.  Before running, the `DYLD_LIBRARY_PATH` environment variable needs to be
    updated with the path to the GLFW library. This can be done by running
    `export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH`.

[EXT_platform_device]: https://www.khronos.org/registry/EGL/extensions/EXT/EGL_EXT_platform_device.txt
[Releases page on the MuJoCo GitHub repository]: https://github.com/deepmind/mujoco/releases
[MuJoCo website]: https://mujoco.org/
[tech report]: https://arxiv.org/abs/2006.12983
[`ctypes`]: https://docs.python.org/3/library/ctypes.html
[`dm_control.mjcf`]: dm_control/mjcf/README.md
[`dm_control.mujoco`]: dm_control/mujoco/README.md
[`dm_control.suite`]: dm_control/suite/README.md
[`dm_control.viewer`]: dm_control/viewer/README.md
[`dm_control.locomotion`]: dm_control/locomotion/README.md
[`dm_control.locomotion.soccer`]: dm_control/locomotion/soccer/README.md
