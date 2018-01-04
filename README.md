# `dm_control`: The DeepMind Control Suite and Control Package

# ![all domains](all_domains.png)

This package contains:

- A set of Python Reinforcement Learning environments powered by the MuJoCo
  physics engine. See the `suite` subdirectory.

- Libraries that provide Python bindings to the MuJoCo physics engine.

If you use this package, please cite our accompanying accompanying [tech report](https://arxiv.org/abs/1801.00690).

## Installation and requirements

Follow these steps to install `dm_control`:

1. Download MuJoCo Pro 1.50 from the download page on the [MuJoCo website](http://www.mujoco.org/).
   MuJoCo Pro must be installed before `dm_control`, since `dm_control`'s
   install script generates Python [`ctypes`](https://docs.python.org/2/library/ctypes.html)
   bindings based on MuJoCo's header files. By default, `dm_control` assumes
   that the MuJoCo Zip archive is extracted as `~/.mujoco/mjpro150`.

2. Install the `dm_control` Python package by running
   `pip install git+git://github.com/deepmind/dm_control.git`
   (PyPI package coming soon) or by cloning the repository and running
   `pip install /path/to/dm_control/`
   At installation time, `dm_control` looks for the MuJoCo headers from Step 1
   in `~/.mujoco/mjpro150/include`, however this path can be configured with the
   `headers-dir` command line argument.

3. Install a license key for MuJoCo, required by `dm_control` at runtime. See
   the [MuJoCo license key page](https://www.roboti.us/license.html) for further
   details. By default, `dm_control` looks for the MuJoCo license key file at
   `~/.mujoco/mjkey.txt`.

4. If the license key (e.g. `mjkey.txt`) or the shared library provided by
   MuJoCo Pro (e.g. `libmujoco150.so` or `libmujoco150.dylib`) are installed at
   non-default paths, specify their locations using the `MJKEY_PATH` and
   `MJLIB_PATH` environment variables respectively.

## Additional instructions for Linux

Install `GLFW` and `GLEW` through your Linux distribution's package manager.
For example, on Debian and Ubuntu, this can be done by running
`sudo apt-get install libglfw3 libglew2.0`.

## Additional instructions for Homebrew users on macOS

1. The above instructions using `pip` should work, provided that you
   use a Python interpreter that is installed by Homebrew (rather than the
   system-default one).

2. To get OpenGL working, install the `glfw` package from Homebrew by running
   `brew install glfw`.

3. Before running, the `DYLD_LIBRARY_PATH` environment variable needs to be
   updated with the path to the GLFW library. This can be done by running
   `export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH`.

## Control Suite quickstart

```python
from dm_control import suite

# Load one task:
env = suite.load(domain_name="cartpole", task_name="swingup")

# Iterate over a task set:
for domain_name, task_name in suite.BENCHMARKING:
  env = suite.load(domain_name, task_name)

# Step through an episode and print out reward, discount and observation.
action_spec = env.action_spec()
time_step = env.reset()
while not time_step.last():
  action = np.random.uniform(action_spec.minimum,
                             action_spec.maximum,
                             size=action_spec.shape)
  time_step = env.step(action)
  print(time_step.reward, time_step.discount, time_step.observation)
```

See our [tech report](https://arxiv.org/abs/1801.00690) for further details.

## Illustration video

Below is a video montage of solved Control Suite tasks, with reward
visualisation enabled.

[![Video montage](https://img.youtube.com/vi/rAai4QzcYbs/0.jpg)](https://www.youtube.com/watch?v=rAai4QzcYbs)
