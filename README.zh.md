<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

# `dm_control`：Google DeepMind 物理动力学仿真基础设施

`dm_control` 是 Google DeepMind 专为物理仿真与强化学习（Reinforcement Learning，RL）打造的开源软件栈，基于 **MuJoCo** 物理引擎构建。

本软件包的**交互式入门教程**已作为 Colaboratory 笔记本提供：
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/dm_control/blob/main/tutorial.ipynb)

## 核心概览 (Overview)

本工具包由以下几大“核心”组件构成：

-   [`dm_control.mujoco`]: 提供对 MuJoCo 物理引擎的 Python 绑定库。

-   [`dm_control.suite`]: 由 MuJoCo 物理引擎驱动的一套标准 Python 强化学习连续控制环境套件。

-   [`dm_control.viewer`]: 交互式环境可视化查看器。

此外，还提供以下扩展组件以构建更为复杂的控制任务：

-   [`dm_control.mjcf`]: 支持在 Python 中程序化组合与修改 MuJoCo MJCF 模型的工具库。

-   `dm_control.composer`: 支持基于可复用、自包含组件定义丰富复杂 RL 环境的组合库。

-   [`dm_control.locomotion`]: 用于自定义足式运动与位移任务的扩展库。

-   [`dm_control.locomotion.soccer`]: 多智能体足球竞技控制任务套件。

如果您在研究或工程中使用了本软件包，请引用我们的配套[学术论文 (Publication)][publication]：

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

## 安装指南 (Installation)

可以通过 PyPI 直接安装 `dm_control`：

```sh
pip install dm_control
```

> **注意**：**`dm_control` 目前不支持以“可编辑模式（editable mode）”安装**（即 `pip install -e`）。
>
> 尽管 `dm_control` 已大部分迁移至通过 `mujoco` 官方包提供的 pybind11 绑定，但目前仍依赖若干通过 MuJoCo C 头文件自动生成的遗留组件，其构建方式与 editable 模式不兼容。若尝试以 editable 模式安装 `dm_control`，在导入时将触发类似如下错误：
>
> ```
> ImportError: cannot import name 'constants' from partially initialized module 'dm_control.mujoco.wrapper.mjbindings' ...
> ```
>
> 解决方法是先执行 `pip uninstall dm_control`，然后重新安装且**不要**携带 `-e` 参数。

## 版本规范 (Versioning)

从 1.0.0 版本开始，本项目全面采用语义化版本规范（Semantic Versioning）。

在 1.0.0 版本之前，`dm_control` 的 Python 包版本号采用 `0.0.N` 形式，其中 `N` 为内部修订号，在每次 Git Commit 时递增任意数值。

如果您希望直接从代码仓库安装未经发布的最新主分支版本，可运行以下命令：
```sh
pip install git+https://github.com/google-deepmind/dm_control.git
```

## 渲染后端配置 (Rendering)

MuJoCo Python 绑定原生支持三种不同的 OpenGL 渲染后端：
**EGL**（无头模式 headless，硬件加速）、**GLFW**（窗口化 windowed，硬件加速）以及 **OSMesa**（纯 CPU 软件光栅化渲染）。使用 `dm_control` 渲染环境画面时，系统必须至少具备其中一种可用后端。

*   **带窗口系统的硬件加速渲染**：通过 GLFW 与 GLEW 实现。在 Linux 系统上，可以通过系统包管理器进行安装。例如在 Debian 和 Ubuntu 上可执行：
    ```sh
    sudo apt-get install libglfw3 libglew2.0
    ```
    请注意：
    -   [`dm_control.viewer`] 查看器**只能**与 GLFW 配合使用。
    -   GLFW 在无桌面环境的纯无头（Headless）服务器上无法运行。

*   **“无头模式 (Headless)”硬件加速渲染**（即没有 X11 等窗口系统的远程服务器）：需要 EGL 驱动支持 [EXT_platform_device] 扩展。近期的 NVIDIA 驱动均原生支持。同时系统需要安装 GLEW。在 Debian 和 Ubuntu 上，可通过如下命令安装：
    ```sh
    sudo apt-get install libglew2.0
    ```

*   **纯软件渲染**：需要 GLX 与 OSMesa 支持。在 Debian 和 Ubuntu 上可通过如下命令安装：
    ```sh
    sudo apt-get install libgl1-mesa-glx libosmesa6
    ```

默认情况下，`dm_control` 会按优先级尝试使用 **GLFW -> EGL -> OSMesa**。
您也可以通过设置环境变量 `MUJOCO_GL=` 为 `"glfw"`、`"egl"` 或 `"osmesa"` 来显式指定渲染后端。当使用 EGL 渲染时，还可以通过将环境变量 `MUJOCO_EGL_DEVICE_ID=` 设置为目标 GPU 编号来指定用于渲染的物理 GPU。

## macOS 平台 Homebrew 用户补充说明

1.  上述通过 `pip` 安装的流程在 macOS 上同样适用，前提是必须使用由 Homebrew 安装的 Python 解释器（而非 macOS 系统自带的 Python）。

2.  在运行前，需要将 `DYLD_LIBRARY_PATH` 环境变量补充 GLFW 动态库的路径。可通过在终端执行以下命令设置：
    ```sh
    export DYLD_LIBRARY_PATH=$(brew --prefix)/lib:$DYLD_LIBRARY_PATH
    ```

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

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（@JasonYeYuhe）翻译维护，最后同步更新于 2026年09月27日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
