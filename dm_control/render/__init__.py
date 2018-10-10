# Copyright 2017-2018 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""OpenGL context management for rendering MuJoCo scenes.

The `Renderer` class will use one of the following rendering APIs, in order of
descending priority: GLFW > OSMesa.

Rendering support can be disabled globally by setting the
`DISABLE_MUJOCO_RENDERING` environment variable before launching the Python
interpreter. This allows the MuJoCo bindings in `dm_control.mujoco` to be used
on platforms where an OpenGL context cannot be created. Attempting to render
when rendering has been disabled will result in a `RuntimeError`.
"""

import os
DISABLED = bool(os.environ.get('DISABLE_MUJOCO_RENDERING', ''))
del os

DISABLED_MESSAGE = (
    'Rendering support has been disabled by the `DISABLE_MUJOCO_RENDERING` '
    'environment variable')

# pylint: disable=g-import-not-at-top
BACKEND = None

if not DISABLED:

  if not BACKEND:
    try:
      from dm_control.render.glfw_renderer import GLFWContext as Renderer
      BACKEND = 'glfw'
    except ImportError:
      pass

  if not BACKEND:
    try:
      from dm_control.render.pyopengl.osmesa_renderer import OSMesaContext as Renderer
      BACKEND = 'osmesa'
    except ImportError:
      pass

  if not BACKEND:

    def Renderer(*args, **kwargs):  # pylint: disable=function-redefined,invalid-name
      del args, kwargs
      raise RuntimeError(
          'No OpenGL rendering backend is available. To use '
          '`dm_control.mujoco` without rendering support, set the '
          '`DISABLE_MUJOCO_RENDERING` environment variable before '
          'launching your interpreter.')
