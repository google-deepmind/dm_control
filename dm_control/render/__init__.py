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

By default, the `Renderer` class will try to load one of the following rendering
APIs, in descending order of priority: EGL > GLFW > OSMesa.

It is also possible to select a specific backend by setting the `MUJOCO_GL=`
environment variable to 'egl', 'glfw' or 'osmesa'.
"""

import collections
import os

BACKEND = os.environ.get('MUJOCO_GL')


# pylint: disable=g-import-not-at-top
def _import_egl():
  from dm_control.render.pyopengl.egl_renderer import EGLContext
  return EGLContext


def _import_glfw():
  from dm_control.render.glfw_renderer import GLFWContext
  return GLFWContext


def _import_osmesa():
  from dm_control.render.pyopengl.osmesa_renderer import OSMesaContext
  return OSMesaContext
# pylint: enable=g-import-not-at-top

_ALL_RENDERERS = collections.OrderedDict([
    ('egl', _import_egl),
    ('glfw', _import_glfw),
    ('osmesa', _import_osmesa),
])


if BACKEND is not None:
  # If a backend was specified, try importing it and error if unsuccessful.
  try:
    import_func = _ALL_RENDERERS[BACKEND]
  except KeyError:
    raise RuntimeError('MUJOCO_GL= must be one of {!r}, got {!r}.'
                       .format(_ALL_RENDERERS.keys(), BACKEND))
  Renderer = import_func()  # pylint: disable=invalid-name
else:
  # Otherwise try importing them in descending order of priority until
  # successful.
  for name, import_func in _ALL_RENDERERS.items():
    try:
      Renderer = import_func()
      BACKEND = name
      break
    except ImportError:
      pass
  if BACKEND is None:

    def Renderer(*args, **kwargs):  # pylint: disable=function-redefined,invalid-name
      del args, kwargs
      raise RuntimeError('No OpenGL rendering backend is available.')

USING_GPU = BACKEND in ('egl', 'glfw')
