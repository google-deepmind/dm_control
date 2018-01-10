# Copyright 2017 The dm_control Authors.
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
descending priority: EGL > GLFW > OSMesa.
"""

# pylint: disable=g-import-not-at-top
try:
  from dm_control.render.glfw_renderer import GLFWContext as _GLFWRenderer
except (ImportError, IOError):
  _GLFWRenderer = None
try:
  from dm_control.render.egl_renderer import EGLContext as _EGLRenderer
except ImportError:
  _EGLRenderer = None
try:
  from dm_control.render.osmesa_renderer import OSMesaContext as _OSMesaRenderer
except ImportError:
  _OSMesaRenderer = None
# pylint: enable=g-import-not-at-top

# pylint: disable=invalid-name
if _EGLRenderer:
  Renderer = _EGLRenderer
elif _GLFWRenderer:
  Renderer = _GLFWRenderer
elif _OSMesaRenderer:
  Renderer = _OSMesaRenderer
else:
  # This is a workaround that allows imports from `dm_control.render` to succeed
  # even when there is no rendering API available. We need this in order to run
  # integration tests on headless servers.

  def Renderer(*args, **kwargs):
    del args, kwargs  # Unused.
    raise ImportError('No OpenGL rendering backend could be imported.')

# pylint: enable=invalid-name


