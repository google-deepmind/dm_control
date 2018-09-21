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

"""An OpenGL renderer backed by GLFW."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# Internal dependencies.


from dm_control.render import base
import six


# Re-raise any exceptions that occur during module import as `ImportError`s.
# This simplifies the conditional imports in `render/__init__.py`.
try:
  import glfw  # pylint: disable=g-import-not-at-top
except (ImportError, IOError, OSError) as exc:
  _, exc, tb = sys.exc_info()
  six.reraise(ImportError, ImportError(str(exc)), tb)
try:
  glfw.init()
except glfw.GLFWError as exc:
  _, exc, tb = sys.exc_info()
  six.reraise(ImportError, ImportError(str(exc)), tb)


class GLFWContext(base.ContextBase):
  """An OpenGL context backed by GLFW."""

  def __init__(self, max_width, max_height):
    """Initializes this context.

    Args:
      max_width: Integer specifying the maximum framebuffer width in pixels.
      max_height: Integer specifying the maximum framebuffer height in pixels.
    """
    super(GLFWContext, self).__init__()
    glfw.window_hint(glfw.VISIBLE, 0)
    glfw.window_hint(glfw.DOUBLEBUFFER, 0)
    self._context = glfw.create_window(width=max_width, height=max_height,
                                       title='Invisible window', monitor=None,
                                       share=None)
    self._previous_context = None
    # This reference prevents `glfw` from being garbage-collected before the
    # last window is destroyed, otherwise we may get `AttributeError`s when the
    # `__del__` method is later called.
    self._glfw = glfw

  def activate(self, width, height):
    """Called when entering the `make_current` context manager.

    Args:
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.
    """
    self._previous_context = glfw.get_current_context()
    glfw.make_context_current(self._context)
    if (width, height) != glfw.get_window_size(self._context):
      glfw.set_window_size(self._context, width, height)

  def deactivate(self):
    """Called when exiting the `make_current` context manager."""
    glfw.make_context_current(self._previous_context)

  def _free(self):
    """Frees resources associated with this context."""
    self._previous_context = None
    if self._context:
      glfw.destroy_window(self._context)
      self._context = None
