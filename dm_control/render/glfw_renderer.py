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

# Internal dependencies.

from dm_control.render import base
import glfw

_done_init_glfw = False


def _maybe_init_glfw():
  global _done_init_glfw
  if not _done_init_glfw:
    if not glfw.init():
      raise OSError('Failed to initialize GLFW.')
    _done_init_glfw = True


class GLFWRenderer(base.Renderer):
  """An OpenGL renderer backed by GLFW."""

  def _create(self, max_width, max_height):
    _maybe_init_glfw()
    glfw.window_hint(glfw.VISIBLE, 0)
    glfw.window_hint(glfw.DOUBLEBUFFER, 0)
    self._context = glfw.create_window(width=max_width, height=max_height,
                                       title='Invisible window', monitor=None,
                                       share=None)
    # This reference prevents `glfw` from being garbage-collected before the
    # last window is destroyed, otherwise we may get `AttributeError`s when the
    # `__del__` method is later called.
    self._glfw = glfw

  def _before_make_current(self, width, height):
    previous_context = glfw.get_current_context()
    glfw.make_context_current(self._context)
    if (width, height) != glfw.get_window_size(self._context):
      glfw.set_window_size(self._context, width, height)
    return previous_context

  def _after_make_current(self, previous_context):
    glfw.make_context_current(previous_context)

  def free_context(self):
    if self._context is not None:
      self._glfw.destroy_window(self._context)
      self._context = None
