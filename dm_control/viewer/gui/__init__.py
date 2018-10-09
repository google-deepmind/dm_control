# Copyright 2018 The dm_control Authors.
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
"""Viewer's windowing systems."""

from dm_control import render

# pylint: disable=g-import-not-at-top
# pylint: disable=invalid-name

RenderWindow = None
try:
  from dm_control.viewer.gui import glfw_gui
  RenderWindow = glfw_gui.GlfwWindow
except ImportError:
  pass
if not RenderWindow:

  def ErrorRenderWindow(*args, **kwargs):
    del args, kwargs
    raise ImportError(
        'Cannot create a window because no windowing system could be imported')
  RenderWindow = ErrorRenderWindow

del render
