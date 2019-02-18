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

"""Tests for GLFWContext."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

# Internal dependencies.
from absl.testing import absltest
from dm_control import _render
from dm_control.mujoco import wrapper
from dm_control.mujoco.testing import decorators


import mock  # pylint: disable=g-import-not-at-top

MAX_WIDTH = 1024
MAX_HEIGHT = 1024

CONTEXT_PATH = _render.__name__ + '.glfw_renderer.glfw'


@unittest.skipUnless(
    _render.BACKEND == _render.constants.GLFW,
    reason='GLFW beckend not selected.')
class GLFWContextTest(absltest.TestCase):

  def test_init(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
    self.assertIs(renderer._context, mock_context)

  def test_make_current(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
      with renderer.make_current():
        pass
    mock_glfw.make_context_current.assert_called_once_with(mock_context)

  def test_freeing(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
      renderer.free()
    mock_glfw.destroy_window.assert_called_once_with(mock_context)
    self.assertIsNone(renderer._context)

  @decorators.run_threaded(num_threads=1, calls_per_thread=20)
  def test_repeatedly_create_and_destroy_context(self):
    renderer = _render.Renderer(MAX_WIDTH, MAX_HEIGHT)
    wrapper.MjrContext(wrapper.MjModel.from_xml_string('<mujoco/>'), renderer)

if __name__ == '__main__':
  absltest.main()
