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

# Internal dependencies.

from absl.testing import absltest

from dm_control.render import glfw_renderer

import mock

MAX_WIDTH = 1024
MAX_HEIGHT = 1024
CONTEXT_PATH = glfw_renderer.__name__ + ".glfw"


@mock.patch(CONTEXT_PATH)
class GLFWContextTest(absltest.TestCase):

  def setUp(self):
    self.context = mock.MagicMock()

    with mock.patch(CONTEXT_PATH):
      self.renderer = glfw_renderer.GLFWContext(MAX_WIDTH, MAX_HEIGHT)

  def tearDown(self):
    self.renderer._context = None

  def test_activation(self, mock_glfw):
    self.renderer.activate(MAX_WIDTH, MAX_HEIGHT)
    mock_glfw.make_context_current.assert_called_once()

  def test_deactivation(self, mock_glfw):
    self.renderer.deactivate()
    mock_glfw.make_context_current.assert_called_once()

  def test_freeing(self, mock_glfw):
    self.renderer._context = mock.MagicMock()
    self.renderer._previous_context = mock.MagicMock()
    self.renderer.free()
    mock_glfw.destroy_window.assert_called_once()
    self.assertIsNone(self.renderer._context)
    self.assertIsNone(self.renderer._previous_context)


if __name__ == "__main__":
  absltest.main()
