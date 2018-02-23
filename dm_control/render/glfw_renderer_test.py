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

try:
  from dm_control.render import glfw_renderer  # pylint: disable=g-import-not-at-top
except (ImportError, IOError, OSError):
  glfw_renderer = None

import mock  # pylint: disable=g-import-not-at-top

MAX_WIDTH = 1024
MAX_HEIGHT = 1024

if glfw_renderer:
  CONTEXT_PATH = glfw_renderer.__name__ + '.glfw'


@unittest.skipUnless(glfw_renderer,
                     reason='GLFW renderer could not be imported.')
class GLFWContextTest(absltest.TestCase):

  def test_init(self, mock_glfw):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = glfw_renderer.GLFWContext(MAX_WIDTH, MAX_HEIGHT)
    mock_glfw.make_context_current.assert_called_once_with(mock_context)
    self.assertIs(renderer._context, mock_context)

  def test_make_current(self, mock_glfw):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = glfw_renderer.GLFWContext(MAX_WIDTH, MAX_HEIGHT)
    with renderer.make_current(MAX_WIDTH, MAX_HEIGHT):
      pass
    mock_glfw.set_window_size.assert_called_once_with(
        mock_context, MAX_WIDTH, MAX_HEIGHT)

  def test_freeing(self, mock_glfw):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_glfw:
      mock_glfw.create_window.return_value = mock_context
      renderer = glfw_renderer.GLFWContext(MAX_WIDTH, MAX_HEIGHT)
    renderer.free()
    mock_glfw.destroy_window.assert_called_once_with(mock_context)
    self.assertIsNone(renderer._context)


if __name__ == '__main__':
  absltest.main()
