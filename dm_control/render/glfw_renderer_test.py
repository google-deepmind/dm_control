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

"""Tests for GLFWRenderer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest

from dm_control import render

import glfw
import mock

MAX_WIDTH = 1024
MAX_HEIGHT = 1024


class GLFWRendererTest(absltest.TestCase):

  @mock.patch(render.__name__ + ".glfw_renderer.glfw", spec=glfw)
  def test_context_activation_and_deactivation(self, mock_glfw):
    context = mock.MagicMock()

    mock_glfw.create_window = mock.MagicMock(return_value=context)
    mock_glfw.get_current_context = mock.MagicMock(return_value=None)

    renderer = render.Renderer(MAX_WIDTH, MAX_HEIGHT)
    renderer.make_context_current = mock.MagicMock()

    with renderer.make_current(2, 2):
      mock_glfw.make_context_current.assert_called_once_with(context)
      mock_glfw.make_context_current.reset_mock()

    mock_glfw.make_context_current.assert_called_once_with(None)


if __name__ == "__main__":
  absltest.main()
