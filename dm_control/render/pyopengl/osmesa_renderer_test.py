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

"""Tests for OSMesaContext."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

# Internal dependencies.
from absl.testing import absltest

try:
  from dm_control.render.pyopengl import osmesa_renderer  # pylint: disable=g-import-not-at-top
except (ImportError, IOError, OSError):
  osmesa_renderer = None

import mock  # pylint: disable=g-import-not-at-top

MAX_WIDTH = 640
MAX_HEIGHT = 480

if osmesa_renderer:
  CONTEXT_PATH = osmesa_renderer.__name__ + '.osmesa'
  GL_ARRAYS_PATH = osmesa_renderer.__name__ + '.arrays'


@unittest.skipUnless(osmesa_renderer,
                     reason='OSMesa renderer could not be imported.')
class OSMesaContextTest(absltest.TestCase):

  def test_init(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_osmesa:
      mock_osmesa.OSMesaCreateContextExt.return_value = mock_context
      renderer = osmesa_renderer.OSMesaContext(MAX_WIDTH, MAX_HEIGHT)
      self.assertIs(renderer._context, mock_context)
      renderer.free()

  def test_make_current(self):
    mock_context = mock.MagicMock()
    mock_buffer = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_osmesa:
      with mock.patch(GL_ARRAYS_PATH) as mock_glarrays:
        mock_osmesa.OSMesaCreateContextExt.return_value = mock_context
        mock_glarrays.GLfloatArray.zeros.return_value = mock_buffer
        renderer = osmesa_renderer.OSMesaContext(MAX_WIDTH, MAX_HEIGHT)
        with renderer.make_current():
          pass
        mock_osmesa.OSMesaMakeCurrent.assert_called_once_with(
            mock_context, mock_buffer,
            osmesa_renderer.GL.GL_FLOAT, MAX_WIDTH, MAX_HEIGHT)
        renderer.free()

  def test_freeing(self):
    mock_context = mock.MagicMock()
    with mock.patch(CONTEXT_PATH) as mock_osmesa:
      mock_osmesa.OSMesaCreateContextExt.return_value = mock_context
      renderer = osmesa_renderer.OSMesaContext(MAX_WIDTH, MAX_HEIGHT)
      renderer.free()
      mock_osmesa.OSMesaDestroyContext.assert_called_once_with(mock_context)
      self.assertIsNone(renderer._context)


if __name__ == '__main__':
  absltest.main()
