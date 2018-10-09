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
"""Tests for the views.py module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control.viewer import views
import mock
import numpy as np
import six
from six.moves import range


class ColumnTextViewTest(absltest.TestCase):

  def setUp(self):
    self.model = mock.MagicMock()
    self.view = views.ColumnTextView(self.model)

    self.context = mock.MagicMock()
    self.viewport = mock.MagicMock()
    self.location = views.PanelLocation.TOP_LEFT

  def test_rendering_empty_columns(self):
    self.model.get_columns.return_value = []
    with mock.patch(views.__name__ + '.mjlib') as mjlib_mock:
      self.view.render(self.context, self.viewport, self.location)
      self.assertEqual(0, mjlib_mock.mjr_overlay.call_count)

  def test_rendering(self):
    self.model.get_columns.return_value = [('', '')]
    with mock.patch(views.__name__ + '.mjlib') as mjlib_mock:
      self.view.render(self.context, self.viewport, self.location)
      mjlib_mock.mjr_overlay.assert_called_once()


class MujocoDepthBufferTests(absltest.TestCase):

  def setUp(self):
    self.component = views.MujocoDepthBuffer()

    self.context = mock.MagicMock()
    self.viewport = mock.MagicMock()

  def test_updating_buffer_size_after_viewport_resize(self):
    self.component._depth_buffer = np.zeros((1, 1), np.float32)
    self.viewport.width = 10
    self.viewport.height = 10

    with mock.patch(views.__name__ + '.mjlib'):
      self.component.render(context=self.context, viewport=self.viewport)
      self.assertEqual((10, 10), self.component._depth_buffer.shape)

  def test_reading_depth_data(self):
    with mock.patch(views.__name__ + '.mjlib') as mjlib_mock:
      self.component.render(context=self.context, viewport=self.viewport)
      mjlib_mock.mjr_readPixels.assert_called_once()
      self.assertIsNone(mjlib_mock.mjr_readPixels.call_args[0][0])

  def test_rendering_position_fixed_to_bottom_right_quarter_of_viewport(self):
    self.viewport.width = 100
    self.viewport.height = 100
    expected_rect = [75, 0, 25, 25]
    with mock.patch(views.__name__ + '.mjlib') as mjlib_mock:
      self.component.render(context=self.context, viewport=self.viewport)
      mjlib_mock.mjr_drawPixels.assert_called_once()
      render_rect = mjlib_mock.mjr_drawPixels.call_args[0][2]
      self.assertEqual(expected_rect[0], render_rect.left)
      self.assertEqual(expected_rect[1], render_rect.bottom)
      self.assertEqual(expected_rect[2], render_rect.width)
      self.assertEqual(expected_rect[3], render_rect.height)


class ViewportLayoutTest(absltest.TestCase):

  def setUp(self):
    self.layout = views.ViewportLayout()

    self.context = mock.MagicMock()
    self.viewport = mock.MagicMock()

  def test_added_elements_need_to_be_a_view(self):
    self.element = mock.MagicMock()
    with self.assertRaises(TypeError):
      self.layout.add(self.element, views.PanelLocation.TOP_LEFT)

  def test_adding_component(self):
    self.element = mock.MagicMock(spec=views.BaseViewportView)
    self.layout.add(self.element, views.PanelLocation.TOP_LEFT)
    self.assertEqual(1, len(self.layout))

  def test_adding_same_component_twice_updates_location(self):
    self.element = mock.MagicMock(spec=views.BaseViewportView)
    self.layout.add(self.element, views.PanelLocation.TOP_LEFT)
    self.layout.add(self.element, views.PanelLocation.TOP_RIGHT)
    self.assertEqual(
        views.PanelLocation.TOP_RIGHT, self.layout._views[self.element])

  def test_removing_component(self):
    self.element = mock.MagicMock(spec=views.BaseViewportView)
    self.layout._views[self.element] = views.PanelLocation.TOP_LEFT
    self.layout.remove(self.element)
    self.assertEqual(0, len(self.layout))

  def test_removing_unregistered_component(self):
    self.element = mock.MagicMock(spec=views.BaseViewportView)
    self.layout.remove(self.element)  # No error is raised

  def test_clearing_layout(self):
    pos = views.PanelLocation.TOP_LEFT
    self.layout._views = {mock.MagicMock(spec=views.BaseViewportView): pos
                          for _ in range(3)}
    self.layout.clear()
    self.assertEqual(0, len(self.layout))

  def test_rendering_layout(self):
    positions = [
        views.PanelLocation.TOP_LEFT,
        views.PanelLocation.TOP_RIGHT,
        views.PanelLocation.BOTTOM_LEFT]
    self.layout._views = {mock.MagicMock(spec=views.BaseViewportView): pos
                          for pos in positions}
    self.layout.render(self.context, self.viewport)
    for view, location in six.iteritems(self.layout._views):
      view.render.assert_called_once()
      self.assertEqual(location, view.render.call_args[0][2])


if __name__ == '__main__':
  absltest.main()
