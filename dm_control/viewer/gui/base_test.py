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
"""Tests for the base windowing system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control.viewer import user_input
import mock


# pylint: disable=g-import-not-at-top
_OPEN_GL_MOCK = mock.MagicMock()
_MOCKED_MODULES = {
    'OpenGL': _OPEN_GL_MOCK,
    'OpenGL.GL': _OPEN_GL_MOCK,
}
with mock.patch.dict('sys.modules', _MOCKED_MODULES):
  from dm_control.viewer.gui import base
# pylint: enable=g-import-not-at-top

_EPSILON = 1e-7


@mock.patch(base.__name__ + '.time')
class DoubleClickDetectorTest(absltest.TestCase):

  def setUp(self):
    self.detector = base.DoubleClickDetector()
    self.double_click_event = (user_input.MOUSE_BUTTON_LEFT, user_input.PRESS)

  def test_two_rapid_clicks_yield_double_click_event(self, mock_time):
    mock_time.time.return_value = 0
    self.assertFalse(self.detector.process(*self.double_click_event))

    mock_time.time.return_value = base._DOUBLE_CLICK_INTERVAL - _EPSILON
    self.assertTrue(self.detector.process(*self.double_click_event))

  def test_two_slow_clicks_dont_yield_double_click_event(self, mock_time):
    mock_time.time.return_value = 0
    self.assertFalse(self.detector.process(*self.double_click_event))

    mock_time.time.return_value = base._DOUBLE_CLICK_INTERVAL
    self.assertFalse(self.detector.process(*self.double_click_event))

  def test_sequence_of_slow_clicks_followed_by_fast_click(self, mock_time):
    click_times = [(0., False),
                   (base._DOUBLE_CLICK_INTERVAL * 2., False),
                   (base._DOUBLE_CLICK_INTERVAL * 3. - _EPSILON, True)]
    for click_time, result in click_times:
      mock_time.time.return_value = click_time
      self.assertEqual(result, self.detector.process(*self.double_click_event))


if __name__ == '__main__':
  absltest.main()
