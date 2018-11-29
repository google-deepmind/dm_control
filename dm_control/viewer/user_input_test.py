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
"""Tests for the user_input module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control.viewer import user_input
import mock


class InputMapTests(absltest.TestCase):

  def setUp(self):
    super(InputMapTests, self).setUp()
    self.mouse = mock.MagicMock()
    self.keyboard = mock.MagicMock()
    self.input_map = user_input.InputMap(self.mouse, self.keyboard)

    self.callback = mock.MagicMock()

  def test_clearing_bindings(self):
    self.input_map._active_exclusive = 1
    self.input_map._action_callbacks = {1: 2}
    self.input_map._double_click_callbacks = {3: 4}
    self.input_map._plane_callback = [5]
    self.input_map._z_axis_callback = [6]

    self.input_map.clear_bindings()

    self.assertEmpty(self.input_map._action_callbacks)
    self.assertEmpty(self.input_map._double_click_callbacks)
    self.assertEmpty(self.input_map._plane_callback)
    self.assertEmpty(self.input_map._z_axis_callback)
    self.assertEqual(
        user_input._NO_EXCLUSIVE_KEY, self.input_map._active_exclusive)

  def test_binding(self):
    self.input_map.bind(self.callback, user_input.KEY_UP)
    expected_dict = {
        (user_input.KEY_UP, user_input.MOD_NONE): (False, self.callback)}
    self.assertDictEqual(expected_dict, self.input_map._action_callbacks)

  def test_binding_exclusive(self):
    self.input_map.bind(self.callback, user_input.Exclusive(user_input.KEY_UP))
    expected_dict = {
        (user_input.KEY_UP, user_input.MOD_NONE): (True, self.callback)}
    self.assertDictEqual(expected_dict, self.input_map._action_callbacks)

  def test_binding_and_invoking_ranges_of_actions(self):
    self.input_map.bind(self.callback, user_input.Range(
        [user_input.KEY_UP, (user_input.KEY_UP, user_input.MOD_ALT)]))

    self.input_map._handle_key(
        user_input.KEY_UP, user_input.PRESS, user_input.MOD_NONE)
    self.callback.assert_called_once_with(0)

    self.callback.reset_mock()
    self.input_map._handle_key(
        user_input.KEY_UP, user_input.PRESS, user_input.MOD_ALT)
    self.callback.assert_called_once_with(1)

  def test_binding_planar_action(self):
    self.input_map.bind_plane(self.callback)
    self.assertLen(self.input_map._plane_callback, 1)
    self.assertEqual(self.callback, self.input_map._plane_callback[0])

  def test_binding_z_axis_action(self):
    self.input_map.bind_z_axis(self.callback)
    self.assertLen(self.input_map._z_axis_callback, 1)
    self.assertEqual(self.callback, self.input_map._z_axis_callback[0])

  def test_invoking_regular_action_in_response_to_click(self):
    self.input_map._action_callbacks = {(1, 2): (False, self.callback)}

    self.input_map._handle_key(1, user_input.PRESS, 2)
    self.callback.assert_called_once()
    self.callback.reset_mock()

    self.input_map._handle_key(1, user_input.RELEASE, 2)
    self.assertEqual(0, self.callback.call_count)

  def test_invoking_exclusive_action_in_response_to_click(self):
    self.input_map._action_callbacks = {(1, 2): (True, self.callback)}

    self.input_map._handle_key(1, user_input.PRESS, 2)
    self.callback.assert_called_once_with(True)
    self.callback.reset_mock()

    self.input_map._handle_key(1, user_input.RELEASE, 2)
    self.callback.assert_called_once_with(False)

  def test_exclusive_action_blocks_other_actions_until_its_finished(self):
    self.input_map._action_callbacks = {
        (1, 2): (True, self.callback), (3, 4): (False, self.callback)}

    self.input_map._handle_key(1, user_input.PRESS, 2)
    self.callback.assert_called_once_with(True)
    self.callback.reset_mock()

    # Attempting to start other actions (PRESS) or end them (RELEASE)
    # amounts to nothing.
    self.input_map._handle_key(3, user_input.PRESS, 4)
    self.assertEqual(0, self.callback.call_count)

    self.input_map._handle_key(3, user_input.RELEASE, 4)
    self.assertEqual(0, self.callback.call_count)

    # Even attempting to start the same action for the 2nd time fails.
    self.input_map._handle_key(1, user_input.PRESS, 2)
    self.assertEqual(0, self.callback.call_count)

    # Only finishing the action frees up the resources.
    self.input_map._handle_key(1, user_input.RELEASE, 2)
    self.callback.assert_called_once_with(False)
    self.callback.reset_mock()

    # Now we can start a new action.
    self.input_map._handle_key(3, user_input.PRESS, 4)
    self.callback.assert_called_once()

  def test_modifiers_required_only_for_exclusive_action_start(self):
    activation_modifiers = 2
    no_modifiers = 0
    self.input_map._action_callbacks = {
        (1, activation_modifiers): (True, self.callback)}

    self.input_map._handle_key(1, user_input.PRESS, activation_modifiers)
    self.callback.assert_called_once_with(True)
    self.callback.reset_mock()

    self.input_map._handle_key(1, user_input.RELEASE, no_modifiers)
    self.callback.assert_called_once_with(False)

  def test_invoking_regular_action_in_response_to_double_click(self):
    self.input_map._double_click_callbacks = {(1, 2): self.callback}
    self.input_map._handle_double_click(1, 2)
    self.callback.assert_called_once()

  def test_exclusive_actions_dont_respond_to_double_clicks(self):
    self.input_map._action_callbacks = {(1, 2): (True, self.callback)}

    self.input_map._handle_double_click(1, 2)
    self.assertEqual(0, self.callback.call_count)

  def test_mouse_move(self):
    position = [1, 2]
    translation = [3, 4]
    self.input_map._plane_callback = [self.callback]
    self.input_map._handle_mouse_move(position, translation)
    self.callback.assert_called_once_with(position, translation)

  def test_mouse_scroll(self):
    value = 5
    self.input_map._z_axis_callback = [self.callback]
    self.input_map._handle_mouse_scroll(value)
    self.callback.assert_called_once_with(value)


if __name__ == '__main__':
  absltest.main()
