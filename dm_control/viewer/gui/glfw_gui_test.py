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
"""Tests for the GLFW based windowing system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
# Internal dependencies.
from absl.testing import absltest
from dm_control.viewer import user_input
import mock
import numpy as np


_OPEN_GL_MOCK = mock.MagicMock()
_GLFW_MOCK = mock.MagicMock()
_MOCKED_MODULES = {
    'OpenGL': _OPEN_GL_MOCK,
    'OpenGL.GL': _OPEN_GL_MOCK,
    'glfw': _GLFW_MOCK,
}
with mock.patch.dict('sys.modules', _MOCKED_MODULES):
  from dm_control.viewer.gui import glfw_gui  # pylint: disable=g-import-not-at-top

glfw_gui.base.GL = _OPEN_GL_MOCK
glfw_gui.base.shaders = _OPEN_GL_MOCK
glfw_gui.glfw = _GLFW_MOCK

# Pretend we are using GLFW for offscreen rendering so that the runtime backend
# check will succeed.
glfw_gui.render.BACKEND = 'glfw'

_EPSILON = 1e-7


class GlfwKeyboardTest(absltest.TestCase):

  def setUp(self):
    _GLFW_MOCK.reset_mock()
    self.context = mock.MagicMock()
    self.handler = glfw_gui.GlfwKeyboard(self.context)

    self.events = [
        (1, user_input.KEY_T, 't', user_input.PRESS, user_input.MOD_ALT),
        (1, user_input.KEY_T, 't', user_input.RELEASE, user_input.MOD_ALT),
        (1, user_input.KEY_A, 't', user_input.PRESS, 0)]

    self.listener = mock.MagicMock()
    self.handler.on_key += [self.listener]

  def test_single_event(self):
    self.handler._handle_key_event(*self.events[0])
    self.handler.process_events()
    self.listener.assert_called_once_with(user_input.KEY_T, user_input.PRESS,
                                          user_input.MOD_ALT)

  def test_sequence_of_events(self):
    for event in self.events:
      self.handler._handle_key_event(*event)
    self.handler.process_events()
    self.assertEqual(3, self.listener.call_count)
    for event, call_args in zip(self.events, self.listener.call_args_list):
      expected_event = tuple([event[1]] + list(event[-2:]))
      self.assertEqual(expected_event, call_args[0])


class FakePassthroughRenderingContext(object):

  def __init__(self):
    self.window = 0

  def call(self, func, *args):
    return func(*args)


class GlfwMouseTest(absltest.TestCase):

  @contextlib.contextmanager
  def fake_make_current(self):
    yield FakePassthroughRenderingContext()

  def setUp(self):
    _GLFW_MOCK.reset_mock()
    _GLFW_MOCK.get_framebuffer_size = mock.MagicMock(return_value=(256, 256))
    _GLFW_MOCK.get_window_size = mock.MagicMock(return_value=(256, 256))
    self.window = mock.MagicMock()
    self.window.make_current = mock.MagicMock(
        side_effect=self.fake_make_current)
    self.handler = glfw_gui.GlfwMouse(self.window)

  def test_moving_mouse(self):
    def move_handler(position, translation):
      self.position = position
      self.translation = translation
    self.new_position = [100, 100]
    self.handler._last_mouse_pos = np.array([99, 101], np.int)
    self.handler.on_move += move_handler

    self.handler._handle_move(self.window, self.new_position[0],
                              self.new_position[1])
    self.handler.process_events()
    np.testing.assert_array_equal(self.new_position, self.position)
    np.testing.assert_array_equal([1, -1], self.translation)

  def test_button_click(self):
    def click_handler(button, action, modifiers):
      self.button = button
      self.action = action
      self.modifiers = modifiers
    self.handler.on_click += click_handler

    self.handler._handle_button(self.window, user_input.MOUSE_BUTTON_LEFT,
                                user_input.PRESS, user_input.MOD_SHIFT)
    self.handler.process_events()

    self.assertEqual(user_input.MOUSE_BUTTON_LEFT, self.button)
    self.assertEqual(user_input.PRESS, self.action)
    self.assertEqual(user_input.MOD_SHIFT, self.modifiers)

  def test_scroll(self):
    def scroll_handler(position):
      self.position = position
    self.handler.on_scroll += scroll_handler

    x_value = 10
    y_value = 20
    self.handler._handle_scroll(self.window, x_value, y_value)
    self.handler.process_events()
    # x_value gets ignored, it's the y_value - the vertical scroll - we're
    # interested in.
    self.assertEqual(y_value, self.position)


class GlfwWindowTest(absltest.TestCase):

  WIDTH = 10
  HEIGHT = 20

  @contextlib.contextmanager
  def fake_make_current(self):
    yield FakePassthroughRenderingContext()

  def setUp(self):
    _GLFW_MOCK.reset_mock()
    _GLFW_MOCK.get_video_mode.return_value = (None, None, 60)
    _GLFW_MOCK.get_framebuffer_size.return_value = (4, 5)
    _GLFW_MOCK.get_window_size.return_value = (self.WIDTH, self.HEIGHT)
    self.context = mock.MagicMock()
    self.context.make_current = mock.MagicMock(
        side_effect=self.fake_make_current)
    self.window = glfw_gui.GlfwWindow(
        self.WIDTH, self.HEIGHT, 'title', self.context)

  def test_window_shape(self):
    expected_shape = (self.WIDTH, self.HEIGHT)
    _GLFW_MOCK.get_framebuffer_size.return_value = expected_shape
    self.assertEqual(expected_shape, self.window.shape)

  def test_window_position(self):
    expected_position = (1, 2)
    _GLFW_MOCK.get_window_pos.return_value = expected_position
    self.assertEqual(expected_position, self.window.position)

  def test_freeing_context(self):
    self.window.close = mock.MagicMock()
    self.window.free()
    self.window.close.assert_called_once()

  def test_close(self):
    self.window.close()
    _GLFW_MOCK.destroy_window.assert_called_once()
    self.assertIsNone(self.window._context)

  def test_closing_window_that_has_already_been_closed(self):
    self.window._context = None
    self.window.close()
    self.assertEqual(0, _GLFW_MOCK.destroy_window.call_count)

  def test_file_drop(self):
    self.expected_paths = ['path1', 'path2']
    def callback(paths):
      self.assertEqual(self.expected_paths, paths)

    was_called_mock = mock.MagicMock()
    self.window.on_files_drop += [callback, was_called_mock]
    self.window._handle_file_drop('window_handle', self.expected_paths)
    was_called_mock.assert_called_once()

  def test_setting_title(self):
    new_title = 'new_title'
    self.window.set_title(new_title)
    self.assertEqual(new_title, _GLFW_MOCK.set_window_title.call_args[0][1])

  def test_enabling_full_screen(self):
    full_screen_pos = (0, 0)
    full_screen_size = (1, 2)
    window_size = (3, 4)
    window_pos = (5, 6)
    reserved_value = 7
    full_size_mode = 8

    _GLFW_MOCK.get_framebuffer_size.return_value = window_size
    _GLFW_MOCK.get_window_pos.return_value = window_pos
    _GLFW_MOCK.get_video_mode.return_value = (
        full_screen_size, reserved_value, full_size_mode)

    self.window.set_full_screen(True)
    _GLFW_MOCK.set_window_monitor.assert_called_once()

    new_position = _GLFW_MOCK.set_window_monitor.call_args[0][2:4]
    new_size = _GLFW_MOCK.set_window_monitor.call_args[0][4:6]
    new_mode = _GLFW_MOCK.set_window_monitor.call_args[0][6]
    self.assertEqual(full_screen_pos, new_position)
    self.assertEqual(full_screen_size, new_size)
    self.assertEqual(full_size_mode, new_mode)


if __name__ == '__main__':
  absltest.main()
