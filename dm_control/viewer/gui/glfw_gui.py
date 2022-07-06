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
"""Windowing system that uses GLFW library."""

import functools
from dm_control import _render
from dm_control._render import glfw_renderer
from dm_control.viewer import util
from dm_control.viewer.gui import base
from dm_control.viewer.gui import fullscreen_quad
import glfw
import numpy as np


def _check_valid_backend(func):
  """Decorator which checks that GLFW is being used for offscreen rendering."""
  @functools.wraps(func)
  def wrapped_func(*args, **kwargs):
    if _render.BACKEND != 'glfw':
      raise RuntimeError(
          '{func} may only be called if using GLFW for offscreen rendering, '
          'got `render.BACKEND={backend!r}`.'.format(
              func=func, backend=_render.BACKEND))
    return func(*args, **kwargs)
  return wrapped_func


class DoubleBufferedGlfwContext(glfw_renderer.GLFWContext):
  """Custom context manager for the GLFW based GUI."""

  def __init__(self, width, height, title):
    self._title = title
    super().__init__(max_width=width, max_height=height)

  @_check_valid_backend
  def _platform_init(self, width, height):
    glfw.window_hint(glfw.SAMPLES, 4)
    glfw.window_hint(glfw.VISIBLE, 1)
    glfw.window_hint(glfw.DOUBLEBUFFER, 1)
    self._context = glfw.create_window(width, height, self._title, None, None)
    self._destroy_window = glfw.destroy_window

  @property
  def window(self):
    return self._context


class GlfwKeyboard(base.InputEventsProcessor):
  """Glfw keyboard device handler.

  Handles the keyboard input in a thread-safe way, and forwards the events
  to the registered callbacks.

  Attributes:
    on_key: Observable subject triggered when a key event is triggered.
      Expects a callback with signature: (key, scancode, activity, modifiers)
  """

  def __init__(self, context):
    super().__init__()
    with context.make_current() as ctx:
      ctx.call(glfw.set_key_callback, context.window, self._handle_key_event)
    self.on_key = util.QuietSet()

  def _handle_key_event(self, window, key, scancode, activity, mods):
    """Broadcasts the notification to registered listeners.

    Args:
      window: The window that received the event.
      key: ID representing the key, a glfw.KEY_ constant.
      scancode: The system-specific scancode of the key.
      activity: glfw.PRESS, glfw.RELEASE or glfw.REPEAT.
      mods: Bit field describing which modifier keys were held down, such as Alt
        or Shift.
    """
    del window, scancode
    self.add_event(self.on_key, key, activity, mods)


class GlfwMouse(base.InputEventsProcessor):
  """Glfw mouse device handler.

  Handles the mouse input in a thread-safe way, forwarding the events to the
  registered callbacks.

  Attributes:
    on_move: Observable subject triggered when a mouse move is detected.
      Expects a callback with signature (position, translation).
    on_click: Observable subject triggered when a mouse click is detected.
      Expects a callback with signature (button, action, modifiers).
    on_double_click: Observable subject triggered when a mouse double click is
      detected. Expects a callback with signature (button, modifiers).
    on_scroll: Observable subject triggered when a mouse scroll is detected.
      Expects a callback with signature (scroll_value).
  """

  def __init__(self, context):
    super().__init__()
    self.on_move = util.QuietSet()
    self.on_click = util.QuietSet()
    self.on_double_click = util.QuietSet()
    self.on_scroll = util.QuietSet()
    self._double_click_detector = base.DoubleClickDetector()
    with context.make_current() as ctx:
      framebuffer_width, window_width = ctx.call(
          self._glfw_setup, context.window)

    self._scale = framebuffer_width * 1.0 / window_width
    self._last_mouse_pos = np.zeros(2, int)

    self._double_clicks = {}

  def _glfw_setup(self, window):
    glfw.set_cursor_pos_callback(window, self._handle_move)
    glfw.set_mouse_button_callback(window, self._handle_button)
    glfw.set_scroll_callback(window, self._handle_scroll)
    framebuffer_width, _ = glfw.get_framebuffer_size(window)
    window_width, _ = glfw.get_window_size(window)
    return framebuffer_width, window_width

  @property
  def position(self):
    return self._last_mouse_pos

  def _handle_move(self, window, x, y):
    """Mouse movement callback.

    Args:
      window: Window object from glfw.
      x: Horizontal position of mouse, in pixels.
      y: Vertical position of mouse, in pixels.
    """
    del window
    position = np.array([x, y], int) * self._scale
    delta = position - self._last_mouse_pos
    self._last_mouse_pos = position
    self.add_event(self.on_move, position, delta)

  def _handle_button(self, window, button, act, mods):
    """Mouse button click event handler."""
    del window
    self.add_event(self.on_click, button, act, mods)
    if self._double_click_detector.process(button, act):
      self.add_event(self.on_double_click, button, mods)

  def _handle_scroll(self, window, x_offset, y_offset):
    """Mouse wheel scroll event handler."""
    del window, x_offset
    self.add_event(self.on_scroll, y_offset)


class GlfwWindow:
  """A GLFW based application window.

  Attributes:
    on_files_drop: An observable subject, instance of util.QuietSet. Attached
      listeners, callables taking one argument, will be invoked every time the
      user drops files onto the window. The callable will be passed an iterable
      with dropped file paths.
    is_full_screen: Boolean, whether the window is currently full-screen.
  """

  def __init__(self, width, height, title, context=None):
    """Instance initializer.

    Args:
      width: Initial window width, in pixels.
      height: Initial window height, in pixels.
      title: A string with a window title.
      context: (Optional) A `render.GLFWContext` instance.

    Raises:
      RuntimeError: If GLFW initialization or window initialization fails.
    """
    super().__init__()
    self._context = context or DoubleBufferedGlfwContext(width, height, title)

    if not self._context.window:
      raise RuntimeError('Failed to create window')

    self._oldsize = None

    with self._context.make_current() as ctx:
      self._fullscreen_quad = ctx.call(self._glfw_setup, self._context.window)
    self.on_files_drop = util.QuietSet()

    self._keyboard = GlfwKeyboard(self._context)
    self._mouse = GlfwMouse(self._context)

  def _glfw_setup(self, window):
    glfw.set_drop_callback(window, self._handle_file_drop)
    return fullscreen_quad.FullscreenQuadRenderer()

  @property
  def shape(self):
    """Returns a tuple with the shape of the window, (width, height)."""
    with self._context.make_current() as ctx:
      return ctx.call(glfw.get_framebuffer_size, self._context.window)

  @property
  def position(self):
    """Returns a tuple with top-left window corner's coordinates, (x, y)."""
    with self._context.make_current() as ctx:
      return ctx.call(glfw.get_window_pos, self._context.window)

  @property
  def keyboard(self):
    """Returns a GlfwKeyboard instance associated with the window."""
    return self._keyboard

  @property
  def mouse(self):
    """Returns a GlfwMouse instance associated with the window."""
    return self._mouse

  def set_title(self, title):
    """Sets the window title.

    Args:
      title: A string, title of the window.
    """
    with self._context.make_current() as ctx:
      ctx.call(glfw.set_window_title, self._context.window, title)

  def set_full_screen(self, enable):
    """Expands the main application window to full screen or minimizes it.

    Args:
      enable: Boolean flag, True expands the window to full-screen mode, False
        minimizes it to its former size.
    """
    if enable == self.is_full_screen:
      return

    if enable:
      self._oldsize = list(self.position) + list(self.shape)
      def enable_full_screen(window):
        display = glfw.get_primary_monitor()
        videomode = glfw.get_video_mode(display)
        glfw.set_window_monitor(window, display, 0, 0, videomode[0][0],
                                videomode[0][1], videomode[2])
      with self._context.make_current() as ctx:
        ctx.call(enable_full_screen, self._context.window)
    else:
      with self._context.make_current() as ctx:
        ctx.call(glfw.set_window_monitor,
                 self._context.window, None, self._oldsize[0],
                 self._oldsize[1], self._oldsize[2],
                 self._oldsize[3], 0)
      self._oldsize = None

  def toggle_full_screen(self):
    """Expands the main application window to full screen or minimizes it."""
    show_full_screen = not self.is_full_screen
    self.set_full_screen(show_full_screen)

  @property
  def is_full_screen(self):
    return self._oldsize is not None

  def free(self):
    """Closes the deleted window."""
    self.close()

  def event_loop(self, tick_func):
    """Runs the window's event loop.

    This is a blocking call that won't exit until the window is closed.

    Args:
      tick_func: A callable, function to call every frame.
    """
    while not glfw.window_should_close(self._context.window):
      pixels = tick_func()
      with self._context.make_current() as ctx:
        ctx.call(
            self._update_gui_on_render_thread, self._context.window, pixels)
      self._mouse.process_events()
      self._keyboard.process_events()

  def update(self, render_func):
    """Updates the window and renders a new image.

    Args:
      render_func: A callable returning a 3D numpy array of bytes (np.uint8),
        with dimensions (width, height, 3).
    """
    pixels = render_func()
    with self._context.make_current() as ctx:
      ctx.call(
          self._update_gui_on_render_thread, self._context.window, pixels)
    self._mouse.process_events()
    self._keyboard.process_events()

  def _update_gui_on_render_thread(self, window, pixels):
    self._fullscreen_quad.render(pixels, self.shape)
    glfw.swap_buffers(window)
    glfw.poll_events()

  def close(self):
    """Closes the window and releases associated resources."""
    if self._context is not None:
      self._context.free()
    self._context = None

  def _handle_file_drop(self, window, paths):
    """Handles events of user dropping files onto the window.

    Args:
      window: GLFW window handle (unused).
      paths: An iterable with paths of files dropped onto the window.
    """
    del window
    for listener in list(self.on_files_drop):
      listener(paths)

  def __del__(self):
    self.free()
