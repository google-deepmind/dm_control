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
"""Utilities for handling keyboard events."""

import collections


# Mapped input values, so that we don't have to reference glfw everywhere.
RELEASE = 0
PRESS = 1
REPEAT = 2

KEY_UNKNOWN = -1
KEY_SPACE = 32
KEY_APOSTROPHE = 39
KEY_COMMA = 44
KEY_MINUS = 45
KEY_PERIOD = 46
KEY_SLASH = 47
KEY_0 = 48
KEY_1 = 49
KEY_2 = 50
KEY_3 = 51
KEY_4 = 52
KEY_5 = 53
KEY_6 = 54
KEY_7 = 55
KEY_8 = 56
KEY_9 = 57
KEY_SEMICOLON = 59
KEY_EQUAL = 61
KEY_A = 65
KEY_B = 66
KEY_C = 67
KEY_D = 68
KEY_E = 69
KEY_F = 70
KEY_G = 71
KEY_H = 72
KEY_I = 73
KEY_J = 74
KEY_K = 75
KEY_L = 76
KEY_M = 77
KEY_N = 78
KEY_O = 79
KEY_P = 80
KEY_Q = 81
KEY_R = 82
KEY_S = 83
KEY_T = 84
KEY_U = 85
KEY_V = 86
KEY_W = 87
KEY_X = 88
KEY_Y = 89
KEY_Z = 90
KEY_LEFT_BRACKET = 91
KEY_BACKSLASH = 92
KEY_RIGHT_BRACKET = 93
KEY_GRAVE_ACCENT = 96
KEY_ESCAPE = 256
KEY_ENTER = 257
KEY_TAB = 258
KEY_BACKSPACE = 259
KEY_INSERT = 260
KEY_DELETE = 261
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_PAGE_UP = 266
KEY_PAGE_DOWN = 267
KEY_HOME = 268
KEY_END = 269
KEY_CAPS_LOCK = 280
KEY_SCROLL_LOCK = 281
KEY_NUM_LOCK = 282
KEY_PRINT_SCREEN = 283
KEY_PAUSE = 284
KEY_F1 = 290
KEY_F2 = 291
KEY_F3 = 292
KEY_F4 = 293
KEY_F5 = 294
KEY_F6 = 295
KEY_F7 = 296
KEY_F8 = 297
KEY_F9 = 298
KEY_F10 = 299
KEY_F11 = 300
KEY_F12 = 301
KEY_KP_0 = 320
KEY_KP_1 = 321
KEY_KP_2 = 322
KEY_KP_3 = 323
KEY_KP_4 = 324
KEY_KP_5 = 325
KEY_KP_6 = 326
KEY_KP_7 = 327
KEY_KP_8 = 328
KEY_KP_9 = 329
KEY_KP_DECIMAL = 330
KEY_KP_DIVIDE = 331
KEY_KP_MULTIPLY = 332
KEY_KP_SUBTRACT = 333
KEY_KP_ADD = 334
KEY_KP_ENTER = 335
KEY_KP_EQUAL = 336
KEY_LEFT_SHIFT = 340
KEY_LEFT_CONTROL = 341
KEY_LEFT_ALT = 342
KEY_LEFT_SUPER = 343
KEY_RIGHT_SHIFT = 344
KEY_RIGHT_CONTROL = 345
KEY_RIGHT_ALT = 346
KEY_RIGHT_SUPER = 347

MOD_NONE = 0
MOD_SHIFT = 0x0001
MOD_CONTROL = 0x0002
MOD_ALT = 0x0004
MOD_SUPER = 0x0008
MOD_SHIFT_CONTROL = MOD_SHIFT | MOD_CONTROL

MOUSE_BUTTON_LEFT = 0
MOUSE_BUTTON_RIGHT = 1
MOUSE_BUTTON_MIDDLE = 2

_NO_EXCLUSIVE_KEY = (None, None)
_NO_CALLBACK = (None, None)


class Exclusive(collections.namedtuple('Exclusive', 'combination')):
  """Defines an exclusive action.

  Exclusive actions can be invoked in response to single key clicks only. The
  callback will be called twice. The first time when the key combination is
  pressed, passing True as the argument to the callback. The second time when
  the key is released (the modifiers don't have to be present then), passing
  False as the callback argument.

  Attributes:
    combination: A list of integers interpreted as key codes, or tuples
      in format (keycode, modifier).
  """
  pass


class DoubleClick(collections.namedtuple('DoubleClick', 'combination')):
  """Defines a mouse double click action.

  It will define a requirement to double click the mouse button specified in the
  combination in order to be triggered.

  Attributes:
    combination: A list of integers interpreted as key codes, or tuples
      in format (keycode, modifier). The keycodes are limited only to mouse
      button codes.
  """
  pass


class Range(collections.namedtuple('Range', 'collection')):
  """Binds a number of key combinations to a callback.

  When triggered, the index of the triggering key combination will be passed
  as an argument to the callback.

  Attributes:
    callback: A callable accepting a single argument - an integer index of the
      triggered callback.
    collection: A collection of combinations. Combinations may either be raw key
      codes, tuples in format (keycode, modifier), or one of the Exclusive or
      DoubleClick instances.
  """
  pass


class InputMap(object):
  """Provides ability to alias key combinations and map them to actions."""

  def __init__(self, mouse, keyboard):
    """Instance initializer.

    Args:
      mouse: GlfwMouse instance.
      keyboard: GlfwKeyboard instance.
    """
    self._keyboard = keyboard
    self._mouse = mouse

    self._keyboard.on_key += self._handle_key
    self._mouse.on_click += self._handle_key
    self._mouse.on_double_click += self._handle_double_click
    self._mouse.on_move += self._handle_mouse_move
    self._mouse.on_scroll += self._handle_mouse_scroll

    self.clear_bindings()

  def __del__(self):
    """Instance deleter."""
    self._keyboard.on_key -= self._handle_key
    self._mouse.on_click -= self._handle_key
    self._mouse.on_double_click -= self._handle_double_click
    self._mouse.on_move -= self._handle_mouse_move
    self._mouse.on_scroll -= self._handle_mouse_scroll

  def clear_bindings(self):
    """Clears registered action bindings, while keeping key aliases."""
    self._action_callbacks = {}
    self._double_click_callbacks = {}
    self._plane_callback = []
    self._z_axis_callback = []
    self._active_exclusive = _NO_EXCLUSIVE_KEY

  def bind(self, callback, key_binding):
    """Binds a key combination to a callback.

    Args:
      callback: An argument-less callable.
      key_binding: A integer with a key code, a tuple (keycode, modifier) or one
        of the actions Exclusive|DoubleClick|Range carrying the key combination.
    """
    def build_callback(index, callback):
      def indexed_callback():
        callback(index)
      return indexed_callback

    if isinstance(key_binding, Range):
      for index, binding in enumerate(key_binding.collection):
        self._add_binding(build_callback(index, callback), binding)
    else:
      self._add_binding(callback, key_binding)

  def _add_binding(self, callback, key_binding):
    key_combination = self._extract_key_combination(key_binding)
    if isinstance(key_binding, Exclusive):
      self._action_callbacks[key_combination] = (True, callback)
    elif isinstance(key_binding, DoubleClick):
      self._double_click_callbacks[key_combination] = callback
    else:
      self._action_callbacks[key_combination] = (False, callback)

  def _extract_key_combination(self, key_binding):
    if isinstance(key_binding, Exclusive):
      key_binding = key_binding.combination
    elif isinstance(key_binding, DoubleClick):
      key_binding = key_binding.combination

    if not isinstance(key_binding, tuple):
      key_binding = (key_binding, MOD_NONE)
    return key_binding

  def bind_plane(self, callback):
    """Binds a callback to a planar motion action (mouse movement)."""
    self._plane_callback.append(callback)

  def bind_z_axis(self, callback):
    """Binds a callback to a z-axis motion action (mouse scroll)."""
    self._z_axis_callback.append(callback)

  def _handle_key(self, key, action, modifiers):
    """Handles a single key press (mouse and keyboard)."""
    alias_key = (key, modifiers)

    exclusive_key, exclusive_callback = self._active_exclusive
    if exclusive_key is not None:
      if action == RELEASE and key == exclusive_key:
        exclusive_callback(False)
        self._active_exclusive = _NO_EXCLUSIVE_KEY
    else:
      is_exclusive, callback = self._action_callbacks.get(
          alias_key, _NO_CALLBACK)
      if callback:
        if action == PRESS:
          if is_exclusive:
            callback(True)
            self._active_exclusive = (key, callback)
          else:
            callback()

  def _handle_double_click(self, key, modifiers):
    """Handles a double mouse click."""
    alias_key = (key, modifiers)
    callback = self._double_click_callbacks.get(alias_key, None)
    if callback is not None:
      callback()

  def _handle_mouse_move(self, position, translation):
    """Handles mouse move."""
    for callback in self._plane_callback:
      callback(position, translation)

  def _handle_mouse_scroll(self, value):
    """Handles mouse wheel scroll."""
    for callback in self._z_axis_callback:
      callback(value)


