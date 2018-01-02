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

"""Base class for OpenGL context handlers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import contextlib

# Internal dependencies.
import six


@six.add_metaclass(abc.ABCMeta)
class Renderer(object):
  """Base `Renderer` class for managing OpenGL contexts."""

  def __init__(self, max_width, max_height):
    """Initializes this `Renderer`.

    Arguments to this method are passed to `_create`.

    Args:
      max_width: Integer specifying the maximum framebuffer width in pixels.
      max_height: Integer specifying the maximum framebuffer height in pixels.
    """
    self._max_width = max_width
    self._max_height = max_height
    self._create(max_width, max_height)

  @abc.abstractmethod
  def _create(self, max_width, max_height):
    """Called internally by `__init__` to create the OpenGL context.

    Args:
      max_width: Integer specifying the maximum framebuffer width in pixels.
      max_height: Integer specifying the maximum framebuffer height in pixels.
    """

  @contextlib.contextmanager
  def make_current(self, width, height):
    """Context manager that makes this Renderer's OpenGL context current.

    Args:
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.

    Yields:
      None

    Raises:
      ValueError: If width > max_width, or height > max_height.
    """
    if width > self._max_width:
      raise ValueError('Maximal framebuffer width is {}. {} given.'
                       .format(self._max_width, width))
    if height > self._max_height:
      raise ValueError('Maximal framebuffer height is {}. {} given.'
                       .format(self._max_height, height))

    previous_context = self._before_make_current(width, height)
    try:
      yield
    finally:
      self._after_make_current(previous_context)

  @abc.abstractmethod
  def _before_make_current(self, width, height):
    """Called when entering the `make_current` context manager.

    Args:
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.

    Returns:
      Either a pointer to the previous OpenGL context to be passed to
      `_after_make_current`, or else None.
    """

  @abc.abstractmethod
  def _after_make_current(self, previous_context):
    """Called when exiting the `make_current` context manager.

    Args:
      previous_context: The return value of `_before_make_current`. This should
        either be a pointer to a previous OpenGL context to be made current, or
        else None.
    """

  @abc.abstractmethod
  def free_context(self):
    """Frees resources associated with this context."""

  def __del__(self):
    self.free_context()
