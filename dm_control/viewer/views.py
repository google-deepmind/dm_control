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
"""Components and views that render custom images into Mujoco render frame."""

import abc
import enum

from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper import util
from dm_control.viewer import renderer
import numpy as np

enums = mjbindings.enums
mjlib = mjbindings.mjlib
types = mjbindings.types


class PanelLocation(enum.Enum):
  TOP_LEFT = enums.mjtGridPos.mjGRID_TOPLEFT
  TOP_RIGHT = enums.mjtGridPos.mjGRID_TOPRIGHT
  BOTTOM_LEFT = enums.mjtGridPos.mjGRID_BOTTOMLEFT
  BOTTOM_RIGHT = enums.mjtGridPos.mjGRID_BOTTOMRIGHT


class BaseViewportView(metaclass=abc.ABCMeta):
  """Base abstract view class."""

  @abc.abstractmethod
  def render(self, context, viewport, location):
    """Renders the view on screen.

    Args:
      context: MjrContext instance.
      viewport: Viewport instance.
      location: Value defined in PanelLocation enum.
    """
    pass


class ColumnTextModel(metaclass=abc.ABCMeta):
  """Data model that returns 2 columns of text."""

  @abc.abstractmethod
  def get_columns(self):
    """Returns the text to display in two columns.

    Returns:
      Returns an iterable of tuples of 2 strings. Each tuple has format
      (left_column_label, right_column_label).
    """
    pass


class ColumnTextView(BaseViewportView):
  """A view displayed in Mujoco render window."""

  def __init__(self, model):
    """Instance initializer.

    Args:
      model: Instance of ColumnTextModel.
    """
    self._model = model

  def render(self, context, viewport, location):
    """Renders the overlay on screen.

    Args:
      context: MjrContext instance.
      viewport: Viewport instance.
      location: Value defined in PanelLocation enum.
    """
    columns = self._model.get_columns()
    if not columns:
      return

    columns = np.asarray(columns)
    left_column = '\n'.join(columns[:, 0])
    right_column = '\n'.join(columns[:, 1])
    mjlib.mjr_overlay(
        enums.mjtFont.mjFONT_NORMAL, location.value,
        viewport.mujoco_rect, util.to_binary_string(left_column),
        util.to_binary_string(right_column), context.ptr)


class MujocoDepthBuffer(renderer.Component):
  """Displays the contents of the scene's depth buffer."""

  def __init__(self):
    self._depth_buffer = np.empty((1, 1), np.float32)

  def render(self, context, viewport):
    """Renders the overlay on screen.

    Args:
      context: MjrContext instance.
      viewport: MJRRECT instance.
    """
    width_adjustment = viewport.width % 4
    rect_shape = (viewport.width - width_adjustment, viewport.height)

    if self._depth_buffer is None or self._depth_buffer.shape != rect_shape:
      self._depth_buffer = np.empty(
          (viewport.width, viewport.height), np.float32)

    mjlib.mjr_readPixels(
        None, self._depth_buffer, viewport.mujoco_rect, context.ptr)

    # Subsample by 4, convert to RGB, and cast to unsigned bytes.
    depth_rgb = np.repeat(self._depth_buffer[::4, ::4, None] * 255, 3,
                          -1).astype(np.ubyte)

    pos = types.MJRRECT(
        int(3 * viewport.width / 4) + width_adjustment, 0,
        int(viewport.width / 4), int(viewport.height / 4))
    mjlib.mjr_drawPixels(depth_rgb, None, pos, context.ptr)


class ViewportLayout(renderer.Component):
  """Layout manager for the render viewport.

  Allows to create a viewport layout by injecting renderer component even in
  absence of a renderer, and then easily reattach it between renderers.
  """

  def __init__(self):
    """Instance initializer."""
    self._views = dict()

  def __len__(self):
    return len(self._views)

  def __contains__(self, key):
    value = self._views.get(key, None)
    return value is not None

  def add(self, view, location):
    """Adds a new view.

    Args:
      view: renderer.BaseViewportView instance.
      location: Value defined in PanelLocation enum, location of the view in the
        viewport.
    """
    if not isinstance(view, BaseViewportView):
      raise TypeError(
          'View added to this layout needs to implement BaseViewportView.')
    self._views[view] = location

  def remove(self, view):
    """Removes a view.

    Args:
      view: renderer.BaseViewportView instance.
    """
    self._views.pop(view, None)

  def clear(self):
    """Removes all attached components."""
    self._views = dict()

  def render(self, context, viewport):
    """Renders the overlay on screen.

    Args:
      context: MjrContext instance.
      viewport: MJRRECT instance.
    """
    for view, location in self._views.items():
      view.render(context, viewport, location)
