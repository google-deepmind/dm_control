# Copyright 2019 The dm_control Authors.
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

"""Detects the presence of registered entities within a cuboidal region."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control import mjcf
import numpy as np

_RENDERED_HEIGHT_IN_2D_MODE = 0.02


class _Detection(object):

  __slots__ = ('entity', 'detected')

  def __init__(self, entity, detected=False):
    self.entity = entity
    self.detected = detected


class PositionDetector(composer.Entity):
  """Detects the presence of registered entities within a cuboidal region.

  An entity is considered "detected" if the `xpos` value of any one of its geom
  lies within the active region defined by this detector. Note that this is NOT
  a contact-based detector. Generally speaking, a geom will not be detected
  until it is already "half inside" the region.

  This detector supports both 2D and 3D modes. In 2D mode, the active region
  has an effective infinite height along the z-direction.

  This detector also provides an "inverted" detection mode, where an entity is
  detected when it is not inside the detector's region.
  """

  def _build(self,
             pos,
             size,
             inverted=False,
             visible=False,
             rgba=(1, 0, 0, 0.25),
             detected_rgba=(0, 1, 0, 0.25),
             name='position_detector'):
    """Builds the detector.

    Args:
      pos: The position at the center of this detector's active region. Should
        be an array-like object of length 3 in 3D mode, or length 2 in 2D mode.
      size: The half-lengths of this detector's active region. Should
        be an array-like object of length 3 in 3D mode, or length 2 in 2D mode.
      inverted: (optional) A boolean, whether to operate in inverted detection
        mode. If `True`, an entity is detected when it is not in the active
        region.
      visible: (optional) A boolean, whether this detector is visible by
        default in rendered images. If `False`, this detector's active zone
        is placed in MuJoCo rendering group 4, which is not rendered by default,
        but can be toggled on (e.g. in Domain Explorer) for debugging purposes.
      rgba: (optional) The color to render when nothing is detected.
      detected_rgba: (optional) The color to render when an entity is detected.
      name: (optional) XML element name of this position detector.

    Raises:
      ValueError: If the `pos` and `size` arrays do not have the same length.
    """
    if len(pos) != len(size):
      raise ValueError('`pos` and `size` should have the same length: '
                       'got {!r} and {!r}'.format(pos, size))

    self._inverted = inverted
    self._detected = False
    self._lower = np.array(pos) - np.array(size)
    self._upper = np.array(pos) + np.array(size)

    self._entities = []
    self._entity_geoms = {}

    self._rgba = np.asarray(rgba)
    self._detected_rgba = np.asarray(detected_rgba)

    render_pos = np.zeros(3)
    render_pos[:len(pos)] = pos

    render_size = np.full(3, _RENDERED_HEIGHT_IN_2D_MODE)
    render_size[:len(size)] = size

    self._mjcf_root = mjcf.RootElement(model=name)
    self._site = self._mjcf_root.worldbody.add(
        'site', name='detection_zone', type='box',
        pos=render_pos, size=render_size, rgba=self._rgba)
    if not visible:
      self._site.group = composer.SENSOR_SITES_GROUP

  def resize(self, pos, size):
    if len(pos) != len(size):
      raise ValueError('`pos` and `size` should have the same length: '
                       'got {!r} and {!r}'.format(pos, size))
    self._lower = np.array(pos) - np.array(size)
    self._upper = np.array(pos) + np.array(size)

    render_pos = np.zeros(3)
    render_pos[:len(pos)] = pos

    render_size = np.full(3, _RENDERED_HEIGHT_IN_2D_MODE)
    render_size[:len(size)] = size

    self._site.pos = render_pos
    self._site.size = render_size

  def set_colors(self, rgba, detected_rgba):
    self.set_color(rgba)
    self.set_detected_color(detected_rgba)

  def set_color(self, rgba):
    self._rgba[:3] = rgba
    self._site.rgba = self._rgba

  def set_detected_color(self, detected_rgba):
    self._detected_rgba[:3] = detected_rgba

  def set_position(self, physics, pos):
    physics.bind(self._site).pos = pos
    size = physics.bind(self._site).size[:3]
    self._lower = np.array(pos) - np.array(size)
    self._upper = np.array(pos) + np.array(size)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  def register_entities(self, *entities):
    for entity in entities:
      self._entities.append(_Detection(entity))
      self._entity_geoms[entity] = entity.mjcf_model.find_all('geom')

  def deregister_entities(self):
    self._entities = []

  @property
  def detected_entities(self):
    """A list of detected entities."""
    return [
        detection.entity for detection in self._entities if detection.detected]

  def initialize_episode_mjcf(self, unused_random_state):
    self._entity_geoms = {}
    for detection in self._entities:
      entity = detection.entity
      self._entity_geoms[entity] = entity.mjcf_model.find_all('geom')

  def initialize_episode(self, physics, unused_random_state):
    self._update_detection(physics)

  def after_substep(self, physics, unused_random_state):
    self._update_detection(physics)

  def _is_in_zone(self, xpos):
    return (np.all(self._lower < xpos[:len(self._lower)])
            and np.all(self._upper > xpos[:len(self._upper)]))

  def _update_detection(self, physics):
    previously_detected = self._detected
    self._detected = False
    for detection in self._entities:
      detection.detected = False
      for geom in self._entity_geoms[detection.entity]:
        if self._is_in_zone(physics.bind(geom).xpos) != self._inverted:
          detection.detected = True
          self._detected = True
          break

    if self._detected and not previously_detected:
      physics.bind(self._site).rgba = self._detected_rgba
    elif previously_detected and not self._detected:
      physics.bind(self._site).rgba = self._rgba

  def site_pos(self, physics):
    return physics.bind(self._site).pos

  @property
  def activated(self):
    return self._detected

  @property
  def upper(self):
    return self._upper

  @property
  def lower(self):
    return self._lower

  @property
  def mid(self):
    return (self._lower + self._upper) / 2.
