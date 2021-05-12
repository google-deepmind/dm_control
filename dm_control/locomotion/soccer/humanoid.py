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

"""Walkers based on an actuated jumping ball."""

import enum
import os

from dm_control.locomotion.walkers import cmu_humanoid
import numpy as np


_ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'humanoid')
_MAX_WALKER_ID = 10
_INVALID_WALKER_ID = 'walker_id must be in [0-{}], got: {{}}.'.format(
    _MAX_WALKER_ID)

_INTERIOR_GEOMS = frozenset({
    'lhipjoint', 'rhipjoint', 'lfemur', 'lowerback', 'upperback', 'rclavicle',
    'lclavicle', 'thorax', 'lhumerus', 'root_geom', 'lowerneck', 'rhumerus',
    'rfemur'
})


def _add_visual_only_geoms(mjcf_root):
  """Introduce visual only geoms to complement the `JERSEY` visual."""
  lowerneck = mjcf_root.find('body', 'lowerneck')
  neck_offset = 0.066 - 0.0452401
  lowerneck.add(
      'geom',
      name='halfneck',
      # shrink neck radius from 0.06 to 0.05 else it pokes through shirt
      size=(0.05, 0.02279225 - neck_offset),
      pos=(-0.00165071, 0.0452401 + neck_offset, 0.00534359),
      quat=(0.66437, 0.746906, 0.027253, 0),
      mass=0.,
      contype=0,
      conaffinity=0,
      rgba=(.7, .5, .3, 1))
  lhumerus = mjcf_root.find('body', 'lhumerus')
  humerus_offset = 0.20 - 0.138421
  lhumerus.add(
      'geom',
      name='lelbow',
      size=(0.035, 0.1245789 - humerus_offset),
      pos=(0.0, -0.138421 - humerus_offset, 0.0),
      quat=(0.612372, -0.612372, 0.353553, 0.353553),
      mass=0.,
      contype=0,
      conaffinity=0,
      rgba=(.7, .5, .3, 1))
  rhumerus = mjcf_root.find('body', 'rhumerus')
  humerus_offset = 0.20 - 0.138421
  rhumerus.add(
      'geom',
      name='relbow',
      size=(0.035, 0.1245789 - humerus_offset),
      pos=(0.0, -0.138421 - humerus_offset, 0.0),
      quat=(0.612372, -0.612372, -0.353553, -0.353553),
      mass=0.,
      contype=0,
      conaffinity=0,
      rgba=(.7, .5, .3, 1))
  lfemur = mjcf_root.find('body', 'lfemur')
  femur_offset = 0.384 - 0.202473
  lfemur.add(
      'geom',
      name='lknee',
      # shrink knee radius from 0.06 to 0.055 else it pokes through short
      size=(0.055, 0.1822257 - femur_offset),
      pos=(-5.0684e-08, -0.202473 - femur_offset, 0),
      quat=(0.696364, -0.696364, -0.122788, -0.122788),
      mass=0.,
      contype=0,
      conaffinity=0,
      rgba=(.7, .5, .3, 1))
  rfemur = mjcf_root.find('body', 'rfemur')
  femur_offset = 0.384 - 0.202473
  rfemur.add(
      'geom',
      name='rknee',
      # shrink knee radius from 0.06 to 0.055 else it pokes through short
      size=(0.055, 0.1822257 - femur_offset),
      pos=(-5.0684e-08, -0.202473 - femur_offset, 0),
      quat=(0.696364, -0.696364, 0.122788, 0.122788),
      mass=0.,
      contype=0,
      conaffinity=0,
      rgba=(.7, .5, .3, 1))


class Humanoid(cmu_humanoid.CMUHumanoidPositionControlled):
  """A CMU humanoid walker specialised visually for soccer."""

  class Visual(enum.Enum):
    GEOM = 1
    JERSEY = 2

  def _build(self,
             visual,
             marker_rgba,
             walker_id=None,
             initializer=None,
             name='walker'):
    """Build a soccer-specific Humanoid walker."""
    if not isinstance(visual, Humanoid.Visual):
      raise ValueError('`visual` must be one of `Humanoid.Visual`.')

    if len(marker_rgba) != 4:
      raise ValueError('`marker_rgba` must be a sequence of length 4.')

    if walker_id is None and visual != Humanoid.Visual.GEOM:
      raise ValueError(
          '`walker_id` must be set unless `visual` is set to `Visual.GEOM`.')

    if walker_id is not None and not 0 <= walker_id <= _MAX_WALKER_ID:
      raise ValueError(_INVALID_WALKER_ID.format(walker_id))

    if visual == Humanoid.Visual.JERSEY:
      team = 'R' if marker_rgba[0] > marker_rgba[2] else 'B'
      marker_rgba = None  # disable geom coloring for None geom visual.
    else:
      marker_rgba[-1] = .7

    super(Humanoid, self)._build(
        marker_rgba=marker_rgba,
        initializer=initializer,
        include_face=True)

    self._mjcf_root.model = name

    # Changes to humanoid geoms for visual improvements.
    # Hands: hide hand geoms and add slightly larger visual geoms.
    for hand_name in ['lhand', 'rhand']:
      hand = self._mjcf_root.find('body', hand_name)
      for geom in hand.find_all('geom'):
        geom.rgba = (0, 0, 0, 0)
        if geom.name == hand_name:
          geom_size = geom.size * 1.3  # Palm rescaling.
        else:
          geom_size = geom.size * 1.5  # Finger rescaling.
        geom.parent.add(
            'geom',
            name=geom.name + '_visual',
            type=geom.type,
            quat=geom.quat,
            mass=0,
            contype=0,
            conaffinity=0,
            size=geom_size,
            pos=geom.pos * 1.5)

    # Lighting: remove tracking light as we have multiple walkers in the scene.
    tracking_light = self._mjcf_root.find('light', 'tracking_light')
    tracking_light.remove()

    if visual == Humanoid.Visual.JERSEY:
      shirt_number = walker_id + 1
      self._mjcf_root.asset.add(
          'texture',
          name='skin',
          type='2d',
          file=os.path.join(_ASSETS_PATH, f'{team}_{walker_id + 1:02d}.png'))
      self._mjcf_root.asset.add('material', name='skin', texture='skin')
      self._mjcf_root.asset.add(
          'skin',
          name='skin',
          file=os.path.join(_ASSETS_PATH, 'jersey.skn'),
          material='skin')
      for geom in self._mjcf_root.find_all('geom'):
        if geom.name in _INTERIOR_GEOMS:
          geom.rgba = (0.0, 0.0, 0.0, 0.0)
      _add_visual_only_geoms(self._mjcf_root)

    # Initialize previous action.
    self._prev_action = np.zeros(shape=self.action_spec.shape,
                                 dtype=self.action_spec.dtype)

  @property
  def marker_geoms(self):
    """Returns a sequence of marker geoms to be colored visually."""
    marker_geoms = []

    face = self._mjcf_root.find('geom', 'face')
    if face is not None:
      marker_geoms.append(face)

    marker_geoms += self._mjcf_root.find('body', 'rfoot').find_all('geom')
    marker_geoms += self._mjcf_root.find('body', 'lfoot').find_all('geom')
    return marker_geoms + [
        self._mjcf_root.find('geom', 'lowerneck'),
        self._mjcf_root.find('geom', 'lclavicle'),
        self._mjcf_root.find('geom', 'rclavicle'),
        self._mjcf_root.find('geom', 'thorax'),
        self._mjcf_root.find('geom', 'upperback'),
        self._mjcf_root.find('geom', 'lowerback'),
        self._mjcf_root.find('geom', 'rfemur'),
        self._mjcf_root.find('geom', 'lfemur'),
        self._mjcf_root.find('geom', 'root_geom'),
    ]

  def initialize_episode(self, physics, random_state):
    self._prev_action = np.zeros(shape=self.action_spec.shape,
                                 dtype=self.action_spec.dtype)

  def apply_action(self, physics, action, random_state):
    super().apply_action(physics, action, random_state)

    # Updates previous action.
    self._prev_action[:] = action

  @property
  def prev_action(self):
    return self._prev_action
