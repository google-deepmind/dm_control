# Copyright 2020 The dm_control Authors.
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
"""A non-colliding sphere that is activated through touch."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control import mjcf


class TargetSphere(composer.Entity):
  """A non-colliding sphere that is activated through touch.

  Once the target has been reached, it remains in the "activated" state
  for the remainder of the current episode.

  The target is automatically reset to "not activated" state at episode
  initialization time.
  """

  def _build(self,
             radius=0.6,
             height_above_ground=1,
             rgb1=(0, 0.4, 0),
             rgb2=(0, 0.7, 0),
             specific_collision_geom_ids=None,
             name='target'):
    """Builds this target sphere.

    Args:
      radius: The radius (in meters) of this target sphere.
      height_above_ground: The height (in meters) of this target above ground.
      rgb1: A sequence of three floating point values between 0.0 and 1.0
        (inclusive) representing the color of the first element in the stripe
        pattern of the target.
      rgb2: A sequence of three floating point values between 0.0 and 1.0
        (inclusive) representing the color of the second element in the stripe
        pattern of the target.
      specific_collision_geom_ids: Only activate if collides with these geoms.
      name: The name of this entity.
    """
    self._mjcf_root = mjcf.RootElement(model=name)
    self._texture = self._mjcf_root.asset.add(
        'texture', name='target_sphere', type='cube',
        builtin='checker', rgb1=rgb1, rgb2=rgb2,
        width='100', height='100')
    self._material = self._mjcf_root.asset.add(
        'material', name='target_sphere', texture=self._texture)
    self._geom = self._mjcf_root.worldbody.add(
        'geom', type='sphere', name='geom', gap=2*radius,
        pos=[0, 0, height_above_ground], size=[radius], material=self._material)
    self._geom_id = -1
    self._activated = False
    self._specific_collision_geom_ids = specific_collision_geom_ids

  @property
  def geom(self):
    return self._geom

  @property
  def material(self):
    return self._material

  @property
  def activated(self):
    """Whether this target has been reached during this episode."""
    return self._activated

  def reset(self, physics):
    self._activated = False
    physics.bind(self._material).rgba[-1] = 1

  @property
  def mjcf_model(self):
    return self._mjcf_root

  def initialize_episode_mjcf(self, unused_random_state):
    self._activated = False

  def _update_activation(self, physics):
    if not self._activated:
      for contact in physics.data.contact:
        if self._specific_collision_geom_ids:
          has_specific_collision = (
              contact.geom1 in self._specific_collision_geom_ids or
              contact.geom2 in self._specific_collision_geom_ids)
        else:
          has_specific_collision = True
        if (has_specific_collision and
            self._geom_id in (contact.geom1, contact.geom2)):
          self._activated = True
          physics.bind(self._material).rgba[-1] = 0

  def initialize_episode(self, physics, unused_random_state):
    self._geom_id = physics.model.name2id(self._geom.full_identifier, 'geom')
    self._update_activation(physics)

  def after_substep(self, physics, unused_random_state):
    self._update_activation(physics)


class TargetSphereTwoTouch(composer.Entity):
  """A non-colliding sphere that is activated through touch.

  The target indicates if it has been touched at least once and touched at least
  twice this episode with a two-bit activated state tuple.  It remains activated
  for the remainder of the current episode.

  The target is automatically reset at episode initialization.
  """

  def _build(self,
             radius=0.6,
             height_above_ground=1,
             rgb_initial=((0, 0.4, 0), (0, 0.7, 0)),
             rgb_interval=((1., 1., .4), (0.7, 0.7, 0.)),
             rgb_final=((.4, 0.7, 1.), (0, 0.4, .7)),
             touch_debounce=.2,
             specific_collision_geom_ids=None,
             name='target'):
    """Builds this target sphere.

    Args:
      radius: The radius (in meters) of this target sphere.
      height_above_ground: The height (in meters) of this target above ground.
      rgb_initial: A tuple of two colors for the stripe pattern of the target.
      rgb_interval: A tuple of two colors for the stripe pattern of the target.
      rgb_final: A tuple of two colors for the stripe pattern of the target.
      touch_debounce: duration to not count second touch.
      specific_collision_geom_ids: Only activate if collides with these geoms.
      name: The name of this entity.
    """
    self._mjcf_root = mjcf.RootElement(model=name)
    self._texture_initial = self._mjcf_root.asset.add(
        'texture', name='target_sphere_init', type='cube',
        builtin='checker', rgb1=rgb_initial[0], rgb2=rgb_initial[1],
        width='100', height='100')
    self._texture_interval = self._mjcf_root.asset.add(
        'texture', name='target_sphere_inter', type='cube',
        builtin='checker', rgb1=rgb_interval[0], rgb2=rgb_interval[1],
        width='100', height='100')
    self._texture_final = self._mjcf_root.asset.add(
        'texture', name='target_sphere_final', type='cube',
        builtin='checker', rgb1=rgb_final[0], rgb2=rgb_final[1],
        width='100', height='100')
    self._material = self._mjcf_root.asset.add(
        'material', name='target_sphere_init', texture=self._texture_initial)
    self._geom = self._mjcf_root.worldbody.add(
        'geom', type='sphere', name='geom', gap=2*radius,
        pos=[0, 0, height_above_ground], size=[radius],
        material=self._material)
    self._geom_id = -1
    self._touched_once = False
    self._touched_twice = False
    self._touch_debounce = touch_debounce
    self._specific_collision_geom_ids = specific_collision_geom_ids

  @property
  def geom(self):
    return self._geom

  @property
  def material(self):
    return self._material

  @property
  def activated(self):
    """Whether this target has been reached during this episode."""
    return (self._touched_once, self._touched_twice)

  def reset(self, physics):
    self._touched_once = False
    self._touched_twice = False
    self._geom.material = self._material
    physics.bind(self._material).texid = physics.bind(
        self._texture_initial).element_id

  @property
  def mjcf_model(self):
    return self._mjcf_root

  def initialize_episode_mjcf(self, unused_random_state):
    self._touched_once = False
    self._touched_twice = False

  def _update_activation(self, physics):
    if not (self._touched_once and self._touched_twice):
      for contact in physics.data.contact:
        if self._specific_collision_geom_ids:
          has_specific_collision = (
              contact.geom1 in self._specific_collision_geom_ids or
              contact.geom2 in self._specific_collision_geom_ids)
        else:
          has_specific_collision = True
        if (has_specific_collision and
            self._geom_id in (contact.geom1, contact.geom2)):
          if not self._touched_once:
            self._touched_once = True
            self._touch_time = physics.time()
            physics.bind(self._material).texid = physics.bind(
                self._texture_interval).element_id
          if self._touched_once and (
              physics.time() > (self._touch_time + self._touch_debounce)):
            self._touched_twice = True
            physics.bind(self._material).texid = physics.bind(
                self._texture_final).element_id

  def initialize_episode(self, physics, unused_random_state):
    self._geom_id = physics.model.name2id(self._geom.full_identifier, 'geom')
    self._update_activation(physics)

  def after_substep(self, physics, unused_random_state):
    self._update_activation(physics)
