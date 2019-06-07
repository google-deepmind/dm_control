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

"""Task for a walker to move to a target."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.observation import observable as base_observable
from dm_control.composer.variation import distributions
import numpy as np

DEFAULT_DISTANCE_TOLERANCE_TO_TARGET = 1.0


class GoToTarget(composer.Task):
  """A task that requires a walker to move towards a target."""

  def __init__(self,
               walker,
               arena,
               moving_target=False,
               steps_before_moving_target=10,
               distance_tolerance=DEFAULT_DISTANCE_TOLERANCE_TO_TARGET,
               target_spawn_position=None,
               walker_spawn_position=None,
               walker_spawn_rotation=None,
               physics_timestep=0.005,
               control_timestep=0.025):
    """Initializes this task.

    Args:
      walker: an instance of `locomotion.walkers.base.Walker`.
      arena: an instance of `locomotion.arenas.floors.Floor`.
      moving_target: bool, Whether the target should move after receiving the
        walker reaches it.
      steps_before_moving_target: int, the number of steps before the target
        moves, if moving_target==True.
      distance_tolerance: Accepted to distance to the target position before
        providing reward.
      target_spawn_position: a sequence of 2 numbers, or a `composer.Variation`
        instance that generates such sequences, specifying the position at
        which the target is spawned at the beginning of an episode.
        If None, the entire arena is used to generate random target positions.
      walker_spawn_position: a sequence of 2 numbers, or a `composer.Variation`
        instance that generates such sequences, specifying the position at
        which the walker is spawned at the beginning of an episode.
        If None, the entire arena is used to generate random spawn positions.
      walker_spawn_rotation: a number, or a `composer.Variation` instance that
        generates a number, specifying the yaw angle offset (in radians) that is
        applied to the walker at the beginning of an episode.
      physics_timestep: a number specifying the timestep (in seconds) of the
        physics simulation.
      control_timestep: a number specifying the timestep (in seconds) at which
        the agent applies its control inputs (in seconds).
    """

    self._arena = arena
    self._walker = walker
    self._walker.create_root_joints(self._arena.attach(self._walker))

    arena_position = distributions.Uniform(
        low=-np.array(arena.size) / 2, high=np.array(arena.size) / 2)
    if target_spawn_position is not None:
      self._target_spawn_position = target_spawn_position
    else:
      self._target_spawn_position = arena_position

    if walker_spawn_position is not None:
      self._walker_spawn_position = walker_spawn_position
    else:
      self._walker_spawn_position = arena_position

    self._walker_spawn_rotation = walker_spawn_rotation

    self._distance_tolerance = distance_tolerance
    self._moving_target = moving_target
    self._steps_before_moving_target = steps_before_moving_target
    self._reward_step_counter = 0

    self._target = self.root_entity.mjcf_model.worldbody.add(
        'site',
        name='target',
        type='sphere',
        pos=(0., 0., 0.),
        size=(0.1,),
        rgba=(0.9, 0.6, 0.6, 1.0))

    enabled_observables = []
    enabled_observables += self._walker.observables.proprioception
    enabled_observables += self._walker.observables.kinematic_sensors
    enabled_observables += self._walker.observables.dynamic_sensors
    enabled_observables.append(self._walker.observables.sensors_touch)
    for observable in enabled_observables:
      observable.enabled = True

    walker.observables.add_egocentric_vector(
        'target',
        base_observable.Generic(lambda physics: physics.bind(self._target).pos),
        origin_callable=lambda physics: physics.bind(walker.root_body).xpos)

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

  @property
  def root_entity(self):
    return self._arena

  def target_position(self, physics):
    return np.array(physics.bind(self._target).pos)

  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate(random_state=random_state)

    target_x, target_y = variation.evaluate(
        self._target_spawn_position, random_state=random_state)
    self._target.pos = [target_x, target_y, 0.]

  def initialize_episode(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)
    if self._walker_spawn_rotation:
      rotation = variation.evaluate(
          self._walker_spawn_rotation, random_state=random_state)
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
    else:
      quat = None
    walker_x, walker_y = variation.evaluate(
        self._walker_spawn_position, random_state=random_state)
    self._walker.shift_pose(
        physics,
        position=[walker_x, walker_y, 0.],
        quaternion=quat,
        rotate_velocity=True)

    self._failure_termination = False
    walker_foot_geoms = set(self._walker.ground_contact_geoms)
    walker_nonfoot_geoms = [
        geom for geom in self._walker.mjcf_model.find_all('geom')
        if geom not in walker_foot_geoms]
    self._walker_nonfoot_geomids = set(
        physics.bind(walker_nonfoot_geoms).element_id)
    self._ground_geomids = set(
        physics.bind(self._arena.ground_geoms).element_id)
    self._ground_geomids.add(physics.bind(self._target).element_id)

  def _is_disallowed_contact(self, contact):
    set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
    return ((contact.geom1 in set1 and contact.geom2 in set2) or
            (contact.geom1 in set2 and contact.geom2 in set1))

  def should_terminate_episode(self, physics):
    return self._failure_termination

  def get_discount(self, physics):
    if self._failure_termination:
      return 0.
    else:
      return 1.

  def get_reward(self, physics):
    reward = 0.
    distance = np.linalg.norm(
        physics.bind(self._target).pos[:2] -
        physics.bind(self._walker.root_body).xpos[:2])
    if distance < self._distance_tolerance:
      reward = 1.
      if self._moving_target:
        self._reward_step_counter += 1
    return reward

  def after_step(self, physics, random_state):
    self._failure_termination = False
    for contact in physics.data.contact:
      if self._is_disallowed_contact(contact):
        self._failure_termination = True
        break
    if (self._moving_target and
        self._reward_step_counter >= self._steps_before_moving_target):

      # Reset the target position.
      target_x, target_y = variation.evaluate(
          self._target_spawn_position, random_state=random_state)
      physics.bind(self._target).pos = [target_x, target_y, 0.]

      # Reset the number of steps at the target for the moving target.
      self._reward_step_counter = 0
