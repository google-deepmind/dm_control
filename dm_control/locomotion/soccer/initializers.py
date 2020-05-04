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

"""Soccer task episode initializers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import numpy as np
import six


_INIT_BALL_Z = 0.5
_SPAWN_RATIO = 0.6


@six.add_metaclass(abc.ABCMeta)
class Initializer(object):

  @abc.abstractmethod
  def __call__(self, task, physics, random_state):
    """Initialize episode for a task."""


class UniformInitializer(Initializer):
  """Uniformly initialize walkers and soccer ball over spawn_range."""

  def __init__(self,
               spawn_ratio=_SPAWN_RATIO,
               init_ball_z=_INIT_BALL_Z,
               max_collision_avoidance_retries=100):
    self._spawn_ratio = spawn_ratio
    self._init_ball_z = init_ball_z

    # Lazily initialize geom ids for contact avoidance.
    self._ball_geom_ids = None
    self._walker_geom_ids = None
    self._all_geom_ids = None
    self._max_retries = max_collision_avoidance_retries

  def _initialize_ball(self, ball, spawn_range, physics, random_state):
    x, y = random_state.uniform(-spawn_range, spawn_range)
    ball.set_pose(physics, [x, y, self._init_ball_z])
    # Note: this method is not always called immediately after `physics.reset()`
    #       so we need to explicitly zero out the velocity.
    ball.set_velocity(physics, velocity=0., angular_velocity=0.)

  def _initialize_walker(self, walker, spawn_range, physics, random_state):
    """Uniformly initialize walker in spawn_range."""
    walker.reinitialize_pose(physics, random_state)
    x, y = random_state.uniform(-spawn_range, spawn_range)
    (_, _, z), quat = walker.get_pose(physics)
    walker.set_pose(physics, [x, y, z], quat)
    rotation = random_state.uniform(-np.pi, np.pi)
    quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
    walker.shift_pose(physics, quaternion=quat)
    # Note: this method is not always called immediately after `physics.reset()`
    #       so we need to explicitly zero out the velocity.
    walker.set_velocity(physics, velocity=0., angular_velocity=0.)

  def _initialize_entities(self, task, physics, random_state):
    spawn_range = np.asarray(task.arena.size) * self._spawn_ratio
    self._initialize_ball(task.ball, spawn_range, physics, random_state)
    for player in task.players:
      self._initialize_walker(player.walker, spawn_range, physics, random_state)

  def _initialize_geom_ids(self, task, physics):
    self._ball_geom_ids = {physics.bind(task.ball.geom)}
    self._walker_geom_ids = []
    for player in task.players:
      walker_geoms = player.walker.mjcf_model.find_all('geom')
      self._walker_geom_ids.append(set(physics.bind(walker_geoms).element_id))

    self._all_geom_ids = set(self._ball_geom_ids)
    for walker_geom_ids in self._walker_geom_ids:
      self._all_geom_ids |= walker_geom_ids

  def _has_relevant_contact(self, contact, geom_ids):
    other_geom_ids = self._all_geom_ids - geom_ids
    if ((contact.geom1 in geom_ids and contact.geom2 in other_geom_ids) or
        (contact.geom2 in geom_ids and contact.geom1 in other_geom_ids)):
      return True
    return False

  def __call__(self, task, physics, random_state):
    # Initialize geom_ids for collision detection.
    if not self._all_geom_ids:
      self._initialize_geom_ids(task, physics)

    num_retries = 0
    while True:
      self._initialize_entities(task, physics, random_state)

      should_retry = False
      physics.forward()  # forward physics for contact resolution.
      for contact in physics.data.contact:
        if self._has_relevant_contact(contact, self._ball_geom_ids):
          should_retry = True
          break
        for walker_geom_ids in self._walker_geom_ids:
          if self._has_relevant_contact(contact, walker_geom_ids):
            should_retry = True
            break

      if not should_retry:
        break

      num_retries += 1
      if num_retries > self._max_retries:
        raise RuntimeError('UniformInitializer: `max_retries` (%d) exceeded.' %
                           self._max_retries)
