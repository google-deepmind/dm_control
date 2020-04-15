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

  def __init__(self, spawn_ratio=_SPAWN_RATIO, init_ball_z=_INIT_BALL_Z):
    self._spawn_ratio = spawn_ratio
    self._init_ball_z = init_ball_z

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

  def __call__(self, task, physics, random_state):
    spawn_range = np.asarray(task.arena.size) * self._spawn_ratio
    self._initialize_ball(task.ball, spawn_range, physics, random_state)
    for player in task.players:
      self._initialize_walker(player.walker, spawn_range, physics, random_state)
