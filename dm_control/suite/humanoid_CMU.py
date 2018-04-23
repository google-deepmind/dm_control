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

"""Humanoid_CMU Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards

import numpy as np

_DEFAULT_TIME_LIMIT = 20
_CONTROL_TIMESTEP = 0.02

# Height of head above which stand reward is 1.
_STAND_HEIGHT = 1.4

# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10

SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('humanoid_CMU.xml'), common.ASSETS


@SUITE.add()
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, **kwargs):
  """Returns the Stand task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HumanoidCMU(move_speed=0, random=random)
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP, **kwargs)


@SUITE.add()
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, **kwargs):
  """Returns the Run task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = HumanoidCMU(move_speed=_RUN_SPEED, random=random)
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP, **kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the humanoid_CMU domain."""

  def thorax_upright(self):
    """Returns projection from y-axes of thorax to the z-axes of world."""
    return self.named.data.xmat['thorax', 'zy']

  def head_height(self):
    """Returns the height of the head."""
    return self.named.data.xpos['head', 'z']

  def center_of_mass_position(self):
    """Returns position of the center-of-mass."""
    return self.named.data.subtree_com['thorax']

  def center_of_mass_velocity(self):
    """Returns the velocity of the center-of-mass."""
    return self.named.data.subtree_linvel['thorax']

  def torso_vertical_orientation(self):
    """Returns the z-projection of the thorax orientation matrix."""
    return self.named.data.xmat['thorax', ['zx', 'zy', 'zz']]

  def joint_angles(self):
    """Returns the state without global orientation or position."""
    return self.data.qpos[7:]  # Skip the 7 DoFs of the free root joint.

  def extremities(self):
    """Returns end effector positions in egocentric frame."""
    torso_frame = self.named.data.xmat['thorax'].reshape(3, 3)
    torso_pos = self.named.data.xpos['thorax']
    positions = []
    for side in ('l', 'r'):
      for limb in ('hand', 'foot'):
        torso_to_limb = self.named.data.xpos[side + limb] - torso_pos
        positions.append(torso_to_limb.dot(torso_frame))
    return np.hstack(positions)


class HumanoidCMU(base.Task):
  """A task for the CMU Humanoid."""

  def __init__(self, move_speed, random=None):
    """Initializes an instance of `Humanoid_CMU`.

    Args:
      move_speed: A float. If this value is zero, reward is given simply for
        standing up. Otherwise this specifies a target horizontal velocity for
        the walking task.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._move_speed = move_speed
    super(HumanoidCMU, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets a random collision-free configuration at the start of each episode.

    Args:
      physics: An instance of `Physics`.
    """
    penetrating = True
    while penetrating:
      randomizers.randomize_limited_and_rotational_joints(
          physics, self.random)
      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0

  def get_observation(self, physics):
    """Returns a set of egocentric features."""
    obs = collections.OrderedDict()
    obs['joint_angles'] = physics.joint_angles()
    obs['head_height'] = physics.head_height()
    obs['extremities'] = physics.extremities()
    obs['torso_vertical'] = physics.torso_vertical_orientation()
    obs['com_velocity'] = physics.center_of_mass_velocity()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    standing = rewards.tolerance(physics.head_height(),
                                 bounds=(_STAND_HEIGHT, float('inf')),
                                 margin=_STAND_HEIGHT/4)
    upright = rewards.tolerance(physics.thorax_upright(),
                                bounds=(0.9, float('inf')), sigmoid='linear',
                                margin=1.9, value_at_margin=0)
    stand_reward = standing * upright
    small_control = rewards.tolerance(physics.control(), margin=1,
                                      value_at_margin=0,
                                      sigmoid='quadratic').mean()
    small_control = (4 + small_control) / 5
    if self._move_speed == 0:
      horizontal_velocity = physics.center_of_mass_velocity()[[0, 1]]
      dont_move = rewards.tolerance(horizontal_velocity, margin=2).mean()
      return small_control * stand_reward * dont_move
    else:
      com_velocity = np.linalg.norm(physics.center_of_mass_velocity()[[0, 1]])
      move = rewards.tolerance(com_velocity,
                               bounds=(self._move_speed, float('inf')),
                               margin=self._move_speed, value_at_margin=0,
                               sigmoid='linear')
      move = (5*move + 1) / 6
      return small_control * stand_reward * move
