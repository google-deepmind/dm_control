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

"""Hopper domain."""

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


SUITE = containers.TaggedTasks()

_CONTROL_TIMESTEP = .02  # (Seconds)

# Default duration of an episode, in seconds.
_DEFAULT_TIME_LIMIT = 20

# Minimal height of torso over foot above which stand reward is 1.
_STAND_HEIGHT = 0.6

# Hopping speed above which hop reward is 1.
_HOP_SPEED = 2


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('hopper.xml'), common.ASSETS


@SUITE.add('benchmarking')
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns a Hopper that strives to stand upright, balancing its pose."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hopper(hopping=False, random=random)
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP)


@SUITE.add('benchmarking')
def hop(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns a Hopper that strives to hop forward."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Hopper(hopping=True, random=random)
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Hopper domain."""

  def height(self):
    """Returns height of torso with respect to foot."""
    return (self.named.data.xipos['torso', 'z'] -
            self.named.data.xipos['foot', 'z'])

  def speed(self):
    """Returns horizontal speed of the Hopper."""
    return self.named.data.subtree_linvel['torso', 'x']

  def touch(self):
    """Returns the signals from two foot touch sensors."""
    return np.log1p(self.named.data.sensordata[['touch_toe', 'touch_heel']])


class Hopper(base.Task):
  """A Hopper's `Task` to train a standing and a jumping Hopper."""

  def __init__(self, hopping, random=None):
    """Initialize an instance of `Hopper`.

    Args:
      hopping: Boolean, if True the task is to hop forwards, otherwise it is to
        balance upright.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._hopping = hopping
    super(Hopper, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    self._timeout_progress = 0

  def get_observation(self, physics):
    """Returns an observation of positions, velocities and touch sensors."""
    obs = collections.OrderedDict()
    # Ignores horizontal position to maintain translational invariance:
    obs['position'] = physics.data.qpos[1:]
    obs['velocity'] = physics.velocity()
    obs['touch'] = physics.touch()
    return obs

  def get_reward(self, physics):
    """Returns a reward applicable to the performed task."""
    standing = rewards.tolerance(physics.height(), (_STAND_HEIGHT, 2))
    if self._hopping:
      hopping = rewards.tolerance(physics.speed(),
                                  bounds=(_HOP_SPEED, float('inf')),
                                  margin=_HOP_SPEED/2,
                                  value_at_margin=0.5,
                                  sigmoid='linear')
      return standing * hopping
    else:
      small_control = rewards.tolerance(physics.control(),
                                        margin=1, value_at_margin=0,
                                        sigmoid='quadratic').mean()
      small_control = (small_control + 4) / 5
      return standing * small_control
