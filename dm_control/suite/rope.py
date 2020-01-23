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

"""Point-mass domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_env import specs
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
import numpy as np
import random
import os
import math
_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('cloth_v0.xml'), common.ASSETS
  return common.read_model('rope.xml'),common.ASSETS
N_GEOMS = 3

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Rope(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit,n_frame_skip=1, rope_task=True,**environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

class Rope(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, init_flat=True):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._init_flat = init_flat
    super(Rope, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    return specs.BoundedArray(shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3)

  def get_geoms(self, physics):
      geoms = [physics.named.data.geom_xpos['G{}'.format(i)][:2] for i in range(self.n_geoms)]
      return np.array(geoms)

  def initialize_episode(self,physics):
    if not  self._init_flat:
        physics.named.data.xfrc_applied[['B3','B8','B10','B20'], :2] = np.random.uniform(-0.8, 0.8, size=8).reshape((4,2))
    super(Rope, self).initialize_episode(physics)

  def before_step(self, action, physics):
    physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))
    physics.named.data.qfrc_applied[:2]=0

    location = action[0] * 0.5 + 0.5 # to [-1, 1] to [0, 1]
    location = int(min(np.floor(location * N_GEOMS), N_GEOMS - 1))
    delta = action[1:] * 0.075

    action_idx = 'B{}'.format(location)
    geom_idx = 'G{}'.format(location)

    goal_position = delta + physics.named.data.geom_xpos[geom_idx, :2]
    dist = goal_position - physics.named.data.geom_xpos[geom_idx, :2]

    loop = 0
    while np.linalg.norm(dist) > 0.025:
      loop += 1
      if loop > 40:
        break
      physics.named.data.xfrc_applied[action_idx, :2] = dist * 1
      physics.step()
      self.after_step(physics)
      dist = goal_position - physics.named.data.geom_xpos[geom_idx,:2]

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self,physics):
    return 0.0
