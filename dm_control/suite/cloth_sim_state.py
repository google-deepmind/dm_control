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

"""Planar Stacker domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from dm_control import mujoco
from dm_env import specs
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools
import os
from imageio import imsave
from PIL import Image,ImageColor
from lxml import etree
import numpy as np
import math
_TOL = 1e-13
_CLOSE = .01    # (Meters) Distance below which a thing is considered close.
_CONTROL_TIMESTEP = .02  # (Seconds)
_TIME_LIMIT = 30  # (Seconds)



CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
CORNER_INDEX_GEOM=['G0_0','G0_8','G8_0','G8_8']



SUITE = containers.TaggedTasks()

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cloth_gripper.xml'), common.ASSETS



@SUITE.add('hard')
def easy(fully_observable=True, time_limit=_TIME_LIMIT, random=None,
            environment_kwargs=None):
  """Returns stacker task with 2 boxes."""

  physics=Physics.from_xml_string(*get_model_and_assets())

  task = Stack(randomize_gains=False,random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, control_timestep=_CONTROL_TIMESTEP,special_task=True,time_limit=time_limit,
      **environment_kwargs)

class Physics(mujoco.Physics):
  """Physics with additional features for the Planar Manipulator domain."""

class Stack(base.Task):
  """A Stack `Task`: stack the boxes."""

  def __init__(self, randomize_gains, random=None, mode='corners'):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._mode = mode
    self._current_loc = np.zeros((2,))
    print('mode', self._mode)

    super(Stack, self).__init__(random=random)

  def initialize_episode(self, physics):
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0, 0, -2])
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = np.random.uniform(-.3, .3, size=3)
    super(Stack, self).initialize_episode(physics)


  def action_spec(self, physics):
    """Returns a `BoundedArray` matching the `physics` actuators."""
    return specs.BoundedArray(
      shape=(3,), dtype=np.float, minimum=[-1.0,-1.0,-1.0] , maximum=[1.0,1.0,1.0])


  def before_step(self, action, physics):
    """Sets the control signal for the actuators to values in `action`."""
    # clear previous xfrc_force
    physics.named.data.xfrc_applied[:, :3] = np.zeros((3,))

    # scale the position to be a normal range
    goal_position = action[:3] * 0.1
    x = int(self._current_loc % 9)
    y = int(self._current_loc // 9)

    action_id = 'B{}_{}'.format(x, y)
    geom_id = 'G{}_{}'.format(x, y)

    # apply consecutive force to move the point to the target position
    position = goal_position + physics.named.data.geom_xpos[geom_id]
    dist = position - physics.named.data.geom_xpos[geom_id]

    loop = 0
    while np.linalg.norm(dist) > 0.025:
      loop += 1
      if loop > 40:
        break
      physics.named.data.xfrc_applied[action_id,:3] = dist * 20
      physics.step()
      self.after_step(physics)
      dist = position - physics.named.data.geom_xpos[geom_id]

  def get_observation(self, physics):
    """Returns either features or only sensors (to be used with pixels)."""
    obs = collections.OrderedDict()
    obs['position'] = physics.data.geom_xpos[6:, :2].astype('float32').reshape(-1)
    self._current_loc = self._generate_loc()
    obs['force_location'] = np.tile(self._current_loc , 50).reshape(-1).astype('float32')
    return obs

  def _generate_loc(self):
      if self._mode == 'corners':
          loc = np.random.choice([0, 8, 72, 80])
      elif self._mode == 'border':
          loc = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 17, 18, 26,
                                  27, 35, 36, 44, 45, 53, 54, 62, 63, 71,
                                  72, 73, 74, 75, 76, 77, 78, 79, 80])
      elif self._mode == '9x9':
          loc = np.random.choice(81)
      elif self._mode == '5x5':
          loc = np.random.choice([0, 2, 4, 6, 8, 18, 20, 22, 24, 26,
                                  36, 38, 40, 42, 44, 54, 56, 58, 60, 62,
                                  72, 74, 76, 78, 80])
      elif self._mode == '3x3':
          loc = np.random.choice([0, 4, 8, 36, 40, 44, 72, 76, 80])
      elif self._mode == 'inner_border':
          loc = np.random.choice([10, 11, 12, 13, 14, 15, 16, 19, 25,
                                  28, 34, 37, 43, 46, 52, 55, 61, 64,
                                  65, 66, 67, 68, 69, 70])
      else:
          raise Exception(self.mode)
      return loc

  def get_reward(self,physics):
      dist_sum=0
      for i in range(9):
          for j in range(9):
              index='G'+str(i)+'_'+str(j)
              geom_dist=np.sum(abs(physics.named.data.geom_xpos[index]-np.array([-0.09+0.03*i,-0.15+0.03*j,0])))
              dist_sum += geom_dist
      dist_sum = dist_sum/81

      return -dist_sum, dict()
