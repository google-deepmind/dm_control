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
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np
import random

_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('cloth_v0.xml'), common.ASSETS


# @SUITE.add('benchmarking', 'easy')
# def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the easy point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = PointMass(randomize_gains=False, random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, **environment_kwargs)
#
#
# @SUITE.add()
# def hard(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
#   """Returns the hard point_mass task."""
#   physics = Physics.from_xml_string(*get_model_and_assets())
#   task = PointMass(randomize_gains=True, random=random)
#   environment_kwargs = environment_kwargs or {}
#   return control.Environment(
#       physics, task, time_limit=time_limit, **environment_kwargs)


@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs:
  """Returns the easy cloth task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, special_task=True, **environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""



class Cloth(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, pixel_size=64, camera_id=0,
               reward='area'):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self.pixel_size = pixel_size
    self.camera_id = camera_id
    self.reward = reward
    print('pixel_size', self.pixel_size, 'camera_id', self.camera_id, 'reward', self.reward)
    # self.action_spec=specs.BoundedArray(
    # shape=(2,), dtype=np.float, minimum=0.0, maximum=1.0)
    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArray` matching the `physics` actuators."""
    # return specs.BoundedArray(
    # shape=(12,), dtype=np.float, minimum=[-5.0]*12 ,maximum=[5.0]*12)
    # return specs.BoundedArray(
    #     shape=(3,), dtype=np.float, minimum=[-5.0] * 3, maximum=[5.0] * 3)


    return specs.BoundedArray(
      shape=(12,), dtype=np.float32, minimum=[-1.0] * 12, maximum=[1.0] * 12)

  def initialize_episode(self,physics):

    # randomizers.randomize_limited_and_rotational_joints(physics, self.random)
    # for joint in _JOINTS:
    #   physics.named.data.qpos[joint] = self.random.uniform(-.2, .2)


    physics.data.xpos[1:,:2]=physics.data.xpos[1:,:2]+self.random.uniform(-.3, .3)
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])

    # quat = np.random.RandomState(100).rand(4)
    # quat /= np.linalg.norm(quat)

    # qpos['r1'][3:] =  np.array([.1,.1,.1,.1])
    # for _ in range(250):
    #     self.before_step(np.zeros(12),physics)

       # if physics.named.data.qpos[i] < 0:
       #     physics.named.data.qpos[i]+=0.2
    # physics.named.model.geom_pos['G3_5','z']=0
    # physics.named.data.qpos['J1_0_8']=3
    # self.random.uniform(-.2, .2)
    # physics.named.data.xfrc_applied['B0_0', :3] = np.array([5,5,.1])
    # physics.named.data.xfrc_applied['B0_8', :3] = np.array([5,-5,.1])
    # physics.named.data.xfrc_applied['B8_8', :3] = np.array([-5,-5,.1])
    # physics.named.data.xfrc_applied['B8_0', :3] = np.array([-5,5,.1])
    # for i in range(4):
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)
    # ndof = physics.model.nq
    # unit = self.random.randn(ndof)
    # physics.data.qpos[:] = np.sqrt(2) * unit / np.linalg.norm(unit)

    # physics.named.model.geom_pos['G0_0',]=physics.named.model.geom_pos['G0_0',]+\
    #                                       self.random.uniform(-.1, .1, size=2)
    # physics.named.model.geom_pos['G0_8',] = physics.named.model.geom_pos['G0_8',]+\
    #                                         self.random.uniform(-.1, .1, size=82)
    # physics.named.model.geom_pos['table',]=np.zeros(3)
    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #     # Support legacy internal code.
  #     action_corner=action[3:]
  #     index=np.where(action_corner==1)
  #     print(index)
  #     action=action[:3,]
  #     action = getattr(action, "continuous_actions", action)
      # z_axises=physics.data.geom_xpos[CORNER_INDEX_POSITION,2]
      # # # print(z_axises)
      # indices=np.argwhere(z_axises == np.amax(z_axises)).flatten().tolist()
      #
      # if len(indices)!=1:
      #    index=indices[random.randint(0,len(indices)-1)]
      #    # index=np.random.choice(indices)
      # #    # import ipdb;ipdb.set_trace()
      # else:
      #    index=indices[0]
      # physics.named.data.xfrc_applied[CORNER_INDEX_ACTION[index],:3]=action
      action=action.reshape(4,3)
      physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=action

  #     physics.named.data.xfrc_applied['B0_0',:3]=action

      # physics.set_control(action)
      # physics.named.data.xfrc_applied['B0_0',:3]=action[:3]
      # physics.named.data.xfrc_applied['B8_0',:3 ]=action[3:6]
      # physics.named.data.xfrc_applied['B0_8',:3]=action[6:9]
      # physics.named.data.xfrc_applied['B8_8',:3 ]=action[9:]
      # physics.named.data.xfrc_applied['B0_0', 2] = 9.5
      # physics.named.data.xfrc_applied['B0_0',:2]=action[:2]
      # physics.named.data.xfrc_applied['B8_0',:2]=action[2:]
      # physics.named.data.xfrc_applied['B8_0', 2] = 9.5


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position().astype(np.float32)
    obs['velocity'] = physics.velocity().astype(np.float32)
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    if self.reward == 'area':
        pixels = physics.render(width=self.pixel_size, height=self.pixel_size,
                                camera_id=self.camera_id)
        segmentation = (pixels < 100).any(axis=-1).astype('float32')
        reward = segmentation.mean()
        return reward, dict()
    elif self.reward == 'diagonal':
        pos_ll=physics.data.geom_xpos[86,:2]
        pos_lr=physics.data.geom_xpos[81,:2]
        pos_ul=physics.data.geom_xpos[59,:2]
        pos_ur=physics.data.geom_xpos[54,:2]
        diag_dist1=np.linalg.norm(pos_ll-pos_ur)
        diag_dist2=np.linalg.norm(pos_lr-pos_ul)
        reward=diag_dist1+diag_dist2
        return reward, dict()
    raise ValueError(self.reward)
