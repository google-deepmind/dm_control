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

import math


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

CORNER_INDEX_ACTION=['B3','B8','B10','B20']
GEOM_INDEX=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('cloth_v0.xml'), common.ASSETS
  return common.read_model('rope_v2.xml'),common.ASSETS
W=64




@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Rope(randomize_gains=False, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit,n_frame_skip=1, rope_task=True,**environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""



class Rope(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, random_location=True):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._random_location = random_location
    super(Rope, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    if self._random_location:
      return specs.BoundedArray(
          shape=(2,), dtype=np.float, minimum=[-1.0] * 2, maximum=[1.0] * 2)
    else:
      return specs.BoundedArray(
          shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3
      )

  def initialize_episode(self,physics):

    #physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
    # physics.named.data.xfrc_applied['B10', :3] = np.array([0, 0, -2])
    # physics.named.data.xfrc_applied['B0_8', :3] = np.array([0.2,0.2,0.1])

    # self.dof_damping = np.concatenate([np.zeros((6)), np.ones((48)) * 0.002], axis=0)
    # self.body_mass = np.concatenate([np.zeros(1), np.ones(25) * 0.00563])
    # self.body_inertia = np.concatenate([np.zeros((1, 3)), np.tile(np.array([[4.58e-07,  4.58e-07,  1.8e-07]]), (25, 1))],
    #                                    axis=0)
    # self.geom_friction = np.tile(np.array([[1, 0.005, 0.001]]), (26, 1))
    # self.cam_pos = np.array([0, 0, 0.75])
    # self.cam_quat = np.array([1, 0, 0, 0])
    #
    # self.light_diffuse = np.array([0, 0, 0])
    # self.light_specular = np.array([0, 0, 0])
    # self.light_ambient = np.array([0, 0, 0])
    # self.light_castshadow = np.array([1])
    # self.light_dir = np.array([0, 0, -1])
    # self.light_pos = np.array([0, 0, 1])



    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image = image

    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :2] = np.random.uniform(-0.8, 0.8, size=8).reshape((4,2))
    super(Rope, self).initialize_episode(physics)

  def before_step(self, action, physics):

    physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))
    physics.named.data.qfrc_applied[:2]=0
    goal_position = action[:2]

    if self._random_location:
      assert len(action) == 2
      goal_position = action
    else:
      assert len(action) == 3
      goal_position = action[:2]
      location = action[2]

    goal_position = goal_position * 0.05
    location = self.current_loc
    if location is None:
      return

    corner_action = index
    corner_geom = index

    position = goal_position + physics.named.data.geom_xpos[corner_geom,:2]
    dist = position - physics.named.data.geom_xpos[corner_geom,:2]

    loop = 0
    while np.linalg.norm(dist) > 0.025:
      loop += 1
      if loop > 40:
        break
      physics.named.data.xfrc_applied[corner_action, :2] = dist * 20
      physics.step()
      self.after_step(physics)
      dist = position - physics.named.data.geom_xpos[corner_geom,:2]

  def get_termination(self,physics):
    if self.num_loc<1:
      return 1.0
    else:
      return None


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    location = self.sample_location(physics)
    self.current_loc = location
    if location is None:
      obs['location'] = np.tile(np.array([-1,-1]), 50).reshape(-1).astype('float32')
    #

    # location = self.current_loc
    else:
   # self.current_loc = location
       obs['location'] = np.tile(location, 50).reshape(-1).astype('float32')


    # obs['position'] = physics.position()
    # obs['velocity'] = physics.velocity()
    return obs


  def sample_location(self, physics):
    # obs=self.get_observation(physics)
    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    # image_dir = os.path.expanduser('~/softlearning')
    # image_path = os.path.join(image_dir,f'rope_v1.png')
    # imsave(image_path,image)
    self.image = image
  #  image_dim = image[:, :, 1].reshape((W, W, 1))
    location_range = np.transpose(np.where(np.all(image > 150, axis=2)))
    self.location_range = location_range
    num_loc = np.shape(location_range)[0]
    self.num_loc = num_loc
    if num_loc == 0 :
      return None
    index = np.random.randint(num_loc, size=1)
    location = location_range[index]

    return location
  def get_reward(self,physics):
    #image_dim = self.image[:,:,1].reshape((W,W,1))
    current_mask = np.all(self.image>150,axis=2).astype(int)
    reward_mask = current_mask 
    line = np.linspace(0,31,num=32)*(-0.5)
    column = np.concatenate([np.flip(line),line])
    reward =np.sum(reward_mask* np.exp(column).reshape((W,1)))/111.0
    print(reward)
    return reward
  # def l2_norm_dist_2d(self, xpos):
  #   _, _, _, _, error = linregress(xpos[:, 0], xpos[:, 1])
  #   return error

  # def get_reward(self,physics):
  #   geom_xpos= physics.named.data.geom_xpos[5:]
  #   dist = self.l2_norm_dist_2d(geom_xpos)
  #   reward = -dist
  #   print(reward)
  #   return reward


#  def get_reward(self,physics):
    # geom_xpos = physics.named.data.geom_xpos[5:]
#    reward_dist=np.linalg.norm(physics.named.data.geom_xpos['G0',:2]-physics.named.data.geom_xpos['G24',:2])
#    reward=reward_dist
    # sum_dist_1=np.sum(physics.named.data.geom_xpos[5:,1]**2)
    # for i in range(25):
    #   # sum_dist+=np.linalg.norm(physics.named.data.geom_xpos['G'+str(i),:2]-np.array([0.2-0.02*i,0]))
    #   sum_dist_1 += np.linalg.norm(physics.named.data.geom_xpos[i, :2] - np.array([-0.2 + 0.02 * i, 0]))
    #
    # reward= -sum_dist_1
#    print(reward)

#    return reward
  # def get_reward(self, physics):
  #   """Returns a reward to the agent."""
  #   B0_pos=physics.named.data.geom_xpos['G0']
  #   B20_pos=physics.named.data.geom_xpos['G10']
  #
  #   reward_dist_goal = np.linalg.norm(B0_pos - B20_pos)
  #     # reward_ctrl=ctrl_cost_coeff*np.square(action).sum()
  #
  #
  #   reward = reward_dist_goal
  #
  #   return reward
  # def get_reward(self,physics):
  #      # ori
  #     image_dim = self.image[:,:,1].reshape((W,W,1))
  #     current_mask=(~np.all(image_dim > 100, axis=2)).astype(int)
  #     area=np.sum(current_mask*self.mask)
  #     reward=area/np.sum(self.mask)
  #
  #     return reward
