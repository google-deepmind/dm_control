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
import mujoco_py
import os
import math
_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

CORNER_INDEX_ACTION=['B3','B8','B10','B20']
GEOM_INDEX=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('cloth_v0.xml'), common.ASSETS
  return common.read_model('rope_sac.xml'),common.ASSETS
W=64




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

  def __init__(self, randomize_gains, random=None, maxq=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._maxq = maxq

    super(Rope, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""

    return specs.BoundedArray(
        shape=(4,), dtype=np.float, minimum=[-1.0] * 4, maximum=[1.0] * 4)

  def initialize_episode(self,physics):
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
    action = action.reshape((2,-1))
    if self._maxq:
        locations = np.round((action[:, :2] * 0.5 + 0.5) * 63).astype('int32')
        goal_positions = action[:, 2:]
    else:
        goal_positions = action
        locations = self.current_loc.reshape((2,-1))
    goal_positions = goal_positions * 0.05

    # computing the mapping from geom_xpos to location in image
    cam_fovy = physics.named.model.cam_fovy['fixed']
    f = 0.5 * W / math.tan(cam_fovy * math.pi / 360)
    cam_matrix = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
    cam_mat = physics.named.data.cam_xmat['fixed'].reshape((3, 3))
    cam_pos = physics.named.data.cam_xpos['fixed'].reshape((3, 1))
    cam = np.concatenate([cam_mat, cam_pos], axis=1)
    num_bodies = len(physics.data.geom_xpos)
    cam_pos_all = np.zeros((num_bodies, 3, 1))
    for i in range(num_bodies):
      geom_xpos_added = np.concatenate([physics.data.geom_xpos[i], np.array([1])]).reshape((4, 1))
      cam_pos_all[i] = cam_matrix.dot(cam.dot(geom_xpos_added)[:3])

    # cam_pos_xy=cam_pos_all[5:,:]
    cam_pos_xy = np.rint(cam_pos_all[:, :2].reshape((num_bodies, 2)) / cam_pos_all[:, 2])
    cam_pos_xy = cam_pos_xy.astype(int)
    cam_pos_xy[:, 1] = W - cam_pos_xy[:, 1]

    epsilon = 4
    possible_index = []
    possible_z = []
    for i in range(num_bodies):
        for j in range(num_bodies):
            # flipping the x and y to make sure it corresponds to the real location
            if abs(cam_pos_xy[i][0] - locations[0][1]) < epsilon and abs(cam_pos_xy[i][1] - locations[0][0]) < epsilon and i > 4 and np.all(cam_pos_xy[i] < W) and np.all(cam_pos_xy[i] >= 0) and \
               abs(cam_pos_xy[j][0] - locations[1][1]) < epsilon and abs(cam_pos_xy[j][1] - locations[1][0]) < epsilon and j > 4 and np.all(cam_pos_xy[j] < W) and np.all(cam_pos_xy[j] >= 0) \
               and i != j:
                possible_index.append((i,j))
                possible_z.append((physics.data.geom_xpos[i, 2], physics.data.geom_xpos[j, 2]))

    if possible_index != []:
          left_index, right_index = possible_index[possible_z.index(max(possible_z, key=lambda x: np.mean(x)))]

          left_action, left_geom = left_index - 4, left_index
          right_action, right_geom = right_index - 4, right_index

          # apply consecutive force to move the point to the target position
          left_position = goal_positions[0] + physics.named.data.geom_xpos[left_geom, :2]
          left_dist = left_position - physics.named.data.geom_xpos[left_geom, :2]

          right_position = goal_positions[1] + physics.named.data.geom_xpos[right_geom, :2]
          right_dist = right_position - physics.named.data.geom_xpos[right_geom, :2]

          loop = 0
          while np.linalg.norm(np.vstack((left_dist,right_dist))) > 0.025:
            loop += 1
            if loop > 40:
                # print(np.linalg.norm(left_dist), np.linalg.norm(right_dist), 'Timeout exceeded.')
                break
            physics.named.data.xfrc_applied[left_action, :2] = right_dist * 20
            physics.named.data.xfrc_applied[right_action, :2] = left_dist * 20
            physics.step()
            self.after_step(physics)
            left_dist = left_position - physics.named.data.geom_xpos[left_geom, :2]
            right_dist = right_position - physics.named.data.geom_xpos[right_geom, :2]

  def get_termination(self,physics):
    if self.num_loc<1:
      return 1.0
    else:
      return None


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    if self._maxq:
        location = [-1, 1]
        render_kwargs = {}
        render_kwargs['camera_id'] = 0
        render_kwargs['width'] = W
        render_kwargs['height'] = W
        image = physics.render(**render_kwargs)

        self.image = image

        location_range = np.transpose(np.where(np.all(image > 150, axis=2)))
        self.location_range = location_range
        num_loc = np.shape(location_range)[0]
        self.num_loc = num_loc
    else:
        location = self.sample_location(physics)
    self.current_loc = location

    if self.current_loc is None:
        obs['location'] = np.tile([-1, -1, -1, -1], 50).reshape(-1).astype('float32') / 63
    else:
        obs['location'] = np.tile(location, 50).reshape(-1).astype('float32') / 63

    return obs


  def sample_location(self, physics):
    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)

    self.image = image

    location_range = np.transpose(np.where(np.all(image > 150, axis=2)))
    self.location_range = location_range
    num_loc = np.shape(location_range)[0]
    self.num_loc = num_loc
    if num_loc == 0 :
      return None
    index = np.random.randint(num_loc, size=2)
    # Doesn't constrain pick points to left or right
    location = location_range[index]
    return location.flatten() # shape 4, array of locations

  def get_reward(self,physics):
    current_mask = np.all(self.image>150,axis=2).astype(int)
    reward_mask = current_mask
    line = np.linspace(0,31,num=32)*(-0.5)
    column = np.concatenate([np.flip(line),line])
    reward =np.sum(reward_mask* np.exp(column).reshape((W,1)))/111.0
    return reward

if __name__ == '__main__':
  env = easy()
