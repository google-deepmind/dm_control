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
from dm_control import mujoco#, viewer
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
from dm_control.utils import rewards
from imageio import imsave
from PIL import Image,ImageColor
import os
import math
import numpy as np
import random
import mujoco_py


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
# CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
CORNER_INDEX_POSITION=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_point.xml'),common.ASSETS



W=64

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit,n_frame_skip=1,special_task=True, **environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""



class Cloth(base.Task):
  """A point_mass `Task` to reach target with smooth reward."""

  def __init__(self, randomize_gains, random=None, random_location=True, pixels_only=False, maxq=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._random_location = random_location
    self._maxq = maxq

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""

    if not self._maxq and not self._random_location:
        # first 2 are pixel locations for pick point, last 3 are xyz deltas (place point computed as pick point + delta)
        return specs.BoundedArray(
            shape=(10,), dtype=np.float, minimum=[-1.0] * 10, maximum=[1.0] * 10)
    else:
        # xyz deltas (place point computed as pick point + delta)
        # first xyz is left, second xyz is right, currently no distinction of right vs left
        return specs.BoundedArray(
            shape=(6,), dtype=np.float, minimum=[-1.0] * 6, maximum=[1.0] * 6)

  def initialize_episode(self, physics):
      physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
      physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])

      render_kwargs = {}
      render_kwargs['camera_id'] = 0
      render_kwargs['width'] = W
      render_kwargs['height'] = W
      image = physics.render(**render_kwargs)
      self.image = image
      self.mask = np.any(image < 100, axis=-1).astype(int)

      # Apply random force in the beginning for random cloth init state
      physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

      super(Cloth, self).initialize_episode(physics)
      self.current_locs = self.sample_locations(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))
      action = action.reshape((2,-1))
      if self._maxq:
          locations = (action[:, :2] * 0.5 + 0.5) * 63
          locations = np.round(locations).astype('int32')
          goal_positions = action[:, 2:]
      else:
          locations = self.current_locs.reshape((2,-1))
          goal_positions = action
      goal_positions = goal_positions * 0.05

      # computing the mapping from geom_xpos to pixel location in image
      cam_fovy = physics.named.model.cam_fovy['fixed']
      f = 0.5 * W / math.tan(cam_fovy * math.pi / 360)
      cam_matrix = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
      cam_mat = physics.named.data.cam_xmat['fixed'].reshape((3, 3))
      cam_pos = physics.named.data.cam_xpos['fixed'].reshape((3, 1))
      cam = np.concatenate([cam_mat, cam_pos], axis=1)
      cam_pos_all = np.zeros((86, 3, 1))
      num_bodies = len(physics.data.geom_xpos)
      for i in range(num_bodies):
          geom_xpos_added = np.concatenate([physics.data.geom_xpos[i], np.array([1])]).reshape((4, 1))
          cam_pos_all[i] = cam_matrix.dot(cam.dot(geom_xpos_added)[:3])

      cam_pos_xy = np.rint(cam_pos_all[:, :2].reshape((86, 2)) / cam_pos_all[:, 2])
      cam_pos_xy = cam_pos_xy.astype(int)
      cam_pos_xy[:, 1] = W - cam_pos_xy[:, 1]

      epsilon = 3

      possible_index = []
      possible_z = []
      for i in range(num_bodies):
          for j in range(num_bodies):
              # flipping the x and y to make sure it corresponds to the real location
              if abs(cam_pos_xy[i][0] - locations[0][1]) < epsilon and abs(cam_pos_xy[i][1] - locations[0][0]) < epsilon and i > 4 \
              and abs(cam_pos_xy[j][0] - locations[1][1]) < epsilon and abs(cam_pos_xy[j][1] - locations[1][0]) < epsilon and j > 4 \
              and i != j:
                  possible_index.append((i,j))
                  possible_z.append((physics.data.geom_xpos[i, 2], physics.data.geom_xpos[j, 2]))

      # Move the selected joint to the correct goal position
      if possible_index != []:
          left_index, right_index = possible_index[possible_z.index(max(possible_z, key=lambda x: np.mean(x)))]

          left_action, left_geom = left_index - 4, left_index
          right_action, right_geom = right_index - 4, right_index

          # apply consecutive force to move the point to the target position
          left_position = goal_positions[0] + physics.named.data.geom_xpos[left_geom]
          left_dist = left_position - physics.named.data.geom_xpos[left_geom]

          right_position = goal_positions[1] + physics.named.data.geom_xpos[right_geom]
          right_dist = right_position - physics.named.data.geom_xpos[right_geom]

          loop = 0
          while np.linalg.norm(np.vstack((left_dist,right_dist))) > 0.025:
            loop += 1
            if loop > 100:
                # print(np.linalg.norm(left_dist), np.linalg.norm(right_dist), 'Timeout exceeded.')
                break
            physics.named.data.xfrc_applied[left_action, :3] = right_dist * 20
            physics.named.data.xfrc_applied[right_action, :3] = left_dist * 20
            physics.step()
            self.after_step(physics)
            left_dist = left_position - physics.named.data.geom_xpos[left_geom]
            right_dist = right_position - physics.named.data.geom_xpos[right_geom]
    #   else:
    #       print('No pick point geom found.')



  def get_observation(self, physics):
    """Returns an observation of the state. For this env, it is the pick point."""
    obs = collections.OrderedDict()

    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image = image

    # If pick point part of state space, sample a point randomly
    if self._maxq:
        self.current_locs = np.zeros((4,))
    else:
        self.current_locs = self.sample_locations(physics)
    obs['location'] = np.tile(self.current_locs, 50).reshape(-1).astype('float32') / 63.
    return obs

  def sample_locations(self, physics):
      image = self.image
      location_range = np.transpose(np.where(np.any(image < 100, axis=-1)))

      num_loc = np.shape(location_range)[0]
      index = np.random.randint(num_loc, size=2)
      # Doesn't constrain pick points to left or right.
      locations = location_range[index]
      return locations.flatten()

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    # Reward computed as intersection of current binary image with goal binary image
    current_mask = np.any(self.image < 100, axis=-1).astype(int)
    reward = np.sum(current_mask) / np.sum(self.mask)

    return reward


if __name__ == '__main__':
  env = easy()
