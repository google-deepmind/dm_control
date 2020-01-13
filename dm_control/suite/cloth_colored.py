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
from dm_control import mujoco #, viewer
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

  return common.read_model('cloth_colored.xml'),common.ASSETS



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

  def __init__(self, randomize_gains, random=None, random_pick=True):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._random_pick = random_pick

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""

    if not self._random_pick:
        # first 2 are pixel locations for pick point, last 3 are xyz deltas (place point computed as pick point + delta)
        return specs.BoundedArray(
            shape=(5,), dtype=np.float, minimum=[-1.0] * 5, maximum=[1.0] * 5)
    else:
        # xyz deltas (place point computed as pick point + delta)
        return specs.BoundedArray(
            shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3)

  def initialize_episode(self, physics):
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])

    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image = image
    self.mask = self.segment_image(image).astype(int)

    # Apply random force in the beginning for random cloth init state
    #physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))

      if not self._random_pick:
          location = (action[:2] * 0.5 + 0.5) * 63
          location = np.round(location).astype('int32')
          goal_position = action[2:]
      else:
          location = self.current_loc
          goal_position = action
      goal_position = goal_position * 0.05

      # computing the mapping from geom_xpos to pixel location in image
      cam_fovy = physics.named.model.cam_fovy['fixed']
      f = 0.5 * W / math.tan(cam_fovy * math.pi / 360)
      cam_matrix = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1]])
      cam_mat = physics.named.data.cam_xmat['fixed'].reshape((3, 3))
      cam_pos = physics.named.data.cam_xpos['fixed'].reshape((3, 1))
      cam = np.concatenate([cam_mat, cam_pos], axis=1)
      cam_pos_all = np.zeros((86, 3, 1))
      # num_bodies = len(physics.data.geom_xpos)
      for i in range(86):
          geom_xpos_added = np.concatenate([physics.data.geom_xpos[i], np.array([1])]).reshape((4, 1))
          cam_pos_all[i] = cam_matrix.dot(cam.dot(geom_xpos_added)[:3])

      cam_pos_xy = np.rint(cam_pos_all[:, :2].reshape((86, 2)) / cam_pos_all[:, 2])
      cam_pos_xy = cam_pos_xy.astype(int)
      cam_pos_xy[:, 1] = W - cam_pos_xy[:, 1]

      epsilon = 3
      possible_index = []
      possible_z = []
      for i in range(86):
          # flipping the x and y to make sure it corresponds to the real location
          if abs(cam_pos_xy[i][0] - location[1]) < epsilon and abs(
                  cam_pos_xy[i][1] - location[0]) < epsilon and i > 4:
              possible_index.append(i)
              possible_z.append(physics.data.geom_xpos[i, 2])


      # Move the selected joint to the correct goal position

      if possible_index != []:
          index = possible_index[possible_z.index(max(possible_z))]

          corner_action = index - 4
          corner_geom = index


          # apply consecutive force to move the point to the target position
          position = goal_position + physics.named.data.geom_xpos[corner_geom]
          dist = position - physics.named.data.geom_xpos[corner_geom]

          loop = 0
          while np.linalg.norm(dist) > 0.025:
            loop += 1
            if loop > 40:
              break
            physics.named.data.xfrc_applied[corner_action, :3] = dist * 20
            physics.step()
            self.after_step(physics)
            dist = position - physics.named.data.geom_xpos[corner_geom]


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()

    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image = image

    # If pick point part of state space, sample a point randomly
    if self._random_pick:
        self.current_loc = self.sample_location(physics)
        obs['location'] = np.tile(self.current_loc, 50).reshape(-1).astype('float32') / 63.
    return obs

  def sample_location(self, physics):
      image = self.image
      location_range = np.transpose(np.where(self.segment_image(image)))

      num_loc = np.shape(location_range)[0]
      index = np.random.randint(num_loc)
      location = location_range[index]

      return location

  def segment_image(self, image):
      return np.any(self.image < 100, axis=-1)

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    # Reward computed as intersection of current binary image with goal binary image
    current_mask = self.segment_image(self.image).astype(int)
    reward = np.sum(current_mask) / np.sum(self.mask)

    return reward


if __name__ == '__main__':
  env = easy()
