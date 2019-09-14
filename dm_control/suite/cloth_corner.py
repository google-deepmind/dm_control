# Copyright 2017 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,/
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


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
# CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
CORNER_INDEX_POSITION=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""

  return common.read_model('cloth_corner.xml'),common.ASSETS



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

  def __init__(self, randomize_gains, random=None, random_location=True,
               pixels_only=False):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    self._random_location = random_location
    self._pixels_only = pixels_only

    self._current_loc = self._generate_loc()

    print('random_location', self._random_location, 'pixels_only', self._pixels_only)

    super(Cloth, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    # one hot corner + action
    if self._random_location:
      return specs.BoundedArray(
          shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3)
    else:
      return specs.BoundedArray(
        shape=(7,), dtype=np.float, minimum=[-1.0] * 7, maximum=[1.0] * 7
      )

  def initialize_episode(self,physics):
    if self._random_location:
        self._current_loc = self._generate_loc()

    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])
    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image = image
    self.mask = np.any(image < 100, axis=-1).astype(int)

    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
  #     # Support legacy internal code.

      physics.named.data.xfrc_applied[:,:3]=np.zeros((3,))

      if self._random_location:
        index = self._current_loc
      else:
        one_hot = action[:4]
        index = np.argmax(one_hot)
        action = action[4:]

      goal_position = action * 0.05
      corner_action = CORNER_INDEX_ACTION[index]
      corner_geom = CORNER_INDEX_POSITION[index]


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

      if self._random_location:
        self._current_loc = self._generate_loc()



  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    render_kwargs = {}
    render_kwargs['camera_id'] = 0
    render_kwargs['width'] = W
    render_kwargs['height'] = W
    image = physics.render(**render_kwargs)
    self.image=image

    if not self._pixels_only:
        obs['position'] = physics.data.geom_xpos[5:,:].reshape(-1).astype('float32')

    if self._random_location:
      one_hot = np.zeros(4).astype('float32')
      one_hot[self._current_loc] = 1
      obs['location'] = one_hot

    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    current_mask = np.any(self.image < 100, axis=-1).astype(int)
    area = np.sum(current_mask * self.mask)
    reward = area / np.sum(self.mask)

    return reward

  def _generate_loc(self):
    return np.random.choice(4)
