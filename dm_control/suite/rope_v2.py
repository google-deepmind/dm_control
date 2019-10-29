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

  def __init__(self, randomize_gains, random=None, random_location=True, maxq=False):
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
    self.num_loc = 100
    print('maxq', self._maxq, 'random_location', self._random_location)
    super(Rope, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    if self._random_location:
      return specs.BoundedArray(
          shape=(2,), dtype=np.float, minimum=[-1.0] * 2, maximum=[1.0] * 2)
    else:
      return specs.BoundedArray(
          shape=(3,), dtype=np.float, minimum=[-1.0] * 3, maximum=[1.0] * 3)

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

    if self._random_location and not self._maxq:
      assert len(action) == 2
      goal_position = action
      location = self.current_loc
    else:
      assert len(action) == 3
      goal_position = action[1:]
      location = int(np.round((action[0] * 0.5 + 0.5) * 24))

    goal_position = goal_position * 0.05

    corner_action = 'B{}'.format(location)
    corner_geom = 'G{}'.format(location)

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
    if self._random_location and not self._maxq:
      location = np.random.choice(25)
      self.current_loc = location
      obs['location'] = np.tile(location, 50).reshape(-1).astype('float32') / 24
    elif self._maxq:
      obs['location'] = np.tile(-1, 50).reshape(-1).astype('float32')

    obs['position'] = physics.data.geom_xpos[1:, :].reshape(-1).astype('float32')

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

    return obs

  def get_reward(self,physics):
    current_mask = np.all(self.image>150,axis=2).astype(int)
    reward_mask = current_mask
    line = np.linspace(0,31,num=32)*(-0.5)
    column = np.concatenate([np.flip(line),line])
    reward =np.sum(reward_mask* np.exp(column).reshape((W,1)))/111.0
    return reward
