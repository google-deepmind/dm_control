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
import mujoco_py


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()

CORNER_INDEX_ACTION=['B0','B8','B14','B18']
GEOM_INDEX=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('cloth_v0.xml'), common.ASSETS
  return common.read_model('rope_v1.xml'),common.ASSETS





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

  def __init__(self, randomize_gains, random=None):
    """Initialize an instance of `PointMass`.

    Args:
      randomize_gains: A `bool`, whether to randomize the actuator gains.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._randomize_gains = randomize_gains
    # self.action_spec=specs.BoundedArraySpec(
    # shape=(2,), dtype=np.float, minimum=0.0, maximum=1.0)
    super(Rope, self).__init__(random=random)

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""

    return specs.BoundedArray(
        shape=(2,), dtype=np.float, minimum=[-1.0] * 2, maximum=[1.0] * 2)

  def initialize_episode(self,physics):

    #physics.named.data.xfrc_applied['B3_4', :3] = np.array([0, 0, -2])
    # physics.named.data.xfrc_applied['B10', :3] = np.array([0, 0, -2])
    # physics.named.data.xfrc_applied['B0_8', :3] = np.array([0.2,0.2,0.1])
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION, :3] = np.random.uniform(-0.8, 0.8, size=12).reshape((4,3))
    super(Rope, self).initialize_episode(physics)

  def before_step(self, action, physics):

    physics.named.data.xfrc_applied[:,:]=np.zeros((6,))
    physics.named.data.xfrc_applied['B0',:2] = action * 2


  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.data.geom_xpos[1:, :].astype('float32').reshape(-1)
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    B0_pos=physics.named.data.geom_xpos['G0']
    B20_pos=physics.named.data.geom_xpos['G20']

    reward_dist_goal = np.linalg.norm(B0_pos - B20_pos)

    reward = reward_dist_goal

    return reward
