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

"""Input action, chooses random joint"""


_DEFAULT_TIME_LIMIT = 20
SUITE = containers.TaggedTasks()
CORNER_INDEX_POSITION=[86,81,59,54]
CORNER_INDEX_ACTION=['B0_0','B0_8','B8_0','B8_8']
GEOM_INDEX=['G0_0','G0_8','G8_0','G8_8']

def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  # return common.read_model('cloth_v0.xml'), common.ASSETS
  return common.read_model('cloth_v4.xml'),common.ASSETS

@SUITE.add('benchmarking', 'easy')
def easy(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None, **kwargs):
  """Returns the easy cloth task."""

  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Cloth(randomize_gains=False, random=random, **kwargs)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, n_frame_skip=1, special_task=True, **environment_kwargs)

class Physics(mujoco.Physics):
  """physics for the point_mass domain."""

class Cloth(base.Task):
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
    # self.action_spec=specs.BoundedArray(
    # shape=(2,), dtype=np.float, minimum=0.0, maximum=1.0)
    super(Cloth, self).__init__(random=random)
    self._current_loc = None

  def action_spec(self, physics):
    """Returns a `BoundedArray` matching the `physics` actuators."""

    # action force(3) ~[-1,1]

    return specs.BoundedArray(
        shape=(3,), dtype=np.float, minimum=[-1.0,-1.0,-1.0] , maximum=[1.0,1.0,1.0] )

  def initialize_episode(self,physics):
    physics.named.data.xfrc_applied['B3_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied['B4_4', :3] = np.array([0,0,-2])
    physics.named.data.xfrc_applied[CORNER_INDEX_ACTION,:3]=np.random.uniform(-.5,.5,size=3)

    super(Cloth, self).initialize_episode(physics)

  def before_step(self, action, physics):
      """Sets the control signal for the actuators to values in `action`."""
      physics.named.data.xfrc_applied[1:, :3] = np.zeros((3,))
      self._current_loc = np.random.randint(81)
      physics.named.data.xfrc_applied[self._current_loc + 1, :3] = 5 * action

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()

    one_hot = np.zeros(81)
    one_hot[self._current_loc] = 1.
    obs['force_location'] = one_hot
    return obs

  def get_reward(self, physics):
    """Returns a reward to the agent."""

    pos_ll=physics.data.geom_xpos[86,:2]
    pos_lr=physics.data.geom_xpos[81,:2]
    pos_ul=physics.data.geom_xpos[59,:2]
    pos_ur=physics.data.geom_xpos[54,:2]

    diag_dist1=np.linalg.norm(pos_ll-pos_ur)
    diag_dist2=np.linalg.norm(pos_lr-pos_ul)

    reward = diag_dist1 + diag_dist2
    return reward, dict()
