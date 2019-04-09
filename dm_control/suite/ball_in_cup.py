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

"""Ball-in-Cup Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)


SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('ball_in_cup.xml'), common.ASSETS


@SUITE.add('benchmarking', 'easy')
def catch(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Ball-in-Cup task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = BallInCup(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics with additional features for the Ball-in-Cup domain."""

  def ball_to_target(self):
    """Returns the vector from the ball to the target."""
    target = self.named.data.site_xpos['target', ['x', 'z']]
    ball = self.named.data.xpos['ball', ['x', 'z']]
    return target - ball

  def in_target(self):
    """Returns 1 if the ball is in the target, 0 otherwise."""
    ball_to_target = abs(self.ball_to_target())
    target_size = self.named.model.site_size['target', [0, 2]]
    ball_size = self.named.model.geom_size['ball', 0]
    return float(all(ball_to_target < target_size - ball_size))


class BallInCup(base.Task):
  """The Ball-in-Cup task. Put the ball in the cup."""

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Args:
      physics: An instance of `Physics`.

    """
    # Find a collision-free random initial position of the ball.
    penetrating = True
    while penetrating:
      # Assign a random ball position.
      physics.named.data.qpos['ball_x'] = self.random.uniform(-.2, .2)
      physics.named.data.qpos['ball_z'] = self.random.uniform(.2, .5)
      # Check for collisions.
      physics.after_reset()
      penetrating = physics.data.ncon > 0
    super(BallInCup, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns an observation of the state."""
    obs = collections.OrderedDict()
    obs['position'] = physics.position()
    obs['velocity'] = physics.velocity()
    return obs

  def get_reward(self, physics):
    """Returns a sparse reward."""
    return physics.in_target()
