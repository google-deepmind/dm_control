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

"""Base class for tasks in the Control Suite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import mujoco
from dm_control.rl import control

import numpy as np


class Task(control.Task):
  """Base class for tasks in the Control Suite.

  Actions are mapped directly to the states of MuJoCo actuators: each element of
  the action array is used to set the control input for a single actuator. The
  ordering of the actuators is the same as in the corresponding MJCF XML file.

  Attributes:
    random: A `numpy.random.RandomState` instance. This should be used to
      generate all random variables associated with the task, such as random
      starting states, observation noise* etc.

  *If sensor noise is enabled in the MuJoCo model then this will be generated
  using MuJoCo's internal RNG, which has its own independent state.
  """

  def __init__(self, random=None):
    """Initializes a new continuous control task.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an integer
        seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    if not isinstance(random, np.random.RandomState):
      random = np.random.RandomState(random)
    self._random = random
    self._visualize_reward = False

  @property
  def random(self):
    """Task-specific `numpy.random.RandomState` instance."""
    return self._random

  def action_spec(self, physics):
    """Returns a `BoundedArraySpec` matching the `physics` actuators."""
    return mujoco.action_spec(physics)

  def initialize_episode(self, physics):
    """Resets geom colors to their defaults after starting a new episode.

    Subclasses of `base.Task` must delegate to this method after performing
    their own initialization.

    Args:
      physics: An instance of `mujoco.Physics`.
    """
    self.after_step(physics)

  def before_step(self, action, physics):
    """Sets the control signal for the actuators to values in `action`."""
    # Support legacy internal code.
    action = getattr(action, "continuous_actions", action)
    physics.set_control(action)

  def after_step(self, physics):
    """Modifies colors according to the reward."""
    if self._visualize_reward:
      reward = np.clip(self.get_reward(physics), 0.0, 1.0)
      _set_reward_colors(physics, reward)

  @property
  def visualize_reward(self):
    return self._visualize_reward

  @visualize_reward.setter
  def visualize_reward(self, value):
    if not isinstance(value, bool):
      raise ValueError("Expected a boolean, got {}.".format(type(value)))
    self._visualize_reward = value


_MATERIALS = ["self", "effector", "target"]
_DEFAULT = [name + "_default" for name in _MATERIALS]
_HIGHLIGHT = [name + "_highlight" for name in _MATERIALS]


def _set_reward_colors(physics, reward):
  """Sets the highlight, effector and target colors according to the reward."""
  assert 0.0 <= reward <= 1.0
  colors = physics.named.model.mat_rgba
  default = colors[_DEFAULT]
  highlight = colors[_HIGHLIGHT]
  blend_coef = reward ** 4  # Better color distinction near high rewards.
  colors[_MATERIALS] = blend_coef * highlight + (1.0 - blend_coef) * default
