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

"""Finger Domain."""

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.suite.utils import randomizers
from dm_control.utils import containers
import numpy as np
from six.moves import range

_DEFAULT_TIME_LIMIT = 20  # (seconds)
_CONTROL_TIMESTEP = .02   # (seconds)
# For TURN tasks, the 'tip' geom needs to enter a spherical target of sizes:
_EASY_TARGET_SIZE = 0.07
_HARD_TARGET_SIZE = 0.03
# Initial spin velocity for the Stop task.
_INITIAL_SPIN_VELOCITY = 100
# Spinning slower than this value (radian/second) is considered stopped.
_STOP_VELOCITY = 1e-6
# Spinning faster than this value (radian/second) is considered spinning.
_SPIN_VELOCITY = 15.0


SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('finger.xml'), common.ASSETS


@SUITE.add('benchmarking')
def spin(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  """Returns the Spin task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Spin(random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def turn_easy(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the easy Turn task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Turn(target_radius=_EASY_TARGET_SIZE, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


@SUITE.add('benchmarking')
def turn_hard(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  """Returns the hard Turn task."""
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Turn(target_radius=_HARD_TARGET_SIZE, random=random)
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(mujoco.Physics):
  """Physics simulation with additional features for the Finger domain."""

  def touch(self):
    """Returns logarithmically scaled signals from the two touch sensors."""
    return np.log1p(self.named.data.sensordata[['touchtop', 'touchbottom']])

  def hinge_velocity(self):
    """Returns the velocity of the hinge joint."""
    return self.named.data.sensordata['hinge_velocity']

  def tip_position(self):
    """Returns the (x,z) position of the tip relative to the hinge."""
    return (self.named.data.sensordata['tip'][[0, 2]] -
            self.named.data.sensordata['spinner'][[0, 2]])

  def bounded_position(self):
    """Returns the positions, with the hinge angle replaced by tip position."""
    return np.hstack((self.named.data.sensordata[['proximal', 'distal']],
                      self.tip_position()))

  def velocity(self):
    """Returns the velocities (extracted from sensordata)."""
    return self.named.data.sensordata[['proximal_velocity',
                                       'distal_velocity',
                                       'hinge_velocity']]

  def target_position(self):
    """Returns the (x,z) position of the target relative to the hinge."""
    return (self.named.data.sensordata['target'][[0, 2]] -
            self.named.data.sensordata['spinner'][[0, 2]])

  def to_target(self):
    """Returns the vector from the tip to the target."""
    return self.target_position() - self.tip_position()

  def dist_to_target(self):
    """Returns the signed distance to the target surface, negative is inside."""
    return (np.linalg.norm(self.to_target()) -
            self.named.model.site_size['target', 0])


class Spin(base.Task):
  """A Finger `Task` to spin the stopped body."""

  def __init__(self, random=None):
    """Initializes a new `Spin` instance.

    Args:
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    super(Spin, self).__init__(random=random)

  def initialize_episode(self, physics):
    physics.named.model.site_rgba['target', 3] = 0
    physics.named.model.site_rgba['tip', 3] = 0
    physics.named.model.dof_damping['hinge'] = .03
    _set_random_joint_angles(physics, self.random)
    super(Spin, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns state and touch sensors, and target info."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()
    obs['touch'] = physics.touch()
    return obs

  def get_reward(self, physics):
    """Returns a sparse reward."""
    return float(physics.hinge_velocity() <= -_SPIN_VELOCITY)


class Turn(base.Task):
  """A Finger `Task` to turn the body to a target angle."""

  def __init__(self, target_radius, random=None):
    """Initializes a new `Turn` instance.

    Args:
      target_radius: Radius of the target site, which specifies the goal angle.
      random: Optional, either a `numpy.random.RandomState` instance, an
        integer seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
    """
    self._target_radius = target_radius
    super(Turn, self).__init__(random=random)

  def initialize_episode(self, physics):
    target_angle = self.random.uniform(-np.pi, np.pi)
    hinge_x, hinge_z = physics.named.data.xanchor['hinge', ['x', 'z']]
    radius = physics.named.model.geom_size['cap1'].sum()
    target_x = hinge_x + radius * np.sin(target_angle)
    target_z = hinge_z + radius * np.cos(target_angle)
    physics.named.model.site_pos['target', ['x', 'z']] = target_x, target_z
    physics.named.model.site_size['target', 0] = self._target_radius

    _set_random_joint_angles(physics, self.random)

    super(Turn, self).initialize_episode(physics)

  def get_observation(self, physics):
    """Returns state, touch sensors, and target info."""
    obs = collections.OrderedDict()
    obs['position'] = physics.bounded_position()
    obs['velocity'] = physics.velocity()
    obs['touch'] = physics.touch()
    obs['target_position'] = physics.target_position()
    obs['dist_to_target'] = physics.dist_to_target()
    return obs

  def get_reward(self, physics):
    return float(physics.dist_to_target() <= 0)


def _set_random_joint_angles(physics, random, max_attempts=1000):
  """Sets the joints to a random collision-free state."""

  for _ in range(max_attempts):
    randomizers.randomize_limited_and_rotational_joints(physics, random)
    # Check for collisions.
    physics.after_reset()
    if physics.data.ncon == 0:
      break
  else:
    raise RuntimeError('Could not find a collision-free state '
                       'after {} attempts'.format(max_attempts))
