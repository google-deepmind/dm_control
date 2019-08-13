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

"""A dm_env.Environment subclass for control-specific environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import contextlib

import dm_env
from dm_env import specs
import numpy as np
import six
from six.moves import range

FLAT_OBSERVATION_KEY = 'observations'


class Environment(dm_env.Environment):
  """Class for physics-based reinforcement learning environments."""

  def __init__(self,
               physics,
               task,
               time_limit=float('inf'),
               control_timestep=None,
               n_sub_steps=None,
               flat_observation=False,
               n_frame_skip=1,
               special_task=True):
    """Initializes a new `Environment`.

    Args:
      physics: Instance of `Physics`.
      task: Instance of `Task`.
      time_limit: Optional `int`, maximum time for each episode in seconds. By
        default this is set to infinite.
      control_timestep: Optional control time-step, in seconds.
      n_sub_steps: Optional number of physical time-steps in one control
        time-step, aka "action repeats". Can only be supplied if
        `control_timestep` is not specified.
      flat_observation: If True, observations will be flattened and concatenated
        into a single numpy array.

    Raises:
      ValueError: If both `n_sub_steps` and `control_timestep` are supplied.
    """
    self._task = task
    self._physics = physics
    self._flat_observation = flat_observation
    self._n_frame_skip = n_frame_skip
    self._special_task = special_task

    if n_sub_steps is not None and control_timestep is not None:
      raise ValueError('Both n_sub_steps and control_timestep were supplied.')
    elif n_sub_steps is not None:
      self._n_sub_steps = n_sub_steps
    elif control_timestep is not None:
      self._n_sub_steps = compute_n_steps(control_timestep,
                                          self._physics.timestep())
    else:
      self._n_sub_steps = 1

    if time_limit == float('inf'):
      self._step_limit = float('inf')
    else:
      self._step_limit = time_limit / (
          self._physics.timestep() * self._n_sub_steps)
    self._step_count = 0
    self._reset_next_step = True

  def reset(self):
    """Starts a new episode and returns the first `TimeStep`."""
    self._reset_next_step = False
    self._step_count = 0
    with self._physics.reset_context():
      self._task.initialize_episode(self._physics)

    observation = self._task.get_observation(self._physics)
    if self._flat_observation:
      observation = flatten_observation(observation)

    return dm_env.TimeStep(
        step_type=dm_env.StepType.FIRST,
        reward=None,
        discount=None,
        observation=observation)

  def step(self, action):
    """Updates the environment using the action and returns a `TimeStep`."""

    if self._reset_next_step:
      return self.reset()

    if self._special_task:
        if self._step_count > 130:
            self._task.before_step(action, self._physics)
    for _ in range(self._n_sub_steps * self._n_frame_skip):
      self._physics.step()
    self._task.after_step(self._physics)

    reward = self._task.get_reward(self._physics)
    observation = self._task.get_observation(self._physics)
    if self._flat_observation:
      observation = flatten_observation(observation)

    self._step_count += 1
    if self._step_count >= self._step_limit:
      discount = 1.0
    else:
      discount = self._task.get_termination(self._physics)

    episode_over = discount is not None

    if episode_over:
      self._reset_next_step = True
      return dm_env.TimeStep(
          dm_env.StepType.LAST, reward, discount, observation)
    else:
      return dm_env.TimeStep(dm_env.StepType.MID, reward, 1.0, observation)

  def action_spec(self):
    """Returns the action specification for this environment."""
    return self._task.action_spec(self._physics)

  def step_spec(self):
    """May return a specification for the values returned by `step`."""
    return self._task.step_spec(self._physics)

  def observation_spec(self):
    """Returns the observation specification for this environment.

    Infers the spec from the observation, unless the Task implements the
    `observation_spec` method.

    Returns:
      An dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    try:
      return self._task.observation_spec(self._physics)
    except NotImplementedError:
      observation = self._task.get_observation(self._physics)
      if self._flat_observation:
        observation = flatten_observation(observation)
      return _spec_from_observation(observation)

  @property
  def physics(self):
    return self._physics

  @property
  def task(self):
    return self._task

  def control_timestep(self):
    """Returns the interval between agent actions in seconds."""
    return self.physics.timestep() * self._n_sub_steps


def compute_n_steps(control_timestep, physics_timestep, tolerance=1e-8):
  """Returns the number of physics timesteps in a single control timestep.

  Args:
    control_timestep: Control time-step, should be an integer multiple of the
      physics timestep.
    physics_timestep: The time-step of the physics simulation.
    tolerance: Optional tolerance value for checking if `physics_timestep`
      divides `control_timestep`.

  Returns:
    The number of physics timesteps in a single control timestep.

  Raises:
    ValueError: If `control_timestep` is smaller than `physics_timestep` or if
      `control_timestep` is not an integer multiple of `physics_timestep`.
  """
  if control_timestep < physics_timestep:
    raise ValueError(
        'Control timestep ({}) cannot be smaller than physics timestep ({}).'.
        format(control_timestep, physics_timestep))
  if abs((control_timestep / physics_timestep - round(
      control_timestep / physics_timestep))) > tolerance:
    raise ValueError(
        'Control timestep ({}) must be an integer multiple of physics timestep '
        '({})'.format(control_timestep, physics_timestep))
  return int(round(control_timestep / physics_timestep))


def _spec_from_observation(observation):
  result = collections.OrderedDict()
  for key, value in six.iteritems(observation):
    result[key] = specs.Array(value.shape, value.dtype, name=key)
  return result

# Base class definitions for objects supplied to Environment.


@six.add_metaclass(abc.ABCMeta)
class Physics(object):
  """Simulates a physical environment."""

  @abc.abstractmethod
  def step(self, n_sub_steps=1):
    """Updates the simulation state.

    Args:
      n_sub_steps: Optional number of times to repeatedly update the simulation
        state. Defaults to 1.
    """

  @abc.abstractmethod
  def time(self):
    """Returns the elapsed simulation time in seconds."""

  @abc.abstractmethod
  def timestep(self):
    """Returns the simulation timestep."""

  def set_control(self, control):
    """Sets the control signal for the actuators."""
    raise NotImplementedError('set_control is not supported.')

  @contextlib.contextmanager
  def reset_context(self):
    """Context manager for resetting the simulation state.

    Sets the internal simulation to a default state when entering the block.

    ```python
    with physics.reset_context():
      # Set joint and object positions.

    physics.step()
    ```

    Yields:
      The `Physics` instance.
    """
    try:
      self.reset()
    except PhysicsError:
      pass
    yield self
    self.after_reset()

  @abc.abstractmethod
  def reset(self):
    """Resets internal variables of the physics simulation."""

  @abc.abstractmethod
  def after_reset(self):
    """Runs after resetting internal variables of the physics simulation."""

  def check_divergence(self):
    """Raises a `PhysicsError` if the simulation state is divergent.

    The default implementation is a no-op.
    """


class PhysicsError(RuntimeError):
  """Raised if the state of the physics simulation becomes divergent."""


@six.add_metaclass(abc.ABCMeta)
class Task(object):
  """Defines a task in a `control.Environment`."""

  @abc.abstractmethod
  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode.

    Called by `control.Environment` at the start of each episode *within*
    `physics.reset_context()` (see the documentation for `base.Physics`).

    Args:
      physics: Instance of `Physics`.
    """

  @abc.abstractmethod
  def before_step(self, action, physics):
    """Updates the task from the provided action.

    Called by `control.Environment` before stepping the physics engine.

    Args:
      action: numpy array or array-like action values, or a nested structure of
        such arrays. Should conform to the specification returned by
        `self.action_spec(physics)`.
      physics: Instance of `Physics`.
    """

  def after_step(self, physics):
    """Optional method to update the task after the physics engine has stepped.

    Called by `control.Environment` after stepping the physics engine and before
    `control.Environment` calls `get_observation, `get_reward` and
    `get_termination`.

    The default implementation is a no-op.

    Args:
      physics: Instance of `Physics`.
    """

  @abc.abstractmethod
  def action_spec(self, physics):
    """Returns a specification describing the valid actions for this task.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the action array(s) passed to `self.step`.
    """

  def step_spec(self, physics):
    """Returns a specification describing the time_step for this task.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A `BoundedArraySpec`, or a nested structure containing `BoundedArraySpec`s
      that describe the shapes, dtypes and elementwise lower and upper bounds
      for the array(s) returned by `self.step`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_observation(self, physics):
    """Returns an observation from the environment.

    Args:
      physics: Instance of `Physics`.
    """

  @abc.abstractmethod
  def get_reward(self, physics):
    """Returns a reward from the environment.

    Args:
      physics: Instance of `Physics`.
    """

  def get_termination(self, physics):
    """If the episode should end, returns a final discount, otherwise None."""

  def observation_spec(self, physics):
    """Optional method that returns the observation spec.

    If not implemented, the Environment infers the spec from the observation.

    Args:
      physics: Instance of `Physics`.

    Returns:
      A dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    raise NotImplementedError()


def flatten_observation(observation, output_key=FLAT_OBSERVATION_KEY):
  """Flattens multiple observation arrays into a single numpy array.

  Args:
    observation: A mutable mapping from observation names to numpy arrays.
    output_key: The key for the flattened observation array in the output.

  Returns:
    A mutable mapping of the same type as `observation`. This will contain a
    single key-value pair consisting of `output_key` and the flattened
    and concatenated observation array.

  Raises:
    ValueError: If `observation` is not a `collections.MutableMapping`.
  """
  if not isinstance(observation, collections.MutableMapping):
    raise ValueError('Can only flatten dict-like observations.')

  if isinstance(observation, collections.OrderedDict):
    keys = six.iterkeys(observation)
  else:
    # Keep a consistent ordering for other mappings.
    keys = sorted(six.iterkeys(observation))

  observation_arrays = [observation[key].ravel() for key in keys]
  return type(observation)([(output_key, np.concatenate(observation_arrays))])
