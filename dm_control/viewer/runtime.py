# Copyright 2018-2019 The dm_control Authors.
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
"""Environment's execution runtime."""

import collections
import copy
import enum

from dm_control.mujoco.wrapper import mjbindings
from dm_control.viewer import util
import numpy as np

mjlib = mjbindings.mjlib


# Pause interval between simulation steps.
_SIMULATION_STEP_INTERVAL = 0.001

# The longest allowed simulation time step, in seconds.
_DEFAULT_MAX_SIM_STEP = 1./5.


def _get_default_action(action_spec):
  """Generates an action to apply to the environment if there is no agent.

  * For action dimensions that are closed intervals this will be the midpoint.
  * For left-open or right-open intervals this will be the maximum or the
    minimum respectively.
  * For unbounded intervals this will be zero.

  Args:
    action_spec: An instance of `BoundedArraySpec` or a list or tuple
      containing these.

  Returns:
    A numpy array of actions if `action_spec` is a single `BoundedArraySpec`, or
    a tuple of such arrays if `action_spec` is a list or tuple.
  """
  if isinstance(action_spec, (list, tuple)):
    return tuple(_get_default_action(spec) for spec in action_spec)
  elif isinstance(action_spec, collections.MutableMapping):
    # Clones the Mapping, preserving type and key order.
    result = copy.copy(action_spec)

    for key, value in action_spec.items():
      result[key] = _get_default_action(value)

    return result

  minimum = np.broadcast_to(action_spec.minimum, action_spec.shape)
  maximum = np.broadcast_to(action_spec.maximum, action_spec.shape)
  left_bounded = np.isfinite(minimum)
  right_bounded = np.isfinite(maximum)
  action = np.select(
      condlist=[left_bounded & right_bounded, left_bounded, right_bounded],
      choicelist=[0.5 * (minimum + maximum), minimum, maximum],
      default=0.)
  action = action.astype(action_spec.dtype, copy=False)
  action.flags.writeable = False
  return action


class State(enum.Enum):
  """State of the Runtime class."""
  START = 0
  RUNNING = 1
  STOP = 2
  STOPPED = 3
  RESTARTING = 4


class Runtime(object):
  """Base Runtime class.

  Attributes:
    simulation_time_budget: Float value, how much time can be spent on physics
      simulation every frame, in seconds.
    on_episode_begin: An observable subject, an instance of util.QuietSet.
      It contains argumentless callables, invoked, when a new episode begins.
    on_error: An observable subject, an instance of util.QuietSet. It contains
      single argument callables, invoked, when the environment or the agent
      throw an error.
    on_physics_changed: An observable subject, an instance of util.QuietSet.
      During episode restarts, the underlying physics instance may change. If
      you are interested in learning about those changes, attach a listener
      using the += operator. The listener should be a callable with no required
      arguments.
  """

  def __init__(self, environment, policy=None):
    """Instance initializer.

    Args:
      environment: An instance of dm_control.rl.control.Environment.
      policy: Either a callable that accepts a `TimeStep` and returns a numpy
        array of actions conforming to `environment.action_spec()`, or None, in
        which case a default action will be generated for each environment step.
    """
    self.on_error = util.QuietSet()
    self.on_episode_begin = util.QuietSet()
    self.simulation_time_budget = _DEFAULT_MAX_SIM_STEP

    self._state = State.START
    self._simulation_timer = util.Timer()
    self._tracked_simulation_time = 0.0
    self._error_logger = util.ErrorLogger(self.on_error)

    self._env = environment
    self._policy = policy
    self._default_action = _get_default_action(environment.action_spec())
    self._time_step = None
    self._last_action = None
    self.on_physics_changed = util.QuietSet()

  def tick(self, time_elapsed, paused):
    """Advances the simulation by one frame.

    Args:
      time_elapsed: Time elapsed since the last time this method was called.
      paused: A boolean flag telling if the  simulation is paused.
    Returns:
      A boolean flag to determine if the episode has finished.
    """
    with self._simulation_timer.measure_time():
      if self._state == State.RESTARTING:
        self._state = State.START
      if self._state == State.START:
        if self._start():
          self._broadcast_episode_start()
          self._tracked_simulation_time = self.get_time()
          self._state = State.RUNNING
        else:
          self._state = State.STOPPED
      if self._state == State.RUNNING:
        finished = self._step_simulation(time_elapsed, paused)
        if finished:
          self._state = State.STOP
      if self._state == State.STOP:
        self._state = State.STOPPED

  def _step_simulation(self, time_elapsed, paused):
    """Simulate a simulation step."""
    finished = False
    if paused:
      self._step_paused()
    else:
      step_duration = min(time_elapsed, self.simulation_time_budget)
      actual_simulation_time = self.get_time()
      if self._tracked_simulation_time >= actual_simulation_time:
        end_time = actual_simulation_time + step_duration
        while not finished and self.get_time() < end_time:
          finished = self._step()
      self._tracked_simulation_time += step_duration
    return finished

  def single_step(self):
    """Performs a single step of simulation."""
    if self._state == State.RUNNING:
      finished = self._step()
      self._state = State.STOP if finished else State.RUNNING

  def stop(self):
    """Stops the runtime."""
    self._state = State.STOPPED

  def restart(self):
    """Restarts the episode, resetting environment, model, and data."""
    if self._state != State.STOPPED:
      self._state = State.RESTARTING
    else:
      self._state = State.START

  def get_time(self):
    """Elapsed simulation time."""
    return self._env.physics.data.time

  @property
  def state(self):
    """Returns the current state of the state machine.

    Returned states are values of runtime.State enum.
    """
    return self._state

  @property
  def simulation_time(self):
    """Returns the amount of time spent running the simulation."""
    return self._simulation_timer.measured_time

  @property
  def last_action(self):
    """Action passed to the environment on the last step."""
    return self._last_action

  def _broadcast_episode_start(self):
    for listener in self.on_episode_begin:
      listener()

  def _start(self):
    """Starts a new simulation episode.

    Starting a new episode may be associated with changing the physics instance.
    The method tracks that and notifies observers through 'on_physics_changed'
    subject.

    Returns:
      True if the operation was successful, False otherwise.
    """
    # NB: we check the identity of the data pointer rather than the physics
    # instance itself, since this allows us to detect when the physics has been
    # "reloaded" using one of the `reload_from_*` methods.
    old_data_ptr = self._env.physics.data.ptr

    with self._error_logger:
      self._time_step = self._env.reset()

    if self._env.physics.data.ptr is not old_data_ptr:
      for listener in self.on_physics_changed:
        listener()
    return not self._error_logger.errors_found

  def _step_paused(self):
    mjlib.mj_forward(self._env.physics.model.ptr, self._env.physics.data.ptr)

  def _step(self):
    """Generates an action and applies it to the environment.

    If a `policy` was provided, this will be invoked to generate an action to
    feed to the environment, otherwise a default action will be generated.

    Returns:
      A boolean value, True if the environment signaled the episode end, False
      if the episode is still running.
    """
    finished = True
    with self._error_logger:
      if self._policy:
        action = self._policy(self._time_step)
      else:
        action = self._default_action
      self._time_step = self._env.step(action)
      self._last_action = action
      finished = self._time_step.last()
    return finished or self._error_logger.errors_found
