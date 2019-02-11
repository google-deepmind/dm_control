# Copyright 2018 The dm_control Authors.
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

"""Abstract base class for a Composer task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import sys

from dm_control import mujoco
import six
from six.moves import range

from dm_control.rl import specs


def _check_timesteps_divisible(control_timestep, physics_timestep):
  num_steps = control_timestep / physics_timestep
  rounded_num_steps = int(round(num_steps))
  if abs(num_steps - rounded_num_steps) > 1e-6:
    raise ValueError(
        'Control timestep should be an integer multiple of physics timestep'
        ': got {!r} and {!r}'.format(control_timestep, physics_timestep))
  return rounded_num_steps


@six.add_metaclass(abc.ABCMeta)
class Task(object):
  """Abstract base class for a Composer task."""

  @abc.abstractproperty
  def root_entity(self):
    """A `base.Entity` instance for this task."""
    raise NotImplementedError

  def iter_entities(self):
    return self.root_entity.iter_entities()

  @property
  def observables(self):
    """An OrderedDict of `control.Observable` instances for this task.

    Task subclasses should generally NOT override this property.

    This property is automatically computed by combining the observables dict
    provided by each `Entity` present in this task, and any additional
    observables returned via the `task_observables` property.

    To provide an observable to an agent, the task code should either set
    `enabled` property of an `Entity`-bound observable to `True`, or override
    the `task_observables` property to provide additional observables not bound
    to an `Entity`.

    Returns:
      An `collections.OrderedDict` mapping strings to instances of
      `control.Observable`.
    """
    # Make a shallow copy of the OrderedDict, not the Observables themselves.
    observables = copy.copy(self.task_observables)
    for entity in self.root_entity.iter_entities():
      observables.update(entity.observables.as_dict())
    return observables

  @property
  def task_observables(self):
    """An OrderedDict of task-specific `control.Observable` instances.

    A task should override this property if it wants to provide additional
    observables to the agent that are not already provided by any `Entity` that
    forms part of the task's model. For example, this may be used to provide
    observations that is derived from relative poses between two entities.

    Returns:
      An `collections.OrderedDict` mapping strings to instances of
      `control.Observable`.
    """
    return collections.OrderedDict()

  def after_compile(self, physics, random_state):
    """A callback which is executed after the Mujoco Physics is recompiled.

    Args:
      physics: An instance of `control.Physics`.
      random_state: An instance of `np.random.RandomState`.
    """
    pass

  def _check_root_entity(self, callee_name):
    try:
      _ = self.root_entity
    except:  # pylint: disable=bare-except
      err_type, err, tb = sys.exc_info()
      message = (
          'call to `{}` made before `root_entity` is available;\n'
          'original error message: {}'.format(callee_name, str(err)))
      six.reraise(err_type, err_type(message), tb)

  @property
  def control_timestep(self):
    """Returns the agent's control timestep for this task (in seconds)."""
    self._check_root_entity('control_timestep')
    if hasattr(self, '_control_timestep'):
      return self._control_timestep
    else:
      return self.physics_timestep

  @control_timestep.setter
  def control_timestep(self, new_value):
    """Changes the agent's control timestep for this task.

    Args:
      new_value: the new control timestep (in seconds).

    Raises:
      ValueError: if `new_value` is set and is not divisible by
        `physics_timestep`.
    """
    self._check_root_entity('control_timestep')
    _check_timesteps_divisible(new_value, self.physics_timestep)
    self._control_timestep = new_value

  @property
  def physics_timestep(self):
    """Returns the physics timestep for this task (in seconds)."""
    self._check_root_entity('physics_timestep')
    if self.root_entity.mjcf_model.option.timestep is None:
      return 0.002  # MuJoCo's default.
    else:
      return self.root_entity.mjcf_model.option.timestep

  @physics_timestep.setter
  def physics_timestep(self, new_value):
    """Changes the physics simulation timestep for this task.

    Args:
      new_value: the new simulation timestep (in seconds).

    Raises:
      ValueError: if `control_timestep` is set and is not divisible by
        `new_value`.
    """
    self._check_root_entity('physics_timestep')
    if hasattr(self, '_control_timestep'):
      _check_timesteps_divisible(self._control_timestep, new_value)
    self.root_entity.mjcf_model.option.timestep = new_value

  def set_timesteps(self, control_timestep, physics_timestep):
    """Changes the agent's control timestep and physics simulation timestep.

    This is equivalent to modifying `control_timestep` and `physics_timestep`
    simultaneously. The divisibility check is performed between the two
    new values.

    Args:
      control_timestep: the new agent's control timestep (in seconds).
      physics_timestep: the new physics simulation timestep (in seconds).

    Raises:
      ValueError: if `control_timestep` is not divisible by `physics_timestep`.
    """
    self._check_root_entity('set_timesteps')
    _check_timesteps_divisible(control_timestep, physics_timestep)
    self.root_entity.mjcf_model.option.timestep = physics_timestep
    self._control_timestep = control_timestep

  @property
  def physics_steps_per_control_step(self):
    """Returns number of physics steps per agent's control step."""
    return _check_timesteps_divisible(
        self.control_timestep, self.physics_timestep)

  def action_spec(self, physics):
    """Returns an `BoundedArraySpec` matching the `Physics` actuators.

    BoundedArraySpec.name should contain a tab-separated list of actuator names.
    When overloading this method, non-MuJoCo actuators should be added to the
    top of the list when possible, as a matter of convention.

    Args:
      physics: used to query actuator names in the model.
    """
    names = [physics.model.id2name(i, 'actuator') or str(i)
             for i in range(physics.model.nu)]
    action_spec = mujoco.action_spec(physics)
    return specs.BoundedArraySpec(shape=action_spec.shape,
                                  dtype=action_spec.dtype,
                                  minimum=action_spec.minimum,
                                  maximum=action_spec.maximum,
                                  name='\t'.join(names))

  def get_reward_spec(self):
    """Optional method to define non-scalar rewards for a `Task`."""
    return None

  def get_discount_spec(self):
    """Optional method to define non-scalar discounts for a `Task`."""
    return None

  def initialize_episode_mjcf(self, random_state):
    """Modifies the MJCF model of this task before the next episode begins.

    The Environment calls this method and recompiles the physics
    if necessary before calling `initialize_episode`.

    Args:
      random_state: An instance of `np.random.RandomState`.
    """
    pass

  def initialize_episode(self, physics, random_state):
    """Modifies the physics state before the next episode begins.

    The Environment calls this method after `initialize_episode_mjcf`, and also
    after the physics has been recompiled if necessary.

    Args:
      physics: An instance of `control.Physics`.
      random_state: An instance of `np.random.RandomState`.
    """
    pass

  def before_step(self, physics, action, random_state):
    """A callback which is executed before an agent control step.

    The default implementation sets the control signal for the actuators in
    `physics` to be equal to `action`. Subclasses that override this method
    should ensure that the overriding method also sets the control signal before
    returning, either by calling `super(..., self).before_step`, or by setting
    the control signal explicitly (e.g. in order to create a non-trivial mapping
    between `action` and the control signal).

    Args:
      physics: An instance of `control.Physics`.
      action: A NumPy array corresponding to agent actions.
      random_state: An instance of `np.random.RandomState` (unused).
    """
    del random_state  # Unused.
    physics.set_control(action)

  def before_substep(self, physics, action, random_state):
    """A callback which is executed before a simulation step.

    Actuation can be set, or overridden, in this callback.

    Args:
      physics: An instance of `control.Physics`.
      action: A NumPy array corresponding to agent actions.
      random_state: An instance of `np.random.RandomState`.
    """
    pass

  def after_substep(self, physics, random_state):
    """A callback which is executed after a simulation step.

    Args:
      physics: An instance of `control.Physics`.
      random_state: An instance of `np.random.RandomState`.
    """
    pass

  def after_step(self, physics, random_state):
    """A callback which is executed after an agent control step.

    Args:
      physics: An instance of `control.Physics`.
      random_state: An instance of `np.random.RandomState`.
    """
    pass

  @abc.abstractmethod
  def get_reward(self, physics):
    """Calculates the reward signal given the physics state.

    Args:
      physics: A Physics object.

    Returns:
      A float
    """
    raise NotImplementedError

  def should_terminate_episode(self, physics):  # pylint: disable=unused-argument
    """Determines whether the episode should terminate given the physics state.

    Args:
      physics: A Physics object

    Returns:
      A boolean
    """
    return False

  def get_discount(self, physics):  # pylint: disable=unused-argument
    """Calculates the reward discount factor given the physics state.

    Args:
      physics: A Physics object

    Returns:
      A float
    """
    return 1.0


class NullTask(Task):
  """A class that wraps a single `Entity` into a `Task` with no reward."""

  def __init__(self, root_entity):
    self._root_entity = root_entity

  @property
  def root_entity(self):
    return self._root_entity

  def get_reward(self, physics):
    return 0.0
