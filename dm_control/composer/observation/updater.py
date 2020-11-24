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

"""An object that creates and updates buffers for enabled observables."""

import collections
import functools
from absl import logging

from dm_control.composer import variation
from dm_control.composer.observation import obs_buffer
from dm_env import specs
import numpy as np
import six
from six.moves import range

DEFAULT_BUFFER_SIZE = 1
DEFAULT_UPDATE_INTERVAL = 1
DEFAULT_DELAY = 0


class _EnabledObservable(object):
  """Encapsulates an enabled observable, its buffer, and its update schedule."""

  __slots__ = ('observable', 'observation_callable',
               'update_interval', 'delay', 'buffer_size',
               'buffer', 'update_schedule')

  def __init__(self, observable, physics, random_state,
               strip_singleton_buffer_dim):
    self.observable = observable
    self.observation_callable = (
        observable.observation_callable(physics, random_state))

    self._bind_attribute_from_observable('update_interval',
                                         DEFAULT_UPDATE_INTERVAL,
                                         random_state)
    self._bind_attribute_from_observable('delay',
                                         DEFAULT_DELAY,
                                         random_state)
    self._bind_attribute_from_observable('buffer_size',
                                         DEFAULT_BUFFER_SIZE,
                                         random_state)

    obs_spec = self.observable.array_spec
    if obs_spec is None:
      # We take an observation to determine the shape and dtype of the array.
      # This occurs outside of an episode and doesn't affect environment
      # behavior. At this point the physics state is not guaranteed to be valid,
      # so we might get a `PhysicsError` if the observation callable calls
      # `physics.forward`. We suppress such errors since they do not matter as
      # far as the shape and dtype of the observation are concerned.
      with physics.suppress_physics_errors():
        obs_array = self.observation_callable()
      obs_array = np.asarray(obs_array)
      obs_spec = specs.Array(shape=obs_array.shape, dtype=obs_array.dtype)
    self.buffer = obs_buffer.Buffer(
        buffer_size=self.buffer_size,
        shape=obs_spec.shape, dtype=obs_spec.dtype,
        strip_singleton_buffer_dim=strip_singleton_buffer_dim)
    self.update_schedule = collections.deque()

  def _bind_attribute_from_observable(self, attr, default_value, random_state):
    obs_attr = getattr(self.observable, attr)
    if obs_attr:
      if isinstance(obs_attr, variation.Variation):
        setattr(self, attr,
                functools.partial(obs_attr, random_state=random_state))
      else:
        setattr(self, attr, obs_attr)
    else:
      setattr(self, attr, default_value)


def _call_if_callable(arg):
  if callable(arg):
    return arg()
  else:
    return arg


def _validate_structure(structure):
  """Validates the structure of the given observables collection.

  The collection must either be a dict, or a (list or tuple) of dicts.

  Args:
    structure: A candidate collection of observables.

  Returns:
    A boolean that is `True` if `structure` is either a list or a tuple, or
    `False` otherwise.

  Raises:
    ValueError: If `structure` is neither a dict nor a (list or tuple) of dicts.
  """
  is_nested = isinstance(structure, (list, tuple))
  if is_nested:
    is_valid = all(isinstance(obj, dict) for obj in structure)
  else:
    is_valid = isinstance(structure, dict)
  if not is_valid:
    raise ValueError(
        '`observables` should be a dict, or a (list or tuple) of dicts'
        ': got {}'.format(structure))
  return is_nested


class Updater(object):
  """Creates and updates buffers for enabled observables."""

  def __init__(self, observables, physics_steps_per_control_step=1,
               strip_singleton_buffer_dim=False):
    self._physics_steps_per_control_step = physics_steps_per_control_step
    self._strip_singleton_buffer_dim = strip_singleton_buffer_dim
    self._step_counter = 0
    self._observables = observables
    self._is_nested = _validate_structure(observables)
    self._enabled_structure = None
    self._enabled_list = None

  def reset(self, physics, random_state):
    """Resets this updater's state."""

    def make_buffers_dict(observables):
      """Makes observable states in a dict."""
      # Use `type(observables)` so that our output structure respects the
      # original dict subclass (e.g. OrderedDict).
      out_dict = type(observables)()
      for key, value in six.iteritems(observables):
        if value.enabled:
          out_dict[key] = _EnabledObservable(value, physics, random_state,
                                             self._strip_singleton_buffer_dim)
      return out_dict

    if self._is_nested:
      self._enabled_structure = type(self._observables)(
          make_buffers_dict(obs_dict) for obs_dict in self._observables)
      self._enabled_list = []
      for enabled_dict in self._enabled_structure:
        self._enabled_list.extend(enabled_dict.values())
    else:
      self._enabled_structure = make_buffers_dict(self._observables)
      self._enabled_list = self._enabled_structure.values()

    self._step_counter = 0
    for enabled in self._enabled_list:
      first_delay = _call_if_callable(enabled.delay)
      enabled.buffer.insert(
          0, first_delay,
          enabled.observation_callable())

  def observation_spec(self):
    """The observation specification for this environment.

    Returns a dict mapping the names of enabled observations to their
    corresponding `Array` or `BoundedArray` specs.

    If an obs has a BoundedArray spec, but uses an aggregator that
    does not preserve those bounds (such as `sum`), it will be mapped to an
    (unbounded) `Array` spec. If using a bounds-preserving custom aggregator
    `my_agg`, give it an attribute `my_agg.preserves_bounds = True` to indicate
    to this method that it is bounds-preserving.

    The returned specification is only valid as of the previous call
    to `reset`. In particular, it is an error to call this function before
    the first call to `reset`.

    Returns:
      A dict mapping observation name to `Array` or `BoundedArray` spec
      containing the observation shape and dtype, and possibly bounds.

    Raises:
      RuntimeError: If this method is called before `reset` has been called.
    """
    if self._enabled_structure is None:
      raise RuntimeError('`reset` must be called before `observation_spec`.')

    def make_observation_spec_dict(enabled_dict):
      """Makes a dict of enabled observation specs from of observables."""
      out_dict = type(enabled_dict)()
      for name, enabled in six.iteritems(enabled_dict):

        if isinstance(enabled.observable.array_spec, specs.BoundedArray):
          bounds = (enabled.observable.array_spec.minimum,
                    enabled.observable.array_spec.maximum)
        else:
          bounds = None

        if enabled.observable.aggregator:
          aggregator = enabled.observable.aggregator
          aggregated = aggregator(np.zeros(enabled.buffer.shape,
                                           dtype=enabled.buffer.dtype))
          shape = aggregated.shape
          dtype = aggregated.dtype

          # Ditch bounds if the aggregator isn't known to be bounds-preserving.
          if bounds:
            if not hasattr(aggregator, 'preserves_bounds'):
              logging.warning('Ignoring the bounds of this observable\'s spec, '
                              'as its aggregator method has no boolean '
                              '`preserves_bounds` attrubute.')
              bounds = None
            elif not aggregator.preserves_bounds:
              bounds = None
        else:
          shape = enabled.buffer.shape
          dtype = enabled.buffer.dtype

        if bounds:
          spec = specs.BoundedArray(minimum=bounds[0],
                                    maximum=bounds[1],
                                    shape=shape,
                                    dtype=dtype,
                                    name=name)
        else:
          spec = specs.Array(shape=shape, dtype=dtype, name=name)

        out_dict[name] = spec
      return out_dict

    if self._is_nested:
      enabled_specs = type(self._enabled_structure)(
          make_observation_spec_dict(enabled_dict)
          for enabled_dict in self._enabled_structure)
    else:
      enabled_specs = make_observation_spec_dict(self._enabled_structure)

    return enabled_specs

  def prepare_for_next_control_step(self):
    """Simulates the next control step and optimizes the update schedule."""
    if self._enabled_structure is None:
      raise RuntimeError('`reset` must be called before `before_step`.')
    for enabled in self._enabled_list:

      if (enabled.update_interval == DEFAULT_UPDATE_INTERVAL
          and enabled.delay == DEFAULT_DELAY
          and enabled.buffer_size < self._physics_steps_per_control_step):
        for i in reversed(range(enabled.buffer_size)):
          next_step = (
              self._step_counter + self._physics_steps_per_control_step - i)
          next_delay = DEFAULT_DELAY
          enabled.update_schedule.append((next_step, next_delay))
      else:
        if enabled.update_schedule:
          last_scheduled_step = enabled.update_schedule[-1][0]
        else:
          last_scheduled_step = self._step_counter
        max_step = self._step_counter + 2 * self._physics_steps_per_control_step
        while last_scheduled_step < max_step:
          next_update_interval = _call_if_callable(enabled.update_interval)
          next_step = last_scheduled_step + next_update_interval
          next_delay = _call_if_callable(enabled.delay)
          enabled.update_schedule.append((next_step, next_delay))
          last_scheduled_step = next_step
        # Optimize the schedule by planning ahead and dropping unseen entries.
        enabled.buffer.drop_unobserved_upcoming_items(
            enabled.update_schedule, self._physics_steps_per_control_step)

  def update(self):
    if self._enabled_structure is None:
      raise RuntimeError('`reset` must be called before `after_substep`.')
    self._step_counter += 1
    for enabled in self._enabled_list:
      if (enabled.update_schedule and
          enabled.update_schedule[0][0] == self._step_counter):
        timestamp, delay = enabled.update_schedule.popleft()
        enabled.buffer.insert(
            timestamp, delay,
            enabled.observation_callable())

  def get_observation(self):
    """Gets the current observation.

    The returned observation is only valid as of the previous call
    to `reset`. In particular, it is an error to call this function before
    the first call to `reset`.

    Returns:
      A dict, or list of dicts, or tuple of dicts, of observation values.
      The returned structure corresponds to the structure of the `observables`
      that was given at initialization time.

    Raises:
      RuntimeError: If this method is called before `reset` has been called.
    """
    if self._enabled_structure is None:
      raise RuntimeError('`reset` must be called before `observation`.')

    def aggregate_dict(enabled_dict):
      out_dict = type(enabled_dict)()
      for name, enabled in six.iteritems(enabled_dict):
        if enabled.observable.aggregator:
          aggregated = enabled.observable.aggregator(
              enabled.buffer.read(self._step_counter))
        else:
          aggregated = enabled.buffer.read(self._step_counter)
        out_dict[name] = aggregated
      return out_dict

    if self._is_nested:
      return type(self._enabled_structure)(
          aggregate_dict(enabled_dict)
          for enabled_dict in self._enabled_structure)
    else:
      return aggregate_dict(self._enabled_structure)
