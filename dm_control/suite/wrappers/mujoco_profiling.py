# Copyright 2019 The dm_control Authors.
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
"""Wrapper that adds pixel observations to a control environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import dm_env
from dm_env import specs
import numpy as np

STATE_KEY = 'state'


class Wrapper(dm_env.Environment):
  """Wraps a control environment and adds an observation with profile data.

  The profile data describes the time Mujoco spent in a "step", and the
  observation consists of two values: the cumulative time spent on steps
  (in seconds), and the number of times the profiling timer was called.
  """

  def __init__(self, env, observation_key='step_timing'):
    """Initializes a new mujoco_profiling Wrapper.

    Args:
      env: The environment to wrap.
      observation_key: Optional custom string specifying the profile
        observation's key in the `OrderedDict` of observations. Defaults to
        'step_timing'.

    Raises:
      ValueError: If `env`'s observation spec is not compatible with the
        wrapper. Supported formats are a single array, or a dict of arrays.
      ValueError: If `env`'s observation already contains the specified
        `observation_key`.
    """
    wrapped_observation_spec = env.observation_spec()

    if isinstance(wrapped_observation_spec, specs.Array):
      self._observation_is_dict = False
      invalid_keys = set([STATE_KEY])
    elif isinstance(wrapped_observation_spec, collections.MutableMapping):
      self._observation_is_dict = True
      invalid_keys = set(wrapped_observation_spec.keys())
    else:
      raise ValueError('Unsupported observation spec structure.')

    if observation_key in invalid_keys:
      raise ValueError(
          'Duplicate or reserved observation key {!r}.'.format(observation_key))

    if self._observation_is_dict:
      self._observation_spec = wrapped_observation_spec.copy()
    else:
      self._observation_spec = collections.OrderedDict()
      self._observation_spec[STATE_KEY] = wrapped_observation_spec

    env.physics.enable_profiling()

    # Extend observation spec.
    self._observation_spec[observation_key] = specs.Array(
        shape=(2,), dtype=np.double, name=observation_key)

    self._env = env
    self._observation_key = observation_key

  def reset(self):
    return self._add_profile_observation(self._env.reset())

  def step(self, action):
    return self._add_profile_observation(self._env.step(action))

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._env.action_spec()

  def _add_profile_observation(self, time_step):
    if self._observation_is_dict:
      observation = type(time_step.observation)(time_step.observation)
    else:
      observation = collections.OrderedDict()
      observation[STATE_KEY] = time_step.observation

    # timer[0] is the step timer. There are lots of different timers (see
    # mujoco/hdrs/mjdata.h)
    # but we only care about the step timer.
    timing = self._env.physics.data.timer[0]

    observation[self._observation_key] = np.array([timing[0], timing[1]],
                                                  dtype=np.double)
    return time_step._replace(observation=observation)

  def __getattr__(self, name):
    return getattr(self._env, name)
