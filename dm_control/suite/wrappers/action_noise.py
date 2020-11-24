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

"""Wrapper control suite environments that adds Gaussian noise to actions."""

import dm_env
import numpy as np


_BOUNDS_MUST_BE_FINITE = (
    'All bounds in `env.action_spec()` must be finite, got: {action_spec}')


class Wrapper(dm_env.Environment):
  """Wraps a control environment and adds Gaussian noise to actions."""

  def __init__(self, env, scale=0.01):
    """Initializes a new action noise Wrapper.

    Args:
      env: The control suite environment to wrap.
      scale: The standard deviation of the noise, expressed as a fraction
        of the max-min range for each action dimension.

    Raises:
      ValueError: If any of the action dimensions of the wrapped environment are
        unbounded.
    """
    action_spec = env.action_spec()
    if not (np.all(np.isfinite(action_spec.minimum)) and
            np.all(np.isfinite(action_spec.maximum))):
      raise ValueError(_BOUNDS_MUST_BE_FINITE.format(action_spec=action_spec))
    self._minimum = action_spec.minimum
    self._maximum = action_spec.maximum
    self._noise_std = scale * (action_spec.maximum - action_spec.minimum)
    self._env = env

  def step(self, action):
    noisy_action = action + self._env.task.random.normal(scale=self._noise_std)
    # Clip the noisy actions in place so that they fall within the bounds
    # specified by the `action_spec`. Note that MuJoCo implicitly clips out-of-
    # bounds control inputs, but we also clip here in case the actions do not
    # correspond directly to MuJoCo actuators, or if there are other wrapper
    # layers that expect the actions to be within bounds.
    np.clip(noisy_action, self._minimum, self._maximum, out=noisy_action)
    return self._env.step(noisy_action)

  def reset(self):
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._env.action_spec()

  def __getattr__(self, name):
    return getattr(self._env, name)
