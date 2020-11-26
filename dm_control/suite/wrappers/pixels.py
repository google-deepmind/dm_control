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

"""Wrapper that adds pixel observations to a control environment."""

import collections
import dm_env
from dm_env import specs

STATE_KEY = 'state'


class Wrapper(dm_env.Environment):
  """Wraps a control environment and adds a rendered pixel observation."""

  def __init__(self, env, pixels_only=True, render_kwargs=None,
               observation_key='pixels'):
    """Initializes a new pixel Wrapper.

    Args:
      env: The environment to wrap.
      pixels_only: If True (default), the original set of 'state' observations
        returned by the wrapped environment will be discarded, and the
        `OrderedDict` of observations will only contain pixels. If False, the
        `OrderedDict` will contain the original observations as well as the
        pixel observations.
      render_kwargs: Optional `dict` containing keyword arguments passed to the
        `mujoco.Physics.render` method.
      observation_key: Optional custom string specifying the pixel observation's
        key in the `OrderedDict` of observations. Defaults to 'pixels'.

    Raises:
      ValueError: If `env`'s observation spec is not compatible with the
        wrapper. Supported formats are a single array, or a dict of arrays.
      ValueError: If `env`'s observation already contains the specified
        `observation_key`.
    """
    if render_kwargs is None:
      render_kwargs = {}

    wrapped_observation_spec = env.observation_spec()

    if isinstance(wrapped_observation_spec, specs.Array):
      self._observation_is_dict = False
      invalid_keys = set([STATE_KEY])
    elif isinstance(wrapped_observation_spec, collections.abc.MutableMapping):
      self._observation_is_dict = True
      invalid_keys = set(wrapped_observation_spec.keys())
    else:
      raise ValueError('Unsupported observation spec structure.')

    if not pixels_only and observation_key in invalid_keys:
      raise ValueError('Duplicate or reserved observation key {!r}.'
                       .format(observation_key))

    if pixels_only:
      self._observation_spec = collections.OrderedDict()
    elif self._observation_is_dict:
      self._observation_spec = wrapped_observation_spec.copy()
    else:
      self._observation_spec = collections.OrderedDict()
      self._observation_spec[STATE_KEY] = wrapped_observation_spec

    # Extend observation spec.
    pixels = env.physics.render(**render_kwargs)
    pixels_spec = specs.Array(
        shape=pixels.shape, dtype=pixels.dtype, name=observation_key)
    self._observation_spec[observation_key] = pixels_spec

    self._env = env
    self._pixels_only = pixels_only
    self._render_kwargs = render_kwargs
    self._observation_key = observation_key

  def reset(self):
    time_step = self._env.reset()
    return self._add_pixel_observation(time_step)

  def step(self, action):
    time_step = self._env.step(action)
    return self._add_pixel_observation(time_step)

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._env.action_spec()

  def _add_pixel_observation(self, time_step):
    if self._pixels_only:
      observation = collections.OrderedDict()
    elif self._observation_is_dict:
      observation = type(time_step.observation)(time_step.observation)
    else:
      observation = collections.OrderedDict()
      observation[STATE_KEY] = time_step.observation

    pixels = self._env.physics.render(**self._render_kwargs)
    observation[self._observation_key] = pixels
    return time_step._replace(observation=observation)

  def __getattr__(self, name):
    return getattr(self._env, name)
