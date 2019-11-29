"""Wrapper that scales actions to a specific range."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import dm_env

_BOUNDS_MUST_BE_FINITE = (
    "All bounds in `{object_type}` must be finite, got: {action_spec}")

_MUST_BE_BROADCASTABLE = (
    "{bound_type} must be broadcastable to `action_spec.{bound_type}`. Got"
    " {bound_type}={bound_result} and"
    " action_spec.{bound_type}={action_spec.{bound_type}}.")


class Wrapper(dm_env.Environment):
  """Wraps a control environment to rescale actions to a specific range."""

  def __init__(self, env, minimum, maximum):
    """Initializes a new action scale Wrapper.

    Args:
      env: The control suite environment to wrap.
      minimum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.

    Raises:
      ValueError: If `minimum` or `maximum` are not broadcastable to
        `shape` or if any of the elements in minimum/maximum are not
        finite.
    """
    action_spec = env.action_spec()
    if not (np.all(np.isfinite(action_spec.minimum)) and
            np.all(np.isfinite(action_spec.maximum))):
      raise ValueError(
          _BOUNDS_MUST_BE_FINITE.format(object_type='env.action_spec()',
                                        action_spec=action_spec))

    if minimum is None:
      minimum = action_spec.minimum
    minimum = minimum + np.zeros_like(action_spec.minimum)

    if not np.all(np.isfinite(minimum)):
      raise ValueError(
          _BOUNDS_MUST_BE_FINITE.format(object_type='minimum',
                                        action_spec=minimum))

    assert minimum.shape == action_spec.minimum.shape, (
        _MUST_BE_BROADCASTABLE.format(bound_type="minimum",
                                      bound_result=minimum,
                                      action_spec=action_spec))

    if maximum is None:
      maximum = action_spec.maximum
    maximum = maximum + np.zeros_like(action_spec.maximum)

    if not np.all(np.isfinite(maximum)):
      raise ValueError(
          _BOUNDS_MUST_BE_FINITE.format(object_type='maximum',
                                        action_spec=maximum))

    assert maximum.shape == action_spec.maximum.shape, (
        _MUST_BE_BROADCASTABLE.format(bound_type="maximum",
                                      bound_result=maximum,
                                      action_spec=action_spec))

    self._minimum_unscaled = action_spec.minimum
    self._maximum_unscaled = action_spec.maximum

    self._minimum = minimum
    self._maximum = maximum

    self._action_spec = type(action_spec)(
        shape=action_spec.shape,
        dtype=action_spec.dtype,
        name=action_spec.name,
        minimum=minimum,
        maximum=maximum,
    )
    self._env = env

  def step(self, action):
    minimum = self._minimum
    maximum = self._maximum

    minimum_unscaled = self._minimum_unscaled
    maximum_unscaled = self._maximum_unscaled

    assert np.all(np.less_equal(minimum, action)), (action, minimum)
    assert np.all(np.less_equal(action, maximum)), (action, maximum)

    action = (
      minimum_unscaled
      + (maximum_unscaled - minimum_unscaled)
      * ((action - minimum) / (maximum - minimum)))

    action = np.clip(action, minimum_unscaled, maximum_unscaled)

    return self._env.step(action)

  def reset(self):
    return self._env.reset()

  def observation_spec(self):
    return self._env.observation_spec()

  def action_spec(self):
    return self._action_spec

  def __getattr__(self, name):
    return getattr(self._env, name)
