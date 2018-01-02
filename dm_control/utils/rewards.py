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

"""Soft indicator function evaluating whether a number is within bounds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
import numpy as np

# The value returned by tolerance() at `margin` distance from `bounds` interval.
_DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoids(x, value_at_1, sigmoid):
  """Returns 1 when `x` == 0, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
    sigmoid: String, choice of sigmoid type.

  Returns:
    A numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
      `quadratic` sigmoids which allow `value_at_1` == 0.
    ValueError: If `sigmoid` is of an unknown type.
  """
  if sigmoid in ('cosine', 'linear', 'quadratic'):
    if not 0 <= value_at_1 < 1:
      raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
                       'got {}.'.format(value_at_1))
  else:
    if not 0 < value_at_1 < 1:
      raise ValueError('`value_at_1` must be strictly between 0 and 1, '
                       'got {}.'.format(value_at_1))

  if sigmoid == 'gaussian':
    scale = np.sqrt(-2 * np.log(value_at_1))
    return np.exp(-0.5 * (x*scale)**2)

  elif sigmoid == 'hyperbolic':
    scale = np.arccosh(1/value_at_1)
    return 1 / np.cosh(x*scale)

  elif sigmoid == 'long_tail':
    scale = np.sqrt(1/value_at_1 - 1)
    return 1 / ((x*scale)**2 + 1)

  elif sigmoid == 'cosine':
    scale = np.arccos(2*value_at_1 - 1) / np.pi
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi*scaled_x))/2, 0.0)

  elif sigmoid == 'linear':
    scale = 1-value_at_1
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

  elif sigmoid == 'quadratic':
    scale = np.sqrt(1-value_at_1)
    scaled_x = x*scale
    return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

  elif sigmoid == 'tanh_squared':
    scale = np.arctanh(np.sqrt(1-value_at_1))
    return 1 - np.tanh(x*scale)**2

  else:
    raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
              value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
  """Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.

  Args:
    x: A scalar or numpy array.
    bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
      the target interval. These can be infinite if the interval is unbounded
      at one or both ends, or they can be equal to one another if the target
      value is exact.
    margin: Float. Parameter that controls how steeply the output decreases as
      `x` moves out-of-bounds.
      * If `margin == 0` then the output will be 0 for all values of `x`
        outside of `bounds`.
      * If `margin > 0` then the output will decrease sigmoidally with
        increasing distance from the nearest bound.
    sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
       'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
    value_at_margin: A float between 0 and 1 specifying the output value when
      the distance from `x` to the nearest bound is equal to `margin`. Ignored
      if `margin == 0`.

  Returns:
    A float or numpy array with values between 0.0 and 1.0.

  Raises:
    ValueError: If `bounds[0] > bounds[1]`.
    ValueError: If `margin` is negative.
  """
  lower, upper = bounds
  if lower > upper:
    raise ValueError('Lower bound must be <= upper bound.')
  if margin < 0:
    raise ValueError('`margin` must be non-negative.')

  in_bounds = np.logical_and(lower <= x, x <= upper)
  if margin == 0:
    value = np.where(in_bounds, 1.0, 0.0)
  else:
    d = np.where(x < lower, lower - x, x - upper) / margin
    value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

  return float(value) if np.isscalar(x) else value

