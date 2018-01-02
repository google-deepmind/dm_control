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

"""Corruptors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import copy
import functools

# Internal dependencies.

import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class CorruptorBase(object):

  @abc.abstractmethod
  def __call__(self, x):
    """Returns a corrupted version of the input x."""

  @abc.abstractmethod
  def reset(self):
    """Resets the internal state of the corruptor."""


class Delay(CorruptorBase):
  """Applies a delay to the input."""

  def __init__(self, steps, padding=None):
    """Initialize an instance of `Delay`.

    Args:
      steps: An int, number of steps for the delay.
      padding: An optional numpy array or a function. The output in the
        first `steps`.

    Raises:
      ValueError: When `steps` <= 0.
    """
    if steps <= 0:
      raise ValueError('Delay steps should be greater than 0, %d found', steps)
    self._buffer = collections.deque(maxlen=steps + 1)
    self._padding = padding or (lambda x: np.zeros(x.shape))

  def __call__(self, x):
    """Returns the input to this function from `steps` calls ago."""
    self._buffer.append(copy.deepcopy(x))
    if len(self._buffer) == self._buffer.maxlen:
      return self._buffer.popleft()
    else:
      return self._padding(x) if callable(self._padding) else self._padding

  def reset(self):
    """Resets the buffer."""
    self._buffer.clear()


class StatelessNoise(CorruptorBase):
  """Applies noise to an input without relying on any internal state."""

  def __init__(self, noise_function, **noise_parameters):
    """Initialize an instance of `StatelessNoise`.

    Args:
      noise_function: A function, adding noise to its input.
      **noise_parameters: Additional keyword arguments taken by the
        `noise_function`.
    """
    self._noise_function = functools.partial(noise_function, **noise_parameters)

  def __call__(self, x):
    """Returns the input to this function with noise added."""
    return self._noise_function(x)

  def reset(self):
    pass


def gaussian_noise(x, std):
  """Adds gaussian noise to each dimension of x.

  Example of gaussian noise corruptor:
  ```python
  corruptor = StatelessNoise(noise_function=gaussian_noise,
                             noise_parameter={'std': .1})
  ```

  Args:
    x: A numpy array, the input.
    std: A number, standard deviation of the gaussian noise.

  Returns:
    A numpy array with the same dimension as x, which adds a noise draw from a
    normal distribution to each dimension of x.
  """
  return x + np.random.standard_normal(x.shape) * std
