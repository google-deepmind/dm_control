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

"""Standard statistical distributions that conform to the Variation API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import functools

from dm_control.composer import variation
from dm_control.composer.variation import base
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class Distribution(base.Variation):
  """Base Distribution class for sampling a parametrized distribution.

  Subclasses need to implement `_callable`, which needs to return a callable
  based on the random_state passed as arg. This callable then gets called using
  the arguments passed to the constructor, after being evaluated. This allows
  the distribution parameters themselves to be instances of `base.Variation`.
  By default samples are drawn in the shape of `initial_value`, unless the
  optional `single_sample` constructor arg is set to `True`, in which case only
  a single sample is drawn.
  """
  __slots__ = ('_single_sample', '_args', '_kwargs')

  def __init__(self, *args, **kwargs):
    self._single_sample = kwargs.pop('single_sample', False)
    self._args = args
    self._kwargs = kwargs

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    local_random_state = random_state or np.random
    size = (None if self._single_sample or initial_value is None  # pylint: disable=g-long-ternary
            else np.shape(initial_value))
    local_args = variation.evaluate(self._args,
                                    initial_value=initial_value,
                                    current_value=current_value,
                                    random_state=random_state)
    local_kwargs = variation.evaluate(self._kwargs,
                                      initial_value=initial_value,
                                      current_value=current_value,
                                      random_state=random_state)
    return self._callable(local_random_state)(*local_args,
                                              size=size,
                                              **local_kwargs)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError  # Stops infinite recursion during deepcopy.
    elif name in self._kwargs:
      return self._kwargs[name]
    else:
      raise AttributeError(
          '{!r} object has no attribute {!r}'.format(type(self).__name__, name))

  @abc.abstractmethod
  def _callable(self, random_state):
    raise NotImplementedError


class Uniform(Distribution):
  __slots__ = ()

  def __init__(self, low=0.0, high=1.0, single_sample=False):
    super(Uniform, self).__init__(low=low, high=high,
                                  single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.uniform


class UniformInteger(Distribution):
  __slots__ = ()

  def __init__(self, low, high=None, single_sample=False):
    super(UniformInteger, self).__init__(low, high=high,
                                         single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.randint


class UniformChoice(Distribution):
  __slots__ = ()

  def __init__(self, choices, single_sample=False):
    super(UniformChoice, self).__init__(choices, single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.choice


class UniformPointOnSphere(base.Variation):
  """Samples a point on the unit sphere, i.e. a 3D vector with norm 1."""
  __slots__ = ()

  def __init__(self, single_sample=False):
    self._single_sample = single_sample

  def __call__(self, initial_value=None,
               current_value=None, random_state=None):
    random_state = random_state or np.random
    size = (3 if self._single_sample or initial_value is None  # pylint: disable=g-long-ternary
            else np.append(np.shape(initial_value), 3))
    axis = random_state.normal(size=size)
    axis /= np.linalg.norm(axis, axis=-1, keepdims=True)
    return axis


class Normal(Distribution):
  __slots__ = ()

  def __init__(self, loc=0.0, scale=1.0, single_sample=False):
    super(Normal, self).__init__(loc=loc, scale=scale,
                                 single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.normal


class LogNormal(Distribution):
  __slots__ = ()

  def __init__(self, mean=0.0, sigma=1.0, single_sample=False):
    super(LogNormal, self).__init__(mean=mean, sigma=sigma,
                                    single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.lognormal


class Exponential(Distribution):
  __slots__ = ()

  def __init__(self, scale=1.0, single_sample=False):
    super(Exponential, self).__init__(scale=scale, single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.exponential


class Poisson(Distribution):
  __slots__ = ()

  def __init__(self, lam=1.0, single_sample=False):
    super(Poisson, self).__init__(lam=lam, single_sample=single_sample)

  def _callable(self, random_state):
    return random_state.poisson


class Bernoulli(Distribution):
  __slots__ = ()

  def __init__(self, prob=0.5, single_sample=False):
    super(Bernoulli, self).__init__(prob, single_sample=single_sample)

  def _callable(self, random_state):
    return functools.partial(random_state.binomial, 1)


_NEGATIVE_STDEV = '`stdev` must be >= 0, got {}.'
_NEGATIVE_TIMESCALE = '`timescale` must be >= 0, got {}.'


class BiasedRandomWalk(base.Variation):
  """A Class for generating noise from a zero-mean Ornstein-Uhlenbeck process.

  Let
  `retain = np.exp(-1. / timescale)`
  and
  `scale = stdev * sqrt(1 - (retain * retain))`
  Then the discete-time first-order filtered diffusion process
  `x_next = retain * x + N(0, scale))`
  has standard deviation `stdev` and characteristic timescale `timescale`.
  """
  __slots__ = ('_scale', '_value')

  def __init__(self, stdev=0.1, timescale=10.):
    """Initializes a `BiasedRandomWalk`.

    Args:
      stdev: Float. Standard deviation of the output sequence.
      timescale: Integer. Number of timesteps characteristic of the random walk.
        After `timescale` steps the correlation is reduced by exp(-1). Larger
        or equal to 0, where a value of 0 is an uncorrelated normal
        distribution.

    Raises:
      ValueError: if either `stdev` or `timescale` is negative.
    """
    if stdev < 0:
      raise ValueError(_NEGATIVE_STDEV.format(stdev))
    if timescale < 0:
      raise ValueError(_NEGATIVE_TIMESCALE.format(timescale))
    elif timescale == 0:
      self._retain = 0.
    else:
      self._retain = np.exp(-1. / timescale)
    self._scale = stdev * np.sqrt(1 - (self._retain * self._retain))
    self._value = 0.0

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    random_state = random_state or np.random
    self._value = (self._retain * self._value +
                   random_state.normal(loc=0.0, scale=self._scale))
    return self._value
