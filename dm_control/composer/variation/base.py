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

"""Base class for variations and binary operations on variations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import operator

from dm_control.composer.variation import variation_values
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class Variation(object):
  """Abstract base class for variations."""

  @abc.abstractmethod
  def __call__(self, initial_value, current_value, random_state):
    """Generates a value for this variation.

    Args:
      initial_value: The original value of the attribute being varied.
        Absolute variations may ignore this argument.
      current_value: The current value of the attribute being varied.
        Absolute variations may ignore this argument.
      random_state: A `numpy.RandomState` used to generate the value.
        Deterministic variations may ignore this argument.

    Returns:
      The next value for this variation.
    """

  def __add__(self, other):
    return _BinaryOperation(operator.add, self, other)

  def __radd__(self, other):
    return _BinaryOperation(operator.add, other, self)

  def __sub__(self, other):
    return _BinaryOperation(operator.sub, self, other)

  def __rsub__(self, other):
    return _BinaryOperation(operator.sub, other, self)

  def __mul__(self, other):
    return _BinaryOperation(operator.mul, self, other)

  def __rmul__(self, other):
    return _BinaryOperation(operator.mul, other, self)

  def __truediv__(self, other):
    return _BinaryOperation(operator.truediv, self, other)

  def __rtruediv__(self, other):
    return _BinaryOperation(operator.truediv, other, self)

  def __floordiv__(self, other):
    return _BinaryOperation(operator.floordiv, self, other)

  def __rfloordiv__(self, other):
    return _BinaryOperation(operator.floordiv, other, self)

  def __pow__(self, other):
    return _BinaryOperation(operator.pow, self, other)

  def __rpow__(self, other):
    return _BinaryOperation(operator.pow, other, self)

  def __getitem__(self, index):
    return _GetItemOperation(self, index)

  def __neg__(self):
    return _UnaryOperation(operator.neg, self)


class _UnaryOperation(Variation):
  """Represents the result of applying a unary operator to a Variation."""

  def __init__(self, op, variation):
    self._op = op
    self._variation = variation

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    value = variation_values.evaluate(
        self._variation, initial_value, current_value, random_state)
    return self._op(value)


class _BinaryOperation(Variation):
  """Represents the result of applying a binary operator to two Variations."""

  def __init__(self, op, first, second):
    self._first = first
    self._second = second
    self._op = op

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    first_value = variation_values.evaluate(
        self._first, initial_value, current_value, random_state)
    second_value = variation_values.evaluate(
        self._second, initial_value, current_value, random_state)
    return self._op(first_value, second_value)


class _GetItemOperation(Variation):

  def __init__(self, variation, index):
    self._variation = variation
    self._index = index

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    value = variation_values.evaluate(
        self._variation, initial_value, current_value, random_state)
    return np.asarray(value)[self._index]
