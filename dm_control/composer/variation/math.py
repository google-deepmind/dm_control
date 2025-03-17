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

"""Math operations on variation objects."""

import abc

from dm_control.composer.variation import base
from dm_control.composer.variation.variation_values import evaluate

import numpy as np


class MathOp(base.Variation):
  """Base MathOp class for applying math operations on variation objects.

  Subclasses need to implement `_callable`, which takes in a single value and
  applies the desired math operation. This operation gets applied to the result
  of the evaluated base variation object passed at construction. Structured
  variation objects are automatically traversed.
  """

  def __init__(self, *args, **kwargs):
    self._args = args
    self._kwargs = kwargs

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    local_args = evaluate(
        self._args,
        initial_value=initial_value,
        current_value=current_value,
        random_state=random_state)
    local_kwargs = evaluate(
        self._kwargs,
        initial_value=initial_value,
        current_value=current_value,
        random_state=random_state)
    return self._callable(*local_args, **local_kwargs)

  @property
  @abc.abstractmethod
  def _callable(self):
    pass

  def __eq__(self, other):
    if not isinstance(other, type(self)):
      return False
    return (
        self._args == other._args
        and self._kwargs == other._kwargs
    )

  def __repr__(self):
    return '{}(args={}, kwargs={})'.format(
        type(self).__name__,
        self._args,
        self._kwargs,
    )


class Log(MathOp):

  @property
  def _callable(self):
    return np.log


class Max(MathOp):

  @property
  def _callable(self):
    return np.max


class Min(MathOp):

  @property
  def _callable(self):
    return np.min


class Norm(MathOp):

  @property
  def _callable(self):
    return np.linalg.norm
