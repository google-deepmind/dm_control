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

"""Deterministic variations."""


from dm_control.composer.variation import base


class Constant(base.Variation):
  """Wraps a constant value into a Variation object.

  This class is provided mainly for use in tests, to check that variations are
  invoked correctly without having to introduce randomness in test cases.
  """

  def __init__(self, value):
    self._value = value

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    return self._value


class Sequence(base.Variation):
  """Variation representing a fixed sequence of values."""

  def __init__(self, values):
    self._values = values
    self._iterator = iter(self._values)

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    try:
      return next(self._iterator)
    except StopIteration:
      self._iterator = iter(self._values)
      return next(self._iterator)


class Identity(base.Variation):

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    return current_value
