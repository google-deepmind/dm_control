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

"""A module that helps manage model variation in Composer environments."""

import collections
import copy

from dm_control.composer.variation.base import Variation
from dm_control.composer.variation.variation_values import evaluate
import six


class _VariationInfo(object):

  __slots__ = ['initial_value', 'variation']

  def __init__(self, initial_value=None, variation=None):
    self.initial_value = initial_value
    self.variation = variation


class MJCFVariator(object):
  """Helper object for applying variations to MJCF attributes.

  An instance of this class remembers the original value of each MJCF attribute
  the first time a variation is applied. The original value is then passed as an
  argument to each variation callable.
  """

  def __init__(self):
    self._variations = collections.defaultdict(dict)

  def bind_attributes(self, element, **kwargs):
    """Binds variations to attributes of an MJCF element.

    Args:
      element: An `mjcf.Element` object.
      **kwargs: Keyword arguments mapping attribute names to the corresponding
        variations. A variation is either a fixed value or a callable that
        optionally takes the original value of an attribute and returns a
        new value.
    """
    for attribute_name, variation in six.iteritems(kwargs):
      if variation is None and attribute_name in self._variations[element]:
        del self._variations[element][attribute_name]
      else:
        initial_value = copy.copy(getattr(element, attribute_name))
        self._variations[element][attribute_name] = (
            _VariationInfo(initial_value, variation))

  def apply_variations(self, random_state):
    """Applies variations in-place to the specified MJCF element.

    Args:
      random_state: A `numpy.random.RandomState` instance.
    """
    for element, attribute_variations in six.iteritems(self._variations):
      new_values = {}
      for attribute_name, variation_info in six.iteritems(attribute_variations):
        current_value = getattr(element, attribute_name)
        if variation_info.initial_value is None:
          variation_info.initial_value = copy.copy(current_value)
        new_values[attribute_name] = evaluate(
            variation_info.variation, variation_info.initial_value,
            current_value, random_state)
      element.set_attributes(**new_values)

  def clear(self):
    """Clears all bound attribute variations."""
    self._variations.clear()

  def reset_initial_values(self):
    for variations in six.itervalues(self._variations):
      for variation_info in six.itervalues(variations):
        variation_info.initial_value = None


class PhysicsVariator(object):
  """Helper object for applying variations to MjModel and MjData.

  An instance of this class remembers the original value of each attribute
  the first time a variation is applied. The original value is then passed as an
  argument to each variation callable.
  """

  def __init__(self):
    self._variations = collections.defaultdict(dict)

  def bind_attributes(self, element, **kwargs):
    """Binds variations to attributes of an MJCF element.

    Args:
      element: An `mjcf.Element` object.
      **kwargs: Keyword arguments mapping attribute names to the corresponding
        variations. A variation is either a fixed value or a callable that
        optionally takes the original value of an attribute and returns a
        new value.
    """
    for attribute_name, variation in six.iteritems(kwargs):
      if variation is None and attribute_name in self._variations[element]:
        del self._variations[element][attribute_name]
      else:
        self._variations[element][attribute_name] = (
            _VariationInfo(None, variation))

  def apply_variations(self, physics, random_state):
    for element, variations in six.iteritems(self._variations):
      binding = physics.bind(element)
      for attribute_name, variation_info in six.iteritems(variations):
        current_value = getattr(binding, attribute_name)
        if variation_info.initial_value is None:
          variation_info.initial_value = copy.copy(current_value)
        setattr(binding, attribute_name, evaluate(
            variation_info.variation, variation_info.initial_value,
            current_value, random_state))

  def clear(self):
    """Clears all bound attribute variations."""
    self._variations.clear()

  def reset_initial_values(self):
    for variations in six.itervalues(self._variations):
      for variation_info in six.itervalues(variations):
        variation_info.initial_value = None
