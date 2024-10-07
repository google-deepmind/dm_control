# Copyright 2024 The dm_control Authors.
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

"""A broadcaster that allows sharing of variation values across many callers."""

import collections
import weakref

from dm_control.composer import variation


class VariationBroadcaster:
  """Allows a variation to be broadcasted to multiple callers.

  This class wraps a `Variation` object and generates multiple proxies that
  can be used in place of the wrapped `Variation`. The broadcaster updates its
  value in rounds. At the beginning of each round, the broadcaster re-evaluates
  the wrapped `Variation` and caches the new value internally. When a proxy
  is called, the broadcaster will return this cached value, thus ensuring that
  all proxied values are the same. The round ends when all of the proxies have
  been called exactly once. It is an error to call any particular proxy more
  than once per round.
  """

  def __init__(self, wrapped_variation: variation.Variation):
    self._wrapped_variation = wrapped_variation
    self._cached_values = weakref.WeakKeyDictionary()

  def get_proxy(self) -> variation.Variation:
    """Returns a `Variation` to be used in place of the wrapped `Variation`."""
    new_proxy = _BroadcastedValueProxy(self)
    self._cached_values[new_proxy] = collections.deque()
    return new_proxy

  def _get_value(self, proxy, random_state):
    """Returns the variation value for a proxy owned by this broadcaster."""
    cached_values = self._cached_values[proxy]
    if not cached_values:
      new_value = variation.evaluate(
          self._wrapped_variation, None, None, random_state)
      for values in self._cached_values.values():
        values.append(new_value)
    return cached_values.popleft()


class _BroadcastedValueProxy(variation.Variation):

  def __init__(self, broadcaster):
    self._broadcaster = broadcaster

  def __call__(self, initial_value=None, current_value=None, random_state=None):
    value = self._broadcaster._get_value(self, random_state)  # pylint: disable=protected-access
    return value
