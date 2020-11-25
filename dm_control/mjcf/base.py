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

"""Base class for all MJCF elements in the object model."""

import abc


class Element(metaclass=abc.ABCMeta):
  """Abstract base class for an MJCF element.

  This class is provided so that `isinstance(foo, Element)` is `True` for all
  Element-like objects. We do not implement the actual element here because
  the actual object returned from traversing the object hierarchy is a
  weakproxy-like proxy to an actual element. This is because we do not allow
  orphaned non-root elements, so when a particular element is removed from the
  tree, all references held automatically become invalid.
  """
  __slots__ = []
