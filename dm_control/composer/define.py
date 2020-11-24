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

"""Decorators for Entity methods returning elements and observables."""

import abc
import threading


class cached_property(property):  # pylint: disable=invalid-name
  """A property that is evaluated only once per object instance."""

  def __init__(self, func, doc=None):
    super(cached_property, self).__init__(fget=func, doc=doc)
    self.lock = threading.RLock()

  def __get__(self, obj, cls):
    if obj is None:
      return self
    name = self.fget.__name__
    obj_dict = obj.__dict__
    try:
      # Try returning a precomputed value without locking first.
      # Profiling shows that the lock takes up a non-trivial amount of time.
      return obj_dict[name]
    except KeyError:
      # The value hasn't been computed, now we have to lock.
      with self.lock:
        try:
          # Check again whether another thread has already computed the value.
          return obj_dict[name]
        except KeyError:
          # Otherwise call the function, cache the result, and return it
          return obj_dict.setdefault(name, self.fget(obj))


# A decorator for base.Observables methods returning an observable. This
# decorator should be used by abstract base classes to indicate sub-classes need
# to implement a corresponding @observavble annotated method.
abstract_observable = abc.abstractproperty  # pylint: disable=invalid-name


class observable(cached_property):  # pylint: disable=invalid-name
  """A decorator for base.Observables methods returning an observable.

  The body of the decorated function is evaluated at Entity construction time
  and the observable is cached.
  """
  pass
