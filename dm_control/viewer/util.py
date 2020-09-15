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
"""Utility classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import itertools
import sys
import time
import traceback

from absl import logging
import six

# Lower bound of the time multiplier set through TimeMultiplier class.
_MIN_TIME_MULTIPLIER = 1./32.
# Upper bound of the time multiplier set through TimeMultiplier class.
_MAX_TIME_MULTIPLIER = 1.


def is_scalar(value):
  """Checks if the supplied value can be converted to a scalar."""
  try:
    float(value)
  except (TypeError, ValueError):
    return False
  else:
    return True


def to_iterable(item):
  """Converts an item or iterable into an iterable."""
  if isinstance(item, six.string_types):
    return [item]
  elif isinstance(item, collections.Iterable):
    return item
  else:
    return [item]


class QuietSet(object):
  """A set-like container that quietly processes removals of missing keys."""

  def __init__(self):
    self._items = set()

  def __iadd__(self, items):
    """Adds `items`, avoiding duplicates.

    Args:
      items: An iterable of items to add, or a single item to add.

    Returns:
      This instance of `QuietSet`.
    """
    self._items.update(to_iterable(items))
    self._items.discard(None)
    return self

  def __isub__(self, items):
    """Detaches `items`.

    Args:
      items: An iterable of items to detach, or a single item to detach.

    Returns:
      This instance of `QuietSet`.
    """
    for item in to_iterable(items):
      self._items.discard(item)
    return self

  def __len__(self):
    return len(self._items)

  def __iter__(self):
    return iter(self._items)


def interleave(a, b):
  """Interleaves the contents of two iterables."""
  return itertools.chain.from_iterable(zip(a, b))


class TimeMultiplier(object):
  """Controls the relative speed of the simulation compared to realtime."""

  def __init__(self, initial_time_multiplier):
    """Instance initializer.

    Args:
      initial_time_multiplier: A float scalar specifying the initial speed of
        the simulation with 1.0 corresponding to realtime.
    """
    self.set(initial_time_multiplier)

  def get(self):
    """Returns the current time factor value."""
    return self._real_time_multiplier

  def set(self, value):
    """Modifies the time factor.

    Args:
      value: A float scalar, new value of the time factor.
    """
    self._real_time_multiplier = max(
        _MIN_TIME_MULTIPLIER, min(_MAX_TIME_MULTIPLIER, value))

  def __str__(self):
    """Returns a formatted string containing the time factor."""
    if self._real_time_multiplier >= 1.0:
      time_factor = '%d' % self._real_time_multiplier
    else:
      time_factor = '1/%d' % (1.0 // self._real_time_multiplier)
    return time_factor

  def increase(self):
    """Doubles the current time factor value."""
    self.set(self._real_time_multiplier * 2.)

  def decrease(self):
    """Halves the current time factor value."""
    self.set(self._real_time_multiplier / 2.)


class Integrator(object):
  """Integrates a value and averages it for the specified period of time."""

  def __init__(self, refresh_rate=.5):
    """Instance initializer.

    Args:
      refresh_rate: How often, in seconds, is the integrated value averaged.
    """
    self._value = 0
    self._value_acc = 0
    self._num_samples = 0
    self._sampling_timestamp = time.time()
    self._refresh_rate = refresh_rate

  @property
  def value(self):
    """Returns the averaged value."""
    return self._value

  @value.setter
  def value(self, val):
    """Integrates the new value."""
    self._value_acc += val
    self._num_samples += 1

    time_elapsed = time.time() - self._sampling_timestamp
    if time_elapsed >= self._refresh_rate:
      self._value = self._value_acc / self._num_samples
      self._value_acc = 0
      self._num_samples = 0
      self._sampling_timestamp = time.time()


class AtomicAction(object):
  """An action that cannot be interrupted."""

  def __init__(self, state_change_callback=None):
    """Instance initializer.

    Args:
      state_change_callback: Callable invoked when action changes its state.
    """
    self._state_change_callback = state_change_callback
    self._watermark = None

  def begin(self, watermark):
    """Begins the action, signing it with the specified watermark."""
    if self._watermark is None:
      self._watermark = watermark
      if self._state_change_callback is not None:
        self._state_change_callback(watermark)

  def end(self, watermark):
    """Ends a started action, provided the watermarks match."""
    if self._watermark == watermark:
      self._watermark = None
      if self._state_change_callback is not None:
        self._state_change_callback(None)

  @property
  def in_progress(self):
    """Returns a boolean value to indicate if the being method was called."""
    return self._watermark is not None

  @property
  def watermark(self):
    """Returns the watermark passed to begin() method call, or None.

    None will be returned if the action is not in progress.
    """
    return self._watermark


class ObservableFlag(QuietSet):
  """Observable boolean flag.

  The QuietState provides necessary functionality for managing listeners.

  A listener is a callable that takes one boolean parameter.
  """

  def __init__(self, initial_value):
    """Instance initializer.

    Args:
      initial_value: A boolean value with the initial state of the flag.
    """
    self._value = initial_value
    super(ObservableFlag, self).__init__()

  def toggle(self):
    """Toggles the value True/False."""
    self._value = not self._value
    for listener in self._items:
      listener(self._value)

  def __iadd__(self, value):
    """Add new listeners and update them about the state."""
    listeners = to_iterable(value)
    super(ObservableFlag, self).__iadd__(listeners)
    for listener in listeners:
      listener(self._value)
    return self

  @property
  def value(self):
    """Value of the flag."""
    return self._value

  @value.setter
  def value(self, val):
    if self._value != val:
      for listener in self._items:
        listener(self._value)
    self._value = val


class Timer(object):
  """Measures time elapsed between two ticks."""

  def __init__(self):
    """Instance initializer."""
    self._previous_time = time.time()
    self._measured_time = 0.

  def tick(self):
    """Updates the timer.

    Returns:
      Time elapsed since the last call to this method.
    """
    curr_time = time.time()
    self._measured_time = curr_time - self._previous_time
    self._previous_time = curr_time
    return self._measured_time

  @contextlib.contextmanager
  def measure_time(self):
    start_time = time.time()
    yield
    self._measured_time = time.time() - start_time

  @property
  def measured_time(self):
    return self._measured_time


class ErrorLogger(object):
  """A context manager that catches and logs all errors."""

  def __init__(self, listeners):
    """Instance initializer.

    Args:
      listeners: An iterable of callables, listeners to inform when an error
        is caught. Each callable should accept a single string argument.
    """
    self._error_found = False
    self._listeners = listeners

  def __enter__(self, *args):
    self._error_found = False

  def __exit__(self, exception_type, exception_value, tb):
    if exception_type:
      self._error_found = True
      error_message = ('dm_control viewer intercepted an environment error.\n'
                       'Original message: {}'.format(exception_value))
      logging.error(error_message)
      sys.stderr.write(error_message + '\nTraceback:\n')
      traceback.print_tb(tb)
      for listener in self._listeners:
        listener('{}'.format(exception_value))
      return True

  @property
  def errors_found(self):
    """Returns True if any errors were caught."""
    return self._error_found


class NullErrorLogger(object):
  """A context manager that replaces an ErrorLogger.

  This error logger will pass all thrown errors through.
  """

  def __enter__(self, *args):
    pass

  def __exit__(self, error_type, value, tb):
    pass

  @property
  def errors_found(self):
    """Returns True if any errors were caught."""
    return False
