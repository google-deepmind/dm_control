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
"""Utilities and base classes used exclusively in the gui package."""

import abc
import threading
import time

from dm_control.viewer import user_input
import six

_DOUBLE_CLICK_INTERVAL = 0.25  # seconds


@six.add_metaclass(abc.ABCMeta)
class InputEventsProcessor(object):
  """Thread safe input events processor."""

  def __init__(self):
    """Instance initializer."""
    self._lock = threading.RLock()
    self._events = []

  def add_event(self, receivers, *args):
    """Adds a new event to the processing queue."""
    if not all(callable(receiver) for receiver in receivers):
      raise TypeError('Receivers are expected to be callables.')
    def event():
      for receiver in list(receivers):
        receiver(*args)
    with self._lock:
      self._events.append(event)

  def process_events(self):
    """Invokes each of the events in the queue.

    Thread safe for queue access but not during event invocations.

    This method must be called regularly on the main thread.
    """
    with self._lock:
      # Swap event buffers quickly so that we don't block the input thread for
      # too long.
      events_to_process = self._events
      self._events = []

    # Now that we made the swap, process the received events in our own time.
    for event in events_to_process:
      event()


class DoubleClickDetector(object):
  """Detects double click events."""

  def __init__(self):
    self._double_clicks = {}

  def process(self, button, action):
    """Attempts to identify a mouse button click as a double click event."""
    if action != user_input.PRESS:
      return False

    curr_time = time.time()
    timestamp = self._double_clicks.get(button, None)
    if timestamp is None:
      # No previous click registered.
      self._double_clicks[button] = curr_time
      return False
    else:
      time_elapsed = curr_time - timestamp
      if time_elapsed < _DOUBLE_CLICK_INTERVAL:
        # Double click discovered.
        self._double_clicks[button] = None
        return True
      else:
        # The previous click was too long ago, so discard it and start a fresh
        # timer.
        self._double_clicks[button] = curr_time
        return False
