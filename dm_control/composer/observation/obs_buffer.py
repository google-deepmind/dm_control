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

""""An object that manages the buffering and delaying of observation."""

import collections
import numpy as np


class InFlightObservation:
  """Represents a delayed observation that may not have arrived yet.

  Attributes:
    arrival: The time at which this observation will be delivered.
    timestamp: The time at which this observation was made.
    delay: The amount of delay between the time at which this observation was
      made and the time at which it is delivered.
    value: The value of this observation.
  """

  __slots__ = ('arrival', 'timestamp', 'delay', 'value')

  def __init__(self, timestamp, delay, value):
    self.arrival = timestamp + delay
    self.timestamp = timestamp
    self.delay = delay
    self.value = value

  def __lt__(self, other):
    # This is implemented to facilitate sorting.
    return self.arrival < other.arrival


class Buffer:
  """An object that manages the buffering and delaying of observation."""

  def __init__(self, buffer_size, shape, dtype, pad_with_initial_value=False,
               strip_singleton_buffer_dim=False):
    """Initializes this observation buffer.

    Args:
      buffer_size: The size of the buffer returned by `read`. Note
        that this does *not* affect size of the internal buffer held by this
        object, which always grow as large as is necessary in the presence of
        large delays.
      shape: The shape of a single observation held by this buffer, which can
        either be a single integer or an iterable of integers. The shape of the
        buffer returned by `read` will then be
        `(buffer_size, shape[0], ..., shape[n])`, unless `buffer_size == 1`
        and `strip_singleton_buffer_dim == True`.
      dtype: The NumPy dtype of observation entries.
      pad_with_initial_value: (optional) A boolean. If `True` then the buffer
        returned by `read` is padded with the first observation value when there
        are fewer observation entries than `buffer_size`. If `False` then the
        buffer returned by `read` is padded with zeroes.
      strip_singleton_buffer_dim: (optional) A boolean, if `True` and
        `buffer_size == 1` then the leading dimension will not be added to the
        shape of the array returned by `read`.
    """
    self._buffer_size = buffer_size
    try:
      shape = tuple(shape)
    except TypeError:
      if isinstance(shape, int):
        shape = (shape,)
      else:
        raise

    self._has_buffer_dim = not (strip_singleton_buffer_dim and buffer_size == 1)
    if self._has_buffer_dim:
      self._buffered_shape = (buffer_size,) + shape
    else:
      self._buffered_shape = shape
    self._dtype = dtype

    # The "arrived" deque contains entries that are due to be delivered now.
    # This deque should never grow beyond buffer_size.
    self._arrived_deque = collections.deque(maxlen=buffer_size)
    if not pad_with_initial_value:
      for _ in range(buffer_size):
        self._arrived_deque.append(
            InFlightObservation(-np.inf, 0, np.full(shape, 0, dtype)))

    # The "pending" deque contains entries that are stored for future delivery.
    # This deque can grow arbitrarily large in presence of long delays.
    self._pending_deque = collections.deque()

  def _update_arrived_deque(self, timestamp):
    while self._pending_deque and self._pending_deque[0].arrival <= timestamp:
      self._arrived_deque.append(self._pending_deque.popleft())

  @property
  def shape(self):
    return self._buffered_shape

  @property
  def dtype(self):
    return self._dtype

  def insert(self, timestamp, delay, value):
    """Inserts a new observation to the buffer.

    This function implicitly updates the internal "clock" of this buffer to
    the timestamp of the new observation, and the internal buffer is trimmed
    accordingly, i.e. at most `buffer_size` items whose delayed arrival time
    preceeds `timestamp` are kept.

    Args:
      timestamp: The time at which this observation was made.
      delay: The amount of delay between the time at which this observation was
        made and the time at which it is delivered.
      value: The value of this observation.

    Raises:
      ValueError: if `delay` is negative.
    """
    # If using `pad_with_initial_value`, the `arrived_deque` would be empty.
    # We can now pad it with the initial value now.
    if not self._arrived_deque:
      for _ in range(self._buffer_size):
        self._arrived_deque.append(InFlightObservation(-np.inf, 0, value))

    self._update_arrived_deque(timestamp)
    new_obs = InFlightObservation(timestamp, delay, np.array(value))
    arrival = new_obs.arrival
    if delay == 0:
      # No delay, so the new observation is due for immediate delivery.
      # Add it to the arrived deque.
      self._arrived_deque.append(new_obs)
    elif delay > 0:
      if not self._pending_deque or arrival > self._pending_deque[-1].arrival:
        # New observation's arrival time is monotonic.
        # Technically, we can handle this in the general code branch below,
        # but since this is assumed to be the "typical" case, the special
        # handling here saves us from repeatedly allocating and deallocating
        # an empty temporary deque.
        self._pending_deque.append(new_obs)
      else:
        # General, out-of-order observation.
        arriving_after_new_obs = collections.deque()
        while self._pending_deque and arrival < self._pending_deque[-1].arrival:
          arriving_after_new_obs.appendleft(self._pending_deque.pop())
        self._pending_deque.append(new_obs)
        for existing_obs in arriving_after_new_obs:
          self._pending_deque.append(existing_obs)
    else:
      raise ValueError('`delay` should not be negative: '
                       'got {!r}'.format(delay))

  def read(self, current_time):
    """Reads the content of the buffer at the given timestamp."""
    self._update_arrived_deque(current_time)
    if self._has_buffer_dim:
      out = np.empty(self._buffered_shape, dtype=self._dtype)
      for i, obs in enumerate(self._arrived_deque):
        out[i] = obs.value
    else:
      out = self._arrived_deque[0].value.copy()
    return out

  def drop_unobserved_upcoming_items(self, observation_schedule, read_interval):
    """Plans an optimal observation schedule for an upcoming control period.

    This function determines which of the proposed upcoming observations will
    never in fact be delivered and removes them from the observation schedule.

    We assume that observations will only be queried at times that are integer
    multiples of `read_interval`. If more observations are generated during
    the upcoming control step than the `buffer_size` of this `Buffer`
    then of those new observations will never be required. This function takes
    into account the delayed arrival time and existing buffered items in the
    planning process.

    Args:
      observation_schedule: An list of `(timestamp, delay)` tuples, where
        `timestamp` is the time at which the observation value will be produced,
        and `delay` is the amount of time the observation will be delayed by.
        This list will be modified in place.
      read_interval: The time interval between successive calls to `read`.
        We assume that observations will only be queried at times that are
        integer multiples of `read_interval`.
    """
    # Private deques to simulate what the deques will look like in the future,
    # according to the proposed upcoming observation schedule.
    future_arrived_deque = collections.deque()
    future_pending_deque = collections.deque()

    # Take existing buffered observations into account when planning the
    # upcoming schedule.
    def get_next_existing_timestamp():
      for obs in reversed(self._pending_deque):
        yield InFlightObservation(obs.timestamp, obs.delay, None)
      while True:
        yield InFlightObservation(-np.inf, 0, None)
    existing_timestamp_iter = get_next_existing_timestamp()
    existing_timestamp = next(existing_timestamp_iter)

    # Build the simulated state of the pending deque at the end of the proposed
    # schedule.
    sorted_schedule = sorted([InFlightObservation(time[0], time[1], None)
                              for time in observation_schedule])
    for new_timestamp in reversed(sorted_schedule):
      # We don't need to worry about any existing item that are delivered before
      # the first new item, since those are purged independently of our
      # proposed new observations.
      while existing_timestamp.arrival > new_timestamp.arrival:
        future_pending_deque.appendleft(existing_timestamp)
        existing_timestamp = next(existing_timestamp_iter)
      future_pending_deque.appendleft(new_timestamp)

    # Find the next timestep at which `read` is called.
    first_proposed_timestamp = min(t for t, _ in observation_schedule)
    next_read_time = read_interval * int(np.ceil(
        first_proposed_timestamp // read_interval))

    # Build the simulated state of the arrived deque at each subsequent
    # control steps.
    while future_pending_deque:
      # Keep track of observations that are delivered for the first time
      # during this control timestep.
      newly_arrived = collections.deque()
      while (future_pending_deque and
             future_pending_deque[0].arrival <= next_read_time):
        # `fake_observation` is an `InFlightObservation` without `value`.
        fake_observation = future_pending_deque.popleft()
        future_arrived_deque.append(fake_observation)
        newly_arrived.append(fake_observation)
      while len(future_arrived_deque) > self._buffer_size:
        stale = future_arrived_deque.popleft()
        # Newly-arrived items that become immediately stale are never actually
        # delivered.
        if newly_arrived and stale == newly_arrived[0]:
          newly_arrived.popleft()
          # `stale` might either be one of the existing pending observations or
          # from the proposed schedule.
          if stale.timestamp >= first_proposed_timestamp:
            observation_schedule.remove((stale.timestamp, stale.delay))

      next_read_time += read_interval
