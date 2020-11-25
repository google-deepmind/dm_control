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
"""Tests for the keyboard module."""

import collections
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.viewer import util
import mock
import numpy as np


class QuietSetTest(absltest.TestCase):

  def test_add_listeners(self):
    subject = util.QuietSet()
    listeners = [object() for _ in range(5)]
    for listener in listeners:
      subject += listener
    self.assertLen(subject, 5)

  def test_add_collection_of_listeners(self):
    subject = util.QuietSet()
    subject += [object() for _ in range(5)]
    self.assertLen(subject, 5)

  def test_add_collection_and_individual_listeners(self):
    subject = util.QuietSet()
    subject += object()
    subject += [object() for _ in range(5)]
    subject += object()
    self.assertLen(subject, 7)

  def test_add_duplicate_listeners(self):
    subject = util.QuietSet()
    listener = object()
    subject += listener
    self.assertLen(subject, 1)
    subject += listener
    self.assertLen(subject, 1)

  def test_remove_listeners(self):
    subject = util.QuietSet()
    listeners = [object() for _ in range(3)]
    for listener in listeners:
      subject += listener

    subject -= listeners[1]
    self.assertLen(subject, 2)

  def test_remove_unregistered_listener(self):
    subject = util.QuietSet()
    listeners = [object() for _ in range(3)]
    for listener in listeners:
      subject += listener

    subject -= object()
    self.assertLen(subject, 3)


class ToIterableTest(parameterized.TestCase):

  def test_scalars_converted_to_iterables(self):
    original_value = 3

    result = util.to_iterable(original_value)
    self.assertIsInstance(result, collections.Iterable)
    self.assertLen(result, 1)
    self.assertEqual(original_value, result[0])

  def test_strings_wrappe_by_list(self):
    original_value = 'test_string'

    result = util.to_iterable(original_value)
    self.assertIsInstance(result, collections.Iterable)
    self.assertLen(result, 1)
    self.assertEqual(original_value, result[0])

  @parameterized.named_parameters(
      ('list', [1, 2, 3]),
      ('set', set([1, 2, 3])),
      ('dict', {'1': 2, '3': 4, '5': 6})
  )
  def test_iterables_remain_unaffected(self, original_value):
    result = util.to_iterable(original_value)
    self.assertEqual(result, original_value)


class InterleaveTest(absltest.TestCase):

  def test_equal_sized_iterables(self):
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [i for i in util.interleave(a, b)]
    np.testing.assert_array_equal([1, 4, 2, 5, 3, 6], c)

  def test_iteration_ends_when_smaller_iterable_runs_out_of_elements(self):
    a = [1, 2, 3]
    b = [4, 5, 6, 7, 8]
    c = [i for i in util.interleave(a, b)]
    np.testing.assert_array_equal([1, 4, 2, 5, 3, 6], c)


class TimeMultiplierTests(absltest.TestCase):

  def setUp(self):
    super(TimeMultiplierTests, self).setUp()
    self.factor = util.TimeMultiplier(initial_time_multiplier=1.0)

  def test_custom_initial_factor(self):
    initial_value = 0.5
    factor = util.TimeMultiplier(initial_time_multiplier=initial_value)
    self.assertEqual(initial_value, factor.get())

  def test_initial_factor_clamped_to_valid_value_range(self):
    too_large_multiplier = util._MAX_TIME_MULTIPLIER + 1.
    too_small_multiplier = util._MIN_TIME_MULTIPLIER - 1.

    factor = util.TimeMultiplier(initial_time_multiplier=too_large_multiplier)
    self.assertEqual(util._MAX_TIME_MULTIPLIER, factor.get())

    factor = util.TimeMultiplier(initial_time_multiplier=too_small_multiplier)
    self.assertEqual(util._MIN_TIME_MULTIPLIER, factor.get())

  def test_increase(self):
    self.factor.decrease()
    self.factor.decrease()
    self.factor.increase()
    self.assertEqual(self.factor._real_time_multiplier, 0.5)

  def test_increase_limit(self):
    self.factor._real_time_multiplier = util._MAX_TIME_MULTIPLIER
    self.factor.increase()
    self.assertEqual(util._MAX_TIME_MULTIPLIER, self.factor.get())

  def test_decrease(self):
    self.factor.decrease()
    self.factor.decrease()
    self.assertEqual(self.factor._real_time_multiplier, 0.25)

  def test_decrease_limit(self):
    self.factor._real_time_multiplier = util._MIN_TIME_MULTIPLIER
    self.factor.decrease()
    self.assertEqual(util._MIN_TIME_MULTIPLIER, self.factor.get())

  def test_stringify_when_less_than_one(self):
    self.assertEqual('1', str(self.factor))
    self.factor.decrease()
    self.assertEqual('1/2', str(self.factor))
    self.factor.decrease()
    self.assertEqual('1/4', str(self.factor))


class IntegratorTests(absltest.TestCase):

  def setUp(self):
    super(IntegratorTests, self).setUp()
    self.integration_step = 1
    self.integrator = util.Integrator(self.integration_step)
    self.integrator._sampling_timestamp = 0.0

  def test_initial_value(self):
    self.assertEqual(0, self.integrator.value)

  def test_integration_step(self):
    with mock.patch(util.__name__ + '.time') as time_mock:
      time_mock.time.return_value = self.integration_step
      self.integrator.value = 1
    self.assertEqual(1, self.integrator.value)

  def test_averaging(self):
    with mock.patch(util.__name__ + '.time') as time_mock:
      time_mock.time.return_value = 0
      self.integrator.value = 1
      self.integrator.value = 1
      self.integrator.value = 1
      time_mock.time.return_value = self.integration_step
      self.integrator.value = 1
    self.assertEqual(1, self.integrator.value)


class AtomicActionTests(absltest.TestCase):

  def setUp(self):
    super(AtomicActionTests, self).setUp()
    self.callback = mock.MagicMock()
    self.action = util.AtomicAction(self.callback)

  def test_starting_and_ending_one_action(self):
    self.action.begin(1)
    self.assertEqual(1, self.action.watermark)
    self.callback.assert_called_once_with(1)

    self.callback.reset_mock()

    self.action.end(1)
    self.assertIsNone(self.action.watermark)
    self.callback.assert_called_once_with(None)

  def test_trying_to_interrupt_with_another_action(self):
    self.action.begin(1)
    self.assertEqual(1, self.action.watermark)
    self.callback.assert_called_once_with(1)

    self.callback.reset_mock()

    self.action.begin(2)
    self.assertEqual(1, self.action.watermark)
    self.assertEqual(0, self.callback.call_count)

  def test_trying_to_end_another_action(self):
    self.action.begin(1)
    self.callback.reset_mock()

    self.action.end(2)
    self.assertEqual(1, self.action.watermark)
    self.assertEqual(0, self.callback.call_count)


class ObservableFlagTest(absltest.TestCase):

  def test_update_each_added_listener(self):
    listener = mock.MagicMock(spec=object)

    subject = util.ObservableFlag(True)

    subject += listener
    listener.assert_called_once_with(True)

  def test_update_listeners_on_toggle(self):
    listeners = [mock.MagicMock(spec=object) for _ in range(10)]

    subject = util.ObservableFlag(True)
    subject += listeners

    for listener in listeners:
      listener.reset_mock()
    subject.toggle()
    for listener in listeners:
      listener.assert_called_once_with(False)


class TimerTest(absltest.TestCase):

  def setUp(self):
    super(TimerTest, self).setUp()
    self.timer = util.Timer()

  def test_time_elapsed(self):
    with mock.patch(util.__name__ + '.time') as time_mock:
      time_mock.time.return_value = 1
      self.timer.tick()
      time_mock.time.return_value = 2
      self.assertEqual(1, self.timer.tick())

  def test_time_measurement(self):
    with mock.patch(util.__name__ + '.time') as time_mock:
      time_mock.time.return_value = 1
      with self.timer.measure_time():
        time_mock.time.return_value = 4
      self.assertEqual(3, self.timer.measured_time)


class ErrorLoggerTest(absltest.TestCase):

  def setUp(self):
    super(ErrorLoggerTest, self).setUp()
    self.callback = mock.MagicMock()
    self.logger = util.ErrorLogger([self.callback])

  def test_no_errors_found_on_initialization(self):
    self.assertFalse(self.logger.errors_found)

  def test_no_error_caught(self):
    with self.logger:
      pass
    self.assertFalse(self.logger.errors_found)

  def test_error_caught(self):
    with self.logger:
      raise Exception('error message')
    self.assertTrue(self.logger.errors_found)

  def test_notifying_callbacks(self):
    error_message = 'error message'
    with self.logger:
      raise Exception(error_message)
    self.callback.assert_called_once_with(error_message)


class NullErrorLoggerTest(absltest.TestCase):

  def setUp(self):
    super(NullErrorLoggerTest, self).setUp()
    self.logger = util.NullErrorLogger()

  def test_thrown_errors_are_not_being_intercepted(self):
    with self.assertRaises(Exception):
      with self.logger:
        raise Exception()

  def test_errors_found_always_returns_false(self):
    self.assertFalse(self.logger.errors_found)


if __name__ == '__main__':
  absltest.main()
