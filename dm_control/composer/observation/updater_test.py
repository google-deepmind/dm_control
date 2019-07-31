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

"""Tests for observation.observation_updater."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools
import math

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.composer.observation import fake_physics
from dm_control.composer.observation import observable
from dm_control.composer.observation import updater
from dm_env import specs
import numpy as np
import six
from six.moves import range


class DeterministicSequence(object):

  def __init__(self, sequence):
    self._iter = itertools.cycle(sequence)

  def __call__(self, random_state=None):
    del random_state  # unused
    return six.next(self._iter)


class UpdaterTest(parameterized.TestCase):

  @parameterized.parameters(list, tuple)
  def testNestedSpecsAndValues(self, list_or_tuple):
    observables = list_or_tuple((
        {'one': observable.Generic(lambda _: 1.),
         'two': observable.Generic(lambda _: [2, 2]),
        }, collections.OrderedDict([
            ('three', observable.Generic(lambda _: np.full((2, 2), 3))),
            ('four', observable.Generic(lambda _: [4.])),
            ('five', observable.Generic(lambda _: 5)),
        ])
    ))

    observables[0]['two'].enabled = True
    observables[1]['three'].enabled = True
    observables[1]['five'].enabled = True

    observation_updater = updater.Updater(observables)
    observation_updater.reset(physics=None, random_state=None)

    def make_spec(obs):
      array = np.array(obs.observation_callable(None, None)())
      return specs.Array((1,) + array.shape, array.dtype)
    expected_specs = list_or_tuple((
        {'two': make_spec(observables[0]['two'])},
        collections.OrderedDict([
            ('three', make_spec(observables[1]['three'])),
            ('five', make_spec(observables[1]['five']))
        ])
    ))

    actual_specs = observation_updater.observation_spec()
    self.assertIs(type(actual_specs), type(expected_specs))
    for actual_dict, expected_dict in zip(actual_specs, expected_specs):
      self.assertIs(type(actual_dict), type(expected_dict))
      self.assertEqual(actual_dict, expected_dict)

    expected_values = list_or_tuple((
        {'two': observables[0]['two'](physics=None, random_state=None)},
        collections.OrderedDict([
            ('three', observables[1]['three'](physics=None, random_state=None)),
            ('five', observables[1]['five'](physics=None, random_state=None))
        ])
    ))

    actual_values = observation_updater.get_observation()
    self.assertIs(type(actual_values), type(expected_values))
    for actual_dict, expected_dict in zip(actual_values, expected_values):
      self.assertIs(type(actual_dict), type(expected_dict))
      self.assertLen(actual_dict, len(expected_dict))
      for actual, expected in zip(six.iteritems(actual_dict),
                                  six.iteritems(expected_dict)):
        actual_name, actual_value = actual
        expected_name, expected_value = expected
        self.assertEqual(actual_name, expected_name)
        np.testing.assert_array_equal(actual_value[0], expected_value)

  def assertCorrectSpec(
      self, spec, expected_shape, expected_dtype, expected_name):
    self.assertEqual(spec.shape, expected_shape)
    self.assertEqual(spec.dtype, expected_dtype)
    self.assertEqual(spec.name, expected_name)

  def testObservationSpecInference(self):
    physics = fake_physics.FakePhysics()
    physics.observables['repeated'].buffer_size = 5
    physics.observables['matrix'].buffer_size = 4
    physics.observables['sqrt'] = observable.Generic(
        fake_physics.FakePhysics.sqrt, buffer_size=3)

    for obs in six.itervalues(physics.observables):
      obs.enabled = True

    observation_updater = updater.Updater(physics.observables)
    observation_updater.reset(physics=physics, random_state=None)

    spec = observation_updater.observation_spec()
    self.assertCorrectSpec(spec['repeated'], (5, 2), np.int, 'repeated')
    self.assertCorrectSpec(spec['matrix'], (4, 2, 3), np.int, 'matrix')
    self.assertCorrectSpec(spec['sqrt'], (3,), np.float, 'sqrt')

  def testObservation(self):
    physics = fake_physics.FakePhysics()
    physics.observables['repeated'].buffer_size = 5
    physics.observables['matrix'].delay = 1
    physics.observables['sqrt'] = observable.Generic(
        fake_physics.FakePhysics.sqrt, update_interval=7,
        buffer_size=3, delay=2)
    for obs in six.itervalues(physics.observables):
      obs.enabled = True
    with physics.reset_context():
      pass

    physics_steps_per_control_step = 5
    observation_updater = updater.Updater(
        physics.observables, physics_steps_per_control_step)
    observation_updater.reset(physics=physics, random_state=None)

    for control_step in range(0, 200):
      observation_updater.prepare_for_next_control_step()
      for _ in range(physics_steps_per_control_step):
        physics.step()
        observation_updater.update()

      step_counter = (control_step + 1) * physics_steps_per_control_step

      observation = observation_updater.get_observation()
      def assert_correct_buffer(obs_name, expected_callable,
                                observation=observation,
                                step_counter=step_counter):
        update_interval = (physics.observables[obs_name].update_interval
                           or updater.DEFAULT_UPDATE_INTERVAL)
        buffer_size = (physics.observables[obs_name].buffer_size
                       or updater.DEFAULT_BUFFER_SIZE)
        delay = (physics.observables[obs_name].delay
                 or updater.DEFAULT_DELAY)

        # The final item in the buffer is the current time, less the delay,
        # rounded _down_ to the nearest multiple of the update interval.
        end = update_interval * int(
            math.floor((step_counter - delay) / update_interval))

        # Figure out the first item in the buffer by working backwards from
        # the final item in multiples of the update interval.
        start = end - (buffer_size - 1) * update_interval

        # Clamp both the start and end step number below by zero.
        buffer_range = range(max(0, start), max(0, end + 1), update_interval)

        # Arrays with expected shapes, filled with expected default values.
        expected_value_spec = observation_updater.observation_spec()[obs_name]
        expected_values = np.zeros(shape=expected_value_spec.shape,
                                   dtype=expected_value_spec.dtype)

        # The arrays are filled from right to left, such that the most recent
        # entry is the rightmost one, and any padding is on the left.
        for index, timestamp in enumerate(reversed(buffer_range)):
          expected_values[-(index+1)] = expected_callable(timestamp)

        np.testing.assert_array_equal(observation[obs_name], expected_values)

      assert_correct_buffer('twice', lambda x: 2*x)
      assert_correct_buffer('matrix', lambda x: [[x]*3]*2)
      assert_correct_buffer('repeated', lambda x: [x, x])
      assert_correct_buffer('sqrt', np.sqrt)

  def testVariableRatesAndDelays(self):
    physics = fake_physics.FakePhysics()
    physics.observables['time'] = observable.Generic(
        lambda physics: physics.time(),
        buffer_size=3,
        # observations produced on step numbers 20*N + [0, 3, 5, 8, 11, 15, 16]
        update_interval=DeterministicSequence([3, 2, 3, 3, 4, 1, 4]),
        # observations arrive on step numbers 20*N + [3, 8, 7, 12, 11, 17, 20]
        delay=DeterministicSequence([3, 5, 2, 5, 1, 2, 4]))
    physics.observables['time'].enabled = True

    physics_steps_per_control_step = 10
    observation_updater = updater.Updater(
        physics.observables, physics_steps_per_control_step)
    observation_updater.reset(physics=physics, random_state=None)

    # Run through a few cycles of the variation sequences to make sure that
    # cross-control-boundary behaviour is correct.
    for i in range(5):
      observation_updater.prepare_for_next_control_step()
      for _ in range(physics_steps_per_control_step):
        physics.step()
        observation_updater.update()
      np.testing.assert_array_equal(
          observation_updater.get_observation()['time'],
          20*i + np.array([0, 5, 3]))

      observation_updater.prepare_for_next_control_step()
      for _ in range(physics_steps_per_control_step):
        physics.step()
        observation_updater.update()
      # Note that #11 is dropped since it arrives after #8,
      # whose large delay caused it to cross the control step boundary at #10.
      np.testing.assert_array_equal(
          observation_updater.get_observation()['time'],
          20*i + np.array([8, 15, 16]))

if __name__ == '__main__':
  absltest.main()
