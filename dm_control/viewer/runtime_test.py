# Copyright 2018-2019 The dm_control Authors.
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
"""Runtime tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.viewer import runtime
import dm_env
from dm_env import specs
import mock
import numpy as np
import six
from six.moves import zip


class RuntimeStateMachineTest(parameterized.TestCase):

  def setUp(self):
    super(RuntimeStateMachineTest, self).setUp()
    env = mock.MagicMock()
    env.action_spec.return_value = specs.BoundedArray((1,), np.float64, -1, 1)
    self.runtime = runtime.Runtime(env, mock.MagicMock())
    self.runtime._start = mock.MagicMock()
    self.runtime.get_time = mock.MagicMock()
    self.runtime.get_time.return_value = 0
    self.runtime._step_simulation = mock.MagicMock(return_value=False)

  def test_initial_state(self):
    self.assertEqual(self.runtime._state, runtime.State.START)

  def test_successful_starting(self):
    self.runtime._start.return_value = True
    self.runtime._state = runtime.State.START
    self.runtime.tick(0, False)
    self.assertEqual(self.runtime._state, runtime.State.RUNNING)
    self.runtime._start.assert_called_once()
    self.runtime._step_simulation.assert_called_once()

  def test_failure_during_start(self):
    self.runtime._start.return_value = False
    self.runtime._state = runtime.State.START
    self.runtime.tick(0, False)
    self.assertEqual(self.runtime._state, runtime.State.STOPPED)
    self.runtime._start.assert_called_once()
    self.runtime._step_simulation.assert_not_called()

  def test_restarting(self):
    self.runtime._state = runtime.State.RUNNING
    self.runtime.restart()
    self.runtime.tick(0, False)
    self.assertEqual(self.runtime._state, runtime.State.RUNNING)
    self.runtime._start.assert_called_once()
    self.runtime._step_simulation.assert_called_once()

  def test_running(self):
    self.runtime._state = runtime.State.RUNNING
    self.runtime.tick(0, False)
    self.assertEqual(self.runtime._state, runtime.State.RUNNING)
    self.runtime._step_simulation.assert_called_once()

  def test_ending_a_running_episode(self):
    self.runtime._state = runtime.State.RUNNING
    self.runtime._step_simulation.return_value = True
    self.runtime.tick(0, False)
    self.assertEqual(self.runtime._state, runtime.State.STOPPED)
    self.runtime._step_simulation.assert_called_once()

  def test_calling_stop_has_immediate_effect_on_state(self):
    self.runtime.stop()
    self.assertEqual(self.runtime._state, runtime.State.STOPPED)

  @parameterized.parameters(runtime.State.RUNNING,
                            runtime.State.RESTARTING,)
  def test_states_affected_by_stop(self, state):
    self.runtime._state = state
    self.runtime.stop()
    self.assertEqual(self.runtime._state, runtime.State.STOPPED)

  def test_notifying_listeners_about_successful_start(self):
    callback = mock.MagicMock()
    self.runtime.on_episode_begin += [callback]
    self.runtime._start.return_value = True
    self.runtime._state = runtime.State.START
    self.runtime.tick(0, False)
    callback.assert_called_once()

  def test_listeners_not_notified_when_start_fails(self):
    callback = mock.MagicMock()
    self.runtime.on_episode_begin += [callback]
    self.runtime._start.return_value = False
    self.runtime._state = runtime.State.START
    self.runtime.tick(0, False)
    callback.assert_not_called()


class RuntimeSingleStepTest(parameterized.TestCase):

  def setUp(self):
    super(RuntimeSingleStepTest, self).setUp()
    env = mock.MagicMock(spec=dm_env.Environment)
    env.action_spec.return_value = specs.BoundedArray((1,), np.float64, -1, 1)
    self.runtime = runtime.Runtime(env, mock.MagicMock())
    self.runtime._step = mock.MagicMock()
    self.runtime._step.return_value = False

  def test_when_running(self):
    self.runtime._state = runtime.State.RUNNING
    self.runtime.single_step()
    self.assertEqual(self.runtime._state, runtime.State.RUNNING)
    self.runtime._step.assert_called_once()

  def test_ending_episode(self):
    self.runtime._state = runtime.State.RUNNING
    self.runtime._step.return_value = True
    self.runtime.single_step()
    self.assertEqual(self.runtime._state, runtime.State.STOP)
    self.runtime._step.assert_called_once()

  @parameterized.parameters(runtime.State.START,
                            runtime.State.STOP,
                            runtime.State.STOPPED,
                            runtime.State.RESTARTING)
  def test_runs_only_in_running_state(self, state):
    self.runtime._state = state
    self.runtime.single_step()
    self.assertEqual(self.runtime._state, state)
    self.assertEqual(0, self.runtime._step.call_count)


class RuntimeTest(absltest.TestCase):

  def setUp(self):
    super(RuntimeTest, self).setUp()
    env = mock.MagicMock(spec=dm_env.Environment)
    env.action_spec.return_value = specs.BoundedArray((1,), np.float64, -1, 1)
    self.runtime = runtime.Runtime(env, mock.MagicMock())
    self.runtime._step_paused = mock.MagicMock()
    self.runtime._step = mock.MagicMock(return_value=True)
    self.runtime.get_time = mock.MagicMock(return_value=0)

    self.time_step = 1e-2

  def set_loop(self, num_iterations, finish_after=0):
    finish_after = finish_after or num_iterations + 1
    self.delta_time = self.time_step / float(num_iterations)
    self.time = 0
    self.iteration = 0
    def fakeget_time():
      return self.time
    def fake_step():
      self.time += self.delta_time
      self.iteration += 1
      return self.iteration >= finish_after

    self.runtime._step = mock.MagicMock(side_effect=fake_step)
    self.runtime.get_time = mock.MagicMock(side_effect=fakeget_time)

  def test_num_step_calls(self):
    expected_call_count = 5
    self.set_loop(num_iterations=expected_call_count)
    finished = self.runtime._step_simulation(self.time_step, False)
    self.assertFalse(finished)
    self.assertEqual(expected_call_count, self.runtime._step.call_count)

  def test_finishing_if_episode_ends(self):
    num_iterations = 5
    finish_after = 2
    self.set_loop(num_iterations=num_iterations, finish_after=finish_after)
    finished = self.runtime._step_simulation(self.time_step, False)
    self.assertTrue(finished)
    self.assertEqual(finish_after, self.runtime._step.call_count)

  def test_stepping_paused(self):
    self.runtime._step_simulation(0, True)
    self.runtime._step_paused.assert_called_once()
    self.assertEqual(0, self.runtime._step.call_count)

  def test_physics_step_takes_less_time_than_tick(self):
    self.physics_time_step = runtime._DEFAULT_MAX_SIM_STEP * 0.5
    self.physics_time = 0.0
    def mock_get_time():
      return self.physics_time
    def mock_step():
      self.physics_time += self.physics_time_step
    self.runtime._step = mock.MagicMock(side_effect=mock_step)
    self.runtime.get_time = mock.MagicMock(side_effect=mock_get_time)
    self.runtime._step_simulation(
        time_elapsed=runtime._DEFAULT_MAX_SIM_STEP, paused=False)
    self.assertEqual(2, self.runtime._step.call_count)

  def test_physics_step_takes_more_time_than_tick(self):
    self.physics_time_step = runtime._DEFAULT_MAX_SIM_STEP * 2
    self.physics_time = 0.0
    def mock_get_time():
      return self.physics_time
    def mock_step():
      self.physics_time += self.physics_time_step
    self.runtime._step = mock.MagicMock(side_effect=mock_step)
    self.runtime.get_time = mock.MagicMock(side_effect=mock_get_time)

    # Simulates after the first frame
    self.runtime._step_simulation(
        time_elapsed=runtime._DEFAULT_MAX_SIM_STEP, paused=False)
    self.assertEqual(1, self.runtime._step.call_count)
    self.runtime._step.reset_mock()

    # Then pauses for one frame to let the internal timer catch up with the
    # simulation timer.
    self.runtime._step_simulation(
        time_elapsed=runtime._DEFAULT_MAX_SIM_STEP, paused=False)
    self.assertEqual(0, self.runtime._step.call_count)

    # Resumes simulation on the subsequent frame.
    self.runtime._step_simulation(
        time_elapsed=runtime._DEFAULT_MAX_SIM_STEP, paused=False)
    self.assertEqual(1, self.runtime._step.call_count)

  def test_updating_tracked_time_during_start(self):
    invalid_time = 20
    self.runtime.get_time = mock.MagicMock(return_value=invalid_time)

    valid_time = 2
    def mock_start():
      self.runtime.get_time = mock.MagicMock(return_value=valid_time)
      return True

    self.runtime._start = mock.MagicMock(side_effect=mock_start)
    self.runtime._step_simulation = mock.MagicMock()

    self.runtime.tick(time_elapsed=runtime._DEFAULT_MAX_SIM_STEP, paused=False)
    self.assertEqual(valid_time, self.runtime._tracked_simulation_time)

  def test_error_logger_forward_errors_to_listeners(self):
    callback = mock.MagicMock()
    self.runtime.on_error += [callback]
    with self.runtime._error_logger:
      raise Exception('error message')
    callback.assert_called_once()


class EnvironmentRuntimeTest(parameterized.TestCase):

  def setUp(self):
    super(EnvironmentRuntimeTest, self).setUp()
    self.observation = mock.MagicMock()
    self.env = mock.MagicMock(spec=dm_env.Environment)
    self.env.physics = mock.MagicMock()
    self.env.step = mock.MagicMock()
    self.env.action_spec.return_value = specs.BoundedArray(
        (1,), np.float64, -1, 1)
    self.policy = mock.MagicMock()
    self.actions = mock.MagicMock()
    self.runtime = runtime.Runtime(self.env, self.policy)

  def test_start(self):
    with mock.patch(runtime.__name__ + '.mjlib'):
      result = self.runtime._start()
      self.assertTrue(result)
      self.env.reset.assert_called_once()
      self.policy.assert_not_called()

  def test_step_with_policy(self):
    time_step = mock.Mock(spec=dm_env.TimeStep)
    self.runtime._time_step = time_step
    self.runtime._step()
    self.policy.assert_called_once_with(time_step)
    self.env.step.assert_called_once_with(self.policy.return_value)

  def test_step_without_policy(self):
    with mock.patch(
        runtime.__name__ + '._get_default_action') as mock_get_default_action:
      this_runtime = runtime.Runtime(environment=self.env, policy=None)
    this_runtime._step()
    self.env.step.assert_called_once_with(mock_get_default_action.return_value)

  def test_stepping_paused(self):
    with mock.patch(runtime.__name__ + '.mjlib') as mjlib:
      self.runtime._step_paused()
      mjlib.mj_forward.assert_called_once()

  def test_get_time(self):
    expected_time = 20
    self.env.physics = mock.MagicMock()
    self.env.physics.data = mock.MagicMock()
    self.env.physics.data.time = expected_time
    self.assertEqual(expected_time, self.runtime.get_time())

  def test_tracking_physics_instance_changes(self):
    callback = mock.MagicMock()
    self.runtime.on_physics_changed += [callback]

    def begin_episode_and_reload_physics():
      self.env.physics.data.ptr = mock.MagicMock()
    self.env.reset.side_effect = begin_episode_and_reload_physics

    self.runtime._start()
    callback.assert_called_once_with()

  def test_tracking_physics_instance_that_doesnt_change(self):
    callback = mock.MagicMock()
    self.runtime.on_physics_changed += [callback]

    self.runtime._start()
    callback.assert_not_called()

  def test_exception_thrown_during_start(self):
    def raise_exception(*unused_args, **unused_kwargs):
      raise Exception('test error message')
    self.runtime._env.reset.side_effect = raise_exception
    result = self.runtime._start()
    self.assertFalse(result)

  def test_exception_thrown_during_step(self):
    def raise_exception(*unused_args, **unused_kwargs):
      raise Exception('test error message')
    self.runtime._env.step.side_effect = raise_exception
    finished = self.runtime._step()
    self.assertTrue(finished)


class DefaultActionFromSpecTest(parameterized.TestCase):

  def assertNestedArraysEqual(self, expected, actual):
    """Asserts that two potentially nested structures of arrays are equal."""
    self.assertIs(type(actual), type(expected))
    if isinstance(expected, (list, tuple)):
      self.assertIsInstance(actual, (list, tuple))
      self.assertLen(actual, len(expected))
      for expected_item, actual_item in zip(expected, actual):
        self.assertNestedArraysEqual(expected_item, actual_item)
    elif isinstance(expected, collections.MutableMapping):
      keys_type = list if isinstance(expected, collections.OrderedDict) else set
      self.assertEqual(keys_type(actual.keys()), keys_type(expected.keys()))
      for key, expected_value in six.iteritems(expected):
        self.assertNestedArraysEqual(actual[key], expected_value)
    else:
      np.testing.assert_array_equal(expected, actual)

  _SHAPE = (2,)
  _DTYPE = np.float64
  _ACTION = np.zeros(_SHAPE)
  _ACTION_SPEC = specs.BoundedArray(_SHAPE, np.float64, -1, 1)

  @parameterized.named_parameters(
      ('single_array', _ACTION_SPEC, _ACTION),
      ('tuple', (_ACTION_SPEC, _ACTION_SPEC), (_ACTION, _ACTION)),
      ('list', [_ACTION_SPEC, _ACTION_SPEC], (_ACTION, _ACTION)),
      ('dict',
       {'a': _ACTION_SPEC, 'b': _ACTION_SPEC},
       {'a': _ACTION, 'b': _ACTION}),
      ('OrderedDict',
       collections.OrderedDict([('a', _ACTION_SPEC), ('b', _ACTION_SPEC)]),
       collections.OrderedDict([('a', _ACTION), ('b', _ACTION)])),
      )
  def test_action_structure(self, action_spec, expected_action):
    self.assertNestedArraysEqual(expected_action,
                                 runtime._get_default_action(action_spec))

  def test_ordered_dict_action_structure_with_bad_ordering(self):
    reversed_spec = collections.OrderedDict([('a', self._ACTION_SPEC),
                                             ('b', self._ACTION_SPEC)])
    expected_action = collections.OrderedDict([('b', self._ACTION),
                                               ('a', self._ACTION)])
    with six.assertRaisesRegex(self, AssertionError,
                               r"Lists differ: \['a', 'b'\] != \['b', 'a'\]"):
      self.assertNestedArraysEqual(expected_action,
                                   runtime._get_default_action(reversed_spec))

  @parameterized.named_parameters(
      ('closed',
       specs.BoundedArray(_SHAPE, _DTYPE, minimum=1., maximum=2.),
       np.full(_SHAPE, fill_value=1.5, dtype=_DTYPE)),
      ('left_open',
       specs.BoundedArray(_SHAPE, _DTYPE, minimum=-np.inf, maximum=2.),
       np.full(_SHAPE, fill_value=2., dtype=_DTYPE)),
      ('right_open',
       specs.BoundedArray(_SHAPE, _DTYPE, minimum=1., maximum=np.inf),
       np.full(_SHAPE, fill_value=1., dtype=_DTYPE)),
      ('unbounded',
       specs.BoundedArray(_SHAPE, _DTYPE, minimum=-np.inf, maximum=np.inf),
       np.full(_SHAPE, fill_value=0., dtype=_DTYPE)))
  def test_action_spec_interval(self, action_spec, expected_action):
    self.assertNestedArraysEqual(expected_action,
                                 runtime._get_default_action(action_spec))


if __name__ == '__main__':
  absltest.main()
