# Copyright 2019 The dm_control Authors.
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

"""Tests for the action scale wrapper."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.rl import control
from dm_control.suite.wrappers import action_scale
from dm_env import specs
import mock
import numpy as np


def make_action_spec(lower=(-1.,), upper=(1.,)):
  lower, upper = np.broadcast_arrays(lower, upper)
  return specs.BoundedArray(
      shape=lower.shape, dtype=float, minimum=lower, maximum=upper)


def make_mock_env(action_spec):
  env = mock.Mock(spec=control.Environment)
  env.action_spec.return_value = action_spec
  return env


class ActionScaleTest(parameterized.TestCase):

  def assertStepCalledOnceWithCorrectAction(self, env, expected_action):
    # NB: `assert_called_once_with()` doesn't support numpy arrays.
    env.step.assert_called_once()
    actual_action = env.step.call_args_list[0][0][0]
    np.testing.assert_array_equal(expected_action, actual_action)

  @parameterized.parameters(
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': np.r_[-2., -2.],
          'scaled_maximum': np.r_[2., 2.],
      },
      {
          'minimum': np.r_[-2., -2.],
          'maximum': np.r_[2., 2.],
          'scaled_minimum': np.r_[-1., -1.],
          'scaled_maximum': np.r_[1., 1.],
      },
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': np.r_[-2., -2.],
          'scaled_maximum': np.r_[1., 1.],
      },
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': np.r_[-1., -1.],
          'scaled_maximum': np.r_[2., 2.],
      },
  )
  def test_step(self, minimum, maximum, scaled_minimum, scaled_maximum):
    action_spec = make_action_spec(lower=minimum, upper=maximum)
    env = make_mock_env(action_spec=action_spec)
    wrapped_env = action_scale.Wrapper(
        env, minimum=scaled_minimum, maximum=scaled_maximum)

    time_step = wrapped_env.step(scaled_minimum)
    self.assertStepCalledOnceWithCorrectAction(env, minimum)
    self.assertIs(time_step, env.step(minimum))

    env.reset_mock()

    time_step = wrapped_env.step(scaled_maximum)
    self.assertStepCalledOnceWithCorrectAction(env, maximum)
    self.assertIs(time_step, env.step(maximum))

  @parameterized.parameters(
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
      },
      {
          'minimum': np.r_[0, 1],
          'maximum': np.r_[2, 3],
      },
  )
  def test_correct_action_spec(self, minimum, maximum):
    original_action_spec = make_action_spec(
        lower=np.r_[-2., -2.], upper=np.r_[2., 2.])
    env = make_mock_env(action_spec=original_action_spec)
    wrapped_env = action_scale.Wrapper(env, minimum=minimum, maximum=maximum)
    new_action_spec = wrapped_env.action_spec()
    np.testing.assert_array_equal(new_action_spec.minimum, minimum)
    np.testing.assert_array_equal(new_action_spec.maximum, maximum)

  @parameterized.parameters('reset', 'observation_spec', 'control_timestep')
  def test_method_delegated_to_underlying_env(self, method_name):
    env = make_mock_env(action_spec=make_action_spec())
    wrapped_env = action_scale.Wrapper(env, minimum=0, maximum=1)
    env_method = getattr(env, method_name)
    wrapper_method = getattr(wrapped_env, method_name)
    out = wrapper_method()
    env_method.assert_called_once_with()
    self.assertIs(out, env_method())

  def test_invalid_action_spec_type(self):
    action_spec = [make_action_spec()] * 2
    env = make_mock_env(action_spec=action_spec)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        action_scale._ACTION_SPEC_MUST_BE_BOUNDED_ARRAY.format(action_spec)):
      action_scale.Wrapper(env, minimum=0, maximum=1)

  @parameterized.parameters(
      {'name': 'minimum', 'bounds': np.r_[np.nan]},
      {'name': 'minimum', 'bounds': np.r_[-np.inf]},
      {'name': 'maximum', 'bounds': np.r_[np.inf]},
  )
  def test_non_finite_bounds(self, name, bounds):
    kwargs = {'minimum': np.r_[-1.], 'maximum': np.r_[1.]}
    kwargs[name] = bounds
    env = make_mock_env(action_spec=make_action_spec())
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        action_scale._MUST_BE_FINITE.format(name=name, bounds=bounds)):
      action_scale.Wrapper(env, **kwargs)

  @parameterized.parameters(
      {'name': 'minimum', 'bounds': np.r_[1., 2., 3.]},
      {'name': 'minimum', 'bounds': np.r_[[1.], [2.], [3.]]},
  )
  def test_invalid_bounds_shape(self, name, bounds):
    shape = (2,)
    kwargs = {'minimum': np.zeros(shape), 'maximum': np.ones(shape)}
    kwargs[name] = bounds
    action_spec = make_action_spec(lower=[-1, -1], upper=[2, 3])
    env = make_mock_env(action_spec=action_spec)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        action_scale._MUST_BROADCAST.format(
            name=name, bounds=bounds, shape=shape)):
      action_scale.Wrapper(env, **kwargs)

if __name__ == '__main__':
  absltest.main()
