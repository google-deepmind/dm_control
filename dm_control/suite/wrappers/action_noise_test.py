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

"""Tests for the action noise wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.rl import control
from dm_control.suite.wrappers import action_noise
import mock
import numpy as np
from dm_control.rl import specs


class ActionNoiseTest(parameterized.TestCase):

  def make_action_spec(self, lower=(-1.,), upper=(1.,)):
    lower, upper = np.broadcast_arrays(lower, upper)
    return specs.BoundedArraySpec(
        shape=lower.shape, dtype=float, minimum=lower, maximum=upper)

  def make_mock_env(self, action_spec=None):
    action_spec = action_spec or self.make_action_spec()
    env = mock.Mock(spec=control.Environment)
    env.action_spec.return_value = action_spec
    return env

  def assertStepCalledOnceWithCorrectAction(self, env, expected_action):
    # NB: `assert_called_once_with()` doesn't support numpy arrays.
    env.step.assert_called_once()
    actual_action = env.step.call_args_list[0][0][0]
    np.testing.assert_array_equal(expected_action, actual_action)

  @parameterized.parameters([
      dict(lower=np.r_[-1., 0.], upper=np.r_[1., 2.], scale=0.05),
      dict(lower=np.r_[-1., 0.], upper=np.r_[1., 2.], scale=0.),
      dict(lower=np.r_[-1., 0.], upper=np.r_[-1., 0.], scale=0.05),
  ])
  def test_step(self, lower, upper, scale):
    seed = 0
    std = scale * (upper - lower)
    expected_noise = np.random.RandomState(seed).normal(scale=std)
    action = np.random.RandomState(seed).uniform(lower, upper)
    expected_noisy_action = np.clip(action + expected_noise, lower, upper)
    task = mock.Mock(spec=control.Task)
    task.random = np.random.RandomState(seed)
    action_spec = self.make_action_spec(lower=lower, upper=upper)
    env = self.make_mock_env(action_spec=action_spec)
    env.task = task
    wrapped_env = action_noise.Wrapper(env, scale=scale)
    time_step = wrapped_env.step(action)
    self.assertStepCalledOnceWithCorrectAction(env, expected_noisy_action)
    self.assertIs(time_step, env.step(expected_noisy_action))

  @parameterized.named_parameters([
      dict(testcase_name='within_bounds', action=np.r_[-1.], noise=np.r_[0.1]),
      dict(testcase_name='below_lower', action=np.r_[-1.], noise=np.r_[-0.1]),
      dict(testcase_name='above_upper', action=np.r_[1.], noise=np.r_[0.1]),
  ])
  def test_action_clipping(self, action, noise):
    lower = -1.
    upper = 1.
    expected_noisy_action = np.clip(action + noise, lower, upper)
    task = mock.Mock(spec=control.Task)
    task.random = mock.Mock(spec=np.random.RandomState)
    task.random.normal.return_value = noise
    action_spec = self.make_action_spec(lower=lower, upper=upper)
    env = self.make_mock_env(action_spec=action_spec)
    env.task = task
    wrapped_env = action_noise.Wrapper(env)
    time_step = wrapped_env.step(action)
    self.assertStepCalledOnceWithCorrectAction(env, expected_noisy_action)
    self.assertIs(time_step, env.step(expected_noisy_action))

  @parameterized.parameters([
      dict(lower=np.r_[-1., 0.], upper=np.r_[1., np.inf]),
      dict(lower=np.r_[np.nan, 0.], upper=np.r_[1., 2.]),
  ])
  def test_error_if_action_bounds_non_finite(self, lower, upper):
    action_spec = self.make_action_spec(lower=lower, upper=upper)
    env = self.make_mock_env(action_spec=action_spec)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        action_noise._BOUNDS_MUST_BE_FINITE.format(action_spec=action_spec)):
      _ = action_noise.Wrapper(env)

  def test_reset(self):
    env = self.make_mock_env()
    wrapped_env = action_noise.Wrapper(env)
    time_step = wrapped_env.reset()
    env.reset.assert_called_once_with()
    self.assertIs(time_step, env.reset())

  def test_observation_spec(self):
    env = self.make_mock_env()
    wrapped_env = action_noise.Wrapper(env)
    observation_spec = wrapped_env.observation_spec()
    env.observation_spec.assert_called_once_with()
    self.assertIs(observation_spec, env.observation_spec())

  def test_action_spec(self):
    env = self.make_mock_env()
    wrapped_env = action_noise.Wrapper(env)
    # `env.action_spec()` is called in `Wrapper.__init__()`
    env.action_spec.reset_mock()
    action_spec = wrapped_env.action_spec()
    env.action_spec.assert_called_once_with()
    self.assertIs(action_spec, env.action_spec())

  @parameterized.parameters(['task', 'physics', 'control_timestep'])
  def test_getattr(self, attribute_name):
    env = self.make_mock_env()
    wrapped_env = action_noise.Wrapper(env)
    attr = getattr(wrapped_env, attribute_name)
    self.assertIs(attr, getattr(env, attribute_name))


if __name__ == '__main__':
  absltest.main()
