"""Tests for the action scale wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.rl import control
from dm_control.suite.wrappers import action_scale
from dm_env import specs
import mock
import numpy as np


class ActionScaleTest(parameterized.TestCase):

  def make_action_spec(self, lower=(-1.,), upper=(1.,)):
    lower, upper = np.broadcast_arrays(lower, upper)
    return specs.BoundedArray(
        shape=lower.shape, dtype=float, minimum=lower, maximum=upper)

  def make_mock_env(self, action_spec=None):
    action_spec = action_spec or self.make_action_spec()
    env = mock.Mock(spec=control.Environment)
    env.action_spec.return_value = action_spec
    env.step.side_effect = lambda action: action
    return env

  def assertStepCalledOnceWithCorrectAction(self, env, expected_action):
    # NB: `assert_called_once_with()` doesn't support numpy arrays.
    env.step.assert_called_once()
    actual_action = env.step.call_args_list[0][0][0]
    np.testing.assert_array_equal(expected_action, actual_action)

  @parameterized.parameters([
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
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': None,
          'scaled_maximum': np.r_[2., 2.],
      },
      {
          'minimum': np.r_[-1., -1.],
          'maximum': np.r_[1., 1.],
          'scaled_minimum': None,
          'scaled_maximum': None,
      },
  ])

  def test_step(self, minimum, maximum, scaled_minimum, scaled_maximum, seed=0):
    task = mock.Mock(spec=control.Task)
    task.random = np.random.RandomState(seed)
    action_spec = self.make_action_spec(lower=minimum, upper=maximum)
    env = self.make_mock_env(action_spec=action_spec)
    env.task = task

    if scaled_minimum is None:
      scaled_minimum = minimum

    if scaled_maximum is None:
      scaled_maximum = maximum

    wrapped_env = action_scale.Wrapper(
      env, minimum=scaled_minimum, maximum=scaled_maximum)

    time_step = wrapped_env.step(scaled_minimum)
    self.assertStepCalledOnceWithCorrectAction(env, minimum)
    np.testing.assert_array_equal(time_step, env.step(minimum))

    env.reset_mock()

    time_step = wrapped_env.step(scaled_maximum)
    self.assertStepCalledOnceWithCorrectAction(env, maximum)
    np.testing.assert_array_equal(time_step, env.step(maximum))


if __name__ == '__main__':
  absltest.main()
