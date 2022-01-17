# Copyright 2017 The dm_control Authors.
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

"""Control Environment tests."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.rl import control
from dm_env import specs
import mock
import numpy as np

_CONSTANT_REWARD_VALUE = 1.0
_CONSTANT_OBSERVATION = {'observations': np.asarray(_CONSTANT_REWARD_VALUE)}

_ACTION_SPEC = specs.BoundedArray(
    shape=(1,), dtype=float, minimum=0.0, maximum=1.0)
_OBSERVATION_SPEC = {'observations': specs.Array(shape=(), dtype=float)}


class EnvironmentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._task = mock.Mock(spec=control.Task)
    self._task.initialize_episode = mock.Mock()
    self._task.get_observation = mock.Mock(return_value=_CONSTANT_OBSERVATION)
    self._task.get_reward = mock.Mock(return_value=_CONSTANT_REWARD_VALUE)
    self._task.get_termination = mock.Mock(return_value=None)
    self._task.action_spec = mock.Mock(return_value=_ACTION_SPEC)
    self._task.observation_spec.side_effect = NotImplementedError()

    self._physics = mock.Mock(spec=control.Physics)
    self._physics.time = mock.Mock(return_value=0.0)

    self._physics.reset_context = mock.MagicMock()

    self._env = control.Environment(physics=self._physics, task=self._task)

  def test_environment_calls(self):
    self._env.action_spec()
    self._task.action_spec.assert_called_with(self._physics)

    self._env.reset()
    self._task.initialize_episode.assert_called_with(self._physics)
    self._task.get_observation.assert_called_with(self._physics)

    action = [1]
    time_step = self._env.step(action)

    self._task.before_step.assert_called()
    self._task.after_step.assert_called_with(self._physics)
    self._task.get_termination.assert_called_with(self._physics)

    self.assertEqual(_CONSTANT_REWARD_VALUE, time_step.reward)

  @parameterized.parameters(
      {'physics_timestep': .01, 'control_timestep': None,
       'expected_steps': 1000},
      {'physics_timestep': .01, 'control_timestep': .05,
       'expected_steps': 5000})
  def test_timeout(self, expected_steps, physics_timestep, control_timestep):
    self._physics.timestep.return_value = physics_timestep
    time_limit = expected_steps * (control_timestep or physics_timestep)
    env = control.Environment(
        physics=self._physics, task=self._task, time_limit=time_limit,
        control_timestep=control_timestep)

    time_step = env.reset()
    steps = 0
    while not time_step.last():
      time_step = env.step([1])
      steps += 1

    self.assertEqual(steps, expected_steps)
    self.assertTrue(time_step.last())

    time_step = env.step([1])
    self.assertTrue(time_step.first())

  def test_observation_spec(self):
    observation_spec = self._env.observation_spec()
    self.assertEqual(_OBSERVATION_SPEC, observation_spec)

  def test_redundant_args_error(self):
    with self.assertRaises(ValueError):
      control.Environment(physics=self._physics, task=self._task,
                          n_sub_steps=2, control_timestep=0.1)

  def test_control_timestep(self):
    self._physics.timestep.return_value = .002
    env = control.Environment(
        physics=self._physics, task=self._task, n_sub_steps=5)
    self.assertEqual(.01, env.control_timestep())

  def test_flatten_observations(self):
    multimodal_obs = dict(_CONSTANT_OBSERVATION)
    multimodal_obs['sensor'] = np.zeros(7, dtype=bool)
    self._task.get_observation = mock.Mock(return_value=multimodal_obs)
    env = control.Environment(
        physics=self._physics, task=self._task, flat_observation=True)
    timestep = env.reset()
    self.assertLen(timestep.observation, 1)
    self.assertEqual(timestep.observation[control.FLAT_OBSERVATION_KEY].size,
                     1 + 7)


class ComputeNStepsTest(parameterized.TestCase):

  @parameterized.parameters((0.2, 0.1, 2), (.111, .111, 1), (100, 5, 20),
                            (0.03, 0.005, 6))
  def testComputeNSteps(self, control_timestep, physics_timestep, expected):
    steps = control.compute_n_steps(control_timestep, physics_timestep)
    self.assertEqual(expected, steps)

  @parameterized.parameters((3, 2), (.003, .00101))
  def testComputeNStepsFailures(self, control_timestep, physics_timestep):
    with self.assertRaises(ValueError):
      control.compute_n_steps(control_timestep, physics_timestep)

if __name__ == '__main__':
  absltest.main()
