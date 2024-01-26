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

"""Tests for dm_control.composer.environment."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
import dm_env
import mock
import numpy as np


class DummyTask(composer.NullTask):

  def __init__(self):
    null_entity = composer.ModelWrapperEntity(mjcf.RootElement())
    super().__init__(null_entity)

  @property
  def task_observables(self):
    time = observable.Generic(lambda physics: physics.time())
    time.enabled = True
    return {'time': time}


class DummyTaskWithResetFailures(DummyTask):

  def __init__(self, num_reset_failures):
    super().__init__()
    self.num_reset_failures = num_reset_failures
    self.reset_counter = 0

  def initialize_episode_mjcf(self, random_state):
    self.reset_counter += 1

  def initialize_episode(self, physics, random_state):
    if self.reset_counter <= self.num_reset_failures:
      raise composer.EpisodeInitializationError()


class DummyTaskWithRandomObservation(composer.NullTask):

  def __init__(self):
    null_entity = composer.ModelWrapperEntity(mjcf.RootElement())
    super().__init__(null_entity)

    self._observation = [0.0] * 1000

  def initialize_episode(self, physics, random_state):
    del physics
    self._observation = random_state.randint(1000, size=1000)

  @property
  def task_observables(self):
    random_int = observable.Generic(lambda physics: self._observation)
    random_int.enabled = True
    return {'random_int': random_int}


class EnvironmentTest(parameterized.TestCase):

  def test_failed_resets(self):
    total_reset_failures = 5
    env_reset_attempts = 2
    task = DummyTaskWithResetFailures(num_reset_failures=total_reset_failures)
    env = composer.Environment(task, max_reset_attempts=env_reset_attempts)
    for _ in range(total_reset_failures // env_reset_attempts):
      with self.assertRaises(composer.EpisodeInitializationError):
        env.reset()
    env.reset()  # should not raise an exception
    self.assertEqual(task.reset_counter, total_reset_failures + 1)

  @parameterized.parameters(
      dict(name='reward_spec', defined_in_task=True),
      dict(name='reward_spec', defined_in_task=False),
      dict(name='discount_spec', defined_in_task=True),
      dict(name='discount_spec', defined_in_task=False))
  def test_get_spec(self, name, defined_in_task):
    task = DummyTask()
    env = composer.Environment(task)
    with mock.patch.object(task, 'get_' + name) as mock_task_get_spec:
      if defined_in_task:
        expected_spec = mock.Mock()
        mock_task_get_spec.return_value = expected_spec
      else:
        expected_spec = getattr(dm_env.Environment, name)(env)
        mock_task_get_spec.return_value = None
      spec = getattr(env, name)()
    mock_task_get_spec.assert_called_once_with()
    self.assertSameStructure(spec, expected_spec)

  def test_can_provide_observation(self):
    task = DummyTask()
    env = composer.Environment(task)
    obs = env.reset().observation
    self.assertLen(obs, 1)
    np.testing.assert_array_equal(obs['time'], env.physics.time())
    for _ in range(20):
      obs = env.step([]).observation
      self.assertLen(obs, 1)
      np.testing.assert_array_equal(obs['time'], env.physics.time())

  def test_dont_compile_mjcf_between_episodes(self):
    class AfterCompileHook(object):

      def __init__(self):
        self.after_compile_call_count = 0

      def __call__(self, physics, random_state):
        del physics, random_state
        self.after_compile_call_count += 1

    after_compile_hook = AfterCompileHook()
    task = DummyTask()
    env = composer.Environment(task, recompile_mjcf_every_episode=False)
    env.add_extra_hook('after_compile', after_compile_hook)
    env.reset()
    self.assertEqual(after_compile_hook.after_compile_call_count, 1)
    for _ in range(4):
      env.reset()
      env.step([])

    # Check the hook is not called.
    self.assertEqual(after_compile_hook.after_compile_call_count, 1)

  def test_fixed_initial_state(self):
    task = DummyTaskWithRandomObservation()
    fixed_env = composer.Environment(task, fixed_initial_state=True)
    non_fixed_env = composer.Environment(task, fixed_initial_state=False)
    fixed_obs = fixed_env.reset().observation['random_int']
    non_fixed_obs = non_fixed_env.reset().observation['random_int']
    for _ in range(3):
      np.testing.assert_array_equal(
          fixed_env.reset().observation['random_int'], fixed_obs
      )
      self.assertTrue(
          np.any(
              np.not_equal(
                  np.asarray(non_fixed_obs),
                  np.asarray(non_fixed_env.reset().observation['random_int']),
              )
          )
      )


if __name__ == '__main__':
  absltest.main()
