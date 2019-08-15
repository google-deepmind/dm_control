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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
import dm_env
import mock
from six.moves import range


class DummyTask(composer.NullTask):

  def __init__(self):
    null_entity = composer.ModelWrapperEntity(mjcf.RootElement())
    super(DummyTask, self).__init__(null_entity)


class DummyTaskWithResetFailures(DummyTask):

  def __init__(self, num_reset_failures):
    super(DummyTaskWithResetFailures, self).__init__()
    self.num_reset_failures = num_reset_failures
    self.reset_counter = 0

  def initialize_episode_mjcf(self, random_state):
    self.reset_counter += 1

  def initialize_episode(self, physics, random_state):
    if self.reset_counter <= self.num_reset_failures:
      raise composer.EpisodeInitializationError()


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


if __name__ == '__main__':
  absltest.main()
