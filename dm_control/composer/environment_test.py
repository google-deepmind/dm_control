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
from dm_control import composer
from dm_control import mjcf
from six.moves import range


class TaskWithResetFailures(composer.NullTask):

  def __init__(self, num_reset_failures):
    self.num_reset_failures = num_reset_failures
    self.reset_counter = 0
    null_entity = composer.ModelWrapperEntity(mjcf.RootElement())
    super(TaskWithResetFailures, self).__init__(null_entity)

  def initialize_episode_mjcf(self, random_state):
    self.reset_counter += 1

  def initialize_episode(self, physics, random_state):
    if self.reset_counter <= self.num_reset_failures:
      raise composer.EpisodeInitializationError()


class EnvironmentTest(absltest.TestCase):

  def test_failed_resets(self):
    total_reset_failures = 5
    env_reset_attempts = 2
    task = TaskWithResetFailures(num_reset_failures=total_reset_failures)
    env = composer.Environment(task, max_reset_attempts=env_reset_attempts)
    for _ in range(total_reset_failures // env_reset_attempts):
      with self.assertRaises(composer.EpisodeInitializationError):
        env.reset()
    env.reset()  # should not raise an exception
    self.assertEqual(task.reset_counter, total_reset_failures + 1)


if __name__ == '__main__':
  absltest.main()
