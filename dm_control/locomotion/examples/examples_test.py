# Copyright 2020 The dm_control Authors.
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

"""Tests for `dm_control.locomotion.examples`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.locomotion.examples import basic_cmu_2019
from dm_control.locomotion.examples import basic_rodent_2020

import numpy as np
from six.moves import range


_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10


class ExampleEnvironmentsTest(parameterized.TestCase):
  """Tests run on all the tasks registered."""

  def _validate_observation(self, observation, observation_spec):
    self.assertEqual(list(observation.keys()), list(observation_spec.keys()))
    for name, array_spec in observation_spec.items():
      array_spec.validate(observation[name])

  def _validate_reward_range(self, reward):
    self.assertIsInstance(reward, float)
    self.assertBetween(reward, 0, 1)

  def _validate_discount(self, discount):
    self.assertIsInstance(discount, float)
    self.assertBetween(discount, 0, 1)

  @parameterized.named_parameters(
      ('cmu_humanoid_run_walls', basic_cmu_2019.cmu_humanoid_run_walls),
      ('cmu_humanoid_run_gaps', basic_cmu_2019.cmu_humanoid_run_gaps),
      ('cmu_humanoid_go_to_target', basic_cmu_2019.cmu_humanoid_go_to_target),
      ('cmu_humanoid_maze_forage', basic_cmu_2019.cmu_humanoid_maze_forage),
      ('cmu_humanoid_heterogeneous_forage',
       basic_cmu_2019.cmu_humanoid_heterogeneous_forage),
      ('rodent_escape_bowl', basic_rodent_2020.rodent_escape_bowl),
      ('rodent_run_gaps', basic_rodent_2020.rodent_run_gaps),
      ('rodent_maze_forage', basic_rodent_2020.rodent_maze_forage),
      ('rodent_two_touch', basic_rodent_2020.rodent_two_touch),
  )
  def test_env_runs(self, env_constructor):
    """Tests that the environment runs and is coherent with its specs."""
    random_state = np.random.RandomState(99)

    env = env_constructor(random_state=random_state)
    observation_spec = env.observation_spec()
    action_spec = env.action_spec()
    self.assertTrue(np.all(np.isfinite(action_spec.minimum)))
    self.assertTrue(np.all(np.isfinite(action_spec.maximum)))

    # Run a partial episode, check observations, rewards, discount.
    for _ in range(_NUM_EPISODES):
      time_step = env.reset()
      for _ in range(_NUM_STEPS_PER_EPISODE):
        self._validate_observation(time_step.observation, observation_spec)
        if time_step.first():
          self.assertIsNone(time_step.reward)
          self.assertIsNone(time_step.discount)
        else:
          self._validate_reward_range(time_step.reward)
          self._validate_discount(time_step.discount)
        action = random_state.uniform(action_spec.minimum, action_spec.maximum)
        env.step(action)

if __name__ == '__main__':
  absltest.main()
