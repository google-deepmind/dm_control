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

"""Tests for `dm_control.manipulation_suite`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import manipulation
import numpy as np
from six.moves import range


flags.DEFINE_boolean(
    'fix_seed', True,
    'Whether to fix the seed for the environment\'s random number generator. '
    'This the default since it prevents non-deterministic failures, but it may '
    'be useful to allow the seed to vary in some cases, for example when '
    'repeating a test many times in order to detect rare failure events.')

FLAGS = flags.FLAGS

_FIX_SEED = None
_NUM_EPISODES = 5
_NUM_STEPS_PER_EPISODE = 10


def _get_fix_seed():
  global _FIX_SEED
  if _FIX_SEED is None:
    if FLAGS.is_parsed():
      _FIX_SEED = FLAGS.fix_seed
    else:
      _FIX_SEED = FLAGS['fix_seed'].default
  return _FIX_SEED


class ManipulationTest(parameterized.TestCase):
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

  @parameterized.parameters(*manipulation.ALL)
  def test_task_runs(self, task_name):
    """Tests that the environment runs and is coherent with its specs."""
    seed = 99 if _get_fix_seed() else None
    env = manipulation.load(task_name, seed=seed)
    random_state = np.random.RandomState(seed)

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
