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

"""Tests for dm_control.locomotion.soccer.load."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.locomotion import soccer
import numpy as np
from six.moves import range


class LoadTest(parameterized.TestCase):

  @parameterized.parameters(1, 2)
  def test_load_env(self, team_size):
    env = soccer.load(team_size=team_size, time_limit=2.)
    action_specs = env.action_spec()

    random_state = np.random.RandomState(0)
    time_step = env.reset()
    while not time_step.last():
      actions = []
      for action_spec in action_specs:
        action = random_state.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
      time_step = env.step(actions)

      for i in range(len(action_specs)):
        logging.info(
            "Player %d: reward = %s, discount = %s, observations = %s.", i,
            time_step.reward[i], time_step.discount, time_step.observation[i])


if __name__ == "__main__":
  absltest.main()
