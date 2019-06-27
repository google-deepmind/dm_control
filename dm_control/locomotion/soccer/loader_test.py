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

  @parameterized.named_parameters(
      ("2vs2_nocontacts", 2, True), ("2vs2_contacts", 2, False),
      ("1vs1_nocontacts", 1, True), ("1vs1_contacts", 1, False))
  def test_load_env(self, team_size, disable_walker_contacts):
    env = soccer.load(team_size=team_size, time_limit=2.,
                      disable_walker_contacts=disable_walker_contacts)
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

  def assertSameObservation(self, expected_observation, actual_observation):
    self.assertLen(actual_observation, len(expected_observation))
    for player_id in range(len(expected_observation)):
      expected_player_observations = expected_observation[player_id]
      actual_player_observations = actual_observation[player_id]
      expected_keys = expected_player_observations.keys()
      actual_keys = actual_player_observations.keys()
      msg = ("Observation keys differ for player {}.\nExpected: {}.\nActual: {}"
             .format(player_id, expected_keys, actual_keys))
      self.assertEqual(expected_keys, actual_keys, msg)
      for key in expected_player_observations:
        expected_array = expected_player_observations[key]
        actual_array = actual_player_observations[key]
        msg = ("Observation {!r} differs for player {}.\nExpected:\n{}\n"
               "Actual:\n{}"
               .format(key, player_id, expected_array, actual_array))
        np.testing.assert_array_equal(expected_array, actual_array,
                                      err_msg=msg)

  @parameterized.parameters(True, False)
  def test_same_first_observation_if_same_seed(self, disable_walker_contacts):
    seed = 42
    timestep_1 = soccer.load(
        team_size=2,
        random_state=seed,
        disable_walker_contacts=disable_walker_contacts).reset()
    timestep_2 = soccer.load(
        team_size=2,
        random_state=seed,
        disable_walker_contacts=disable_walker_contacts).reset()
    self.assertSameObservation(timestep_1.observation, timestep_2.observation)

  @parameterized.parameters(True, False)
  def test_different_first_observation_if_different_seed(
      self, disable_walker_contacts):
    timestep_1 = soccer.load(
        team_size=2,
        random_state=1,
        disable_walker_contacts=disable_walker_contacts).reset()
    timestep_2 = soccer.load(
        team_size=2,
        random_state=2,
        disable_walker_contacts=disable_walker_contacts).reset()
    try:
      self.assertSameObservation(timestep_1.observation, timestep_2.observation)
    except AssertionError:
      pass
    else:
      self.fail("Observations are unexpectedly identical.")


if __name__ == "__main__":
  absltest.main()
