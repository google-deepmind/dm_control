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

"""Tests for dm_control.locomotion.soccer.pitch."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.locomotion.soccer import pitch as pitch_lib
from dm_control.locomotion.soccer import team as team_lib
import numpy as np


class PitchTest(parameterized.TestCase):

  def _pitch_with_ball(self, pitch_size, ball_pos):
    pitch = pitch_lib.Pitch(size=pitch_size)
    self.assertEqual(pitch.size, pitch_size)

    sphere = props.Primitive(geom_type='sphere', size=(0.1,), pos=ball_pos)
    pitch.register_ball(sphere)
    pitch.attach(sphere)

    env = composer.Environment(
        composer.NullTask(pitch), random_state=np.random.RandomState(42))
    env.reset()
    return pitch

  def test_pitch_none_detected(self):
    pitch = self._pitch_with_ball((12, 9), (0, 0, 0))
    self.assertEmpty(pitch.detected_off_court())
    self.assertIsNone(pitch.detected_goal())

  def test_pitch_detected_off_court(self):
    pitch = self._pitch_with_ball((12, 9), (20, 0, 0))
    self.assertLen(pitch.detected_off_court(), 1)
    self.assertIsNone(pitch.detected_goal())

  def test_pitch_detected_away_goal(self):
    pitch = self._pitch_with_ball((12, 9), (-9.5, 0, 1))
    self.assertLen(pitch.detected_off_court(), 1)
    self.assertEqual(team_lib.Team.AWAY, pitch.detected_goal())

  def test_pitch_detected_home_goal(self):
    pitch = self._pitch_with_ball((12, 9), (9.5, 0, 1))
    self.assertLen(pitch.detected_off_court(), 1)
    self.assertEqual(team_lib.Team.HOME, pitch.detected_goal())

  @parameterized.parameters((True, distributions.Uniform()),
                            (False, distributions.Uniform()))
  def test_randomize_pitch(self, keep_aspect_ratio, randomizer):
    pitch = pitch_lib.RandomizedPitch(
        min_size=(4, 3),
        max_size=(8, 6),
        randomizer=randomizer,
        keep_aspect_ratio=keep_aspect_ratio)
    pitch.initialize_episode_mjcf(np.random.RandomState(42))

    self.assertBetween(pitch.size[0], 4, 8)
    self.assertBetween(pitch.size[1], 3, 6)

    if keep_aspect_ratio:
      self.assertAlmostEqual((pitch.size[0] - 4) / (8. - 4.),
                             (pitch.size[1] - 3) / (6. - 3.))


if __name__ == '__main__':
  absltest.main()
