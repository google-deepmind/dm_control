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

"""Tests for dm_control.locomotion.soccer.soccer_ball."""


from absl.testing import absltest
from dm_control import composer
from dm_control import mjcf
from dm_control.entities import props
from dm_control.locomotion.soccer import soccer_ball
from dm_control.locomotion.soccer import team
import numpy as np


class SoccerBallTest(absltest.TestCase):

  def test_detect_hit(self):
    arena = composer.Arena()
    ball = soccer_ball.SoccerBall(radius=0.35, mass=0.045, name='test_ball')
    player = team.Player(
        team=team.Team.HOME,
        walker=props.Primitive(geom_type='sphere', size=(0.1,), name='home'))
    arena.add_free_entity(player.walker)
    ball.register_player(player)
    arena.add_free_entity(ball)

    random_state = np.random.RandomState(42)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    physics.step()

    ball.initialize_episode(physics, random_state)
    ball.before_step(physics, random_state)
    self.assertEqual(ball.hit, False)
    self.assertEqual(ball.repossessed, False)
    self.assertEqual(ball.intercepted, False)
    self.assertIsNone(ball.last_hit)
    self.assertIsNone(ball.dist_between_last_hits)

    ball.after_substep(physics, random_state)
    ball.after_step(physics, random_state)

    self.assertEqual(ball.hit, True)
    self.assertEqual(ball.repossessed, True)
    self.assertEqual(ball.intercepted, True)
    self.assertEqual(ball.last_hit, player)
    # Only one hit registered.
    self.assertIsNone(ball.dist_between_last_hits)

  def test_has_tracking_cameras(self):
    ball = soccer_ball.SoccerBall(radius=0.35, mass=0.045, name='test_ball')
    expected_camera_names = ['ball_cam_near', 'ball_cam', 'ball_cam_far']
    camera_names = [cam.name for cam in ball.mjcf_model.find_all('camera')]
    self.assertCountEqual(expected_camera_names, camera_names)


if __name__ == '__main__':
  absltest.main()
