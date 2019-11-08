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
"""Tests for locomotion.tasks.random_goal_maze."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl.testing import absltest

from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import cmu_humanoid

import numpy as np
from six.moves import range


class RandomGoalMazeTest(absltest.TestCase):

  def test_observables(self):
    walker = cmu_humanoid.CMUHumanoid()

    # Build a maze with rooms and targets.
    skybox_texture = labmaze_textures.SkyBox(style='sky_03')
    wall_textures = labmaze_textures.WallTextures(style='style_01')
    floor_textures = labmaze_textures.FloorTextures(style='style_01')
    arena = mazes.RandomMazeWithTargets(
        x_cells=11,
        y_cells=11,
        xy_scale=3,
        max_rooms=4,
        room_min_size=4,
        room_max_size=5,
        spawns_per_room=1,
        targets_per_room=3,
        skybox_texture=skybox_texture,
        wall_textures=wall_textures,
        floor_textures=floor_textures,
    )

    task = random_goal_maze.ManyGoalsMaze(
        walker=walker,
        maze_arena=arena,
        target_builder=functools.partial(
            target_sphere.TargetSphere,
            radius=0.4,
            rgb1=(0, 0, 0.4),
            rgb2=(0, 0, 0.7)),
        control_timestep=.03,
        physics_timestep=.005,
    )
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    timestep = env.reset()

    self.assertIn('walker/joints_pos', timestep.observation)

  def test_termination_and_discount(self):
    walker = cmu_humanoid.CMUHumanoid()

    # Build a maze with rooms and targets.
    skybox_texture = labmaze_textures.SkyBox(style='sky_03')
    wall_textures = labmaze_textures.WallTextures(style='style_01')
    floor_textures = labmaze_textures.FloorTextures(style='style_01')
    arena = mazes.RandomMazeWithTargets(
        x_cells=11,
        y_cells=11,
        xy_scale=3,
        max_rooms=4,
        room_min_size=4,
        room_max_size=5,
        spawns_per_room=1,
        targets_per_room=3,
        skybox_texture=skybox_texture,
        wall_textures=wall_textures,
        floor_textures=floor_textures,
    )

    task = random_goal_maze.ManyGoalsMaze(
        walker=walker,
        maze_arena=arena,
        target_builder=functools.partial(
            target_sphere.TargetSphere,
            radius=0.4,
            rgb1=(0, 0, 0.4),
            rgb2=(0, 0, 0.7)),
        control_timestep=.03,
        physics_timestep=.005,
    )

    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()

    zero_action = np.zeros_like(env.physics.data.ctrl)

    # Walker starts in upright position.
    # Should not trigger failure termination in the first few steps.
    for _ in range(5):
      env.step(zero_action)
      self.assertFalse(task.should_terminate_episode(env.physics))
      np.testing.assert_array_equal(task.get_discount(env.physics), 1)

    # Rotate the walker upside down and run the physics until it makes contact.
    current_time = env.physics.data.time
    walker.shift_pose(env.physics, position=(0, 0, 10), quaternion=(0, 1, 0, 0))
    env.physics.forward()
    while env.physics.data.ncon == 0:
      env.physics.step()
    env.physics.data.time = current_time

    # Should now trigger a failure termination.
    env.step(zero_action)
    self.assertTrue(task.should_terminate_episode(env.physics))


if __name__ == '__main__':
  absltest.main()
