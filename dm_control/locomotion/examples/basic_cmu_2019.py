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

"""Produces reference environments for CMU humanoid locomotion tasks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.tasks import go_to_target
from dm_control.locomotion.tasks import random_goal_maze
from dm_control.locomotion.walkers import cmu_humanoid
from labmaze import fixed_maze


def cmu_humanoid_run_walls(random_state=None):
  """Requires a CMU humanoid to run down a corridor obstructed by walls."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a corridor-shaped arena that is obstructed by walls.
  arena = corr_arenas.WallsCorridor(
      wall_gap=4.,
      wall_width=distributions.Uniform(1, 7),
      wall_height=3.0,
      corridor_width=10,
      corridor_length=100,
      include_initial_padding=False)

  # Build a task that rewards the agent for running down the corridor at a
  # specific velocity.
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(0.5, 0, 0),
      target_velocity=3.0,
      physics_timestep=0.005,
      control_timestep=0.03)

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def cmu_humanoid_run_gaps(random_state=None):
  """Requires a CMU humanoid to run down a corridor with gaps."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a corridor-shaped arena with gaps, where the sizes of the gaps and
  # platforms are uniformly randomized.
  arena = corr_arenas.GapsCorridor(
      platform_length=distributions.Uniform(.3, 2.5),
      gap_length=distributions.Uniform(.5, 1.25),
      corridor_width=10,
      corridor_length=100)

  # Build a task that rewards the agent for running down the corridor at a
  # specific velocity.
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(0.5, 0, 0),
      target_velocity=3.0,
      physics_timestep=0.005,
      control_timestep=0.03)

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def cmu_humanoid_go_to_target(random_state=None):
  """Requires a CMU humanoid to go to a target."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled()

  # Build a standard floor arena.
  arena = floors.Floor()

  # Build a task that rewards the agent for going to a target.
  task = go_to_target.GoToTarget(
      walker=walker,
      arena=arena,
      physics_timestep=0.005,
      control_timestep=0.03)

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def cmu_humanoid_maze_forage(random_state=None):
  """Requires a CMU humanoid to find all items in a maze."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

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

  # Build a task that rewards the agent for obtaining targets.
  task = random_goal_maze.ManyGoalsMaze(
      walker=walker,
      maze_arena=arena,
      target_builder=functools.partial(
          target_sphere.TargetSphere,
          radius=0.4,
          rgb1=(0, 0, 0.4),
          rgb2=(0, 0, 0.7)),
      target_reward_scale=50.,
      physics_timestep=0.005,
      control_timestep=0.03,
  )

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def cmu_humanoid_heterogeneous_forage(random_state=None):
  """Requires a CMU humanoid to find all items of a particular type in a maze."""
  level = ('*******\n'
           '*     *\n'
           '*  P  *\n'
           '*     *\n'
           '*  G  *\n'
           '*     *\n'
           '*******\n')

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  skybox_texture = labmaze_textures.SkyBox(style='sky_03')
  wall_textures = labmaze_textures.WallTextures(style='style_01')
  floor_textures = labmaze_textures.FloorTextures(style='style_01')
  maze = fixed_maze.FixedMazeWithRandomGoals(
      entity_layer=level,
      variations_layer=None,
      num_spawns=1,
      num_objects=6,
  )
  arena = mazes.MazeWithTargets(
      maze=maze,
      xy_scale=3.0,
      z_height=2.0,
      skybox_texture=skybox_texture,
      wall_textures=wall_textures,
      floor_textures=floor_textures,
  )
  task = random_goal_maze.ManyHeterogeneousGoalsMaze(
      walker=walker,
      maze_arena=arena,
      target_builders=[
          functools.partial(
              target_sphere.TargetSphere,
              radius=0.4,
              rgb1=(0, 0.4, 0),
              rgb2=(0, 0.7, 0)),
          functools.partial(
              target_sphere.TargetSphere,
              radius=0.4,
              rgb1=(0.4, 0, 0),
              rgb2=(0.7, 0, 0)),
      ],
      randomize_spawn_rotation=False,
      target_type_rewards=[30., -10.],
      target_type_proportions=[1, 1],
      shuffle_target_builders=True,
      aliveness_reward=0.01,
      control_timestep=.03,
  )

  return composer.Environment(
      time_limit=25,
      task=task,
      random_state=random_state,
      strip_singleton_obs_buffer_dim=True)
