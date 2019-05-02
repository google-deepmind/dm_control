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

from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.locomotion.arenas import corridors as arenas
from dm_control.locomotion.tasks import corridors as tasks
from dm_control.locomotion.walkers import cmu_humanoid


def cmu_humanoid_run_walls(random_state=None):
  """Requires a CMU humanoid to run down a corridor obstructed by walls."""

  # Build a position-controlled CMU humanoid walker.
  walker = cmu_humanoid.CMUHumanoidPositionControlled(
      observable_options={'egocentric_camera': dict(enabled=True)})

  # Build a corridor-shaped arena that is obstructed by walls.
  arena = arenas.WallsCorridor(
      wall_gap=4.,
      wall_width=distributions.Uniform(1, 7),
      wall_height=3.0,
      corridor_width=10,
      corridor_length=100)

  # Build a task that rewards the agent for running down the corridor at a
  # specific velocity.
  task = tasks.RunThroughCorridor(
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
  arena = arenas.GapsCorridor(
      platform_length=distributions.Uniform(.3, 2.5),
      gap_length=distributions.Uniform(.5, 1.25),
      corridor_width=10,
      corridor_length=100)

  # Build a task that rewards the agent for running down the corridor at a
  # specific velocity.
  task = tasks.RunThroughCorridor(
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
