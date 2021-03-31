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

"""Produces reference environments for CMU humanoid tracking task."""


from dm_control import composer
from dm_control.locomotion import arenas

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking

from dm_control.locomotion.walkers import cmu_humanoid


def cmu_humanoid_tracking(random_state=None):
  """Requires a CMU humanoid to run down a corridor obstructed by walls."""

  # Use a position-controlled CMU humanoid walker.
  walker_type = cmu_humanoid.CMUHumanoidPositionControlledV2020

  # Build an empty arena.
  arena = arenas.Floor()

  # Build a task that rewards the agent for tracking motion capture reference
  # data.
  task = tracking.MultiClipMocapTracking(
      walker=walker_type,
      arena=arena,
      ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'),
      dataset='walk_tiny',
      ref_steps=(1, 2, 3, 4, 5),
      min_steps=10,
      reward_type='comic',
  )

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)

