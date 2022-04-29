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

"""Simple script to visualize motion capture data."""

from absl import app

from dm_control import composer
from dm_control import viewer

from dm_control.locomotion import arenas
from dm_control.locomotion import walkers

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.tasks.reference_pose import tracking


def mocap_playback_env(random_state=None):
  """Constructs mocap playback environment."""

  # Use a position-controlled CMU humanoid walker.
  walker_type = walkers.CMUHumanoidPositionControlledV2020

  # Build an empty arena.
  arena = arenas.Floor()

  # Build a task that rewards the agent for tracking motion capture reference
  # data.
  task = tracking.PlaybackTask(
      walker=walker_type,
      arena=arena,
      ref_path=cmu_mocap_data.get_path_for_cmu(version='2020'),
      dataset='run_jump_tiny',
  )

  return composer.Environment(time_limit=30,
                              task=task,
                              random_state=random_state,
                              strip_singleton_obs_buffer_dim=True)


def main(unused_argv):
  # The viewer calls the environment_loader on episode resets. However the task
  # cycles through one clip per episode. To avoid replaying the first clip again
  # and again we construct the environment outside the viewer to make it
  # persistent across resets.
  env = mocap_playback_env()
  viewer.launch(environment_loader=lambda: env)

if __name__ == '__main__':
  app.run(main)
