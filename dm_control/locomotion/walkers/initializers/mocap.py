# Copyright 2021 The dm_control Authors.
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

"""Initializers for walkers that use motion capture data."""

from dm_control.locomotion.mocap import cmu_mocap_data
from dm_control.locomotion.mocap import loader
from dm_control.locomotion.walkers import initializers


class CMUMocapInitializer(initializers.UprightInitializer):
  """Initializer that uses data from a CMU mocap dataset.

     Only suitable if walker matches the motion capture data.
  """

  def __init__(self, mocap_key='CMU_077_02', version='2019'):
    """Load the trajectory."""
    ref_path = cmu_mocap_data.get_path_for_cmu(version)
    self._loader = loader.HDF5TrajectoryLoader(ref_path)
    self._trajectory = self._loader.get_trajectory(mocap_key)

  def initialize_pose(self, physics, walker, random_state):
    super(CMUMocapInitializer, self).initialize_pose(
        physics, walker, random_state)
    random_time = (self._trajectory.start_time +
                   self._trajectory.dt * random_state.randint(
                       self._trajectory.num_steps))
    (walker_timestep,) = self._trajectory.get_timestep_data(
        random_time).walkers
    physics.bind(walker.mocap_joints).qpos = walker_timestep.joints
    physics.bind(walker.mocap_joints).qvel = (
        walker_timestep.joints_velocity)
    walker.set_velocity(physics,
                        velocity=walker_timestep.velocity,
                        angular_velocity=walker_timestep.angular_velocity)
