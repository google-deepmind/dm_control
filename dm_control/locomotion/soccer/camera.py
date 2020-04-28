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

"""Cameras for recording soccer videos."""

from dm_control.mujoco import engine
import numpy as np


class MultiplayerTrackingCamera(object):
  """Camera that smoothly tracks multiple entities."""

  def __init__(
      self,
      min_distance,
      distance_factor,
      smoothing_update_speed,
      azimuth=90,
      elevation=-45,
      width=1920,
      height=1080,
  ):
    """Construct a new MultiplayerTrackingcamera.

    The target lookat point is the centroid of all tracked entities.
    Target camera distance is set to min_distance + distance_factor * d_max,
    where d_max is the maximum distance of any entity to the lookat point.

    Args:
      min_distance: minimum camera distance.
      distance_factor: camera distance multiplier (see above).
      smoothing_update_speed: exponential filter parameter to smooth camera
        movement. 1 means no filter; smaller values mean less change per step.
      azimuth: constant azimuth to use for camera.
      elevation: constant elevation to use for camera.
      width: width to use for rendered video.
      height: height to use for rendered video.
    """
    self._min_distance = min_distance
    self._distance_factor = distance_factor
    if smoothing_update_speed < 0 or smoothing_update_speed > 1:
      raise ValueError("Filter speed must be in range [0, 1].")
    self._smoothing_update_speed = smoothing_update_speed
    self._azimuth = azimuth
    self._elevation = elevation
    self._width = width
    self._height = height
    self._camera = None

  @property
  def camera(self):
    return self._camera

  def render(self):
    """Render the current frame."""
    if self._camera is None:
      raise ValueError(
          "Camera has not been initialized yet."
          " render can only be called after physics has been compiled."
      )
    return self._camera.render()

  def after_compile(self, physics):
    """Instantiate the camera and ensure rendering buffer is large enough."""
    buffer_height = max(self._height, physics.model.vis.global_.offheight)
    buffer_width = max(self._width, physics.model.vis.global_.offwidth)
    physics.model.vis.global_.offheight = buffer_height
    physics.model.vis.global_.offwidth = buffer_width
    self._camera = engine.MovableCamera(
        physics, height=self._height, width=self._width)

  def _get_target_camera_pose(self, entity_positions):
    """Returns the pose that the camera should be pulled toward.

    Args:
      entity_positions: list of numpy arrays representing current positions of
        the entities to be tracked.
    Returns: mujoco.engine.Pose representing the target camera pose.
    """
    stacked_positions = np.stack(entity_positions)
    centroid = np.mean(stacked_positions, axis=0)
    radii = np.linalg.norm(stacked_positions - centroid, axis=1)
    assert len(radii) == len(entity_positions)
    camera_distance = self._min_distance + self._distance_factor * np.max(radii)
    return engine.Pose(
        lookat=centroid,
        distance=camera_distance,
        azimuth=self._azimuth,
        elevation=self._elevation,
    )

  def initialize_episode(self, entity_positions):
    """Begin the episode with the camera set to its target pose."""
    target_pose = self._get_target_camera_pose(entity_positions)
    self._camera.set_pose(*target_pose)

  def after_step(self, entity_positions):
    """Move camera toward its target poses."""
    target_pose = self._get_target_camera_pose(entity_positions)
    cur_pose = self._camera.get_pose()
    smoothing_update_speed = self._smoothing_update_speed
    filtered_pose = [
        target_val * smoothing_update_speed + \
        current_val * (1 - smoothing_update_speed)
        for target_val, current_val in zip(target_pose, cur_pose)
    ]
    self._camera.set_pose(*filtered_pose)
