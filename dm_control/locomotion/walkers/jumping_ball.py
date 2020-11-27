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
"""Walkers based on an actuated jumping ball."""

import os

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.walkers import legacy_base
import numpy as np

_ASSETS_PATH = os.path.join(os.path.dirname(__file__),
                            'assets/jumping_ball')


class JumpingBallWithHead(legacy_base.Walker):
  """A rollable and jumpable ball with a head."""

  def _build(self, name='walker', marker_rgba=None, camera_control=False,
             initializer=None, add_ears=False, camera_height=None):
    """Build a JumpingBallWithHead.

    Args:
      name: name of the walker.
      marker_rgba: RGBA value set to walker.marker_geoms to distinguish between
        walkers (in multi-agent setting).
      camera_control: If `True`, the walker exposes two additional actuated
        degrees of freedom to control the egocentric camera height and tilt.
      initializer: (Optional) A `WalkerInitializer` object.
      add_ears: a boolean. Same as the nose above but the red/blue balls are
        placed to the left/right of the agent. Better for egocentric vision.
      camera_height: A float specifying the height of the camera, or `None` if
        the camera height should be left as specified in the XML model.
    """
    super()._build(initializer=initializer)
    self._mjcf_root = self._mjcf_root = mjcf.from_path(self._xml_path)

    if name:
      self._mjcf_root.model = name

    if camera_height is not None:
      self._mjcf_root.find('body', 'egocentric_camera').pos[2] = camera_height

    if add_ears:
      # Large ears
      head = self._mjcf_root.find('body', 'head_body')
      head.add('site', type='sphere', size=(.26,),
               pos=(.22, 0, 0),
               rgba=(.7, 0, 0, 1))
      head.add('site', type='sphere', size=(.26,),
               pos=(-.22, 0, 0),
               rgba=(0, 0, .7, 1))
    # Set corresponding marker color if specified.
    if marker_rgba is not None:
      for geom in self.marker_geoms:
        geom.set_attributes(rgba=marker_rgba)

    self._root_joints = None
    self._camera_control = camera_control
    if not camera_control:
      for name in ('camera_height', 'camera_tilt'):
        self._mjcf_root.find('actuator', name).remove()
        self._mjcf_root.find('joint', name).remove()

  @property
  def _xml_path(self):
    return os.path.join(_ASSETS_PATH, 'jumping_ball_with_head.xml')

  @property
  def marker_geoms(self):
    return [self._mjcf_root.find('geom', 'head')]

  def create_root_joints(self, attachment_frame):
    root_class = self._mjcf_root.find('default', 'root')
    root_x = attachment_frame.add(
        'joint', name='root_x', type='slide', axis=[1, 0, 0], dclass=root_class)
    root_y = attachment_frame.add(
        'joint', name='root_y', type='slide', axis=[0, 1, 0], dclass=root_class)
    root_z = attachment_frame.add(
        'joint', name='root_z', type='slide', axis=[0, 0, 1], dclass=root_class)
    self._root_joints = [root_x, root_y, root_z]

  def set_pose(self, physics, position=None, quaternion=None):
    if position is not None:
      if self._root_joints is not None:
        physics.bind(self._root_joints).qpos = position
      else:
        super().set_pose(physics, position, quaternion=None)
    physics.bind(self._mjcf_root.find_all('joint')).qpos = 0.
    if quaternion is not None:
      # This walker can only rotate along the z-axis, so we extract only that
      # component from the quaternion.
      z_angle = np.arctan2(
          2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
          1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))
      physics.bind(self._mjcf_root.find('joint', 'steer')).qpos = z_angle

  def initialize_episode(self, physics, unused_random_state):
    # gravity compensation
    if self._camera_control:
      gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
      comp_bodies = physics.bind(self._mjcf_root.find('body',
                                                      'egocentric_camera'))
      comp_bodies.xfrc_applied = -gravity * comp_bodies.mass[..., None]

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @composer.cached_property
  def actuators(self):
    return self._mjcf_root.find_all('actuator')

  @composer.cached_property
  def root_body(self):
    return self._mjcf_root.find('body', 'head_body')

  @composer.cached_property
  def end_effectors(self):
    return [self._mjcf_root.find('body', 'head_body')]

  @composer.cached_property
  def observable_joints(self):
    return [self._mjcf_root.find('joint', 'kick')]

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'egocentric')

  @composer.cached_property
  def ground_contact_geoms(self):
    return (self._mjcf_root.find('geom', 'shell'),)


class RollingBallWithHead(JumpingBallWithHead):
  """A rollable ball with a head."""

  def _build(self, **kwargs):
    super()._build(**kwargs)
    self._mjcf_root.find('actuator', 'kick').remove()
    self._mjcf_root.find('joint', 'kick').remove()

  @composer.cached_property
  def observable_joints(self):
    return []
