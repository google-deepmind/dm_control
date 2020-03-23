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

"""Walkers based on an actuated jumping ball."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import initializers
from dm_control.locomotion.walkers import legacy_base
import numpy as np
from PIL import Image
import six

from dm_control.utils import io as resources

_ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'boxhead')
_MAX_WALKER_ID = 10
_INVALID_WALKER_ID = 'walker_id must be in [0-{}], got: {{}}.'.format(
    _MAX_WALKER_ID)


def _compensate_gravity(physics, body_elements):
  """Applies Cartesian forces to bodies in order to exactly counteract gravity.

  Note that this will also affect the output of pressure, force, or torque
  sensors within the kinematic chain leading from the worldbody to the bodies
  that are being gravity-compensated.

  Args:
    physics: An `mjcf.Physics` instance to modify.
    body_elements: An iterable of `mjcf.Element`s specifying the bodies to which
      gravity compensation will be applied.
  """
  gravity = np.hstack([physics.model.opt.gravity, [0, 0, 0]])
  bodies = physics.bind(body_elements)
  bodies.xfrc_applied = -gravity * bodies.mass[..., None]


def _alpha_blend(foreground, background):
  """Does alpha compositing of two RGBA images.

  Both inputs must be (..., 4) numpy arrays whose shapes are compatible for
  broadcasting. They are assumed to contain float RGBA values in [0, 1].

  Args:
    foreground: foreground RGBA image.
    background: background RGBA image.

  Returns:
    A numpy array of shape (..., 4) containing the blended image.
  """
  fg, bg = np.broadcast_arrays(foreground, background)
  fg_rgb = fg[..., :3]
  fg_a = fg[..., 3:]
  bg_rgb = bg[..., :3]
  bg_a = bg[..., 3:]
  out = np.empty_like(bg)
  out_a = out[..., 3:]
  out_rgb = out[..., :3]
  # https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
  out_a[:] = fg_a + bg_a * (1. - fg_a)
  out_rgb[:] = fg_rgb * fg_a + bg_rgb * bg_a * (1. - fg_a)
  # Avoid division by zero if foreground and background are both transparent.
  out_rgb[:] = np.where(out_a, out_rgb / out_a, out_rgb)
  return out


def _asset_png_with_background_rgba_bytes(asset_fname, background_rgba):
  """Decode PNG from asset file and add solid background."""

  # Retrieve PNG image contents as a bytestring, convert to a numpy array.
  contents = resources.GetResource(os.path.join(_ASSETS_PATH, asset_fname))
  digit_rgba = np.array(Image.open(six.BytesIO(contents)), dtype=np.double)

  # Add solid background with `background_rgba`.
  blended = 255. * _alpha_blend(digit_rgba / 255., np.asarray(background_rgba))

  # Encode composite image array to a PNG bytestring.
  img = Image.fromarray(blended.astype(np.uint8), mode='RGBA')
  buf = six.BytesIO()
  img.save(buf, format='PNG')
  png_encoding = buf.getvalue()
  buf.close()

  return png_encoding


class BoxHeadObservables(legacy_base.WalkerObservables):
  """BoxHead observables with low-res camera and modulo'd rotational joints."""

  def __init__(self, entity, camera_resolution):
    self._camera_resolution = camera_resolution
    super(BoxHeadObservables, self).__init__(entity)

  @composer.observable
  def egocentric_camera(self):
    width, height = self._camera_resolution
    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=width, height=height)

  @property
  def proprioception(self):
    proprioception = super(BoxHeadObservables, self).proprioception
    if self._entity.observable_camera_joints:
      return proprioception + [self.camera_joints_pos, self.camera_joints_vel]
    return proprioception

  @composer.observable
  def camera_joints_pos(self):

    def _sin(value, random_state):
      del random_state
      return np.sin(value)

    def _cos(value, random_state):
      del random_state
      return np.cos(value)

    sin_rotation_joints = observable.MJCFFeature(
        'qpos', self._entity.observable_camera_joints, corruptor=_sin)

    cos_rotation_joints = observable.MJCFFeature(
        'qpos', self._entity.observable_camera_joints, corruptor=_cos)

    def _camera_joints(physics):
      return np.concatenate([
          sin_rotation_joints(physics),
          cos_rotation_joints(physics)
      ], -1)

    return observable.Generic(_camera_joints)

  @composer.observable
  def camera_joints_vel(self):
    return observable.MJCFFeature(
        'qvel', self._entity.observable_camera_joints)


class BoxHead(legacy_base.Walker):
  """A rollable and jumpable ball with a head."""

  def _build(self,
             name='walker',
             marker_rgba=None,
             camera_control=False,
             camera_resolution=(28, 28),
             roll_gear=-60,
             steer_gear=55,
             walker_id=None,
             initializer=None):
    """Build a BoxHead.

    Args:
      name: name of the walker.
      marker_rgba: RGBA value set to walker.marker_geoms to distinguish between
        walkers (in multi-agent setting).
      camera_control: If `True`, the walker exposes two additional actuated
        degrees of freedom to control the egocentric camera height and tilt.
      camera_resolution: egocentric camera rendering resolution.
      roll_gear: gear determining forward acceleration.
      steer_gear: gear determining steering (spinning) torque.
      walker_id: (Optional) An integer in [0-10], this number will be shown on
        the walker's head. Defaults to `None` which does not show any number.
      initializer: (Optional) A `WalkerInitializer` object.

    Raises:
      ValueError: if received invalid walker_id.
    """
    super(BoxHead, self)._build(
        initializer=initializer or initializers.NoOpInitializer())
    xml_path = os.path.join(_ASSETS_PATH, 'boxhead.xml')
    self._mjcf_root = mjcf.from_xml_string(resources.GetResource(xml_path, 'r'))
    if name:
      self._mjcf_root.model = name

    if walker_id is not None and not 0 <= walker_id <= _MAX_WALKER_ID:
      raise ValueError(_INVALID_WALKER_ID.format(walker_id))

    self._walker_id = walker_id
    if walker_id is not None:
      png_bytes = _asset_png_with_background_rgba_bytes(
          'digits/%02d.png' % walker_id, marker_rgba)
      head_texture = self._mjcf_root.asset.add(
          'texture',
          name='head_texture',
          type='2d',
          file=mjcf.Asset(png_bytes, '.png'))
      head_material = self._mjcf_root.asset.add(
          'material', name='head_material', texture=head_texture)
      self._mjcf_root.find('geom', 'head').material = head_material
      self._mjcf_root.find('geom', 'head').rgba = None

      self._mjcf_root.find('geom', 'top_down_cam_box').material = head_material
      self._mjcf_root.find('geom', 'top_down_cam_box').rgba = None

    self._body_texture = self._mjcf_root.asset.add(
        'texture',
        name='ball_body',
        type='cube',
        builtin='checker',
        rgb1=marker_rgba[:-1] if marker_rgba else '.4 .4 .4',
        rgb2='.8 .8 .8',
        width='100',
        height='100')
    self._body_material = self._mjcf_root.asset.add(
        'material', name='ball_body', texture=self._body_texture)
    self._mjcf_root.find('geom', 'shell').material = self._body_material

    # Set corresponding marker color if specified.
    if marker_rgba is not None:
      for geom in self.marker_geoms:
        geom.set_attributes(rgba=marker_rgba)

    self._root_joints = None
    self._camera_control = camera_control
    self._camera_resolution = camera_resolution
    if not camera_control:
      for name in ('camera_pitch', 'camera_yaw'):
        self._mjcf_root.find('actuator', name).remove()
        self._mjcf_root.find('joint', name).remove()
    self._roll_gear = roll_gear
    self._steer_gear = steer_gear
    self._mjcf_root.find('actuator', 'roll').gear[0] = self._roll_gear
    self._mjcf_root.find('actuator', 'steer').gear[0] = self._steer_gear

    # Initialize previous action.
    self._prev_action = np.zeros(shape=self.action_spec.shape,
                                 dtype=self.action_spec.dtype)

  def _build_observables(self):
    return BoxHeadObservables(self, camera_resolution=self._camera_resolution)

  @property
  def marker_geoms(self):
    geoms = [
        self._mjcf_root.find('geom', 'arm_l'),
        self._mjcf_root.find('geom', 'arm_r'),
        self._mjcf_root.find('geom', 'eye_l'),
        self._mjcf_root.find('geom', 'eye_r'),
    ]
    if self._walker_id is None:
      geoms.append(self._mjcf_root.find('geom', 'head'))
    return geoms

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
        super(BoxHead, self).set_pose(physics, position, quaternion=None)
    physics.bind(self._mjcf_root.find_all('joint')).qpos = 0.
    if quaternion is not None:
      # This walker can only rotate along the z-axis, so we extract only that
      # component from the quaternion.
      z_angle = np.arctan2(
          2 * (quaternion[0] * quaternion[3] + quaternion[1] * quaternion[2]),
          1 - 2 * (quaternion[2] ** 2 + quaternion[3] ** 2))
      physics.bind(self._mjcf_root.find('joint', 'steer')).qpos = z_angle

  def initialize_episode(self, physics, random_state):
    self.reinitialize_pose(physics, random_state)

    if self._camera_control:
      _compensate_gravity(physics,
                          self._mjcf_root.find('body', 'egocentric_camera'))
    self._prev_action = np.zeros(shape=self.action_spec.shape,
                                 dtype=self.action_spec.dtype)

  def apply_action(self, physics, action, random_state):
    super(BoxHead, self).apply_action(physics, action, random_state)

    # Updates previous action.
    self._prev_action[:] = action

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
    return (self._mjcf_root.find('body', 'head_body'),)

  @composer.cached_property
  def observable_joints(self):
    return (self._mjcf_root.find('joint', 'kick'),)

  @composer.cached_property
  def observable_camera_joints(self):
    if self._camera_control:
      return (
          self._mjcf_root.find('joint', 'camera_yaw'),
          self._mjcf_root.find('joint', 'camera_pitch'),
      )
    return ()

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'egocentric')

  @composer.cached_property
  def ground_contact_geoms(self):
    return (self._mjcf_root.find('geom', 'shell'),)

  @property
  def prev_action(self):
    return self._prev_action
