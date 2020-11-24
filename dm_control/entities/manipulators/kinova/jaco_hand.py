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

"""Module containing the standard Jaco hand."""

import collections
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.entities.manipulators import base
from dm_control.entities.manipulators.kinova import assets_path

_JACO_HAND_XML_PATH = os.path.join(assets_path.KINOVA_ROOT, 'jaco_hand.xml')
_HAND_BODY = 'hand'
_PINCH_SITE = 'pinchsite'
_GRIP_SITE = 'gripsite'


class JacoHand(base.RobotHand):
  """A composer entity representing a Jaco hand."""

  def _build(self,
             name=None,
             use_pinch_site_as_tcp=False):
    """Initializes the JacoHand.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
      use_pinch_site_as_tcp: (optional) A boolean, if `True` the pinch site
        will be used as the tool center point. If `False` the grip site is used.
    """
    self._mjcf_root = mjcf.from_path(_JACO_HAND_XML_PATH)
    if name:
      self._mjcf_root.model = name
    # Find MJCF elements that will be exposed as attributes.
    self._bodies = self.mjcf_model.find_all('body')
    self._tool_center_point = self._mjcf_root.find(
        'site', _PINCH_SITE if use_pinch_site_as_tcp else _GRIP_SITE)
    self._joints = self._mjcf_root.find_all('joint')
    self._hand_geoms = list(self._mjcf_root.find('body', _HAND_BODY).geom)
    self._finger_geoms = [geom for geom in self._mjcf_root.find_all('geom')
                          if geom.name and geom.name.startswith('finger')]
    self._grip_site = self._mjcf_root.find('site', _GRIP_SITE)
    self._pinch_site = self._mjcf_root.find('site', _PINCH_SITE)

    # Add actuators.
    self._finger_actuators = [
        _add_velocity_actuator(joint) for joint in self._joints]

  def _build_observables(self):
    return JacoHandObservables(self)

  @property
  def tool_center_point(self):
    """Tool center point for the Jaco hand."""
    return self._tool_center_point

  @property
  def joints(self):
    """List of joint elements."""
    return self._joints

  @property
  def actuators(self):
    """List of finger actuators."""
    return self._finger_actuators

  @property
  def hand_geom(self):
    """List of geoms belonging to the hand."""
    return self._hand_geoms

  @property
  def finger_geoms(self):
    """List of geoms belonging to the fingers."""
    return self._finger_geoms

  @property
  def grip_site(self):
    """Grip site."""
    return self._grip_site

  @property
  def pinch_site(self):
    """Pinch site."""
    return self._pinch_site

  @property
  def pinch_site_pos_sensor(self):
    """Sensor that returns the cartesian position of the pinch site."""
    return self._pinch_site_pos_sensor

  @property
  def pinch_site_quat_sensor(self):
    """Sensor that returns the orientation of the pinch site as a quaternion."""
    return self._pinch_site_quat_sensor

  @property
  def mjcf_model(self):
    """Returns the `mjcf.RootElement` object corresponding to this robot."""
    return self._mjcf_root

  def set_grasp(self, physics, close_factors):
    """Sets the finger position to the desired positions.

    Args:
      physics: An instance of `mjcf.Physics`.
      close_factors: A number or list of numbers defining the desired grasp
        position of each finger. A value of 0 corresponds to fully opening a
        finger, while a value of 1 corresponds to fully closing it. If a single
        number is specified, the same position is applied to all fingers.
    """
    if not isinstance(close_factors, collections.Iterable):
      close_factors = (close_factors,) * len(self.joints)
    for joint, finger_factor in zip(self.joints, close_factors):
      joint_mj = physics.bind(joint)
      min_value, max_value = joint_mj.range
      joint_mj.qpos = min_value + (max_value - min_value) * finger_factor
    physics.after_reset()

    # Set target joint velocities to zero.
    physics.bind(self.actuators).ctrl = 0


def _add_velocity_actuator(joint):
  """Adds a velocity actuator to a joint, returns the new MJCF element."""
  # These parameters were adjusted to achieve a grip force of ~25 N and a finger
  # closing time of ~1.2 s, as specified in the datasheet for the hand.
  gain = 10.
  forcerange = (-1., 1.)
  ctrlrange = (-5., 5.)  # Based on Kinova's URDF.
  return joint.root.actuator.add(
      'velocity',
      joint=joint,
      name=joint.name,
      kv=gain,
      ctrllimited=True,
      ctrlrange=ctrlrange,
      forcelimited=True,
      forcerange=forcerange)


class JacoHandObservables(base.JointsObservables):
  """Observables for the Jaco hand."""

  @composer.observable
  def pinch_site_pos(self):
    """The position of the pinch site, in global coordinates."""
    return observable.MJCFFeature('xpos', self._entity.pinch_site)

  @composer.observable
  def pinch_site_rmat(self):
    """The rotation matrix of the pinch site in global coordinates."""
    return observable.MJCFFeature('xmat', self._entity.pinch_site)

