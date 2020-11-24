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

"""Module containing the Jaco robot class."""

import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable
from dm_control.entities.manipulators import base
from dm_control.entities.manipulators.kinova import assets_path
import numpy as np

_JACO_ARM_XML_PATH = os.path.join(assets_path.KINOVA_ROOT, 'jaco_arm.xml')
_LARGE_JOINTS = ('joint_1', 'joint_2', 'joint_3')
_SMALL_JOINTS = ('joint_4', 'joint_5', 'joint_6')
_ALL_JOINTS = _LARGE_JOINTS + _SMALL_JOINTS
_WRIST_SITE = 'wristsite'

# These are the peak torque limits taken from Kinova's datasheet:
# https://www.kinovarobotics.com/sites/default/files/AS-ACT-KA58-KA75-SP-INT-EN%20201804-1.2%20%28KINOVA%E2%84%A2%20Actuator%20series%20KA75%2B%20KA-58%20Specifications%29.pdf
_LARGE_JOINT_MAX_TORQUE = 30.5
_SMALL_JOINT_MAX_TORQUE = 6.8

# On the real robot these limits are imposed by the actuator firmware. It's
# technically possible to exceed them via the low-level API, but this can reduce
# the lifetime of the actuators.
_LARGE_JOINT_MAX_VELOCITY = np.deg2rad(36.)
_SMALL_JOINT_MAX_VELOCITY = np.deg2rad(48.)

# The velocity actuator gain is a very rough estimate, and should be considered
# a placeholder for proper system identification.
_VELOCITY_GAIN = 500.


class JacoArm(base.RobotArm):
  """A composer entity representing a Jaco arm."""

  def _build(self, name=None):
    """Initializes the JacoArm.

    Args:
      name: String, the name of this robot. Used as a prefix in the MJCF name
        name attributes.
    """
    self._mjcf_root = mjcf.from_path(_JACO_ARM_XML_PATH)
    if name:
      self._mjcf_root.model = name
    # Find MJCF elements that will be exposed as attributes.
    self._joints = [self._mjcf_root.find('joint', name) for name in _ALL_JOINTS]
    self._wrist_site = self._mjcf_root.find('site', _WRIST_SITE)
    self._bodies = self.mjcf_model.find_all('body')
    # Add actuators.
    self._actuators = [_add_velocity_actuator(joint) for joint in self._joints]
    # Add torque sensors.
    self._joint_torque_sensors = [
        _add_torque_sensor(joint) for joint in self._joints]

  def _build_observables(self):
    return JacoArmObservables(self)

  @property
  def joints(self):
    """List of joint elements belonging to the arm."""
    return self._joints

  @property
  def actuators(self):
    """List of actuator elements belonging to the arm."""
    return self._actuators

  @property
  def joint_torque_sensors(self):
    """List of torque sensors for each joint belonging to the arm."""
    return self._joint_torque_sensors

  @property
  def wrist_site(self):
    """Wrist site of the arm (attachment point for the hand)."""
    return self._wrist_site

  @property
  def mjcf_model(self):
    """Returns the `mjcf.RootElement` object corresponding to this robot."""
    return self._mjcf_root


def _add_velocity_actuator(joint):
  """Adds a velocity actuator to a joint, returns the new MJCF element."""

  if joint.name in _LARGE_JOINTS:
    max_torque = _LARGE_JOINT_MAX_TORQUE
    max_velocity = _LARGE_JOINT_MAX_VELOCITY
  elif joint.name in _SMALL_JOINTS:
    max_torque = _SMALL_JOINT_MAX_TORQUE
    max_velocity = _SMALL_JOINT_MAX_VELOCITY
  else:
    raise ValueError('`joint.name` must be one of {}, got {!r}.'
                     .format(_ALL_JOINTS, joint.name))
  return joint.root.actuator.add(
      'velocity',
      joint=joint,
      name=joint.name,
      kv=_VELOCITY_GAIN,
      ctrllimited=True,
      ctrlrange=(-max_velocity, max_velocity),
      forcelimited=True,
      forcerange=(-max_torque, max_torque))


def _add_torque_sensor(joint):
  """Adds a torque sensor to a joint, returns the new MJCF element."""
  site = joint.parent.add(
      'site', size=[1e-3], group=composer.SENSOR_SITES_GROUP,
      name=joint.name+'_site')
  return joint.root.sensor.add('torque', site=site, name=joint.name+'_torque')


class JacoArmObservables(base.JointsObservables):
  """Jaco arm obserables."""

  @define.observable
  def joints_pos(self):
    # Because most of the Jaco arm joints are unlimited, we return the joint
    # angles as sine/cosine pairs so that the observations are bounded.
    def get_sin_cos_joint_angles(physics):
      joint_pos = physics.bind(self._entity.joints).qpos
      return np.vstack([np.sin(joint_pos), np.cos(joint_pos)]).T
    return observable.Generic(get_sin_cos_joint_angles)

  @define.observable
  def joints_torque(self):
    # MuJoCo's torque sensors are 3-axis, but we are only interested in torques
    # acting about the axis of rotation of the joint. We therefore project the
    # torques onto the joint axis.
    def get_torques(physics):
      torques = physics.bind(self._entity.joint_torque_sensors).sensordata
      joint_axes = physics.bind(self._entity.joints).axis
      return np.einsum('ij,ij->i', torques.reshape(-1, 3), joint_axes)
    return observable.Generic(get_torques)
