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

"""Abstract base classes for robot arms and hands."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics
import numpy as np
import six
from six.moves import range
from six.moves import zip


DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])

_INVALID_JOINTS_ERROR = (
    'All non-hinge joints must have limits. Model contains the following '
    'non-hinge joints which are unbounded:\n{invalid_str}')


@six.add_metaclass(abc.ABCMeta)
class RobotArm(composer.Robot):
  """The abstract base class for robotic arms."""

  def _build_observables(self):
    return JointsObservables(self)

  @property
  def attachment_site(self):
    return self.wrist_site

  def _get_joint_pos_sampling_bounds(self, physics):
    """Returns lower and upper bounds for sampling arm joint positions.

    Args:
      physics: An `mjcf.Physics` instance.

    Returns:
      A (2, num_joints) numpy array containing (lower, upper) position bounds.
      For hinge joints without limits the bounds are defined as [0, 2pi].

    Raises:
      RuntimeError: If the model contains unlimited joints that are not hinges.
    """
    bound_joints = physics.bind(self.joints)
    limits = np.array(bound_joints.range, copy=True)
    is_hinge = bound_joints.type == mjbindings.enums.mjtJoint.mjJNT_HINGE
    is_limited = bound_joints.limited.astype(np.bool)
    invalid = ~is_hinge & ~is_limited  # All non-hinge joints must have limits.
    if any(invalid):
      invalid_str = '\n'.join(str(self.joints[i]) for i in np.where(invalid)[0])
      raise RuntimeError(_INVALID_JOINTS_ERROR.format(invalid_str=invalid_str))
    # For unlimited hinges we sample positions between 0 and 2pi.
    limits[is_hinge & ~is_limited] = 0., 2*np.pi
    return limits.T

  def randomize_arm_joints(self, physics, random_state):
    """Randomizes the qpos of all arm joints.

    The ranges of qpos values is determined from the MJCF model.

    Args:
      physics: A `mujoco.Physics` instance.
      random_state: An `np.random.RandomState` instance.
    """
    lower, upper = self._get_joint_pos_sampling_bounds(physics)
    physics.bind(self.joints).qpos = random_state.uniform(lower, upper)

  def set_site_to_xpos(self, physics, random_state, site, target_pos,
                       target_quat=None, max_ik_attempts=10):
    """Moves the arm so that a site occurs at the specified location.

    This function runs the inverse kinematics solver to find a configuration
    arm joints for which the pinch site occurs at the specified location in
    Cartesian coordinates.

    Args:
      physics: A `mujoco.Physics` instance.
      random_state: An `np.random.RandomState` instance.
      site: Either a `mjcf.Element` or a string specifying the full name
        of the site whose position is being set.
      target_pos: The desired Cartesian location of the site.
      target_quat: (optional) The desired orientation of the site, expressed
        as a quaternion. If `None`, the default orientation is to point
        vertically downwards.
      max_ik_attempts: (optional) Maximum number of attempts to make at finding
        a solution satisfying `target_pos` and `target_quat`. The joint
        positions will be randomized after each unsuccessful attempt.

    Returns:
      A boolean indicating whether the desired configuration is obtained.

    Raises:
      ValueError: If site is neither a string nor an `mjcf.Element`.
    """
    if isinstance(site, mjcf.Element):
      site_name = site.full_identifier
    elif isinstance(site, str):
      site_name = site
    else:
      raise ValueError('site should either be a string or mjcf.Element: got {}'
                       .format(site))
    if target_quat is None:
      target_quat = DOWN_QUATERNION
    lower, upper = self._get_joint_pos_sampling_bounds(physics)
    arm_joint_names = [joint.full_identifier for joint in self.joints]

    for _ in range(max_ik_attempts):
      result = inverse_kinematics.qpos_from_site_pose(
          physics=physics,
          site_name=site_name,
          target_pos=target_pos,
          target_quat=target_quat,
          joint_names=arm_joint_names,
          rot_weight=2,
          inplace=True)
      success = result.success

      # Canonicalise the angle to [0, 2*pi]
      if success:
        for arm_joint, low, high in zip(self.joints, lower, upper):
          arm_joint_mj = physics.bind(arm_joint)
          while arm_joint_mj.qpos >= high:
            arm_joint_mj.qpos -= 2*np.pi
          while arm_joint_mj.qpos < low:
            arm_joint_mj.qpos += 2*np.pi
            if arm_joint_mj.qpos > high:
              success = False
              break

      # If succeeded or only one attempt, break and do not randomize joints.
      if success or max_ik_attempts <= 1:
        break
      else:
        self.randomize_arm_joints(physics, random_state)

    return success

  @abc.abstractproperty
  def joints(self):
    """Returns the joint elements of the arm."""
    raise NotImplementedError

  @abc.abstractproperty
  def wrist_site(self):
    """Returns the wrist site element of the arm."""
    raise NotImplementedError


class JointsObservables(composer.Observables):
  """Observables common to all robot arms."""

  @define.observable
  def joints_pos(self):
    return observable.MJCFFeature('qpos', self._entity.joints)

  @define.observable
  def joints_vel(self):
    return observable.MJCFFeature('qvel', self._entity.joints)


@six.add_metaclass(abc.ABCMeta)
class RobotHand(composer.Robot):
  """The abstract base class for robotic hands."""

  @abc.abstractmethod
  def set_grasp(self, physics, close_factors):
    """Sets the finger position to the desired positions.

    Args:
      physics: An instance of `mjcf.Physics`.
      close_factors: A number or list of numbers defining the desired grasp
        position of each finger. A value of 0 corresponds to fully opening a
        finger, while a value of 1 corresponds to fully closing it. If a single
        number is specified, the same position is applied to all fingers.
    """

  @abc.abstractproperty
  def tool_center_point(self):
    """Returns the tool center point element of the hand."""
