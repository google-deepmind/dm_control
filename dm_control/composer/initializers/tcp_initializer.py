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

"""An initializer that sets the pose of a hand's tool center point."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import variation
from dm_control.entities.manipulators import base
from six.moves import range


_REJECTION_SAMPLING_FAILED = (
    'Failed to find a valid initial configuration for the robot after '
    '{max_rejection_samples} TCP poses sampled and up to {max_ik_attempts} '
    'initial joint configurations per pose.')


class ToolCenterPointInitializer(composer.Initializer):
  """An initializer that sets the position of a hand's tool center point.

  This initializer calls the RobotArm's internal method to try and set the
  hand's TCP to a randomized Cartesian position within the specified bound.
  By default the initializer performs rejection sampling in order to avoid
  poses that result in "relevant collisions", which are defined as:

  * Collisions between links of the robot arm
  * Collisions between the arm and the hand
  * Collisions between either the arm or hand and an external body without a
    free joint
  """

  def __init__(self,
               hand,
               arm,
               position,
               quaternion=base.DOWN_QUATERNION,
               ignore_collisions=False,
               max_ik_attempts=10,
               max_rejection_samples=10):
    """Initializes this ToolCenterPointInitializer.

    Args:
      hand: Either a `base.RobotHand` instance or None, in which case
        `arm.wrist_site` is used as the TCP site in place of
        `hand.tool_center_point`.
      arm: A `base.RobotArm` instance.
      position: A single fixed Cartesian position, or a `Variation`
        object that generates Cartesian positions. If a fixed sequence of
        positions for multiple props is desired, use
        `variation.deterministic.Sequence`.
      quaternion: (optional) A single fixed unit quaternion, or a
        `composer.Variation` object that generates unit quaternions. If a fixed
        sequence of quaternions for ultiple props is desired, use
        `variation.deterministic.Sequence`.
      ignore_collisions: (optional) If True all collisions are ignored, i.e.
        rejection sampling is disabled.
      max_ik_attempts: (optional) Maximum number of attempts for the inverse
        kinematics solver to find a solution satisfying `target_pos` and
        `target_quat`. These are attempts per rejection sample. If more than
        one attempt is performed, the joint configuration will be randomized
        before the second trial. To avoid randomizing joint positions, set this
        parameter to 1.
      max_rejection_samples (optional): Maximum number of TCP target poses to
        sample while attempting to find a non-colliding configuration. For each
        sampled pose, up to `max_ik_attempts` may be performed in order to find
        an IK solution satisfying this pose.
    """
    super(ToolCenterPointInitializer, self).__init__()
    self._arm = arm
    self._hand = hand
    self._position = position
    self._quaternion = quaternion
    self._ignore_collisions = ignore_collisions
    self._max_ik_attempts = max_ik_attempts
    self._max_rejection_samples = max_rejection_samples

  def _has_relevant_collisions(self, physics):
    mjcf_root = self._arm.mjcf_model.root_model
    all_geoms = mjcf_root.find_all('geom')
    free_body_geoms = set()
    for body in mjcf_root.worldbody.get_children('body'):
      if mjcf.get_freejoint(body):
        free_body_geoms.update(body.find_all('geom'))

    arm_model = self._arm.mjcf_model
    hand_model = None
    if self._hand is not None:
      hand_model = self._hand.mjcf_model

    def is_robot(geom):
      return geom.root is arm_model or geom.root is hand_model

    def is_external_body_without_freejoint(geom):
      return not (is_robot(geom) or geom in free_body_geoms)

    for contact in physics.data.contact:
      geom_1 = all_geoms[contact.geom1]
      geom_2 = all_geoms[contact.geom2]
      if contact.dist > 0:
        # Ignore "contacts" with positive distance (i.e. not actually touching).
        continue
      if (
          # Include arm-arm and arm-hand self-collisions (but not hand-hand).
          (geom_1.root is arm_model and geom_2.root is arm_model) or
          (geom_1.root is arm_model and geom_2.root is hand_model) or
          (geom_1.root is hand_model and geom_2.root is arm_model) or
          # Include collisions between the arm or hand and an external body
          # provided that the external body does not have a freejoint.
          (is_robot(geom_1) and is_external_body_without_freejoint(geom_2)) or
          (is_external_body_without_freejoint(geom_1) and is_robot(geom_2))):
        return True
    return False

  def __call__(self, physics, random_state):
    """Sets initial tool center point pose via inverse kinematics.

    Args:
      physics: An `mjcf.Physics` instance.
      random_state: An `np.random.RandomState` instance.

    Raises:
      RuntimeError: If a collision-free pose could not be found within
        `max_ik_attempts`.
    """
    if self._hand is not None:
      target_site = self._hand.tool_center_point
    else:
      target_site = self._arm.wrist_site

    initial_qpos = physics.bind(self._arm.joints).qpos.copy()

    for _ in range(self._max_rejection_samples):
      target_pos = variation.evaluate(self._position,
                                      random_state=random_state)
      target_quat = variation.evaluate(self._quaternion,
                                       random_state=random_state)
      success = self._arm.set_site_to_xpos(
          physics=physics, random_state=random_state, site=target_site,
          target_pos=target_pos, target_quat=target_quat,
          max_ik_attempts=self._max_ik_attempts)

      if success:
        physics.forward()  # Recalculate contacts.
        if (self._ignore_collisions
            or not self._has_relevant_collisions(physics)):
          return

      # If IK failed to find a solution for this target pose, or if the solution
      # resulted in contacts, then reset the arm joints to their original
      # positions and try again with a new target.
      physics.bind(self._arm.joints).qpos = initial_qpos

    raise RuntimeError(_REJECTION_SAMPLING_FAILED.format(
        max_rejection_samples=self._max_rejection_samples,
        max_ik_attempts=self._max_ik_attempts))
