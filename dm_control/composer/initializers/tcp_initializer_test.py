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

import functools

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.initializers import tcp_initializer
from dm_control.entities import props
from dm_control.entities.manipulators import kinova
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib


class TcpInitializerTest(parameterized.TestCase):

  def make_model(self, with_hand=True):
    arm = kinova.JacoArm()
    arena = composer.Arena()
    arena.attach(arm)
    if with_hand:
      hand = kinova.JacoHand()
      arm.attach(hand)
    else:
      hand = None
    return arena, arm, hand

  def assertTargetPoseAchieved(self, frame_binding, target_pos, target_quat):
    np.testing.assert_array_almost_equal(target_pos, frame_binding.xpos)
    target_xmat = np.empty(9, np.double)
    mjlib.mju_quat2Mat(target_xmat, target_quat / np.linalg.norm(target_quat))
    np.testing.assert_array_almost_equal(target_xmat, frame_binding.xmat)

  def assertEntitiesInContact(self, physics, first, second):
    first_geom_ids = physics.bind(
        first.mjcf_model.find_all('geom')).element_id
    second_geom_ids = physics.bind(
        second.mjcf_model.find_all('geom')).element_id
    contact = physics.data.contact
    first_to_second = (np.isin(contact.geom1, first_geom_ids).ravel() &
                       np.isin(contact.geom2, second_geom_ids).ravel())
    second_to_first = (np.isin(contact.geom1, second_geom_ids).ravel() &
                       np.isin(contact.geom2, first_geom_ids).ravel())
    touching = contact.dist <= 0
    valid_contact = touching & (first_to_second | second_to_first)
    self.assertTrue(np.any(valid_contact), msg='Entities are not in contact.')

  @parameterized.parameters([
      dict(target_pos=np.array([0.1, 0.2, 0.3]),
           target_quat=np.array([0., 1., 1., 0.]),
           with_hand=True),
      dict(target_pos=np.array([0., -0.1, 0.5]),
           target_quat=np.array([1., 1., 0., 0.]),
           with_hand=False),
  ])
  def test_initialize_to_fixed_pose(self, target_pos, target_quat, with_hand):
    arena, arm, hand = self.make_model(with_hand=with_hand)
    initializer = tcp_initializer.ToolCenterPointInitializer(
        hand=hand, arm=arm, position=target_pos, quaternion=target_quat)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    initializer(physics=physics, random_state=np.random.RandomState(0))
    site = hand.tool_center_point if with_hand else arm.wrist_site
    self.assertTargetPoseAchieved(physics.bind(site), target_pos, target_quat)

  def test_exception_if_hand_colliding_with_fixed_body(self):
    arena, arm, hand = self.make_model()
    target_pos = np.array([0.1, 0.2, 0.3])
    target_quat = np.array([0., 1., 1., 0.])
    max_rejection_samples = 10
    max_ik_attempts = 5

    # Place a fixed obstacle at the target location so that the TCP can't reach
    # the target without colliding with it.
    obstacle = props.Primitive(geom_type='sphere', size=[0.3])
    attachment_frame = arena.attach(obstacle)
    attachment_frame.pos = target_pos
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    make_initializer = functools.partial(
        tcp_initializer.ToolCenterPointInitializer,
        hand=hand,
        arm=arm,
        position=target_pos,
        quaternion=target_quat,
        max_ik_attempts=max_ik_attempts,
        max_rejection_samples=max_rejection_samples)

    initializer = make_initializer()
    with self.assertRaisesWithLiteralMatch(
        composer.EpisodeInitializationError,
        tcp_initializer._REJECTION_SAMPLING_FAILED.format(
            max_rejection_samples=max_rejection_samples,
            max_ik_attempts=max_ik_attempts)):
      initializer(physics=physics, random_state=np.random.RandomState(0))

    # The initializer should succeed if we ignore collisions.
    initializer_ignore_collisions = make_initializer(ignore_collisions=True)
    initializer_ignore_collisions(physics=physics,
                                  random_state=np.random.RandomState(0))
    self.assertTargetPoseAchieved(
        physics.bind(hand.tool_center_point), target_pos, target_quat)

    # Confirm that the obstacle and the hand are in contact.
    self.assertEntitiesInContact(physics, hand, obstacle)

  @parameterized.named_parameters([
      dict(testcase_name='between_arm_and_arm', with_hand=False),
      dict(testcase_name='between_arm_and_hand', with_hand=True),
  ])
  def test_exception_if_self_collision(self, with_hand):
    arena, arm, hand = self.make_model(with_hand=with_hand)
    # This pose places the wrist or hand partially inside the base of the arm.
    target_pos = np.array([0., 0.1, 0.1])
    target_quat = np.array([-1., 1., 0., 0.])
    max_rejection_samples = 10
    max_ik_attempts = 5
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    make_initializer = functools.partial(
        tcp_initializer.ToolCenterPointInitializer,
        hand=hand,
        arm=arm,
        position=target_pos,
        quaternion=target_quat,
        max_ik_attempts=max_ik_attempts,
        max_rejection_samples=max_rejection_samples)

    initializer = make_initializer()
    with self.assertRaisesWithLiteralMatch(
        composer.EpisodeInitializationError,
        tcp_initializer._REJECTION_SAMPLING_FAILED.format(
            max_rejection_samples=max_rejection_samples,
            max_ik_attempts=max_ik_attempts)):
      initializer(physics=physics, random_state=np.random.RandomState(0))

    # The initializer should succeed if we ignore collisions.
    initializer_ignore_collisions = make_initializer(ignore_collisions=True)
    initializer_ignore_collisions(physics=physics,
                                  random_state=np.random.RandomState(0))
    site = hand.tool_center_point if with_hand else arm.wrist_site
    self.assertTargetPoseAchieved(
        physics.bind(site), target_pos, target_quat)

    # Confirm that there is self-collision.
    self.assertEntitiesInContact(physics, arm, hand if with_hand else arm)

  def test_ignore_robot_collision_with_free_body(self):
    arena, arm, hand = self.make_model()
    target_pos = np.array([0.1, 0.2, 0.3])
    target_quat = np.array([0., 1., 1., 0.])

    # The obstacle is still placed at the target location, but this time it has
    # a freejoint and is held in place by a weld constraint.
    obstacle = props.Primitive(geom_type='sphere', size=[0.3], pos=target_pos)
    attachment_frame = arena.add_free_entity(obstacle)
    attachment_frame.pos = target_pos
    arena.mjcf_model.equality.add(
        'weld', body1=attachment_frame,
        relpose=np.hstack([target_pos, [1., 0., 0., 0.]]))
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    initializer = tcp_initializer.ToolCenterPointInitializer(
        hand=hand,
        arm=arm,
        position=target_pos,
        quaternion=target_quat)

    # Check that the initializer succeeds.
    initializer(physics=physics, random_state=np.random.RandomState(0))
    self.assertTargetPoseAchieved(
        physics.bind(hand.tool_center_point), target_pos, target_quat)

    # Confirm that the obstacle and the hand are in contact.
    self.assertEntitiesInContact(physics, hand, obstacle)

  def test_ignore_collision_not_involving_robot(self):
    arena, arm, hand = self.make_model()
    target_pos = np.array([0.1, 0.2, 0.3])
    target_quat = np.array([0., 1., 1., 0.])

    # Add two boxes that are always in contact with each other, but never with
    # the arm or hand (since they are not within reach).
    side_length = 0.1
    x_offset = 10.
    bottom_box = props.Primitive(
        geom_type='box', size=[side_length]*3, pos=[x_offset, 0, 0])
    top_box = props.Primitive(
        geom_type='box', size=[side_length]*3, pos=[x_offset, 0, 2*side_length])
    arena.attach(bottom_box)
    arena.add_free_entity(top_box)

    initializer = tcp_initializer.ToolCenterPointInitializer(
        hand=hand,
        arm=arm,
        position=target_pos,
        quaternion=target_quat)

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    # Confirm that there are actually contacts between the two boxes.
    self.assertEntitiesInContact(physics, bottom_box, top_box)

    # Check that the initializer still succeeds.
    initializer(physics=physics, random_state=np.random.RandomState(0))
    self.assertTargetPoseAchieved(
        physics.bind(hand.tool_center_point), target_pos, target_quat)

if __name__ == '__main__':
  absltest.main()
