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

"""Tests for the Jaco arm class."""

import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.entities.manipulators import kinova
from dm_control.entities.manipulators.kinova import jaco_arm
from dm_control.mujoco.wrapper import mjbindings
import numpy as np

mjlib = mjbindings.mjlib


class JacoArmTest(parameterized.TestCase):

  def test_can_compile_and_step_model(self):
    arm = kinova.JacoArm()
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    physics.step()

  def test_can_attach_hand(self):
    arm = kinova.JacoArm()
    hand = kinova.JacoHand()
    arm.attach(hand)
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    physics.step()

  # TODO(b/159974149): Investigate why the mass does not match the datasheet.
  @unittest.expectedFailure
  def test_mass(self):
    arm = kinova.JacoArm()
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    mass = physics.bind(arm.mjcf_model.worldbody).subtreemass
    expected_mass = 4.4
    self.assertAlmostEqual(mass, expected_mass)

  @parameterized.parameters([
      dict(actuator_index=0,
           control_input=0,
           expected_velocity=0.),
      dict(actuator_index=0,
           control_input=jaco_arm._LARGE_JOINT_MAX_VELOCITY,
           expected_velocity=jaco_arm._LARGE_JOINT_MAX_VELOCITY),
      dict(actuator_index=4,
           control_input=jaco_arm._SMALL_JOINT_MAX_VELOCITY,
           expected_velocity=jaco_arm._SMALL_JOINT_MAX_VELOCITY),
      dict(actuator_index=0,
           control_input=-jaco_arm._LARGE_JOINT_MAX_VELOCITY,
           expected_velocity=-jaco_arm._LARGE_JOINT_MAX_VELOCITY),
      dict(actuator_index=0,
           control_input=2*jaco_arm._LARGE_JOINT_MAX_VELOCITY,  # Test clipping
           expected_velocity=jaco_arm._LARGE_JOINT_MAX_VELOCITY),
  ])
  def test_velocity_actuation(
      self, actuator_index, control_input, expected_velocity):
    arm = kinova.JacoArm()
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    actuator = arm.actuators[actuator_index]
    bound_actuator = physics.bind(actuator)
    bound_joint = physics.bind(actuator.joint)
    acceleration_threshold = 1e-6
    with physics.model.disable('contact', 'gravity'):
      bound_actuator.ctrl = control_input
      # Step until the joint has stopped accelerating.
      while abs(bound_joint.qacc) > acceleration_threshold:
        physics.step()
      self.assertAlmostEqual(bound_joint.qvel[0], expected_velocity, delta=0.01)

  @parameterized.parameters([
      dict(joint_index=0, min_expected_torque=1.7, max_expected_torque=5.2),
      dict(joint_index=5, min_expected_torque=0.8, max_expected_torque=7.0)])
  def test_backdriving_torque(
      self, joint_index, min_expected_torque, max_expected_torque):
    arm = kinova.JacoArm()
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    bound_joint = physics.bind(arm.joints[joint_index])
    torque = min_expected_torque * 0.8
    velocity_threshold = 0.1*2*np.pi/60.  # 0.1 RPM
    torque_increment = 0.01
    seconds_per_torque_increment = 1.
    max_torque = max_expected_torque * 1.1
    while torque < max_torque:
      # Ensure that no other forces are acting on the arm.
      with physics.model.disable('gravity', 'contact', 'actuation'):
        # Reset the simulation so that the initial velocity is zero.
        physics.reset()
        bound_joint.qfrc_applied = torque
        while physics.time() < seconds_per_torque_increment:
          physics.step()
      if bound_joint.qvel[0] >= velocity_threshold:
        self.assertBetween(torque, min_expected_torque, max_expected_torque)
        return
      # If we failed to accelerate the joint to the target velocity within the
      # time limit we'll reset the simulation and increase the torque.
      torque += torque_increment
    self.fail('Torque of {} Nm insufficient to backdrive joint.'.format(torque))

  @parameterized.parameters([
      dict(joint_pos=0., expected_obs=[0., 1.]),
      dict(joint_pos=-0.5*np.pi, expected_obs=[-1., 0.]),
      dict(joint_pos=np.pi, expected_obs=[0., -1.]),
      dict(joint_pos=10*np.pi, expected_obs=[0., 1.])])
  def test_joints_pos_observables(self, joint_pos, expected_obs):
    joint_index = 0
    arm = kinova.JacoArm()
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    physics.bind(arm.joints).qpos[joint_index] = joint_pos
    actual_obs = arm.observables.joints_pos(physics)[joint_index]
    np.testing.assert_array_almost_equal(expected_obs, actual_obs)

  @parameterized.parameters(
      dict(joint_index=idx, applied_torque=t)
      for idx, t in itertools.product([0, 2, 4], [0., -6.8, 30.5]))
  def test_joints_torque_observables(self, joint_index, applied_torque):
    arm = kinova.JacoArm()
    joint = arm.joints[joint_index]
    physics = mjcf.Physics.from_mjcf_model(arm.mjcf_model)
    with physics.model.disable('gravity', 'limit', 'contact', 'actuation'):
      # Apply a cartesian torque to the body containing the joint. We use
      # `xfrc_applied` rather than `qfrc_applied` because forces in
      # `qfrc_applied` are not measured by the torque sensor).
      physics.bind(joint.parent).xfrc_applied[3:] = (
          applied_torque * physics.bind(joint).xaxis)
      observed_torque = arm.observables.joints_torque(physics)[joint_index]
    # Note the change in sign, since the sensor measures torques in the
    # child->parent direction.
    self.assertAlmostEqual(observed_torque, -applied_torque, delta=0.1)


class JacoHandTest(parameterized.TestCase):

  def test_can_compile_and_step_model(self):
    hand = kinova.JacoHand()
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    physics.step()

  # TODO(b/159974149): Investigate why the mass does not match the datasheet.
  @unittest.expectedFailure
  def test_hand_mass(self):
    hand = kinova.JacoHand()
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    mass = physics.bind(hand.mjcf_model.worldbody).subtreemass
    expected_mass = 0.727
    self.assertAlmostEqual(mass, expected_mass)

  def test_grip_force(self):
    arena = composer.Arena()
    hand = kinova.JacoHand()
    arena.attach(hand)

    # A sphere with a touch sensor for measuring grip force.
    prop_model = mjcf.RootElement(model='grip_target')
    prop_model.worldbody.add('geom', type='sphere', size=[0.02])
    touch_site = prop_model.worldbody.add('site', type='sphere', size=[0.025])
    touch_sensor = prop_model.sensor.add('touch', site=touch_site)
    prop = composer.ModelWrapperEntity(prop_model)

    # Add some slide joints to allow movement of the target in the XY plane.
    # This helps the contact solver to converge more reliably.
    prop_frame = arena.attach(prop)
    prop_frame.add('joint', name='slide_x', type='slide', axis=(1, 0, 0))
    prop_frame.add('joint', name='slide_y', type='slide', axis=(0, 1, 0))

    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    bound_pinch_site = physics.bind(hand.pinch_site)
    bound_actuators = physics.bind(hand.actuators)
    bound_joints = physics.bind(hand.joints)
    bound_touch = physics.bind(touch_sensor)

    # Position the grip target at the pinch site.
    prop.set_pose(physics, position=bound_pinch_site.xpos)

    # Close the fingers with as much force as the actuators will allow.
    bound_actuators.ctrl = bound_actuators.ctrlrange[:, 1]

    # Run the simulation forward until the joints stop moving.
    physics.step()
    qvel_thresh = 1e-3  # radians / s
    while max(abs(bound_joints.qvel)) > qvel_thresh:
      physics.step()
    expected_min_grip_force = 20.
    expected_max_grip_force = 30.
    grip_force = bound_touch.sensordata
    self.assertBetween(
        grip_force, expected_min_grip_force, expected_max_grip_force,
        msg='Expected grip force to be between {} and {} N, got {} N.'.format(
            expected_min_grip_force, expected_max_grip_force, grip_force))

  @parameterized.parameters([dict(opening=True), dict(opening=False)])
  def test_finger_travel_time(self, opening):
    hand = kinova.JacoHand()
    physics = mjcf.Physics.from_mjcf_model(hand.mjcf_model)
    bound_actuators = physics.bind(hand.actuators)
    bound_joints = physics.bind(hand.joints)
    min_ctrl, max_ctrl = bound_actuators.ctrlrange.T
    min_qpos, max_qpos = bound_joints.range.T

    # Measure the time taken for the finger joints to traverse 99.9% of their
    # total range.
    qpos_tol = 1e-3 * (max_qpos - min_qpos)
    if opening:
      hand.set_grasp(physics=physics, close_factors=1.)  # Fully closed.
      np.testing.assert_array_almost_equal(bound_joints.qpos, max_qpos)
      target_pos = min_qpos  # Fully open.
      ctrl = min_ctrl  # Open the fingers as fast as the actuators will allow.
    else:
      hand.set_grasp(physics=physics, close_factors=0.)  # Fully open.
      np.testing.assert_array_almost_equal(bound_joints.qpos, min_qpos)
      target_pos = max_qpos  # Fully closed.
      ctrl = max_ctrl  # Close the fingers as fast as the actuators will allow.

    # Run the simulation until all joints have reached their target positions.
    bound_actuators.ctrl = ctrl
    while np.any(abs(bound_joints.qpos - target_pos) > qpos_tol):
      with physics.model.disable('gravity'):
        physics.step()
    expected_travel_time = 1.2  # Seconds.
    self.assertAlmostEqual(physics.time(), expected_travel_time, delta=0.1)

  @parameterized.parameters([
      dict(pos=np.r_[0., 0., 0.3], quat=np.r_[0., 1., 0., 1.]),
      dict(pos=np.r_[0., -0.1, 0.5], quat=np.r_[1., 1., 0., 0.]),
  ])
  def test_pinch_site_observables(self, pos, quat):
    arm = kinova.JacoArm()
    hand = kinova.JacoHand()
    arena = composer.Arena()
    arm.attach(hand)
    arena.attach(arm)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)

    # Normalize the quaternion.
    quat /= np.linalg.norm(quat)

    # Drive the arm so that the pinch site is at the desired position and
    # orientation.
    success = arm.set_site_to_xpos(
        physics=physics,
        random_state=np.random.RandomState(0),
        site=hand.pinch_site,
        target_pos=pos,
        target_quat=quat)
    self.assertTrue(success)

    # Check that the observations are as expected.
    observed_pos = hand.observables.pinch_site_pos(physics)
    np.testing.assert_allclose(observed_pos, pos, atol=1e-3)

    observed_rmat = hand.observables.pinch_site_rmat(physics).reshape(3, 3)
    expected_rmat = np.empty((3, 3), np.double)
    mjlib.mju_quat2Mat(expected_rmat.ravel(), quat)
    difference_rmat = observed_rmat.dot(expected_rmat.T)
    angular_difference = np.arccos((np.trace(difference_rmat) - 1) / 2)
    self.assertLess(angular_difference, 1e-3)


if __name__ == '__main__':
  absltest.main()
