# Copyright 2017-2018 The dm_control Authors.
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

"""Tests for inverse_kinematics."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import inverse_kinematics as ik
import numpy as np
import os

mjlib = mjbindings.mjlib

_ARM_XML = assets.get_contents('arm.xml')
_MODEL_WITH_BALL_JOINTS_XML = assets.get_contents('model_with_ball_joints.xml')

_SITE_NAME = 'gripsite'
_JOINTS = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
_TOL = 1.2e-14
_MAX_STEPS = 100
_MAX_RESETS = 10

_TARGETS = [
    # target_pos              # target_quat
    (np.array([0., 0., 0.3]), np.array([0., 1., 0., 1.])),
    (np.array([-0.5, 0., 0.5]), None),
    (np.array([0., 0., 0.8]), np.array([0., 1., 0., 1.])),
    (np.array([0., 0., 0.8]), None),
    (np.array([0., -0.1, 0.5]), None),
    (np.array([0., -0.1, 0.5]), np.array([1., 1., 0., 0.])),
    (np.array([0.5, 0., 0.5]), None),
    (np.array([0.4, 0.1, 0.5]), None),
    (np.array([0.4, 0.1, 0.5]), np.array([1., 0., 0., 0.])),
    (np.array([0., 0., 0.3]), None),
    (np.array([0., 0.5, -0.2]), None),
    (np.array([0.5, 0., 0.3]), np.array([1., 0., 0., 1.])),
    (None, np.array([1., 0., 0., 1.])),
    (None, np.array([0., 1., 0., 1.])),
]
_INPLACE = [False, True]


class _ResetArm:

  def __init__(self, seed=None):
    self._rng = np.random.RandomState(seed)
    self._lower = None
    self._upper = None

  def _cache_bounds(self, physics):
    self._lower, self._upper = physics.named.model.jnt_range[_JOINTS].T
    limited = physics.named.model.jnt_limited[_JOINTS].astype(bool)
    # Positions for hinge joints without limits are sampled between 0 and 2pi
    self._lower[~limited] = 0
    self._upper[~limited] = 2 * np.pi

  def __call__(self, physics):
    if self._lower is None:
      self._cache_bounds(physics)
    # NB: This won't work for joints with > 1 DOF
    new_qpos = self._rng.uniform(self._lower, self._upper)
    physics.named.data.qpos[_JOINTS] = new_qpos


class InverseKinematicsTest(parameterized.TestCase):

  def testQposFromMultipleSitesPose(self):
      dir_path = os.path.dirname(os.path.realpath(__file__))
      model_dir = os.path.join(dir_path, "./testing/assets/task.xml")
      physics = mujoco.Physics.from_xml_path(model_dir)

      target_pos = physics.model.key_mpos[0]
      target_pos = target_pos.reshape((-1, 3))
      target_quat = None

      _SITES_NAMES = []

      body_names = [
        "pelvis",    "head",      "ltoe",  "rtoe",  "lheel",  "rheel",
        "lknee",     "rknee",     "lhand", "rhand", "lelbow", "relbow",
        "lshoulder", "rshoulder", "lhip",  "rhip",
      ]

      for name in body_names:
          _SITES_NAMES.append("tracking[" + name + "]")

      _MAX_STEPS = 5000
      result = ik.qpos_from_site_pose(
          physics=physics,
          sites_names=_SITES_NAMES,
          target_pos=target_pos,
          target_quat=target_quat,
          joint_names=None,
          tol=1e-14,
          regularization_threshold=0.5,
          regularization_strength=1e-2,
          max_update_norm=2.0,
          progress_thresh=5000.0,
          max_steps=_MAX_STEPS,
          inplace=False,
          null_space_method=False
      )

      self.assertLessEqual(result.steps, _MAX_STEPS)
      physics.data.qpos[:] = result.qpos

      save_path = os.path.join(dir_path, "./testing/assets/result_qpos")
      np.save(save_path, result.qpos)
      mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)

      pos = physics.named.data.site_xpos[_SITES_NAMES]
      err_norm = np.linalg.norm(target_pos - pos)
      self.assertLessEqual(err_norm, 0.11)

  @parameterized.parameters(itertools.product(_TARGETS, _INPLACE))
  def testQposFromSitePose(self, target, inplace):
    physics = mujoco.Physics.from_xml_string(_ARM_XML)
    target_pos, target_quat = target
    count = 0
    physics2 = physics.copy(share_model=True)
    resetter = _ResetArm(seed=0)
    while True:
      result = ik.qpos_from_site_pose(
          physics=physics2,
          sites_names=[_SITE_NAME],
          target_pos=target_pos,
          target_quat=target_quat,
          joint_names=_JOINTS,
          tol=_TOL,
          max_steps=_MAX_STEPS,
          inplace=inplace,
      )
      if result.success:
        break
      elif count < _MAX_RESETS:
        resetter(physics2)
        count += 1
      else:
        raise RuntimeError(
            'Failed to find a solution within %i attempts.' % _MAX_RESETS)

    self.assertLessEqual(result.steps, _MAX_STEPS)
    self.assertLessEqual(result.err_norm, _TOL)
    physics.data.qpos[:] = result.qpos
    mjlib.mj_fwdPosition(physics.model.ptr, physics.data.ptr)
    if target_pos is not None:
      pos = physics.named.data.site_xpos[_SITE_NAME]
      np.testing.assert_array_almost_equal(pos, target_pos)
    if target_quat is not None:
      xmat = physics.named.data.site_xmat[_SITE_NAME]
      quat = np.empty_like(target_quat)
      mjlib.mju_mat2Quat(quat, xmat)
      quat /= np.ptp(quat)  # Normalize xquat so that its max-min range is 1
      np.testing.assert_array_almost_equal(quat, target_quat)

  def testNamedJointsWithMultipleDOFs(self):
    """Regression test for b/77506142."""
    physics = mujoco.Physics.from_xml_string(_MODEL_WITH_BALL_JOINTS_XML)
    site_name = 'gripsite'
    joint_names = ['joint_1', 'joint_2']

    # This target position can only be achieved by rotating both ball joints. If
    # DOFs are incorrectly indexed by joint index then only the first two DOFs
    # in the first ball joint can change, and IK will fail to find a solution.
    target_pos = (0.05, 0.05, 0)
    result = ik.qpos_from_site_pose(
        physics=physics,
        sites_names=[site_name],
        target_pos=target_pos,
        joint_names=joint_names,
        tol=_TOL,
        max_steps=_MAX_STEPS,
        inplace=True)

    self.assertTrue(result.success)
    self.assertLessEqual(result.steps, _MAX_STEPS)
    self.assertLessEqual(result.err_norm, _TOL)
    pos = physics.named.data.site_xpos[site_name]
    np.testing.assert_array_almost_equal(pos, target_pos)

    # IK should fail to converge if only the first joint is allowed to move.
    physics.reset()
    result = ik.qpos_from_site_pose(
        physics=physics,
        sites_names=[site_name],
        target_pos=target_pos,
        joint_names=joint_names[:1],
        tol=_TOL,
        max_steps=_MAX_STEPS,
        inplace=True)
    self.assertFalse(result.success)

  @parameterized.named_parameters(
      ('None', None),
      ('list', ['joint_1', 'joint_2']),
      ('tuple', ('joint_1', 'joint_2')),
      ('numpy.array', np.array(['joint_1', 'joint_2'])))
  def testAllowedJointNameTypes(self, joint_names):
    """Test allowed types for joint_names parameter."""
    physics = mujoco.Physics.from_xml_string(_ARM_XML)
    site_name = 'gripsite'
    target_pos = (0.05, 0.05, 0)
    ik.qpos_from_site_pose(
        physics=physics,
        sites_names=[site_name],
        target_pos=target_pos,
        joint_names=joint_names,
        tol=_TOL,
        max_steps=_MAX_STEPS,
        inplace=True)

  @parameterized.named_parameters(
      ('int', 1),
      ('dict', {'joint_1': 1, 'joint_2': 2}),
      ('function', lambda x: x),
  )
  def testDisallowedJointNameTypes(self, joint_names):
    physics = mujoco.Physics.from_xml_string(_ARM_XML)
    site_name = 'gripsite'
    target_pos = (0.05, 0.05, 0)

    expected_message = ik._INVALID_JOINT_NAMES_TYPE.format(type(joint_names))

    with self.assertRaisesWithLiteralMatch(ValueError, expected_message):
      ik.qpos_from_site_pose(
          physics=physics,
          sites_names=[site_name],
          target_pos=target_pos,
          joint_names=joint_names,
          tol=_TOL,
          max_steps=_MAX_STEPS,
          inplace=True)

  def testNoTargetPosOrQuat(self):
    physics = mujoco.Physics.from_xml_string(_ARM_XML)
    site_name = 'gripsite'
    with self.assertRaisesWithLiteralMatch(
        ValueError, ik._REQUIRE_TARGET_POS_OR_QUAT):
      ik.qpos_from_site_pose(
          physics=physics,
          sites_names=[site_name],
          tol=_TOL,
          max_steps=_MAX_STEPS,
          inplace=True)

if __name__ == '__main__':
  absltest.main()
