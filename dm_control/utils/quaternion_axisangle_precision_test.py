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

"""Regression tests for small quaternion rotations."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.utils import transformations
import mujoco
import numpy as np


class QuaternionAxisanglePrecisionTest(parameterized.TestCase):

  @parameterized.product(angle=(1e-9, 1e-8, 1e-7, 0.1, 1.5, 3.0), sign=(1, -1))
  def test_small_and_regular_rotations_match_their_axis_angle(self, angle, sign):
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14.0)
    quat = sign * np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]
    before = quat.copy()
    actual = transformations.quat_to_axisangle(quat)
    np.testing.assert_allclose(actual, angle * axis, rtol=1e-12, atol=1e-16)
    np.testing.assert_array_equal(quat, before)

  @parameterized.product(angle=(1e-9, 1e-8, 0.2, 2.0, 3.5, 5.0), sign=(1, -1))
  def test_matches_native_mujoco_conversion(self, angle, sign):
    axis = np.array([2.0, 1.0, -2.0]) / 3.0
    quat = np.empty(4)
    mujoco.mju_axisAngle2Quat(quat, axis, angle)
    quat *= sign
    expected = np.empty(3)
    mujoco.mju_quat2Vel(expected, quat, 1.0)
    actual = transformations.quat_to_axisangle(quat)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(
        transformations.quat_to_mat(transformations.axisangle_to_quat(actual)),
        transformations.quat_to_mat(quat),
        rtol=1e-12,
        atol=1e-15,
    )

  @parameterized.parameters(1, -1)
  def test_identity_and_subthreshold_rotations_keep_zero_output(self, sign):
    for angle in (0.0, 1e-12):
      with self.subTest(angle=angle):
        quat = sign * np.array([np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0])
        np.testing.assert_array_equal(
            transformations.quat_to_axisangle(quat), np.zeros(3)
        )

  def test_half_turn_convention_and_invalid_scalar_validation_are_preserved(self):
    for scalar in (0.0, np.cos(np.pi / 2)):
      with self.subTest(half_turn_scalar=scalar):
        np.testing.assert_allclose(
            transformations.quat_to_axisangle(np.array([scalar, 1.0, 0.0, 0.0])),
            [-np.pi, 0.0, 0.0],
        )
    for scalar in (-1.01, 1.01):
      with self.subTest(scalar=scalar), self.assertRaises(ValueError):
        transformations.quat_to_axisangle(np.array([scalar, 0.0, 0.0, 0.0]))


if __name__ == "__main__":
  absltest.main()
