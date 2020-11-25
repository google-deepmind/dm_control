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

"""Tests for dm_control.locomotion.tasks.transformations."""

import itertools

from absl.testing import absltest
from absl.testing import parameterized

from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import transformations

import numpy as np

mjlib = mjbindings.mjlib

_NUM_RANDOM_SAMPLES = 1000


class TransformationsTest(parameterized.TestCase):

  def __init__(self, *args, **kwargs):
    super(TransformationsTest, self).__init__(*args, **kwargs)
    self._random_state = np.random.RandomState()

  @parameterized.parameters(
      {
          'quat': [-0.41473841, 0.59483601, -0.45089078, 0.52044181],
          'truemat':
              np.array([[0.05167565, -0.10471773, 0.99315851],
                        [-0.96810656, -0.24937912, 0.02407785],
                        [0.24515162, -0.96272751, -0.11426475]])
      },
      {
          'quat': [0.08769298, 0.69897558, 0.02516888, 0.7093022],
          'truemat':
              np.array([[-0.00748615, -0.08921678, 0.9959841],
                        [0.15958651, -0.98335294, -0.08688582],
                        [0.98715556, 0.15829519, 0.02159933]])
      },
      {
          'quat': [0.58847272, 0.44682507, 0.51443343, -0.43520737],
          'truemat':
              np.array([[0.09190557, 0.97193884, 0.21653695],
                        [-0.05249182, 0.22188379, -0.97365918],
                        [-0.99438321, 0.07811829, 0.07141119]])
      },
  )
  def test_quat_to_mat(self, quat, truemat):
    """Tests hard-coded quat-mat pairs generated from mujoco if mj not avail."""
    mat = transformations.quat_to_mat(quat)
    np.testing.assert_allclose(mat[0:3, 0:3], truemat, atol=1e-7)

  def test_quat_to_mat_mujoco_special(self):
    # Test for special values that often cause numerical issues.
    rng = [-np.pi, np.pi / 2, 0, np.pi / 2, np.pi]
    for euler_tup in itertools.product(rng, rng, rng):
      euler_vec = np.array(euler_tup, dtype=np.float)
      mat = transformations.euler_to_rmat(euler_vec, ordering='XYZ')
      quat = transformations.mat_to_quat(mat)
      tr_mat = transformations.quat_to_mat(quat)
      mj_mat = np.zeros(9)
      mjlib.mju_quat2Mat(mj_mat, quat)
      mj_mat = mj_mat.reshape(3, 3)
      np.testing.assert_allclose(tr_mat[0:3, 0:3], mj_mat, atol=1e-10)
      np.testing.assert_allclose(tr_mat[0:3, 0:3], mat, atol=1e-10)

  def test_quat_to_mat_mujoco_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat = self._random_quaternion()
      tr_mat = transformations.quat_to_mat(quat)
      mj_mat = np.zeros(9)
      mjlib.mju_quat2Mat(mj_mat, quat)
      mj_mat = mj_mat.reshape(3, 3)
      np.testing.assert_allclose(tr_mat[0:3, 0:3], mj_mat)

  def test_mat_to_quat_mujoco(self):
    subsamps = 10
    rng = np.linspace(-np.pi, np.pi, subsamps)
    for euler_tup in itertools.product(rng, rng, rng):
      euler_vec = np.array(euler_tup, dtype=np.float)
      mat = transformations.euler_to_rmat(euler_vec, ordering='XYZ')
      mj_quat = np.empty(4, dtype=euler_vec.dtype)
      mjlib.mju_mat2Quat(mj_quat, mat.flatten())
      tr_quat = transformations.mat_to_quat(mat)
      self.assertTrue(
          np.allclose(mj_quat, tr_quat) or np.allclose(mj_quat, -tr_quat))

  @parameterized.parameters(
      {'angles': (0, 0, 0)},
      {'angles': (-0.1, 0.4, -1.3)}
  )
  def test_euler_to_rmat_special(self, angles):
    # Test for special values that often cause numerical issues.
    r1, r2, r3 = angles
    for ordering in transformations._eulermap.keys():
      r = transformations.euler_to_rmat(np.array([r1, r2, r3]), ordering)
      euler_angles = transformations.rmat_to_euler(r, ordering)
      np.testing.assert_allclose(euler_angles, [r1, r2, r3])

  def test_quat_mul_vs_mat_mul_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat1 = self._random_quaternion()
      quat2 = self._random_quaternion()
      rmat1 = transformations.quat_to_mat(quat1)[0:3, 0:3]
      rmat2 = transformations.quat_to_mat(quat2)[0:3, 0:3]
      quat_prod = transformations.quat_mul(quat1, quat2)
      rmat_prod_q = transformations.quat_to_mat(quat_prod)[0:3, 0:3]
      rmat_prod = rmat1.dot(rmat2)
      np.testing.assert_allclose(rmat_prod, rmat_prod_q)

  def test_quat_mul_mujoco_special(self):
    # Test for special values that often cause numerical issues.
    rng = [-np.pi, np.pi / 2, 0, np.pi / 2, np.pi]
    quat1 = np.array([1, 0, 0, 0], dtype=np.float64)
    for euler_tup in itertools.product(rng, rng, rng):
      euler_vec = np.array(euler_tup, dtype=np.float64)
      quat2 = transformations.euler_to_quat(euler_vec, ordering='XYZ')
      quat_prod_tr = transformations.quat_mul(quat1, quat2)
      quat_prod_mj = np.zeros(4)
      mjlib.mju_mulQuat(quat_prod_mj, quat1, quat2)
      np.testing.assert_allclose(quat_prod_tr, quat_prod_mj, atol=1e-14)
      quat1 = quat2

  def test_quat_mul_mujoco_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat1 = self._random_quaternion()
      quat2 = self._random_quaternion()
      quat_prod_tr = transformations.quat_mul(quat1, quat2)
      quat_prod_mj = np.zeros(4)
      mjlib.mju_mulQuat(quat_prod_mj, quat1, quat2)
      np.testing.assert_allclose(quat_prod_tr, quat_prod_mj)

  def test_quat_rotate_mujoco_special(self):
    # Test for special values that often cause numerical issues.
    rng = [-np.pi, np.pi / 2, 0, np.pi / 2, np.pi]
    vec = np.array([1, 0, 0], dtype=np.float64)
    for euler_tup in itertools.product(rng, rng, rng):
      euler_vec = np.array(euler_tup, dtype=np.float64)
      quat = transformations.euler_to_quat(euler_vec, ordering='XYZ')
      rotated_vec_tr = transformations.quat_rotate(quat, vec)
      rotated_vec_mj = np.zeros(3)
      mjlib.mju_rotVecQuat(rotated_vec_mj, vec, quat)
      np.testing.assert_allclose(rotated_vec_tr, rotated_vec_mj, atol=1e-14)

  def test_quat_rotate_mujoco_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      quat = self._random_quaternion()
      vec = self._random_state.rand(3)
      rotated_vec_tr = transformations.quat_rotate(quat, vec)
      rotated_vec_mj = np.zeros(3)
      mjlib.mju_rotVecQuat(rotated_vec_mj, vec, quat)
      np.testing.assert_allclose(rotated_vec_tr, rotated_vec_mj)

  def test_quat_diff_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      source = self._random_quaternion()
      target = self._random_quaternion()
      np.testing.assert_allclose(
          transformations.quat_diff(source, target),
          transformations.quat_mul(transformations.quat_conj(source), target))

  def test_quat_dist_random(self):
    for _ in range(_NUM_RANDOM_SAMPLES):
      # test with normalized quaternions for stability of test
      source = self._random_quaternion()
      target = self._random_quaternion()
      source /= np.linalg.norm(source)
      target /= np.linalg.norm(target)
      self.assertGreater(transformations.quat_dist(source, target), 0)
      np.testing.assert_allclose(
          transformations.quat_dist(source, source), 0, atol=1e-9)

  def _random_quaternion(self):
    rand = self._random_state.rand(3)
    r1 = np.sqrt(1.0 - rand[0])
    r2 = np.sqrt(rand[0])
    pi2 = np.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return np.array(
        (np.cos(t2) * r2, np.sin(t1) * r1, np.cos(t1) * r1, np.sin(t2) * r2),
        dtype=np.float64)


if __name__ == '__main__':
  absltest.main()
