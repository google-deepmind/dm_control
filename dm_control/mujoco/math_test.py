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

"""Tests for index."""
import math
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mujoco import math as mjmath
import numpy as np


class MathTest(parameterized.TestCase):

  def testQuatProd(self):
    np.testing.assert_allclose(
        mjmath.mj_quatprod([0., 1., 0., 0.], [0., 0., 1., 0.]),
        [0., 0., 0., 1.])
    np.testing.assert_allclose(
        mjmath.mj_quatprod([0., 0., 1., 0.], [0., 0., 0., 1.]),
        [0., 1., 0., 0.])
    np.testing.assert_allclose(
        mjmath.mj_quatprod([0., 0., 0., 1.], [0., 1., 0., 0.]),
        [0., 0., 1., 0.])

  def testQuat2Vel(self):
    np.testing.assert_allclose(
        mjmath.mj_quat2vel([0., 1., 0., 0.], 0.1), [math.pi / 0.1, 0., 0.])

  def testQuatNeg(self):
    np.testing.assert_allclose(
        mjmath.mj_quatneg([math.sqrt(0.5), math.sqrt(0.5), 0., 0.]),
        [math.sqrt(0.5), -math.sqrt(0.5), 0., 0.])

  def testQuatDiff(self):
    np.testing.assert_allclose(
        mjmath.mj_quatdiff([0., 1., 0., 0.], [0., 0., 1., 0.]),
        [0., 0., 0., -1.])

  def testEuler2Quat(self):
    np.testing.assert_allclose(
        mjmath.euler2quat(0., 0., 0.), [1., 0., 0., 0.])


if __name__ == '__main__':
  absltest.main()
