# Copyright 2017 The dm_control Authors.
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

"""Tests for corruptors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized

from dm_control.utils import corruptors

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin


class DelayTest(absltest.TestCase):

  def setUp(self):
    self.n = 10
    self.delay = corruptors.Delay(steps=self.n)
    super(DelayTest, self).setUp()

  def testProcess(self):
    obs = np.array(range(2 * self.n))
    actual_obs = []
    for i in obs:
      actual_obs.append(self.delay(i))
    expected = np.hstack(([.0] * self.n, obs[:self.n]))
    np.testing.assert_array_equal(expected, actual_obs)

    actual_obs = []
    for i in obs:
      actual_obs.append(self.delay(i))
    expected = np.hstack((obs[self.n:], obs[:self.n]))
    np.testing.assert_array_equal(expected, actual_obs)

  def testReset(self):
    obs = np.array(range(2 * self.n))
    for _ in xrange(2):
      actual_obs = []
      for i in obs:
        actual_obs.append(self.delay(i))
      self.delay.reset()

      expected = np.hstack(([.0] * self.n, obs[:self.n]))
      np.testing.assert_array_equal(expected, actual_obs)


class StatelessNoiseTest(absltest.TestCase):

  def testProcess(self):
    c = corruptors.StatelessNoise(noise_function=corruptors.gaussian_noise,
                                  std=1e-3)
    x = np.array([.0] * 3)
    y = np.array([.0] * 3)
    n = 1e3
    for _ in xrange(int(n)):
      y += c(x)
    y /= n
    np.testing.assert_allclose(x, y, atol=1e-4)


class NoiseFunctionTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('1D', np.array([3., 4.]), .1),
      ('2D', np.array([[.0, .1], [1., 2.]]), .4)
  )
  def testGaussianNoise_Shape(self, x, std):
    noisy_x = corruptors.gaussian_noise(x, std)
    self.assertEqual(x.shape, noisy_x.shape)

if __name__ == '__main__':
  absltest.main()
