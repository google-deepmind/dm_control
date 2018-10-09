# Copyright 2018 The dm_control Authors.
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

"""Tests for distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from dm_control.composer.variation import distributions
import numpy as np
from six.moves import range

RANDOM_SEED = 123
NUM_ITERATIONS = 100


def _make_random_state():
  return np.random.RandomState(RANDOM_SEED)


class DistributionsTest(absltest.TestCase):

  def setUp(self):
    self._variation_random_state = _make_random_state()
    self._np_random_state = _make_random_state()

  def testUniform(self):
    lower, upper = [2, 3, 4], [5, 6, 7]
    variation = distributions.Uniform(low=lower, high=upper)
    for _ in range(NUM_ITERATIONS):
      np.testing.assert_array_equal(
          variation(random_state=self._variation_random_state),
          self._np_random_state.uniform(lower, upper))

  def testUniformChoice(self):
    choices = ['apple', 'banana', 'cherry']
    variation = distributions.UniformChoice(choices)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.choice(choices))

  def testUniformPointOnSphere(self):
    variation = distributions.UniformPointOnSphere()
    samples = []
    for _ in range(NUM_ITERATIONS):
      sample = variation(random_state=self._variation_random_state)
      self.assertEqual(sample.size, 3)
      np.testing.assert_approx_equal(np.linalg.norm(sample), 1.0)
      samples.append(sample)
    # Make sure that none of the samples are the same.
    self.assertEqual(
        len(set(np.reshape(np.asarray(samples), -1))), 3 * NUM_ITERATIONS)

  def testNormal(self):
    loc, scale = 1, 2
    variation = distributions.Normal(loc=loc, scale=scale)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.normal(loc, scale))

  def testExponential(self):
    scale = 3
    variation = distributions.Exponential(scale=scale)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.exponential(scale))

  def testPoisson(self):
    lam = 4
    variation = distributions.Poisson(lam=lam)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.poisson(lam))

if __name__ == '__main__':
  absltest.main()
