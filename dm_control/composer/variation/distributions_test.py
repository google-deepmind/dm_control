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

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.composer.variation import distributions
import numpy as np

RANDOM_SEED = 123
NUM_ITERATIONS = 100


def _make_random_state():
  return np.random.RandomState(RANDOM_SEED)


class DistributionsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._variation_random_state = _make_random_state()
    self._np_random_state = _make_random_state()

  def testUniform(self):
    lower, upper = [2, 3, 4], [5, 6, 7]
    variation = distributions.Uniform(low=lower, high=upper)
    for _ in range(NUM_ITERATIONS):
      np.testing.assert_array_equal(
          variation(random_state=self._variation_random_state),
          self._np_random_state.uniform(lower, upper))

    self.assertEqual(variation, distributions.Uniform(low=lower, high=upper))
    self.assertNotEqual(variation, distributions.Uniform(low=upper, high=upper))
    self.assertIn('[2, 3, 4]', repr(variation))

  def testUniformChoice(self):
    choices = ['apple', 'banana', 'cherry']
    variation = distributions.UniformChoice(choices)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.choice(choices))

    self.assertIn('banana', repr(variation))

  def testUniformPointOnSphere(self):
    variation = distributions.UniformPointOnSphere()
    samples = []
    for _ in range(NUM_ITERATIONS):
      sample = variation(random_state=self._variation_random_state)
      self.assertEqual(sample.size, 3)
      np.testing.assert_approx_equal(np.linalg.norm(sample), 1.0)
      samples.append(sample)
    # Make sure that none of the samples are the same.
    self.assertLen(set(np.reshape(np.asarray(samples), -1)), 3 * NUM_ITERATIONS)
    self.assertEqual(variation, distributions.UniformPointOnSphere())
    self.assertNotEqual(
        variation, distributions.UniformPointOnSphere(single_sample=True)
    )

  def testNormal(self):
    loc, scale = 1, 2
    variation = distributions.Normal(loc=loc, scale=scale)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.normal(loc, scale))
    self.assertEqual(variation, distributions.Normal(loc=loc, scale=scale))
    self.assertNotEqual(
        variation, distributions.Normal(loc=loc*2, scale=scale)
    )
    self.assertEqual(
        "Normal(args=(), kwargs={'loc': 1, 'scale': 2}, single_sample=False)",
        repr(variation),
    )

  def testExponential(self):
    scale = 3
    variation = distributions.Exponential(scale=scale)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.exponential(scale))
    self.assertEqual(variation, distributions.Exponential(scale=scale))
    self.assertNotEqual(
        variation, distributions.Exponential(scale=scale*2)
    )
    self.assertEqual(
        "Exponential(args=(), kwargs={'scale': 3}, single_sample=False)",
        repr(variation),
    )

  def testPoisson(self):
    lam = 4
    variation = distributions.Poisson(lam=lam)
    for _ in range(NUM_ITERATIONS):
      self.assertEqual(
          variation(random_state=self._variation_random_state),
          self._np_random_state.poisson(lam))
    self.assertEqual(variation, distributions.Poisson(lam=lam))
    self.assertNotEqual(
        variation, distributions.Poisson(lam=lam*2)
    )
    self.assertEqual(
        "Poisson(args=(), kwargs={'lam': 4}, single_sample=False)",
        repr(variation),
    )

  @parameterized.parameters(0, 10)
  def testBiasedRandomWalk(self, timescale):
    stdev = 1.
    variation = distributions.BiasedRandomWalk(stdev=stdev, timescale=timescale)
    sequence = [variation(random_state=self._variation_random_state)
                for _ in range(int(max(timescale, 1)*NUM_ITERATIONS*1000))]
    self.assertAlmostEqual(np.mean(sequence), 0., delta=0.01)
    self.assertAlmostEqual(np.std(sequence), stdev, delta=0.01)

  @parameterized.parameters(
      dict(arg_name='stdev', template=distributions._NEGATIVE_STDEV),
      dict(arg_name='timescale', template=distributions._NEGATIVE_TIMESCALE))
  def testBiasedRandomWalkExceptions(self, arg_name, template):
    bad_value = -1.
    with self.assertRaisesWithLiteralMatch(
        ValueError, template.format(bad_value)):
      _ = distributions.BiasedRandomWalk(**{arg_name: bad_value})

if __name__ == '__main__':
  absltest.main()
