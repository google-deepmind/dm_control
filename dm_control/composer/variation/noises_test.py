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

"""Tests for noises."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import noises

NUM_ITERATIONS = 100


class NoisesTest(parameterized.TestCase):

  @parameterized.parameters(False, True)
  def testAdditive(self, use_constant_variation_object):
    amount = 2
    if use_constant_variation_object:
      variation = noises.Additive(deterministic.Constant(amount))
    else:
      variation = noises.Additive(amount)
    initial_value = 0
    current_value = initial_value
    for _ in range(NUM_ITERATIONS):
      current_value = variation(
          initial_value=initial_value, current_value=current_value)
      self.assertEqual(current_value, initial_value + amount)

  @parameterized.parameters(False, True)
  def testAdditiveCumulative(self, use_constant_variation_object):
    amount = 3
    if use_constant_variation_object:
      variation = noises.Additive(
          deterministic.Constant(amount), cumulative=True)
    else:
      variation = noises.Additive(amount, cumulative=True)
    initial_value = 1
    current_value = initial_value
    for i in range(NUM_ITERATIONS):
      current_value = variation(
          initial_value=initial_value, current_value=current_value)
      self.assertEqual(current_value, initial_value + amount * (i + 1))

  @parameterized.parameters(False, True)
  def testMultiplicative(self, use_constant_variation_object):
    amount = 23
    if use_constant_variation_object:
      variation = noises.Multiplicative(deterministic.Constant(amount))
    else:
      variation = noises.Multiplicative(amount)
    initial_value = 3
    current_value = initial_value
    for _ in range(NUM_ITERATIONS):
      current_value = variation(
          initial_value=initial_value, current_value=current_value)
      self.assertEqual(current_value, initial_value * amount)

  @parameterized.parameters(False, True)
  def testMultiplicativeCumulative(self, use_constant_variation_object):
    amount = 2
    if use_constant_variation_object:
      variation = noises.Multiplicative(
          deterministic.Constant(amount), cumulative=True)
    else:
      variation = noises.Multiplicative(amount, cumulative=True)
    initial_value = 3
    current_value = initial_value
    for i in range(NUM_ITERATIONS):
      current_value = variation(
          initial_value=initial_value, current_value=current_value)
      self.assertEqual(current_value, initial_value * amount ** (i + 1))

if __name__ == '__main__':
  absltest.main()
