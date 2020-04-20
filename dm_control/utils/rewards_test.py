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

"""Tests for dm_control.utils.rewards."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized

from dm_control.utils import rewards

import numpy as np


_INPUT_VECTOR_SIZE = 10
EPS = np.finfo(np.double).eps
INF = float("inf")


class ToleranceTest(parameterized.TestCase):

  @parameterized.parameters((0.5, 0.95), (1e12, 1-EPS), (1e12, EPS),
                            (EPS, 1-EPS), (EPS, EPS))
  def test_tolerance_sigmoid_parameterisation(self, margin, value_at_margin):
    actual = rewards.tolerance(x=margin, bounds=(0, 0), margin=margin,
                               value_at_margin=value_at_margin)
    self.assertAlmostEqual(actual, value_at_margin)

  @parameterized.parameters(("gaussian",), ("hyperbolic",), ("long_tail",),
                            ("cosine",), ("tanh_squared",), ("linear",),
                            ("quadratic",), ("reciprocal",))
  def test_tolerance_sigmoids(self, sigmoid):
    margins = [0.01, 1.0, 100, 10000]
    values_at_margin = [0.1, 0.5, 0.9]
    bounds_list = [(0, 0), (-1, 1), (-np.pi, np.pi), (-100, 100)]
    for bounds in bounds_list:
      for margin in margins:
        for value_at_margin in values_at_margin:
          upper_margin = bounds[1]+margin
          value = rewards.tolerance(x=upper_margin, bounds=bounds,
                                    margin=margin,
                                    value_at_margin=value_at_margin,
                                    sigmoid=sigmoid)
          self.assertAlmostEqual(value, value_at_margin, delta=np.sqrt(EPS))
          lower_margin = bounds[0]-margin
          value = rewards.tolerance(x=lower_margin, bounds=bounds,
                                    margin=margin,
                                    value_at_margin=value_at_margin,
                                    sigmoid=sigmoid)
          self.assertAlmostEqual(value, value_at_margin, delta=np.sqrt(EPS))

  @parameterized.parameters((-1, 0), (-0.5, 0.1), (0, 1), (0.5, 0.1), (1, 0))
  def test_tolerance_margin_loss_shape(self, x, expected):
    actual = rewards.tolerance(x=x, bounds=(0, 0), margin=0.5,
                               value_at_margin=0.1)
    self.assertAlmostEqual(actual, expected, delta=1e-3)

  def test_tolerance_vectorization(self):
    bounds = (-.1, .1)
    margin = 0.2
    x_array = np.random.randn(2, 3, 4)
    value_array = rewards.tolerance(x=x_array, bounds=bounds, margin=margin)
    self.assertEqual(x_array.shape, value_array.shape)
    for i, x in enumerate(x_array.ravel()):
      value = rewards.tolerance(x=x, bounds=bounds, margin=margin)
      self.assertEqual(value, value_array.ravel()[i])

  # pylint: disable=bad-whitespace
  @parameterized.parameters(
      # Exact target.
      (0,    (0, 0), 1),
      (EPS,  (0, 0), 0),
      (-EPS, (0, 0), 0),
      # Interval with one open end.
      (0,    (0, INF), 1),
      (EPS,  (0, INF), 1),
      (-EPS, (0, INF), 0),
      # Closed interval.
      (0,     (0, 1), 1),
      (EPS,   (0, 1), 1),
      (-EPS,  (0, 1), 0),
      (1,     (0, 1), 1),
      (1+EPS, (0, 1), 0))
  def test_tolerance_bounds(self, x, bounds, expected):
    actual = rewards.tolerance(x, bounds=bounds, margin=0)
    self.assertEqual(actual, expected)  # Should be exact, since margin == 0.

  def test_tolerance_incorrect_bounds_order(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Lower bound must be <= upper bound."):
      rewards.tolerance(0, bounds=(1, 0), margin=0.05)

  def test_tolerance_negative_margin(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "`margin` must be non-negative."):
      rewards.tolerance(0, bounds=(0, 1), margin=-0.05)

  def test_tolerance_bad_value_at_margin(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "`value_at_1` must be strictly between 0 and 1, got 0."):
      rewards.tolerance(0, bounds=(0, 1), margin=1, value_at_margin=0)

  def test_tolerance_unknown_sigmoid(self):
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Unknown sigmoid type 'unsupported_sigmoid'."):
      rewards.tolerance(0, bounds=(0, 1), margin=.1,
                        sigmoid="unsupported_sigmoid")

if __name__ == "__main__":
  absltest.main()
