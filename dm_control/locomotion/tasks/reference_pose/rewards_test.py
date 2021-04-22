# Copyright 2021 The dm_control Authors.
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
"""Tests for dm_control.locomotion.tasks.reference_pose.rewards."""

from absl.testing import absltest
from dm_control.locomotion.tasks.reference_pose import rewards
import numpy as np

WALKER_FEATURES = {
    'scalar': 0.,
    'vector': np.ones(3),
    'match': 0.1,
}

REFERENCE_FEATURES = {
    'scalar': 1.5,
    'vector': np.full(3, 2),
    'match': 0.1,
}

QUATERNION_FEATURES = {
    'unmatched_quaternion': (1., 0., 0., 0.),
    'matched_quaternions': [(1., 0., 1., 0.), (0.707, 0.707, 0., 0.)],
}

REFERENCE_QUATERNION_FEATURES = {
    'unmatched_quaternion': (0., 0., 0., 1.),
    'matched_quaternions': [(1., 0., 1., 0.), (0.707, 0.707, 0., 0.)],
}


EXPECTED_DIFFERENCES = {
    'scalar': 2.25,
    'vector': 3.,
    'match': 0.,
    'unmatched_quaternion': np.sum(rewards.bounded_quat_dist(
        QUATERNION_FEATURES['unmatched_quaternion'],
        REFERENCE_QUATERNION_FEATURES['unmatched_quaternion']))**2,
    'matched_quaternions': 0.,
}

EXCLUDE_KEYS = ('scalar', 'match')


class RewardsTest(absltest.TestCase):

  def test_compute_squared_differences(self):
    """Basic usage."""
    differences = rewards.compute_squared_differences(
        WALKER_FEATURES, REFERENCE_FEATURES)
    for key, difference in differences.items():
      self.assertEqual(difference, EXPECTED_DIFFERENCES[key])

  def test_compute_squared_differences_exclude_keys(self):
    """Test excluding some keys from squared difference computation."""
    differences = rewards.compute_squared_differences(
        WALKER_FEATURES, REFERENCE_FEATURES, exclude_keys=EXCLUDE_KEYS)
    for key in EXCLUDE_KEYS:
      self.assertNotIn(key, differences)

  def test_compute_squared_differences_quaternion(self):
    """Test that quaternions use a different distance computation."""

    differences = rewards.compute_squared_differences(
        QUATERNION_FEATURES, REFERENCE_QUATERNION_FEATURES)

    for key, difference in differences.items():
      self.assertAlmostEqual(difference, EXPECTED_DIFFERENCES[key])


if __name__ == '__main__':
  absltest.main()
