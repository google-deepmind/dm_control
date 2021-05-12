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
"""Tests for dm_control.locomotion.soccer.Humanoid."""

import itertools
from absl.testing import absltest
from absl.testing import parameterized
from dm_control.locomotion.soccer import humanoid
from dm_control.locomotion.soccer import team


class HumanoidTest(parameterized.TestCase):

  @parameterized.parameters(
      itertools.product(
          humanoid.Humanoid.Visual,
          (team.RGBA_RED, team.RGBA_BLUE),
          (None, 0, 10),
      ))
  def test_instantiation(self, visual, marker_rgba, walker_id):
    if visual != humanoid.Humanoid.Visual.GEOM and walker_id is None:
      self.skipTest('Invalid configuration skipped.')
    humanoid.Humanoid(
        visual=visual, marker_rgba=marker_rgba, walker_id=walker_id)

  @parameterized.parameters(-1, 11)
  def test_invalid_walker_id(self, walker_id):
    with self.assertRaisesWithLiteralMatch(
        ValueError, humanoid._INVALID_WALKER_ID.format(walker_id)):
      humanoid.Humanoid(
          visual=humanoid.Humanoid.Visual.JERSEY,
          walker_id=walker_id,
          marker_rgba=team.RGBA_BLUE)


if __name__ == '__main__':
  absltest.main()
