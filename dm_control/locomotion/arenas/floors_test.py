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

"""Tests for locomotion.arenas.floors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control import mjcf
from dm_control.locomotion.arenas import floors
import numpy as np


class FloorsTest(absltest.TestCase):

  def test_can_compile_mjcf(self):
    arena = floors.Floor()
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)

  def test_size(self):
    floor_size = (12.9, 27.1)
    arena = floors.Floor(size=floor_size)
    self.assertEqual(tuple(arena.ground_geoms[0].size[:2]), floor_size)

  def test_top_camera(self):
    floor_width, floor_height = 12.9, 27.1
    arena = floors.Floor(size=[floor_width, floor_height])

    self.assertGreater(arena._top_camera_y_padding_factor, 1)
    np.testing.assert_array_equal(arena._top_camera.quat, (1, 0, 0, 0))

    expected_camera_y = floor_height * arena._top_camera_y_padding_factor
    np.testing.assert_allclose(
        np.tan(np.deg2rad(arena._top_camera.fovy / 2)),
        expected_camera_y / arena._top_camera.pos[2])


if __name__ == '__main__':
  absltest.main()
