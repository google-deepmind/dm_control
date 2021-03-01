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
"""Tests for locomotion.arenas.padded_room."""

from absl.testing import absltest
from dm_control import mjcf
from dm_control.locomotion.arenas import labmaze_textures
from dm_control.locomotion.arenas import mazes
from dm_control.locomotion.arenas import padded_room


class PaddedRoomTest(absltest.TestCase):

  def test_can_compile_mjcf(self):
    # Set the wall and floor textures to match DMLab and set the skybox.
    skybox_texture = labmaze_textures.SkyBox(style='sky_03')
    wall_textures = labmaze_textures.WallTextures(style='style_01')
    floor_textures = labmaze_textures.FloorTextures(style='style_01')

    maze = padded_room.PaddedRoom(room_size=4, num_objects=2)
    arena = mazes.MazeWithTargets(
        maze=maze,
        skybox_texture=skybox_texture,
        wall_textures=wall_textures,
        floor_textures=floor_textures)
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)


if __name__ == '__main__':
  absltest.main()
