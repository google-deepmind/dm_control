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
"""Tests for arenas.mazes.covering."""


from absl.testing import absltest
from dm_control.locomotion.arenas import covering
import labmaze
import numpy as np
import six
from six.moves import range

if six.PY3:
  _STRING_DTYPE = '|U1'
else:
  _STRING_DTYPE = '|S1'


class CoveringTest(absltest.TestCase):

  def testRandomMazes(self):
    maze = labmaze.RandomMaze(height=17, width=17,
                              max_rooms=5, room_min_size=3, room_max_size=5,
                              spawns_per_room=0, objects_per_room=0,
                              random_seed=54321)
    for _ in range(1000):
      maze.regenerate()
      walls = covering.make_walls(maze.entity_layer)
      reconstructed = np.full(maze.entity_layer.shape, ' ', dtype=_STRING_DTYPE)
      for wall in walls:
        reconstructed[wall.start.y:wall.end.y, wall.start.x:wall.end.x] = '*'
      np.testing.assert_array_equal(reconstructed, maze.entity_layer)

  def testOddCovering(self):
    maze = labmaze.RandomMaze(height=17, width=17,
                              max_rooms=5, room_min_size=3, room_max_size=5,
                              spawns_per_room=0, objects_per_room=0,
                              random_seed=54321)
    for _ in range(1000):
      maze.regenerate()
      walls = covering.make_walls(maze.entity_layer, make_odd_sized_walls=True)
      reconstructed = np.full(maze.entity_layer.shape, ' ', dtype=_STRING_DTYPE)
      for wall in walls:
        reconstructed[wall.start.y:wall.end.y, wall.start.x:wall.end.x] = '*'
      np.testing.assert_array_equal(reconstructed, maze.entity_layer)
      for wall in walls:
        self.assertEqual((wall.end.y - wall.start.y) % 2, 1)
        self.assertEqual((wall.end.x - wall.start.x) % 2, 1)

  def testNoOverlappingWalls(self):
    maze_string = """..**
                     .***
                     .***
                     """.replace(' ', '')
    walls = covering.make_walls(labmaze.TextGrid(maze_string))
    surface = 0
    for wall in walls:
      size_x = wall.end.x - wall.start.x
      size_y = wall.end.y - wall.start.y
      surface += size_x * size_y
    self.assertEqual(surface, 8)


if __name__ == '__main__':
  absltest.main()
