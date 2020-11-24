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
"""Calculates a covering of text mazes with overlapping rectangular walls."""

import collections
import numpy as np
from six.moves import range

GridCoordinates = collections.namedtuple('GridCoordinates', ('y', 'x'))
MazeWall = collections.namedtuple('MazeWall', ('start', 'end'))


class _MazeWallCoveringContext(object):
  """Calculates a covering of text mazes with overlapping rectangular walls.

  This class uses a greedy algorithm to try and minimize the number of geoms
  generated to create a given maze. The solution is not guaranteed to be
  optimal, but in most cases should result in a significantly smaller number of
  geoms than if each cell were treated as an individual box.
  """

  def __init__(self, text_maze, wall_char='*', make_odd_sized_walls=False):
    """Initializes this _MazeWallCoveringContext.

    Args:
      text_maze: A `labmaze.TextGrid` instance.
      wall_char: (optional) The character that signifies a wall.
      make_odd_sized_walls: (optional) A boolean, if `True` all wall sections
        generated span odd numbers of grid cells. This option exists primarily
        to appease MuJoCo's texture repeating algorithm.
    """
    self._text_maze = text_maze
    self._wall_char = wall_char
    self._make_odd_sized_walls = make_odd_sized_walls
    self._covered = np.full(text_maze.shape, False, dtype=np.bool)
    self._maze_size = GridCoordinates(*text_maze.shape)
    self._next_start = GridCoordinates(0, 0)
    self._calculated = False
    self._walls = ()

  def calculate(self):
    """Calculates a covering of text mazes with overlapping rectangular walls.

    Returns:
      A tuple of `MazeWall` objects, each describing the corners of a wall.
    """
    if not self._calculated:
      self._calculated = True
      self._find_next_start()
      walls = []
      while self._next_start.y < self._maze_size.y:
        walls.append(self._find_next_wall())
        self._find_next_start()
      self._walls = tuple(walls)
    return self._walls

  def _find_next_start(self):
    """Moves `self._next_start` to the top-left corner of the next wall."""
    for y in range(self._next_start.y, self._maze_size.y):
      start_x = self._next_start.x if y == self._next_start.y else 0
      for x in range(start_x, self._maze_size.x):
        if self._text_maze[y, x] == self._wall_char and not self._covered[y, x]:
          self._next_start = GridCoordinates(y, x)
          return
    self._next_start = self._maze_size

  def _scan_row(self, row, start_col, end_col):
    """Scans a row of text maze to find the longest strip of wall."""
    for col in range(start_col, end_col):
      if (self._text_maze[row, col] != self._wall_char
          or self._covered[row, col]):
        return col
    return end_col

  def _find_next_wall(self):
    """Finds the largest piece of rectangular wall at the current location.

    This function assumes that `self._next_start` is already at the top-left
    corner of the next piece of wall.

    Returns:
      A `MazeWall` named tuple representing the next piece of wall created.
    """
    start = self._next_start
    x = self._maze_size.x
    end_x_for_rows = []
    total_cells = []

    for y in range(start.y, self._maze_size.y):
      x = self._scan_row(y, start.x, x)
      if x > start.x:
        if self._make_odd_sized_walls and (x - start.x) % 2 == 0:
          x -= 1
        end_x_for_rows.append(x)
        total_cells.append((x - start.x) * (y - start.y + 1))
        y += 1
      else:
        break

    if not self._make_odd_sized_walls:
      end_y_offset = total_cells.index(max(total_cells))
    else:
      end_y_offset = 2 * total_cells[::2].index(max(total_cells[::2]))
    end = GridCoordinates(start.y + end_y_offset + 1,
                          end_x_for_rows[end_y_offset])
    self._covered[start.y:end.y, start.x:end.x] = True
    self._next_start = GridCoordinates(start.y, end.x)
    return MazeWall(start, end)


def make_walls(text_maze, wall_char='*', make_odd_sized_walls=False):
  """Calculates a covering of text mazes with overlapping rectangular walls.

  Args:
    text_maze: A `labmaze.TextMaze` instance.
    wall_char: (optional) The character that signifies a wall.
    make_odd_sized_walls: (optional) A boolean, if `True` all wall sections
      generated span odd numbers of grid cells. This option exists primarily
      to appease MuJoCo's texture repeating algorithm.

  Returns:
    A tuple of `MazeWall` objects, each describing the corners of a wall.
  """
  wall_covering_context = _MazeWallCoveringContext(
      text_maze, wall_char=wall_char, make_odd_sized_walls=make_odd_sized_walls)
  return wall_covering_context.calculate()
