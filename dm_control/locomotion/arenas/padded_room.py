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
"""A LabMaze square room where the outermost cells are always empty."""

import labmaze
import numpy as np
_PADDING = 4


class PaddedRoom(labmaze.BaseMaze):
  """A LabMaze square room where the outermost cells are always empty."""

  def __init__(self,
               room_size,
               num_objects=0,
               random_state=None,
               pad_with_walls=True,
               num_agent_spawn_positions=1):
    self._room_size = room_size
    self._num_objects = num_objects
    self._num_agent_spawn_positions = num_agent_spawn_positions
    self._random_state = random_state or np.random

    empty_maze = '\n'.join(['.' * (room_size + _PADDING)] *
                           (room_size + _PADDING) + [''])

    self._entity_layer = labmaze.TextGrid(empty_maze)

    if pad_with_walls:
      self._entity_layer[0, :] = '*'
      self._entity_layer[-1, :] = '*'
      self._entity_layer[:, 0] = '*'
      self._entity_layer[:, -1] = '*'

    self._variations_layer = labmaze.TextGrid(empty_maze)

  def regenerate(self):
    self._entity_layer[1:-1, 1:-1] = ' '
    self._variations_layer[:, :] = '.'

    generated = list(
        self._random_state.choice(
            self._room_size * self._room_size,
            self._num_objects + self._num_agent_spawn_positions,
            replace=False))
    for i, obj in enumerate(generated):
      if i < self._num_agent_spawn_positions:
        token = labmaze.defaults.SPAWN_TOKEN
      else:
        token = labmaze.defaults.OBJECT_TOKEN
      obj_y, obj_x = obj // self._room_size, obj % self._room_size
      self._entity_layer[obj_y + int(_PADDING / 2),
                         obj_x + int(_PADDING / 2)] = token

  @property
  def entity_layer(self):
    return self._entity_layer

  @property
  def variations_layer(self):
    return self._variations_layer

  @property
  def width(self):
    return self._room_size + _PADDING

  @property
  def height(self):
    return self._room_size + _PADDING
