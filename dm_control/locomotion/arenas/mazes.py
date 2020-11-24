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
"""Maze-based arenas."""

import string

from absl import logging
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
from dm_control.locomotion.arenas import covering
import labmaze
import numpy as np
import six
from six.moves import range
from six.moves import zip


# Put all "actual" wall geoms in a separate group since they are not rendered.
_WALL_GEOM_GROUP = 3

_TOP_CAMERA_DISTANCE = 100
_TOP_CAMERA_Y_PADDING_FACTOR = 1.1

_DEFAULT_WALL_CHAR = '*'
_DEFAULT_FLOOR_CHAR = '.'


class MazeWithTargets(composer.Arena):
  """A 2D maze with target positions specified by a LabMaze-style text maze."""

  def _build(self, maze, xy_scale=2.0, z_height=2.0,
             skybox_texture=None, wall_textures=None, floor_textures=None,
             aesthetic='default', name='maze'):
    """Initializes this maze arena.

    Args:
      maze: A `labmaze.BaseMaze` instance.
      xy_scale: The size of each maze cell in metres.
      z_height: The z-height of the maze in metres.
      skybox_texture: (optional) A `composer.Entity` that provides a texture
        asset for the skybox.
      wall_textures: (optional) Either a `composer.Entity` that provides texture
        assets for the maze walls, or a dict mapping printable characters to
        such Entities. In the former case, the maze walls are assumed to be
        represented by '*' in the maze's entity layer. In the latter case,
        the dict's keys specify the different characters that can be present
        in the maze's entity layer, and the dict's values are the corresponding
        texture providers.
      floor_textures: (optional) A `composer.Entity` that provides texture
        assets for the maze floor. Unlike with walls, we do not currently
        support per-variation floor texture. Instead, we sample textures from
        the same texture provider for each variation in the variations layer.
      aesthetic: option to adjust the material properties and skybox
      name: (optional) A string, the name of this arena.
    """
    super(MazeWithTargets, self)._build(name)
    self._maze = maze
    self._xy_scale = xy_scale
    self._z_height = z_height

    self._x_offset = (self._maze.width - 1) / 2
    self._y_offset = (self._maze.height - 1) / 2

    self._mjcf_root.default.geom.rgba = [1, 1, 1, 1]

    if aesthetic != 'default':
      sky_info = locomotion_arenas_assets.get_sky_texture_info(aesthetic)
      texturedir = locomotion_arenas_assets.get_texturedir(aesthetic)
      self._mjcf_root.compiler.texturedir = texturedir
      self._skybox = self._mjcf_root.asset.add(
          'texture', name='aesthetic_skybox', file=sky_info.file,
          type='skybox', gridsize=sky_info.gridsize,
          gridlayout=sky_info.gridlayout)
    elif skybox_texture:
      self._skybox_texture = skybox_texture.texture
      self.attach(skybox_texture)
    else:
      self._skybox_texture = self._mjcf_root.asset.add(
          'texture', type='skybox', name='skybox', builtin='gradient',
          rgb1=[.4, .6, .8], rgb2=[0, 0, 0], width=100, height=100)

    self._texturing_geom_names = []
    self._texturing_material_names = []
    if wall_textures:
      if isinstance(wall_textures, dict):
        for texture_provider in set(wall_textures.values()):
          self.attach(texture_provider)
        self._wall_textures = {
            wall_char: texture_provider.textures
            for wall_char, texture_provider in six.iteritems(wall_textures)
        }
      else:
        self.attach(wall_textures)
        self._wall_textures = {_DEFAULT_WALL_CHAR: wall_textures.textures}
    else:
      self._wall_textures = {_DEFAULT_WALL_CHAR: [self._mjcf_root.asset.add(
          'texture', type='2d', name='wall', builtin='flat',
          rgb1=[.8, .8, .8], width=100, height=100)]}

    if aesthetic != 'default':
      ground_info = locomotion_arenas_assets.get_ground_texture_info(aesthetic)
      self._floor_textures = [
          self._mjcf_root.asset.add(
              'texture',
              name='aesthetic_texture_main',
              file=ground_info.file,
              type=ground_info.type),
          self._mjcf_root.asset.add(
              'texture',
              name='aesthetic_texture',
              file=ground_info.file,
              type=ground_info.type)
      ]
    elif floor_textures:
      self._floor_textures = floor_textures.textures
      self.attach(floor_textures)
    else:
      self._floor_textures = [self._mjcf_root.asset.add(
          'texture', type='2d', name='floor', builtin='flat',
          rgb1=[.2, .2, .2], width=100, height=100)]

    ground_x = ((self._maze.width - 1) + 1) * (xy_scale / 2)
    ground_y = ((self._maze.height - 1) + 1) * (xy_scale / 2)
    self._mjcf_root.worldbody.add(
        'geom', name='ground', type='plane',
        pos=[0, 0, 0], size=[ground_x, ground_y, 1], rgba=[0, 0, 0, 0])

    self._maze_body = self._mjcf_root.worldbody.add('body', name='maze_body')

    self._mjcf_root.visual.map.znear = 0.0005

    # Choose the FOV so that the maze always fits nicely within the frame
    # irrespective of actual maze size.
    maze_size = max(self._maze.width, self._maze.height)
    top_camera_fovy = (360 / np.pi) * np.arctan2(
        _TOP_CAMERA_Y_PADDING_FACTOR * maze_size * self._xy_scale / 2,
        _TOP_CAMERA_DISTANCE)
    self._top_camera = self._mjcf_root.worldbody.add(
        'camera', name='top_camera',
        pos=[0, 0, _TOP_CAMERA_DISTANCE], zaxis=[0, 0, 1], fovy=top_camera_fovy)

    self._target_positions = ()
    self._spawn_positions = ()

    self._text_maze_regenerated_hook = None
    self._tile_geom_names = {}

  def _build_observables(self):
    return MazeObservables(self)

  @property
  def top_camera(self):
    return self._top_camera

  @property
  def xy_scale(self):
    return self._xy_scale

  @property
  def z_height(self):
    return self._z_height

  @property
  def maze(self):
    return self._maze

  @property
  def text_maze_regenerated_hook(self):
    """A callback that is executed after the LabMaze object is regenerated."""
    return self._text_maze_modifier

  @text_maze_regenerated_hook.setter
  def text_maze_regenerated_hook(self, hook):
    self._text_maze_regenerated_hook = hook

  @property
  def target_positions(self):
    """A tuple of Cartesian target positions generated for the current maze."""
    return self._target_positions

  @property
  def spawn_positions(self):
    """The Cartesian position at which the agent should be spawned."""
    return self._spawn_positions

  @property
  def target_grid_positions(self):
    """A tuple of grid coordinates of targets generated for the current maze."""
    return self._target_grid_positions

  @property
  def spawn_grid_positions(self):
    """The grid-coordinate position at which the agent should be spawned."""
    return self._spawn_grid_positions

  def regenerate(self, random_state=np.random.RandomState()):
    """Generates a new maze layout."""
    del random_state
    self._maze.regenerate()
    logging.debug('GENERATED MAZE:\n%s', self._maze.entity_layer)
    self._find_spawn_and_target_positions()

    if self._text_maze_regenerated_hook:
      self._text_maze_regenerated_hook()

    # Remove old texturing planes.
    for geom_name in self._texturing_geom_names:
      del self._mjcf_root.worldbody.geom[geom_name]
    self._texturing_geom_names = []

    # Remove old texturing materials.
    for material_name in self._texturing_material_names:
      del self._mjcf_root.asset.material[material_name]
    self._texturing_material_names = []

    # Remove old actual-wall geoms.
    self._maze_body.geom.clear()

    self._current_wall_texture = {
        wall_char: np.random.choice(wall_textures)
        for wall_char, wall_textures in six.iteritems(self._wall_textures)
    }

    for wall_char in self._wall_textures:
      self._make_wall_geoms(wall_char)
    self._make_floor_variations()

  def _make_wall_geoms(self, wall_char):
    walls = covering.make_walls(
        self._maze.entity_layer, wall_char=wall_char, make_odd_sized_walls=True)
    for i, wall in enumerate(walls):
      wall_mid = covering.GridCoordinates(
          (wall.start.y + wall.end.y - 1) / 2,
          (wall.start.x + wall.end.x - 1) / 2)
      wall_pos = np.array([(wall_mid.x - self._x_offset) * self._xy_scale,
                           -(wall_mid.y - self._y_offset) * self._xy_scale,
                           self._z_height / 2])
      wall_size = np.array([(wall.end.x - wall_mid.x - 0.5) * self._xy_scale,
                            (wall.end.y - wall_mid.y - 0.5) * self._xy_scale,
                            self._z_height / 2])
      self._maze_body.add('geom', name='wall{}_{}'.format(wall_char, i),
                          type='box', pos=wall_pos, size=wall_size,
                          group=_WALL_GEOM_GROUP)
      self._make_wall_texturing_planes(wall_char, i, wall_pos, wall_size)

  def _make_wall_texturing_planes(self, wall_char, wall_id,
                                  wall_pos, wall_size):
    xyaxes = {
        'x': {-1: [0, -1, 0, 0, 0, 1], 1: [0, 1, 0, 0, 0, 1]},
        'y': {-1: [1, 0, 0, 0, 0, 1], 1: [-1, 0, 0, 0, 0, 1]},
        'z': {-1: [-1, 0, 0, 0, 1, 0], 1: [1, 0, 0, 0, 1, 0]}
    }
    for direction_index, direction in enumerate(('x', 'y', 'z')):
      index = list(i for i in range(3) if i != direction_index)
      delta_vector = np.array([int(i == direction_index) for i in range(3)])
      material_name = 'wall{}_{}_{}'.format(wall_char, wall_id, direction)
      self._texturing_material_names.append(material_name)
      mat = self._mjcf_root.asset.add(
          'material', name=material_name,
          texture=self._current_wall_texture[wall_char],
          texrepeat=(2 * wall_size[index] / self._xy_scale))
      for sign, sign_name in zip((-1, 1), ('neg', 'pos')):
        if direction == 'z' and sign == -1:
          continue
        geom_name = (
            'wall{}_{}_texturing_{}_{}'.format(
                wall_char, wall_id, sign_name, direction))
        self._texturing_geom_names.append(geom_name)
        self._mjcf_root.worldbody.add(
            'geom', type='plane', name=geom_name,
            pos=(wall_pos + sign * delta_vector * wall_size),
            size=np.concatenate([wall_size[index], [self._xy_scale]]),
            xyaxes=xyaxes[direction][sign], material=mat,
            contype=0, conaffinity=0)

  def _make_floor_variations(self, build_tile_geoms_fn=None):
    """Builds the floor tiles.

    Args:
      build_tile_geoms_fn: An optional callable returning floor tile geoms.
        If not passed, the floor will be built using a default covering method.
        Takes a kwarg `wall_char` that can be used control how active floor
        tiles are selected.
    """
    main_floor_texture = np.random.choice(self._floor_textures)
    for variation in _DEFAULT_FLOOR_CHAR + string.ascii_uppercase:
      if variation not in self._maze.variations_layer:
        break

      if build_tile_geoms_fn is None:
        # Break the floor variation down to odd-sized tiles.
        tiles = covering.make_walls(self._maze.variations_layer,
                                    wall_char=variation,
                                    make_odd_sized_walls=True)
      else:
        tiles = build_tile_geoms_fn(wall_char=variation)

      # Sample a texture that's not the same as the main floor texture.
      variation_texture = main_floor_texture
      if variation != _DEFAULT_FLOOR_CHAR:
        if len(self._floor_textures) == 1:
          return
        else:
          while variation_texture is main_floor_texture:
            variation_texture = np.random.choice(self._floor_textures)

      for i, tile in enumerate(tiles):
        tile_mid = covering.GridCoordinates(
            (tile.start.y + tile.end.y - 1) / 2,
            (tile.start.x + tile.end.x - 1) / 2)
        tile_pos = np.array([(tile_mid.x - self._x_offset) * self._xy_scale,
                             -(tile_mid.y - self._y_offset) * self._xy_scale,
                             0.0])
        tile_size = np.array([(tile.end.x - tile_mid.x - 0.5) * self._xy_scale,
                              (tile.end.y - tile_mid.y - 0.5) * self._xy_scale,
                              self._xy_scale])
        if variation == _DEFAULT_FLOOR_CHAR:
          tile_name = 'floor_{}'.format(i)
        else:
          tile_name = 'floor_{}_{}'.format(variation, i)
        self._tile_geom_names[tile.start] = tile_name
        self._texturing_material_names.append(tile_name)
        self._texturing_geom_names.append(tile_name)
        material = self._mjcf_root.asset.add(
            'material', name=tile_name, texture=variation_texture,
            texrepeat=(2 * tile_size[[0, 1]] / self._xy_scale))
        self._mjcf_root.worldbody.add(
            'geom', name=tile_name, type='plane', material=material,
            pos=tile_pos, size=tile_size, contype=0, conaffinity=0)

  @property
  def ground_geoms(self):
    return tuple([
        geom for geom in self.mjcf_model.find_all('geom')
        if 'ground' in geom.name
    ])

  def find_token_grid_positions(self, tokens):
    out = {token: [] for token in tokens}
    for y in range(self._maze.entity_layer.shape[0]):
      for x in range(self._maze.entity_layer.shape[1]):
        for token in tokens:
          if self._maze.entity_layer[y, x] == token:
            out[token].append((y, x))
    return out

  def grid_to_world_positions(self, grid_positions):
    out = []
    for y, x in grid_positions:
      out.append(np.array([(x - self._x_offset) * self._xy_scale,
                           -(y - self._y_offset) * self._xy_scale,
                           0.0]))
    return out

  def world_to_grid_positions(self, world_positions):
    out = []
    # the order of x, y is reverse between grid positions format and
    # world positions format.
    for x, y, _ in world_positions:
      out.append(np.array([self._y_offset - y / self._xy_scale,
                           self._x_offset + x / self._xy_scale]))
    return out

  def _find_spawn_and_target_positions(self):
    grid_positions = self.find_token_grid_positions([
        labmaze.defaults.OBJECT_TOKEN, labmaze.defaults.SPAWN_TOKEN])
    self._target_grid_positions = tuple(
        grid_positions[labmaze.defaults.OBJECT_TOKEN])
    self._spawn_grid_positions = tuple(
        grid_positions[labmaze.defaults.SPAWN_TOKEN])
    self._target_positions = tuple(
        self.grid_to_world_positions(self._target_grid_positions))
    self._spawn_positions = tuple(
        self.grid_to_world_positions(self._spawn_grid_positions))


class MazeObservables(composer.Observables):

  @composer.observable
  def top_camera(self):
    return observable.MJCFCamera(self._entity.top_camera)


class RandomMazeWithTargets(MazeWithTargets):
  """A randomly generated 2D maze with target positions."""

  def _build(self,
             x_cells,
             y_cells,
             xy_scale=2.0,
             z_height=2.0,
             max_rooms=labmaze.defaults.MAX_ROOMS,
             room_min_size=labmaze.defaults.ROOM_MIN_SIZE,
             room_max_size=labmaze.defaults.ROOM_MAX_SIZE,
             spawns_per_room=labmaze.defaults.SPAWN_COUNT,
             targets_per_room=labmaze.defaults.OBJECT_COUNT,
             max_variations=labmaze.defaults.MAX_VARIATIONS,
             simplify=labmaze.defaults.SIMPLIFY,
             skybox_texture=None,
             wall_textures=None,
             floor_textures=None,
             aesthetic='default',
             name='random_maze'):
    """Initializes this random maze arena.

    Args:
      x_cells: The number of cells along the x-direction of the maze. Must be
        an odd integer.
      y_cells: The number of cells along the y-direction of the maze. Must be
        an odd integer.
      xy_scale: The size of each maze cell in metres.
      z_height: The z-height of the maze in metres.
      max_rooms: (optional) The maximum number of rooms in each generated maze.
      room_min_size: (optional) The minimum size of each room generated.
      room_max_size: (optional) The maximum size of each room generated.
      spawns_per_room: (optional) Number of spawn points
        to generate in each room.
      targets_per_room: (optional) Number of targets to generate in each room.
      max_variations: (optional) Maximum number of variations to generate
        in the variations layer.
      simplify: (optional) flag to simplify the maze.
      skybox_texture: (optional) A `composer.Entity` that provides a texture
        asset for the skybox.
      wall_textures: (optional) A `composer.Entity` that provides texture
        assets for the maze walls.
      floor_textures: (optional) A `composer.Entity` that provides texture
        assets for the maze floor.
      aesthetic: option to adjust the material properties and skybox
      name: (optional) A string, the name of this arena.
    """
    random_seed = np.random.randint(2147483648)  # 2**31
    super(RandomMazeWithTargets, self)._build(
        maze=labmaze.RandomMaze(
            height=y_cells,
            width=x_cells,
            max_rooms=max_rooms,
            room_min_size=room_min_size,
            room_max_size=room_max_size,
            max_variations=max_variations,
            spawns_per_room=spawns_per_room,
            objects_per_room=targets_per_room,
            simplify=simplify,
            random_seed=random_seed),
        xy_scale=xy_scale,
        z_height=z_height,
        skybox_texture=skybox_texture,
        wall_textures=wall_textures,
        floor_textures=floor_textures,
        aesthetic=aesthetic,
        name=name)
