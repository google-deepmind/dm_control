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

"""Corridor-based arenas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from dm_control import composer
from dm_control.composer import variation
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
import six

_SIDE_WALLS_GEOM_GROUP = 3
_CORRIDOR_X_PADDING = 2.0
_WALL_THICKNESS = 0.16
_SIDE_WALL_HEIGHT = 4.0
_DEFAULT_ALPHA = 0.5


@six.add_metaclass(abc.ABCMeta)
class Corridor(composer.Arena):
  """Abstract base class for corridor-type arenas."""

  @abc.abstractmethod
  def regenerate(self, random_state):
    raise NotImplementedError

  @abc.abstractproperty
  def corridor_length(self):
    raise NotImplementedError

  @abc.abstractproperty
  def corridor_width(self):
    raise NotImplementedError

  @abc.abstractproperty
  def ground_geoms(self):
    raise NotImplementedError

  def is_at_target_position(self, position, tolerance=0.0):
    """Checks if a `position` is within `tolerance' of an end of the corridor.

    This can also be used to evaluate more complicated T-shaped or L-shaped
    corridors.

    Args:
      position: An iterable of 2 elements corresponding to the x and y location
        of the position to evaluate.
      tolerance: A `float` tolerance to use while evaluating the position.

    Returns:
    A `bool` indicating whether the `position` is within the `tolerance` of an
    end of the corridor.
    """
    x, _ = position
    return x > self.corridor_length - tolerance


class EmptyCorridor(Corridor):
  """An empty corridor with planes around the perimeter."""

  def _build(self,
             corridor_width=4,
             corridor_length=40,
             visible_side_planes=True,
             name='empty_corridor'):
    """Builds the corridor.

    Args:
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      name: The name of this arena.
    """
    super(EmptyCorridor, self)._build(name=name)

    self._corridor_width = corridor_width
    self._corridor_length = corridor_length

    self._walls_body = self._mjcf_root.worldbody.add('body', name='walls')

    self._mjcf_root.visual.map.znear = 0.0005
    self._mjcf_root.asset.add(
        'texture', type='skybox', builtin='gradient',
        rgb1=[0.4, 0.6, 0.8], rgb2=[0, 0, 0], width=100, height=600)
    self._mjcf_root.visual.headlight.set_attributes(
        ambient=[0.4, 0.4, 0.4], diffuse=[0.8, 0.8, 0.8],
        specular=[0.1, 0.1, 0.1])

    alpha = _DEFAULT_ALPHA if visible_side_planes else 0.0
    self._ground_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', rgba=[0.5, 0.5, 0.5, 1], size=[1, 1, 1])
    self._left_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[1, 0, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
    self._right_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[-1, 0, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
    self._near_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[0, 1, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)
    self._far_plane = self._mjcf_root.worldbody.add(
        'geom', type='plane', xyaxes=[0, -1, 0, 0, 0, 1], size=[1, 1, 1],
        rgba=[1, 0, 0, alpha], group=_SIDE_WALLS_GEOM_GROUP)

    self._current_corridor_length = None
    self._current_corridor_width = None

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor is resized accordingly.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    self._walls_body.geom.clear()
    corridor_width = variation.evaluate(self._corridor_width,
                                        random_state=random_state)
    corridor_length = variation.evaluate(self._corridor_length,
                                         random_state=random_state)
    self._current_corridor_length = corridor_length
    self._current_corridor_width = corridor_width

    self._ground_plane.pos = [corridor_length / 2, 0, 0]
    self._ground_plane.size = [
        corridor_length / 2 + _CORRIDOR_X_PADDING, corridor_width / 2, 1]

    self._left_plane.pos = [
        corridor_length / 2, corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
    self._left_plane.size = [
        corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

    self._right_plane.pos = [
        corridor_length / 2, -corridor_width / 2, _SIDE_WALL_HEIGHT / 2]
    self._right_plane.size = [
        corridor_length / 2 + _CORRIDOR_X_PADDING, _SIDE_WALL_HEIGHT / 2, 1]

    self._near_plane.pos = [
        -_CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
    self._near_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

    self._far_plane.pos = [
        corridor_length + _CORRIDOR_X_PADDING, 0, _SIDE_WALL_HEIGHT / 2]
    self._far_plane.size = [corridor_width / 2, _SIDE_WALL_HEIGHT / 2, 1]

  @property
  def corridor_length(self):
    return self._current_corridor_length

  @property
  def corridor_width(self):
    return self._current_corridor_width

  @property
  def ground_geoms(self):
    return (self._ground_plane,)


class GapsCorridor(EmptyCorridor):
  """A corridor that consists of multiple platforms separated by gaps."""

  def _build(self,
             platform_length=1.,
             gap_length=2.5,
             corridor_width=4,
             corridor_length=40,
             ground_rgba=(0.5, 0.5, 0.5, 1),
             visible_side_planes=False,
             aesthetic='default',
             name='gaps_corridor'):
    """Builds the corridor.

    Args:
      platform_length: A number or a `composer.variation.Variation` object that
        specifies the size of the platforms along the corridor.
      gap_length: A number or a `composer.variation.Variation` object that
        specifies the size of the gaps along the corridor.
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      ground_rgba: A sequence of 4 numbers or a `composer.variation.Variation`
        object specifying the color of the ground.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      aesthetic: option to adjust the material properties and skybox
      name: The name of this arena.
    """
    super(GapsCorridor, self)._build(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes,
        name=name)

    self._platform_length = platform_length
    self._gap_length = gap_length
    self._ground_rgba = ground_rgba
    self._aesthetic = aesthetic

    if self._aesthetic != 'default':
      ground_info = locomotion_arenas_assets.get_ground_texture_info(aesthetic)
      sky_info = locomotion_arenas_assets.get_sky_texture_info(aesthetic)
      texturedir = locomotion_arenas_assets.get_texturedir(aesthetic)
      self._mjcf_root.compiler.texturedir = texturedir

      self._ground_texture = self._mjcf_root.asset.add(
          'texture', name='aesthetic_texture', file=ground_info.file,
          type=ground_info.type)
      self._ground_material = self._mjcf_root.asset.add(
          'material', name='aesthetic_material', texture=self._ground_texture,
          texuniform='true')
      # remove existing skybox
      for texture in self._mjcf_root.asset.find_all('texture'):
        if texture.type == 'skybox':
          texture.remove()
      self._skybox = self._mjcf_root.asset.add(
          'texture', name='aesthetic_skybox', file=sky_info.file,
          type='skybox', gridsize=sky_info.gridsize,
          gridlayout=sky_info.gridlayout)

    self._ground_body = self._mjcf_root.worldbody.add('body', name='ground')

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor resized accordingly, and
    new sets of platforms are created according to values drawn from the
    `platform_length`, `gap_length`, and `ground_rgba` distributions specified
    in `_build`.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    # Resize the entire corridor first.
    super(GapsCorridor, self).regenerate(random_state)

    # Move the ground plane down and make it invisible.
    self._ground_plane.pos = [self._current_corridor_length / 2, 0, -10]
    self._ground_plane.rgba = [0, 0, 0, 0]

    # Clear the existing platform pieces.
    self._ground_body.geom.clear()

    # Make the first platform larger.
    platform_length = 3. * _CORRIDOR_X_PADDING
    platform_pos = [
        platform_length / 2,
        0,
        -_WALL_THICKNESS,
    ]
    platform_size = [
        platform_length / 2,
        self._current_corridor_width / 2,
        _WALL_THICKNESS,
    ]
    if self._aesthetic != 'default':
      self._ground_body.add(
          'geom',
          type='box',
          name='start_floor',
          pos=platform_pos,
          size=platform_size,
          material=self._ground_material)
    else:
      self._ground_body.add(
          'geom',
          type='box',
          rgba=variation.evaluate(self._ground_rgba, random_state),
          name='start_floor',
          pos=platform_pos,
          size=platform_size)

    current_x = platform_length
    platform_id = 0
    while current_x < self._current_corridor_length:
      platform_length = variation.evaluate(
          self._platform_length, random_state=random_state)
      platform_pos = [
          current_x + platform_length / 2.,
          0,
          -_WALL_THICKNESS,
      ]
      platform_size = [
          platform_length / 2,
          self._current_corridor_width / 2,
          _WALL_THICKNESS,
      ]
      if self._aesthetic != 'default':
        self._ground_body.add(
            'geom',
            type='box',
            name='floor_{}'.format(platform_id),
            pos=platform_pos,
            size=platform_size,
            material=self._ground_material)
      else:
        self._ground_body.add(
            'geom',
            type='box',
            rgba=variation.evaluate(self._ground_rgba, random_state),
            name='floor_{}'.format(platform_id),
            pos=platform_pos,
            size=platform_size)

      platform_id += 1

      # Move x to start of the next platform.
      current_x += platform_length + variation.evaluate(
          self._gap_length, random_state=random_state)

  @property
  def ground_geoms(self):
    return (self._ground_plane,) + tuple(self._ground_body.find_all('geom'))


class WallsCorridor(EmptyCorridor):
  """A corridor obstructed by multiple walls aligned against the two sides."""

  def _build(self,
             wall_gap=2.5,
             wall_width=2.5,
             wall_height=2.0,
             swap_wall_side=True,
             wall_rgba=(1, 1, 1, 1),
             corridor_width=4,
             corridor_length=40,
             visible_side_planes=False,
             include_initial_padding=True,
             name='walls_corridor'):
    """Builds the corridor.

    Args:
      wall_gap: A number or a `composer.variation.Variation` object that
        specifies the gap between each consecutive pair obstructing walls.
      wall_width: A number or a `composer.variation.Variation` object that
        specifies the width that the obstructing walls extend into the corridor.
      wall_height: A number or a `composer.variation.Variation` object that
        specifies the height of the obstructing walls.
      swap_wall_side: A boolean or a `composer.variation.Variation` object that
        specifies whether the next obstructing wall should be aligned against
        the opposite side of the corridor compared to the previous one.
      wall_rgba: A sequence of 4 numbers or a `composer.variation.Variation`
        object specifying the color of the walls.
      corridor_width: A number or a `composer.variation.Variation` object that
        specifies the width of the corridor.
      corridor_length: A number or a `composer.variation.Variation` object that
        specifies the length of the corridor.
      visible_side_planes: Whether to the side planes that bound the corridor's
        perimeter should be rendered.
      include_initial_padding: Whether to include initial offset before first
        obstacle.
      name: The name of this arena.
    """
    super(WallsCorridor, self)._build(
        corridor_width=corridor_width,
        corridor_length=corridor_length,
        visible_side_planes=visible_side_planes,
        name=name)

    self._wall_height = wall_height
    self._wall_rgba = wall_rgba
    self._wall_gap = wall_gap
    self._wall_width = wall_width
    self._swap_wall_side = swap_wall_side
    self._include_initial_padding = include_initial_padding

  def regenerate(self, random_state):
    """Regenerates this corridor.

    New values are drawn from the `corridor_width` and `corridor_height`
    distributions specified in `_build`. The corridor resized accordingly, and
    new sets of obstructing walls are created according to values drawn from the
    `wall_gap`, `wall_width`, `wall_height`, and `wall_rgba` distributions
    specified in `_build`.

    Args:
      random_state: A `numpy.random.RandomState` object that is passed to the
        `Variation` objects.
    """
    super(WallsCorridor, self).regenerate(random_state)

    wall_x = variation.evaluate(
        self._wall_gap, random_state=random_state) - _CORRIDOR_X_PADDING
    if self._include_initial_padding:
      wall_x += 2*_CORRIDOR_X_PADDING
    wall_side = 0
    wall_id = 0
    while wall_x < self._current_corridor_length:
      wall_width = variation.evaluate(
          self._wall_width, random_state=random_state)
      wall_height = variation.evaluate(
          self._wall_height, random_state=random_state)
      wall_rgba = variation.evaluate(self._wall_rgba, random_state=random_state)
      if variation.evaluate(self._swap_wall_side, random_state=random_state):
        wall_side = 1 - wall_side

      wall_pos = [
          wall_x,
          (2 * wall_side - 1) * (self._current_corridor_width - wall_width) / 2,
          wall_height / 2
      ]
      wall_size = [_WALL_THICKNESS / 2, wall_width / 2, wall_height / 2]
      self._walls_body.add(
          'geom',
          type='box',
          name='wall_{}'.format(wall_id),
          pos=wall_pos,
          size=wall_size,
          rgba=wall_rgba)

      wall_id += 1
      wall_x += variation.evaluate(self._wall_gap, random_state=random_state)

  @property
  def ground_geoms(self):
    return (self._ground_plane,)
