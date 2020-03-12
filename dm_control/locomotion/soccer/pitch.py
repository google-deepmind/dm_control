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

"""A soccer pitch with home/away goals and one field with position detection."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging
from dm_control import composer
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.locomotion.soccer import team
import numpy as np


_TOP_CAMERA_Y_PADDING_FACTOR = 1.1
_TOP_CAMERA_DISTANCE = 100.
_WALL_HEIGHT = 10.
_WALL_THICKNESS = .5
_SIDE_WIDTH = 32. / 6.
_GROUND_GEOM_GRID_RATIO = 1. / 100  # Grid size for lighting.
_FIELD_BOX_CONTACT_BIT = 1 << 7  # Use a higher bit to prevent potential clash.

_DEFAULT_PITCH_SIZE = (12, 9)
_DEFAULT_GOAL_LENGTH_RATIO = 0.33  # Goal length / pitch width.


def _top_down_cam_fovy(size, top_camera_distance):
  return (360 / np.pi) * np.arctan2(_TOP_CAMERA_Y_PADDING_FACTOR * max(size),
                                    top_camera_distance)


def _wall_pos_xyaxes(size):
  """Infers position and size of bounding walls given pitch size.

  Walls are placed around `ground_geom` that represents the pitch. Note that
  the ball cannot travel beyond `field` but walkers can walk outside of the
  `field` but not the surrounding walls.

  Args:
    size: a tuple of (length, width) of the pitch.

  Returns:
    a list of 4 tuples, each representing the position and xyaxes of a wall
    plane. In order, walls are placed along x-negative, x-positive, y-negative,
    y-positive relative the center of the pitch.
  """
  return [
      ((0., -size[1], 0.), (-1, 0, 0, 0, 0, 1)),
      ((0., size[1], 0.), (1, 0, 0, 0, 0, 1)),
      ((-size[0], 0., 0.), (0, 1, 0, 0, 0, 1)),
      ((size[0], 0., 0.), (0, -1, 0, 0, 0, 1)),
  ]


def _roof_size(size):
  return (size[0], size[1], _WALL_THICKNESS)


class Pitch(composer.Arena):
  """A pitch with a plane, two goals and a field with position detection."""

  def _build(self,
             size=_DEFAULT_PITCH_SIZE,
             goal_size=None,
             top_camera_distance=_TOP_CAMERA_DISTANCE,
             field_box=False,
             name='pitch'):
    """Construct a pitch with walls and position detectors.

    Args:
      size: a tuple of (length, width) of the pitch.
      goal_size: optional (depth, width, height) indicating the goal size.
        If not specified, the goal size is inferred from pitch size with a fixed
        default ratio.
      top_camera_distance: the distance of the top-down camera to the pitch.
      field_box: adds a "field box" that collides with the ball but not the
        walkers.
      name: the name of this arena.
    """
    super(Pitch, self)._build(name=name)
    self._size = size
    self._goal_size = goal_size
    self._top_camera_distance = top_camera_distance

    self._top_camera = self._mjcf_root.worldbody.add(
        'camera',
        name='top_down',
        pos=[0, 0, top_camera_distance],
        zaxis=[0, 0, 1],
        fovy=_top_down_cam_fovy(self._size, top_camera_distance))

    self._mjcf_root.visual.headlight.set_attributes(
        ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

    # Ensure close up geoms are rendered by egocentric cameras.
    self._mjcf_root.visual.map.znear = 0.0005

    # Build groundplane.
    if len(self._size) != 2:
      raise ValueError('`size` should be a sequence of length 2: got {!r}'
                       .format(self._size))
    self._ground_texture = self._mjcf_root.asset.add(
        'texture',
        type='2d',
        builtin='checker',
        name='groundplane',
        rgb1=[0.3, 0.8, 0.3],
        rgb2=[0.1, 0.6, 0.1],
        width=300,
        height=300,
        mark='edge',
        markrgb=[0.8, 0.8, 0.8])
    self._ground_material = self._mjcf_root.asset.add(
        'material', name='groundplane', texture=self._ground_texture)
    self._ground_geom = self._mjcf_root.worldbody.add(
        'geom',
        type='plane',
        material=self._ground_material,
        size=list(self._size) + [max(self._size) * _GROUND_GEOM_GRID_RATIO])

    # Build walls.
    self._walls = []
    for wall_pos, wall_xyaxes in _wall_pos_xyaxes(self._size):
      self._walls.append(
          self._mjcf_root.worldbody.add(
              'geom',
              type='plane',
              rgba=[.1, .1, .1, .8],
              pos=wall_pos,
              size=[1e-7, 1e-7, 1e-7],
              xyaxes=wall_xyaxes))

    # Build goal position detectors.
    # If field_box is enabled, offset goal by 1.0 such that ball reaches the
    # goal position detector before bouncing off the field_box.
    self._fb_offset = 0.5 if field_box else 0.0
    goal_size = self._get_goal_size()
    self._home_goal = props.PositionDetector(
        pos=(-self._size[0] + goal_size[0] - self._fb_offset, 0,
             goal_size[2]),
        size=goal_size,
        rgba=(0, 0, 1, 0.5),
        visible=True,
        name='home_goal')
    self.attach(self._home_goal)

    self._away_goal = props.PositionDetector(
        pos=(self._size[0] - goal_size[0] + self._fb_offset, 0, goal_size[2]),
        size=goal_size,
        rgba=(1, 0, 0, 0.5),
        visible=True,
        name='away_goal')
    self.attach(self._away_goal)

    # Build inverted field position detectors.
    self._field = props.PositionDetector(
        pos=(0, 0),
        size=(self._size[0] - 2 * goal_size[0],
              self._size[1] - 2 * goal_size[0]),
        rgba=(1, 0, 0, 0.1),
        inverted=True,
        visible=True,
        name='field')
    self.attach(self._field)

    # Build field box.
    self._field_box = []
    if field_box:
      for wall_pos, wall_xyaxes in _wall_pos_xyaxes(
          (self._field.upper - self._field.lower) / 2.0):
        self._field_box.append(
            self._mjcf_root.worldbody.add(
                'geom',
                type='plane',
                rgba=[.3, .3, .3, .3],
                pos=wall_pos,
                size=[1e-7, 1e-7, 1e-7],
                xyaxes=wall_xyaxes))

  def _get_goal_size(self):
    goal_size = self._goal_size
    if goal_size is None:
      goal_size = (
          _SIDE_WIDTH / 2,
          self._size[1] * _DEFAULT_GOAL_LENGTH_RATIO,
          _SIDE_WIDTH / 2,
      )
    return goal_size

  def register_ball(self, ball):
    self._home_goal.register_entities(ball)
    self._away_goal.register_entities(ball)

    if self._field_box:
      # Geoms a and b collides if:
      #   (a.contype & b.conaffinity) || (b.contype & a.conaffinity) != 0.
      #   See: http://www.mujoco.org/book/computation.html#Collision
      ball.geom.contype = (ball.geom.contype or 1) | _FIELD_BOX_CONTACT_BIT
      for wall in self._field_box:
        wall.conaffinity = _FIELD_BOX_CONTACT_BIT
        wall.contype = _FIELD_BOX_CONTACT_BIT
    else:
      self._field.register_entities(ball)

  def detected_goal(self):
    """Returning the team that scored a goal."""
    if self._home_goal.detected_entities:
      return team.Team.AWAY
    if self._away_goal.detected_entities:
      return team.Team.HOME
    return None

  def detected_off_court(self):
    return self._field.detected_entities

  @property
  def size(self):
    return self._size

  @property
  def home_goal(self):
    return self._home_goal

  @property
  def away_goal(self):
    return self._away_goal

  @property
  def field(self):
    return self._field

  @property
  def ground_geom(self):
    return self._ground_geom


class RandomizedPitch(Pitch):
  """RandomizedPitch that randomizes its size between (min_size, max_size)."""

  def __init__(self,
               min_size,
               max_size,
               randomizer=None,
               keep_aspect_ratio=False,
               goal_size=None,
               field_box=False,
               top_camera_distance=_TOP_CAMERA_DISTANCE,
               name='randomized_pitch'):
    """Construct a randomized pitch.

    Args:
      min_size: a tuple of minimum (length, width) of the pitch.
      max_size: a tuple of maximum (length, width) of the pitch.
      randomizer: a callable that returns ratio between [0., 1.] that scales
        between min_size, max_size.
      keep_aspect_ratio: if `True`, keep the aspect ratio constant during
        randomization.
      goal_size: optional (depth, width, height) indicating the goal size.
        If not specified, the goal size is inferred from pitch size with a fixed
        default ratio.
      field_box: optional indicating if we should construct field box containing
        the ball (but not the walkers).
      top_camera_distance: the distance of the top-down camera to the pitch.
      name: the name of this arena.
    """
    super(RandomizedPitch, self).__init__(
        size=max_size,
        goal_size=goal_size,
        top_camera_distance=top_camera_distance,
        field_box=field_box,
        name=name)

    self._min_size = min_size
    self._max_size = max_size

    self._randomizer = randomizer or distributions.Uniform()
    self._keep_aspect_ratio = keep_aspect_ratio

    # Sample a new size and regenerate the soccer pitch.
    logging.info('%s between (%s, %s) with %s', self.__class__.__name__,
                 min_size, max_size, self._randomizer)

  def _resize_goals(self, goal_size):
    self._home_goal.resize(
        pos=(-self._size[0] + goal_size[0] + self._fb_offset, 0, goal_size[2]),
        size=goal_size)
    self._away_goal.resize(
        pos=(self._size[0] - goal_size[0] - self._fb_offset, 0, goal_size[2]),
        size=goal_size)

  def initialize_episode_mjcf(self, random_state):
    super(RandomizedPitch, self).initialize_episode_mjcf(random_state)
    min_len, min_wid = self._min_size
    max_len, max_wid = self._max_size

    if self._keep_aspect_ratio:
      len_ratio = self._randomizer(random_state=random_state)
      wid_ratio = len_ratio
    else:
      len_ratio = self._randomizer(random_state=random_state)
      wid_ratio = self._randomizer(random_state=random_state)

    self._size = (min_len + len_ratio * (max_len - min_len),
                  min_wid + wid_ratio * (max_wid - min_wid))

    # Reset top_down camera field of view.
    self._top_camera.fovy = _top_down_cam_fovy(self._size,
                                               self._top_camera_distance)

    # Resize ground geom size.
    self._ground_geom.size = list(
        self._size) + [max(self._size) * _GROUND_GEOM_GRID_RATIO]

    # Resize and reposition walls and roof geoms.
    for i, (wall_pos, _) in enumerate(_wall_pos_xyaxes(self._size)):
      self._walls[i].pos = wall_pos

    goal_size = self._get_goal_size()
    self._resize_goals(goal_size)

    # Resize inverted field position detectors.
    self._field.resize(
        pos=(0, 0),
        size=(self._size[0] - 2 * goal_size[0],
              self._size[1] - 2 * goal_size[0]))

    # Resize and reposition field box geoms.
    if self._field_box:
      for i, (pos, _) in enumerate(
          _wall_pos_xyaxes((self._field.upper - self._field.lower) / 2.0)):
        self._field_box[i].pos = pos
