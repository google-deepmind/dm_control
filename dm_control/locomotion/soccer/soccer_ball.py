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

"""A soccer ball that keeps track of ball-player contacts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from dm_control import mjcf
from dm_control.entities import props
import numpy as np

from dm_control.utils import io as resources

_ASSETS_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'soccer_ball')


def _get_texture(name):
  contents = resources.GetResource(
      os.path.join(_ASSETS_PATH, '{}.png'.format(name)))
  return mjcf.Asset(contents, '.png')


class SoccerBall(props.Primitive):
  """A soccer ball that keeps track of entities that come into contact."""

  def _build(self, radius=0.35, mass=0.045, name='soccer_ball'):
    """Builds this soccer ball.

    Args:
      radius: The radius (in meters) of this target sphere.
      mass: Mass (in kilograms) of the ball.
      name: The name of this entity.
    """
    super(SoccerBall, self)._build(
        geom_type='sphere', size=(radius,), name=name)
    texture = self._mjcf_root.asset.add(
        'texture',
        name='soccer_ball',
        type='cube',
        fileup=_get_texture('up'),
        filedown=_get_texture('down'),
        filefront=_get_texture('front'),
        fileback=_get_texture('back'),
        fileleft=_get_texture('left'),
        fileright=_get_texture('right'))
    material = self._mjcf_root.asset.add(
        'material', name='soccer_ball', texture=texture)
    self._geom.set_attributes(
        pos=[0, 0, radius],
        size=[radius],
        condim=6,
        friction=[.7, .075, .075],
        mass=mass,
        material=material)

    # Add some tracking cameras for visualization and logging.
    self._mjcf_root.worldbody.add(
        'camera',
        name='ball_cam_near',
        pos=[0, -2, 2],
        zaxis=[0, -1, 1],
        fovy=70,
        mode='trackcom')
    self._mjcf_root.worldbody.add(
        'camera',
        name='ball_cam',
        pos=[0, -7, 7],
        zaxis=[0, -1, 1],
        fovy=70,
        mode='trackcom')
    self._mjcf_root.worldbody.add(
        'camera',
        name='ball_cam_far',
        pos=[0, -10, 10],
        zaxis=[0, -1, 1],
        fovy=70,
        mode='trackcom')

    # Keep track of entities to team mapping.
    self._players = []

    # Initialize tracker attributes.
    self.initialize_entity_trackers()

  def register_player(self, player):
    self._players.append(player)

  def initialize_entity_trackers(self):
    self._last_hit = None
    self._hit = False
    self._repossessed = False
    self._intercepted = False

    # Tracks distance traveled by the ball in between consecutive hits.
    self._pos_at_last_step = None
    self._dist_since_last_hit = None
    self._dist_between_last_hits = None

  def initialize_episode(self, physics, unused_random_state):
    self._geom_id = physics.model.name2id(self._geom.full_identifier, 'geom')
    self._geom_id_to_player = {}
    for player in self._players:
      geoms = player.walker.mjcf_model.find_all('geom')
      for geom in geoms:
        geom_id = physics.model.name2id(geom.full_identifier, 'geom')
        self._geom_id_to_player[geom_id] = player

    self.initialize_entity_trackers()

  def after_substep(self, physics, unused_random_state):
    """Resolve contacts and update ball-player contact trackers."""
    if self._hit:
      # Ball has already registered a valid contact within step (during one of
      # previous after_substep calls).
      return

    # Iterate through all contacts to find the first contact between the ball
    # and one of the registered entities.
    for contact in physics.data.contact:
      # Keep contacts that involve the ball and one of the registered entities.
      has_self = False
      for geom_id in (contact.geom1, contact.geom2):
        if geom_id == self._geom_id:
          has_self = True
        else:
          player = self._geom_id_to_player.get(geom_id)

      if has_self and player:
        # Detected a contact between the ball and an registered player.
        if self._last_hit is not None:
          self._intercepted = player.team != self._last_hit.team
        else:
          self._intercepted = True

        # Register repossessed before updating last_hit player.
        self._repossessed = player is not self._last_hit
        self._last_hit = player
        # Register hit event.
        self._hit = True
        break

  def before_step(self, physics, random_state):
    super(SoccerBall, self).before_step(physics, random_state)
    # Reset per simulation step indicator.
    self._hit = False
    self._repossessed = False
    self._intercepted = False

  def after_step(self, physics, random_state):
    super(SoccerBall, self).after_step(physics, random_state)
    pos = physics.bind(self._geom).xpos
    if self._hit:
      # SoccerBall is hit on this step. Update dist_between_last_hits
      # to dist_since_last_hit before resetting dist_since_last_hit.
      self._dist_between_last_hits = self._dist_since_last_hit
      self._dist_since_last_hit = 0.
      self._pos_at_last_step = pos.copy()

    if self._dist_since_last_hit is not None:
      # Accumulate distance traveled since last hit event.
      self._dist_since_last_hit += np.linalg.norm(pos - self._pos_at_last_step)

    self._pos_at_last_step = pos.copy()

  @property
  def last_hit(self):
    """The player that last came in contact with the ball or `None`."""
    return self._last_hit

  @property
  def hit(self):
    """Indicates if the ball is hit during the last simulation step.

    For a timeline shown below:
      ..., agent.step, simulation, agent.step, ...

    Returns:
      True: if the ball is hit by a registered player during simulation step.
      False: if not.
    """
    return self._hit

  @property
  def repossessed(self):
    """Indicates if the ball has been repossessed by a different player.

    For a timeline shown below:
      ..., agent.step, simulation, agent.step, ...

    Returns:
      True if the ball is hit by a registered player during simulation step
        and that player is different from `last_hit`.
      False: if the ball is not hit, or the ball is hit by `last_hit` player.
    """
    return self._repossessed

  @property
  def intercepted(self):
    """Indicates if the ball has been intercepted by a different team.

    For a timeline shown below:
      ..., agent.step, simulation, agent.step, ...

    Returns:
      True: if the ball is hit for the first time, or repossessed by an player
        from a different team.
      False: if the ball is not hit, not repossessed, or repossessed by a
        teammate to `last_hit`.
    """
    return self._intercepted

  @property
  def dist_between_last_hits(self):
    """Distance between last consecutive hits.

    Returns:
      Distance between last two consecutive hit events or `None` if there has
        not been two consecutive hits on the ball.
    """
    return self._dist_between_last_hits
