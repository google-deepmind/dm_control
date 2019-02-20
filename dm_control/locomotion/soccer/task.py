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

""""A task where players play a soccer game."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control.locomotion.soccer import initializers
from dm_control.locomotion.soccer import observables as observables_lib
from dm_control.locomotion.soccer import soccer_ball
import numpy as np
from six.moves import zip

_THROW_IN_BALL_Z = 0.5

_REWARD_LOSE = -30.
_REWARD_WIN = 30.

_HOME_XMAT = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
_AWAY_XMAT = np.asarray([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])


def _disable_geom_contacts(entities):
  for entity in entities:
    mjcf_model = entity.mjcf_model
    for geom in mjcf_model.find_all('geom'):
      geom.set_attributes(contype=0)


class Task(composer.Task):
  """A task where two teams of walkers play soccer."""

  def __init__(self,
               players,
               arena,
               ball=None,
               initializer=None,
               observables=None,
               disable_walker_contacts=False,
               nconmax_per_player=200,
               njmax_per_player=200,
               control_timestep=0.025):
    """Construct an instance of soccer.Task.

    This task implements the high-level game logic of multi-agent MuJoCo soccer.

    Args:
      players: a sequence of `soccer.Player` instances, representing
        participants to the game from both teams.
      arena: an instance of `soccer.Pitch`, implementing the physical geoms and
        the sensors associated with the pitch.
      ball: optional instance of `soccer.SoccerBall`, implementing the physical
        geoms and sensors associated with the soccer ball. If None, defaults to
        using `soccer_ball.SoccerBall()`.
      initializer: optional instance of `soccer.Initializer` that initializes
        the task at the start of each episode. If None, defaults to
        `initializers.UniformInitializer()`.
      observables: optional instance of `soccer.Observables` that adds
        observables for each player. If None, defaults to
        `observables.CoreObservables()`.
      disable_walker_contacts: if `True`, disable physical contacts between
        players.
      nconmax_per_player: allocated maximum number of contacts per player. It
        may be necessary to increase this value if you encounter errors due to
        `mjWARN_CONTACTFULL`.
      njmax_per_player: allocated maximum number of scalar constraints per
        player. It may be necessary to increase this value if you encounter
        errors due to `mjWARN_CNSTRFULL`.
      control_timestep: control timestep of the agent.
    """
    self.arena = arena
    self.players = players

    self._initializer = initializer or initializers.UniformInitializer()
    self._observables = observables or observables_lib.CoreObservables()

    if disable_walker_contacts:
      _disable_geom_contacts([p.walker for p in self.players])

    # Create ball and attach ball to arena.
    self.ball = ball or soccer_ball.SoccerBall()
    self.arena.add_free_entity(self.ball)
    self.arena.register_ball(self.ball)

    # Register soccer ball contact tracking for players.
    for player in self.players:
      player.walker.create_root_joints(self.arena.attach(player.walker))
      self.ball.register_player(player)
      # Add per-walkers observables.
      self._observables(self, player)

    self.set_timesteps(
        physics_timestep=0.005, control_timestep=control_timestep)
    self.root_entity.mjcf_model.size.nconmax = nconmax_per_player * len(players)
    self.root_entity.mjcf_model.size.njmax = njmax_per_player * len(players)

  @property
  def observables(self):
    observables = []
    for player in self.players:
      observables.append(
          player.walker.observables.as_dict(fully_qualified=False))
    return observables

  def _throw_in(self, physics, random_state, ball):
    x, y, _ = physics.bind(ball.geom).xpos
    shrink_x, shrink_y = random_state.uniform([0.7, 0.7], [0.9, 0.9])
    ball.set_pose(physics, [x * shrink_x, y * shrink_y, _THROW_IN_BALL_Z])
    ball.set_velocity(
        physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))
    ball.initialize_entity_trackers()

  def initialize_episode_mjcf(self, random_state):
    self.arena.initialize_episode_mjcf(random_state)

  def initialize_episode(self, physics, random_state):
    self.arena.initialize_episode(physics, random_state)
    self._initializer(self, physics, random_state)

  @property
  def root_entity(self):
    return self.arena

  def get_reward(self, physics):
    return [np.zeros((), dtype=np.float32)] * len(self.players)

  def get_discount(self, physics):
    if self.arena.detected_goal():
      return np.zeros((), np.float32)
    return np.ones((), np.float32)

  def should_terminate_episode(self, physics):
    # TerminationType determined by get_discount(physics).
    return self.arena.detected_goal()

  def before_step(self, physics, actions, random_state):
    for player, action in zip(self.players, actions):
      player.walker.apply_action(physics, action, random_state)

    if self.arena.detected_off_court():
      self._throw_in(physics, random_state, self.ball)

  def action_spec(self, physics):
    """Return multi-agent action_spec."""
    return [player.walker.action_spec for player in self.players]
