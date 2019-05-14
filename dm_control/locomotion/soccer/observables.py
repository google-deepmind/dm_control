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
"""Soccer observables modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from dm_control.composer.observation import observable as base_observable
from dm_control.locomotion.soccer import team as team_lib
import numpy as np
import six
from six.moves import zip


@six.add_metaclass(abc.ABCMeta)
class ObservablesAdder(object):
  """A callable that adds a set of per-player observables for a task."""

  @abc.abstractmethod
  def __call__(self, task, player):
    """Adds observables to a player for the given task.

    Args:
      task: A `soccer.Task` instance.
      player: A `Walker` instance to which observables will be added.
    """


class MultiObservablesAdder(ObservablesAdder):
  """Applies multiple `ObservablesAdder`s to a soccer task and player."""

  def __init__(self, observables):
    """Initializes a `MultiObservablesAdder` instance.

    Args:
      observables: A list of `ObservablesAdder` instances.
    """
    self._observables = observables

  def __call__(self, task, player):
    """Adds observables to a player for the given task.

    Args:
      task: A `soccer.Task` instance.
      player: A `Walker` instance to which observables will be added.
    """
    for observable in self._observables:
      observable(task, player)


class CoreObservablesAdder(ObservablesAdder):
  """Core set of per player observables."""

  def __call__(self, task, player):
    """Adds observables to a player for the given task.

    Args:
      task: A `soccer.Task` instance.
      player: A `Walker` instance to which observables will be added.
    """
    # Enable proprioceptive observables.
    self._add_player_proprio_observables(player)

    # Add egocentric observations of soccer ball.
    self._add_player_observables_on_ball(player, task.ball)

    # Add egocentric observations of others.
    teammate_id = 0
    opponent_id = 0
    for other in task.players:
      if other is player:
        continue
      # Infer team prefix for `other` conditioned on `player.team`.
      if player.team != other.team:
        prefix = 'opponent_{}'.format(opponent_id)
        opponent_id += 1
      else:
        prefix = 'teammate_{}'.format(teammate_id)
        teammate_id += 1

      self._add_player_observables_on_other(player, other, prefix)

    self._add_player_arena_observables(player, task.arena)

    # Add per player game statistics.
    self._add_player_stats_observables(task, player)

  def _add_player_observables_on_other(self, player, other, prefix):
    """Add observables of another player in this player's egocentric frame.

    Args:
      player: A `Walker` instance, the player we are adding observables to.
      other: A `Walker` instance corresponding to a different player.
      prefix: A string specifying a prefix to apply to the names of observables
        belonging to `player`.
    """
    if player is other:
      raise ValueError('Cannot add egocentric observables of player on itself.')
    # Origin callable in xpos, xvel for `player`.
    xpos_xyz_callable = lambda p: p.bind(player.walker.root_body).xpos
    xvel_xyz_callable = lambda p: p.bind(player.walker.root_body).cvel[3:]

    # Egocentric observation of other's position, orientation and
    # linear velocities.
    def _cvel_observation(physics, other=other):
      # Velocitmeter reads in local frame but we need world frame observable
      # for egocentric transformation.
      return physics.bind(other.walker.root_body).cvel[3:]

    player.walker.observables.add_egocentric_vector(
        '{}_ego_linear_velocity'.format(prefix),
        base_observable.Generic(_cvel_observation),
        origin_callable=xvel_xyz_callable)
    player.walker.observables.add_egocentric_vector(
        '{}_ego_position'.format(prefix),
        other.walker.observables.position,
        origin_callable=xpos_xyz_callable)
    player.walker.observables.add_egocentric_xmat(
        '{}_ego_orientation'.format(prefix),
        other.walker.observables.orientation)

    # Adds end effectors of the other agents in the other's egocentric frame.
    # A is seeing B's hand extended to B's right.
    player.walker.observables.add_observable(
        '{}_end_effectors_pos'.format(prefix),
        other.walker.observables.end_effectors_pos)

  def _add_player_observables_on_ball(self, player, ball):
    """Add observables of the soccer ball in this player's egocentric frame.

    Args:
      player: A `Walker` instance, the player we are adding observations for.
      ball: A `SoccerBall` instance.
    """
    # Origin callables for egocentric transformations.
    xpos_xyz_callable = lambda p: p.bind(player.walker.root_body).xpos
    xvel_xyz_callable = lambda p: p.bind(player.walker.root_body).cvel[3:]

    # Add egocentric ball observations.
    player.walker.observables.add_egocentric_vector(
        'ball_ego_angular_velocity', ball.observables.angular_velocity)
    player.walker.observables.add_egocentric_vector(
        'ball_ego_position',
        ball.observables.position,
        origin_callable=xpos_xyz_callable)
    player.walker.observables.add_egocentric_vector(
        'ball_ego_linear_velocity',
        ball.observables.linear_velocity,
        origin_callable=xvel_xyz_callable)

  def _add_player_proprio_observables(self, player):
    """Add proprioceptive observables to the given player.

    Args:
      player: A `Walker` instance, the player we are adding observations for.
    """
    for observable in (player.walker.observables.proprioception +
                       player.walker.observables.kinematic_sensors):
      observable.enabled = True

    # Also enable previous action observable as part of proprioception.
    player.walker.observables.prev_action.enabled = True

  def _add_player_arena_observables(self, player, arena):
    """Add observables of the arena.

    Args:
      player: A `Walker` instance to which observables will be added.
      arena: A `Pitch` instance.
    """
    # Enable egocentric view of position detectors (goal, field).
    # Corners named according to walker *facing towards opponent goal*.
    clockwise_names = [
        'team_goal_back_right',
        'team_goal_mid',
        'team_goal_front_left',
        'field_front_left',
        'opponent_goal_back_left',
        'opponent_goal_mid',
        'opponent_goal_front_right',
        'field_back_right',
    ]
    clockwise_features = [
        lambda _: arena.home_goal.lower[:2],
        lambda _: arena.home_goal.mid,
        lambda _: arena.home_goal.upper[:2],
        lambda _: arena.field.upper,
        lambda _: arena.away_goal.upper[:2],
        lambda _: arena.away_goal.mid,
        lambda _: arena.away_goal.lower[:2],
        lambda _: arena.field.lower,
    ]
    xpos_xyz_callable = lambda p: p.bind(player.walker.root_body).xpos
    xpos_xy_callable = lambda p: p.bind(player.walker.root_body).xpos[:2]
    # A list of egocentric reference origin for each one of clockwise_features.
    clockwise_origins = [
        xpos_xy_callable,
        xpos_xyz_callable,
        xpos_xy_callable,
        xpos_xy_callable,
        xpos_xy_callable,
        xpos_xyz_callable,
        xpos_xy_callable,
        xpos_xy_callable,
    ]
    if player.team != team_lib.Team.HOME:
      half = len(clockwise_features) // 2
      clockwise_features = clockwise_features[half:] + clockwise_features[:half]
      clockwise_origins = clockwise_origins[half:] + clockwise_origins[:half]

    for name, feature, origin in zip(clockwise_names, clockwise_features,
                                     clockwise_origins):
      player.walker.observables.add_egocentric_vector(
          name, base_observable.Generic(feature), origin_callable=origin)

  def _add_player_stats_observables(self, task, player):
    """Add observables corresponding to game statistics.

    Args:
      task: A `soccer.Task` instance.
      player: A `Walker` instance to which observables will be added.
    """

    def _stats_vel_to_ball(physics):
      dir_ = (
          physics.bind(task.ball.geom).xpos -
          physics.bind(player.walker.root_body).xpos)
      vel_to_ball = np.dot(dir_[:2] / (np.linalg.norm(dir_[:2]) + 1e-7),
                           physics.bind(player.walker.root_body).cvel[3:5])
      return np.sum(vel_to_ball)

    player.walker.observables.add_observable(
        'stats_vel_to_ball', base_observable.Generic(_stats_vel_to_ball))

    def _stats_closest_vel_to_ball(physics):
      """Velocity to the ball if this walker is the team's closest."""
      closest = None
      min_team_dist_to_ball = np.inf
      for player_ in task.players:
        if player_.team == player.team:
          dist_to_ball = np.linalg.norm(
              physics.bind(task.ball.geom).xpos -
              physics.bind(player_.walker.root_body).xpos)
          if dist_to_ball < min_team_dist_to_ball:
            min_team_dist_to_ball = dist_to_ball
            closest = player_
      if closest is player:
        return _stats_vel_to_ball(physics)
      return 0.

    player.walker.observables.add_observable(
        'stats_closest_vel_to_ball',
        base_observable.Generic(_stats_closest_vel_to_ball))

    def _stats_vel_ball_to_goal(physics):
      """Ball velocity towards opponents' goal."""
      if player.team == team_lib.Team.HOME:
        goal = task.arena.away_goal
      else:
        goal = task.arena.home_goal

      goal_center = (goal.upper + goal.lower) / 2.
      direction = goal_center - physics.bind(task.ball.geom).xpos
      ball_vel_observable = task.ball.observables.linear_velocity
      ball_vel = ball_vel_observable.observation_callable(physics)()

      norm_dir = np.linalg.norm(direction)
      normalized_dir = direction / norm_dir if norm_dir else direction
      return np.sum(np.dot(normalized_dir, ball_vel))

    player.walker.observables.add_observable(
        'stats_vel_ball_to_goal',
        base_observable.Generic(_stats_vel_ball_to_goal))

    def _stats_avg_teammate_dist(physics):
      """Compute average distance from `walker` to its teammates."""
      teammate_dists = []
      for other in task.players:
        if player is other:
          continue
        if other.team != player.team:
          continue
        dist = np.linalg.norm(
            physics.bind(player.walker.root_body).xpos -
            physics.bind(other.walker.root_body).xpos)
        teammate_dists.append(dist)
      return np.mean(teammate_dists) if teammate_dists else 0.

    player.walker.observables.add_observable(
        'stats_home_avg_teammate_dist',
        base_observable.Generic(_stats_avg_teammate_dist))

    def _stats_teammate_spread_out(physics):
      """Compute average distance from `walker` to its teammates."""
      return _stats_avg_teammate_dist(physics) > 5.

    player.walker.observables.add_observable(
        'stats_teammate_spread_out',
        base_observable.Generic(_stats_teammate_spread_out))

    def _stats_home_score(unused_physics):
      if (task.arena.detected_goal() and
          task.arena.detected_goal() == player.team):
        return 1.
      return 0.

    player.walker.observables.add_observable(
        'stats_home_score', base_observable.Generic(_stats_home_score))

    has_opponent = any([p.team != player.team for p in task.players])

    def _stats_away_score(unused_physics):
      if (has_opponent and task.arena.detected_goal() and
          task.arena.detected_goal() != player.team):
        return 1.
      return 0.

    player.walker.observables.add_observable(
        'stats_away_score', base_observable.Generic(_stats_away_score))


# TODO(b/124848293): add unit-test interception observables.
class InterceptionObservablesAdder(ObservablesAdder):
  """Adds obervables representing interception events.

  These observables represent events where this player received the ball from
  another player, or when an opponent intercepted the ball from this player's
  team. For each type of event there are three different thresholds applied to
  the distance travelled by the ball since it last made contact with a player
  (5, 10, or 15 meters).

  For example, on a given timestep `stats_i_received_ball_10m` will be 1 if
  * This player just made contact with the ball
  * The last player to have made contact with the ball was a different player
  * The ball travelled for at least 10 m since it last hit a player
  and 0 otherwise.

  Conversely, `stats_opponent_intercepted_ball_10m` will be 1 if:
  * An opponent just made contact with the ball
  * The last player to have made contact with the ball was on this player's team
  * The ball travelled for at least 10 m since it last hit a player
  """

  def __call__(self, task, player):
    """Adds observables to a player for the given task.

    Args:
      task: A `soccer.Task` instance.
      player: A `Walker` instance to which observables will be added.
    """

    def _stats_i_received_ball(unused_physics):
      if (task.ball.hit and task.ball.repossessed and
          task.ball.last_hit is player):
        return 1.
      return 0.

    player.walker.observables.add_observable(
        'stats_i_received_ball',
        base_observable.Generic(_stats_i_received_ball))

    def _stats_opponent_intercepted_ball(unused_physics):
      """Indicator on if an opponent intercepted the ball."""
      if (task.ball.hit and task.ball.intercepted and
          task.ball.last_hit.team != player.team):
        return 1.
      return 0.

    player.walker.observables.add_observable(
        'stats_opponent_intercepted_ball',
        base_observable.Generic(_stats_opponent_intercepted_ball))

    for dist in [5, 10, 15]:

      def _stats_i_received_ball_dist(physics, dist=dist):
        if (_stats_i_received_ball(physics) and
            task.ball.dist_between_last_hits is not None and
            task.ball.dist_between_last_hits > dist):
          return 1.
        return 0.

      player.walker.observables.add_observable(
          'stats_i_received_ball_%dm' % dist,
          base_observable.Generic(_stats_i_received_ball_dist))

      def _stats_opponent_intercepted_ball_dist(physics, dist=dist):
        if (_stats_opponent_intercepted_ball(physics) and
            task.ball.dist_between_last_hits is not None and
            task.ball.dist_between_last_hits > dist):
          return 1.
        return 0.

      player.walker.observables.add_observable(
          'stats_opponent_intercepted_ball_%dm' % dist,
          base_observable.Generic(_stats_opponent_intercepted_ball_dist))
