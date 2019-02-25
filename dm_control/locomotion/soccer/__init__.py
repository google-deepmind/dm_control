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

"""Multi-agent MuJoCo soccer environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control.locomotion.soccer.boxhead import BoxHead
from dm_control.locomotion.soccer.initializers import Initializer
from dm_control.locomotion.soccer.initializers import UniformInitializer
from dm_control.locomotion.soccer.observables import CoreObservablesAdder
from dm_control.locomotion.soccer.observables import InterceptionObservablesAdder
from dm_control.locomotion.soccer.observables import MultiObservablesAdder
from dm_control.locomotion.soccer.observables import ObservablesAdder
from dm_control.locomotion.soccer.pitch import Pitch
from dm_control.locomotion.soccer.pitch import RandomizedPitch
from dm_control.locomotion.soccer.soccer_ball import SoccerBall
from dm_control.locomotion.soccer.task import Task
from dm_control.locomotion.soccer.team import Player
from dm_control.locomotion.soccer.team import Team
from six.moves import range

_RGBA_BLUE = [.1, .1, .8, 1.]
_RGBA_RED = [.8, .1, .1, 1.]


def _make_walker(name, walker_id, marker_rgba):
  """Construct a BoxHead walker."""
  return BoxHead(
      name=name,
      walker_id=walker_id,
      marker_rgba=marker_rgba,
  )


def _make_players(team_size):
  """Construct home and away teams each of `team_size` players."""
  home_players = []
  away_players = []
  for i in range(team_size):
    home_players.append(
        Player(Team.HOME, _make_walker("home%d" % i, i, _RGBA_BLUE)))
    away_players.append(
        Player(Team.AWAY, _make_walker("away%d" % i, i, _RGBA_RED)))
  return home_players + away_players


def load(team_size, time_limit=45.):
  """Construct `team_size`-vs-`team_size` soccer environment."""
  if team_size < 0 or team_size > 11:
    raise ValueError(
        "Team size must be between 1 and 11 (received %d)." % team_size)

  return composer.Environment(
      task=Task(
          players=_make_players(team_size),
          arena=RandomizedPitch(
              min_size=(32, 24), max_size=(48, 36), keep_aspect_ratio=True),
      ),
      time_limit=time_limit)
