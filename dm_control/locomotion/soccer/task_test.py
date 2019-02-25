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

"""Tests for locomotion.tasks.soccer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control.locomotion import soccer
import numpy as np
from six.moves import range
from six.moves import zip


RGBA_BLUE = [.1, .1, .8, 1.]
RGBA_RED = [.8, .1, .1, 1.]


def _walker(name, walker_id, marker_rgba):
  return soccer.BoxHead(
      name=name,
      walker_id=walker_id,
      marker_rgba=marker_rgba,
  )


def _team_players(team_size, team, team_name, team_color):
  team_of_players = []
  for i in range(team_size):
    team_of_players.append(
        soccer.Player(team, _walker("%s%d" % (team_name, i), i, team_color)))
  return team_of_players


def _home_team(team_size):
  return _team_players(team_size, soccer.Team.HOME, "home", RGBA_BLUE)


def _away_team(team_size):
  return _team_players(team_size, soccer.Team.AWAY, "away", RGBA_RED)


def _env(players, disable_walker_contacts=True, observables=None):
  return composer.Environment(
      task=soccer.Task(
          players=players,
          arena=soccer.Pitch((20, 15)),
          observables=observables,
          disable_walker_contacts=disable_walker_contacts,
      ),
      time_limit=1)


def _observables_adder(observables_adder):
  if observables_adder == "core":
    return soccer.CoreObservablesAdder()
  if observables_adder == "core_interception":
    return soccer.MultiObservablesAdder(
        [soccer.CoreObservablesAdder(),
         soccer.InterceptionObservablesAdder()])
  raise ValueError("Unrecognized observable_adder %s" % observables_adder)


class TaskTest(parameterized.TestCase):

  def _assert_all_count_equal(self, list_of_lists):
    """Check all lists in the list are count equal."""
    if not list_of_lists:
      return

    first = sorted(list_of_lists[0])
    for other in list_of_lists[1:]:
      self.assertCountEqual(first, other)

  @parameterized.named_parameters(
      ("1vs1_core", 1, "core", 31, True),
      ("2vs2_core", 2, "core", 39, True),
      ("1vs1_interception", 1, "core_interception", 39, True),
      ("2vs2_interception", 2, "core_interception", 47, True),
      ("1vs1_core_contact", 1, "core", 31, False),
      ("2vs2_core_contact", 2, "core", 39, False),
      ("1vs1_interception_contact", 1, "core_interception", 39, False),
      ("2vs2_interception_contact", 2, "core_interception", 47, False),
  )
  def test_step_environment(self, team_size, observables_adder, num_obs,
                            disable_walker_contacts):
    env = _env(
        _home_team(team_size) + _away_team(team_size),
        observables=_observables_adder(observables_adder),
        disable_walker_contacts=disable_walker_contacts)
    self.assertLen(env.action_spec(), 2 * team_size)
    self.assertLen(env.observation_spec(), 2 * team_size)

    actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]

    timestep = env.reset()

    for observation, spec in zip(timestep.observation, env.observation_spec()):
      self.assertLen(spec, num_obs)
      self.assertCountEqual(list(observation.keys()), list(spec.keys()))
      for key in observation.keys():
        self.assertEqual(observation[key].shape, spec[key].shape)

    while not timestep.last():
      timestep = env.step(actions)

  # TODO(b/124848293): consolidate environment stepping loop for task tests.
  @parameterized.named_parameters(
      ("1vs2", 1, 2, 35),
      ("2vs1", 2, 1, 35),
      ("3vs0", 3, 0, 35),
      ("0vs2", 0, 2, 31),
      ("2vs2", 2, 2, 39),
      ("0vs0", 0, 0, None),
  )
  def test_num_players(self, home_size, away_size, num_observations):
    env = _env(_home_team(home_size) + _away_team(away_size))
    self.assertLen(env.action_spec(), home_size + away_size)
    self.assertLen(env.observation_spec(), home_size + away_size)

    actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]

    timestep = env.reset()

    # Members of the same team should have identical specs.
    self._assert_all_count_equal(
        [spec.keys() for spec in env.observation_spec()[:home_size]])
    self._assert_all_count_equal(
        [spec.keys() for spec in env.observation_spec()[-away_size:]])

    for observation, spec in zip(timestep.observation, env.observation_spec()):
      self.assertCountEqual(list(observation.keys()), list(spec.keys()))
      for key in observation.keys():
        self.assertEqual(observation[key].shape, spec[key].shape)

      self.assertLen(spec, num_observations)

    while not timestep.last():
      timestep = env.step(actions)

  def test_all_contacts(self):
    env = _env(_home_team(1) + _away_team(1))

    def _all_contact_configuration(physics, unused_random_state):
      walkers = [p.walker for p in env.task.players]
      ball = env.task.ball

      x, y, rotation = 0., 0., np.pi / 6.
      ball.set_pose(physics, [x, y, 5.])
      ball.set_velocity(
          physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))

      x, y, rotation = 0., 0., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[0].set_pose(physics, [x, y, 3.], quat)
      walkers[0].set_velocity(
          physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))

      x, y, rotation = 0., 0., np.pi / 3. + np.pi
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[1].set_pose(physics, [x, y, 1.], quat)
      walkers[1].set_velocity(
          physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))

    env.add_extra_hook("initialize_episode", _all_contact_configuration)

    actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]

    timestep = env.reset()
    while not timestep.last():
      timestep = env.step(actions)

  def test_symmetric_observations(self):
    env = _env(_home_team(1) + _away_team(1))

    def _symmetric_configuration(physics, unused_random_state):
      walkers = [p.walker for p in env.task.players]
      ball = env.task.ball

      x, y, rotation = 0., 0., np.pi / 6.
      ball.set_pose(physics, [x, y, 0.5])
      ball.set_velocity(
          physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))

      x, y, rotation = 5., 3., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[0].set_pose(physics, [x, y, 0.], quat)
      walkers[0].set_velocity(
          physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))

      x, y, rotation = -5., -3., np.pi / 3. + np.pi
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[1].set_pose(physics, [x, y, 0.], quat)
      walkers[1].set_velocity(
          physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))

    env.add_extra_hook("initialize_episode", _symmetric_configuration)

    timestep = env.reset()
    obs_a, obs_b = timestep.observation
    self.assertCountEqual(list(obs_a.keys()), list(obs_b.keys()))
    for k in sorted(obs_a.keys()):
      o_a, o_b = obs_a[k], obs_b[k]
      np.testing.assert_allclose(o_a, o_b, err_msg=k + " not equal.", atol=1e-6)

  def test_symmetric_dynamic_observations(self):
    env = _env(_home_team(1) + _away_team(1))

    def _symmetric_configuration(physics, unused_random_state):
      walkers = [p.walker for p in env.task.players]
      ball = env.task.ball

      x, y, rotation = 0., 0., np.pi / 6.
      ball.set_pose(physics, [x, y, 0.5])
      # Ball shooting up. Walkers going tangent.
      ball.set_velocity(physics, velocity=[0., 0., 1.],
                        angular_velocity=[0., 0., 0.])

      x, y, rotation = 5., 3., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[0].set_pose(physics, [x, y, 0.], quat)
      walkers[0].set_velocity(physics, velocity=[y, -x, 0.],
                              angular_velocity=[0., 0., 0.])

      x, y, rotation = -5., -3., np.pi / 3. + np.pi
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[1].set_pose(physics, [x, y, 0.], quat)
      walkers[1].set_velocity(physics, velocity=[y, -x, 0.],
                              angular_velocity=[0., 0., 0.])

    env.add_extra_hook("initialize_episode", _symmetric_configuration)

    timestep = env.reset()
    obs_a, obs_b = timestep.observation
    self.assertCountEqual(list(obs_a.keys()), list(obs_b.keys()))
    for k in sorted(obs_a.keys()):
      o_a, o_b = obs_a[k], obs_b[k]
      np.testing.assert_allclose(o_a, o_b, err_msg=k + " not equal.", atol=1e-6)

  def test_prev_actions(self):
    env = _env(_home_team(1) + _away_team(1))

    actions = []
    for i, player in enumerate(env.task.players):
      spec = player.walker.action_spec
      actions.append((i + 1) * np.ones(spec.shape, dtype=spec.dtype))

    env.reset()
    timestep = env.step(actions)

    for walker_idx, obs in enumerate(timestep.observation):
      np.testing.assert_allclose(
          np.squeeze(obs["prev_action"], axis=0),
          actions[walker_idx],
          err_msg="Walker {}: incorrect previous action.".format(walker_idx))

  def test_throw_in(self):
    env = _env(_home_team(1) + _away_team(1))

    def _throw_in_configuration(physics, unused_random_state):
      walkers = [p.walker for p in env.task.players]
      ball = env.task.ball

      x, y, rotation = 0., 3., np.pi / 6.
      ball.set_pose(physics, [x, y, 0.5])
      # Ball shooting up. Walkers going tangent.
      ball.set_velocity(physics, velocity=[0., 50., 0.],
                        angular_velocity=[0., 0., 0.])

      x, y, rotation = 0., -3., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[0].set_pose(physics, [x, y, 0.], quat)
      walkers[0].set_velocity(physics, velocity=[0., 0., 0.],
                              angular_velocity=[0., 0., 0.])
      x, y, rotation = 0., -5., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[1].set_pose(physics, [x, y, 0.], quat)
      walkers[1].set_velocity(physics, velocity=[0., 0., 0.],
                              angular_velocity=[0., 0., 0.])

    env.add_extra_hook("initialize_episode", _throw_in_configuration)

    actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]

    timestep = env.reset()

    while not timestep.last():
      timestep = env.step(actions)

    terminal_ball_vel = np.linalg.norm(
        timestep.observation[0]["ball_ego_linear_velocity"])
    self.assertAlmostEqual(terminal_ball_vel, 0.)

  @parameterized.named_parameters(("score", 50., 0.), ("timeout", 0., 1.))
  def test_terminal_discount(self, init_ball_vel_x, expected_terminal_discount):
    env = _env(_home_team(1) + _away_team(1))

    def _initial_configuration(physics, unused_random_state):
      walkers = [p.walker for p in env.task.players]
      ball = env.task.ball

      x, y, rotation = 0., 0., np.pi / 6.
      ball.set_pose(physics, [x, y, 0.5])
      # Ball shooting up. Walkers going tangent.
      ball.set_velocity(physics, velocity=[init_ball_vel_x, 0., 0.],
                        angular_velocity=[0., 0., 0.])

      x, y, rotation = 0., -3., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[0].set_pose(physics, [x, y, 0.], quat)
      walkers[0].set_velocity(physics, velocity=[0., 0., 0.],
                              angular_velocity=[0., 0., 0.])
      x, y, rotation = 0., 3., np.pi / 3.
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
      walkers[1].set_pose(physics, [x, y, 0.], quat)
      walkers[1].set_velocity(physics, velocity=[0., 0., 0.],
                              angular_velocity=[0., 0., 0.])

    env.add_extra_hook("initialize_episode", _initial_configuration)

    actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]

    timestep = env.reset()

    while not timestep.last():
      timestep = env.step(actions)

    self.assertEqual(timestep.discount, expected_terminal_discount)


if __name__ == "__main__":
  absltest.main()
