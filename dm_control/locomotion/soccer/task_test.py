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

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion import soccer
from dm_control.locomotion.soccer import camera
from dm_control.locomotion.soccer import initializers
from dm_control.mujoco.wrapper import mjbindings
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


def _env(players, disable_walker_contacts=True, observables=None,
         random_state=42, **task_kwargs):
  return composer.Environment(
      task=soccer.Task(
          players=players,
          arena=soccer.Pitch((20, 15)),
          observables=observables,
          disable_walker_contacts=disable_walker_contacts,
          **task_kwargs
      ),
      random_state=random_state,
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
      ("1vs1_core", 1, "core", 33, True),
      ("2vs2_core", 2, "core", 43, True),
      ("1vs1_interception", 1, "core_interception", 41, True),
      ("2vs2_interception", 2, "core_interception", 51, True),
      ("1vs1_core_contact", 1, "core", 33, False),
      ("2vs2_core_contact", 2, "core", 43, False),
      ("1vs1_interception_contact", 1, "core_interception", 41, False),
      ("2vs2_interception_contact", 2, "core_interception", 51, False),
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
      ("1vs2", 1, 2, 38),
      ("2vs1", 2, 1, 38),
      ("3vs0", 3, 0, 38),
      ("0vs2", 0, 2, 33),
      ("2vs2", 2, 2, 43),
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

      self.assertLen(timestep.observation, home_size + away_size)

      self.assertLen(timestep.reward, home_size + away_size)
      for player_spec, player_reward in zip(env.reward_spec(), timestep.reward):
        player_spec.validate(player_reward)

      discount_spec = env.discount_spec()
      discount_spec.validate(timestep.discount)

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

  @parameterized.named_parameters(
      dict(testcase_name="1vs2_draw",
           home_size=1, away_size=2, ball_vel_x=0, expected_home_score=0),
      dict(testcase_name="1vs2_home_score",
           home_size=1, away_size=2, ball_vel_x=50, expected_home_score=1),
      dict(testcase_name="2vs1_away_score",
           home_size=2, away_size=1, ball_vel_x=-50, expected_home_score=-1),
      dict(testcase_name="3vs0_home_score",
           home_size=3, away_size=0, ball_vel_x=50, expected_home_score=1),
      dict(testcase_name="0vs2_home_score",
           home_size=0, away_size=2, ball_vel_x=50, expected_home_score=1),
      dict(testcase_name="2vs2_away_score",
           home_size=2, away_size=2, ball_vel_x=-50, expected_home_score=-1),
  )
  def test_scoring_rewards(
      self, home_size, away_size, ball_vel_x, expected_home_score):
    env = _env(_home_team(home_size) + _away_team(away_size))

    def _score_configuration(physics, random_state):
      del random_state  # Unused.
      # Send the ball shooting towards either the home or away goal.
      env.task.ball.set_pose(physics, [0., 0., 0.5])
      env.task.ball.set_velocity(physics,
                                 velocity=[ball_vel_x, 0., 0.],
                                 angular_velocity=[0., 0., 0.])

    env.add_extra_hook("initialize_episode", _score_configuration)

    actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]

    # Disable contacts and gravity so that the ball follows a straight path.
    with env.physics.model.disable("contact", "gravity"):

      timestep = env.reset()
      with self.subTest("Reward and discount are None on the first timestep"):
        self.assertTrue(timestep.first())
        self.assertIsNone(timestep.reward)
        self.assertIsNone(timestep.discount)

      # Step until the episode ends.
      timestep = env.step(actions)
      while not timestep.last():
        self.assertTrue(timestep.mid())
        # For non-terminal timesteps, the reward should always be 0 and the
        # discount should always be 1.
        np.testing.assert_array_equal(np.hstack(timestep.reward), 0.)
        self.assertEqual(timestep.discount, 1.)
        timestep = env.step(actions)

    # If a goal was scored then the epsiode should have ended with a discount of
    # 0. If neither team scored and the episode ended due to hitting the time
    # limit then the discount should be 1.
    with self.subTest("Correct terminal discount"):
      if expected_home_score != 0:
        expected_discount = 0.
      else:
        expected_discount = 1.
      self.assertEqual(timestep.discount, expected_discount)

    with self.subTest("Correct terminal reward"):
      reward = np.hstack(timestep.reward)
      np.testing.assert_array_equal(reward[:home_size], expected_home_score)
      np.testing.assert_array_equal(reward[home_size:], -expected_home_score)

  def test_throw_in(self):
    env = _env(_home_team(1) + _away_team(1))

    def _throw_in_configuration(physics, unused_random_state):
      walkers = [p.walker for p in env.task.players]
      ball = env.task.ball

      x, y, rotation = 0., 3., np.pi / 6.
      ball.set_pose(physics, [x, y, 0.5])
      # Ball shooting out of bounds.
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

  @parameterized.named_parameters(("reset_only", False), ("step", True))
  def test_render(self, take_step):
    height = 100
    width = 150
    tracking_cameras = []
    for min_distance in [1, 1, 2]:
      tracking_cameras.append(
          camera.MultiplayerTrackingCamera(
              min_distance=min_distance,
              distance_factor=1,
              smoothing_update_speed=0.1,
              width=width,
              height=height,
          ))
    env = _env(_home_team(1) + _away_team(1), tracking_cameras=tracking_cameras)
    env.reset()
    if take_step:
      actions = [np.zeros(s.shape, s.dtype) for s in env.action_spec()]
      env.step(actions)
    rendered_frames = [cam.render() for cam in tracking_cameras]
    for frame in rendered_frames:
      assert frame.shape == (height, width, 3)
    self.assertTrue(np.array_equal(rendered_frames[0], rendered_frames[1]))
    self.assertFalse(np.array_equal(rendered_frames[1], rendered_frames[2]))


class UniformInitializerTest(parameterized.TestCase):

  @parameterized.parameters([0.3, 0.7])
  def test_walker_position(self, spawn_ratio):
    initializer = initializers.UniformInitializer(spawn_ratio=spawn_ratio)
    env = _env(_home_team(2) + _away_team(2), initializer=initializer)
    root_bodies = [p.walker.root_body for p in env.task.players]
    xy_bounds = np.asarray(env.task.arena.size) * spawn_ratio
    env.reset()
    xy = env.physics.bind(root_bodies).xpos[:, :2].copy()
    with self.subTest("X and Y positions within bounds"):
      if np.any(abs(xy) > xy_bounds):
        self.fail("Walker(s) spawned out of bounds. Expected abs(xy) "
                  "<= {}, got:\n{}".format(xy_bounds, xy))
    env.reset()
    xy2 = env.physics.bind(root_bodies).xpos[:, :2].copy()
    with self.subTest("X and Y positions change after reset"):
      if np.any(xy == xy2):
        self.fail("Walker(s) have the same X and/or Y coordinates before and "
                  "after reset. Before: {}, after: {}.".format(xy, xy2))

  def test_walker_rotation(self):
    initializer = initializers.UniformInitializer()
    env = _env(_home_team(2) + _away_team(2), initializer=initializer)

    def quats_to_eulers(quats):
      eulers = np.empty((len(quats), 3), dtype=np.double)
      dt = 1.
      for i, quat in enumerate(quats):
        mjbindings.mjlib.mju_quat2Vel(eulers[i], quat, dt)
      return eulers

    # TODO(b/132671988): Switch to using `get_pose` to get the quaternion once
    #                    `BoxHead.get_pose` and `BoxHead.set_pose` are
    #                    implemented in a consistent way.
    def get_quat(walker):
      return env.physics.bind(walker.root_body).xquat

    env.reset()
    quats = [get_quat(p.walker) for p in env.task.players]
    eulers = quats_to_eulers(quats)
    with self.subTest("Rotation is about the Z-axis only"):
      np.testing.assert_array_equal(eulers[:, :2], 0.)

    env.reset()
    quats2 = [get_quat(p.walker) for p in env.task.players]
    eulers2 = quats_to_eulers(quats2)
    with self.subTest("Rotation about Z changes after reset"):
      if np.any(eulers[:, 2] == eulers2[:, 2]):
        self.fail("Walker(s) have the same rotation about Z before and "
                  "after reset. Before: {}, after: {}."
                  .format(eulers[:, 2], eulers2[:, 2]))

  # TODO(b/132759890): Remove `expectedFailure` decorator once `set_velocity`
  #                    works correctly for the `BoxHead` walker.
  @unittest.expectedFailure
  def test_walker_velocity(self):
    initializer = initializers.UniformInitializer()
    env = _env(_home_team(2) + _away_team(2), initializer=initializer)
    root_joints = []
    non_root_joints = []
    for player in env.task.players:
      attachment_frame = mjcf.get_attachment_frame(player.walker.mjcf_model)
      root_joints.extend(
          attachment_frame.find_all("joint", immediate_children_only=True))
      non_root_joints.extend(player.walker.mjcf_model.find_all("joint"))
    # Assign a non-zero sentinel value to the velocities of all root and
    # non-root joints.
    sentinel_velocity = 3.14
    env.physics.bind(root_joints + non_root_joints).qvel = sentinel_velocity
    # The initializer should zero the velocities of the root joints, but not the
    # non-root joints.
    initializer(env.task, env.physics, env.random_state)
    np.testing.assert_array_equal(env.physics.bind(non_root_joints).qvel,
                                  sentinel_velocity)
    np.testing.assert_array_equal(env.physics.bind(root_joints).qvel, 0.)

  @parameterized.parameters([
      dict(spawn_ratio=0.3, init_ball_z=0.4),
      dict(spawn_ratio=0.5, init_ball_z=0.6),
  ])
  def test_ball_position(self, spawn_ratio, init_ball_z):
    initializer = initializers.UniformInitializer(
        spawn_ratio=spawn_ratio, init_ball_z=init_ball_z)
    env = _env(_home_team(2) + _away_team(2), initializer=initializer)
    xy_bounds = np.asarray(env.task.arena.size) * spawn_ratio
    env.reset()
    position, _ = env.task.ball.get_pose(env.physics)
    xyz = position.copy()
    with self.subTest("X and Y positions within bounds"):
      if np.any(abs(xyz[:2]) > xy_bounds):
        self.fail("Ball spawned out of bounds. Expected abs(xy) "
                  "<= {}, got:\n{}".format(xy_bounds, xyz[:2]))
    with self.subTest("Z position equal to `init_ball_z`"):
      self.assertEqual(xyz[2], init_ball_z)
    env.reset()
    position, _ = env.task.ball.get_pose(env.physics)
    xyz2 = position.copy()
    with self.subTest("X and Y positions change after reset"):
      if np.any(xyz[:2] == xyz2[:2]):
        self.fail("Ball has the same XY position before and after reset. "
                  "Before: {}, after: {}.".format(xyz[:2], xyz2[:2]))

  def test_ball_velocity(self):
    initializer = initializers.UniformInitializer()
    env = _env(_home_team(1) + _away_team(1), initializer=initializer)
    ball_root_joint = mjcf.get_frame_freejoint(env.task.ball.mjcf_model)
    # Set the velocities of the ball root joint to a non-zero sentinel value.
    env.physics.bind(ball_root_joint).qvel = 3.14
    initializer(env.task, env.physics, env.random_state)
    # The initializer should set the ball velocity to zero.
    ball_velocity = env.physics.bind(ball_root_joint).qvel
    np.testing.assert_array_equal(ball_velocity, 0.)

if __name__ == "__main__":
  absltest.main()
