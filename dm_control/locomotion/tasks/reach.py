# Copyright 2020 The dm_control Authors.
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
"""A (visuomotor) task consisting of reaching to targets for reward."""

import collections
import enum
import itertools

from dm_control import composer
from dm_control.composer.observation import observable as dm_observable
import numpy as np
from six.moves import range
from six.moves import zip

DEFAULT_ALIVE_THRESHOLD = -1.0
DEFAULT_PHYSICS_TIMESTEP = 0.005
DEFAULT_CONTROL_TIMESTEP = 0.03


class TwoTouchState(enum.IntEnum):
  PRE_TOUCH = 0
  TOUCHED_ONCE = 1
  TOUCHED_TWICE = 2  # at appropriate time
  TOUCHED_TOO_SOON = 3
  NO_SECOND_TOUCH = 4


class TwoTouch(composer.Task):
  """Task with target to tap with short delay (for Rat)."""

  def __init__(self,
               walker,
               arena,
               target_builders,
               target_type_rewards,
               shuffle_target_builders=False,
               randomize_spawn_position=False,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               touch_interval=0.8,
               interval_tolerance=0.1,  # consider making a curriculum
               failure_timeout=1.2,
               reset_delay=0.,
               z_height=.14,  # 5.5" in real experiments
               target_area=(),
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    self._walker = walker
    self._arena = arena
    self._walker.create_root_joints(self._arena.attach(self._walker))
    if 'CMUHumanoid' in str(type(self._walker)):
      self._lhand_body = walker.mjcf_model.find('body', 'lhand')
      self._rhand_body = walker.mjcf_model.find('body', 'rhand')
    elif 'Rat' in str(type(self._walker)):
      self._lhand_body = walker.mjcf_model.find('body', 'hand_L')
      self._rhand_body = walker.mjcf_model.find('body', 'hand_R')
    else:
      raise ValueError('Expects Rat or CMUHumanoid.')
    self._lhand_geoms = self._lhand_body.find_all('geom')
    self._rhand_geoms = self._rhand_body.find_all('geom')

    self._targets = []
    self._target_builders = target_builders
    self._target_type_rewards = tuple(target_type_rewards)
    self._shuffle_target_builders = shuffle_target_builders

    self._randomize_spawn_position = randomize_spawn_position
    self._spawn_position = [0.0, 0.0]  # x, y
    self._randomize_spawn_rotation = randomize_spawn_rotation
    self._rotation_bias_factor = rotation_bias_factor

    self._aliveness_reward = aliveness_reward
    self._discount = 1.0

    self._touch_interval = touch_interval
    self._interval_tolerance = interval_tolerance
    self._failure_timeout = failure_timeout
    self._reset_delay = reset_delay
    self._target_positions = []
    self._state_logic = TwoTouchState.PRE_TOUCH

    self._z_height = z_height
    arena_size = self._arena.size
    if target_area:
      self._target_area = target_area
    else:
      self._target_area = [1/2*arena_size[0], 1/2*arena_size[1]]
    target_x = 1.
    target_y = 1.
    self._target_positions.append((target_x, target_y, self._z_height))

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

    self._task_observables = collections.OrderedDict()
    def task_state(physics):
      del physics
      return np.array([self._state_logic])
    self._task_observables['task_logic'] = dm_observable.Generic(task_state)

    self._walker.observables.egocentric_camera.height = 64
    self._walker.observables.egocentric_camera.width = 64

    for observable in (self._walker.observables.proprioception +
                       self._walker.observables.kinematic_sensors +
                       self._walker.observables.dynamic_sensors +
                       list(self._task_observables.values())):
      observable.enabled = True
    self._walker.observables.egocentric_camera.enabled = True

  def _get_targets(self, total_target_count, random_state):
    # Multiply total target count by the fraction for each type, rounded down.
    target_numbers = np.array([1, len(self._target_positions)-1])

    if self._shuffle_target_builders:
      random_state.shuffle(self._target_builders)

    all_targets = []
    for target_type, num in enumerate(target_numbers):
      targets = []
      if num < 1:
        break
      target_builder = self._target_builders[target_type]
      for i in range(num):
        target = target_builder(name='target_{}_{}'.format(target_type, i))
        targets.append(target)
      all_targets.append(targets)
    return all_targets

  @property
  def name(self):
    return 'two_touch'

  @property
  def task_observables(self):
    return self._task_observables

  @property
  def root_entity(self):
    return self._arena

  def _randomize_targets(self, physics, random_state=np.random):
    for ii in range(len(self._target_positions)):
      target_x = self._target_area[0]*random_state.uniform(-1., 1.)
      target_y = self._target_area[1]*random_state.uniform(-1., 1.)
      self._target_positions[ii] = (target_x, target_y, self._z_height)
    target_positions = np.copy(self._target_positions)
    random_state.shuffle(target_positions)
    all_targets = self._targets
    for pos, target in zip(target_positions, itertools.chain(*all_targets)):
      target.reset(physics)
      physics.bind(target.geom).pos = pos
    self._targets = all_targets
    self._target_rewarded_once = [
        [False] * len(targets) for targets in all_targets]
    self._target_rewarded_twice = [
        [False] * len(targets) for targets in all_targets]
    self._first_touch_time = None
    self._second_touch_time = None
    self._do_time_out = False
    self._state_logic = TwoTouchState.PRE_TOUCH

  def initialize_episode_mjcf(self, random_state):
    self._arena.regenerate(random_state)
    for target in itertools.chain(*self._targets):
      target.detach()
    target_positions = np.copy(self._target_positions)
    random_state.shuffle(target_positions)
    all_targets = self._get_targets(len(self._target_positions), random_state)
    for pos, target in zip(target_positions, itertools.chain(*all_targets)):
      self._arena.attach(target)
      target.geom.pos = pos
      target.initialize_episode_mjcf(random_state)
    self._targets = all_targets

  def _respawn_walker(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)

    if self._randomize_spawn_position:
      self._spawn_position = self._arena.spawn_positions[
          random_state.randint(0, len(self._arena.spawn_positions))]

    if self._randomize_spawn_rotation:
      rotation = 2*np.pi*np.random.uniform()
      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]

    self._walker.shift_pose(
        physics,
        [self._spawn_position[0], self._spawn_position[1], 0.0],
        quat,
        rotate_velocity=True)

  def initialize_episode(self, physics, random_state):
    super(TwoTouch, self).initialize_episode(physics, random_state)
    self._respawn_walker(physics, random_state)
    self._state_logic = TwoTouchState.PRE_TOUCH
    self._discount = 1.0
    self._lhand_geomids = set(physics.bind(self._lhand_geoms).element_id)
    self._rhand_geomids = set(physics.bind(self._rhand_geoms).element_id)
    self._hand_geomids = self._lhand_geomids | self._rhand_geomids
    self._randomize_targets(physics)
    self._must_randomize_targets = False
    for target in itertools.chain(*self._targets):
      target._specific_collision_geom_ids = self._hand_geomids  # pylint: disable=protected-access

  def before_step(self, physics, action, random_state):
    super(TwoTouch, self).before_step(physics, action, random_state)
    if self._must_randomize_targets:
      self._randomize_targets(physics)
      self._must_randomize_targets = False

  def should_terminate_episode(self, physics):
    failure_termination = False
    if failure_termination:
      self._discount = 0.0
      return True
    else:
      return False

  def get_reward(self, physics):
    reward = self._aliveness_reward
    lhand_pos = physics.bind(self._lhand_body).xpos
    rhand_pos = physics.bind(self._rhand_body).xpos
    target_pos = physics.bind(self._targets[0][0].geom).xpos
    lhand_rew = np.exp(-3.*sum(np.abs(lhand_pos-target_pos)))
    rhand_rew = np.exp(-3.*sum(np.abs(rhand_pos-target_pos)))
    closeness_reward = np.maximum(lhand_rew, rhand_rew)
    reward += .01*closeness_reward*self._target_type_rewards[0]
    if self._state_logic == TwoTouchState.PRE_TOUCH:
      # touch the first time
      for target_type, targets in enumerate(self._targets):
        for i, target in enumerate(targets):
          if (target.activated[0] and
              not self._target_rewarded_once[target_type][i]):
            self._first_touch_time = physics.time()
            self._state_logic = TwoTouchState.TOUCHED_ONCE
            self._target_rewarded_once[target_type][i] = True
            reward += self._target_type_rewards[target_type]
    elif self._state_logic == TwoTouchState.TOUCHED_ONCE:
      for target_type, targets in enumerate(self._targets):
        for i, target in enumerate(targets):
          if (target.activated[1] and
              not self._target_rewarded_twice[target_type][i]):
            self._second_touch_time = physics.time()
            self._state_logic = TwoTouchState.TOUCHED_TWICE
            self._target_rewarded_twice[target_type][i] = True
            # check if touched too soon
            if ((self._second_touch_time - self._first_touch_time) <
                (self._touch_interval - self._interval_tolerance)):
              self._do_time_out = True
              self._state_logic = TwoTouchState.TOUCHED_TOO_SOON
            # check if touched at correct time
            elif ((self._second_touch_time - self._first_touch_time) <=
                  (self._touch_interval + self._interval_tolerance)):
              reward += self._target_type_rewards[target_type]
      # check if no second touch within time interval
      if ((physics.time() - self._first_touch_time) >
          (self._touch_interval + self._interval_tolerance)):
        self._do_time_out = True
        self._state_logic = TwoTouchState.NO_SECOND_TOUCH
        self._second_touch_time = physics.time()
    elif (self._state_logic == TwoTouchState.TOUCHED_TWICE or
          self._state_logic == TwoTouchState.TOUCHED_TOO_SOON or
          self._state_logic == TwoTouchState.NO_SECOND_TOUCH):
      # hold here due to timeout
      if self._do_time_out:
        if physics.time() > (self._second_touch_time + self._failure_timeout):
          self._do_time_out = False
      # reset/re-randomize
      elif physics.time() > (self._second_touch_time + self._reset_delay):
        self._must_randomize_targets = True
    return reward

  def get_discount(self, physics):
    del physics
    return self._discount
