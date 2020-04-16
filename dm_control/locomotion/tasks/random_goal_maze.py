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
"""A task consisting of finding goals/targets in a random maze."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable as observable_lib
from dm_control.locomotion.props import target_sphere
from dm_control.mujoco.wrapper import mjbindings
import numpy as np
from six.moves import range
from six.moves import zip

_NUM_RAYS = 10

# Aliveness in [-1., 0.].
DEFAULT_ALIVE_THRESHOLD = -0.5

DEFAULT_PHYSICS_TIMESTEP = 0.001
DEFAULT_CONTROL_TIMESTEP = 0.025


class NullGoalMaze(composer.Task):
  """A base task for maze with goals."""

  def __init__(self,
               walker,
               maze_arena,
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=True,
               enable_global_task_observables=False,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    """Initializes goal-directed maze task.

    Args:
      walker: The body to navigate the maze.
      maze_arena: The physical maze arena object.
      randomize_spawn_position: Flag to randomize position of spawning.
      randomize_spawn_rotation: Flag to randomize orientation of spawning.
      rotation_bias_factor: A non-negative number that concentrates initial
        orientation away from walls. When set to zero, the initial orientation
        is uniformly random. The larger the value of this number, the more
        likely it is that the initial orientation would face the direction that
        is farthest away from a wall.
      aliveness_reward: Reward for being alive.
      aliveness_threshold: Threshold if should terminate based on walker
        aliveness feature.
      contact_termination: whether to terminate if a non-foot geom touches the
        ground.
      enable_global_task_observables: Flag to provide task observables that
        contain global information, including map layout.
      physics_timestep: timestep of simulation.
      control_timestep: timestep at which agent changes action.
    """
    self._walker = walker
    self._maze_arena = maze_arena
    self._walker.create_root_joints(self._maze_arena.attach(self._walker))

    self._randomize_spawn_position = randomize_spawn_position
    self._randomize_spawn_rotation = randomize_spawn_rotation
    self._rotation_bias_factor = rotation_bias_factor

    self._aliveness_reward = aliveness_reward
    self._aliveness_threshold = aliveness_threshold
    self._contact_termination = contact_termination
    self._discount = 1.0

    self.set_timesteps(
        physics_timestep=physics_timestep, control_timestep=control_timestep)

    self._walker.observables.egocentric_camera.height = 64
    self._walker.observables.egocentric_camera.width = 64

    for observable in (self._walker.observables.proprioception +
                       self._walker.observables.kinematic_sensors +
                       self._walker.observables.dynamic_sensors):
      observable.enabled = True
    self._walker.observables.egocentric_camera.enabled = True

    if enable_global_task_observables:
      # Reveal maze text map as observable.
      maze_obs = observable_lib.Generic(
          lambda _: self._maze_arena.maze.entity_layer)
      maze_obs.enabled = True

      # absolute walker position
      def get_walker_pos(physics):
        walker_pos = physics.bind(self._walker.root_body).xpos
        return walker_pos
      absolute_position = observable_lib.Generic(get_walker_pos)
      absolute_position.enabled = True

      # absolute walker orientation
      def get_walker_ori(physics):
        walker_ori = np.reshape(
            physics.bind(self._walker.root_body).xmat, (3, 3))
        return walker_ori
      absolute_orientation = observable_lib.Generic(get_walker_ori)
      absolute_orientation.enabled = True

      # grid element of player in maze cell: i,j cell in maze layout
      def get_walker_ij(physics):
        walker_xypos = physics.bind(self._walker.root_body).xpos[:-1]
        walker_rel_origin = (
            (walker_xypos +
             np.sign(walker_xypos) * self._maze_arena.xy_scale / 2) /
            (self._maze_arena.xy_scale)).astype(int)
        x_offset = (self._maze_arena.maze.width - 1) / 2
        y_offset = (self._maze_arena.maze.height - 1) / 2
        walker_ij = walker_rel_origin + np.array([x_offset, y_offset])
        return walker_ij
      absolute_position_discrete = observable_lib.Generic(get_walker_ij)
      absolute_position_discrete.enabled = True

      self._task_observables = collections.OrderedDict({
          'maze_layout': maze_obs,
          'absolute_position': absolute_position,
          'absolute_orientation': absolute_orientation,
          'location_in_maze': absolute_position_discrete,  # from bottom left
      })
    else:
      self._task_observables = collections.OrderedDict({})

  @property
  def task_observables(self):
    return self._task_observables

  @property
  def name(self):
    return 'goal_maze'

  @property
  def root_entity(self):
    return self._maze_arena

  def initialize_episode_mjcf(self, unused_random_state):
    self._maze_arena.regenerate()

  def _respawn(self, physics, random_state):
    self._walker.reinitialize_pose(physics, random_state)

    if self._randomize_spawn_position:
      self._spawn_position = self._maze_arena.spawn_positions[
          random_state.randint(0, len(self._maze_arena.spawn_positions))]

    if self._randomize_spawn_rotation:
      # Move walker up out of the way before raycasting.
      self._walker.shift_pose(physics, [0.0, 0.0, 100.0])

      distances = []
      geomid_out = np.array([-1], dtype=np.intc)
      for i in range(_NUM_RAYS):
        theta = 2 * np.pi * i / _NUM_RAYS
        pos = np.array([self._spawn_position[0], self._spawn_position[1], 0.1],
                       dtype=np.float64)
        vec = np.array([np.cos(theta), np.sin(theta), 0], dtype=np.float64)
        dist = mjbindings.mjlib.mj_ray(
            physics.model.ptr, physics.data.ptr, pos, vec,
            None, 1, -1, geomid_out)
        distances.append(dist)

      def remap_with_bias(x):
        """Remaps values [-1, 1] -> [-1, 1] with bias."""
        return np.tanh((1 + self._rotation_bias_factor) * np.arctanh(x))

      max_theta = 2 * np.pi * np.argmax(distances) / _NUM_RAYS
      rotation = max_theta + np.pi * (
          1 + remap_with_bias(random_state.uniform(-1, 1)))

      quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]

      # Move walker back down.
      self._walker.shift_pose(physics, [0.0, 0.0, -100.0])
    else:
      quat = None

    self._walker.shift_pose(
        physics, [self._spawn_position[0], self._spawn_position[1], 0.0],
        quat,
        rotate_velocity=True)

  def initialize_episode(self, physics, random_state):
    super(NullGoalMaze, self).initialize_episode(physics, random_state)
    self._respawn(physics, random_state)
    self._discount = 1.0

    walker_foot_geoms = set(self._walker.ground_contact_geoms)
    walker_nonfoot_geoms = [
        geom for geom in self._walker.mjcf_model.find_all('geom')
        if geom not in walker_foot_geoms]
    self._walker_nonfoot_geomids = set(
        physics.bind(walker_nonfoot_geoms).element_id)
    self._ground_geomids = set(
        physics.bind(self._maze_arena.ground_geoms).element_id)

  def _is_disallowed_contact(self, contact):
    set1, set2 = self._walker_nonfoot_geomids, self._ground_geomids
    return ((contact.geom1 in set1 and contact.geom2 in set2) or
            (contact.geom1 in set2 and contact.geom2 in set1))

  def after_step(self, physics, random_state):
    self._failure_termination = False
    if self._contact_termination:
      for c in physics.data.contact:
        if self._is_disallowed_contact(c):
          self._failure_termination = True
          break

  def should_terminate_episode(self, physics):
    if self._walker.aliveness(physics) < self._aliveness_threshold:
      self._failure_termination = True
    if self._failure_termination:
      self._discount = 0.0
      return True
    else:
      return False

  def get_reward(self, physics):
    del physics
    return self._aliveness_reward

  def get_discount(self, physics):
    del physics
    return self._discount


class RepeatSingleGoalMaze(NullGoalMaze):
  """Requires an agent to repeatedly find the same goal in a maze."""

  def __init__(self,
               walker,
               maze_arena,
               target=None,
               target_reward_scale=1.0,
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=True,
               max_repeats=0,
               enable_global_task_observables=False,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    super(RepeatSingleGoalMaze, self).__init__(
        walker=walker,
        maze_arena=maze_arena,
        randomize_spawn_position=randomize_spawn_position,
        randomize_spawn_rotation=randomize_spawn_rotation,
        rotation_bias_factor=rotation_bias_factor,
        aliveness_reward=aliveness_reward,
        aliveness_threshold=aliveness_threshold,
        contact_termination=contact_termination,
        enable_global_task_observables=enable_global_task_observables,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep)
    if target is None:
      target = target_sphere.TargetSphere()
    self._target = target
    self._rewarded_this_step = False
    self._maze_arena.attach(target)
    self._target_reward_scale = target_reward_scale
    self._max_repeats = max_repeats
    self._targets_obtained = 0

    if enable_global_task_observables:
      xpos_origin_callable = lambda phys: phys.bind(walker.root_body).xpos

      def _target_pos(physics, target=target):
        return physics.bind(target.geom).xpos

      walker.observables.add_egocentric_vector(
          'target_0',
          observable_lib.Generic(_target_pos),
          origin_callable=xpos_origin_callable)

  def initialize_episode_mjcf(self, random_state):
    super(RepeatSingleGoalMaze, self).initialize_episode_mjcf(random_state)
    self._target_position = self._maze_arena.target_positions[
        random_state.randint(0, len(self._maze_arena.target_positions))]
    mjcf.get_attachment_frame(
        self._target.mjcf_model).pos = self._target_position

  def initialize_episode(self, physics, random_state):
    super(RepeatSingleGoalMaze, self).initialize_episode(physics, random_state)
    self._rewarded_this_step = False
    self._targets_obtained = 0

  def after_step(self, physics, random_state):
    super(RepeatSingleGoalMaze, self).after_step(physics, random_state)
    if self._target.activated:
      self._rewarded_this_step = True
      self._targets_obtained += 1
      if self._targets_obtained <= self._max_repeats:
        self._respawn(physics, random_state)
        self._target.reset(physics)
    else:
      self._rewarded_this_step = False

  def should_terminate_episode(self, physics):
    if super(RepeatSingleGoalMaze, self).should_terminate_episode(physics):
      return True
    if self._targets_obtained > self._max_repeats:
      return True

  def get_reward(self, physics):
    del physics
    if self._rewarded_this_step:
      target_reward = self._target_reward_scale
    else:
      target_reward = 0.0
    return target_reward + self._aliveness_reward


class ManyHeterogeneousGoalsMaze(NullGoalMaze):
  """Requires an agent to find multiple goals with different rewards."""

  def __init__(self,
               walker,
               maze_arena,
               target_builders,
               target_type_rewards,
               target_type_proportions,
               shuffle_target_builders=False,
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=True,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    super(ManyHeterogeneousGoalsMaze, self).__init__(
        walker=walker,
        maze_arena=maze_arena,
        randomize_spawn_position=randomize_spawn_position,
        randomize_spawn_rotation=randomize_spawn_rotation,
        rotation_bias_factor=rotation_bias_factor,
        aliveness_reward=aliveness_reward,
        aliveness_threshold=aliveness_threshold,
        contact_termination=contact_termination,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep)
    self._active_targets = []
    self._target_builders = target_builders
    self._target_type_rewards = tuple(target_type_rewards)
    self._target_type_fractions = (
        np.array(target_type_proportions, dtype=float) /
        np.sum(target_type_proportions))
    self._shuffle_target_builders = shuffle_target_builders

  def _get_targets(self, total_target_count, random_state):
    # Multiply total target count by the fraction for each type, rounded down.
    target_numbers = np.array([int(frac * total_target_count)
                               for frac in self._target_type_fractions])

    # Calculate deviations from the ideal ratio incurred by rounding.
    errors = (self._target_type_fractions -
              target_numbers / float(total_target_count))

    # Sort the target types by deviations from ideal ratios.
    target_types_sorted_by_errors = list(np.argsort(errors))

    # Top up individual target classes until we reach the desired total,
    # starting from the class that is furthest away from the ideal ratio.
    current_total = np.sum(target_numbers)
    while current_total < total_target_count:
      target_numbers[target_types_sorted_by_errors.pop()] += 1
      current_total += 1

    if self._shuffle_target_builders:
      random_state.shuffle(self._target_builders)

    all_targets = []
    for target_type, num in enumerate(target_numbers):
      targets = []
      target_builder = self._target_builders[target_type]
      for i in range(num):
        target = target_builder(name='target_{}_{}'.format(target_type, i))
        targets.append(target)
      all_targets.append(targets)
    return all_targets

  def initialize_episode_mjcf(self, random_state):
    super(
        ManyHeterogeneousGoalsMaze, self).initialize_episode_mjcf(random_state)
    for target in itertools.chain(*self._active_targets):
      target.detach()
    target_positions = list(self._maze_arena.target_positions)
    random_state.shuffle(target_positions)
    all_targets = self._get_targets(len(target_positions), random_state)
    for pos, target in zip(target_positions, itertools.chain(*all_targets)):
      self._maze_arena.attach(target)
      mjcf.get_attachment_frame(target.mjcf_model).pos = pos
      target.initialize_episode_mjcf(random_state)
    self._active_targets = all_targets
    self._target_rewarded = [[False] * len(targets) for targets in all_targets]

  def get_reward(self, physics):
    del physics
    reward = self._aliveness_reward
    for target_type, targets in enumerate(self._active_targets):
      for i, target in enumerate(targets):
        if target.activated and not self._target_rewarded[target_type][i]:
          reward += self._target_type_rewards[target_type]
          self._target_rewarded[target_type][i] = True
    return reward

  def should_terminate_episode(self, physics):
    if super(ManyHeterogeneousGoalsMaze,
             self).should_terminate_episode(physics):
      return True
    else:
      for target in itertools.chain(*self._active_targets):
        if not target.activated:
          return False
      # All targets have been activated: successful termination.
      return True


class ManyGoalsMaze(ManyHeterogeneousGoalsMaze):
  """Requires an agent to find all goals in a random maze."""

  def __init__(self,
               walker,
               maze_arena,
               target_builder,
               target_reward_scale=1.0,
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=True,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    super(ManyGoalsMaze, self).__init__(
        walker=walker,
        maze_arena=maze_arena,
        target_builders=[target_builder],
        target_type_rewards=[target_reward_scale],
        target_type_proportions=[1],
        randomize_spawn_position=randomize_spawn_position,
        randomize_spawn_rotation=randomize_spawn_rotation,
        rotation_bias_factor=rotation_bias_factor,
        aliveness_reward=aliveness_reward,
        aliveness_threshold=aliveness_threshold,
        contact_termination=contact_termination,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep)


class RepeatSingleGoalMazeAugmentedWithTargets(RepeatSingleGoalMaze):
  """Augments the single goal maze with many lower reward targets."""

  def __init__(self,
               walker,
               main_target,
               maze_arena,
               num_subtargets=20,
               target_reward_scale=10.0,
               subtarget_reward_scale=1.0,
               subtarget_colors=((0, 0, 0.4), (0, 0, 0.7)),
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=True,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    super(RepeatSingleGoalMazeAugmentedWithTargets, self).__init__(
        walker=walker,
        target=main_target,
        maze_arena=maze_arena,
        target_reward_scale=target_reward_scale,
        randomize_spawn_position=randomize_spawn_position,
        randomize_spawn_rotation=randomize_spawn_rotation,
        rotation_bias_factor=rotation_bias_factor,
        aliveness_reward=aliveness_reward,
        aliveness_threshold=aliveness_threshold,
        contact_termination=contact_termination,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep)
    self._subtarget_reward_scale = subtarget_reward_scale
    self._subtargets = []
    for i in range(num_subtargets):
      subtarget = target_sphere.TargetSphere(
          radius=0.4, rgb1=subtarget_colors[0], rgb2=subtarget_colors[1],
          name='subtarget_{}'.format(i)
      )
      self._subtargets.append(subtarget)
      self._maze_arena.attach(subtarget)
    self._subtarget_rewarded = None

  def initialize_episode_mjcf(self, random_state):
    super(RepeatSingleGoalMazeAugmentedWithTargets,
          self).initialize_episode_mjcf(random_state)
    subtarget_positions = self._maze_arena.target_positions
    for pos, subtarget in zip(subtarget_positions, self._subtargets):
      mjcf.get_attachment_frame(subtarget.mjcf_model).pos = pos
    self._subtarget_rewarded = [False] * len(self._subtargets)

  def get_reward(self, physics):
    main_reward = super(RepeatSingleGoalMazeAugmentedWithTargets,
                        self).get_reward(physics)
    subtarget_reward = 0
    for i, subtarget in enumerate(self._subtargets):
      if subtarget.activated and not self._subtarget_rewarded[i]:
        subtarget_reward += 1
        self._subtarget_rewarded[i] = True
    subtarget_reward *= self._subtarget_reward_scale
    return main_reward + subtarget_reward

  def should_terminate_episode(self, physics):
    if super(RepeatSingleGoalMazeAugmentedWithTargets,
             self).should_terminate_episode(physics):
      return True
    else:
      for subtarget in self._subtargets:
        if not subtarget.activated:
          return False
      # All subtargets have been activated.
      return True
