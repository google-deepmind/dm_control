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

"""Tasks involving assembly and/or disassembly of bricks."""

import collections

from absl import logging
from dm_control import composer
from dm_control.composer import initializers
from dm_control.composer import variation
from dm_control.composer.observation import observable
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.manipulation.shared import arenas
from dm_control.manipulation.shared import cameras
from dm_control.manipulation.shared import constants
from dm_control.manipulation.shared import observations
from dm_control.manipulation.shared import registry
from dm_control.manipulation.shared import robots
from dm_control.manipulation.shared import tags
from dm_control.manipulation.shared import workspaces
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import rewards
import numpy as np

mjlib = mjbindings.mjlib


_BrickWorkspace = collections.namedtuple(
    '_BrickWorkspace',
    ['prop_bbox', 'tcp_bbox', 'goal_hint_pos', 'goal_hint_quat', 'arm_offset'])

# Ensures that the prop does not collide with the table during initialization.
_PROP_Z_OFFSET = 1e-6

_WORKSPACE = _BrickWorkspace(
    prop_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, _PROP_Z_OFFSET),
        upper=(0.1, 0.1, _PROP_Z_OFFSET)),
    tcp_bbox=workspaces.BoundingBox(
        lower=(-0.1, -0.1, 0.15),
        upper=(0.1, 0.1, 0.4)),
    goal_hint_pos=(0.2, 0.1, 0.),
    goal_hint_quat=(-0.38268343, 0., 0., 0.92387953),
    arm_offset=robots.ARM_OFFSET)

# Alpha value of the visual goal hint representing the goal state for each task.
_HINT_ALPHA = 0.75

# Distance thresholds for the shaping rewards for getting the top brick close
# to the bottom brick, and for 'clicking' them together.
_CLOSE_THRESHOLD = 0.01
_CLICK_THRESHOLD = 0.001

# Sequence of colors for the brick(s).
_COLOR_VALUES, _COLOR_NAMES = list(
    zip(
        ((1., 0., 0.), 'red'),
        ((0., 1., 0.), 'green'),
        ((0., 0., 1.), 'blue'),
        ((0., 1., 1.), 'cyan'),
        ((1., 0., 1.), 'magenta'),
        ((1., 1., 0.), 'yellow'),
    ))


class _Common(composer.Task):
  """Common components of brick tasks."""

  def __init__(self,
               arena,
               arm,
               hand,
               num_bricks,
               obs_settings,
               workspace,
               control_timestep):
    if not 2 <= num_bricks <= 6:
      raise ValueError('`num_bricks` must be between 2 and 6, got {}.'
                       .format(num_bricks))

    if num_bricks > 3:
      # The default values computed by MuJoCo's compiler are too small if there
      # are more than three stacked bricks, since each stacked pair generates
      # a large number of contacts. The values below are sufficient for up to
      # 6 stacked bricks.
      # TODO(b/78331644): It may be useful to log the size of `physics.model`
      #                   and `physics.data` after compilation to gauge the
      #                   impact of these changes on MuJoCo's memory footprint.
      arena.mjcf_model.size.nconmax = 400
      arena.mjcf_model.size.njmax = 1200

    self._arena = arena
    self._arm = arm
    self._hand = hand
    self._arm.attach(self._hand)
    self._arena.attach_offset(self._arm, offset=workspace.arm_offset)
    self.control_timestep = control_timestep

    # Add custom camera observable.
    self._task_observables = cameras.add_camera_observables(
        arena, obs_settings, cameras.FRONT_CLOSE)

    color_sequence = iter(_COLOR_VALUES)
    brick_obs_options = observations.make_options(
        obs_settings, observations.FREEPROP_OBSERVABLES)

    bricks = []
    brick_frames = []
    goal_hint_bricks = []
    for _ in range(num_bricks):
      color = next(color_sequence)
      brick = props.Duplo(color=color,
                          observable_options=brick_obs_options)
      brick_frames.append(arena.add_free_entity(brick))
      bricks.append(brick)

      # Translucent, contactless brick with no observables. These are used to
      # provide a visual hint representing the goal state for each task.
      hint_brick = props.Duplo(color=color)
      _hintify(hint_brick, alpha=_HINT_ALPHA)
      arena.attach(hint_brick)
      goal_hint_bricks.append(hint_brick)

    self._bricks = bricks
    self._brick_frames = brick_frames
    self._goal_hint_bricks = goal_hint_bricks

    # Position and quaternion for the goal hint.
    self._goal_hint_pos = workspace.goal_hint_pos
    self._goal_hint_quat = workspace.goal_hint_quat

    self._tcp_initializer = initializers.ToolCenterPointInitializer(
        self._hand, self._arm,
        position=distributions.Uniform(*workspace.tcp_bbox),
        quaternion=workspaces.DOWN_QUATERNION)

    # Add sites for visual debugging.
    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.tcp_bbox.lower,
        upper=workspace.tcp_bbox.upper,
        rgba=constants.GREEN, name='tcp_spawn_area')

    workspaces.add_bbox_site(
        body=self.root_entity.mjcf_model.worldbody,
        lower=workspace.prop_bbox.lower,
        upper=workspace.prop_bbox.upper,
        rgba=constants.BLUE, name='prop_spawn_area')

  @property
  def task_observables(self):
    return self._task_observables

  @property
  def root_entity(self):
    return self._arena

  @property
  def arm(self):
    return self._arm

  @property
  def hand(self):
    return self._hand


class Stack(_Common):
  """Build a stack of Duplo bricks."""

  def __init__(self,
               arena,
               arm,
               hand,
               num_bricks,
               target_height,
               moveable_base,
               randomize_order,
               obs_settings,
               workspace,
               control_timestep):
    """Initializes a new `Stack` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      num_bricks: The total number of bricks; must be between 2 and 6.
      target_height: The target number of bricks in the stack in order to get
        maximum reward. Must be between 2 and `num_bricks`.
      moveable_base: Boolean specifying whether or not the bottom brick should
        be moveable.
      randomize_order: Boolean specifying whether to randomize the desired order
        of bricks in the stack at the start of each episode.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: A `_BrickWorkspace` instance.
      control_timestep: Float specifying the control timestep in seconds.

    Raises:
      ValueError: If `num_bricks` is not between 2 and 6, or if
        `target_height` is not between 2 and `num_bricks - 1`.
    """
    if not 2 <= target_height <= num_bricks:
      raise ValueError('`target_height` must be between 2 and {}, got {}.'
                       .format(num_bricks, target_height))

    super(Stack, self).__init__(arena=arena,
                                arm=arm,
                                hand=hand,
                                num_bricks=num_bricks,
                                obs_settings=obs_settings,
                                workspace=workspace,
                                control_timestep=control_timestep)

    self._moveable_base = moveable_base
    self._randomize_order = randomize_order
    self._target_height = target_height
    self._prop_bbox = workspace.prop_bbox

    # Shuffled at the start of each episode if `randomize_order` is True.
    self._desired_order = np.arange(target_height)

    # In the random order case, create a `prop_pose` observable that informs the
    # agent of the desired order.
    if randomize_order:
      desired_order_observable = observable.Generic(self._get_desired_order)
      desired_order_observable.configure(**obs_settings.prop_pose._asdict())
      self._task_observables['desired_order'] = desired_order_observable

  def _get_desired_order(self, physics):
    del physics  # Unused
    return self._desired_order.astype(np.double)

  def initialize_episode_mjcf(self, random_state):
    if self._randomize_order:
      self._desired_order = random_state.choice(
          len(self._bricks), size=self._target_height, replace=False)
      logging.info('Desired stack order (from bottom to top): [%s]',
                   ' '.join(_COLOR_NAMES[i] for i in self._desired_order))

    # If the base of the stack should be fixed, remove the freejoint for the
    # first brick (and ensure that all the others have freejoints).
    fixed_indices = [] if self._moveable_base else [self._desired_order[0]]
    _add_or_remove_freejoints(attachment_frames=self._brick_frames,
                              fixed_indices=fixed_indices)

    # We need to define the prop initializer for the bricks here rather than in
    # the `__init__`, since `PropPlacer` looks for freejoints on instantiation.
    self._brick_placer = initializers.PropPlacer(
        props=self._bricks,
        position=distributions.Uniform(*self._prop_bbox),
        quaternion=workspaces.uniform_z_rotation,
        settle_physics=True)

  def initialize_episode(self, physics, random_state):
    self._brick_placer(physics, random_state)
    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._tcp_initializer(physics, random_state)
    # Arrange the goal hint bricks in the desired stack order.
    _build_stack(physics,
                 bricks=self._goal_hint_bricks,
                 base_pos=self._goal_hint_pos,
                 base_quat=self._goal_hint_quat,
                 order=self._desired_order,
                 random_state=random_state)

  def get_reward(self, physics):
    pairs = list(zip(self._desired_order[:-1], self._desired_order[1:]))
    pairwise_rewards = _get_pairwise_stacking_rewards(
        physics=physics, bricks=self._bricks, pairs=pairs)
    # The final reward is an average over the pairwise rewards.
    return np.mean(pairwise_rewards)


class Reassemble(_Common):
  """Disassemble a stack of Duplo bricks and reassemble it in another order."""

  def __init__(self,
               arena,
               arm,
               hand,
               num_bricks,
               randomize_initial_order,
               randomize_desired_order,
               obs_settings,
               workspace,
               control_timestep):
    """Initializes a new `Reassemble` task.

    Args:
      arena: `composer.Entity` instance.
      arm: `robot_base.RobotArm` instance.
      hand: `robot_base.RobotHand` instance.
      num_bricks: The total number of bricks; must be between 2 and 6.
      randomize_initial_order: Boolean specifying whether to randomize the
        initial order  of bricks in the stack at the start of each episode.
      randomize_desired_order: Boolean specifying whether to independently
        randomize the desired order of bricks in the stack at the start of each
        episode. By default the desired order will be the reverse of the initial
        order, with the exception of the base brick which is always the same as
        in the initial order since it is welded in place.
      obs_settings: `observations.ObservationSettings` instance.
      workspace: A `_BrickWorkspace` instance.
      control_timestep: Float specifying the control timestep in seconds.

    Raises:
      ValueError: If `num_bricks` is not between 2 and 6.
    """
    super(Reassemble, self).__init__(arena=arena,
                                     arm=arm,
                                     hand=hand,
                                     num_bricks=num_bricks,
                                     obs_settings=obs_settings,
                                     workspace=workspace,
                                     control_timestep=control_timestep)
    self._randomize_initial_order = randomize_initial_order
    self._randomize_desired_order = randomize_desired_order

    # Randomized at the start of each episode if `randomize_initial_order` is
    # True.
    self._initial_order = np.arange(num_bricks)

    # Randomized at the start of each episode if `randomize_desired_order` is
    # True.
    self._desired_order = self._initial_order.copy()
    self._desired_order[1:] = self._desired_order[-1:0:-1]

    # In the random order case, create a `prop_pose` observable that informs the
    # agent of the desired order.
    if randomize_desired_order:
      desired_order_observable = observable.Generic(self._get_desired_order)
      desired_order_observable.configure(**obs_settings.prop_pose._asdict())
      self._task_observables['desired_order'] = desired_order_observable

    # Distributions of positions and orientations for the base of the stack.
    self._base_pos = distributions.Uniform(*workspace.prop_bbox)
    self._base_quat = workspaces.uniform_z_rotation

  def _get_desired_order(self, physics):
    del physics  # Unused
    return self._desired_order.astype(np.double)

  def initialize_episode_mjcf(self, random_state):
    if self._randomize_initial_order:
      random_state.shuffle(self._initial_order)

    # The bottom brick will be fixed to the table, so it must be the same in
    # both the initial and desired order.
    self._desired_order[0] = self._initial_order[0]
    # By default the desired order of the other bricks is the opposite of their
    # initial order.
    self._desired_order[1:] = self._initial_order[-1:0:-1]

    if self._randomize_desired_order:
      random_state.shuffle(self._desired_order[1:])

    logging.info('Desired stack order (from bottom to top): [%s]',
                 ' '.join(_COLOR_NAMES[i] for i in self._desired_order))

    # Remove the freejoint from the bottom brick in the stack.
    _add_or_remove_freejoints(attachment_frames=self._brick_frames,
                              fixed_indices=[self._initial_order[0]])

  def initialize_episode(self, physics, random_state):
    # Build the initial stack.
    _build_stack(physics,
                 bricks=self._bricks,
                 base_pos=self._base_pos,
                 base_quat=self._base_quat,
                 order=self._initial_order,
                 random_state=random_state)
    # Arrange the goal hint bricks into a stack with the desired order.
    _build_stack(physics,
                 bricks=self._goal_hint_bricks,
                 base_pos=self._goal_hint_pos,
                 base_quat=self._goal_hint_quat,
                 order=self._desired_order,
                 random_state=random_state)
    self._hand.set_grasp(physics, close_factors=random_state.uniform())
    self._tcp_initializer(physics, random_state)

  def get_reward(self, physics):
    pairs = list(zip(self._desired_order[:-1], self._desired_order[1:]))
    # We set `close_coef=0.` because the coarse shaping reward causes problems
    # for this task (it means there is a strong disincentive to break up the
    # initial stack).
    pairwise_rewards = _get_pairwise_stacking_rewards(
        physics=physics,
        bricks=self._bricks,
        pairs=pairs,
        close_coef=0.)
    # The final reward is an average over the pairwise rewards.
    return np.mean(pairwise_rewards)


def _distance(pos1, pos2):
  diff = pos1 - pos2
  return sum(np.sqrt((diff * diff).sum(1)))


def _min_stud_to_hole_distance(physics, bottom_brick, top_brick):
  # Positions of the top left and bottom right studs on the `bottom_brick` and
  # the top left and bottom right holes on the `top_brick`.
  stud_pos = physics.bind(bottom_brick.studs[[0, -1], [0, -1]]).xpos
  hole_pos = physics.bind(top_brick.holes[[0, -1], [0, -1]]).xpos
  # Bricks are rotationally symmetric, so we compute top left -> top left and
  # top left -> bottom right distances and return whichever of these is smaller.
  dist1 = _distance(stud_pos, hole_pos)
  dist2 = _distance(stud_pos[::-1], hole_pos)
  return min(dist1, dist2)


def _get_pairwise_stacking_rewards(physics, bricks, pairs, close_coef=0.1):
  """Returns a vector of shaping reward components based on pairwise distances.

  Args:
    physics: An `mjcf.Physics` instance.
    bricks: A list of `composer.Entity` instances corresponding to bricks.
    pairs: A list of `(bottom_idx, top_idx)` tuples specifying which pairs of
      bricks should be measured.
    close_coef: Float specfying the relative weight given to the coarse-
      tolerance shaping component for getting the bricks close to one another
      (as opposed to the fine-tolerance component for clicking them together).

  Returns:
    A numpy array of size `len(pairs)` containing values in (0, 1], where
    1 corresponds to a stacked pair of bricks.
  """
  distances = []
  for bottom_idx, top_idx in pairs:
    bottom_brick = bricks[bottom_idx]
    top_brick = bricks[top_idx]
    distances.append(
        _min_stud_to_hole_distance(physics, bottom_brick, top_brick))
  distances = np.hstack(distances)

  # Coarse-tolerance component for bringing the holes close to the studs.
  close = rewards.tolerance(
      distances, bounds=(0, _CLOSE_THRESHOLD), margin=(_CLOSE_THRESHOLD * 10))

  # Fine-tolerance component for clicking the bricks together.
  clicked = rewards.tolerance(
      distances, bounds=(0, _CLICK_THRESHOLD), margin=_CLICK_THRESHOLD)

  # Weighted average of coarse and fine components for each pair of bricks.
  return np.average([close, clicked], weights=[close_coef, 1.], axis=0)


def _build_stack(physics, bricks, base_pos, base_quat, order, random_state):
  """Builds a stack of bricks.

  Args:
    physics: Instance of `mjcf.Physics`.
    bricks: Sequence of `composer.Entity` instances corresponding to bricks.
    base_pos: Position of the base brick in the stack; either a (3,) numpy array
      or a `variation.Variation` that yields such arrays.
    base_quat: Quaternion of the base brick in the stack; either a (4,) numpy
      array or a `variation.Variation` that yields such arrays.
    order: Sequence of indices specifying the order in which to stack the
      bricks.
    random_state: An `np.random.RandomState` instance.
  """
  base_pos = variation.evaluate(base_pos, random_state=random_state)
  base_quat = variation.evaluate(base_quat, random_state=random_state)
  bricks[order[0]].set_pose(physics, position=base_pos, quaternion=base_quat)
  for bottom_idx, top_idx in zip(order[:-1], order[1:]):
    bottom = bricks[bottom_idx]
    top = bricks[top_idx]
    stud_pos = physics.bind(bottom.studs[0, 0]).xpos
    _, quat = bottom.get_pose(physics)
    # The reward function treats top left -> top left and top left -> bottom
    # right configurations as identical, so the orientations of the bricks are
    # randomized so that 50% of the time the top brick is rotated 180 degrees
    # relative to the brick below.
    if random_state.rand() < 0.5:
      quat = quat.copy()
      axis = np.array([0., 0., 1.])
      angle = np.pi
      mjlib.mju_quatIntegrate(quat, axis, angle)
      hole_idx = (-1, -1)
    else:
      hole_idx = (0, 0)
    top.set_pose(physics, quaternion=quat)

    # Set the position of the top brick so that its holes line up with the studs
    # of the brick below.
    offset = physics.bind(top.holes[hole_idx]).xpos
    top_pos = stud_pos - offset
    top.set_pose(physics, position=top_pos)


def _add_or_remove_freejoints(attachment_frames, fixed_indices):
  """Adds or removes freejoints from props.

  Args:
    attachment_frames: A list of `mjcf.Elements` corresponding to attachment
      frames.
    fixed_indices: A list of indices of attachment frames that should be fixed
      to the world (i.e. have their freejoints removed). Freejoints will be
      added to all other elements in `attachment_frames` if they do not already
      possess them.
  """
  for i, frame in enumerate(attachment_frames):
    if i in fixed_indices:
      if frame.freejoint:
        frame.freejoint.remove()
    elif not frame.freejoint:
      frame.add('freejoint')


def _replace_alpha(rgba, alpha=0.3):
  new_rgba = rgba.copy()
  new_rgba[3] = alpha
  return new_rgba


def _hintify(entity, alpha=None):
  """Modifies an entity for use as a 'visual hint'.

  Contacts will be disabled for all geoms within the entity, and its bodies will
  be converted to "mocap" bodies (which are viewed as fixed from the perspective
  of the dynamics). The geom alpha values may also be overridden to render the
  geoms as translucent.

  Args:
    entity: A `composer.Entity`, modified in place.
    alpha: Optional float between 0 and 1, used to override the alpha values for
      all of the geoms in this entity.
  """
  for subentity in entity.iter_entities():
    # TODO(b/112084359): This assumes that all geoms either define explicit RGBA
    #                    values, or inherit from the top-level default. It will
    #                    not correctly handle more complicated hierarchies of
    #                    default classes.
    if (alpha is not None
        and subentity.mjcf_model.default.geom is not None
        and subentity.mjcf_model.default.geom.rgba is not None):
      subentity.mjcf_model.default.geom.rgba = _replace_alpha(
          subentity.mjcf_model.default.geom.rgba, alpha=alpha)
    for body in subentity.mjcf_model.find_all('body'):
      body.mocap = 'true'
    for geom in subentity.mjcf_model.find_all('geom'):
      if alpha is not None and geom.rgba is not None:
        geom.rgba = _replace_alpha(geom.rgba, alpha=alpha)
      geom.contype = 0
      geom.conaffinity = 0


def _stack(obs_settings, num_bricks, moveable_base, randomize_order,
           target_height=None):
  """Configure and instantiate a Stack task.

  Args:
    obs_settings: `observations.ObservationSettings` instance.
    num_bricks: The total number of bricks; must be between 2 and 6.
    moveable_base: Boolean specifying whether or not the bottom brick should
        be moveable.
    randomize_order: Boolean specifying whether to randomize the desired order
      of bricks in the stack at the start of each episode.
    target_height: The target number of bricks in the stack in order to get
      maximum reward. Must be between 2 and `num_bricks`. Defaults to
      `num_bricks`.

  Returns:
    An instance of `Stack`.
  """
  if target_height is None:
    target_height = num_bricks
  arena = arenas.Standard()
  arm = robots.make_arm(obs_settings=obs_settings)
  hand = robots.make_hand(obs_settings=obs_settings)
  return Stack(arena=arena,
               arm=arm,
               hand=hand,
               num_bricks=num_bricks,
               target_height=target_height,
               moveable_base=moveable_base,
               randomize_order=randomize_order,
               obs_settings=obs_settings,
               workspace=_WORKSPACE,
               control_timestep=constants.CONTROL_TIMESTEP)


@registry.add(tags.FEATURES)
def stack_2_bricks_features():
  return _stack(obs_settings=observations.PERFECT_FEATURES, num_bricks=2,
                moveable_base=False, randomize_order=False)


@registry.add(tags.VISION)
def stack_2_bricks_vision():
  return _stack(obs_settings=observations.VISION, num_bricks=2,
                moveable_base=False, randomize_order=False)


@registry.add(tags.FEATURES)
def stack_2_bricks_moveable_base_features():
  return _stack(obs_settings=observations.PERFECT_FEATURES, num_bricks=2,
                moveable_base=True, randomize_order=False)


@registry.add(tags.VISION)
def stack_2_bricks_moveable_base_vision():
  return _stack(obs_settings=observations.VISION, num_bricks=2,
                moveable_base=True, randomize_order=False)


@registry.add(tags.FEATURES)
def stack_3_bricks_features():
  return _stack(obs_settings=observations.PERFECT_FEATURES, num_bricks=3,
                moveable_base=False, randomize_order=False)


@registry.add(tags.VISION)
def stack_3_bricks_vision():
  return _stack(obs_settings=observations.VISION, num_bricks=3,
                moveable_base=False, randomize_order=False)


@registry.add(tags.FEATURES)
def stack_3_bricks_random_order_features():
  return _stack(obs_settings=observations.PERFECT_FEATURES, num_bricks=3,
                moveable_base=False, randomize_order=True)


@registry.add(tags.FEATURES)
def stack_2_of_3_bricks_random_order_features():
  return _stack(obs_settings=observations.PERFECT_FEATURES, num_bricks=3,
                moveable_base=False, randomize_order=True, target_height=2)


@registry.add(tags.VISION)
def stack_2_of_3_bricks_random_order_vision():
  return _stack(obs_settings=observations.VISION, num_bricks=3,
                moveable_base=False, randomize_order=True, target_height=2)


def _reassemble(obs_settings, num_bricks, randomize_initial_order,
                randomize_desired_order):
  """Configure and instantiate a `Reassemble` task.

  Args:
    obs_settings: `observations.ObservationSettings` instance.
    num_bricks: The total number of bricks; must be between 2 and 6.
    randomize_initial_order: Boolean specifying whether to randomize the
      initial order  of bricks in the stack at the start of each episode.
    randomize_desired_order: Boolean specifying whether to independently
      randomize the desired order of bricks in the stack at the start of each
      episode. By default the desired order will be the reverse of the initial
      order, with the exception of the base brick which is always the same as
      in the initial order since it is welded in place.

  Returns:
    An instance of `Reassemble`.
  """
  arena = arenas.Standard()
  arm = robots.make_arm(obs_settings=obs_settings)
  hand = robots.make_hand(obs_settings=obs_settings)
  return Reassemble(arena=arena,
                    arm=arm,
                    hand=hand,
                    num_bricks=num_bricks,
                    randomize_initial_order=randomize_initial_order,
                    randomize_desired_order=randomize_desired_order,
                    obs_settings=obs_settings,
                    workspace=_WORKSPACE,
                    control_timestep=constants.CONTROL_TIMESTEP)


@registry.add(tags.FEATURES)
def reassemble_3_bricks_fixed_order_features():
  return _reassemble(obs_settings=observations.PERFECT_FEATURES, num_bricks=3,
                     randomize_initial_order=False,
                     randomize_desired_order=False)


@registry.add(tags.VISION)
def reassemble_3_bricks_fixed_order_vision():
  return _reassemble(obs_settings=observations.VISION, num_bricks=3,
                     randomize_initial_order=False,
                     randomize_desired_order=False)


@registry.add(tags.FEATURES)
def reassemble_5_bricks_random_order_features():
  return _reassemble(obs_settings=observations.PERFECT_FEATURES, num_bricks=5,
                     randomize_initial_order=True,
                     randomize_desired_order=True)


@registry.add(tags.VISION)
def reassemble_5_bricks_random_order_vision():
  return _reassemble(obs_settings=observations.VISION, num_bricks=5,
                     randomize_initial_order=True,
                     randomize_desired_order=True)
