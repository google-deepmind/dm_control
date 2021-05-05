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
"""Tasks for multi-clip mocap tracking with RL."""

import abc
import collections

import typing
from typing import Any, Callable, Mapping, Optional, Sequence, Set, Text, Union

from absl import logging
from dm_control import composer
from dm_control.composer.observation import observable as base_observable
from dm_control.locomotion.mocap import loader

from dm_control.locomotion.tasks.reference_pose import datasets
from dm_control.locomotion.tasks.reference_pose import types
from dm_control.locomotion.tasks.reference_pose import utils
from dm_control.locomotion.tasks.reference_pose import rewards

from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import transformations as tr

from dm_env import specs

import numpy as np
import tree

if typing.TYPE_CHECKING:
  from dm_control.locomotion.walkers import legacy_base
  from dm_control import mjcf

mjlib = mjbindings.mjlib
DEFAULT_PHYSICS_TIMESTEP = 0.005
_MAX_END_STEP = 10000


def _strip_reference_prefix(dictionary: Mapping[Text, Any],
                            prefix: Text,
                            keep_prefixes: Optional[Set[Text]] = None):
  """Strips a prefix from dictionary keys and remove keys without the prefix.

  Strips a prefix from the keys of a dictionary and removes any key from the
  result dictionary that doesn't match the determined prefix, unless explicitly
  excluded in keep_prefixes.

  E.g.
  dictionary={
    'example_key': 1,
    'example_another_key': 2,
    'doesnt_match': 3,
    'keep_this': 4,
  }, prefix='example_', keep_prefixes=['keep_']

  would return
  {
    'key': 1,
    'another_key': 2,
    'keep_this': 4,
  }

  Args:
    dictionary: The dictionary whose keys will be stripped.
    prefix: The prefix to strip.
    keep_prefixes: Optionally specify prefixes for keys that will be unchanged
      and retained in the result dictionary.

  Returns:
    The dictionary with the modified keys and original values (and unchanged
    keys specified by keep_prefixes).
  """
  keep_prefixes = keep_prefixes or []
  new_dictionary = dict()
  for key in dictionary:
    if key.startswith(prefix):
      key_without_prefix = key.split(prefix)[1]
      # note that this will not copy the underlying array.
      new_dictionary[key_without_prefix] = dictionary[key]
    else:
      for keep_prefix in keep_prefixes:
        if key.startswith(keep_prefix):
          new_dictionary[key] = dictionary[key]

  return new_dictionary


class ReferencePosesTask(composer.Task, metaclass=abc.ABCMeta):
  """Abstract base class for task that uses reference data."""

  def __init__(
      self,
      walker: Callable[..., 'legacy_base.Walker'],
      arena: composer.Arena,
      ref_path: Text,
      ref_steps: Sequence[int],
      dataset: Union[Text, types.ClipCollection],
      termination_error_threshold: float = 0.3,
      prop_termination_error_threshold: float = 0.1,
      min_steps: int = 10,
      reward_type: Text = 'termination_reward',
      physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
      always_init_at_clip_start: bool = False,
      proto_modifier: Optional[Any] = None,
      prop_factory: Optional[Any] = None,
      disable_props: bool = False,
      ghost_offset: Optional[Sequence[Union[int, float]]] = None,
      body_error_multiplier: Union[int, float] = 1.0,
      actuator_force_coeff: float = 0.015,
      enabled_reference_observables: Optional[Sequence[Text]] = None,
  ):
    """Abstract task that uses reference data.

    Args:
      walker: Walker constructor to be used.
      arena: Arena to be used.
      ref_path: Path to the dataset containing reference poses.
      ref_steps: tuples of indices of reference observation. E.g if
        ref_steps=(1, 2, 3) the walker/reference observation at time t will
        contain information from t+1, t+2, t+3.
      dataset: A ClipCollection instance or a name of a dataset that appears as
        a key in DATASETS in datasets.py
      termination_error_threshold: Error threshold for episode terminations for
        hand body position and joint error only.
      prop_termination_error_threshold: Error threshold for episode terminations
        for prop position.
      min_steps: minimum number of steps within an episode. This argument
        determines the latest allowable starting point within a given reference
        trajectory.
      reward_type: type of reward to use, must be a string that appears as a key
        in the REWARD_FN dict in rewards.py.
      physics_timestep: Physics timestep to use for simulation.
      always_init_at_clip_start: only initialize epsidodes at the start of a
        reference trajectory.
      proto_modifier: Optional proto modifier to modify reference trajectories,
        e.g. adding a vertical offset.
      prop_factory: Optional function that takes the mocap proto and returns
        the corresponding props for the trajectory.
      disable_props: If prop_factory is specified but disable_props is True,
        no props will be created.
      ghost_offset: if not None, include a ghost rendering of the walker with
        the reference pose at the specified position offset.
      body_error_multiplier: A multiplier that is applied to the body error term
        when determining failure termination condition.
      actuator_force_coeff: A coefficient for the actuator force reward channel.
      enabled_reference_observables: Optional iterable of enabled observables.
        If not specified, a reasonable default set will be enabled.
    """
    self._ref_steps = np.sort(ref_steps)
    self._max_ref_step = self._ref_steps[-1]
    self._termination_error_threshold = termination_error_threshold
    self._prop_termination_error_threshold = prop_termination_error_threshold
    self._reward_fn = rewards.get_reward(reward_type)
    self._reward_keys = rewards.get_reward_channels(reward_type)
    self._min_steps = min_steps
    self._always_init_at_clip_start = always_init_at_clip_start
    self._ghost_offset = ghost_offset
    self._body_error_multiplier = body_error_multiplier
    self._actuator_force_coeff = actuator_force_coeff
    logging.info('Reward type %s', reward_type)

    if isinstance(dataset, Text):
      try:
        dataset = datasets.DATASETS[dataset]
      except KeyError:
        logging.error('Dataset %s not found in datasets.py', dataset)
        raise
    self._load_reference_data(
        ref_path=ref_path, proto_modifier=proto_modifier, dataset=dataset)

    self._get_possible_starts()

    logging.info('%d starting points found.', len(self._possible_starts))

    # load a dummy trajectory
    self._current_clip_index = 0
    self._current_clip = self._loader.get_trajectory(
        self._dataset.ids[0], zero_out_velocities=False)
    # Create the environment.
    self._arena = arena
    self._walker = utils.add_walker(walker, self._arena)
    self.set_timesteps(
        physics_timestep=physics_timestep,
        control_timestep=self._current_clip.dt)

    # Identify the desired body components.
    try:
      walker_bodies = self._walker.mocap_tracking_bodies
    except AttributeError:
      logging.info('Walker must implement mocap bodies for this task.')
      raise

    walker_bodies_names = [bdy.name for bdy in walker_bodies]
    self._body_idxs = np.array(
        [walker_bodies_names.index(bdy) for bdy in walker_bodies_names])

    self._prop_factory = prop_factory
    if disable_props:
      self._props = []
    else:
      self._props = self._current_clip.create_props(prop_factory=prop_factory)
    for prop in self._props:
      self._arena.add_free_entity(prop)

    # Create the observables.
    self._add_observables(enabled_reference_observables)

    # initialize counters etc.
    self._time_step = 0
    self._current_start_time = 0.0
    self._last_step = 0
    self._current_clip_index = 0
    self._reference_observations = dict()
    self._end_mocap = False
    self._should_truncate = False

    # Set up required dummy quantities for observations
    self._prop_prefixes = []

    self._disable_props = disable_props
    if not disable_props:
      if len(self._props) == 1:
        self._prop_prefixes += ['prop/']
      else:
        self._prop_prefixes += [f'prop_{i}/' for i in range(len(self._props))]
    self._clip_reference_features = self._current_clip.as_dict()
    self._strip_reference_prefix()

    self._walker_joints = self._clip_reference_features['joints'][0]
    self._walker_features = tree.map_structure(lambda x: x[0],
                                               self._clip_reference_features)
    self._walker_features_prev = tree.map_structure(
        lambda x: x[0], self._clip_reference_features)

    self._current_reference_features = dict()
    self._reference_ego_bodies_quats = collections.defaultdict(dict)
    # if requested add ghost body to visualize motion capture reference.
    if self._ghost_offset is not None:
      self._ghost = utils.add_walker(
          walker, self._arena, name='ghost', ghost=True)
      self._ghost.observables.disable_all()

      if disable_props:
        self._ghost_props = []
      else:
        self._ghost_props = self._current_clip.create_props(
            prop_factory=self._ghost_prop_factory)
        for prop in self._ghost_props:
          self._arena.add_free_entity(prop)
          prop.observables.disable_all()
    else:
      self._ghost_props = []

    # initialize reward channels
    self._reset_reward_channels()

  def _strip_reference_prefix(self):
    self._clip_reference_features = _strip_reference_prefix(
        self._clip_reference_features,
        'walker/',
        keep_prefixes=self._prop_prefixes)

    positions = []
    quaternions = []
    for prefix in self._prop_prefixes:
      position_key, quaternion_key = f'{prefix}position', f'{prefix}quaternion'
      positions.append(self._clip_reference_features[position_key])
      quaternions.append(self._clip_reference_features[quaternion_key])
      del self._clip_reference_features[position_key]
      del self._clip_reference_features[quaternion_key]
    # positions has dimension (#props, #timesteps, 3). However, the convention
    # for reference observations is (#timesteps, #props, 3). Therefore we
    # transpose the dimensions by specifying the desired positions in the list
    # for each dimension as an argument to np.transpose.
    axes = [1, 0, 2]
    if self._prop_prefixes:
      self._clip_reference_features['prop_positions'] = np.transpose(
          positions, axes=axes)
      self._clip_reference_features['prop_quaternions'] = np.transpose(
          quaternions, axes=axes)

  def _ghost_prop_factory(self, prop_proto, priority_friction=False):
    if self._prop_factory is None:
      return None

    prop = self._prop_factory(prop_proto, priority_friction=priority_friction)
    for geom in prop.mjcf_model.find_all('geom'):
      geom.set_attributes(contype=0, conaffinity=0, rgba=(0.5, 0.5, 0.5, .999))
    prop.observables.disable_all()
    return prop

  def _load_reference_data(self, ref_path, proto_modifier,
                           dataset: types.ClipCollection):
    self._loader = loader.HDF5TrajectoryLoader(
        ref_path, proto_modifier=proto_modifier)

    self._dataset = dataset
    self._num_clips = len(self._dataset.ids)

    if self._dataset.end_steps is None:
      # load all trajectories to infer clip end steps.
      self._all_clips = [
          self._loader.get_trajectory(  # pylint: disable=g-complex-comprehension
              clip_id,
              start_step=clip_start_step,
              end_step=_MAX_END_STEP) for clip_id, clip_start_step in zip(
                  self._dataset.ids, self._dataset.start_steps)
      ]
      # infer clip end steps to set sampling distribution
      self._dataset.end_steps = tuple(clip.end_step for clip in self._all_clips)
    else:
      self._all_clips = [None] * self._num_clips

  def _add_observables(self, enabled_reference_observables):

    # pylint: disable=g-long-lambda
    self._walker.observables.add_observable(
        'reference_rel_joints',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_rel_joints']))
    self._walker.observables.add_observable(
        'reference_rel_bodies_pos_global',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_rel_bodies_pos_global']))
    self._walker.observables.add_observable(
        'reference_rel_bodies_quats',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_rel_bodies_quats']))
    self._walker.observables.add_observable(
        'reference_rel_bodies_pos_local',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_rel_bodies_pos_local']))
    self._walker.observables.add_observable(
        'reference_ego_bodies_quats',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_ego_bodies_quats']))
    self._walker.observables.add_observable(
        'reference_rel_root_quat',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_rel_root_quat']))
    self._walker.observables.add_observable(
        'reference_rel_root_pos_local',
        base_observable.Generic(lambda _: self._reference_observations[
            'walker/reference_rel_root_pos_local']))
    # pylint: enable=g-long-lambda
    self._walker.observables.add_observable(
        'reference_appendages_pos',
        base_observable.Generic(self.get_reference_appendages_pos))

    if enabled_reference_observables:
      for name, observable in self.observables.items():
        observable.enabled = name in enabled_reference_observables
    self._walker.observables.add_observable(
        'clip_id', base_observable.Generic(self.get_clip_id))
    self._walker.observables.add_observable(
        'velocimeter_control', base_observable.Generic(self.get_veloc_control))
    self._walker.observables.add_observable(
        'gyro_control', base_observable.Generic(self.get_gyro_control))
    self._walker.observables.add_observable(
        'joints_vel_control',
        base_observable.Generic(self.get_joints_vel_control))

    self._arena.observables.add_observable(
        'reference_props_pos_global',
        base_observable.Generic(self.get_reference_props_pos_global))
    self._arena.observables.add_observable(
        'reference_props_quat_global',
        base_observable.Generic(self.get_reference_props_quat_global))
    observables = []
    observables += self._walker.observables.proprioception
    observables += self._walker.observables.kinematic_sensors
    observables += self._walker.observables.dynamic_sensors

    for observable in observables:
      observable.enabled = True

    for prop in self._props:
      prop.observables.position.enabled = True
      prop.observables.orientation.enabled = True

  def _get_possible_starts(self):
    # List all possible (clip, step) starting points.
    self._possible_starts = []
    self._start_probabilities = []
    dataset = self._dataset
    for clip_number, (start, end, weight) in enumerate(
        zip(dataset.start_steps, dataset.end_steps, dataset.weights)):
      # length - required lookahead - minimum number of steps
      last_possible_start = end - self._max_ref_step - self._min_steps

      if self._always_init_at_clip_start:
        self._possible_starts += [(clip_number, start)]
        self._start_probabilities += [weight]
      else:
        self._possible_starts += [
            (clip_number, j) for j in range(start, last_possible_start)
        ]
        self._start_probabilities += [
            weight for _ in range(start, last_possible_start)
        ]

    # normalize start probabilities
    self._start_probabilities = np.array(self._start_probabilities) / np.sum(
        self._start_probabilities)

  def initialize_episode_mjcf(self, random_state: np.random.RandomState):
    if hasattr(self._arena, 'regenerate'):
      self._arena.regenerate(random_state)

    # Get a new clip here to instantiate the right prop for this episode.
    self._get_clip_to_track(random_state)
    # Set up props.
    # We call the prop factory here to ensure that props can change per episode.
    for prop in self._props:
      prop.detach()
      del prop

    if not self._disable_props:
      self._props = self._current_clip.create_props(
          prop_factory=self._prop_factory)
      for prop in self._props:
        self._arena.add_free_entity(prop)
        prop.observables.position.enabled = True
        prop.observables.orientation.enabled = True

      if self._ghost_offset is not None:
        for prop in self._ghost_props:
          prop.detach()
          del prop
        self._ghost_props = self._current_clip.create_props(
            prop_factory=self._ghost_prop_factory)
        for prop in self._ghost_props:
          self._arena.add_free_entity(prop)
          prop.observables.disable_all()

  def _get_clip_to_track(self, random_state: np.random.RandomState):
    # Randomly select a starting point.
    index = random_state.choice(
        len(self._possible_starts), p=self._start_probabilities)
    clip_index, start_step = self._possible_starts[index]

    self._current_clip_index = clip_index
    clip_id = self._dataset.ids[self._current_clip_index]

    if self._all_clips[self._current_clip_index] is None:
      # fetch selected trajectory
      logging.info('Loading clip %s', clip_id)
      self._all_clips[self._current_clip_index] = self._loader.get_trajectory(
          clip_id,
          start_step=self._dataset.start_steps[self._current_clip_index],
          end_step=self._dataset.end_steps[self._current_clip_index],
          zero_out_velocities=False)
    self._current_clip = self._all_clips[self._current_clip_index]
    self._clip_reference_features = self._current_clip.as_dict()
    self._strip_reference_prefix()

    # The reference features are already restricted to
    # clip_start_step:clip_end_step. However start_step is in
    # [clip_start_step:clip_end_step]. Hence we subtract clip_start_step to
    # obtain a valid index for the reference features.
    self._time_step = start_step - self._dataset.start_steps[
        self._current_clip_index]
    self._current_start_time = (start_step - self._dataset.start_steps[
        self._current_clip_index]) * self._current_clip.dt
    self._last_step = len(
        self._clip_reference_features['joints']) - self._max_ref_step - 1
    logging.info('Mocap %s at step %d with remaining length %d.', clip_id,
                 start_step, self._last_step - start_step)

  def initialize_episode(self, physics: 'mjcf.Physics',
                         random_state: np.random.RandomState):
    """Randomly selects a starting point and set the walker."""

    # Set the walker at the beginning of the clip.
    self._set_walker(physics)
    self._walker_features = utils.get_features(
        physics, self._walker, props=self._props)
    self._walker_features_prev = self._walker_features.copy()

    self._walker_joints = np.array(physics.bind(self._walker.mocap_joints).qpos)  # pytype: disable=attribute-error

    # compute initial error
    self._compute_termination_error()
    # assert error is 0 at initialization. In particular this will prevent
    # a proto/walker mismatch.
    if self._termination_error > 1e-2:
      raise ValueError(('The termination exceeds 1e-2 at initialization. '
                        'This is likely due to a proto/walker mismatch.'))

    self._update_ghost(physics)
    self._reference_observations.update(
        self.get_all_reference_observations(physics))

    # reset reward channels
    self._reset_reward_channels()

  def _reset_reward_channels(self):
    if self._reward_keys:
      self.last_reward_channels = collections.OrderedDict([
          (k, 0.0) for k in self._reward_keys
      ])
    else:
      self.last_reward_channels = None

  def _compute_termination_error(self):
    target_joints = self._clip_reference_features['joints'][self._time_step]
    error_joints = np.mean(np.abs(target_joints - self._walker_joints))
    target_bodies = self._clip_reference_features['body_positions'][
        self._time_step]
    error_bodies = np.mean(
        np.abs((target_bodies -
                self._walker_features['body_positions'])[self._body_idxs]))
    self._termination_error = (
        0.5 * self._body_error_multiplier * error_bodies + 0.5 * error_joints)

    if self._props:
      target_props = self._clip_reference_features['prop_positions'][
          self._time_step]
      cur_props = self._walker_features['prop_positions']
      # Separately compute prop termination error as euclidean distance.
      self._prop_termination_error = np.mean(
          np.linalg.norm(target_props - cur_props, axis=-1))

  def before_step(self, physics: 'mjcf.Physics', action,
                  random_state: np.random.RandomState):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics: 'mjcf.Physics',
                 random_state: np.random.RandomState):
    """Update the data after step."""
    del random_state  # unused by after_step.

    self._walker_features_prev = self._walker_features.copy()

  def after_compile(self, physics: 'mjcf.Physics',
                    random_state: np.random.RandomState):
    # populate reference observations field to initialize observations.
    if not self._reference_observations:
      self._reference_observations.update(
          self.get_all_reference_observations(physics))

  def should_terminate_episode(self, physics: 'mjcf.Physics'):
    del physics  # physics unused by should_terminate_episode.

    if self._should_truncate:
      logging.info('Truncate with error %f.', self._termination_error)
      return True

    if self._end_mocap:
      logging.info('End of mocap.')
      return True

    return False

  def get_discount(self, physics: 'mjcf.Physics'):
    del physics  # unused by get_discount.

    if self._should_truncate:
      return 0.0
    return 1.0

  def get_reference_rel_joints(self, physics: 'mjcf.Physics'):
    """Observation of the reference joints relative to walker."""
    del physics  # physics unused by reference observations.
    time_steps = self._time_step + self._ref_steps
    diff = (self._clip_reference_features['joints'][time_steps] -
            self._walker_joints)
    return diff[:, self._walker.mocap_to_observable_joint_order].flatten()

  def get_reference_rel_bodies_pos_global(self, physics: 'mjcf.Physics'):
    """Observation of the reference bodies relative to walker."""
    del physics  # physics unused by reference observations.

    time_steps = self._time_step + self._ref_steps
    return (self._clip_reference_features['body_positions'][time_steps] -
            self._walker_features['body_positions'])[:,
                                                     self._body_idxs].flatten()

  def get_reference_rel_bodies_quats(self, physics: 'mjcf.Physics'):
    """Observation of the reference bodies quats relative to walker."""
    del physics  # physics unused by reference observations.

    time_steps = self._time_step + self._ref_steps
    obs = []
    for t in time_steps:
      for b in self._body_idxs:
        obs.append(
            tr.quat_diff(
                self._walker_features['body_quaternions'][b, :],
                self._clip_reference_features['body_quaternions'][t, b, :]))
    return np.concatenate([o.flatten() for o in obs])

  def get_reference_rel_bodies_pos_local(self, physics: 'mjcf.Physics'):
    """Observation of the reference bodies relative to walker in local frame."""
    time_steps = self._time_step + self._ref_steps
    obs = self._walker.transform_vec_to_egocentric_frame(
        physics, (self._clip_reference_features['body_positions'][time_steps] -
                  self._walker_features['body_positions'])[:, self._body_idxs])
    return np.concatenate([o.flatten() for o in obs])

  def get_reference_ego_bodies_quats(self, unused_physics: 'mjcf.Physics'):
    """Body quat of the reference relative to the reference root quat."""
    time_steps = self._time_step + self._ref_steps
    obs = []
    quats_for_clip = self._reference_ego_bodies_quats[self._current_clip_index]
    for t in time_steps:
      if t not in quats_for_clip:
        root_quat = self._clip_reference_features['quaternion'][t, :]
        quats_for_clip[t] = [
            tr.quat_diff(  # pylint: disable=g-complex-comprehension
                root_quat,
                self._clip_reference_features['body_quaternions'][t, b, :])
            for b in self._body_idxs
        ]
      obs.extend(quats_for_clip[t])
    return np.concatenate([o.flatten() for o in obs])

  def get_reference_rel_root_quat(self, physics: 'mjcf.Physics'):
    """Root quaternion of reference relative to current root quat."""
    del physics  # physics unused by reference observations.

    time_steps = self._time_step + self._ref_steps
    obs = []
    for t in time_steps:
      obs.append(
          tr.quat_diff(self._walker_features['quaternion'],
                       self._clip_reference_features['quaternion'][t, :]))
    return np.concatenate([o.flatten() for o in obs])

  def get_reference_appendages_pos(self, physics: 'mjcf.Physics'):
    """Reference appendage positions in reference frame."""
    del physics  # physics unused by reference observations.

    time_steps = self._time_step + self._ref_steps
    return self._clip_reference_features['appendages'][time_steps].flatten()

  def get_reference_rel_root_pos_local(self, physics: 'mjcf.Physics'):
    """Reference position relative to current root position in root frame."""
    time_steps = self._time_step + self._ref_steps
    obs = self._walker.transform_vec_to_egocentric_frame(
        physics, (self._clip_reference_features['position'][time_steps] -
                  self._walker_features['position']))
    return np.concatenate([o.flatten() for o in obs])

  def get_reference_props_pos_global(self, physics: 'mjcf.Physics'):
    time_steps = self._time_step + self._ref_steps
    # size N x 3 where N = # of props
    if self._props:
      return self._clip_reference_features['prop_positions'][
          time_steps].flatten()
    else:
      return []

  def get_reference_props_quat_global(self, physics: 'mjcf.Physics'):
    time_steps = self._time_step + self._ref_steps
    # size N x 4 where N = # of props
    if self._props:
      return self._clip_reference_features['prop_quaternions'][
          time_steps].flatten()
    else:
      return []

  def get_veloc_control(self, physics: 'mjcf.Physics'):
    """Velocity measurements in the prev root frame at the control timestep."""
    del physics  # physics unused by get_veloc_control.

    rmat_prev = tr.quat_to_mat(self._walker_features_prev['quaternion'])[:3, :3]
    veloc_world = (
        self._walker_features['position'] -
        self._walker_features_prev['position']) / self._control_timestep
    return np.dot(veloc_world, rmat_prev)

  def get_gyro_control(self, physics: 'mjcf.Physics'):
    """Gyro measurements in the prev root frame at the control timestep."""
    del physics  # physics unused by get_gyro_control.

    quat_curr, quat_prev = (self._walker_features['quaternion'],
                            self._walker_features_prev['quaternion'])
    normed_diff = tr.quat_diff(quat_prev, quat_curr)
    normed_diff /= np.linalg.norm(normed_diff)
    return tr.quat_to_axisangle(normed_diff) / self._control_timestep

  def get_joints_vel_control(self, physics: 'mjcf.Physics'):
    """Joint velocity measurements at the control timestep."""
    del physics  # physics unused by get_joints_vel_control.

    joints_curr, joints_prev = (self._walker_features['joints'],
                                self._walker_features_prev['joints'])
    return (joints_curr - joints_prev)[
        self._walker.mocap_to_observable_joint_order]/self._control_timestep

  def get_clip_id(self, physics: 'mjcf.Physics'):
    """Observation of the clip id."""
    del physics  # physics unused by get_clip_id.

    return np.array([self._current_clip_index])

  def get_all_reference_observations(self, physics: 'mjcf.Physics'):
    reference_observations = dict()
    reference_observations[
        'walker/reference_rel_bodies_pos_local'] = self.get_reference_rel_bodies_pos_local(
            physics)
    reference_observations[
        'walker/reference_rel_joints'] = self.get_reference_rel_joints(physics)
    reference_observations[
        'walker/reference_rel_bodies_pos_global'] = self.get_reference_rel_bodies_pos_global(
            physics)
    reference_observations[
        'walker/reference_ego_bodies_quats'] = self.get_reference_ego_bodies_quats(
            physics)
    reference_observations[
        'walker/reference_rel_root_quat'] = self.get_reference_rel_root_quat(
            physics)
    reference_observations[
        'walker/reference_rel_bodies_quats'] = self.get_reference_rel_bodies_quats(
            physics)
    reference_observations[
        'walker/reference_rel_root_pos_local'] = self.get_reference_rel_root_pos_local(
            physics)
    if self._props:
      reference_observations[
          'props/reference_pos_global'] = self.get_reference_props_pos_global(
              physics)
      reference_observations[
          'props/reference_quat_global'] = self.get_reference_props_quat_global(
              physics)
    return reference_observations

  def get_reward(self, physics: 'mjcf.Physics') -> float:
    reward, unused_debug_outputs, reward_channels = self._reward_fn(
        termination_error=self._termination_error,
        termination_error_threshold=self._termination_error_threshold,
        reference_features=self._current_reference_features,
        walker_features=self._walker_features,
        reference_observations=self._reference_observations)

    if 'actuator_force' in self._reward_keys:
      reward_channels['actuator_force'] = -self._actuator_force_coeff*np.mean(
          np.square(self._walker.actuator_force(physics)))

    self._should_truncate = self._termination_error > self._termination_error_threshold

    if self._props:
      prop_termination = self._prop_termination_error > self._prop_termination_error_threshold
      self._should_truncate = self._should_truncate or prop_termination

    self.last_reward_channels = reward_channels
    return reward

  def _set_walker(self, physics: 'mjcf.Physics'):
    timestep_features = tree.map_structure(lambda x: x[self._time_step],
                                           self._clip_reference_features)
    utils.set_walker_from_features(physics, self._walker, timestep_features)
    if self._props:
      utils.set_props_from_features(physics, self._props, timestep_features)
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

  def _update_ghost(self, physics: 'mjcf.Physics'):
    if self._ghost_offset is not None:
      target = tree.map_structure(lambda x: x[self._time_step],
                                  self._clip_reference_features)
      utils.set_walker_from_features(physics, self._ghost, target,
                                     self._ghost_offset)
      if self._ghost_props:
        utils.set_props_from_features(
            physics, self._ghost_props, target, z_offset=self._ghost_offset)
      mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

  def action_spec(self, physics: 'mjcf.Physics'):
    """Action spec of the walker only."""
    ctrl = physics.bind(self._walker.actuators).ctrl  # pytype: disable=attribute-error
    shape = ctrl.shape
    dtype = ctrl.dtype
    minimum = []
    maximum = []
    for actuator in self._walker.actuators:
      if physics.bind(actuator).ctrllimited:  # pytype: disable=attribute-error
        ctrlrange = physics.bind(actuator).ctrlrange  # pytype: disable=attribute-error
        minimum.append(ctrlrange[0])
        maximum.append(ctrlrange[1])
      else:
        minimum.append(-float('inf'))
        maximum.append(float('inf'))
    return specs.BoundedArray(
        shape=shape,
        dtype=dtype,
        minimum=np.asarray(minimum, dtype=dtype),
        maximum=np.asarray(maximum, dtype=dtype),
        name='\t'.join(actuator.full_identifier  # pytype: disable=attribute-error
                       for actuator in self._walker.actuators))

  @abc.abstractproperty
  def name(self):
    raise NotImplementedError

  @property
  def root_entity(self):
    return self._arena


class MultiClipMocapTracking(ReferencePosesTask):
  """Task for multi-clip mocap tracking."""

  def __init__(
      self,
      walker: Callable[..., 'legacy_base.Walker'],
      arena: composer.Arena,
      ref_path: Text,
      ref_steps: Sequence[int],
      dataset: Union[Text, Sequence[Any]],
      termination_error_threshold: float = 0.3,
      prop_termination_error_threshold: float = 0.1,
      min_steps: int = 10,
      reward_type: Text = 'termination_reward',
      physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
      always_init_at_clip_start: bool = False,
      proto_modifier: Optional[Any] = None,
      prop_factory: Optional[Any] = None,
      disable_props: bool = True,
      ghost_offset: Optional[Sequence[Union[int, float]]] = None,
      body_error_multiplier: Union[int, float] = 1.0,
      actuator_force_coeff: float = 0.015,
      enabled_reference_observables: Optional[Sequence[Text]] = None,
  ):
    """Mocap tracking task.

    Args:
      walker: Walker constructor to be used.
      arena: Arena to be used.
      ref_path: Path to the dataset containing reference poses.
      ref_steps: tuples of indices of reference observation. E.g if
        ref_steps=(1, 2, 3) the walker/reference observation at time t will
        contain information from t+1, t+2, t+3.
      dataset: dataset: A ClipCollection instance or a named dataset that
        appears as a key in DATASETS in datasets.py
      termination_error_threshold: Error threshold for episode terminations for
        hand body position and joint error only.
      prop_termination_error_threshold: Error threshold for episode terminations
        for prop position.
      min_steps: minimum number of steps within an episode. This argument
        determines the latest allowable starting point within a given reference
        trajectory.
      reward_type: type of reward to use, must be a string that appears as a key
        in the REWARD_FN dict in rewards.py.
      physics_timestep: Physics timestep to use for simulation.
      always_init_at_clip_start: only initialize epsidodes at the start of a
        reference trajectory.
      proto_modifier: Optional proto modifier to modify reference trajectories,
        e.g. adding a vertical offset.
      prop_factory: Optional function that takes the mocap proto and returns
        the corresponding props for the trajectory.
      disable_props: If prop_factory is specified but disable_props is True,
        no props will be created.
      ghost_offset: if not None, include a ghost rendering of the walker with
        the reference pose at the specified position offset.
      body_error_multiplier: A multiplier that is applied to the body error term
        when determining failure termination condition.
      actuator_force_coeff: A coefficient for the actuator force reward channel.
      enabled_reference_observables: Optional iterable of enabled observables.
        If not specified, a reasonable default set will be enabled.
    """
    super().__init__(
        walker=walker,
        arena=arena,
        ref_path=ref_path,
        ref_steps=ref_steps,
        termination_error_threshold=termination_error_threshold,
        prop_termination_error_threshold=prop_termination_error_threshold,
        min_steps=min_steps,
        dataset=dataset,
        reward_type=reward_type,
        physics_timestep=physics_timestep,
        always_init_at_clip_start=always_init_at_clip_start,
        proto_modifier=proto_modifier,
        prop_factory=prop_factory,
        disable_props=disable_props,
        ghost_offset=ghost_offset,
        body_error_multiplier=body_error_multiplier,
        actuator_force_coeff=actuator_force_coeff,
        enabled_reference_observables=enabled_reference_observables)
    self._walker.observables.add_observable(
        'time_in_clip',
        base_observable.Generic(self.get_normalized_time_in_clip))

  def after_step(self, physics: 'mjcf.Physics', random_state):
    """Update the data after step."""
    super().after_step(physics, random_state)
    self._time_step += 1

    # Update the walker's data for this timestep.
    self._walker_features = utils.get_features(
        physics, self._walker, props=self._props)
    # features for default error
    self._walker_joints = np.array(physics.bind(self._walker.mocap_joints).qpos)  # pytype: disable=attribute-error

    self._current_reference_features = {
        k: v[self._time_step].copy()
        for k, v in self._clip_reference_features.items()
    }

    # Error.
    self._compute_termination_error()

    # Terminate based on the error.
    self._end_mocap = self._time_step == self._last_step

    self._reference_observations.update(
        self.get_all_reference_observations(physics))

    self._update_ghost(physics)

  def get_normalized_time_in_clip(self, physics: 'mjcf.Physics'):
    """Observation of the normalized time in the mocap clip."""
    normalized_time_in_clip = (self._current_start_time +
                               physics.time()) / self._current_clip.duration
    return np.array([normalized_time_in_clip])

  @property
  def name(self):
    return 'MultiClipMocapTracking'


class PlaybackTask(ReferencePosesTask):
  """Simple task to visualize mocap data."""

  def __init__(self,
               walker,
               arena,
               ref_path: Text,
               dataset: Union[Text, types.ClipCollection],
               proto_modifier: Optional[Any] = None,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP):
    super().__init__(walker=walker,
                     arena=arena,
                     ref_path=ref_path,
                     ref_steps=(1,),
                     dataset=dataset,
                     termination_error_threshold=np.inf,
                     physics_timestep=physics_timestep,
                     always_init_at_clip_start=True,
                     proto_modifier=proto_modifier)
    self._current_clip_index = -1

  def _get_clip_to_track(self, random_state: np.random.RandomState):
    self._current_clip_index = (self._current_clip_index + 1) % self._num_clips

    start_step = self._dataset.start_steps[self._current_clip_index]
    clip_id = self._dataset.ids[self._current_clip_index]
    logging.info('Showing clip %d of %d, clip id %s',
                 self._current_clip_index+1, self._num_clips, clip_id)

    if self._all_clips[self._current_clip_index] is None:
      # fetch selected trajectory
      logging.info('Loading clip %s', clip_id)
      self._all_clips[self._current_clip_index] = self._loader.get_trajectory(
          clip_id,
          start_step=self._dataset.start_steps[self._current_clip_index],
          end_step=self._dataset.end_steps[self._current_clip_index],
          zero_out_velocities=False)
    self._current_clip = self._all_clips[self._current_clip_index]
    self._clip_reference_features = self._current_clip.as_dict()
    self._clip_reference_features = _strip_reference_prefix(
        self._clip_reference_features, 'walker/')
    # The reference features are already restricted to
    # clip_start_step:clip_end_step. However start_step is in
    # [clip_start_step:clip_end_step]. Hence we subtract clip_start_step to
    # obtain a valid index for the reference features.
    self._time_step = start_step - self._dataset.start_steps[
        self._current_clip_index]
    self._current_start_time = (start_step - self._dataset.start_steps[
        self._current_clip_index]) * self._current_clip.dt
    self._last_step = len(
        self._clip_reference_features['joints']) - self._max_ref_step - 1
    logging.info('Mocap %s at step %d with remaining length %d.', clip_id,
                 start_step, self._last_step - start_step)

  def _set_walker(self, physics: 'mjcf.Physics'):
    timestep_features = tree.map_structure(lambda x: x[self._time_step],
                                           self._clip_reference_features)
    utils.set_walker_from_features(physics, self._walker, timestep_features)
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

  def after_step(self, physics, random_state: np.random.RandomState):
    super().after_step(physics, random_state)
    self._time_step += 1

    self._set_walker(physics)
    self._end_mocap = self._time_step == self._last_step

  def get_reward(self, physics):
    return 0.0

  @property
  def name(self):
    return 'PlaybackTask'
