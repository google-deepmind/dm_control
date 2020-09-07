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
from typing import Any, Callable, Mapping, Optional, Sequence, Text, Union

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
import six
import tree

if typing.TYPE_CHECKING:
  from dm_control.locomotion.walkers import base
  from dm_control import mjcf

mjlib = mjbindings.mjlib
DEFAULT_PHYSICS_TIMESTEP = 0.005
_MAX_END_STEP = 10000


def _strip_reference_prefix(dictionary: Mapping[Text, Any], prefix: Text):
  new_dictionary = dict()
  for key in list(dictionary.keys()):
    if key.startswith(prefix):
      key_without_prefix = key.split(prefix)[1]
      # note that this will not copy the underlying array.
      new_dictionary[key_without_prefix] = dictionary[key]
  return new_dictionary


@six.add_metaclass(abc.ABCMeta)
class ReferencePosesTask(composer.Task):
  """Abstract base class for task that uses reference data."""

  def __init__(
      self,
      walker: Union['base.Walker', Callable],  # pylint: disable=g-bare-generic
      arena: composer.Arena,
      ref_path: Text,
      ref_steps: Sequence[int],
      dataset: Union[Text, types.ClipCollection],
      termination_error_threshold: float = 0.3,
      min_steps: int = 10,
      reward_type: Text = 'default',
      physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
      always_init_at_clip_start: bool = False,
      proto_modifier: Optional[Any] = None,
      ghost_offset: Optional[Sequence[Union[int, float]]] = None,
      body_error_multiplier: Union[int, float] = 1.0,
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
      termination_error_threshold: Error threshold for episode terminations.
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
      ghost_offset: if not None, include a ghost rendering of the walker with
        the reference pose at the specified position offset.
      body_error_multiplier: A multiplier that is applied to the body error term
        when determining failure termination condition.
    """
    self._ref_steps = np.sort(ref_steps)
    self._max_ref_step = self._ref_steps[-1]
    self._termination_error_threshold = termination_error_threshold
    self._reward_fn = rewards.get_reward(reward_type)
    self._reward_keys = rewards.get_reward_channels(reward_type)
    self._min_steps = min_steps
    self._always_init_at_clip_start = always_init_at_clip_start
    self._ghost_offset = ghost_offset
    self._body_error_multiplier = body_error_multiplier
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
    # Create the observables.
    self._add_observables()

    # initialize counters etc.
    self._time_step = 0
    self._current_start_time = 0.0
    self._last_step = 0
    self._current_clip_index = 0
    self._end_mocap = False
    self._should_truncate = False

    # Set up required dummy quantities for observations
    self._clip_reference_features = self._current_clip.as_dict()
    self._clip_reference_features = _strip_reference_prefix(
        self._clip_reference_features, 'walker/')

    self._walker_joints = self._clip_reference_features['joints'][0]
    self._walker_features = tree.map_structure(lambda x: x[0],
                                               self._clip_reference_features)
    self._walker_features_prev = tree.map_structure(
        lambda x: x[0], self._clip_reference_features)

    self._current_reference_features = dict()

    # if requested add ghost body to visualize motion capture reference.
    if self._ghost_offset is not None:
      self._ghost = utils.add_walker(walker, self._arena, 'ghost', ghost=True)

    # initialize reward channels
    self._reset_reward_channels()

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

  def _add_observables(self):
    observables = []
    observables += self._walker.observables.proprioception
    observables += self._walker.observables.kinematic_sensors
    observables += self._walker.observables.dynamic_sensors

    for observable in observables:
      observable.enabled = True
    self._walker.observables.add_observable(
        'clip_id', base_observable.Generic(self.get_clip_id))
    self._walker.observables.add_observable(
        'reference_rel_joints',
        base_observable.Generic(self.get_reference_rel_joints))
    self._walker.observables.add_observable(
        'reference_rel_bodies_pos_global',
        base_observable.Generic(self.get_reference_rel_bodies_pos_global))
    self._walker.observables.add_observable(
        'reference_rel_bodies_quats',
        base_observable.Generic(self.get_reference_rel_bodies_quats))
    self._walker.observables.add_observable(
        'reference_rel_bodies_pos_local',
        base_observable.Generic(self.get_reference_rel_bodies_pos_local))
    self._walker.observables.add_observable(
        'reference_ego_bodies_quats',
        base_observable.Generic(self.get_reference_ego_bodies_quats))
    self._walker.observables.add_observable(
        'reference_rel_root_quat',
        base_observable.Generic(self.get_reference_rel_root_quat))
    self._walker.observables.add_observable(
        'reference_rel_root_pos_local',
        base_observable.Generic(self.get_reference_rel_root_pos_local))
    self._walker.observables.add_observable(
        'reference_appendages_pos',
        base_observable.Generic(self.get_reference_appendages_pos))
    self._walker.observables.add_observable(
        'velocimeter_control', base_observable.Generic(self.get_veloc_control))
    self._walker.observables.add_observable(
        'gyro_control', base_observable.Generic(self.get_gyro_control))
    self._walker.observables.add_observable(
        'joints_vel_control',
        base_observable.Generic(self.get_joints_vel_control))

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

  def initialize_episode(self, physics: 'mjcf.Physics',
                         random_state: np.random.RandomState):
    """Randomly selects a starting point and set the walker."""

    self._get_clip_to_track(random_state)

    # Set the walker at the beginning of the clip.
    self._set_walker(physics)
    self._walker_features = utils.get_features(physics, self._walker)
    self._walker_features_prev = utils.get_features(physics, self._walker)

    self._walker_joints = np.array(physics.bind(self._walker.mocap_joints).qpos)  # pytype: disable=attribute-error

    # compute initial error
    self._compute_termination_error()
    # assert error is 0 at initialization. In particular this will prevent
    # a proto/walker mismatch.
    if self._termination_error > 1e-2:
      raise ValueError(('The termination exceeds 1e-2 at initialization. '
                        'This is likely due to a proto/walker mismatch.'))

    self._update_ghost(physics)

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

  def before_step(self, physics: 'mjcf.Physics', action,
                  random_state: np.random.RandomState):
    self._walker.apply_action(physics, action, random_state)

  def after_step(self, physics: 'mjcf.Physics',
                 random_state: np.random.RandomState):
    """Update the data after step."""
    del random_state  # unused by after_step.

    self._walker_features_prev = self._walker_features.copy()

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

  def get_reference_ego_bodies_quats(self, physics: 'mjcf.Physics'):
    """Body quat of the reference relative to the reference root quat."""
    del physics  # physics unused by reference observations.

    time_steps = self._time_step + self._ref_steps
    obs = []
    for t in time_steps:
      for b in self._body_idxs:
        obs.append(
            tr.quat_diff(
                self._clip_reference_features['quaternion'][t, :],
                self._clip_reference_features['body_quaternions'][t, b, :]))
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
    return reference_observations

  def get_reward(self, physics: 'mjcf.Physics') -> float:
    reference_observations = self.get_all_reference_observations(physics)
    reward, unused_debug_outputs, reward_channels = self._reward_fn(
        termination_error=self._termination_error,
        termination_error_threshold=self._termination_error_threshold,
        reference_features=self._current_reference_features,
        walker_features=self._walker_features,
        reference_observations=reference_observations)

    self._should_truncate = self._termination_error > self._termination_error_threshold

    self.last_reward_channels = reward_channels
    return reward

  def _set_walker(self, physics: 'mjcf.Physics'):
    timestep_features = tree.map_structure(lambda x: x[self._time_step],
                                           self._clip_reference_features)
    utils.set_walker_from_features(physics, self._walker, timestep_features)
    mjlib.mj_kinematics(physics.model.ptr, physics.data.ptr)

  def _update_ghost(self, physics: 'mjcf.Physics'):
    if self._ghost_offset is not None:
      target = tree.map_structure(lambda x: x[self._time_step],
                                  self._clip_reference_features)
      utils.set_walker_from_features(physics, self._ghost, target,
                                     self._ghost_offset)
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
      walker: Union['base.Walker', Callable],  # pylint: disable=g-bare-generic
      arena: composer.Arena,
      ref_path: Text,
      ref_steps: Sequence[int],
      dataset: Union[Text, Sequence[Any]],
      termination_error_threshold: float = 0.3,
      min_steps: int = 10,
      reward_type: Text = 'default',
      physics_timestep: float = DEFAULT_PHYSICS_TIMESTEP,
      always_init_at_clip_start: bool = False,
      proto_modifier: Optional[Any] = None,
      ghost_offset: Optional[Sequence[Union[int, float]]] = None,
      body_error_multiplier: Union[int, float] = 1.0,
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
      termination_error_threshold: Error threshold for episode terminations.
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
      ghost_offset: if not None, include a ghost rendering of the walker with
        the reference pose at the specified position offset.
      body_error_multiplier: A multiplier that is applied to the body error term
        when determining failure termination condition.
    """
    super(MultiClipMocapTracking, self).__init__(
        walker=walker,
        arena=arena,
        ref_path=ref_path,
        ref_steps=ref_steps,
        termination_error_threshold=termination_error_threshold,
        min_steps=min_steps,
        dataset=dataset,
        reward_type=reward_type,
        physics_timestep=physics_timestep,
        always_init_at_clip_start=always_init_at_clip_start,
        proto_modifier=proto_modifier,
        ghost_offset=ghost_offset,
        body_error_multiplier=body_error_multiplier)
    self._walker.observables.add_observable(
        'time_in_clip',
        base_observable.Generic(self.get_normalized_time_in_clip))

  def after_step(self, physics: 'mjcf.Physics', random_state):
    """Update the data after step."""
    super(MultiClipMocapTracking, self).after_step(physics, random_state)
    self._time_step += 1

    # Update the walker's data for this timestep.
    self._walker_features = utils.get_features(physics, self._walker)
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

    self._update_ghost(physics)

  def get_normalized_time_in_clip(self, physics: 'mjcf.Physics'):
    """Observation of the normalized time in the mocap clip."""
    normalized_time_in_clip = (self._current_start_time +
                               physics.time()) / self._current_clip.duration
    return np.array([normalized_time_in_clip])

  @property
  def name(self):
    return 'MultiClipMocapTracking'
