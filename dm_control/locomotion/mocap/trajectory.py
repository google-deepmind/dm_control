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
"""Represents a motion-captured trajectory."""

import collections
import copy

from dm_control.locomotion.mocap import mocap_pb2
from dm_control.locomotion.mocap import props as mocap_props
from dm_control.locomotion.mocap import walkers as mocap_walkers
import numpy as np

STEP_TIME_TOLERANCE = 1e-4

_REPEATED_POSITION_FIELDS = ('end_effectors', 'appendages', 'body_positions')
_REPEATED_QUATERNION_FIELDS = ('body_quaternions',)


def _zero_out_velocities(timestep_proto):
  out_proto = copy.deepcopy(timestep_proto)
  for walker in out_proto.walkers:
    walker.velocity[:] = np.zeros_like(walker.velocity)
    walker.angular_velocity[:] = np.zeros_like(walker.angular_velocity)
    walker.joints_velocity[:] = np.zeros_like(walker.joints_velocity)
  for prop in out_proto.props:
    prop.velocity[:] = np.zeros_like(prop.velocity)
    prop.angular_velocity[:] = np.zeros_like(prop.angular_velocity)
  return out_proto


class Trajectory(object):
  """Represents a motion-captured trajectory."""

  def __init__(self, proto, start_time=None, end_time=None, start_step=None,
               end_step=None, zero_out_velocities=True):
    """A wrapper around a mocap trajectory proto.

    Args:
      proto: proto representing the mocap trajectory.
      start_time: Start time of the mocap trajectory if only a subset of the
        underlying clip is desired. Defaults to the start of the full clip.
        Cannot be used when start_step is provided.
      end_time: End time of the mocap trajectory if only a subset of the
        underlying clip is desired. Defaults to the end of the full clip.
        Cannot be used when end_step is provided.
      start_step: Like start_time but using time indices. Defaults to the start
        of the full clip. Cannot be used when start_time is provided.
      end_step: Like end_time but using time indices. Defaults to the start
        of the full clip. Cannot be used when end_time is provided.
      zero_out_velocities: Whether to zero out the velocities in the last time
        step of the requested trajectory. Depending on the use-case it may be
        beneficial to use a stable end pose.
    """
    self._proto = proto
    self._zero_out_velocities = zero_out_velocities

    if (start_time and start_step) or (end_time and end_step):
      raise ValueError(('Please specify either start and end times'
                        'or start and end steps but not both.'))
    if start_step:
      start_time = start_step * self._proto.dt
    if end_step:
      end_time = end_step * self._proto.dt
    self._set_start_time(start_time or 0.)
    self._set_end_time(end_time or (len(self._proto.timesteps)*self._proto.dt))
    self._walkers_info = tuple(mocap_walkers.WalkerInfo(walker_proto)
                               for walker_proto in self._proto.walkers)
    self._dict = None

  def as_dict(self):
    """Return trajectory as dictionary."""
    if self._dict is None:
      self._dict = dict()

      if self._proto.timesteps:
        initial_timestep = self._proto.timesteps[0]

        num_walkers = len(initial_timestep.walkers)
        for i in range(num_walkers):
          key_prefix = 'walker_{:d}/'.format(
              i) if num_walkers > 1 else 'walker/'
          for field in mocap_pb2.WalkerPose.DESCRIPTOR.fields:
            field_name = field.name

            def walker_field(timestep, i=i, field_name=field_name):
              values = getattr(timestep.walkers[i], field_name)
              if field_name in _REPEATED_POSITION_FIELDS:
                values = np.reshape(values, (-1, 3))
              elif field_name in _REPEATED_QUATERNION_FIELDS:
                values = np.reshape(values, (-1, 4))
              return np.array(values)

            self._dict[key_prefix + field_name] = walker_field

        num_props = len(initial_timestep.walkers)
        for i in range(len(initial_timestep.props)):
          key_prefix = 'prop_{:d}/'.format(i) if num_props > 1 else 'prop/'
          for field in mocap_pb2.PropPose.DESCRIPTOR.fields:
            field_name = field.name

            def prop_field(timestep, i=i, field_name=field_name):
              return np.array(getattr(timestep.props[i], field_name))

            self._dict[key_prefix + field_name] = prop_field

        self._create_all_items(self._dict)
        for k in self._dict:
          # make trajectory immutable by default
          self._dict[k].flags.writeable = False  # pytype: disable=attribute-error

    return {k: v[self._start_step:self._end_step]
            for k, v in self._dict.items()}

  def _create_single_item(self, get_field_in_timestep):
    if not self._proto.timesteps:
      return np.empty((0))
    for i, timestep in enumerate(self._proto.timesteps):
      values = get_field_in_timestep(timestep)
      if i == 0:
        array = np.empty((len(self._proto.timesteps),) + values.shape)
      array[i, :] = values
    return array

  def _create_all_items(self, dictionary):
    for key, value in dictionary.items():
      if callable(value):
        dictionary[key] = self._create_single_item(value)
    return dictionary

  def _get_quantized_time(self, time):
    if time == float('inf'):
      return len(self._proto.timesteps) - 1
    else:
      divided_time = time / self._proto.dt
      quantized_time = int(np.round(divided_time))
      if np.abs(quantized_time - divided_time) > STEP_TIME_TOLERANCE:
        raise ValueError('`time` should be a multiple of dt = {}: got {}'
                         .format(self._proto.dt, time))
      return quantized_time

  def _get_step_id(self, time):
    quantized_time = self._get_quantized_time(time)
    return np.clip(quantized_time + self._start_step,
                   self._start_step, self._end_step - 1)

  def get_modified_trajectory(self, proto_modifier, random_state=None):
    modified_proto = copy.deepcopy(self._proto)
    if isinstance(proto_modifier, collections.Iterable):
      for proto_mod in proto_modifier:
        proto_mod(modified_proto, random_state=random_state)
    else:
      proto_modifier(modified_proto, random_state=random_state)
    return type(self)(modified_proto, self.start_time, self.end_time)

  @property
  def identifier(self):
    return self._proto.identifier

  @property
  def start_time(self):
    return self._start_step * self._proto.dt

  def _set_start_time(self, new_value):
    self._start_step = np.clip(self._get_quantized_time(new_value),
                               0, len(self._proto.timesteps) - 1)

  @start_time.setter
  def start_time(self, new_value):
    self._set_start_time(new_value)

  @property
  def start_step(self):
    return self._start_step

  @start_step.setter
  def start_step(self, new_value):
    self._start_step = np.clip(int(new_value), 0,
                               len(self._proto.timesteps) - 1)

  @property
  def end_step(self):
    return self._end_step

  @end_step.setter
  def end_step(self, new_value):
    self._end_step = np.clip(int(new_value), 0,
                             len(self._proto.timesteps) - 1)

  @property
  def end_time(self):
    return (self._end_step - 1) * self._proto.dt

  @property
  def clip_end_time(self):
    """Length of the full clip."""
    return (len(self._proto.timesteps) -1) * self._proto.dt

  def _set_end_time(self, new_value):
    self._end_step = 1 + np.clip(self._get_quantized_time(new_value),
                                 0, len(self._proto.timesteps) - 1)
    if self._zero_out_velocities:
      self._last_timestep = _zero_out_velocities(
          self._proto.timesteps[self._end_step - 1])
    else:
      self._last_timestep = self._proto.timesteps[self._end_step - 1]

  @end_time.setter
  def end_time(self, new_value):
    self._set_end_time(new_value)

  @property
  def duration(self):
    return self.end_time - self.start_time

  @property
  def num_steps(self):
    return self._end_step - self._start_step

  @property
  def dt(self):
    return self._proto.dt

  def configure_walkers(self, walkers):
    try:
      walkers = iter(walkers)
    except TypeError:
      walkers = iter((walkers,))
    for walker, walker_info in zip(walkers, self._walkers_info):
      walker_info.rescale_walker(walker)
      walker_info.add_marker_sites(walker)

  def create_props(self, proto_modifier=None, priority_friction=False):
    proto = self._proto
    if proto_modifier is not None:
      proto = copy.copy(proto)
      proto_modifier(proto)
    return tuple(
        mocap_props.Prop(prop_proto, priority_friction=priority_friction)
        for prop_proto in proto.props)

  def get_timestep_data(self, time):
    step_id = self._get_step_id(time)
    if step_id == self._end_step - 1:
      return self._last_timestep
    else:
      return self._proto.timesteps[step_id]

  def set_walker_poses(self, physics, walkers):
    timestep = self._proto.timesteps[self._get_step_id(physics.time())]
    for walker, walker_timestep in zip(walkers, timestep.walkers):
      walker.set_pose(physics,
                      position=walker_timestep.position,
                      quaternion=walker_timestep.quaternion)
      physics.bind(walker.mocap_joints).qpos = walker_timestep.joints

  def set_prop_poses(self, physics, props):
    timestep = self._proto.timesteps[self._get_step_id(physics.time())]
    for prop, prop_timestep in zip(props, timestep.props):
      prop.set_pose(physics,
                    position=prop_timestep.position,
                    quaternion=prop_timestep.quaternion)
