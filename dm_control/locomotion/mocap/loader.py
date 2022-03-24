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
"""Helpers for loading a collection of trajectories."""

import abc
import collections
import operator

from dm_control.composer import variation
from dm_control.locomotion.mocap import mocap_pb2
from dm_control.locomotion.mocap import trajectory
import numpy as np

from google.protobuf import descriptor


class TrajectoryLoader(metaclass=abc.ABCMeta):
  """Base class for helpers that load and decode mocap trajectories."""

  def __init__(self, trajectory_class=trajectory.Trajectory,
               proto_modifier=()):
    """Initializes this loader.

    Args:
      trajectory_class: A Python class that wraps a loaded trajectory proto.
      proto_modifier: (optional) A callable, or an iterable of callables, that
        modify each trajectory proto in-place after it has been deserialized
        from the SSTable.

    Raises:
      ValueError: If `proto_modifier` is specified, but contains a
        non-callable entry.
    """
    self._trajectory_class = trajectory_class
    if not isinstance(proto_modifier, collections.abc.Iterable):
      if proto_modifier is None:  # backwards compatibility
        proto_modifier = ()
      else:
        proto_modifier = (proto_modifier,)
    for modifier in proto_modifier:
      if not callable(modifier):
        raise ValueError('{} is not callable'.format(modifier))
    self._proto_modifiers = proto_modifier

  @abc.abstractmethod
  def keys(self):
    """The sequence of identifiers for the loaded trajectories."""

  @abc.abstractmethod
  def _get_proto_for_key(self, key):
    """Returns a protocol buffer message corresponding to the requested key."""

  def get_trajectory(self, key, start_time=None, end_time=None, start_step=None,
                     end_step=None, zero_out_velocities=True):
    """Retrieves a trajectory identified by `key` from the SSTable."""
    proto = self._get_proto_for_key(key)
    for modifier in self._proto_modifiers:
      modifier(proto)
    return self._trajectory_class(proto, start_time=start_time,
                                  end_time=end_time, start_step=start_step,
                                  end_step=end_step,
                                  zero_out_velocities=zero_out_velocities)


class HDF5TrajectoryLoader(TrajectoryLoader):
  """A helper for loading and decoding mocap trajectories from HDF5.

  In order to use this class, h5py must be installed (it's an optional
  dependency of dm_control).
  """

  def __init__(self, path, trajectory_class=trajectory.Trajectory,
               proto_modifier=()):
    # h5py is an optional dependency of dm_control, so only try to import
    # if it's used.
    try:
      import h5py  # pylint: disable=g-import-not-at-top
    except ImportError as e:
      raise ImportError(
          'h5py not found. When installing dm_control, '
          'use `pip install dm_control[HDF5]` to enable HDF5TrajectoryLoader.'
      ) from e
    self._h5_file = h5py.File(path, mode='r')
    self._keys = tuple(sorted(self._h5_file.keys()))
    super().__init__(
        trajectory_class=trajectory_class, proto_modifier=proto_modifier)

  def keys(self):
    return self._keys

  def _fill_primitive_proto_fields(self, proto, h5_group, skip_fields=()):
    for field in proto.DESCRIPTOR.fields:
      if field.name in skip_fields or field.name not in h5_group.attrs:
        continue
      elif field.type not in (descriptor.FieldDescriptor.TYPE_GROUP,
                              descriptor.FieldDescriptor.TYPE_MESSAGE):
        if field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          getattr(proto, field.name).extend(h5_group.attrs[field.name])
        else:
          setattr(proto, field.name, h5_group.attrs[field.name])

  def _fill_repeated_proto_message_fields(self, proto_container,
                                          h5_container, h5_prefix):
    for item_id in range(len(h5_container)):
      h5_item = h5_container['{:s}_{:d}'.format(h5_prefix, item_id)]
      proto = proto_container.add()
      self._fill_primitive_proto_fields(proto, h5_item)

  def _get_proto_for_key(self, key):
    """Returns a trajectory protocol buffer message for the specified key."""
    if isinstance(key, str):
      key = key.encode('utf-8')

    h5_trajectory = self._h5_file[key]
    num_steps = h5_trajectory.attrs['num_steps']

    proto = mocap_pb2.FittedTrajectory()
    proto.identifier = key
    self._fill_primitive_proto_fields(proto, h5_trajectory,
                                      skip_fields=('identifier',))

    for _ in range(num_steps):
      proto.timesteps.add()

    h5_walkers = h5_trajectory['walkers']
    for walker_id in range(len(h5_walkers)):
      h5_walker = h5_walkers['walker_{:d}'.format(walker_id)]
      walker_proto = proto.walkers.add()
      self._fill_primitive_proto_fields(walker_proto, h5_walker)
      self._fill_repeated_proto_message_fields(
          walker_proto.scaling.subtree,
          h5_walker['scaling'], h5_prefix='subtree')
      self._fill_repeated_proto_message_fields(
          walker_proto.markers.marker,
          h5_walker['markers'], h5_prefix='marker')

      walker_fields = dict()
      for field in mocap_pb2.WalkerPose.DESCRIPTOR.fields:
        walker_fields[field.name] = np.asarray(h5_walker[field.name])

      for timestep_id, timestep in enumerate(proto.timesteps):
        walker_timestep = timestep.walkers.add()
        for k, v in walker_fields.items():
          getattr(walker_timestep, k).extend(v[:, timestep_id])

    h5_props = h5_trajectory['props']
    for prop_id in range(len(h5_props)):
      h5_prop = h5_props['prop_{:d}'.format(prop_id)]
      prop_proto = proto.props.add()
      self._fill_primitive_proto_fields(prop_proto, h5_prop)

      prop_fields = dict()
      for field in mocap_pb2.PropPose.DESCRIPTOR.fields:
        prop_fields[field.name] = np.asarray(h5_prop[field.name])

      for timestep_id, timestep in enumerate(proto.timesteps):
        prop_timestep = timestep.props.add()
        for k, v in prop_fields.items():
          getattr(prop_timestep, k).extend(v[:, timestep_id])

    return proto


class PropMassLimiter:
  """A trajectory proto modifier that enforces a maximum mass for each prop."""

  def __init__(self, max_mass):
    self._max_mass = max_mass

  def __call__(self, proto, random_state=None):
    for prop in proto.props:
      prop.mass = min(prop.mass, self._max_mass)


class PropResizer:
  """A trajectory proto modifier that changes prop sizes and mass."""

  def __init__(self, size_factor=None, size_delta=None, mass=None):
    if size_factor and size_delta:
      raise ValueError(
          'Only one of `size_factor` or `size_delta` can be specified.')
    elif size_factor:
      self._size_variation = size_factor
      self._size_op = operator.mul
    else:
      self._size_variation = size_delta
      self._size_op = operator.add
    self._mass = mass

  def __call__(self, proto, random_state=None):
    for prop in proto.props:
      size_value = variation.evaluate(self._size_variation,
                                      random_state=random_state)
      if not np.shape(size_value):
        size_value = np.full(len(prop.size), size_value)
      for i in range(len(prop.size)):
        prop.size[i] = self._size_op(prop.size[i], size_value[i])
      prop.mass = variation.evaluate(self._mass, random_state=random_state)


class ZOffsetter:
  """A trajectory proto modifier that shifts the z position of a trajectory."""

  def __init__(self, z_offset=0.0):
    self._z_offset = z_offset

  def _add_z_offset(self, proto_field):
    if len(proto_field) % 3:
      raise ValueError('Length of proto_field is not a multiple of 3.')
    for i in range(2, len(proto_field), 3):
      proto_field[i] += self._z_offset

  def __call__(self, proto, random_state=None):
    for t in proto.timesteps:
      for walker_pose in t.walkers:
        # shift walker position.
        self._add_z_offset(walker_pose.position)
        self._add_z_offset(walker_pose.body_positions)
        self._add_z_offset(walker_pose.center_of_mass)
      for prop_pose in t.props:
        # shift prop position
        self._add_z_offset(prop_pose.position)
