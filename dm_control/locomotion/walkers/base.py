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

"""Base class for Walkers."""

import abc
import collections

from dm_control import composer
from dm_control.composer.observation import observable

from dm_env import specs
import numpy as np


def _make_readonly_float64_copy(value):
  if np.isscalar(value):
    return np.float64(value)
  else:
    out = np.array(value, dtype=np.float64)
    out.flags.writeable = False
    return out


class WalkerPose(collections.namedtuple(
    'WalkerPose', ('qpos', 'xpos', 'xquat'))):
  """A named tuple representing a walker's joint and Cartesian pose."""

  __slots__ = ()

  def __new__(cls, qpos=None, xpos=(0, 0, 0), xquat=(1, 0, 0, 0)):
    """Creates a new WalkerPose.

    Args:
      qpos: The joint position for the pose, or `None` if the `qpos0` values in
        the `mjModel` should be used.
      xpos: A Cartesian displacement, for example if the walker should be lifted
        or lowered by a specific amount for this pose.
      xquat: A quaternion displacement for the root body.

    Returns:
      A new instance of `WalkerPose`.
    """
    return super(WalkerPose, cls).__new__(
        cls,
        qpos=_make_readonly_float64_copy(qpos) if qpos is not None else None,
        xpos=_make_readonly_float64_copy(xpos),
        xquat=_make_readonly_float64_copy(xquat))

  def __eq__(self, other):
    return (np.all(self.qpos == other.qpos) and
            np.all(self.xpos == other.xpos) and
            np.all(self.xquat == other.xquat))


class Walker(composer.Robot, metaclass=abc.ABCMeta):
  """Abstract base class for Walker robots."""

  def create_root_joints(self, attachment_frame) -> None:
    attachment_frame.add('freejoint')

  def _build_observables(self) -> 'WalkerObservables':
    return WalkerObservables(self)

  def transform_vec_to_egocentric_frame(self, physics, vec_in_world_frame):
    """Linearly transforms a world-frame vector into walker's egocentric frame.

    Note that this function does not perform an affine transformation of the
    vector. In other words, the input vector is assumed to be specified with
    respect to the same origin as this walker's egocentric frame. This function
    can also be applied to matrices whose innermost dimensions are either 2 or
    3. In this case, a matrix with the same leading dimensions is returned
    where the innermost vectors are replaced by their values computed in the
    egocentric frame.

    Args:
      physics: An `mjcf.Physics` instance.
      vec_in_world_frame: A NumPy array with last dimension of shape (2,) or
      (3,) that represents a vector quantity in the world frame.

    Returns:
      The same quantity as `vec_in_world_frame` but reexpressed in this
      entity's egocentric frame. The returned np.array has the same shape as
      np.asarray(vec_in_world_frame).

    Raises:
      ValueError: if `vec_in_world_frame` does not have shape ending with (2,)
        or (3,).
    """
    return super().global_vector_to_local_frame(physics, vec_in_world_frame)

  def transform_xmat_to_egocentric_frame(self, physics, xmat):
    """Transforms another entity's `xmat` into this walker's egocentric frame.

    This function takes another entity's (E) xmat, which is an SO(3) matrix
    from E's frame to the world frame, and turns it to a matrix that transforms
    from E's frame into this walker's egocentric frame.

    Args:
      physics: An `mjcf.Physics` instance.
      xmat: A NumPy array of shape (3, 3) or (9,) that represents another
        entity's xmat.

    Returns:
      The `xmat` reexpressed in this entity's egocentric frame. The returned
      np.array has the same shape as np.asarray(xmat).

    Raises:
      ValueError: if `xmat` does not have shape (3, 3) or (9,).
    """
    return super().global_xmat_to_local_frame(physics, xmat)

  @property
  @abc.abstractmethod
  def root_body(self):
    raise NotImplementedError

  @property
  @abc.abstractmethod
  def observable_joints(self):
    raise NotImplementedError

  @property
  def action_spec(self):
    if not self.actuators:
      minimum, maximum = (), ()
    else:
      minimum, maximum = zip(*[
          a.ctrlrange if a.ctrlrange is not None else (-1., 1.)
          for a in self.actuators
      ])
    return specs.BoundedArray(
        shape=(len(self.actuators),),
        dtype=float,
        minimum=minimum,
        maximum=maximum,
        name='\t'.join([actuator.name for actuator in self.actuators]))

  def apply_action(self, physics, action, random_state):
    """Apply action to walker's actuators."""
    del random_state
    physics.bind(self.actuators).ctrl = action


class WalkerObservables(composer.Observables):
  """Base class for Walker obserables."""

  @composer.observable
  def joints_pos(self):
    return observable.MJCFFeature('qpos', self._entity.observable_joints)

  @composer.observable
  def sensors_gyro(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.gyro)

  @composer.observable
  def sensors_accelerometer(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.accelerometer)

  @composer.observable
  def sensors_framequat(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.framequat)

  # Semantic groupings of Walker observables.
  def _collect_from_attachments(self, attribute_name):
    out = []
    for entity in self._entity.iter_entities(exclude_self=True):
      out.extend(getattr(entity.observables, attribute_name, []))
    return out

  @property
  def proprioception(self):
    return ([self.joints_pos] +
            self._collect_from_attachments('proprioception'))

  @property
  def kinematic_sensors(self):
    return ([self.sensors_gyro,
             self.sensors_accelerometer,
             self.sensors_framequat] +
            self._collect_from_attachments('kinematic_sensors'))

  @property
  def dynamic_sensors(self):
    return self._collect_from_attachments('dynamic_sensors')
