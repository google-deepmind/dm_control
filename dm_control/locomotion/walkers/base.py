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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import initializers
from dm_control.mujoco.wrapper.mjbindings import mjlib

import numpy as np
import six

from dm_control.rl import specs

_RANGEFINDER_SCALE = 10.0
_TOUCH_THRESHOLD = 1e-3


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


@six.add_metaclass(abc.ABCMeta)
class Walker(composer.Robot):
  """Abstract base class for Walker robots."""

  def _build(self, initializer=None):
    self._initializer = initializer or initializers.UprightInitializer()

  def create_root_joints(self, attachment_frame):
    attachment_frame.add('freejoint')

  @property
  def upright_pose(self):
    return WalkerPose()

  def _build_observables(self):
    return WalkerObservables(self)

  def reinitialize_pose(self, physics, random_state):
    self._initializer.initialize_pose(physics, self, random_state)

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
    vec_in_world_frame = np.asarray(vec_in_world_frame)

    xmat = np.reshape(physics.bind(self.root_body).xmat, (3, 3))
    # The ordering of the np.dot is such that the transformation holds for any
    # matrix whose final dimensions are (2,) or (3,).
    if vec_in_world_frame.shape[-1] == 2:
      return np.dot(vec_in_world_frame, xmat[:2, :2])
    elif vec_in_world_frame.shape[-1] == 3:
      return np.dot(vec_in_world_frame, xmat)
    else:
      raise ValueError('`vec_in_world_frame` should have shape with final '
                       'dimension 2 or 3: got {}'.format(
                           vec_in_world_frame.shape))

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
    xmat = np.asarray(xmat)

    input_shape = xmat.shape
    if xmat.shape == (9,):
      xmat = np.reshape(xmat, (3, 3))

    self_xmat = np.reshape(physics.bind(self.root_body).xmat, (3, 3))
    if xmat.shape == (3, 3):
      return np.reshape(np.dot(self_xmat.T, xmat), input_shape)
    else:
      raise ValueError('`xmat` should have shape (3, 3) or (9,): got {}'.format(
          xmat.shape))

  @abc.abstractproperty
  def root_body(self):
    raise NotImplementedError

  def aliveness(self, physics):
    """A measure of the aliveness of the walker.

    Aliveness measure could be used for deciding on termination (ant flipped
    over and it's impossible for it to recover), or used as a shaping reward
    to maintain an alive pose that we desired (humanoids remaining upright).

    Args:
      physics: an instance of `Physics`.

    Returns:
      a `float` in the range of [-1., 0.] where -1 means not alive and 0. means
      alive. In walkers for which the concept of aliveness does not make sense,
      the default implementation is to always return 0.0.
    """
    return 0.

  @abc.abstractproperty
  def ground_contact_geoms(self):
    """Geoms in this walker that are expected to be in contact with the ground.

    This property is used by some tasks to determine contact-based failure
    termination. It should only contain geoms that are expected to be in
    contact with the ground during "normal" locomotion. For example, for a
    humanoid model, this property would be expected to contain only the geoms
    that make up the two feet.

    Note that certain specialized tasks may also allow geoms that are not listed
    here to be in contact with the ground. For example, a humanoid cartwheel
    task would also allow the hands to touch the ground in addition to the feet.
    """
    raise NotImplementedError

  def after_compile(self, physics, unused_random_state):
    super(Walker, self).after_compile(physics, unused_random_state)
    self._end_effector_geom_ids = set()
    for eff_body in self.end_effectors:
      eff_geom = eff_body.find_all('geom')
      self._end_effector_geom_ids |= set(physics.bind(eff_geom).element_id)
    self._body_geom_ids = set(
        physics.bind(geom).element_id
        for geom in self.mjcf_model.find_all('geom'))
    self._body_geom_ids.difference_update(self._end_effector_geom_ids)

  @property
  def end_effector_geom_ids(self):
    return self._end_effector_geom_ids

  @property
  def body_geom_ids(self):
    return self._body_geom_ids

  def end_effector_contacts(self, physics):
    """Collect the contacts with the end effectors.

    This function returns any contacts being made with any of the end effectors,
    both the other geom with which contact is being made as well as the
    magnitude.

    Args:
      physics: an instance of `Physics`.

    Returns:
      a dict with as key a tuple of geom ids, of which one is an end effector,
      and as value the total magnitude of all contacts between these geoms
    """
    return self.collect_contacts(physics, self._end_effector_geom_ids)

  def body_contacts(self, physics):
    """Collect the contacts with the body.

    This function returns any contacts being made with any of body geoms, except
    the end effectors, both the other geom with which contact is being made as
    well as the magnitude.

    Args:
      physics: an instance of `Physics`.

    Returns:
      a dict with as key a tuple of geom ids, of which one is a body geom,
      and as value the total magnitude of all contacts between these geoms
    """
    return self.collect_contacts(physics, self._body_geom_ids)

  def collect_contacts(self, physics, geom_ids):
    contacts = {}
    forcetorque = np.zeros(6)
    for i, contact in enumerate(physics.data.contact):
      if ((contact.geom1 in geom_ids) or
          (contact.geom2 in geom_ids)) and contact.dist < contact.includemargin:
        mjlib.mj_contactForce(physics.model.ptr, physics.data.ptr, i,
                              forcetorque)
        contacts[(contact.geom1, contact.geom2)] = (forcetorque[0]
                                                    + contacts.get(
                                                        (contact.geom1,
                                                         contact.geom2), 0.))
    return contacts

  @abc.abstractproperty
  def end_effectors(self):
    raise NotImplementedError

  @abc.abstractproperty
  def observable_joints(self):
    raise NotImplementedError

  @abc.abstractproperty
  def egocentric_camera(self):
    raise NotImplementedError

  @composer.cached_property
  def touch_sensors(self):
    return self._mjcf_root.sensor.get_children('touch')

  @property
  def prev_action(self):
    """Returns the actuation actions applied in the previous step.

    Concrete walker implementations should provide caching mechanism themselves
      in order to access this observable (for example, through `apply_action`).
    """
    raise NotImplementedError

  def after_substep(self, physics, random_state):
    del random_state  # Unused.
    # As of MuJoCo v2.0, updates to `mjData->subtree_linvel` will be skipped
    # unless these quantities are needed by the simulation. We need these in
    # order to calculate `torso_{x,y}vel`, so we therefore call `mj_subtreeVel`
    # explicitly.
    # TODO(b/123065920): Consider using a `subtreelinvel` sensor instead.
    mjlib.mj_subtreeVel(physics.model.ptr, physics.data.ptr)

  @property
  def action_spec(self):
    minimum, maximum = zip(*[
        a.ctrlrange if a.ctrlrange is not None else (-1., 1.)
        for a in self.actuators
    ])
    return specs.BoundedArraySpec(
        shape=(len(self.actuators),),
        dtype=np.float,
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
  def joints_vel(self):
    return observable.MJCFFeature('qvel', self._entity.observable_joints)

  @composer.observable
  def body_height(self):
    return observable.Generic(
        lambda physics: physics.bind(self._entity.root_body).xpos[2])

  @composer.observable
  def end_effectors_pos(self):
    """Position of end effectors relative to torso, in the egocentric frame."""
    def relative_pos_in_egocentric_frame(physics):
      end_effector = physics.bind(self._entity.end_effectors).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)
    return observable.Generic(relative_pos_in_egocentric_frame)

  @composer.observable
  def world_zaxis(self):
    """The world's z-vector in this Walker's torso frame."""
    return observable.Generic(
        lambda physics: physics.bind(self._entity.root_body).xmat[6:])

  @composer.observable
  def sensors_gyro(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.gyro)

  @composer.observable
  def sensors_velocimeter(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.velocimeter)

  @composer.observable
  def sensors_accelerometer(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.accelerometer)

  @composer.observable
  def sensors_force(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.force)

  @composer.observable
  def sensors_torque(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.torque)

  @composer.observable
  def sensors_touch(self):
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.sensor.touch,
        corruptor=
        lambda v, random_state: np.array(v > _TOUCH_THRESHOLD, dtype=np.float))

  @composer.observable
  def sensors_rangefinder(self):
    def tanh_rangefinder(physics):
      raw = physics.bind(self._entity.mjcf_model.sensor.rangefinder).sensordata
      raw = np.array(raw)
      raw[raw == -1.0] = np.inf
      return _RANGEFINDER_SCALE * np.tanh(raw / _RANGEFINDER_SCALE)
    return observable.Generic(tanh_rangefinder)

  @composer.observable
  def egocentric_camera(self):
    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=64, height=64)

  @composer.observable
  def position(self):
    return observable.MJCFFeature('xpos', self._entity.root_body)

  @composer.observable
  def orientation(self):
    return observable.MJCFFeature('xmat', self._entity.root_body)

  def add_egocentric_vector(self,
                            name,
                            world_frame_observable,
                            enabled=True,
                            origin_callable=None,
                            **kwargs):

    def _egocentric(physics, origin_callable=origin_callable):
      vec = world_frame_observable.observation_callable(physics)()
      origin_callable = origin_callable or (lambda physics: np.zeros(vec.size))
      delta = vec - origin_callable(physics)
      return self._entity.transform_vec_to_egocentric_frame(physics, delta)

    self._observables[name] = observable.Generic(_egocentric, **kwargs)
    self._observables[name].enabled = enabled

  def add_egocentric_xmat(self, name, xmat_observable, enabled=True, **kwargs):

    def _egocentric(physics):
      return self._entity.transform_xmat_to_egocentric_frame(
          physics,
          xmat_observable.observation_callable(physics)())

    self._observables[name] = observable.Generic(_egocentric, **kwargs)
    self._observables[name].enabled = enabled

  # Semantic groupings of Walker observables.
  def _collect_from_attachments(self, attribute_name):
    out = []
    for entity in self._entity.iter_entities(exclude_self=True):
      out.extend(getattr(entity.observables, attribute_name, []))
    return out

  @property
  def proprioception(self):
    return ([self.joints_pos, self.joints_vel,
             self.body_height, self.end_effectors_pos, self.world_zaxis] +
            self._collect_from_attachments('proprioception'))

  @property
  def kinematic_sensors(self):
    return ([self.sensors_gyro, self.sensors_velocimeter,
             self.sensors_accelerometer] +
            self._collect_from_attachments('kinematic_sensors'))

  @property
  def dynamic_sensors(self):
    return ([self.sensors_force, self.sensors_torque, self.sensors_touch] +
            self._collect_from_attachments('dynamic_sensors'))

  # Convenience observables for defining rewards and terminations.
  @composer.observable
  def veloc_strafe(self):
    velocimeter = self._entity.mjcf_model.sensor.velocimeter
    return observable.Generic(
        lambda physics: physics.bind(velocimeter).sensordata[1])

  @composer.observable
  def veloc_up(self):
    velocimeter = self._entity.mjcf_model.sensor.velocimeter
    return observable.Generic(
        lambda physics: physics.bind(velocimeter).sensordata[2])

  @composer.observable
  def veloc_forward(self):
    velocimeter = self._entity.mjcf_model.sensor.velocimeter
    return observable.Generic(
        lambda physics: physics.bind(velocimeter).sensordata[0])

  @composer.observable
  def gyro_forward_roll(self):
    gyro = self._entity.mjcf_model.sensor.gyro
    return observable.Generic(lambda physics: -physics.bind(gyro).sensordata[0])

  @composer.observable
  def gyro_rightward_roll(self):
    gyro = self._entity.mjcf_model.sensor.gyro
    return observable.Generic(lambda physics: physics.bind(gyro).sensordata[1])

  @composer.observable
  def gyro_clockwise_spin(self):
    gyro = self._entity.mjcf_model.sensor.gyro
    return observable.Generic(lambda physics: -physics.bind(gyro).sensordata[2])

  @composer.observable
  def torso_xvel(self):
    return observable.Generic(
        lambda physics: physics.bind(self._entity.root_body).subtree_linvel[0])

  @composer.observable
  def torso_yvel(self):
    return observable.Generic(
        lambda physics: physics.bind(self._entity.root_body).subtree_linvel[1])

  @composer.observable
  def prev_action(self):
    return observable.Generic(lambda _: self._entity.prev_action)
