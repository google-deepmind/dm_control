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

from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import initializers
from dm_control.mujoco.wrapper.mjbindings import mjlib
import numpy as np

_RANGEFINDER_SCALE = 10.0
_TOUCH_THRESHOLD = 1e-3


class Walker(base.Walker):
  """Legacy base class for Walker robots."""

  def _build(self, initializer=None):
    try:
      self._initializers = tuple(initializer)
    except TypeError:
      self._initializers = (initializer or initializers.UprightInitializer(),)

  @property
  def upright_pose(self):
    return base.WalkerPose()

  def _build_observables(self):
    return WalkerObservables(self)

  def reinitialize_pose(self, physics, random_state):
    for initializer in self._initializers:
      initializer.initialize_pose(physics, self, random_state)

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
    super().after_compile(physics, unused_random_state)
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

  @composer.cached_property
  def mocap_joints(self):
    return tuple(self.mjcf_model.find_all('joint'))

  @composer.cached_property
  def mocap_to_observable_joint_order(self):
    mocap_to_obs = [self.mocap_joints.index(j) for j in self.observable_joints]
    return mocap_to_obs

  @composer.cached_property
  def observable_to_mocap_joint_order(self):
    obs_to_mocap = [self.observable_joints.index(j) for j in self.mocap_joints]
    return obs_to_mocap


class WalkerObservables(base.WalkerObservables):
  """Legacy base class for Walker obserables."""

  @composer.observable
  def joints_vel(self):
    return observable.MJCFFeature('qvel', self._entity.observable_joints)

  @composer.observable
  def body_height(self):
    return observable.MJCFFeature('xpos', self._entity.root_body)[2]

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
    return observable.MJCFFeature('xmat', self._entity.root_body)[6:]

  @composer.observable
  def sensors_velocimeter(self):
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.velocimeter)

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
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.velocimeter)[1]

  @composer.observable
  def veloc_up(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.velocimeter)[2]

  @composer.observable
  def veloc_forward(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.velocimeter)[0]

  @composer.observable
  def gyro_backward_roll(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.gyro)[0]

  @composer.observable
  def gyro_rightward_roll(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.gyro)[1]

  @composer.observable
  def gyro_anticlockwise_spin(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.gyro)[2]

  @composer.observable
  def torso_xvel(self):
    return observable.MJCFFeature('subtree_linvel', self._entity.root_body)[0]

  @composer.observable
  def torso_yvel(self):
    return observable.MJCFFeature('subtree_linvel', self._entity.root_body)[1]

  @composer.observable
  def prev_action(self):
    return observable.Generic(lambda _: self._entity.prev_action)
