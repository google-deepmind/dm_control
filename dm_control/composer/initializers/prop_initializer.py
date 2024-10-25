# Copyright 2018 The dm_control Authors.
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

"""An initializer that places props at various poses."""

from absl import logging

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import variation
from dm_control.composer.initializers import utils
from dm_control.composer.variation import rotations
from dm_control.rl import control
import numpy as np


# Absolute velocity threshold for a prop joint to be considered settled.
_SETTLE_QVEL_TOL = 1e-3
# Absolute acceleration threshold for a prop joint to be considered settled.
_SETTLE_QACC_TOL = 1e-2

_REJECTION_SAMPLING_FAILED = '\n'.join([
    'Failed to find a non-colliding pose for prop {model_name!r} within '  # pylint: disable=implicit-str-concat
    '{max_attempts} attempts.',
    'You may be able to avoid this error by:',
    '1. Sampling from a broader distribution over positions and/or quaternions',
    '2. Increasing `max_attempts_per_prop`',
    '3. Disabling collision detection by setting `ignore_collisions=False`'])


_SETTLING_PHYSICS_FAILED = '\n'.join([
    'Failed to settle physics after {max_attempts} attempts of '  # pylint: disable=implicit-str-concat
    '{max_time} seconds.',
    'Last residual velocity={max_qvel} and acceleration={max_qacc}.',
    'This suggests your dynamics are unstable. Consider:',
    '\t1. Increasing `max_settle_physics_attempts`',
    '\t2. Increasing `max_settle_physics_time`',
    '\t3. Tuning your contact parameters or initial pose distributions.'])


class PropPlacer(composer.Initializer):
  """An initializer that places props at various positions and orientations."""

  def __init__(self,
               props,
               position,
               quaternion=rotations.IDENTITY_QUATERNION,
               ignore_collisions=False,
               max_qvel_tol=_SETTLE_QVEL_TOL,
               max_qacc_tol=_SETTLE_QACC_TOL,
               max_attempts_per_prop=20,
               settle_physics=False,
               min_settle_physics_time=0.,
               max_settle_physics_time=2.,
               max_settle_physics_attempts=1,
               raise_exception_on_settle_failure=False):
    """Initializes this PropPlacer.

    Args:
      props: A sequence of `composer.Entity` instances representing props.
      position: A single fixed Cartesian position, or a `composer.Variation`
        object that generates Cartesian positions. If a fixed sequence of
        positions for multiple props is desired, use
        `variation.deterministic.Sequence`.
      quaternion: (optional) A single fixed unit quaternion, or a
        `Variation` object that generates unit quaternions. If a fixed
        sequence of quaternions for multiple props is desired, use
        `variation.deterministic.Sequence`.
      ignore_collisions: (optional) If True, ignore collisions between props,
        i.e. do not run rejection sampling.
      max_qvel_tol: Maximum post-initialization joint velocity for props. If
        `settle_physics=True`, the simulation will be run until all prop joint
        velocities are less than this threshold.
      max_qacc_tol: Maximum post-initialization joint acceleration for props. If
        `settle_physics=True`, the simulation will be run until all prop joint
        velocities are less than this threshold.
      max_attempts_per_prop: The maximum number of rejection sampling attempts
        per prop. If a non-colliding pose cannot be found before this limit is
        reached, a `RuntimeError` will be raised.
      settle_physics: (optional) If True, the physics simulation will be
        advanced for a few steps to allow the prop positions to settle.
      min_settle_physics_time: (optional) When `settle_physics` is True, lower
        bound on time (in seconds) the physics simulation is advanced.
      max_settle_physics_time: (optional) When `settle_physics` is True, upper
        bound on time (in seconds) the physics simulation is advanced.
      max_settle_physics_attempts: (optional) When `settle_physics` is True, the
        number of attempts at sampling overall scene pose and settling.
      raise_exception_on_settle_failure: If True, raises an exception if
        settling physics is unsuccessful.
    """
    super().__init__()
    self._props = props
    self._prop_joints = []
    for prop in props:
      freejoint = mjcf.get_frame_freejoint(prop.mjcf_model)
      if freejoint is not None:
        self._prop_joints.append(freejoint)
      self._prop_joints.extend(prop.mjcf_model.find_all('joint'))
    self._position = position
    self._quaternion = quaternion
    self._ignore_collisions = ignore_collisions
    self._max_attempts_per_prop = max_attempts_per_prop
    self._settle_physics = settle_physics
    self._max_qvel_tol = max_qvel_tol
    self._max_qacc_tol = max_qacc_tol
    self._min_settle_physics_time = min_settle_physics_time
    self._max_settle_physics_time = max_settle_physics_time
    self._max_settle_physics_attempts = max_settle_physics_attempts
    self._raise_exception_on_settle_failure = raise_exception_on_settle_failure

    if max_settle_physics_attempts < 1:
      raise ValueError('max_settle_physics_attempts should be greater than '
                       'zero to have any effect, but is '
                       f'{max_settle_physics_attempts}')

  def _has_collisions_with_prop(self, physics, prop):
    prop_geom_ids = physics.bind(prop.mjcf_model.find_all('geom')).element_id
    contacts = physics.data.contact
    for contact in contacts:
      # Ignore contacts with positive distances (i.e. not actually touching).
      if contact.dist <= 0 and (contact.geom1 in prop_geom_ids or
                                contact.geom2 in prop_geom_ids):
        return True

  def _disable_and_cache_contact_parameters(self, physics, props):
    cached_contact_params = {}
    for prop in props:
      geoms = prop.mjcf_model.find_all('geom')
      param_list = []
      for geom in geoms:
        bound_geom = physics.bind(geom)
        param_list.append((bound_geom.contype, bound_geom.conaffinity))
        bound_geom.contype = 0
        bound_geom.conaffinity = 0
      cached_contact_params[prop] = param_list
    return cached_contact_params

  def _restore_contact_parameters(self, physics, prop, cached_contact_params):
    geoms = prop.mjcf_model.find_all('geom')
    param_list = cached_contact_params[prop]
    for i, geom in enumerate(geoms):
      contype, conaffinity = param_list[i]
      bound_geom = physics.bind(geom)
      bound_geom.contype = contype
      bound_geom.conaffinity = conaffinity

  def __call__(self, physics, random_state, ignore_contacts_with_entities=None):
    """Sets initial prop poses.

    Args:
      physics: An `mjcf.Physics` instance.
      random_state: a `np.random.RandomState` instance.
      ignore_contacts_with_entities: a list of `composer.Entity` instances
        to ignore when detecting collisions. This can be used to ignore props
        that are not being placed by this initializer, but are known to be
        colliding in the current state of the simulation (for example if they
        are going to be placed by a different initializer that will be called
        subsequently).

    Raises:
      RuntimeError: If `ignore_collisions == False` and a non-colliding prop
        pose could not be found within `max_attempts_per_prop`.
    """
    if ignore_contacts_with_entities is None:
      ignore_contacts_with_entities = []
    # Temporarily disable contacts for all geoms that belong to props which
    # haven't yet been placed in order to free up space in the contact buffer.
    cached_contact_params = self._disable_and_cache_contact_parameters(
        physics, self._props + ignore_contacts_with_entities)

    try:
      physics.forward()
    except control.PhysicsError as cause:
      effect = control.PhysicsError(
          'Despite disabling contact for all props in this initializer, '
          '`physics.forward()` resulted in a `PhysicsError`')
      raise effect from cause

    def place_props():
      for prop in self._props:
        # Restore the original contact parameters for all geoms in the prop
        # we're about to place, so that we can detect if the new pose results in
        # collisions.
        self._restore_contact_parameters(physics, prop, cached_contact_params)

        success = False
        initial_position, initial_quaternion = prop.get_pose(physics)
        next_position, next_quaternion = initial_position, initial_quaternion
        for _ in range(self._max_attempts_per_prop):
          next_position = variation.evaluate(self._position,
                                             initial_value=initial_position,
                                             current_value=next_position,
                                             random_state=random_state)
          next_quaternion = variation.evaluate(self._quaternion,
                                               initial_value=initial_quaternion,
                                               current_value=next_quaternion,
                                               random_state=random_state)
          prop.set_pose(physics, next_position, next_quaternion)
          try:
            # If this pose results in collisions then there's a chance we'll
            # encounter a PhysicsError error here due to a full contact buffer,
            # in which case reject this pose and sample another.
            physics.forward()
          except control.PhysicsError:
            continue

          if (self._ignore_collisions
              or not self._has_collisions_with_prop(physics, prop)):
            success = True
            break

        if not success:
          raise RuntimeError(_REJECTION_SAMPLING_FAILED.format(
              model_name=prop.mjcf_model.model,
              max_attempts=self._max_attempts_per_prop))

      for prop in ignore_contacts_with_entities:
        self._restore_contact_parameters(physics, prop, cached_contact_params)

    # Place the props and settle the physics. If settling was requested and it
    # it fails, re-place the props.
    def place_and_settle():
      for _ in range(self._max_settle_physics_attempts):
        place_props()

        # Step physics and check prop states.
        original_time = physics.data.time
        try:
          props_isolator = utils.JointStaticIsolator(physics, self._prop_joints)
          prop_joints_mj = physics.bind(self._prop_joints)
          while (
              physics.data.time - original_time < self._max_settle_physics_time
          ):
            with props_isolator:
              physics.step()
            max_qvel = np.max(np.abs(prop_joints_mj.qvel))
            max_qacc = np.max(np.abs(prop_joints_mj.qacc))
            if (max_qvel < self._max_qvel_tol) and (
                max_qacc < self._max_qacc_tol) and (
                    physics.data.time - original_time
                    ) > self._min_settle_physics_time:
              return True
        finally:
          physics.data.time = original_time

      if self._raise_exception_on_settle_failure:
        raise RuntimeError(
            _SETTLING_PHYSICS_FAILED.format(
                max_attempts=self._max_settle_physics_attempts,
                max_time=self._max_settle_physics_time,
                max_qvel=max_qvel,
                max_qacc=max_qacc,
            ))
      else:
        log_str = _SETTLING_PHYSICS_FAILED.format(
            max_attempts='%s',
            max_time='%s',
            max_qvel='%s',
            max_qacc='%s',
        )
        logging.warning(log_str, self._max_settle_physics_attempts,
                        self._max_settle_physics_time, max_qvel, max_qacc)

      return False

    if self._settle_physics:
      place_and_settle()
    else:
      place_props()
