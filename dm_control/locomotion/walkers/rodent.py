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
"""A Rodent walker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np

_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'assets/rodent.xml')

_RAT_MOCAP_JOINTS = [
    'vertebra_1_extend', 'vertebra_2_bend', 'vertebra_3_twist',
    'vertebra_4_extend', 'vertebra_5_bend', 'vertebra_6_twist',
    'hip_L_supinate', 'hip_L_abduct', 'hip_L_extend', 'knee_L', 'ankle_L',
    'toe_L', 'hip_R_supinate', 'hip_R_abduct', 'hip_R_extend', 'knee_R',
    'ankle_R', 'toe_R', 'vertebra_C1_extend', 'vertebra_C1_bend',
    'vertebra_C2_extend', 'vertebra_C2_bend', 'vertebra_C3_extend',
    'vertebra_C3_bend', 'vertebra_C4_extend', 'vertebra_C4_bend',
    'vertebra_C5_extend', 'vertebra_C5_bend', 'vertebra_C6_extend',
    'vertebra_C6_bend', 'vertebra_C7_extend', 'vertebra_C9_bend',
    'vertebra_C11_extend', 'vertebra_C13_bend', 'vertebra_C15_extend',
    'vertebra_C17_bend', 'vertebra_C19_extend', 'vertebra_C21_bend',
    'vertebra_C23_extend', 'vertebra_C25_bend', 'vertebra_C27_extend',
    'vertebra_C29_bend', 'vertebra_cervical_5_extend',
    'vertebra_cervical_4_bend', 'vertebra_cervical_3_twist',
    'vertebra_cervical_2_extend', 'vertebra_cervical_1_bend',
    'vertebra_axis_twist', 'vertebra_atlant_extend', 'atlas', 'mandible',
    'scapula_L_supinate', 'scapula_L_abduct', 'scapula_L_extend', 'shoulder_L',
    'shoulder_sup_L', 'elbow_L', 'wrist_L', 'finger_L', 'scapula_R_supinate',
    'scapula_R_abduct', 'scapula_R_extend', 'shoulder_R', 'shoulder_sup_R',
    'elbow_R', 'wrist_R', 'finger_R'
]


_UPRIGHT_POS = (0.0, 0.0, 0.0)
_UPRIGHT_QUAT = (1., 0., 0., 0.)
_TORQUE_THRESHOLD = 60


class Rat(legacy_base.Walker):
  """A position-controlled rat with control range scaled to [-1, 1]."""

  def _build(self,
             params=None,
             name='walker',
             initializer=None):
    self.params = params
    self._mjcf_root = mjcf.from_path(_XML_PATH)
    if name:
      self._mjcf_root.model = name

    self.body_sites = []
    super(Rat, self)._build(initializer=initializer)

  @property
  def upright_pose(self):
    """Reset pose to upright position."""
    return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

  @property
  def mjcf_model(self):
    """Return the model root."""
    return self._mjcf_root

  @composer.cached_property
  def actuators(self):
    """Return all actuators."""
    return tuple(self._mjcf_root.find_all('actuator'))

  @composer.cached_property
  def root_body(self):
    """Return the body."""
    return self._mjcf_root.find('body', 'torso')

  @composer.cached_property
  def pelvis_body(self):
    """Return the body."""
    return self._mjcf_root.find('body', 'pelvis')

  @composer.cached_property
  def head(self):
    """Return the head."""
    return self._mjcf_root.find('body', 'skull')

  @composer.cached_property
  def left_arm_root(self):
    """Return the left arm."""
    return self._mjcf_root.find('body', 'scapula_L')

  @composer.cached_property
  def right_arm_root(self):
    """Return the right arm."""
    return self._mjcf_root.find('body', 'scapula_R')

  @composer.cached_property
  def ground_contact_geoms(self):
    """Return ground contact geoms."""
    return tuple(
        self._mjcf_root.find('body', 'foot_L').find_all('geom') +
        self._mjcf_root.find('body', 'foot_R').find_all('geom'))

  @composer.cached_property
  def standing_height(self):
    """Return standing height."""
    return self.params['_STAND_HEIGHT']

  @composer.cached_property
  def end_effectors(self):
    """Return end effectors."""
    return (self._mjcf_root.find('body', 'lower_arm_R'),
            self._mjcf_root.find('body', 'lower_arm_L'),
            self._mjcf_root.find('body', 'foot_R'),
            self._mjcf_root.find('body', 'foot_L'))

  @composer.cached_property
  def observable_joints(self):
    """Return observable joints."""
    return tuple(actuator.joint
                 for actuator in self.actuators  #  This lint is mistaken; pylint: disable=not-an-iterable
                 if actuator.joint is not None)

  @composer.cached_property
  def observable_tendons(self):
    return self._mjcf_root.find_all('tendon')

  @composer.cached_property
  def mocap_joints(self):
    return tuple(
        self._mjcf_root.find('joint', name) for name in _RAT_MOCAP_JOINTS)

  @composer.cached_property
  def mocap_joint_order(self):
    return tuple([jnt.name for jnt in self.mocap_joints])  #  This lint is mistaken; pylint: disable=not-an-iterable

  @composer.cached_property
  def bodies(self):
    """Return all bodies."""
    return tuple(self._mjcf_root.find_all('body'))

  @composer.cached_property
  def mocap_bodies(self):
    """Return bodies for mocap comparison."""
    return tuple(body for body in self._mjcf_root.find_all('body')
                 if not re.match(r'(vertebra|hand|toe)', body.name))

  @composer.cached_property
  def primary_joints(self):
    """Return primary (non-vertebra) joints."""
    return tuple(jnt for jnt in self._mjcf_root.find_all('joint')
                 if 'vertebra' not in jnt.name)

  @composer.cached_property
  def vertebra_joints(self):
    """Return vertebra joints."""
    return tuple(jnt for jnt in self._mjcf_root.find_all('joint')
                 if 'vertebra' in jnt.name)

  @composer.cached_property
  def primary_joint_order(self):
    joint_names = self.mocap_joint_order
    primary_names = tuple([jnt.name for jnt in self.primary_joints])  # pylint: disable=not-an-iterable
    primary_order = []
    for nm in primary_names:
      primary_order.append(joint_names.index(nm))
    return primary_order

  @composer.cached_property
  def vertebra_joint_order(self):
    joint_names = self.mocap_joint_order
    vertebra_names = tuple([jnt.name for jnt in self.vertebra_joints])  # pylint: disable=not-an-iterable
    vertebra_order = []
    for nm in vertebra_names:
      vertebra_order.append(joint_names.index(nm))
    return vertebra_order

  @composer.cached_property
  def egocentric_camera(self):
    """Return the egocentric camera."""
    return self._mjcf_root.find('camera', 'egocentric')

  @property
  def _xml_path(self):
    """Return the path to th model .xml file."""
    return self.params['_XML_PATH']

  @composer.cached_property
  def joint_actuators(self):
    """Return all joint actuators."""
    return tuple([act for act in self._mjcf_root.find_all('actuator')
                  if act.joint])

  @composer.cached_property
  def joint_actuators_range(self):
    act_joint_range = []
    for act in self.joint_actuators:  #  This lint is mistaken; pylint: disable=not-an-iterable
      associated_joint = self._mjcf_root.find('joint', act.name)
      act_range = associated_joint.dclass.joint.range
      act_joint_range.append(act_range)
    return act_joint_range

  def pose_to_actuation(self, pose):
    # holds for joint actuators, find desired torque = 0
    # u_ref = [2 q_ref - (r_low + r_up) ]/(r_up - r_low)
    r_lower = np.array([ajr[0] for ajr in self.joint_actuators_range])  #  This lint is mistaken; pylint: disable=not-an-iterable
    r_upper = np.array([ajr[1] for ajr in self.joint_actuators_range])  #  This lint is mistaken; pylint: disable=not-an-iterable
    num_tendon_actuators = len(self.actuators) - len(self.joint_actuators)
    tendon_actions = np.zeros(num_tendon_actuators)
    return np.hstack([tendon_actions, (2*pose[self.joint_actuator_order]-
                                       (r_lower+r_upper))/(r_upper-r_lower)])

  @composer.cached_property
  def joint_actuator_order(self):
    joint_names = self.mocap_joint_order
    joint_actuator_names = tuple([act.name for act in self.joint_actuators])  #  This lint is mistaken; pylint: disable=not-an-iterable
    actuator_order = []
    for nm in joint_actuator_names:
      actuator_order.append(joint_names.index(nm))
    return actuator_order

  def _build_observables(self):
    return RodentObservables(self)


class RodentObservables(legacy_base.WalkerObservables):
  """Observables for the Rat."""

  @composer.observable
  def head_height(self):
    """Observe the head height."""
    return observable.MJCFFeature('xpos', self._entity.head)[2]

  @composer.observable
  def sensors_torque(self):
    """Observe the torque sensors."""
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.sensor.torque,
        corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_THRESHOLD)
        )

  @composer.observable
  def tendons_pos(self):
    return observable.MJCFFeature('length', self._entity.observable_tendons)

  @composer.observable
  def tendons_vel(self):
    return observable.MJCFFeature('velocity', self._entity.observable_tendons)

  @composer.observable
  def actuator_activation(self):
    """Observe the actuator activation."""
    model = self._entity.mjcf_model
    return observable.MJCFFeature('act', model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` with head's position appended."""

    def relative_pos_in_egocentric_frame(physics):
      end_effectors_with_head = (
          self._entity.end_effectors + (self._entity.head,))
      end_effector = physics.bind(end_effectors_with_head).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = \
          np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)

    return observable.Generic(relative_pos_in_egocentric_frame)

  @property
  def proprioception(self):
    """Return proprioceptive information."""
    return [
        self.joints_pos, self.joints_vel,
        self.tendons_pos, self.tendons_vel,
        self.actuator_activation,
        self.body_height, self.end_effectors_pos, self.appendages_pos,
        self.world_zaxis
    ] + self._collect_from_attachments('proprioception')

  @composer.observable
  def egocentric_camera(self):
    """Observable of the egocentric camera."""

    if not hasattr(self, '_scene_options'):
      # Don't render this walker's geoms.
      self._scene_options = mj_wrapper.MjvOption()
      collision_geom_group = 2
      self._scene_options.geomgroup[collision_geom_group] = 0
      cosmetic_geom_group = 1
      self._scene_options.geomgroup[cosmetic_geom_group] = 0

    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=64, height=64,
                                 scene_option=self._scene_options
                                )
