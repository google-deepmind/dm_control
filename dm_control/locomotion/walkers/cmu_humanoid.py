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
"""A CMU humanoid walker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.locomotion.walkers import rescale
from dm_control.locomotion.walkers import scaled_actuators
from dm_control.mujoco import wrapper as mj_wrapper
import numpy as np
import six
from six.moves import zip

_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'assets/humanoid_CMU_V{model_version}.xml')
_WALKER_GEOM_GROUP = 2
_WALKER_INVIS_GROUP = 1

_CMU_MOCAP_JOINTS = (
    'lfemurrz', 'lfemurry', 'lfemurrx', 'ltibiarx', 'lfootrz', 'lfootrx',
    'ltoesrx', 'rfemurrz', 'rfemurry', 'rfemurrx', 'rtibiarx', 'rfootrz',
    'rfootrx', 'rtoesrx', 'lowerbackrz', 'lowerbackry', 'lowerbackrx',
    'upperbackrz', 'upperbackry', 'upperbackrx', 'thoraxrz', 'thoraxry',
    'thoraxrx', 'lowerneckrz', 'lowerneckry', 'lowerneckrx', 'upperneckrz',
    'upperneckry', 'upperneckrx', 'headrz', 'headry', 'headrx', 'lclaviclerz',
    'lclaviclery', 'lhumerusrz', 'lhumerusry', 'lhumerusrx', 'lradiusrx',
    'lwristry', 'lhandrz', 'lhandrx', 'lfingersrx', 'lthumbrz', 'lthumbrx',
    'rclaviclerz', 'rclaviclery', 'rhumerusrz', 'rhumerusry', 'rhumerusrx',
    'rradiusrx', 'rwristry', 'rhandrz', 'rhandrx', 'rfingersrx', 'rthumbrz',
    'rthumbrx')


# pylint: disable=bad-whitespace
PositionActuatorParams = collections.namedtuple(
    'PositionActuatorParams', ['name', 'forcerange', 'kp'])
_POSITION_ACTUATORS = [
    PositionActuatorParams('headrx',      [-20,   20 ], 20 ),
    PositionActuatorParams('headry',      [-20,   20 ], 20 ),
    PositionActuatorParams('headrz',      [-20,   20 ], 20 ),
    PositionActuatorParams('lclaviclery', [-20,   20 ], 20 ),
    PositionActuatorParams('lclaviclerz', [-20,   20 ], 20 ),
    PositionActuatorParams('lfemurrx',    [-120,  120], 120),
    PositionActuatorParams('lfemurry',    [-80,   80 ], 80 ),
    PositionActuatorParams('lfemurrz',    [-80,   80 ], 80 ),
    PositionActuatorParams('lfingersrx',  [-20,   20 ], 20 ),
    PositionActuatorParams('lfootrx',     [-50,   50 ], 50 ),
    PositionActuatorParams('lfootrz',     [-50,   50 ], 50 ),
    PositionActuatorParams('lhandrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('lhandrz',     [-20,   20 ], 20 ),
    PositionActuatorParams('lhumerusrx',  [-60,   60 ], 60 ),
    PositionActuatorParams('lhumerusry',  [-60,   60 ], 60 ),
    PositionActuatorParams('lhumerusrz',  [-60,   60 ], 60 ),
    PositionActuatorParams('lowerbackrx', [-120,  120], 150),
    PositionActuatorParams('lowerbackry', [-120,  120], 150),
    PositionActuatorParams('lowerbackrz', [-120,  120], 150),
    PositionActuatorParams('lowerneckrx', [-20,   20 ], 20 ),
    PositionActuatorParams('lowerneckry', [-20,   20 ], 20 ),
    PositionActuatorParams('lowerneckrz', [-20,   20 ], 20 ),
    PositionActuatorParams('lradiusrx',   [-60,   60 ], 60 ),
    PositionActuatorParams('lthumbrx',    [-20,   20 ], 20) ,
    PositionActuatorParams('lthumbrz',    [-20,   20 ], 20 ),
    PositionActuatorParams('ltibiarx',    [-80,   80 ], 80 ),
    PositionActuatorParams('ltoesrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('lwristry',    [-20,   20 ], 20 ),
    PositionActuatorParams('rclaviclery', [-20,   20 ], 20 ),
    PositionActuatorParams('rclaviclerz', [-20,   20 ], 20 ),
    PositionActuatorParams('rfemurrx',    [-120,  120], 120),
    PositionActuatorParams('rfemurry',    [-80,   80 ], 80 ),
    PositionActuatorParams('rfemurrz',    [-80,   80 ], 80 ),
    PositionActuatorParams('rfingersrx',  [-20,   20 ], 20 ),
    PositionActuatorParams('rfootrx',     [-50,   50 ], 50 ),
    PositionActuatorParams('rfootrz',     [-50,   50 ], 50 ),
    PositionActuatorParams('rhandrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('rhandrz',     [-20,   20 ], 20 ),
    PositionActuatorParams('rhumerusrx',  [-60,   60 ], 60 ),
    PositionActuatorParams('rhumerusry',  [-60,   60 ], 60 ),
    PositionActuatorParams('rhumerusrz',  [-60,   60 ], 60 ),
    PositionActuatorParams('rradiusrx',   [-60,   60 ], 60 ),
    PositionActuatorParams('rthumbrx',    [-20,   20 ], 20 ),
    PositionActuatorParams('rthumbrz',    [-20,   20 ], 20 ),
    PositionActuatorParams('rtibiarx',    [-80,   80 ], 80 ),
    PositionActuatorParams('rtoesrx',     [-20,   20 ], 20 ),
    PositionActuatorParams('rwristry',    [-20,   20 ], 20 ),
    PositionActuatorParams('thoraxrx',    [-80,   80 ], 100),
    PositionActuatorParams('thoraxry',    [-80,   80 ], 100),
    PositionActuatorParams('thoraxrz',    [-80,   80 ], 100),
    PositionActuatorParams('upperbackrx', [-80,   80 ], 80 ),
    PositionActuatorParams('upperbackry', [-80,   80 ], 80 ),
    PositionActuatorParams('upperbackrz', [-80,   80 ], 80 ),
    PositionActuatorParams('upperneckrx', [-20,   20 ], 20 ),
    PositionActuatorParams('upperneckry', [-20,   20 ], 20 ),
    PositionActuatorParams('upperneckrz', [-20,   20 ], 20 ),
]
PositionActuatorParamsV2020 = collections.namedtuple(
    'PositionActuatorParams', ['name', 'forcerange', 'kp', 'damping'])
_POSITION_ACTUATORS_V2020 = [
    PositionActuatorParamsV2020('headrx',      [-40,   40 ], 40 , 2 ),
    PositionActuatorParamsV2020('headry',      [-40,   40 ], 40 , 2 ),
    PositionActuatorParamsV2020('headrz',      [-40,   40 ], 40 , 2 ),
    PositionActuatorParamsV2020('lclaviclery', [-80,   80 ], 80 , 20),
    PositionActuatorParamsV2020('lclaviclerz', [-80,   80 ], 80 , 20),
    PositionActuatorParamsV2020('lfemurrx',    [-300,  300], 300, 15),
    PositionActuatorParamsV2020('lfemurry',    [-200,  200], 200, 10),
    PositionActuatorParamsV2020('lfemurrz',    [-200,  200], 200, 10),
    PositionActuatorParamsV2020('lfingersrx',  [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('lfootrx',     [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('lfootrz',     [-50,   50 ], 50 , 3 ),
    PositionActuatorParamsV2020('lhandrx',     [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('lhandrz',     [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('lhumerusrx',  [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('lhumerusry',  [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('lhumerusrz',  [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('lowerbackrx', [-300,  300], 300, 15),
    PositionActuatorParamsV2020('lowerbackry', [-180,  180], 180, 20),
    PositionActuatorParamsV2020('lowerbackrz', [-200,  200], 200, 20),
    PositionActuatorParamsV2020('lowerneckrx', [-120,  120 ],120, 20),
    PositionActuatorParamsV2020('lowerneckry', [-120,  120 ],120, 20),
    PositionActuatorParamsV2020('lowerneckrz', [-120,  120 ],120, 20),
    PositionActuatorParamsV2020('lradiusrx',   [-90,   90 ], 90 , 5 ),
    PositionActuatorParamsV2020('lthumbrx',    [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('lthumbrz',    [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('ltibiarx',    [-160,  160], 160, 8 ),
    PositionActuatorParamsV2020('ltoesrx',     [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('lwristry',    [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rclaviclery', [-80,   80 ], 80 , 20),
    PositionActuatorParamsV2020('rclaviclerz', [-80,   80 ], 80 , 20),
    PositionActuatorParamsV2020('rfemurrx',    [-300,  300], 300, 15),
    PositionActuatorParamsV2020('rfemurry',    [-200,  200], 200, 10),
    PositionActuatorParamsV2020('rfemurrz',    [-200,  200], 200, 10),
    PositionActuatorParamsV2020('rfingersrx',  [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rfootrx',     [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('rfootrz',     [-50,   50 ], 50 , 3 ),
    PositionActuatorParamsV2020('rhandrx',     [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rhandrz',     [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rhumerusrx',  [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('rhumerusry',  [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('rhumerusrz',  [-120,  120], 120, 6 ),
    PositionActuatorParamsV2020('rradiusrx',   [-90,   90 ], 90 , 5 ),
    PositionActuatorParamsV2020('rthumbrx',    [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rthumbrz',    [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rtibiarx',    [-160,  160], 160, 8 ),
    PositionActuatorParamsV2020('rtoesrx',     [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('rwristry',    [-20,   20 ], 20 , 1 ),
    PositionActuatorParamsV2020('thoraxrx',    [-300,  300], 300, 15),
    PositionActuatorParamsV2020('thoraxry',    [-80,   80],  80 , 8 ),
    PositionActuatorParamsV2020('thoraxrz',    [-200,  200], 200, 12),
    PositionActuatorParamsV2020('upperbackrx', [-300,  300], 300, 15),
    PositionActuatorParamsV2020('upperbackry', [-80,   80],  80 , 8 ),
    PositionActuatorParamsV2020('upperbackrz', [-200,  200], 200, 12),
    PositionActuatorParamsV2020('upperneckrx', [-60,   60 ], 60 , 10),
    PositionActuatorParamsV2020('upperneckry', [-60,   60 ], 60 , 10),
    PositionActuatorParamsV2020('upperneckrz', [-60,   60 ], 60 , 10),
]

# pylint: enable=bad-whitespace

_UPRIGHT_POS = (0.0, 0.0, 0.94)
_UPRIGHT_QUAT = (0.859, 1.0, 1.0, 0.859)

# Height of head above which the humanoid is considered standing.
_STAND_HEIGHT = 1.5

_TORQUE_THRESHOLD = 60


@six.add_metaclass(abc.ABCMeta)
class _CMUHumanoidBase(legacy_base.Walker):
  """The abstract base class for walkers compatible with the CMU humanoid."""

  def _build(self,
             name='walker',
             marker_rgba=None,
             include_face=False,
             initializer=None):
    self._mjcf_root = mjcf.from_path(self._xml_path)
    if name:
      self._mjcf_root.model = name

    # Set corresponding marker color if specified.
    if marker_rgba is not None:
      for geom in self.marker_geoms:
        geom.set_attributes(rgba=marker_rgba)

    self._actuator_order = np.argsort(_CMU_MOCAP_JOINTS)
    self._inverse_order = np.argsort(self._actuator_order)

    super(_CMUHumanoidBase, self)._build(initializer=initializer)

    if include_face:
      head = self._mjcf_root.find('body', 'head')
      face_forwardness = head.pos[1]-.02
      head_geom = self._mjcf_root.find('geom', 'head')
      nose_size = head_geom.size[0] / 4.75
      face = head.add(
          'body', name='face', pos=(0.0, 0.039, face_forwardness))
      face.add('geom',
               type='capsule',
               name='nose',
               size=(nose_size, 0.01),
               pos=(0.0, 0.0, 0.0),
               quat=(1, 0.7, 0, 0),
               mass=0.,
               contype=0,
               conaffinity=0,
               group=_WALKER_INVIS_GROUP)

  def _build_observables(self):
    return CMUHumanoidObservables(self)

  @abc.abstractproperty
  def _xml_path(self):
    raise NotImplementedError

  @composer.cached_property
  def mocap_joints(self):
    return tuple(
        self._mjcf_root.find('joint', name) for name in _CMU_MOCAP_JOINTS)

  @property
  def actuator_order(self):
    """Index of joints from the CMU mocap dataset sorted alphabetically by name.

    Actuators in this walkers are ordered alphabetically by name. This property
    provides a mapping between from actuator ordering to canonical CMU ordering.

    Returns:
      A list of integers corresponding to joint indices from the CMU dataset.
      Specifically, the n-th element in the list is the index of the CMU joint
      index that corresponds to the n-th actuator in this walker.
    """
    return self._actuator_order

  @property
  def actuator_to_joint_order(self):
    """Index of actuators corresponding to each CMU mocap joint.

    Actuators in this walkers are ordered alphabetically by name. This property
    provides a mapping between from canonical CMU ordering to actuator ordering.

    Returns:
      A list of integers corresponding to actuator indices within this walker.
      Specifically, the n-th element in the list is the index of the actuator
      in this walker that corresponds to the n-th joint from the CMU mocap
      dataset.
    """
    return self._inverse_order

  @property
  def upright_pose(self):
    return base.WalkerPose(xpos=_UPRIGHT_POS, xquat=_UPRIGHT_QUAT)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @composer.cached_property
  def actuators(self):
    return tuple(self._mjcf_root.find_all('actuator'))

  @composer.cached_property
  def root_body(self):
    return self._mjcf_root.find('body', 'root')

  @composer.cached_property
  def head(self):
    return self._mjcf_root.find('body', 'head')

  @composer.cached_property
  def left_arm_root(self):
    return self._mjcf_root.find('body', 'lclavicle')

  @composer.cached_property
  def right_arm_root(self):
    return self._mjcf_root.find('body', 'rclavicle')

  @composer.cached_property
  def ground_contact_geoms(self):
    return tuple(self._mjcf_root.find('body', 'lfoot').find_all('geom') +
                 self._mjcf_root.find('body', 'rfoot').find_all('geom'))

  @composer.cached_property
  def standing_height(self):
    return _STAND_HEIGHT

  @composer.cached_property
  def end_effectors(self):
    return (self._mjcf_root.find('body', 'rradius'),
            self._mjcf_root.find('body', 'lradius'),
            self._mjcf_root.find('body', 'rfoot'),
            self._mjcf_root.find('body', 'lfoot'))

  @composer.cached_property
  def observable_joints(self):
    return tuple(actuator.joint for actuator in self.actuators
                 if actuator.joint is not None)

  @composer.cached_property
  def bodies(self):
    return tuple(self._mjcf_root.find_all('body'))

  @composer.cached_property
  def mocap_tracking_bodies(self):
    """Collection of bodies for mocap tracking."""
    # remove root body
    root_body = self._mjcf_root.find('body', 'root')
    return tuple(
        b for b in self._mjcf_root.find_all('body') if b != root_body)

  @composer.cached_property
  def egocentric_camera(self):
    return self._mjcf_root.find('camera', 'egocentric')

  @composer.cached_property
  def body_camera(self):
    return self._mjcf_root.find('camera', 'bodycam')

  @property
  def marker_geoms(self):
    return (self._mjcf_root.find('geom', 'rradius'),
            self._mjcf_root.find('geom', 'lradius'))


class CMUHumanoid(_CMUHumanoidBase):
  """A CMU humanoid walker."""

  @property
  def _xml_path(self):
    return _XML_PATH.format(model_version='2019')


class CMUHumanoidPositionControlled(CMUHumanoid):
  """A position-controlled CMU humanoid with control range scaled to [-1, 1]."""

  def _build(self, model_version='2019', **kwargs):
    self._version = model_version
    if 'scale_default' in kwargs:
      scale_default = kwargs['scale_default']
      del kwargs['scale_default']
    else:
      scale_default = False

    super(CMUHumanoidPositionControlled, self)._build(**kwargs)

    if scale_default:
      # NOTE: This rescaling doesn't affect the attached hands
      rescale.rescale_humanoid(self, 1.2, 1.2, 70)

    # modify actuators
    if self._version == '2020':
      position_actuators = _POSITION_ACTUATORS_V2020
    else:
      position_actuators = _POSITION_ACTUATORS
    self._mjcf_root.default.general.forcelimited = 'true'
    self._mjcf_root.actuator.motor.clear()
    for actuator_params in position_actuators:
      associated_joint = self._mjcf_root.find('joint', actuator_params.name)
      if hasattr(actuator_params, 'damping'):
        associated_joint.damping = actuator_params.damping
      actuator = scaled_actuators.add_position_actuator(
          name=actuator_params.name,
          target=associated_joint,
          kp=actuator_params.kp,
          qposrange=associated_joint.range,
          ctrlrange=(-1, 1),
          forcerange=actuator_params.forcerange)
      if self._version == '2020':
        actuator.dyntype = 'filter'
        actuator.dynprm = [0.030]
    limits = zip(*(actuator.joint.range for actuator in self.actuators))  # pylint: disable=not-an-iterable
    lower, upper = (np.array(limit) for limit in limits)
    self._scale = upper - lower
    self._offset = upper + lower

  @property
  def _xml_path(self):
    return _XML_PATH.format(model_version=self._version)

  def cmu_pose_to_actuation(self, target_pose):
    """Creates the control signal corresponding a CMU mocap joints pose.

    Args:
      target_pose: An array containing the target position for each joint.
        These must be given in "canonical CMU order" rather than "qpos order",
        i.e. the order of `target_pose[self.actuator_order]` should correspond
        to the order of `physics.bind(self.actuators).ctrl`.

    Returns:
      An array of the same shape as `target_pose` containing inputs for position
      controllers. Writing these values into `physics.bind(self.actuators).ctrl`
      will cause the actuators to drive joints towards `target_pose`.
    """
    return (2 * target_pose[self.actuator_order] - self._offset) / self._scale


class CMUHumanoidPositionControlledV2020(CMUHumanoidPositionControlled):
  """A 2020 updated CMU humanoid walker; includes nose for head orientation."""

  def _build(self, **kwargs):
    super(CMUHumanoidPositionControlledV2020, self)._build(
        model_version='2020', scale_default=True, include_face=True, **kwargs)


class CMUHumanoidObservables(legacy_base.WalkerObservables):
  """Observables for the Humanoid."""

  @composer.observable
  def body_camera(self):
    options = mj_wrapper.MjvOption()

    # Don't render this walker's geoms.
    options.geomgroup[_WALKER_GEOM_GROUP] = 0
    return observable.MJCFCamera(
        self._entity.body_camera, width=64, height=64, scene_option=options)

  @composer.observable
  def egocentric_camera(self):
    options = mj_wrapper.MjvOption()

    # Don't render this walker's geoms.
    options.geomgroup[_WALKER_INVIS_GROUP] = 0
    return observable.MJCFCamera(self._entity.egocentric_camera,
                                 width=64, height=64, scene_option=options)

  @composer.observable
  def head_height(self):
    return observable.MJCFFeature('xpos', self._entity.head)[2]

  @composer.observable
  def sensors_torque(self):
    return observable.MJCFFeature(
        'sensordata', self._entity.mjcf_model.sensor.torque,
        corruptor=lambda v, random_state: np.tanh(2 * v / _TORQUE_THRESHOLD))

  @composer.observable
  def actuator_activation(self):
    return observable.MJCFFeature('act',
                                  self._entity.mjcf_model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` with the head's position appended."""
    def relative_pos_in_egocentric_frame(physics):
      end_effectors_with_head = (
          self._entity.end_effectors + (self._entity.head,))
      end_effector = physics.bind(end_effectors_with_head).xpos
      torso = physics.bind(self._entity.root_body).xpos
      xmat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(end_effector - torso, xmat), -1)
    return observable.Generic(relative_pos_in_egocentric_frame)

  @property
  def proprioception(self):
    return [
        self.joints_pos,
        self.joints_vel,
        self.actuator_activation,
        self.body_height,
        self.end_effectors_pos,
        self.appendages_pos,
        self.world_zaxis
    ] + self._collect_from_attachments('proprioception')
