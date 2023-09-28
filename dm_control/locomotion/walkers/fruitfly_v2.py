# Copyright 2023 The dm_control Authors.
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

"""Fruit fly model."""

import collections as col
import os
from typing import Sequence

from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.walkers import base
from dm_control.locomotion.walkers import legacy_base
from dm_control.mujoco import wrapper as mj_wrapper
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils import transformations
from dm_env import specs
import numpy as np
enums = mjbindings.enums
mjlib = mjbindings.mjlib


_XML_PATH = os.path.join(os.path.dirname(__file__),
                         'assets/fruitfly_v2/fruitfly.xml')
# === Constants.
_SPAWN_POS = np.array((0, 0, 0.1278))
# OrderedDict used to streamline enabling/disabling of action classes.
_ACTION_CLASSES = col.OrderedDict(adhesion=0,
                                  head=0,
                                  mouth=0,
                                  antennae=0,
                                  wings=0,
                                  abdomen=0,
                                  legs=0,
                                  user=0)


def neg_quat(quat_a):
  """Returns neg(quat_a)."""
  quat_b = quat_a.copy()
  quat_b[0] *= -1
  return quat_b


def mul_quat(quat_a, quat_b):
  """Returns quat_a * quat_b."""
  quat_c = np.zeros(4)
  mjlib.mju_mulQuat(quat_c, quat_a, quat_b)
  return quat_c


def mul_jac_t_vec(physics, efc):
  """Maps forces from constraint space to joint space."""
  qfrc = np.zeros(physics.model.nv)
  mjlib.mj_mulJacTVec(physics.model.ptr, physics.data.ptr, qfrc, efc)
  return qfrc


def rot_vec_quat(vec, quat):
  """Rotates vector with quaternion."""
  res = np.zeros(3)
  mjlib.mju_rotVecQuat(res, vec, quat)
  return res


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
  """Checks if any of substrings is in string."""
  return any(s in string for s in substrings)


def body_quat_from_springrefs(body: 'mjcf.element') -> np.ndarray:
  """Computes new body quat from all joint springrefs and current quat."""
  joints = body.joint
  if not joints:
    return None
  # Construct quaternions for all joint axes.
  quats = []
  for joint in joints:
    theta = joint.springref or joint.dclass.joint.springref or 0
    axis = joint.axis or joint.dclass.joint.axis
    if axis is None:
      axis = joint.dclass.parent.joint.axis
    quats.append(np.hstack((np.cos(theta/2), np.sin(theta/2) * axis)))
  # Compute the new orientation quaternion.
  quat = np.array([1., 0, 0, 0])
  for i in range(len(quats)):
    quat = transformations.quat_mul(quats[-1-i], quat)
  if body.quat is not None:
    quat = transformations.quat_mul(body.quat, quat)
  return quat


def change_body_frame(body, frame_pos, frame_quat):
  """Change the frame of a body while maintaining child locations."""
  frame_pos = np.zeros(3) if frame_pos is None else frame_pos
  frame_quat = np.array((1., 0, 0, 0)) if frame_quat is None else frame_quat
  # Get frame transformation.
  body_pos = np.zeros(3) if body.pos is None else body.pos
  dpos = body_pos - frame_pos
  body_quat = np.array((1., 0, 0, 0)) if body.quat is None else body.quat
  dquat = mul_quat(neg_quat(frame_quat), body_quat)
  # Translate and rotate the body to the new frame.
  body.pos = frame_pos
  body.quat = frame_quat
  # Move all its children to their previous location.
  for child in body.all_children():
    if not hasattr(child, 'pos'):
      continue
    # Rotate:
    if hasattr(child, 'quat'):
      child_quat = np.array((1., 0, 0, 0)) if child.quat is None else child.quat
      child.quat = mul_quat(dquat, child_quat)
    # Translate, accounting for rotations.
    child_pos = np.zeros(3) if child.pos is None else child.pos
    pos_in_parent = rot_vec_quat(child_pos, body_quat) + dpos
    child.pos = rot_vec_quat(pos_in_parent, neg_quat(frame_quat))


#-------------------------------------------------------------------------------


class FruitFly(legacy_base.Walker):
  """A fruit fly model."""

  def _build(self,
             name: str = 'walker',
             use_legs: bool = True,
             use_wings: bool = False,
             use_mouth: bool = False,
             use_antennae: bool = False,
             joint_filter: float = 0.01,
             adhesion_filter: float = 0.01,
             body_pitch_angle: float = 47.5,
             stroke_plane_angle: float = 0.,
             physics_timestep: float = 1e-4,
             control_timestep: float = 2e-3,
             num_user_actions: int = 0,
             eye_camera_fovy: float = 150.,
             eye_camera_size: int = 32,
             ):
    """Build a fruitfly walker.

    Args:
      name: Name of the walker.
      use_legs: Whether to use or retract the legs.
      use_wings: Whether to use or retract the wings.
      use_mouth: Whether to use or retract the mouth.
      use_antennae: Whether to use the antennae.
      joint_filter: Timescale of filter for joint actuators. 0: disabled.
      adhesion_filter: Timescale of filter for adhesion actuators. 0: disabled.
      body_pitch_angle: Body pitch angle for initial flight pose, relative to
        ground, degrees. 0: horizontal body position. Default value from
        https://doi.org/10.1126/science.1248955
      stroke_plane_angle: Angle of wing stroke plane for initial flight pose,
        relative to ground, degrees. 0: horizontal stroke plane.
      physics_timestep: Timestep of the simulation.
      control_timestep: Timestep of the controller.
      num_user_actions: Optional, number of additional actions for custom usage,
        e.g. in before_step callback. The action range is [-1, 1]. 0: Not used.
      eye_camera_fovy: Vertical field of view of the eye cameras, degrees. The
        horizontal field of view is computed automatically given the window
        size.
      eye_camera_size: Size in pixels (height and width) of the eye cameras.
        Height and width are assumed equal.
    """
    self._adhesion_filter = adhesion_filter
    self._control_timestep = control_timestep
    self._buffer_size = int(round(control_timestep/physics_timestep))
    self._eye_camera_size = eye_camera_size
    root = mjcf.from_path(_XML_PATH)
    self._mjcf_root = root
    if name:
      self._mjcf_root.model = name

    # Remove freejoint.
    root.find('joint', 'free').remove()
    # Set eye camera fovy.
    root.find('camera', 'eye_right').fovy = eye_camera_fovy
    root.find('camera', 'eye_left').fovy = eye_camera_fovy

    # Identify actuator/body/joint/tendon classes by substrings in their names.
    name_substr = {'adhesion': [],
                   'head': ['head'],
                   'mouth': ['rostrum', 'haustellum', 'labrum'],
                   'antennae': ['antenna'],
                   'wings': ['wing'],
                   'abdomen': ['abdomen'],
                   'legs': ['T1', 'T2', 'T3'],
                   'user': []}

    # === Retract disabled body parts and remove their actuators.

    # Maybe retract and disable legs.
    if not use_legs:
      # Set orientation quaternions to retracted leg position.
      leg_bodies = [b for b in root.find_all('body')
                    if any_substr_in_str(name_substr['legs'], b.name)]
      for body in leg_bodies:
        body.quat = body_quat_from_springrefs(body)
      # Remove leg tendons and tendon actuators.
      for tendon in root.find_all('tendon'):
        if any_substr_in_str(name_substr['legs'], tendon.name):
          # Assume tendon actuator names are the same as tendon names.
          actuator = root.find('actuator', tendon.name)
          if actuator is not None:
            actuator.remove()
          tendon.remove()
      # Remove leg actuators and joints.
      leg_joints = [j for j in root.find_all('joint')
                    if any_substr_in_str(name_substr['legs'], j.name)]
      for joint in leg_joints:
        # Assume joint actuator names are the same as joint names.
        actuator = root.find('actuator', joint.name)
        if actuator is not None:
          actuator.remove()
        self.observable_joints.remove(joint)
        joint.remove()
      # Remove leg adhesion actuators.
      for actuator in root.find_all('actuator'):
        if ('adhere' in actuator.name and
            any_substr_in_str(name_substr['legs'], actuator.name)):
          actuator.remove()

    # Maybe retract and disable wings.
    if not use_wings:
      wing_joints = [j for j in root.find_all('joint')
                     if any_substr_in_str(name_substr['wings'], j.name)]
      for joint in wing_joints:
        root.find('actuator', joint.name).remove()
        self.observable_joints.remove(joint)

    # Maybe disable mouth.
    if not use_mouth:
      mouth_joints = [j for j in root.find_all('joint')
                      if any_substr_in_str(name_substr['mouth'], j.name)]
      for joint in mouth_joints:
        root.find('actuator', joint.name).remove()
        self.observable_joints.remove(joint)
      # Remove mouth adhesion actuators.
      for actuator in root.find_all('actuator'):
        if ('adhere' in actuator.name and
            any_substr_in_str(name_substr['mouth'], actuator.name)):
          actuator.remove()

    # Maybe disable antennae.
    if not use_antennae:
      antenna_joints = [j for j in root.find_all('joint')
                        if any_substr_in_str(name_substr['antennae'], j.name)]
      for joint in antenna_joints:
        root.find('actuator', joint.name).remove()
        self.observable_joints.remove(joint)

    # === For flight, set body pitch angle and stroke plane angle.
    if use_wings:
      # == Set body pitch angle.
      up_dir = root.find('site', 'hover_up_dir').quat
      up_dir_angle = 2 * np.arccos(up_dir[0])
      delta = np.deg2rad(body_pitch_angle) - up_dir_angle
      dquat = np.array([np.cos(delta/2), 0, np.sin(delta/2), 0])
      # Rotate up_dir to new angle.
      up_dir[:] = mul_quat(dquat, up_dir)
      # == Set stroke plane angle.
      stroke_plane_angle = np.deg2rad(stroke_plane_angle)
      stroke_plane_quat = np.array([np.cos(stroke_plane_angle/2), 0,
                                    np.sin(stroke_plane_angle/2), 0])
      for quat, wing in [(np.array([0., 0, 0, 1]), 'wing_left'),
                         (np.array([0., -1, 0, 0]), 'wing_right')]:
        # Rotate wing-joint frame.
        dquat = mul_quat(neg_quat(stroke_plane_quat), quat)
        new_wing_quat = mul_quat(dquat, neg_quat(up_dir))
        body = root.find('body', wing)
        change_body_frame(body, body.pos, new_wing_quat)

    # === Maybe change actuator dynamics to `filter`.
    if joint_filter > 0:
      for actuator in root.find_all('actuator'):
        if actuator.tag != 'adhesion':
          actuator.dyntype = 'filter'
          actuator.dynprm = (joint_filter,)
    if adhesion_filter > 0:
      for actuator in root.find_all('actuator'):
        if actuator.tag == 'adhesion':
          actuator.dclass.parent.general.dyntype = 'filter'
          actuator.dclass.parent.general.dynprm = (adhesion_filter,)

    # === Get action-class indices into the MuJoCo control vector.
    # Find all ctrl indices except adhesion.
    self._ctrl_indices = _ACTION_CLASSES.copy()
    names = [a.name for a in root.find_all('actuator')]
    for act_class in self._ctrl_indices.keys():
      indices = [i for i, name in enumerate(names)
                 if any_substr_in_str(name_substr[act_class], name)
                 and 'adhere' not in name]
      self._ctrl_indices[act_class] = indices if indices else None
    # Find adhesion ctrl indices.
    indices = [i for i, name in enumerate(names) if 'adhere' in name]
    self._ctrl_indices['adhesion'] = indices if indices else None

    # === Count the number of actions in each action-class.
    self._num_actions = _ACTION_CLASSES.copy()

    # User actions, if any.
    self._num_actions['user'] = num_user_actions

    # The rest of action classes, including adhesion.
    for act_class in self._num_actions.keys():
      if self._ctrl_indices[act_class] is not None:
        self._num_actions[act_class] = len(self._ctrl_indices[act_class])

    # === Get action-class indices into the environment action vector.
    self._action_indices = _ACTION_CLASSES.copy()
    counter = 0
    for act_class in _ACTION_CLASSES.keys():
      if self._num_actions[act_class]:
        indices = list(range(counter, counter + self._num_actions[act_class]))
        self._action_indices[act_class] = indices
        counter += self._num_actions[act_class]
      else:
        self._action_indices[act_class] = []

    super()._build()

  #-----------------------------------------------------------------------------

  def initialize_episode(self, physics: 'mjcf.Physics',
                         random_state: np.random.RandomState):
    """Set the walker."""
    # Save the weight of the body (in Dyne i.e. gram*cm/s^2).
    body_mass = physics.named.model.body_subtreemass['walker/thorax']  # gram.
    self._weight = np.linalg.norm(physics.model.opt.gravity) * body_mass

  #-----------------------------------------------------------------------------

  @property
  def upright_pose(self):
    return base.WalkerPose(xpos=_SPAWN_POS)

  @property
  def weight(self):
    return self._weight

  @property
  def adhesion_filter(self):
    return self._adhesion_filter

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @composer.cached_property
  def root_body(self):
    """Return the body."""
    return self.mjcf_model.find('body', 'thorax')

  @composer.cached_property
  def thorax(self):
    """Return the thorax."""
    return self.mjcf_model.find('body', 'thorax')

  @composer.cached_property
  def abdomen(self):
    """Return the abdomen."""
    return self.mjcf_model.find('body', 'abdomen')

  @composer.cached_property
  def head(self):
    """Return the head."""
    return self.mjcf_model.find('body', 'head')

  @composer.cached_property
  def head_site(self):
    """Return the head."""
    return self.mjcf_model.find('site', 'head')

  @composer.cached_property
  def observable_joints(self):
    return self.mjcf_model.find_all('joint')

  @composer.cached_property
  def actuators(self):
    return self.mjcf_model.find_all('actuator')

  @composer.cached_property
  def mocap_tracking_bodies(self):
    # Which bodies to track?
    body_names = (
        'thorax', 'abdomen', 'head',
        'claw_T1_left', 'claw_T1_right',
        'claw_T2_left', 'claw_T2_right',
        'claw_T3_left', 'claw_T3_right')
    bodies = []
    for body_name in body_names:
      body = self.mjcf_model.find('body', body_name)
      if body:
        bodies.append(body)
    return tuple(bodies)

  @composer.cached_property
  def end_effectors(self):
    site_names = ('claw_T1_left', 'claw_T1_right',
                  'claw_T2_left', 'claw_T2_right',
                  'claw_T3_left', 'claw_T3_right')
    sites = []
    for site_name in site_names:
      site = self.mjcf_model.find('site', site_name)
      if site:
        sites.append(site)
    return tuple(sites)

  @composer.cached_property
  def appendages(self):
    # wings? mouth? antennae?
    additional_site_names = ('head',)
    sites = list(self.end_effectors)
    for site_name in additional_site_names:
      sites.append(self.mjcf_model.find('site', site_name))
    return tuple(sites)

  def _build_observables(self):
    return FruitFlyObservables(self, self._buffer_size, self._eye_camera_size)

  @composer.cached_property
  def left_eye(self):
    """Return the left_eye camera."""
    return self._mjcf_root.find('camera', 'eye_left')

  @composer.cached_property
  def right_eye(self):
    """Return the right_eye camera."""
    return self._mjcf_root.find('camera', 'eye_right')

  @composer.cached_property
  def egocentric_camera(self):
    """Required by legacy_base."""
    return self._mjcf_root.find('camera', 'eye_right')

  @composer.cached_property
  def ground_contact_geoms(self):
    """Return ground contact geoms."""
    return (self._mjcf_root.find('geom', 'tarsal_claw_T1_left_collision'),
            self._mjcf_root.find('geom', 'tarsal_claw_T1_right_collision'),
            self._mjcf_root.find('geom', 'tarsal_claw_T2_left_collision'),
            self._mjcf_root.find('geom', 'tarsal_claw_T2_right_collision'),
            self._mjcf_root.find('geom', 'tarsal_claw_T3_left_collision'),
            self._mjcf_root.find('geom', 'tarsal_claw_T3_right_collision'),
            )

  #-----------------------------------------------------------------------------

  def apply_action(self, physics, action, random_state):
    """Apply action to walker's actuators."""
    del random_state
    if not self.mjcf_model.find_all('actuator'):
      return
    # Apply MuJoCo actions.
    ctrl = np.zeros(physics.model.nu)
    for key, indices in self._action_indices.items():
      if self._ctrl_indices[key] and indices:
        ctrl[self._ctrl_indices[key]] = action[indices]
    physics.set_control(ctrl)

  #-----------------------------------------------------------------------------

  def get_action_spec(self, physics):
    """Returns a `BoundedArray` spec matching this walker's actuators."""
    minimum = []
    maximum = []

    # MuJoCo actions.
    indices = []
    for key, _ in self._action_indices.items():
      if self._ctrl_indices[key] and self._num_actions[key]:
        indices.extend(self._ctrl_indices[key])
    mj_minima, mj_maxima = physics.model.actuator_ctrlrange[indices].T
    names = [physics.model.id2name(i, 'actuator') or str(i)
             for i in indices]
    names = [s.split('/')[-1] for s in names]
    num_actions = len(indices)
    minimum.extend(mj_minima)
    maximum.extend(mj_maxima)

    # User actions.
    if self._num_actions['user']:
      minimum.extend(self._num_actions['user'] * [-1.0])
      maximum.extend(self._num_actions['user'] * [1.0])
      names.extend([f'user_{i}' for i in range(self._num_actions['user'])])
      num_actions += self._num_actions['user']

    return specs.BoundedArray(shape=(num_actions,),
                              dtype=float,
                              minimum=np.asarray(minimum),
                              maximum=np.asarray(maximum),
                              name='\t'.join(names))

#-------------------------------------------------------------------------------


class FruitFlyObservables(legacy_base.WalkerObservables):
  """Observables for the fruit fly."""

  def __init__(self, walker, buffer_size, eye_camera_size):
    self._walker = walker
    self._buffer_size = buffer_size
    self._eye_camera_size = eye_camera_size
    super().__init__(walker)

  @composer.observable
  def thorax_height(self):
    """Observe the thorax height."""
    return observable.MJCFFeature('xpos', self._entity.thorax)[2]

  @composer.observable
  def abdomen_height(self):
    """Observe the abdomen height."""
    return observable.MJCFFeature('xpos', self._entity.abdomen)[2]

  @composer.observable
  def world_zaxis_hover(self):
    """The world's z-vector in this Walker's torso frame."""
    hover_up_dir = self._walker.mjcf_model.find('site', 'hover_up_dir')
    return observable.MJCFFeature('xmat', hover_up_dir)[6:]

  @composer.observable
  def world_zaxis(self):
    """The world's z-vector in this Walker's torso frame."""
    return observable.MJCFFeature('xmat', self._entity.root_body)[6:]

  @composer.observable
  def world_zaxis_abdomen(self):
    """The world's z-vector in this Walker's abdomen frame."""
    return observable.MJCFFeature('xmat', self._entity.abdomen)[6:]

  @composer.observable
  def world_zaxis_head(self):
    """The world's z-vector in this Walker's head frame."""
    return observable.MJCFFeature('xmat', self._entity.head)[6:]

  @composer.observable
  def force(self):
    """Force sensors."""
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.force,
                                  buffer_size=self._buffer_size,
                                  aggregator='mean')

  @composer.observable
  def touch(self):
    """Touch sensors."""
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.touch,
                                  buffer_size=self._buffer_size,
                                  aggregator='mean')

  @composer.observable
  def accelerometer(self):
    """Accelerometer readings."""
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.accelerometer,
                                  buffer_size=self._buffer_size,
                                  aggregator='mean')

  @composer.observable
  def gyro(self):
    """Gyro readings."""
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.gyro,
                                  buffer_size=self._buffer_size,
                                  aggregator='mean')

  @composer.observable
  def velocimeter(self):
    """Velocimeter readings."""
    return observable.MJCFFeature('sensordata',
                                  self._entity.mjcf_model.sensor.velocimeter,
                                  buffer_size=self._buffer_size,
                                  aggregator='mean')

  @composer.observable
  def actuator_activation(self):
    """Observe the actuator activation."""
    model = self._entity.mjcf_model
    return observable.MJCFFeature('act', model.find_all('actuator'))

  @composer.observable
  def appendages_pos(self):
    """Equivalent to `end_effectors_pos` but may include other appendages."""

    def relative_pos_in_egocentric_frame(physics):
      appendages = physics.bind(self._entity.appendages).xpos
      torso_pos = physics.bind(self._entity.root_body).xpos
      torso_mat = np.reshape(physics.bind(self._entity.root_body).xmat, (3, 3))
      return np.reshape(np.dot(appendages - torso_pos, torso_mat), -1)

    return observable.Generic(relative_pos_in_egocentric_frame)

  @composer.observable
  def self_contact(self):
    """Returns the sum of self-contact forces."""
    def sum_body_contact_forces(physics):
      walker_id = physics.model.name2id('walker/', 'body')
      force = np.array((0.0))
      for contact_id, contact in enumerate(physics.data.contact):
        # Both geoms must be descendants of the thorax.
        body1 = physics.model.geom_bodyid[contact.geom1]
        body2 = physics.model.geom_bodyid[contact.geom2]
        root1 = physics.model.body_rootid[body1]
        root2 = physics.model.body_rootid[body2]
        if not(root1 == walker_id and root2 == walker_id):
          continue
        contact_force, _ = physics.data.contact_force(contact_id)
        force += np.linalg.norm(contact_force)
      return force
    return observable.Generic(sum_body_contact_forces,
                              buffer_size=self._buffer_size,
                              aggregator='mean')

  @property
  def vestibular(self):
    """Return vestibular information."""
    return [self.gyro, self.accelerometer,
            self.velocimeter, self.world_zaxis]

  @property
  def proprioception(self):
    """Return proprioceptive information."""
    return [self.joints_pos, self.joints_vel,
            self.actuator_activation]

  @property
  def orientation(self):
    """Return orientation of world z-axis in local frame."""
    return [self.world_zaxis, self.world_zaxis_abdomen, self.world_zaxis_head]

  @composer.observable
  def right_eye(self):
    """Observable of the right_eye camera."""

    if not hasattr(self, '_scene_options'):
      # Render this walker's geoms.
      self._scene_options = mj_wrapper.MjvOption()
      cosmetic_geom_group = 1
      self._scene_options.geomgroup[cosmetic_geom_group] = 1

    return observable.MJCFCamera(self._entity.right_eye,
                                 width=self._eye_camera_size,
                                 height=self._eye_camera_size,
                                 scene_option=self._scene_options)

  @composer.observable
  def left_eye(self):
    """Observable of the left_eye camera."""

    if not hasattr(self, '_scene_options'):
      # Render this walker's geoms.
      self._scene_options = mj_wrapper.MjvOption()
      cosmetic_geom_group = 1
      self._scene_options.geomgroup[cosmetic_geom_group] = 1

    return observable.MJCFCamera(self._entity.left_eye,
                                 width=self._eye_camera_size,
                                 height=self._eye_camera_size,
                                 scene_option=self._scene_options)
