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

"""Make front legs for the dog model."""

import numpy as np

from dm_control import mjcf


def create_front_legs(nails, model, primary_axis, bones, side_sign, parent):
  """Add front legs in the model.

    Args:
      nails: a list of string with the geoms representing nails.
      model: model in which we want to add the front legs.
      primary_axis: a dictionary of numpy arrays representing axis of rotation.
      bones: a list of strings with all the names of the bones.
      side_sign: a dictionary with two axis representing the signs of
          translations.
      parent: parent object on which we should start attaching new components.
    """
  def_scapula_supinate = model.default.find('default', 'scapula_supinate')
  def_scapula_abduct = model.default.find('default', 'scapula_abduct')
  def_scapula_extend = model.default.find('default', 'scapula_extend')

  scapula_defaults = {'_abduct': def_scapula_abduct,
                      '_extend': def_scapula_extend,
                      '_supinate': def_scapula_supinate}

  torso = parent
  # Shoulders
  scapula = {}
  scapulae = [b for b in bones if 'Scapula' in b]
  scapula_pos = np.array((.08, -.02, .14))
  for side in ['_L', '_R']:
    body_pos = scapula_pos * side_sign[side]
    arm = torso.add('body', name='scapula' + side, pos=body_pos)
    scapula[side] = arm
    for bone in [b for b in scapulae if side in b]:
      arm.add(
        'geom',
        name=bone,
        mesh=bone,
        pos=-torso.pos - body_pos,
        dclass='bone')

    # Shoulder joints
    for dof in ['_supinate', '_abduct', '_extend']:
      joint_axis = primary_axis[dof].copy()
      joint_pos = scapula_defaults[dof].joint.pos.copy()
      if dof != '_extend':
        joint_axis *= 1. if side == '_R' else -1.
        joint_pos *= side_sign[side]
      else:
        joint_axis += .3 * (1 if side == '_R' else -1) * primary_axis['_abduct']
      arm.add(
        'joint',
        name='scapula' + side + dof,
        dclass='scapula' + dof,
        axis=joint_axis,
        pos=joint_pos)

    # Shoulder sites
    shoulder_pos = np.array((.075, -.033, -.13))
    arm.add('site', name='shoulder' + side, size=[0.01],
            pos=shoulder_pos * side_sign[side])

  # Upper Arms:
  upper_arm = {}
  parent_pos = {}
  humeri = ['humerus_R', 'humerus_L']
  for side in ['_L', '_R']:
    body_pos = shoulder_pos * side_sign[side]
    parent = scapula[side]
    parent_pos[side] = torso.pos + parent.pos
    arm = parent.add('body', name='upper_arm' + side, pos=body_pos)
    upper_arm[side] = arm
    for bone in [b for b in humeri if side in b]:
      arm.add(
        'geom',
        name=bone,
        mesh=bone,
        pos=-parent_pos[side] - body_pos,
        dclass='bone')
    parent_pos[side] += body_pos

    # Shoulder joints
    for dof in ['_supinate', '_extend']:
      joint_axis = primary_axis[dof].copy()
      if dof == '_supinate':
        joint_axis[0] = 1
        joint_axis *= 1. if side == '_R' else -1.
      arm.add(
        'joint',
        name='shoulder' + side + dof,
        dclass='shoulder' + dof,
        axis=joint_axis)

    # Elbow sites
    elbow_pos = np.array((-.05, -.015, -.145))
    arm.add(
      'site',
      type='cylinder',
      name='elbow' + side,
      size=[0.003, .02],
      zaxis=(0, 1, -(1. if side == '_R' else -1.) * .2),
      pos=elbow_pos * side_sign[side])

  # Lower arms:
  lower_arm = {}
  lower_arm_bones = [
    b for b in bones
    if 'ulna' in b.lower() or 'Radius' in b or 'accessory' in b
  ]
  for side in ['_L', '_R']:
    body_pos = elbow_pos * side_sign[side]
    arm = upper_arm[side].add('body', name='lower_arm' + side, pos=body_pos)
    lower_arm[side] = arm
    for bone in [b for b in lower_arm_bones if side in b]:
      arm.add(
        'geom',
        name=bone,
        mesh=bone,
        pos=-parent_pos[side] - body_pos,
        dclass='bone')
    # Elbow joints
    elbow_axis = upper_arm[side].find_all('site')[0].zaxis
    arm.add('joint', name='elbow' + side, dclass='elbow', axis=elbow_axis)
    parent_pos[side] += body_pos

    # Wrist sites
    wrist_pos = np.array((.003, .015, -0.19))
    arm.add('site', type='cylinder', name='wrist' + side, size=[0.004, .017],
            zaxis=(0, 1, 0), pos=wrist_pos * side_sign[side])

  # Hands:
  hands = {}
  hand_bones = [
    b for b in bones
    if ('carpal' in b.lower() and 'acces' not in b and 'ulna' not in b) or
       ('distalis_digiti_I_' in b)
  ]
  for side in ['_L', '_R']:
    body_pos = wrist_pos * side_sign[side]
    hand_anchor = lower_arm[side].add(
      'body', name='hand_anchor' + side, pos=body_pos)
    hand_anchor.add(
      'geom',
      name=hand_anchor.name,
      dclass='foot_primitive',
      type='box',
      size=(.005, .005, .005),
      contype=0,
      conaffinity=0)
    hand_anchor.add('site', name=hand_anchor.name, dclass='sensor')
    hand = hand_anchor.add('body', name='hand' + side)
    hands[side] = hand
    for bone in [b for b in hand_bones if side in b]:
      hand.add(
        'geom',
        name=bone,
        mesh=bone,
        pos=-parent_pos[side] - body_pos,
        dclass='bone')
    # Wrist joints
    hand.add('joint', name='wrist' + side, dclass='wrist', axis=(0, -1, 0))
    hand.add(
      'geom',
      name=hand.name + '_collision',
      size=[0.03, 0.016, 0.012],
      pos=[.01, 0, -.04],
      euler=(0, 65, 0),
      dclass='collision_primitive',
      type='box')

    parent_pos[side] += body_pos

    # Finger sites
    finger_pos = np.array((.02, 0, -.06))
    hand.add(
      'site',
      type='cylinder',
      name='finger' + side,
      size=[0.003, .025],
      zaxis=((1. if side == '_R' else -1.) * .2, 1, 0),
      pos=finger_pos * side_sign[side])

  # Fingers:
  finger_bones = [
    b for b in bones if 'Phalanx' in b and 'distalis_digiti_I_' not in b
  ]
  palm_sites = []
  for side in ['_L', '_R']:
    body_pos = finger_pos * side_sign[side]
    hand = hands[side].add('body', name='finger' + side, pos=body_pos)
    for bone in [b for b in finger_bones if side in b]:
      geom = hand.add(
        'geom',
        name=bone,
        mesh=bone,
        pos=-parent_pos[side] - body_pos,
        dclass='bone')
      if 'distalis' in bone:
        nails.append(bone)
        geom.dclass = 'visible_bone'
    # Finger joints
    finger_axis = upper_arm[side].find_all('site')[0].zaxis
    hand.add('joint', name='finger' + side, dclass='finger', axis=finger_axis)
    hand.add(
      'geom',
      name=hand.name + '0_collision',
      size=[0.018, 0.012],
      pos=[.012, 0, -.012],
      euler=(90, 0, 0),
      dclass='foot_primitive')
    hand.add(
      'geom',
      name=hand.name + '1_collision',
      size=[0.01, 0.015],
      pos=[.032, 0, -.02],
      euler=(90, 0, 0),
      dclass='foot_primitive')
    hand.add(
      'geom',
      name=hand.name + '2_collision',
      size=[0.008, 0.01],
      pos=[.042, 0, -.022],
      euler=(90, 0, 0),
      dclass='foot_primitive')

    palm = hand.add(
      'site',
      name='palm' + side,
      size=(.028, .03, .007),
      pos=(.02, 0, -.024),
      type='box',
      dclass='sensor')
    palm_sites.append(palm)

  physics = mjcf.Physics.from_mjcf_model(model)

  for side in ['_L', '_R']:
    # scapula:
    scap = scapula[side]
    geom = scap.get_children('geom')[0]
    bound_geom = physics.bind(geom)
    scap.add(
      'geom',
      name=geom.name + '_collision',
      pos=bound_geom.pos,
      size=bound_geom.size * 0.8,
      quat=bound_geom.quat,
      type='box',
      dclass='collision_primitive')
    # upper arm:
    arm = upper_arm[side]
    geom = arm.get_children('geom')[0]
    bound_geom = physics.bind(geom)
    arm.add(
      'geom',
      name=geom.name + '_collision',
      pos=bound_geom.pos,
      size=[.02, .08],
      quat=bound_geom.quat,
      dclass='collision_primitive')

  all_geoms = model.find_all('geom')
  for geom in all_geoms:
    if 'Ulna' in geom.name:
      bound_geom = physics.bind(geom)
      geom.parent.add(
        'geom',
        name=geom.name + '_collision',
        pos=bound_geom.pos,
        size=[.015, .06],
        quat=bound_geom.quat,
        dclass='collision_primitive')
    if 'Radius' in geom.name:
      bound_geom = physics.bind(geom)
      pos = bound_geom.pos + np.array((-.005, 0., -.01))
      geom.parent.add(
        'geom',
        name=geom.name + '_collision',
        pos=pos,
        size=[.017, .09],
        quat=bound_geom.quat,
        dclass='collision_primitive')

  return palm_sites
