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

"""Make neck for the dog model."""

import collections

import numpy as np

from dm_control import mjcf


def create_neck(model, bone_position, cervical_dofs_per_vertebra,
                bones, side_sign, bone_size, parent):
  """Add neck and head in the dog model.

    Args:
      model: model in which we want to add the neck.
      bone_position: a dictionary of bones positions.
      cervical_dofs_per_vertebra: a number that the determines how many
        dofs are going to be used between each pair of cervical vetebrae.
      bones: a list of strings with all the names of the bones.
      side_sign: a dictionary with two axis representing the signs of
          translations.
      bone_size: dictionary containing the scale of the geometry.
      parent: parent object on which we should start attaching new components.
  """
  # Cervical Spine
  def_cervical = model.default.find('default', 'cervical')
  def_cervical_extend = model.default.find('default', 'cervical_extend')
  def_cervical_bend = model.default.find('default', 'cervical_bend')
  def_cervical_twist = model.default.find('default', 'cervical_twist')
  cervical_defaults = {'extend': def_cervical_extend,
                       'bend': def_cervical_bend,
                       'twist': def_cervical_twist}

  cervical_bones = ['C_' + str(i) for i in range(7, 0, -1)]
  parent_pos = parent.pos
  cervical_bodies = []
  cervical_geoms = []
  radius = .07
  for i, bone in enumerate(cervical_bones):
    bone_pos = bone_position[bone]
    rel_pos = bone_pos - parent_pos
    child = parent.add('body', name=bone, pos=rel_pos)
    cervical_bodies.append(child)
    dclass = 'bone' if i > 3 else 'light_bone'
    geom = child.add('geom', name=bone, mesh=bone, pos=-bone_pos,
                     dclass=dclass)
    child.add(
      'geom',
      name=bone + '_collision',
      type='sphere',
      size=[radius],
      dclass='nonself_collision_primitive')
    radius -= .006
    cervical_geoms.append(geom)
    parent = child
    parent_pos = bone_pos

  # Reload
  physics = mjcf.Physics.from_mjcf_model(model)

  # Cervical (neck) spine joints:
  cervical_axis = collections.OrderedDict()
  cervical_axis['extend'] = np.array((0., 1., 0.))
  cervical_axis['bend'] = np.array((0., 0., 1.))
  cervical_axis['twist'] = np.array((1., 0., 0))

  num_dofs = 0
  cervical_joints = []
  cervical_joint_names = []
  torso = model.find('body', 'torso')
  parent = torso.find('geom', 'T_1')
  for i, vertebra in enumerate(cervical_bodies):
    while num_dofs < (i + 1) * cervical_dofs_per_vertebra:
      dof = num_dofs % 3
      dof_name = list(cervical_axis.keys())[dof]
      dof_axis = cervical_axis[dof_name]
      cervical_joint_names.append(vertebra.name + '_' + dof_name)

      rel_pos = physics.bind(vertebra).xpos - physics.bind(parent).xpos
      twist_dir = rel_pos / np.linalg.norm(rel_pos)
      bend_dir = np.cross(twist_dir, cervical_axis['extend'])
      cervical_axis['bend'] = bend_dir
      cervical_axis['twist'] = twist_dir
      joint_frame = np.vstack((twist_dir, cervical_axis['extend'], bend_dir))
      joint_pos = def_cervical.joint.pos * physics.bind(
        vertebra.find('geom', vertebra.name)).size.mean()
      joint = vertebra.add(
        'joint',
        name=cervical_joint_names[-1],
        dclass='cervical_' + dof_name,
        axis=cervical_axis[dof_name],
        pos=joint_pos.dot(joint_frame))
      cervical_joints.append(joint)
      num_dofs += 1
    parent = vertebra

  # Lumbar spine joints:
  lumbar_axis = collections.OrderedDict()
  lumbar_axis['extend'] = np.array((0., 1., 0.))
  lumbar_axis['bend'] = np.array((0., 0., 1.))
  lumbar_axis['twist'] = np.array((1., 0., 0))

  # Scale joint defaults relative to 3 cervical_dofs_per_vertebra
  for dof in lumbar_axis.keys():
    axis_scale = 7.0 / [dof in joint for joint in cervical_joint_names].count(
      True)
    cervical_defaults[dof].joint.range *= axis_scale

  # Reload
  physics = mjcf.Physics.from_mjcf_model(model)

  # Skull
  c_1 = cervical_bodies[-1]
  upper_teeth = [m for m in bones if 'Top' in m]
  skull_bones = upper_teeth + ['Skull', 'Ethmoid', 'Vomer', 'eye_L', 'eye_R']
  skull = c_1.add(
    'body', name='skull', pos=bone_position['Skull'] - physics.bind(c_1).xpos)
  skull_geoms = []
  for bone in skull_bones:
    geom = skull.add(
      'geom',
      name=bone,
      mesh=bone,
      pos=-bone_position['Skull'],
      dclass='light_bone')
    if 'eye' in bone:
      geom.rgba = [1, 1, 1, 1]
      geom.dclass = 'visible_bone'
    skull_geoms.append(geom)
    if bone in upper_teeth:
      geom.dclass = 'visible_bone'

  for side in ['_L', '_R']:
    pos = np.array((0.023, -0.027, 0.01)) * side_sign[side]
    skull.add(
      'geom',
      name='iris' + side,
      type='ellipsoid',
      dclass='visible_bone',
      rgba=(0.45, 0.45, 0.225, .4),
      size=(.003, .007, .007),
      pos=pos,
      euler=[0, 0, -20 * (1. if side == '_R' else -1.)])
    pos = np.array((0.0215, -0.0275, 0.01)) * side_sign[side]
    skull.add(
      'geom',
      name='pupil' + side,
      type='sphere',
      dclass='visible_bone',
      rgba=(0, 0, 0, 1),
      size=(.003, 0, 0),
      pos=pos)

  # collision geoms
  skull.add(
    'geom',
    name='skull0' + '_collision',
    type='ellipsoid',
    dclass='collision_primitive',
    size=(.06, .06, .04),
    pos=(-.02, 0, .01),
    euler=[0, 10, 0])
  skull.add(
    'geom',
    name='skull1' + '_collision',
    type='capsule',
    dclass='collision_primitive',
    size=(0.015, 0.04, 0.015),
    pos=(.06, 0, -.01),
    euler=[0, 110, 0])
  skull.add(
    'geom',
    name='skull2' + '_collision',
    type='box',
    dclass='collision_primitive',
    size=(.03, .028, .008),
    pos=(.02, 0, -.03))
  skull.add(
    'geom',
    name='skull3' + '_collision',
    type='box',
    dclass='collision_primitive',
    size=(.02, .018, .006),
    pos=(.07, 0, -.03))
  skull.add(
    'geom',
    name='skull4' + '_collision',
    type='box',
    dclass='collision_primitive',
    size=(.005, .015, .004),
    pos=(.095, 0, -.03))

  skull.add('joint',
            name='atlas',
            dclass='atlas',
            pos=np.array((-.5, 0, 0)) * bone_size['Skull'])

  head = skull.add(
    'site', name='head', size=(.01, .01, .01), type='box',
    dclass='sensor')
  skull.add(
    'site',
    name='upper_bite',
    size=(.005,),
    dclass='sensor',
    pos=(.065, 0, -.07))
  # Jaw
  lower_teeth = [m for m in bones if 'Bottom' in m]
  jaw_bones = lower_teeth + ['Mandible']
  jaw = skull.add(
    'body',
    name='jaw',
    pos=bone_position['Mandible'] - bone_position['Skull'])
  jaw_geoms = []
  for bone in jaw_bones:
    geom = jaw.add(
      'geom',
      name=bone,
      mesh=bone,
      pos=-bone_position['Mandible'],
      dclass='light_bone')
    jaw_geoms.append(geom)
    if bone in lower_teeth:
      geom.dclass = 'visible_bone'
  # Jaw collision geoms:
  jaw_col_pos = [(-0.03, 0, 0.01), (0, 0, -0.012), (0.03, 0, -0.028),
                 (0.052, 0, -0.035)]
  jaw_col_size = [(0.03, 0.028, 0.008), (0.02, 0.022, 0.005),
                  (0.02, 0.018, 0.005), (0.015, 0.013, 0.003)]
  jaw_col_angle = [55, 30, 25, 15]
  for i in range(4):
    jaw.add(
      'geom',
      name='jaw' + str(i) + '_collision',
      type='box',
      dclass='collision_primitive',
      size=jaw_col_size[i],
      pos=jaw_col_pos[i],
      euler=[0, jaw_col_angle[i], 0])

  jaw.add('joint',
          name='mandible',
          dclass='mandible',
          axis=[0, 1, 0],
          pos=np.array((-0.043, 0, 0.05)))
  jaw.add(
    'site',
    name='lower_bite',
    size=(.005,),
    dclass='sensor',
    pos=(.063, 0, 0.005))

  print('Make collision ellipsoids for teeth.')
  visible_bones = upper_teeth + lower_teeth
  for bone in visible_bones:
    bone_geom = torso.find('geom', bone)
    bone_geom.type = 'ellipsoid'
  physics = mjcf.Physics.from_mjcf_model(model)
  for bone in visible_bones:
    bone_geom = torso.find('geom', bone)
    pos = physics.bind(bone_geom).pos
    quat = physics.bind(bone_geom).quat
    size = physics.bind(bone_geom).size
    bone_geom.parent.add(
      'geom',
      name=bone + '_collision',
      dclass='tooth_primitive',
      pos=pos,
      size=size * 1.2,
      quat=quat,
      type='ellipsoid')
    bone_geom.type = None

  return cervical_joints
