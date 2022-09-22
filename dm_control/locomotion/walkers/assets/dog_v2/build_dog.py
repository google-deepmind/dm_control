# Copyright 2017 The dm_control Authors.
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

"""Make dog model."""

import collections
import os
import struct
from absl import app
from absl import flags

from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import enums
import editdistance
import ipdb  # pylint: disable=unused-import
from lxml import etree

import numpy as np
from scipy import spatial
from scipy.sparse.csgraph import minimum_spanning_tree as mst

from dm_control.utils import io as resources

flags.DEFINE_boolean(
    'make_skin', True, 'Whether to make a new dog_skin.skn')
flags.DEFINE_boolean(
    'use_tendons', False, 'Whether to add tendons to the model.')
flags.DEFINE_float(
    'lumbar_dofs_per_veterbra', 1.5,
    'Number of degrees of freedom per vertebra in lumbar spine.')
flags.DEFINE_float(
    'cervical_dofs_per_veterbra', 1.5,
    'Number of degrees of freedom vertebra in cervical spine.')
flags.DEFINE_float(
    'caudal_dofs_per_veterbra', 1,
    'Number of degrees of freedom vertebra in caudal spine.')

FLAGS = flags.FLAGS

BASE_MODEL = 'dog_base.xml'
ASSET_RELPATH = '../../../../suite/dog_assets'
ASSET_DIR = os.path.dirname(__file__) + '/' + ASSET_RELPATH


def main(argv):
  del argv

  # Read flags.
  if FLAGS.is_parsed():
    use_tendons = FLAGS.use_tendons
    lumbar_dofs_per_veterbra = FLAGS.lumbar_dofs_per_veterbra
    cervical_dofs_per_veterbra = FLAGS.cervical_dofs_per_veterbra
    caudal_dofs_per_veterbra = FLAGS.caudal_dofs_per_veterbra
    make_skin = FLAGS.make_skin
  else:
    use_tendons = FLAGS['use_tendons'].default
    lumbar_dofs_per_veterbra = FLAGS['lumbar_dofs_per_veterbra'].default
    cervical_dofs_per_veterbra = FLAGS['cervical_dofs_per_veterbra'].default
    caudal_dofs_per_veterbra = FLAGS['caudal_dofs_per_veterbra'].default
    make_skin = FLAGS['make_skin'].default

  print('Load base model.')
  with open(BASE_MODEL, 'r') as f:
    model = mjcf.from_file(f)

  # Helper constants:
  side_sign = {'_L': np.array((1., -1., 1.)),
               '_R': np.array((1., 1., 1.))}
  primary_axis = {'_abduct': np.array((-1., 0., 0.)),
                  '_extend': np.array((0., 1., 0.)),
                  '_supinate': np.array((0., 0., -1.))}

  # DEFAULTS

  # Lumbar Spine
  def_lumbar_extend = model.default.find('default', 'lumbar_extend')
  def_lumbar_bend = model.default.find('default', 'lumbar_bend')
  def_lumbar_twist = model.default.find('default', 'lumbar_twist')
  lumbar_defaults = {'extend': def_lumbar_extend,
                     'bend': def_lumbar_bend,
                     'twist': def_lumbar_twist}

  # Cervical Spine
  def_cervical = model.default.find('default', 'cervical')
  def_cervical_extend = model.default.find('default', 'cervical_extend')
  def_cervical_bend = model.default.find('default', 'cervical_bend')
  def_cervical_twist = model.default.find('default', 'cervical_twist')
  cervical_defaults = {'extend': def_cervical_extend,
                       'bend': def_cervical_bend,
                       'twist': def_cervical_twist}

  def_scapula_supinate = model.default.find('default', 'scapula_supinate')
  def_scapula_abduct = model.default.find('default', 'scapula_abduct')
  def_scapula_extend = model.default.find('default', 'scapula_extend')

  scapula_defaults = {'_abduct': def_scapula_abduct,
                      '_extend': def_scapula_extend,
                      '_supinate': def_scapula_supinate}

  # Add meshes:
  print('Loading all meshes, getting positions and sizes.')
  meshdir = ASSET_DIR
  model.compiler.meshdir = meshdir
  texturedir = ASSET_DIR
  model.compiler.texturedir = texturedir
  site_meshes = []
  muscle_meshes = []
  bones = []
  for dirpath, _, filenames in resources.WalkResources(meshdir):
    prefix = 'extras/' if 'extras' in dirpath else ''
    for filename in filenames:
      if 'dog_skin.msh' in filename:
        skin_msh = model.asset.add('mesh', name='skin_msh', file=filename,
                                   scale=(1.25, 1.25, 1.25))
      name = filename[4:-4]
      name = name.replace('*', ':')
      if filename.startswith('BONE'):
        if 'Lingual' not in name:
          bones.append(name)
          model.asset.add('mesh', name=name, file=prefix+filename)
      if use_tendons:
        if filename.startswith('SITE') and use_tendons:
          site = model.asset.add('mesh', name=name, file=prefix+filename)
          site_meshes.append(site)
        elif filename.startswith('MUSC'):
          muscle = model.asset.add('mesh', name=name, file=prefix+filename)
          muscle_meshes.append(muscle)

  # Put all bones in worldbody, get positions, remove bones:
  bone_geoms = []
  for bone in bones:
    geom = model.worldbody.add('geom', name=bone, mesh=bone, type='mesh',
                               contype=0, conaffinity=0, rgba=[1, .5, .5, .4])
    bone_geoms.append(geom)
  physics = mjcf.Physics.from_mjcf_model(model)
  bone_position = {}
  bone_size = {}
  for bone in bones:
    geom = model.find('geom', bone)
    bone_position[bone] = np.array(physics.bind(geom).xpos)
    bone_size[bone] = np.array(physics.bind(geom).rbound)
    geom.remove()

  # Put all sites in worldbody, get positions, remove site_geoms and meshes:
  if use_tendons:
    connector_sites = []
    site_geoms = []
    for site_mesh in site_meshes:
      geom = model.worldbody.add(
          'geom',
          name=site_mesh.name,
          mesh=site_mesh,
          type='mesh',
          contype=0,
          conaffinity=0,
          rgba=[1, .5, .5, .4])
      site_geoms.append(geom)
    physics = mjcf.Physics.from_mjcf_model(model)
    site_position = {}
    for site_mesh in site_meshes:
      geom = model.find('geom', site_mesh.name)
      site_position[site_mesh.name] = np.array(physics.bind(geom).xpos)
      geom.remove()
    for site_mesh in site_meshes:
      site_mesh.remove()
    for key, value in site_position.items():
      site = model.worldbody.add(
          'site', name=key, pos=value, dclass='connector')
      connector_sites.append(site)

    # Make connectors
    connectors = []
    for site in connector_sites:
      bone, muscle = site.name.split(':')
      if not bone.startswith(
          ('m_', 'Mm', 'Ligament', 'Lingual', '_', '-')) and not any(
              s in bone for s in ['_lig_', 'ascia']):
        connectors.append(
            [bone.replace('-', '_'),
             muscle.replace('-', '_'), site])
    connector_bones = set([con[0] for con in connectors])
    bad_bones = [bone for bone in connector_bones if bone not in bones]

    # fix bad bones
    for bb in bad_bones:
      dist = np.zeros(len(bones), dtype=int)
      for i, bone in enumerate(bones):
        dist[i] = int(editdistance.eval(bb, bone))
      nearest = bones[dist.argmin()]  # pylint: disable=invalid-sequence-index
      print('replacing '+bb+' with '+nearest)
      for c in connectors:
        if c[0] == bb:
          c[0] = nearest
    connector_bones = set([con[0] for con in connectors])
    bad_bones = [bone for bone in connector_bones if bone not in bones]
    assert not bad_bones

  # Torso
  print('Torso, lumbar spine, pelvis.')
  thoracic_spine = [m for m in bones if 'T_' in m]
  ribs = [m for m in bones if 'Rib' in m and 'cage' not in m]
  sternum = [m for m in bones if 'Sternum' in m]
  torso_bones = thoracic_spine + ribs + sternum + ['Xiphoid_cartilage']
  torso = model.worldbody.add('body', name='torso')
  root_joint = torso.add('freejoint', name='root')
  torso.add('site', name='root', size=(.01,), rgba=[0, 1, 0, 1])
  torso.add('light', name='light', mode='trackcom', pos=[0, 0, 3])
  torso.add('camera', name='y-axis', mode='trackcom', pos=[0, -1.5, .8],
            xyaxes=[1, 0, 0, 0, .6, 1])
  torso.add('camera', name='x-axis', mode='trackcom', pos=[2, 0, .5],
            xyaxes=[0, 1, 0, -.3, 0, 1])
  torso_geoms = []
  for bone in torso_bones:
    torso_geoms.append(
        torso.add('geom', name=bone, mesh=bone, dclass='light_bone'))

  # Reload, get CoM position, set pos
  physics = mjcf.Physics.from_mjcf_model(model)
  torso_pos = np.array(physics.bind(model.find('body', 'torso')).xipos)
  torso.pos = torso_pos
  for geom in torso_geoms:
    geom.pos = -torso_pos

  # Collision primitive for torso
  torso.add(
      'geom',
      name='collision_torso',
      dclass='nonself_collision_primitive',
      type='ellipsoid',
      pos=[0, 0, 0],
      size=[.2, .09, .11],
      euler=[0, 10, 0],
      density=200)

  # Lumbar spine bodies:
  lumbar_bones = ['L_1', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7']
  parent = torso
  parent_pos = torso_pos
  lumbar_bodies = []
  lumbar_geoms = []
  for i, bone in enumerate(lumbar_bones):
    bone_pos = bone_position[bone]
    child = parent.add('body', name=bone, pos=bone_pos-parent_pos)
    lumbar_bodies.append(child)
    geom = child.add('geom', name=bone, mesh=bone, pos=-bone_pos, dclass='bone')
    child.add(
        'geom',
        name=bone + '_collision',
        type='sphere',
        size=[.05,],
        pos=[0, 0, -.02],
        dclass='nonself_collision_primitive')
    lumbar_geoms.append(geom)
    parent = child
    parent_pos = bone_pos
  l_7 = parent

  # Lumbar spine joints:
  lumbar_axis = collections.OrderedDict()
  lumbar_axis['extend'] = np.array((0., 1., 0.))
  lumbar_axis['bend'] = np.array((0., 0., 1.))
  lumbar_axis['twist'] = np.array((1., 0., 0))

  num_dofs = 0
  lumbar_joints = []
  lumbar_joint_names = []
  for i, vertebra in enumerate(lumbar_bodies):
    while num_dofs < (i+1) * lumbar_dofs_per_veterbra:
      dof = num_dofs % 3
      dof_name = list(lumbar_axis.keys())[dof]
      dof_axis = lumbar_axis[dof_name]
      lumbar_joint_names.append(vertebra.name + '_' + dof_name)
      joint = vertebra.add(
          'joint',
          name=lumbar_joint_names[-1],
          dclass='lumbar_' + dof_name,
          axis=dof_axis)
      lumbar_joints.append(joint)
      num_dofs += 1

  # Scale joint defaults relative to 3 lumbar_dofs_per_veterbra
  for dof in lumbar_axis.keys():
    axis_scale = 7.0 / [dof in joint for joint in lumbar_joint_names
                       ].count(True)
    lumbar_defaults[dof].joint.range *= axis_scale

  # Pelvis:
  pelvis = l_7.add(
      'body', name='pelvis', pos=bone_position['Pelvis'] - bone_position['L_7'])
  pelvic_bones = ['Sacrum', 'Pelvis']
  pelvic_geoms = []
  for bone in pelvic_bones:
    geom = pelvis.add(
        'geom',
        name=bone,
        mesh=bone,
        pos=-bone_position['Pelvis'],
        dclass='bone')
    pelvic_geoms.append(geom)
  # Collision primitives for pelvis
  for side in ['_L', '_R']:
    pos = np.array((.01, -.02, -.01)) * side_sign[side]
    pelvis.add(
        'geom',
        name='collision_pelvis' + side,
        pos=pos,
        size=[.05, .05, 0],
        euler=[0, 70, 0],
        dclass='nonself_collision_primitive')

  print('Neck, skull, jaw.')
  # Cervical spine (neck) bodies:
  cervical_bones = ['C_' + str(i) for i in range(7, 0, -1)]
  parent = torso
  parent_pos = torso.pos
  cervical_bodies = []
  cervical_geoms = []
  radius = .07
  for i, bone in enumerate(cervical_bones):
    bone_pos = bone_position[bone]
    rel_pos = bone_pos-parent_pos
    child = parent.add('body', name=bone, pos=rel_pos)
    cervical_bodies.append(child)
    dclass = 'bone' if i > 3 else 'light_bone'
    geom = child.add('geom', name=bone, mesh=bone, pos=-bone_pos, dclass=dclass)
    child.add(
        'geom',
        name=bone + side + '_collision',
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
  parent = torso.find('geom', 'T_1')
  for i, vertebra in enumerate(cervical_bodies):
    while num_dofs < (i+1) * cervical_dofs_per_veterbra:
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

  # Scale joint defaults relative to 3 cervical_dofs_per_veterbra
  for dof in lumbar_axis.keys():
    axis_scale = 7.0 / [dof in joint for joint in cervical_joint_names
                       ].count(True)
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
    pos = np.array((0.023, -0.027, 0.01))*side_sign[side]
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
      'site', name='head', size=(.01, .01, .01), type='box', dclass='sensor')
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

  print('Back legs.')
  # Hip joint sites:
  scale = np.asarray([bone_size[bone] for bone in pelvic_bones]).mean()
  hip_pos = np.array((-.23, -.6, -.16)) * scale
  for side in ['_L', '_R']:
    pelvis.add(
        'site', name='hip' + side, size=[0.011], pos=hip_pos * side_sign[side])

  # Upper legs:
  upper_leg = {}
  femurs = [b for b in bones if 'Fem' in b]
  if not use_tendons:
    femurs += [b for b in bones if 'Patella' in b]
  for side in ['_L', '_R']:
    body_pos = hip_pos * side_sign[side]
    leg = pelvis.add('body', name='upper_leg' + side, pos=body_pos)
    upper_leg[side] = leg
    for bone in [b for b in femurs if side in b]:
      leg.add(
          'geom',
          name=bone,
          mesh=bone,
          pos=-bone_position['Pelvis'] - body_pos,
          dclass='bone')

    # Hip joints
    for dof in ['_supinate', '_abduct', '_extend']:
      axis = primary_axis[dof].copy()
      if dof != '_extend':
        axis *= 1. if side != '_R' else -1.
      leg.add('joint', name='hip' + side + dof, dclass='hip' + dof, axis=axis)

    # Knee sites
    scale = bone_size['Femoris_L']
    knee_pos = np.array((-0.2, -0.27, -1.45)) * scale
    leg.add('site', type='cylinder', name='knee' + side, size=[0.003, .02],
            zaxis=(0, 1, 0), pos=knee_pos*side_sign[side])
    pos = np.array((-.01, -.02, -.08))*side_sign[side]
    euler = [-10 * (1. if side == '_R' else -1.), 20, 0]
    leg.add(
        'geom',
        name=leg.name + '0_collision',
        pos=pos,
        size=[.04, .08],
        euler=euler,
        dclass='collision_primitive')
    pos = np.array((-.03, 0, -.05))
    euler = [-10*(1. if side == '_R' else -1.), 5, 0]
    leg.add(
        'geom',
        name=leg.name + '1_collision',
        pos=pos,
        size=[.04, .04],
        euler=euler,
        dclass='collision_primitive')

    # Patella
    if use_tendons:
      # Make patella body
      pass

  # Lower legs:
  lower_leg = {}
  lower_leg_bones = [b for b in bones if 'Tibia_' in b or 'Fibula' in b]
  for side in ['_L', '_R']:
    body_pos = knee_pos * side_sign[side]
    leg = upper_leg[side].add('body', name='lower_leg' + side, pos=body_pos)
    lower_leg[side] = leg
    for bone in [b for b in lower_leg_bones if side in b]:
      leg.add(
          'geom',
          name=bone,
          mesh=bone,
          pos=-bone_position['Pelvis'] - upper_leg[side].pos - body_pos,
          dclass='bone')
    # Knee joints
    leg.add('joint', name='knee' + side, dclass='knee', axis=(0, -1, 0))

    # Ankle sites
    scale = bone_size['Tibia_L']
    ankle_pos = np.array((-1.27, 0.04, -0.98)) * scale
    leg.add('site', type='cylinder', name='ankle' + side, size=[0.003, .013],
            zaxis=(0, 1, 0), pos=ankle_pos*side_sign[side])

  # Feet:
  foot = {}
  foot_bones = [b for b in bones if 'tars' in b.lower() or 'tuber' in b]
  for side in ['_L', '_R']:
    body_pos = ankle_pos * side_sign[side]
    leg = lower_leg[side].add('body', name='foot' + side, pos=body_pos)
    foot[side] = leg
    for bone in [b for b in foot_bones if side in b]:
      leg.add(
          'geom',
          name=bone,
          mesh=bone,
          pos=-bone_position['Pelvis'] - upper_leg[side].pos -
          lower_leg[side].pos - body_pos,
          dclass='bone')
    # Ankle joints
    leg.add('joint', name='ankle' + side, dclass='ankle', axis=(0, 1, 0))
    pos = np.array((-.01, -.005, -.05)) * side_sign[side]
    leg.add(
        'geom',
        name=leg.name + '_collision',
        size=[.015, .07],
        pos=pos,
        dclass='collision_primitive')

    # Toe sites
    scale = bone_size['Metatarsi_R_2']
    toe_pos = np.array((-.37, -.2, -2.95)) * scale
    leg.add('site', type='cylinder', name='toe' + side, size=[0.003, .025],
            zaxis=(0, 1, 0), pos=toe_pos*side_sign[side])

  # Toes:
  toe_bones = [b for b in bones if 'Phalange' in b]
  toe_geoms = []
  sole_sites = []
  nails = []
  for side in ['_L', '_R']:
    body_pos = toe_pos * side_sign[side]
    foot_anchor = foot[side].add(
        'body', name='foot_anchor' + side, pos=body_pos)
    foot_anchor.add(
        'geom',
        name=foot_anchor.name,
        dclass='foot_primitive',
        type='box',
        size=(.005, .005, .005),
        contype=0,
        conaffinity=0)
    foot_anchor.add('site', name=foot_anchor.name, dclass='sensor')
    leg = foot_anchor.add('body', name='toe' + side)
    for bone in [b for b in toe_bones if side in b]:
      geom = leg.add(
          'geom',
          name=bone,
          mesh=bone,
          pos=-bone_position['Pelvis'] - upper_leg[side].pos -
          lower_leg[side].pos - foot[side].pos - body_pos,
          dclass='bone')
      if 'B_3' in bone:
        nails.append(bone)
        geom.dclass = 'visible_bone'
      else:
        toe_geoms.append(geom)
    # Toe joints
    leg.add('joint', name='toe' + side, dclass='toe', axis=(0, 1, 0))
    # Collision geoms
    leg.add(
        'geom',
        name=leg.name + '0_collision',
        size=[0.018, 0.012],
        pos=[.015, 0, -.02],
        euler=(90, 0, 0),
        dclass='foot_primitive')
    leg.add(
        'geom',
        name=leg.name + '1_collision',
        size=[0.01, 0.015],
        pos=[.035, 0, -.028],
        euler=(90, 0, 0),
        dclass='foot_primitive')
    leg.add(
        'geom',
        name=leg.name + '2_collision',
        size=[0.008, 0.01],
        pos=[.045, 0, -.03],
        euler=(90, 0, 0),
        dclass='foot_primitive')
    sole = leg.add(
        'site',
        name='sole' + side,
        size=(.025, .03, .008),
        pos=(.026, 0, -.033),
        type='box',
        dclass='sensor')

    sole_sites.append(sole)

  print('Shoulders, front legs.')
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
          pos=-torso_pos - body_pos,
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
            pos=shoulder_pos*side_sign[side])

  # Upper Arms:
  upper_arm = {}
  parent_pos = {}
  humeri = ['humerus_R', 'humerus_L']
  for side in ['_L', '_R']:
    body_pos = shoulder_pos * side_sign[side]
    parent = scapula[side]
    parent_pos[side] = torso_pos + parent.pos
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
            zaxis=(0, 1, 0), pos=wrist_pos*side_sign[side])

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

  print('Tail.')
  # Caudal spine (tail) bodies:
  caudal_bones = ['Ca_' + str(i+1) for i in range(21)]
  parent = pelvis
  parent_pos = bone_position['Pelvis']
  caudal_bodies = []
  caudal_geoms = []
  for bone in caudal_bones:
    bone_pos = bone_position[bone]
    rel_pos = bone_pos-parent_pos
    xyaxes = np.hstack((-rel_pos, (0, 1, 0)))
    xyaxes[1] = 0
    child = parent.add('body', name=bone, pos=rel_pos)
    caudal_bodies.append(child)
    geom = child.add('geom', name=bone, mesh=bone, pos=-bone_pos, dclass='bone')
    caudal_geoms.append(geom)
    parent = child
    parent_pos = bone_pos

  # Reload
  physics = mjcf.Physics.from_mjcf_model(model)

  # Caudal spine joints:
  caudal_axis = collections.OrderedDict()
  caudal_axis['extend'] = np.array((0., 1., 0.))

  scale = np.asarray([bone_size[bone] for bone in caudal_bones]).mean()
  joint_pos = np.array((.3, 0, .26))*scale
  num_dofs = 0
  caudal_joints = []
  caudal_joint_names = []
  parent = pelvic_geoms[0]
  for i, vertebra in enumerate(caudal_bodies):
    while num_dofs < (i+1) * caudal_dofs_per_veterbra:
      dof = num_dofs % 2
      dof_name = list(caudal_axis.keys())[dof]
      dof_axis = caudal_axis[dof_name]
      caudal_joint_names.append(vertebra.name + '_' + dof_name)
      rel_pos = physics.bind(parent).xpos - physics.bind(vertebra).xpos
      twist_dir = rel_pos / np.linalg.norm(rel_pos)
      bend_dir = np.cross(caudal_axis['extend'], twist_dir)
      caudal_axis['bend'] = bend_dir
      joint_pos = twist_dir * physics.bind(caudal_geoms[i]).size[2]

      joint = vertebra.add(
          'joint',
          name=caudal_joint_names[-1],
          dclass='caudal_' + dof_name,
          axis=caudal_axis[dof_name],
          pos=joint_pos)
      caudal_joints.append(joint)
      num_dofs += 1
    parent = vertebra
  vertebra.add('site', name='tail_tip', dclass='sensor', size=(0.005,))

  print('Collision geoms, fixed tendons.')
  physics = mjcf.Physics.from_mjcf_model(model)

  # Add collision geoms that require bind()
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
    if 'Ca_' in geom.name:
      sc = (float(geom.name[3:]) + 1) / 4
      scale = np.array((1.2, sc, 1))
      bound_geom = physics.bind(geom)
      geom.parent.add(
          'geom',
          name=geom.name + '_collision',
          pos=bound_geom.pos,
          size=bound_geom.size * scale,
          quat=bound_geom.quat,
          dclass='collision_primitive')

  for side in ['_L', '_R']:
    # lower leg:
    leg = lower_leg[side]
    leg.add(
        'geom',
        name=leg.name + '_collision',
        pos=physics.bind(leg).ipos * 1.3,
        size=[.02, .1],
        quat=physics.bind(leg).iquat,
        dclass='collision_primitive')
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

  print('Make collision ellipsoids for teeth.')
  visible_bones = upper_teeth+lower_teeth
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

  print('Unify ribcage and jaw meshes.')
  for body in model.find_all('body'):
    body_meshes = [
        geom for geom in body.all_children() if geom.tag == 'geom' and
        hasattr(geom, 'mesh') and geom.mesh is not None
    ]
    if len(body_meshes) > 10:
      mergables = [('torso', 'Ribcage'), ('jaw', 'Jaw'),
                   ('skull', 'MergedSkull')]
      for bodyname, meshname in mergables:
        if body.name == bodyname:
          print('==== Merging ', bodyname)
          for mesh in body_meshes:
            print(mesh.name)
          body.add('inertial',
                   mass=physics.bind(body).mass,
                   pos=physics.bind(body).ipos,
                   quat=physics.bind(body).iquat,
                   diaginertia=physics.bind(body).inertia)

          for mesh in body_meshes:
            if 'eye' not in mesh.name:
              model.find('mesh', mesh.name).remove()
              mesh.remove()
          body.add(
              'geom',
              name=meshname,
              mesh=meshname,
              dclass='bone',
              pos=-bone_position[meshname])

  # Fixed Tendons:
  spinal_joints = collections.OrderedDict()
  spinal_joints['lumbar_'] = lumbar_joints
  spinal_joints['cervical_'] = cervical_joints
  spinal_joints['caudal_'] = caudal_joints
  tendons = []
  for region in spinal_joints.keys():
    for direction in ['extend', 'bend', 'twist']:
      joints = [
          joint for joint in spinal_joints[region] if direction in joint.name
      ]
      if joints:
        tendon = model.tendon.add(
            'fixed', name=region + direction, dclass=joints[0].dclass)
        tendons.append(tendon)
        joint_inertia = physics.bind(joints).M0
        coefs = joint_inertia ** .25
        coefs /= coefs.sum()
        coefs *= len(joints)
        for i, joint in enumerate(joints):
          tendon.add('joint', joint=joint, coef=coefs[i])

  # Actuators:
  all_spinal_joints = [
      joint for region in spinal_joints.values() for joint in region  # pylint: disable=g-complex-comprehension
  ]
  actuated_joints = [
      joint for joint in model.find_all('joint')
      if joint not in all_spinal_joints and joint is not root_joint
  ]
  for tendon in tendons:
    gain = 0.
    for joint in tendon.joint:
      # joint.joint.user = physics.bind(joint.joint).damping
      def_joint = model.default.find('default', joint.joint.dclass)
      j_gain = def_joint.general.gainprm or def_joint.parent.general.gainprm
      gain += j_gain[0] * joint.coef
    gain /= len(tendon.joint)

    model.actuator.add(
        'general', tendon=tendon, name=tendon.name, dclass=tendon.dclass)

  for joint in actuated_joints:
    model.actuator.add(
        'general', joint=joint, name=joint.name, dclass=joint.dclass)

  # make spatial tendons
  if use_tendons:
    print('Muscles, connector sites, spatial tendons.')

    # Put all muscles in worldbody
    muscle_geoms = []
    for muscle in muscle_meshes:
      geom = model.worldbody.add(
          'geom', name=muscle.name, mesh=muscle, type='mesh', dclass='muscle')
      muscle_geoms.append(geom)

    # fix muscle associations using edit and global distance
    muscle_names = [m.name for m in muscle_geoms]
    muscle_pos = physics.bind(muscle_geoms).xpos
    for cc in connectors:
      edist = np.asarray([editdistance.eval(cc[1], m) for m in muscle_names])
      dist = np.asarray([np.linalg.norm(cc[2].pos-p) for p in muscle_pos])
      replace_candidates = np.argsort(edist)[:5]
      combi_dist = edist[replace_candidates] + 10*dist[replace_candidates]
      replacement = muscle_names[replace_candidates[combi_dist.argmin()]]
      if replacement != cc[1] and edist[replace_candidates[
          combi_dist.argmin()]] > 3:
        print('dist %6.2g %50s --> %-50s' %
              (combi_dist.min(), cc[1], replacement))
        cc[1] = replacement

    # Move connectors into bodies
    for c in connectors:
      name = c[2].name
      pos = c[2].pos
      c[2].remove()
      body = model.find('geom', c[0]).parent
      c[2] = body.add(
          'site',
          dclass='connector',
          pos=pos - physics.bind(body).xpos,
          name=name)
      c.append(pos)

  # muscles
  # for i,c in enumerate(connectors):
  #   connectors[i][1] = c[1].replace('(2)','')
  #   connectors[i][1] = c[1].replace('.001','')
    counts = collections.Counter([con[1] for con in connectors])
    muscles = counts.keys()
    singleton_muscles = [k for k in muscles if counts[k] == 1]
    # dist = np.zeros(len(muscles), dtype='int')
    # for st in singleton_muscles:
    #   for i,m in enumerate(muscles):
    #     dist[i] = int(editdistance.eval(st, m))
    #   dist[dist==0] = 99
    #   st2 = muscles[dist.argmin()]
    #   if not ('_L' in st and '_R' in st2 or
    #           '_R' in st and '_L' in st2):
    #     print('replacing '+st+' with '+st2)

    # remove singleton_muscles
    connectors = [c for c in connectors if c[1] not in singleton_muscles]
    counts = collections.Counter([con[1] for con in connectors])
    muscles = counts.keys()
    singleton_muscles = [k for k in muscles if counts[k] == 1]
    assert not singleton_muscles

    for muscle in muscles:
      sites = [(c[2], c[3]) for c in connectors if c[1] == muscle]
      nsite = len(sites)
      dist = np.zeros((nsite, nsite))
      for i in range(nsite):
        for k in range(i):
          if i != k:
            dist[k, i] = np.linalg.norm(sites[i][1] - sites[k][1])
      tree = mst(dist).toarray()
      counter = 0
      rgba = np.random.rand(4)
      rgba[3] = 1.
      for i, row in enumerate(tree):
        for j, v in enumerate(row):
          if v > 0:
            tendon = model.tendon.add(
                'spatial', name=muscle + '_' + str(counter), rgba=rgba)
            tendon.add('site', site=sites[i][0])
            tendon.add('site', site=sites[j][0])
            counter += 1

  print('Excluding contacts.')
  physics = mjcf.Physics.from_mjcf_model(model)
  excluded_pairs = []
  for c in physics.data.contact:
    body1 = physics.model.id2name(physics.model.geom_bodyid[c.geom1], 'body')
    body2 = physics.model.id2name(physics.model.geom_bodyid[c.geom2], 'body')
    pair = body1+':'+body2
    if pair not in excluded_pairs:
      excluded_pairs.append(pair)
      model.contact.add('exclude', name=pair, body1=body1, body2=body2)
  # manual exclusions
  model.contact.add('exclude', name='C_1:jaw', body1=c_1, body2=jaw)
  model.contact.add(
      'exclude', name='torso:lower_arm_L', body1=torso, body2='lower_arm_L')
  model.contact.add(
      'exclude', name='torso:lower_arm_R', body1=torso, body2='lower_arm_R')
  model.contact.add(
      'exclude', name='C_4:scapula_R', body1='C_4', body2='scapula_R')
  model.contact.add(
      'exclude', name='C_4:scapula_L', body1='C_4', body2='scapula_L')
  model.contact.add(
      'exclude', name='C_5:upper_arm_R', body1='C_5', body2='upper_arm_R')
  model.contact.add(
      'exclude', name='C_5:upper_arm_L', body1='C_5', body2='upper_arm_L')
  model.contact.add(
      'exclude', name='C_6:upper_arm_R', body1='C_6', body2='upper_arm_R')
  model.contact.add(
      'exclude', name='C_6:upper_arm_L', body1='C_6', body2='upper_arm_L')
  model.contact.add(
      'exclude', name='C_7:upper_arm_R', body1='C_7', body2='upper_arm_R')
  model.contact.add(
      'exclude', name='C_7:upper_arm_L', body1='C_7', body2='upper_arm_L')
  model.contact.add(
      'exclude',
      name='upper_leg_L:upper_leg_R',
      body1='upper_leg_L',
      body2='upper_leg_R')
  for side in ['_L', '_R']:
    model.contact.add(
        'exclude',
        name='lower_leg' + side + ':pelvis',
        body1='lower_leg' + side,
        body2='pelvis')
    model.contact.add(
        'exclude',
        name='upper_leg' + side + ':foot' + side,
        body1='upper_leg' + side,
        body2='foot' + side)

  if make_skin:
    print('Making Skin.')
    # Add skin mesh:
    skinmesh = model.worldbody.add(
        'geom',
        name='skinmesh',
        mesh='skin_msh',
        type='mesh',
        contype=0,
        conaffinity=0,
        rgba=[1, .5, .5, .5],
        group=1,
        euler=(0, 0, 90))
    physics = mjcf.Physics.from_mjcf_model(model)

    # Get skinmesh vertices in global coordinates
    vertadr = physics.named.model.mesh_vertadr['skin_msh']
    vertnum = physics.named.model.mesh_vertnum['skin_msh']
    skin_vertices = physics.model.mesh_vert[vertadr:vertadr + vertnum, :]
    skin_vertices = skin_vertices.dot(
        physics.named.data.geom_xmat['skinmesh'].reshape(3, 3).T)
    skin_vertices += physics.named.data.geom_xpos['skinmesh']
    skin_normals = physics.model.mesh_normal[vertadr:vertadr + vertnum, :]
    skin_normals = skin_normals.dot(
        physics.named.data.geom_xmat['skinmesh'].reshape(3, 3).T)
    skin_normals += physics.named.data.geom_xpos['skinmesh']

    # Get skinmesh faces
    faceadr = physics.named.model.mesh_faceadr['skin_msh']
    facenum = physics.named.model.mesh_facenum['skin_msh']
    skin_faces = physics.model.mesh_face[faceadr:faceadr + facenum, :]

    # Make skin
    skin = model.asset.add(
        'skin',
        name='skin',
        vertex=skin_vertices.ravel(),
        face=skin_faces.ravel())

    # Functions for capsule vertices
    numslices = 10
    numstacks = 10
    numquads = 8
    def hemisphere(radius):
      positions = []
      for az in np.linspace(0, 2*np.pi, numslices, False):
        for el in np.linspace(0, np.pi, numstacks, False):
          pos = np.asarray(
              [np.cos(el) * np.cos(az),
               np.cos(el) * np.sin(az),
               np.sin(el)])
          positions.append(pos)
      return radius*np.asarray(positions)

    def cylinder(radius, height):
      positions = []
      for az in np.linspace(0, 2*np.pi, numslices, False):
        for el in np.linspace(-1, 1, numstacks):
          pos = np.asarray([radius*np.cos(az), radius*np.sin(az), height*el])
          positions.append(pos)
      return np.asarray(positions)

    def capsule(radius, height):
      hp = hemisphere(radius)
      cy = cylinder(radius, height)
      offset = np.array((0, 0, height))
      return np.unique(np.vstack((cy, hp + offset, -hp - offset)), axis=0)

    def ellipsoid(size):
      hp = hemisphere(1)
      sphere = np.unique(np.vstack((hp, -hp)), axis=0)
      return sphere*size

    def box(sx, sy, sz):
      positions = []
      for x in np.linspace(-sx, sx, numquads+1):
        for y in np.linspace(-sy, sy, numquads+1):
          for z in np.linspace(-sz, sz, numquads+1):
            if abs(x) == sx or abs(y) == sy or abs(z) == sz:
              pos = np.asarray([x, y, z])
              positions.append(pos)
      return np.unique(np.asarray(positions), axis=0)

    # Find smallest distance between
    # each skin vertex and vertices of all meshes in body i
    distance = np.zeros((skin_vertices.shape[0], physics.model.nbody))
    for i in range(1, physics.model.nbody):
      geom_id = np.argwhere(physics.model.geom_bodyid == i).ravel()
      mesh_id = physics.model.geom_dataid[geom_id]
      body_verts = []
      for k, gid in enumerate(geom_id):
        skip = False
        if physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_MESH:
          vertadr = physics.model.mesh_vertadr[mesh_id[k]]
          vertnum = physics.model.mesh_vertnum[mesh_id[k]]
          vertices = physics.model.mesh_vert[vertadr:vertadr+vertnum, :]
        elif physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_CAPSULE:
          radius = physics.model.geom_size[gid, 0]
          height = physics.model.geom_size[gid, 1]
          vertices = capsule(radius, height)
        elif physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_ELLIPSOID:
          vertices = ellipsoid(physics.model.geom_size[gid])
        elif physics.model.geom_type[gid] == enums.mjtGeom.mjGEOM_BOX:
          vertices = box(*physics.model.geom_size[gid])
        else:
          skip = True
        if not skip:
          vertices = vertices.dot(physics.data.geom_xmat[gid].reshape(3, 3).T)
          vertices += physics.data.geom_xpos[gid]
          body_verts.append(vertices)

      body_verts = np.vstack((body_verts))
      # hull = spatial.ConvexHull(body_verts)
      tree = spatial.cKDTree(body_verts)
      distance[:, i], _ = tree.query(skin_vertices)

      # non-KDTree implementation of the above 2 lines:
      # distance[:, i] = np.amin(
      #     spatial.distance.cdist(skin_vertices, body_verts, 'euclidean'),
      #     axis=1)

    # Calculate bone weights from distances
    sigma = .015
    weights = np.exp(-distance[:, 1:]/sigma)
    threshold = .01
    weights /= np.atleast_2d(np.sum(weights, axis=1)).T
    weights[weights < threshold] = 0
    weights /= np.atleast_2d(np.sum(weights, axis=1)).T

    for i in range(1, physics.model.nbody):
      vertweight = weights[weights[:, i - 1] >= threshold, i - 1]
      vertid = np.argwhere(weights[:, i - 1] >= threshold).ravel()
      if vertid.any():
        skin.add(
            'bone',
            body=physics.model.id2name(i, 'body'),
            bindquat=[1, 0, 0, 0],
            bindpos=physics.data.xpos[i, :],
            vertid=vertid,
            vertweight=vertweight)

    # Remove skinmesh
    skinmesh.remove()

    # Convert skin into *.skn file accordong to
    # https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset-skin
    f = open('dog_skin.skn', 'w+b')
    nvert = skin.vertex.size // 3
    f.write(
        struct.pack('4i', nvert, nvert, skin.face.size // 3,
                    physics.model.nbody - 1))
    f.write(struct.pack(str(skin.vertex.size) + 'f', *skin.vertex))
    assert physics.model.mesh_texcoord.shape[0] == physics.bind(
        skin_msh).vertnum
    f.write(
        struct.pack(
            str(2 * nvert) + 'f', *physics.model.mesh_texcoord.flatten()))
    f.write(struct.pack(str(skin.face.size) + 'i', *skin.face))
    for bone in skin.bone:
      name_length = len(bone.body)
      assert name_length <= 40
      f.write(
          struct.pack(str(name_length) + 'c', *[s.encode() for s in bone.body]))
      f.write((40 - name_length) * b'\x00')
      f.write(struct.pack('3f', *bone.bindpos))
      f.write(struct.pack('4f', *bone.bindquat))
      f.write(struct.pack('i', bone.vertid.size))
      f.write(struct.pack(str(bone.vertid.size) + 'i', *bone.vertid))
      f.write(struct.pack(str(bone.vertid.size) + 'f', *bone.vertweight))
    f.close()

    # Remove XML-based skin, add binary skin.
    skin.remove()
    #######  end `if make_skin:` ######

  # Add skin from .skn
  print('Adding Skin.')
  skin_texture = model.asset.add('texture', name='skin',
                                 file='skin_texture.png', type='2d')
  skin_material = model.asset.add('material', name='skin', texture=skin_texture)
  skin = model.asset.add(
      'skin', name='skin', file='dog_skin.skn', material=skin_material)
  skin_msh.remove()

  print('Removing non-essential sites.')
  all_sites = model.find_all('site')
  for site in all_sites:
    if site.dclass is None:
      site.remove()

  # sensors
  model.sensor.add('accelerometer', name='accelerometer', site=head)
  model.sensor.add('velocimeter', name='velocimeter', site=head)
  model.sensor.add('gyro', name='gyro', site=head)
  model.sensor.add('subtreelinvel', name='torso_linvel', body=torso)
  model.sensor.add('subtreeangmom', name='torso_angmom', body=torso)
  for site in palm_sites+sole_sites:
    model.sensor.add('touch', name=site.name, site=site)
  anchors = [site for site in model.find_all('site') if 'anchor' in site.name]
  for site in anchors:
    model.sensor.add('force', name=site.name.replace('_anchor', ''), site=site)

  # Print stuff
  joint_acts = [model.find('actuator', j.name) for j in actuated_joints]
  print('{:20} {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
      'name', 'mass', 'damping', 'stiffness', 'ratio', 'armature'))
  for i, j in enumerate(actuated_joints):
    dmp = physics.bind(j).damping[0]
    mass_eff = physics.bind(j).M0[0]
    dmp = physics.bind(j).damping[0]
    stf = physics.bind(joint_acts[i]).gainprm[0]
    arma = physics.bind(j).armature[0]
    print('{:20} {:10.4} {:10} {:10.4} {:10.4} {:10}'.format(
        j.name, mass_eff, dmp, stf, dmp / (2 * np.sqrt(mass_eff * stf)), arma))

  print('Finalising and saving model.')
  xml_string = model.to_xml_string('float', precision=4, zero_threshold=1e-7)
  root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))

  print('Remove hashes from filenames')
  assets = list(root.find('asset').iter())
  for asset in assets:
    asset_filename = asset.get('file')
    if asset_filename is not None:
      name = asset_filename[:-4]
      extension = asset_filename[-4:]
      asset.set('file', name[:-41] + extension)

  print('Add <compiler meshdir/>, for locally-loadable model')
  compiler = etree.Element(
      'compiler', meshdir=ASSET_RELPATH, texturedir=ASSET_RELPATH)
  root.insert(0, compiler)

  print('Remove class="/"')
  default_elem = root.find('default')
  root.insert(6, default_elem[0])
  root.remove(default_elem)
  xml_string = etree.tostring(root, pretty_print=True)
  xml_string = xml_string.replace(b' class="/"', b'')

  print('Insert spaces between top level elements')
  lines = xml_string.splitlines()
  newlines = []
  for line in lines:
    newlines.append(line)
    if line.startswith(b'  <'):
      if line.startswith(b'  </') or line.endswith(b'/>'):
        newlines.append(b'')
  newlines.append(b'')
  xml_string = b'\n'.join(newlines)

  # Save to file.
  f = open('dog.xml', 'wb')
  f.write(xml_string)
  f.close()

if __name__ == '__main__':
  app.run(main)
