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

"""Make back legs for the dog model."""

from dm_control import mjcf
import numpy as np


def create_back_legs(
    model,
    primary_axis,
    bone_position,
    bones,
    side_sign,
    bone_size,
    pelvic_bones,
    parent,
):
  """Add back legs in the model.

  Args:
    model: model in which we want to add the back legs.
    primary_axis: a dictionary of numpy arrays representing axis of rotation.
    bone_position: a dictionary of bones positions.
    bones: a list of strings with all the names of the bones.
    side_sign: a dictionary with two axis representing the signs of
      translations.
    bone_size: dictionary containing the scale of the geometry.
    pelvic_bones: list of string of the pelvic bones.
    parent: parent object on which we should start attaching new components.

  Returns:
    The tuple `(nails, sole_sites)`.
  """
  pelvis = parent
  # Hip joint sites:
  scale = np.asarray([bone_size[bone] for bone in pelvic_bones]).mean()
  hip_pos = np.array((-0.23, -0.6, -0.16)) * scale
  for side in ['_L', '_R']:
    pelvis.add(
        'site', name='hip' + side, size=[0.011], pos=hip_pos * side_sign[side]
    )

  # Upper legs:
  upper_leg = {}
  femurs = [b for b in bones if 'Fem' in b]
  use_tendons = False
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
          dclass='bone',
      )

    # Hip joints
    for dof in ['_supinate', '_abduct', '_extend']:
      axis = primary_axis[dof].copy()
      if dof != '_extend':
        axis *= 1.0 if side != '_R' else -1.0
      leg.add('joint', name='hip' + side + dof, dclass='hip' + dof, axis=axis)

    # Knee sites
    scale = bone_size['Femoris_L']
    knee_pos = np.array((-0.2, -0.27, -1.45)) * scale
    leg.add(
        'site',
        type='cylinder',
        name='knee' + side,
        size=[0.003, 0.02],
        zaxis=(0, 1, 0),
        pos=knee_pos * side_sign[side],
    )
    pos = np.array((-0.01, -0.02, -0.08)) * side_sign[side]
    euler = [-10 * (1.0 if side == '_R' else -1.0), 20, 0]
    leg.add(
        'geom',
        name=leg.name + '0_collision',
        pos=pos,
        size=[0.04, 0.08],
        euler=euler,
        dclass='collision_primitive',
    )
    pos = np.array((-0.03, 0, -0.05))
    euler = [-10 * (1.0 if side == '_R' else -1.0), 5, 0]
    leg.add(
        'geom',
        name=leg.name + '1_collision',
        pos=pos,
        size=[0.04, 0.04],
        euler=euler,
        dclass='collision_primitive',
    )

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
          dclass='bone',
      )
    # Knee joints
    leg.add('joint', name='knee' + side, dclass='knee', axis=(0, -1, 0))

    # Ankle sites
    scale = bone_size['Tibia_L']
    ankle_pos = np.array((-1.27, 0.04, -0.98)) * scale
    leg.add(
        'site',
        type='cylinder',
        name='ankle' + side,
        size=[0.003, 0.013],
        zaxis=(0, 1, 0),
        pos=ankle_pos * side_sign[side],
    )

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
          pos=-bone_position['Pelvis']
          - upper_leg[side].pos
          - lower_leg[side].pos
          - body_pos,
          dclass='bone',
      )
    # Ankle joints
    leg.add('joint', name='ankle' + side, dclass='ankle', axis=(0, 1, 0))
    pos = np.array((-0.01, -0.005, -0.05)) * side_sign[side]
    leg.add(
        'geom',
        name=leg.name + '_collision',
        size=[0.015, 0.07],
        pos=pos,
        dclass='collision_primitive',
    )

    # Toe sites
    scale = bone_size['Metatarsi_R_2']
    toe_pos = np.array((-0.37, -0.2, -2.95)) * scale
    leg.add(
        'site',
        type='cylinder',
        name='toe' + side,
        size=[0.003, 0.025],
        zaxis=(0, 1, 0),
        pos=toe_pos * side_sign[side],
    )

  # Toes:
  toe_bones = [b for b in bones if 'Phalange' in b]
  toe_geoms = []
  sole_sites = []
  nails = []
  for side in ['_L', '_R']:
    body_pos = toe_pos * side_sign[side]
    foot_anchor = foot[side].add(
        'body', name='foot_anchor' + side, pos=body_pos
    )
    foot_anchor.add(
        'geom',
        name=foot_anchor.name,
        dclass='foot_primitive',
        type='box',
        size=(0.005, 0.005, 0.005),
        contype=0,
        conaffinity=0,
    )
    foot_anchor.add('site', name=foot_anchor.name, dclass='sensor')
    leg = foot_anchor.add('body', name='toe' + side)
    for bone in [b for b in toe_bones if side in b]:
      geom = leg.add(
          'geom',
          name=bone,
          mesh=bone,
          pos=-bone_position['Pelvis']
          - upper_leg[side].pos
          - lower_leg[side].pos
          - foot[side].pos
          - body_pos,
          dclass='bone',
      )
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
        pos=[0.015, 0, -0.02],
        euler=(90, 0, 0),
        dclass='foot_primitive',
    )
    leg.add(
        'geom',
        name=leg.name + '1_collision',
        size=[0.01, 0.015],
        pos=[0.035, 0, -0.028],
        euler=(90, 0, 0),
        dclass='foot_primitive',
    )
    leg.add(
        'geom',
        name=leg.name + '2_collision',
        size=[0.008, 0.01],
        pos=[0.045, 0, -0.03],
        euler=(90, 0, 0),
        dclass='foot_primitive',
    )
    sole = leg.add(
        'site',
        name='sole' + side,
        size=(0.025, 0.03, 0.008),
        pos=(0.026, 0, -0.033),
        type='box',
        dclass='sensor',
    )

    sole_sites.append(sole)

  physics = mjcf.Physics.from_mjcf_model(model)

  for side in ['_L', '_R']:
    # lower leg:
    leg = lower_leg[side]
    leg.add(
        'geom',
        name=leg.name + '_collision',
        pos=physics.bind(leg).ipos * 1.3,
        size=[0.02, 0.1],
        quat=physics.bind(leg).iquat,
        dclass='collision_primitive',
    )

  return nails, sole_sites
