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

"""Make tail for the dog model."""

import collections

import numpy as np

from dm_control import mjcf


def create_tail(caudal_dofs_per_vertebra, bone_size, model, bone_position, parent):
  """Add tail in the dog model.

  Args:
    caudal_dofs_per_vertebra: a number that the determines how many dofs are
      going to be used between each pair of caudal vetebrae.
    bone_size: dictionary containing the scale of the geometry.
    model: model in which we want to add the tail.
    bone_position: a dictionary of bones positions.
    parent: parent object on which we should start attaching new components.

  Returns:
    A list of caudal joints.
  """
  # Caudal spine (tail) bodies:
  caudal_bones = ["Ca_" + str(i + 1) for i in range(21)]
  parent_pos = bone_position["Pelvis"]
  caudal_bodies = []
  caudal_geoms = []
  for bone in caudal_bones:
    bone_pos = bone_position[bone]
    rel_pos = bone_pos - parent_pos
    xyaxes = np.hstack((-rel_pos, (0, 1, 0)))
    xyaxes[1] = 0
    child = parent.add("body", name=bone, pos=rel_pos)
    caudal_bodies.append(child)
    geom = child.add("geom", name=bone, mesh=bone,
                     pos=-bone_pos, dclass="bone")
    caudal_geoms.append(geom)
    parent = child
    parent_pos = bone_pos

  # Reload
  physics = mjcf.Physics.from_mjcf_model(model)

  # Caudal spine joints:
  caudal_axis = collections.OrderedDict()
  caudal_axis["extend"] = np.array((0.0, 1.0, 0.0))

  scale = np.asarray([bone_size[bone] for bone in caudal_bones]).mean()
  joint_pos = np.array((0.3, 0, 0.26)) * scale
  num_dofs = 0
  caudal_joints = []
  caudal_joint_names = []
  parent = model.find("geom", "Sacrum")
  for i, vertebra in enumerate(caudal_bodies):
    while num_dofs < (i + 1) * caudal_dofs_per_vertebra:
      dof = num_dofs % 2
      dof_name = list(caudal_axis.keys())[dof]
      caudal_joint_names.append(vertebra.name + "_" + dof_name)
      rel_pos = physics.bind(parent).xpos - physics.bind(vertebra).xpos
      twist_dir = rel_pos / np.linalg.norm(rel_pos)
      bend_dir = np.cross(caudal_axis["extend"], twist_dir)
      caudal_axis["bend"] = bend_dir
      joint_pos = twist_dir * physics.bind(caudal_geoms[i]).size[2]

      joint = vertebra.add(
          "joint",
          name=caudal_joint_names[-1],
          dclass="caudal_" + dof_name,
          axis=caudal_axis[dof_name],
          pos=joint_pos,
      )
      caudal_joints.append(joint)
      num_dofs += 1
    parent = vertebra

  parent.add("site", name="tail_tip", dclass="sensor", size=(0.005,))

  physics = mjcf.Physics.from_mjcf_model(model)
  all_geoms = model.find_all("geom")

  for geom in all_geoms:
    if "Ca_" in geom.name:
      sc = (float(geom.name[3:]) + 1) / 4
      scale = np.array((1.2, sc, 1))
      bound_geom = physics.bind(geom)
      geom.parent.add(
          "geom",
          name=geom.name + "_collision",
          pos=bound_geom.pos,
          size=bound_geom.size * scale,
          quat=bound_geom.quat,
          dclass="collision_primitive",
      )
  return caudal_joints
