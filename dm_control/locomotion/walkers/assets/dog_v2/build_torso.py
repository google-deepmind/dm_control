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

"""Make torso for the dog model."""

import collections

import numpy as np

from dm_control import mjcf


def create_torso(
    model, bones, bone_position, lumbar_dofs_per_vertebra, side_sign, parent, composer
):
  """Add torso in the dog model.

  Args:
    model: model in which we want to add the torso.
    bones: a list of strings with all the names of the bones.
    bone_position: a dictionary of bones positions.
    lumbar_dofs_per_vertebra: a number that the determines how many dofs are
      going to be used between each pair of lumbar vetebrae.
    side_sign: a dictionary with two axis representing the signs of
      translations.
    parent: parent object on which we should start attaching new components.
    composer: boolean to determine if a model used by the composer is being created.
  Returns:
    The tuple `(pelvic_bones, lumbar_joints)`.
  """
  # Lumbar Spine
  def_lumbar_extend = model.default.find("default", "lumbar_extend")
  def_lumbar_bend = model.default.find("default", "lumbar_bend")
  def_lumbar_twist = model.default.find("default", "lumbar_twist")
  lumbar_defaults = {
      "extend": def_lumbar_extend,
      "bend": def_lumbar_bend,
      "twist": def_lumbar_twist,
  }

  thoracic_spine = [m for m in bones if "T_" in m]
  ribs = [m for m in bones if "Rib" in m and "cage" not in m]
  sternum = [m for m in bones if "Sternum" in m]
  torso_bones = thoracic_spine + ribs + sternum  # + ['Xiphoid_cartilage']
  torso = parent.add("body", name="torso")

  if not composer:
    torso.add("freejoint", name="root")

  torso.add("site", name="root", size=(0.01,), rgba=[0, 1, 0, 1])
  torso.add("light", name="light", mode="trackcom", pos=[0, 0, 3])
  torso.add(
      "camera",
      name="y-axis",
      mode="trackcom",
      pos=[0, -1.5, 0.8],
      xyaxes=[1, 0, 0, 0, 0.6, 1],
  )
  torso.add(
      "camera",
      name="x-axis",
      mode="trackcom",
      pos=[2, 0, 0.5],
      xyaxes=[0, 1, 0, -0.3, 0, 1],
  )
  torso_geoms = []
  for bone in torso_bones:
    torso_geoms.append(
        torso.add("geom", name=bone, mesh=bone, dclass="light_bone"))

  # Reload, get CoM position, set pos
  physics = mjcf.Physics.from_mjcf_model(model)
  torso_pos = np.array(physics.bind(model.find("body", "torso")).xipos)
  torso.pos = torso_pos
  for geom in torso_geoms:
    geom.pos = -torso_pos
  # Collision primitive for torso
  torso.add(
      "geom",
      name="collision_torso",
      dclass="nonself_collision_primitive",
      type="ellipsoid",
      pos=[0, 0, 0],
      size=[0.2, 0.09, 0.11],
      euler=[0, 10, 0],
      density=200,
  )

  # Lumbar spine bodies:
  lumbar_bones = ["L_1", "L_2", "L_3", "L_4", "L_5", "L_6", "L_7"]
  parent = torso
  parent_pos = torso_pos
  lumbar_bodies = []
  lumbar_geoms = []
  for i, bone in enumerate(lumbar_bones):
    bone_pos = bone_position[bone]
    child = parent.add("body", name=bone, pos=bone_pos - parent_pos)
    lumbar_bodies.append(child)
    geom = child.add("geom", name=bone, mesh=bone,
                     pos=-bone_pos, dclass="bone")
    child.add(
        "geom",
        name=bone + "_collision",
        type="sphere",
        size=[0.05],
        pos=[0, 0, -0.02],
        dclass="nonself_collision_primitive",
    )
    lumbar_geoms.append(geom)
    parent = child
    parent_pos = bone_pos
  l_7 = parent

  # Lumbar spine joints:
  lumbar_axis = collections.OrderedDict()
  lumbar_axis["extend"] = np.array((0.0, 1.0, 0.0))
  lumbar_axis["bend"] = np.array((0.0, 0.0, 1.0))
  lumbar_axis["twist"] = np.array((1.0, 0.0, 0))

  num_dofs = 0
  lumbar_joints = []
  lumbar_joint_names = []
  for i, vertebra in enumerate(lumbar_bodies):
    while num_dofs < (i + 1) * lumbar_dofs_per_vertebra:
      dof = num_dofs % 3
      dof_name = list(lumbar_axis.keys())[dof]
      dof_axis = lumbar_axis[dof_name]
      lumbar_joint_names.append(vertebra.name + "_" + dof_name)
      joint = vertebra.add(
          "joint",
          name=lumbar_joint_names[-1],
          dclass="lumbar_" + dof_name,
          axis=dof_axis,
      )
      lumbar_joints.append(joint)
      num_dofs += 1

  # Scale joint defaults relative to 3 lumbar_dofs_per_veterbra
  for dof in lumbar_axis.keys():
    axis_scale = 7.0 / \
        [dof in joint for joint in lumbar_joint_names].count(True)
    lumbar_defaults[dof].joint.range *= axis_scale

  # Pelvis:
  pelvis = l_7.add(
      "body", name="pelvis", pos=bone_position["Pelvis"] - bone_position["L_7"]
  )
  pelvic_bones = ["Sacrum", "Pelvis"]
  pelvic_geoms = []
  for bone in pelvic_bones:
    geom = pelvis.add(
        "geom",
        name=bone,
        mesh=bone,
        pos=-bone_position["Pelvis"],
        dclass="bone",
    )
    pelvic_geoms.append(geom)
  # Collision primitives for pelvis
  for side in ["_L", "_R"]:
    pos = np.array((0.01, -0.02, -0.01)) * side_sign[side]
    pelvis.add(
        "geom",
        name="collision_pelvis" + side,
        pos=pos,
        size=[0.05, 0.05, 0],
        euler=[0, 70, 0],
        dclass="nonself_collision_primitive",
    )

  return pelvic_bones, lumbar_joints
