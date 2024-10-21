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

"""Add muscle actuators to the dog model."""

from os import listdir
from os.path import isfile, join

import numpy as np
import pyvista as pv
import trimesh
from pykdtree.kdtree import KDTree
from tqdm import tqdm
from utils import array_to_string, slices2paths

from dm_control import mjcf as mjcf_module
from dm_control.locomotion.walkers.assets.dog_v2 import muscles as muscles_constants
from dm_control.mujoco.wrapper import mjbindings
from dm_control.utils.transformations import quat_to_mat

MUSCLE_LEGS = (
    muscles_constants.EXTENSORS_BACK
    + muscles_constants.EXTENSORS_FRONT
    + muscles_constants.FLEXORS_BACK
    + muscles_constants.FLEXORS_FRONT
)

WRAP_GEOMS_LEGS = (
    "shoulder_L_wrapping",
    "elbow_L_wrapping",
    "wrist_L_wrapping",
    "finger_L_wrapping",
    "hip_L_wrapping",
    "knee_L_wrapping",
    "toe_L_wrapping",
    "shoulder_R_wrapping",
    "elbow_R_wrapping",
    "wrist_R_wrapping",
    "finger_R_wrapping",
    "hip_R_wrapping",
    "knee_R_wrapping",
    "toe_R_wrapping",
)


def get_closest_geom(kd_tree, point, mjcf, mtu):
  """Finds the closest geometric body to a given point using a KD-tree.

  Args:
      kd_tree : A dictionary containing KD-trees for different bones.
      point : The coordinates of the target point.
      mjcf: The MJCF (MuJoCo XML) object.
      mtu: The muscle-tendon unit (MTU) identifier.

  Returns:
      closest_geom_body: The closest geometric body (as a Mujoco body object)
      to the target point.
  """
  dist = np.inf
  closest_geom_body = 0
  for bone, tree in kd_tree.items():
    dist_new, i = tree.query(np.array([point]), k=1)
    geom = mjcf.find("geom", bone)
    body = geom.parent

    if mtu in MUSCLE_LEGS:
      if body.name not in [
          "upper_leg_L",
          "upper_leg_R",
          "lower_leg_L",
          "lower_leg_R",
          "foot_L",
          "foot_R",
          "toe_L",
          "toe_R",
          "scapula_L",
          "scapula_R",
          "upper_arm_L",
          "upper_arm_R",
          "lower_arm_L",
          "lower_arm_R",
          "hand_L",
          "hand_R",
          "finger_L",
          "finger_R",
          "pelvis",
      ]:
        continue
    else:
      if body.name in [
          "upper_leg_L",
          "upper_leg_R",
          "lower_leg_L",
          "lower_leg_R",
          "foot_L",
          "foot_R",
          "toe_L",
          "toe_R",
          "lower_arm_L",
          "lower_arm_R",
          "hand_L",
          "hand_R",
          "finger_L",
          "finger_R",
      ]:
        continue

    if dist_new < dist:
      dist = dist_new
      closest_geom_body = body

  return closest_geom_body


def calculate_transformation(element):
  """Calculates the transformation matrix from the root to the given element.

  This function iterates through the parent elements of the given element and
  calculates the transformation matrices from the root until reaching the
  given element. It considers the position and orientation (quaternion)
  attributes of each element to construct the transformation matrices.
  The final transformation matrix represents the cumulative transformation
  from the root to the given element.

  Args:
      element: The MuJoCo element for which the transformation matrix
        is calculated.

  Returns:
      T: The transformation matrix from the root to the given element.
  """
  # Calculate all transformation matrices from root until this element
  all_transformations = []

  while type(element) == mjcf_module.element._ElementImpl:
    if element.pos is not None:
      pos = np.array(element.pos, dtype=float)
    else:
      pos = np.zeros(3)

    if element.quat is not None:
      rot = quat_to_mat(element.get("quat"))
    else:
      rot = np.eye(3)

    all_transformations.append(
        np.vstack(
            (np.hstack((rot, pos.reshape([-1, 1]))), np.array([0, 0, 0, 1])))
    )

    element = element.parent

  # Apply all transformations
  transform = np.eye(4)
  for transformation in reversed(all_transformations):
    transform = np.matmul(transform, transformation)

  return transform


def add_muscles(
    model, scale_multiplier, muscle_dynamics, asset_dir, lengthrange_from_joints
):
  """Add muscles to a Mujoco (MJCF) model.

  This function adds muscles to a Mujoco model,
  creating MTUs (muscle-tendon units) and defining
  their properties.

  Args:
      model (MJCFModel): The input Mujoco model to which muscles will be added.
      scale_multiplier (float): A scaling factor for muscle forces (0 to disable anatomical scaling).
      muscle_dynamics (str): Muscle dynamics model, either 'Millard', 'Sigmoid', or 'General'.
      asset_dir (str): The directory path containing muscle and bone assets.
      lengthrange_from_joints (bool): If True, compute length ranges from joint limits.

  Returns:
      None

  Note:
  - The function modifies the input `model` in place by adding muscles and related properties.
  - The specific muscles added depend on the provided `muscle_dynamics` and other parameters.

  Raises:
      NameError: If an unsupported `muscle_dynamics` value is provided.
  """
  physics = mjcf_module.Physics.from_mjcf_model(model)
  muscle_meshes_path = join(asset_dir, "muscles/")
  bones_path = asset_dir

  bones = [
      f for f in listdir(bones_path) if isfile(join(bones_path, f)) and f[0] != "."
  ]

  bones_kd_trees = {}
  bones_meshes = {}
  for bone in bones:
    if (
        "BONE" in bone
        and "simpl" not in bone
        and "Lingual" not in bone
        and "cartilage" not in bone
    ):
      bone_mesh = trimesh.load_mesh(bones_path + "/" + bone, process=False)
      bones_kd_trees[bone[4:-4]] = KDTree(bone_mesh.vertices)
      bones_meshes[bone[4:-4]] = bone_mesh

  muscle = model.default.default["muscle"]
  muscle.tendon.rgba = [0.5, 0, 0, 1]

  length_range = model.compiler.lengthrange
  length_range.inttotal = 5
  length_range.accel = 50
  length_range.interval = 2
  length_range.timestep = 0.001
  length_range.timeconst = 1
  length_range.tolrange = 0.05

  muscles = (
      muscles_constants.EXTENSORS_FRONT
      + muscles_constants.FLEXORS_FRONT
      + muscles_constants.EXTENSORS_BACK
      + muscles_constants.FLEXORS_BACK
      + muscles_constants.TORSO
      + muscles_constants.NECK
      + muscles_constants.TAIL
  )

  used_muscles = []
  volumes = []
  spatials = []
  # create MTUs
  for mtu in muscles:
    # Load stl file of site
    m = trimesh.load_mesh(muscle_meshes_path + mtu + ".stl")
    pv_mesh = pv.read(muscle_meshes_path + mtu + ".stl")
    volumes.append(pv_mesh.volume)
    # check along which axis to slice
    if mtu not in muscles_constants.LATERAL:
      plane_normal = [0, 0, -1]
      start = np.argmax(m.vertices[:, 2])
      end = np.argmin(m.vertices[:, 2])
    else:
      plane_normal = [1, 0, 0]
      start = np.argmin(m.vertices[:, 0])
      end = np.argmax(m.vertices[:, 0])

    muscle_length = np.linalg.norm(m.vertices[start] - m.vertices[end])

    if muscle_length < 0.1:
      print("muscle too short", mtu)
      continue

    heights = np.linspace(start=-0.5, stop=muscle_length,
                          num=1000, dtype=float)

    slices = m.section_multiplane(
        plane_origin=m.vertices[start], plane_normal=plane_normal, heights=heights
    )

    paths = slices2paths(mtu, slices, muscle_length)
    spatial = model.tendon.add(
        "spatial", name=f"{mtu}_tendon", dclass="muscle")
    spatials.append(spatial)
    used_muscles.append(mtu)

    counter = 0  # used for site naming
    prev_body = None
    prev_point = None

    for idx in range(len(paths)):
      point = paths[idx][0]
      closest_geom_body = get_closest_geom(bones_kd_trees, point, model, mtu)

      # Add site to tendon
      min_dist = 0.05
      max_dist = 0.11
      if mtu in muscles_constants.TAIL:
        max_dist = 0.05
      elif mtu in muscles_constants.NECK:
        min_dist = 0.11
        max_dist = 0.15

      d = 0
      if prev_point is not None:
        d = np.linalg.norm(prev_point - point)

      # we add a site to the tendon path if:
      # 1. it's the last or first site
      # 2. the geometry we are attaching it to is different from the previous
      #   one, and we have passed the min distance between sites
      # 3. we passed the max dist between sites
      if (
          (closest_geom_body != prev_body and d > min_dist)
          or idx == len(paths) - 1
          or idx == 0
          or d > max_dist
      ):
        site_matrix = np.eye(4)
        site_matrix[:3, 3] = point
        body_matrix = calculate_transformation(closest_geom_body)
        site_matrix = np.matmul(np.linalg.inv(body_matrix), site_matrix)

        site_name = mtu + "_" + str(counter)
        # Create the site
        closest_geom_body.add(
            "site",
            name=site_name,
            pos=array_to_string(site_matrix[:3, 3]),
            group="4",
            dclass="connector",
        )

        counter += 1
        spatial.add("site", site=site_name)

        prev_point = point
        prev_body = closest_geom_body

        # check for wrapping geoms
        if idx != len(paths) - 1:
          geoms = physics.named.data.geom_xpos.axes.row.names
          dist = np.inf
          closest_g = None
          closest_geom_pos = None
          for g_name in geoms:
            if "wrapping" in g_name:
              pos = physics.named.data.geom_xpos[g_name]
              new_dist = np.linalg.norm(pos - point)
              if (
                  (
                      mtu in muscles_constants.LATERAL
                      or mtu in muscles_constants.NECK
                  )
                  and g_name not in WRAP_GEOMS_LEGS
                  and pos[2] < point[2]
                  and new_dist < dist
              ):
                closest_g = g_name
                dist = new_dist
                closest_geom_pos = pos
              elif (
                  (
                      mtu in muscles_constants.FLEXORS_BACK
                      or mtu in muscles_constants.FLEXORS_FRONT
                      or mtu in muscles_constants.EXTENSORS_FRONT
                      or mtu in muscles_constants.EXTENSORS_BACK
                  )
                  and pos[2] < point[2]
                  and dist > 0.01
                  and new_dist < dist
                  and g_name in WRAP_GEOMS_LEGS
              ):
                # in the legs we are interested only in wrapping
                # geometries that are lower than the site we
                # are adding
                closest_g = g_name
                dist = new_dist
                closest_geom_pos = pos

          if dist < 0.1:
            if (
                mtu in muscles_constants.FLEXORS_BACK
                or mtu in muscles_constants.FLEXORS_FRONT
            ) and closest_geom_pos[2] < point[2]:
              spatial.add(
                  "geom", geom=closest_g, sidesite=closest_g + "_backward"
              )
            elif (
                mtu in muscles_constants.EXTENSORS_BACK
                or mtu in muscles_constants.EXTENSORS_FRONT
            ) and closest_geom_pos[2] < point[2]:
              spatial.add(
                  "geom", geom=closest_g, sidesite=closest_g + "_forward"
              )
            elif mtu in muscles_constants.NECK:
              pos1 = physics.named.data.site_xpos[closest_g + "_forward"]
              d1 = np.linalg.norm(pos1 - point)
              pos2 = physics.named.data.site_xpos[closest_g + "_backward"]
              d2 = np.linalg.norm(pos2 - point)
              if d1 < d2:
                spatial.add(
                    "geom",
                    geom=closest_g,
                    sidesite=closest_g + "_forward",
                )
              else:
                spatial.add(
                    "geom",
                    geom=closest_g,
                    sidesite=closest_g + "_backward",
                )
            elif mtu in muscles_constants.TORSO and dist < 0.05:
              spatial.add(
                  "geom", geom=closest_g, sidesite=closest_g + "_forward"
              )

    print("Added tendon {}".format(spatial.name))

  physics = mjcf_module.Physics.from_mjcf_model(model)
  # Compute length ranges:
  # This computation of lenght ranges differs from the computation
  # provided in MuJoCo. MuJoCo contracts the muscles
  # individually, and records it's minimum and maximum length. However this
  # method relies on the assumption that the muscles dynamics and parameters
  # are perfectly set up. This is not the case, so the method below is used.
  # This method randomly samples the joint limits and records the minimum and
  # maximum length of the muscles. Sampling is necessary because some muscles
  # cross multiple joints (Biarticular muscles). 
  if lengthrange_from_joints:
    vector_min = np.ones(physics.model.ntendon) * np.inf
    vector_max = np.ones(physics.model.ntendon) * -np.inf

    for _ in tqdm(range(500000)):
      hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
      slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE

      for joint_id in range(physics.model.njnt):
        joint_name = physics.model.id2name(joint_id, "joint")
        joint_type = physics.model.jnt_type[joint_id]
        is_limited = physics.model.jnt_limited[joint_id]
        range_min, range_max = physics.model.jnt_range[joint_id]

        if is_limited and (joint_type == hinge or joint_type == slide):
          physics.named.data.qpos[joint_name] = np.random.uniform(
              range_min, range_max
          )
        else:
          continue

      physics.forward()
      vector_min = np.minimum(vector_min, physics.data.ten_length[:])
      vector_max = np.maximum(vector_max, physics.data.ten_length[:])
      physics.reset()

  for tend_idx, spatial in enumerate(spatials):
    if muscle_dynamics in ["Millard", "Sigmoid"]:
      muscle = model.actuator.add(
          "general",
          tendon=spatial,
          name=spatial.name,
          dclass=spatial.dclass,
          dyntype="muscle",
          gaintype="muscle",
          biastype="muscle",
      )
      prms = [0] * 10
      prms[0] = 0.01
      prms[1] = 0.04
      if muscle_dynamics == "Millard":
        muscle.dynprm = prms
      else:
        prms[2] = 2.0
        muscle.dynprm = prms

      if spatial.name[: -len("_tendon")] in muscles_constants.NECK:
        scale = 1500
      else:
        scale = 8000

      # range(2), force, scale, lmin, lmax, vmax, fpmax, fvmax
      gainprm = [0.75, 1.05, -1, scale, 0.5, 1.6, 1.5, 1.3, 1.2, 0]
      muscle.gainprm = gainprm
      muscle.biasprm = gainprm
      muscle.ctrllimited = True
      muscle.ctrlrange = [0.0, 1.0]
      if lengthrange_from_joints:
        muscle.lengthrange = [vector_min[tend_idx], vector_max[tend_idx]]
    elif muscle_dynamics == "General":
      muscle = model.actuator.add(
          "general", tendon=spatial, name=spatial.name, dclass=spatial.dclass
      )
      muscle.ctrllimited = True
      muscle.ctrlrange = [0.0, 1.0]
      muscle.gaintype = "affine"
      muscle.dyntype = "muscle"
      muscle.dynprm = [0.01, 0.04, 0.001]
      muscle.gainprm = [-200, -50, -10]
      if lengthrange_from_joints:
        muscle.lengthrange = [vector_min[tend_idx], vector_max[tend_idx]]
    else:
      raise NameError(muscle_dynamics)

  everything_ok = False
  while not everything_ok:
    try:
      physics = mjcf_module.Physics.from_mjcf_model(model)
      everything_ok = True

      # assign length ranges and forces
      if muscle_dynamics in ["Millard", "Sigmoid"]:
        lr = physics.named.model.actuator_lengthrange
        forces = None

        # compute anatomical forces if we provide a scale_multiplier > 0
        if scale_multiplier > 0:
          volumes = np.array(volumes)
          lm = []
          for mtu in used_muscles:
            mtu_name = mtu + "_tendon"
            L0 = (lr[mtu_name, 1] - lr[mtu_name, 0]) / (1.05 - 0.75)
            LM = physics.named.model.actuator_length0[mtu_name] - (
                lr[mtu_name, 0] - 0.75 * L0
            )
            lm.append(LM)
          lm = np.array(lm)
          forces = (volumes / lm) * scale_multiplier

        actuators = model.find_all("actuator")
        # assign forces and length ranges
        for idx, muscle in enumerate(actuators):
          muscle.lengthrange = lr[idx]
          if forces is not None:
            muscle.gainprm[2] = forces[idx]

    except Exception as inst:
      print(inst)
      s = str(inst)
      s = s.split()
      i = s.index("actuator")
      muscle_index = s[i + 1]
      if ":" in muscle_index:
        muscle_index = muscle_index[:-1]

      muscle_index = int(muscle_index)
      del model.actuator.general[muscle_index]
