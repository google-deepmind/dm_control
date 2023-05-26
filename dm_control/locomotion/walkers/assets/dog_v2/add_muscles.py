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

import csv
import os
from os import listdir
from os.path import isfile, join
import sys

from copy import deepcopy

import numpy as np

from lxml import etree

from scipy.spatial.transform import Rotation as Rot
from pykdtree.kdtree import KDTree
import trimesh

import pyvista as pv

from dm_control import mjcf as mjcf_module
from dm_control import mujoco
from dm_control import suite
from dm_control import viewer
from dm_control.mujoco import math
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper import util
from dm_control.rl import control
from dm_control.suite import common
from dm_control.utils import io as resources
from dm_control.utils import xml_tools

from muscles import extensors_back, extensors_front, flexors_back, \
  flexors_front, lateral, neck, tail, to_skip, torso

from utils import array_to_string, export, get_model_and_assets, slices2paths

muscle_legs = extensors_back + extensors_front + \
              flexors_back + flexors_front

wrap_geoms_legs = ['shoulder_L_wrapping', 'elbow_L_wrapping',
                   'wrist_L_wrapping', 'finger_L_wrapping', 'hip_L_wrapping',
                   'knee_L_wrapping', 'toe_L_wrapping',

                   'shoulder_R_wrapping', 'elbow_R_wrapping',
                   'wrist_R_wrapping', 'finger_R_wrapping', 'hip_R_wrapping',
                   'knee_R_wrapping', 'toe_R_wrapping']


def getClosestGeom(kd_tree, point, mjcf, mtu):
  """
    Finds the closest geometric body to a given point using a KD-tree.

    Args:
        kd_tree (dict): A dictionary containing KD-trees for different bones.
        point (array-like): The coordinates of the target point.
        mjcf: The MJCF (MuJoCo XML) object.
        mtu: The muscle-tendon unit (MTU) identifier.

    Returns:
        closest_geom_body: The closest geometric body (as a Mujoco body object)
        to the target point.
  """
  dist = 100000000000
  closest_geom_body = 0
  for bone, tree in kd_tree.items():
    dist_new, i = tree.query(np.array([point]), k=1)
    geom = mjcf.find("geom", bone)
    body = geom.parent

    if mtu in muscle_legs:
      if body.name not in ["upper_leg_L",
                           "upper_leg_R", "lower_leg_L", "lower_leg_R",
                           "foot_L", "foot_R", "toe_L", "toe_R",
                           "scapula_L", "scapula_R", "upper_arm_L",
                           "upper_arm_R", "lower_arm_L", "lower_arm_R",
                           "hand_L", "hand_R", "finger_L", "finger_R",
                           "pelvis"]:
        continue
    else:
      if body.name in ["upper_leg_L",
                       "upper_leg_R", "lower_leg_L", "lower_leg_R",
                       "foot_L", "foot_R", "toe_L", "toe_R",
                       "lower_arm_L", "lower_arm_R",
                       "hand_L", "hand_R", "finger_L", "finger_R"]:
        continue

    if dist_new < dist:
      dist = dist_new
      closest_geom_body = body

  return closest_geom_body


def calculate_transformation(element):
  """
    Calculates the transformation matrix from the root to the given element.

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
      rot = Rot.from_quat(element.quat).as_matrix()
    else:
      rot = Rot.identity().as_matrix()

    all_transformations.append(
      np.vstack(
        (np.hstack((rot, pos.reshape([-1, 1]))), np.array([0, 0, 0, 1])))
    )

    element = element.parent

  # Apply all transformations
  T = np.eye(4)
  for transformation in reversed(all_transformations):
    T = np.matmul(T, transformation)

  return T


def add_muscles(model, scale_multiplier, muscle_dynamics, asset_dir):
  physics = mjcf_module.Physics.from_mjcf_model(model)

  mjcf = model

  muscle_meshes_path = asset_dir + '/muscles/'
  bones_path = asset_dir

  bones = [f for f in listdir(bones_path)
           if isfile(join(bones_path, f)) and f[0] != '.']

  bones_kd_trees = {}
  bones_meshes = {}
  for bone in bones:
    if "BONE" in bone and "simpl" not in bone and "Lingual" not in bone:
      bone_mesh = trimesh.load_mesh(bones_path + bone, process=False)

      bones_kd_trees[bone[4:-4]] = KDTree(bone_mesh.vertices)
      bones_meshes[bone[4:-4]] = bone_mesh

  muscle = mjcf.default.default["muscle"]

  flexors = muscle.add("default", dclass="flexors")
  flexors.tendon.rgba = [1, 1, 0, 1]

  extensors = muscle.add("default", dclass="extensors")
  extensors.tendon.rgba = [1, 0.6, 0, 1]

  length_range = mjcf.compiler.lengthrange
  length_range.inttotal = 50
  length_range.accel = 20
  length_range.interval = 2
  length_range.timestep = 0.01

  mjcf.option.timestep = 0.005

  muscles = extensors_front + flexors_front + \
            extensors_back + flexors_back + torso + neck + tail

  used_muscles = []
  volumes = []
  cross_sections = []
  # create MTUs
  for mtu in muscles:
    # Load stl file of site
    m = trimesh.load_mesh(muscle_meshes_path + mtu + '.stl')
    pv_mesh = pv.read(muscle_meshes_path + mtu + '.stl')
    volumes.append(pv_mesh.volume)
    # check along which axis to slice
    if mtu not in lateral:
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

    heights = np.linspace(start=-0.5,
                          stop=muscle_length,
                          num=1000, dtype=float)

    slices = m.section_multiplane(plane_origin=m.vertices[start],
                                  plane_normal=plane_normal, heights=heights)

    paths, area = slices2paths(mtu, slices, muscle_length)
    cross_sections.append(area)

    spatial = mjcf.tendon.add('spatial', name=f'{mtu}_tendon')
    used_muscles.append(mtu)

    # print("adding spatial", spatial.name)
    if mtu in flexors_back or mtu in flexors_front:
      spatial.dclass = "flexors"
    elif mtu in extensors_back or mtu in extensors_front:
      spatial.dclass = "extensors"

    counter = 0  # used for site naming
    prev_body = None
    prev_point = None

    for idx in range(len(paths)):
      point = paths[idx][0]
      closest_geom_body = getClosestGeom(bones_kd_trees, point, mjcf, mtu)

      # Add site to tendon
      min_dist = 0.05
      max_dist = 0.11
      if mtu in tail:
        max_dist = 0.05
      elif mtu in neck:
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
      if (closest_geom_body != prev_body and d > min_dist) or \
          idx == len(paths) - 1 or idx == 0 or d > max_dist:
        site_matrix = np.eye(4)
        site_matrix[:3, 3] = point
        body_matrix = calculate_transformation(closest_geom_body)
        site_matrix = np.matmul(np.linalg.inv(body_matrix), site_matrix)

        site_name = mtu + "_" + str(counter)
        # Create the site
        closest_geom_body.add("site", name=site_name,
                              pos=array_to_string(site_matrix[:3, 3]),
                              group='4', dclass='connector')

        counter += 1
        spatial.add('site', site=site_name)

        prev_point = point
        prev_body = closest_geom_body

        # check for wrapping geoms
        if idx != len(paths) - 1:
          geoms = physics.named.data.geom_xpos.axes.row.names
          dist = 100000000000
          closest_g = None
          closest_geom_pos = None
          for g_name in geoms:
            if 'wrapping' in g_name:
              pos = physics.named.data.geom_xpos[g_name]
              new_dist = np.linalg.norm(pos - point)
              if (mtu in lateral or mtu in neck) and \
                  g_name not in wrap_geoms_legs and \
                  pos[2] < point[2] and \
                  new_dist < dist:
                closest_g = g_name
                dist = new_dist
                closest_geom_pos = pos
              elif (mtu in flexors_back or
                    mtu in flexors_front or
                    mtu in extensors_front or
                    mtu in extensors_back) and \
                  pos[2] < point[2] and \
                  dist > 0.01 and \
                  new_dist < dist and \
                  g_name in wrap_geoms_legs:
                # in the legs we are interested only in wrapping
                # geometries that are lower than the site we
                # are adding
                closest_g = g_name
                dist = new_dist
                closest_geom_pos = pos

          if dist < 0.1:
            if (mtu in flexors_back or mtu in flexors_front) \
                  and closest_geom_pos[2] < point[2]:
              spatial.add('geom', geom=closest_g,
                          sidesite=closest_g + '_backward')
            elif (mtu in extensors_back or mtu in extensors_front) \
                    and closest_geom_pos[2] < point[2]:
              spatial.add('geom', geom=closest_g,
                          sidesite=closest_g + '_forward')
            elif mtu in neck:
              pos1 = physics.named.data.site_xpos[closest_g + '_forward']
              d1 = np.linalg.norm(pos1 - point)
              pos2 = physics.named.data.site_xpos[closest_g + '_backward']
              d2 = np.linalg.norm(pos2 - point)
              if d1 < d2:
                spatial.add('geom', geom=closest_g,
                            sidesite=closest_g + '_forward')
              else:
                spatial.add('geom', geom=closest_g,
                            sidesite=closest_g + '_backward')
            elif mtu in torso and dist < 0.05:
              spatial.add('geom', geom=closest_g,
                          sidesite=closest_g + '_forward')

    if muscle_dynamics in ['Millard', 'Sigmoid']:
      muscle = mjcf.actuator.add('muscle', tendon=spatial, name=spatial.name,
                                 dclass=spatial.dclass)

      prms = [0] * 10
      prms[0] = 0.01
      prms[1] = 0.04
      if muscle_dynamics == 'Millard':
        muscle.dynprm = prms
      else:
        prms[2] = 0.1
        muscle.dynprm = prms

      muscle.range = [0.75, 1.05]
      muscle.force = -1
      muscle.scale = 2620
      muscle.lmin = 0.5
      muscle.lmax = 1.6
      muscle.vmax = 1.5
      muscle.fvmax = 1.2
      muscle.fpmax = 1.3
      muscle.ctrllimited = True
      muscle.ctrlrange = [0.0, 1.0]
    elif muscle_dynamics == 'General':
      muscle = mjcf.actuator.add('general', tendon=spatial,
                                 name=spatial.name, dclass=spatial.dclass)
      muscle.ctrllimited = True
      muscle.ctrlrange = [-1.0, 0.0]
      muscle.dyntype = "filter"
      muscle.dynprm = [0.05]
      if mtu in torso or mtu in neck:
        muscle.gainprm = [100]
      else:
        muscle.gainprm = [300]
    else:
      raise NameError(muscle_dynamics)

    print("Added mtu {}".format(muscle.name))

  everything_ok = False
  while not everything_ok:
    try:
      physics = mjcf_module.Physics.from_mjcf_model(mjcf)
      everything_ok = True

      forces = None
      # compute anatomical forces if we provide a scale_multiplier > 0
      if muscle_dynamics in ['Millard', 'Sigmoid'] and scale_multiplier > 0:
        volumes = np.array(volumes)
        cross_sections = np.array(cross_sections)
        lm = []
        lr = physics.named.model.actuator_lengthrange
        for mtu in used_muscles:
          mtu_name = mtu + "_tendon"
          L0 = (lr[mtu_name, 1] - lr[mtu_name, 0]) / (1.05 - 0.75)
          LM = physics.named.model.actuator_length0[mtu_name] - \
               (lr[mtu_name] - 0.75 * L0)
          lm.append(LM)

        lm = np.array(lm)
        forces = ((cross_sections * 10) + (volumes / lm)) * scale_multiplier

      actuators = mjcf.find_all('actuator')
      print("actuators", len(actuators))

      # assign forces and length ranges
      idx = 0
      for muscle in actuators:
        muscle.lengthrange = lr[idx]
        if forces is not None:
          muscle.force = forces[idx]
        print("mtu {}".format(muscle.name))
        idx += 1

    except Exception as inst:
      print(inst)
      s = str(inst)
      s = s.split()
      i = s.index('actuator')
      muscle_index = s[i + 1]

      if ':' in muscle_index:
        muscle_index = muscle_index[:-1]

      muscle_index = int(muscle_index)

      del mjcf.actuator.muscle[muscle_index]
