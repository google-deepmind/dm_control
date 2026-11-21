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


def add_wrapping_geoms(model):
  """ Adds wrapping geometries and sites to the specified model.

  Args:
      model: The model to which the wrapping geometries and sites will be added.
  """
  pelvis = model.find("body", "pelvis")
  pelvis.add(
      "geom",
      name="hip_R_wrapping",
      dclass="wrapping",
      size=[0.05, 0.04],
      pos=[0, 0.05, -0.02],
  )
  pelvis.add(
      "site",
      name="hip_R_wrapping_forward",
      dclass="wrapping",
      pos=[0.08, -0.05, -0.00],
  )
  pelvis.add(
      "site",
      name="hip_R_wrapping_backward",
      dclass="wrapping",
      pos=[-0.04, -0.05, -0.04],
  )

  pelvis.add(
      "geom",
      name="hip_L_wrapping",
      dclass="wrapping",
      size=[0.05, 0.04],
      pos=[0, -0.05, -0.02],
  )
  pelvis.add(
      "site",
      name="hip_L_wrapping_forward",
      dclass="wrapping",
      pos=[0.08, 0.05, -0.00],
  )
  pelvis.add(
      "site",
      name="hip_L_wrapping_backward",
      dclass="wrapping",
      pos=[-0.04, 0.05, -0.04],
  )

  C_2 = model.find("body", "C_2")
  C_2.add(
      "geom",
      name="C_2_wrapping",
      dclass="wrapping",
      pos=[0.02, 0, 0.03],
      size=[0.015, 0.04],
  )
  C_2.add("site", name="C_2_wrapping_forward",
          pos=[0.05, 0, 0.02], dclass="wrapping")
  C_2.add(
      "site", name="C_2_wrapping_backward", pos=[-0.005, 0, 0.04], dclass="wrapping"
  )
  C_7 = model.find("body", "C_7")
  C_7.add("geom", name="C_7_wrapping", dclass="wrapping", size=[0.03, 0.07])

  Ca_1 = model.find("body", "Ca_1")
  Ca_1.add("geom", name="Ca_1_wrapping", size=[0.008, 0.04], dclass="wrapping")
  Ca_1.add("site", name="Ca_1_wrapping_forward",
           pos=[0, 0, -0.05], dclass="wrapping")
  Ca_1.add("site", name="Ca_1_wrapping_backward",
           pos=[0, 0, 0.05], dclass="wrapping")

  torso = model.find("body", "torso")
  torso.add(
      "site", name="C_7_wrapping_forward", pos=[0.19, 0, 0.055], dclass="wrapping"
  )
  torso.add(
      "site", name="C_7_wrapping_backward", pos=[0.09, 0, 0.12], dclass="wrapping"
  )

  for i in range(1, 8, 1):
    body = model.find("body", f"L_{i}")
    body.add("geom", name=f"L_{i}_wrapping", size=[
             0.009, 0.04], dclass="wrapping")
    body.add(
        "site",
        name=f"L_{i}_wrapping_forward",
        pos=[0, 0, 0.1],
        dclass="wrapping",
    )
    body.add(
        "site",
        name=f"L_{i}_wrapping_backward",
        pos=[0, 0, -0.1],
        dclass="wrapping",
    )

  # Left Side
  upper_leg_L = model.find("body", "upper_leg_L")
  upper_leg_L.add(
      "site",
      name="knee_L_wrapping_forward",
      pos=[0.07, 0.0311, -0.175],
      dclass="wrapping",
  )
  upper_leg_L.add(
      "site",
      name="knee_L_wrapping_backward",
      pos=[-0.09, 0.0311, 0],
      dclass="wrapping",
  )

  lower_leg_L = model.find("body", "lower_leg_L")
  lower_leg_L.add(
      "geom", name="knee_L_wrapping", dclass="wrapping", size=[0.02, 0.05]
  )
  lower_leg_L.add(
      "site",
      name="ankle_L_wrapping_forward",
      pos=[-0.08, 0, -0.13824],
      dclass="wrapping",
  )
  lower_leg_L.add(
      "site",
      name="ankle_L_wrapping_backward",
      pos=[-0.25, 0, -0.13824],
      dclass="wrapping",
  )

  foot_L = model.find("body", "foot_L")
  foot_L.add("geom", name="ankle_L_wrapping",
             dclass="wrapping", size=[0.02, 0.05])
  foot_L.add(
      "site",
      name="toe_L_wrapping_forward",
      pos=[0.07, 0, -0.11993],
      dclass="wrapping",
  )
  foot_L.add(
      "site",
      name="toe_L_wrapping_backward",
      pos=[-0.07, 0, -0.11993],
      dclass="wrapping",
  )

  toe_L = model.find("body", "toe_L")
  toe_L.add("geom", name="toe_L_wrapping",
            dclass="wrapping", size=[0.01, 0.05])

  scapula_L = model.find("body", "scapula_L")
  scapula_L.add(
      "geom",
      name="shoulder_L_wrapping",
      pos=[0.075, 0.033, -0.13],
      size=[0.015, 0.05],
      dclass="wrapping",
  )
  scapula_L.add(
      "site",
      name="shoulder_L_wrapping_forward",
      pos=[0.1, 0.033, -0.13],
      dclass="wrapping",
  )
  scapula_L.add(
      "site",
      name="shoulder_L_wrapping_backward",
      pos=[0.02, 0.033, -0.13],
      dclass="wrapping",
  )
  upper_arm_L = model.find("body", "upper_arm_L")
  upper_arm_L.add(
      "geom",
      name="elbow_L_wrapping",
      pos=[-0.05, 0.015, -0.135],
      size=[0.015, 0.05],
      dclass="wrapping",
  )
  upper_arm_L.add(
      "site",
      name="elbow_L_wrapping_forward",
      pos=[0.03, 0.015, -0.15],
      dclass="wrapping",
  )
  upper_arm_L.add(
      "site",
      name="elbow_L_wrapping_backward",
      pos=[-0.1, 0.015, -0.15],
      dclass="wrapping",
  )
  lower_arm_L = model.find("body", "lower_arm_L")
  lower_arm_L.add(
      "site",
      name="wrist_L_wrapping_forward",
      pos=[0.015, -0.015, -0.18],
      dclass="wrapping",
  )
  lower_arm_L.add(
      "site",
      name="wrist_L_wrapping_backward",
      pos=[-0.05, -0.015, -0.19],
      dclass="wrapping",
  )
  hand_anchor_L = model.find("body", "hand_anchor_L")
  hand_anchor_L.add(
      "geom",
      name="wrist_L_wrapping",
      size=[0.011, 0.05],
      pos=[0, 0, 0.012],
      dclass="wrapping",
  )
  hand_L = model.find("body", "hand_L")
  hand_L.add(
      "geom",
      name="finger_L_wrapping",
      size=[0.015, 0.05],
      pos=[0.01, 0, -0.05],
      dclass="wrapping",
  )
  hand_L.add(
      "site",
      name="finger_L_wrapping_forward",
      pos=[0.05, 0, -0.06],
      dclass="wrapping",
  )
  hand_L.add(
      "site",
      name="finger_L_wrapping_backward",
      pos=[-0.05, 0, -0.06],
      dclass="wrapping",
  )

  # Right Side
  upper_leg_R = model.find("body", "upper_leg_R")
  upper_leg_R.add(
      "site",
      name="knee_R_wrapping_forward",
      pos=[0.07, -0.0311, -0.175],
      dclass="wrapping",
  )
  upper_leg_R.add(
      "site",
      name="knee_R_wrapping_backward",
      pos=[-0.09, -0.0311, 0],
      dclass="wrapping",
  )

  lower_leg_R = model.find("body", "lower_leg_R")
  lower_leg_R.add(
      "geom", name="knee_R_wrapping", dclass="wrapping", size=[0.02, 0.05]
  )
  lower_leg_R.add(
      "site",
      name="ankle_R_wrapping_forward",
      pos=[-0.08, 0, -0.13824],
      dclass="wrapping",
  )
  lower_leg_R.add(
      "site",
      name="ankle_R_wrapping_backward",
      pos=[-0.25, 0, -0.13824],
      dclass="wrapping",
  )

  foot_R = model.find("body", "foot_R")
  foot_R.add("geom", name="ankle_R_wrapping",
             dclass="wrapping", size=[0.02, 0.05])
  foot_R.add(
      "site",
      name="toe_R_wrapping_forward",
      pos=[0.07, 0, -0.11993],
      dclass="wrapping",
  )
  foot_R.add(
      "site",
      name="toe_R_wrapping_backward",
      pos=[-0.07, 0, -0.11993],
      dclass="wrapping",
  )

  toe_R = model.find("body", "toe_R")
  toe_R.add("geom", name="toe_R_wrapping",
            dclass="wrapping", size=[0.01, 0.05])

  scapula_R = model.find("body", "scapula_R")
  scapula_R.add(
      "geom",
      name="shoulder_R_wrapping",
      pos=[0.075, -0.033, -0.13],
      size=[0.015, 0.05],
      dclass="wrapping",
  )
  scapula_R.add(
      "site",
      name="shoulder_R_wrapping_forward",
      pos=[0.1, -0.033, -0.13],
      dclass="wrapping",
  )
  scapula_R.add(
      "site",
      name="shoulder_R_wrapping_backward",
      pos=[0.02, -0.033, -0.13],
      dclass="wrapping",
  )
  upper_arm_R = model.find("body", "upper_arm_R")
  upper_arm_R.add(
      "geom",
      name="elbow_R_wrapping",
      pos=[-0.05, -0.015, -0.135],
      size=[0.015, 0.05],
      dclass="wrapping",
  )
  upper_arm_R.add(
      "site",
      name="elbow_R_wrapping_forward",
      pos=[0.03, -0.015, -0.15],
      dclass="wrapping",
  )
  upper_arm_R.add(
      "site",
      name="elbow_R_wrapping_backward",
      pos=[-0.1, -0.015, -0.15],
      dclass="wrapping",
  )
  lower_arm_R = model.find("body", "lower_arm_R")
  lower_arm_R.add(
      "site",
      name="wrist_R_wrapping_forward",
      pos=[0.015, 0.015, -0.18],
      dclass="wrapping",
  )
  lower_arm_R.add(
      "site",
      name="wrist_R_wrapping_backward",
      pos=[-0.05, 0.015, -0.19],
      dclass="wrapping",
  )
  hand_anchor_R = model.find("body", "hand_anchor_R")
  hand_anchor_R.add(
      "geom",
      name="wrist_R_wrapping",
      size=[0.011, 0.05],
      pos=[0, 0, 0.012],
      dclass="wrapping",
  )
  hand_R = model.find("body", "hand_R")
  hand_R.add(
      "geom",
      name="finger_R_wrapping",
      size=[0.015, 0.05],
      pos=[0.01, 0, -0.05],
      dclass="wrapping",
  )
  hand_R.add(
      "site",
      name="finger_R_wrapping_forward",
      pos=[0.05, 0, -0.06],
      dclass="wrapping",
  )
  hand_R.add(
      "site",
      name="finger_R_wrapping_backward",
      pos=[-0.05, 0, -0.06],
      dclass="wrapping",
  )
