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

"""Add markers to the dog model, used for motion capture tracking."""

MARKERS_PER_BODY = {
    "torso": (
        (-0.25, 0, 0.13),
        (-0.1, 0, 0.13),
        (0.05, 0, 0.15),
        (-0.25, -0.06, -0.01),
        (-0.1, -0.08, -0.02),
        (0.05, -0.09, -0.02),
        (-0.25, 0.06, -0.01),
        (-0.1, 0.08, -0.02),
        (0.05, 0.09, -0.02),
    ),
    "C_6": ((0.05, -0.05, 0), (0.05, 0.05, 0),),
    "upper_arm": ((0.04, 0, 0),),
    "lower_arm": ((0.025, 0, 0), (0.025, 0, -0.1),),
    "hand": ((0, -0.02, 0), (0, 0.02, 0), (0.02, -0.02, -0.08), (0.02, 0.02, -0.08),),
    "finger": ((0.065, 0, 0),),
    "pelvis": ((0.04, -0.07, 0.02), (0.04, 0.07, 0.02),),
    "upper_leg": ((-0.03, 0.04, 0),),
    "lower_leg": ((0.04, 0, 0), (-0.04, 0.035, 0),),
    "foot": ((0, -0.02, 0), (0, 0.02, 0), (0, -0.02, -0.1), (0, 0.03, -0.1),),
    "toe": ((0.055, 0, 0),),
    "Ca_10": ((0, 0, 0.01),),
    "skull": ((0.01, 0, 0.04), (0, -0.05, 0), (0, 0.05, 0), (-0.1, 0, 0.02),),
}


def add_markers(model):
  """Add markers to the given model.

  Args:
      model: The model to add markers to.
  """
  bodies = model.find_all("body")

  total_markers = 0
  for body in bodies:
    for name, markers_pos in MARKERS_PER_BODY.items():
      if name in body.name and "anchor" not in body.name:
        marker_idx = 0
        for pos in markers_pos:
          pos = list(pos)
          if "_R" in body.name:
            pos[1] *= -1
          body.add(
              "site",
              name="marker_" + body.name + "_" + str(marker_idx),
              pos=pos,
              dclass="marker",
          )

          marker_idx += 1
          total_markers += 1
  
  for i in range(total_markers):
    marker_body = model.worldbody.add(
        "body", name="marker_" + str(i), mocap=True)
    marker_body.add("site", name="marker_" + str(i), dclass="mocap_marker")
