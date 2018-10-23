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

"""Randomization functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco.wrapper import mjbindings
import numpy as np
from six.moves import range


def random_limited_quaternion(random, limit):
  """Generates a random quaternion limited to the specified rotations."""
  axis = random.randn(3)
  axis /= np.linalg.norm(axis)
  angle = random.rand() * limit

  quaternion = np.zeros(4)
  mjbindings.mjlib.mju_axisAngle2Quat(quaternion, axis, angle)

  return quaternion


def randomize_limited_and_rotational_joints(physics, random=None):
  """Randomizes the positions of joints defined in the physics body.

  The following randomization rules apply:
    - Bounded joints (hinges or sliders) are sampled uniformly in the bounds.
    - Unbounded hinges are samples uniformly in [-pi, pi]
    - Quaternions for unlimited free joints and ball joints are sampled
      uniformly on the unit 3-sphere.
    - Quaternions for limited ball joints are sampled uniformly on a sector
      of the unit 3-sphere.
    - The linear degrees of freedom of free joints are not randomized.

  Args:
    physics: Instance of 'Physics' class that holds a loaded model.
    random: Optional instance of 'np.random.RandomState'. Defaults to the global
      NumPy random state.
  """
  random = random or np.random

  hinge = mjbindings.enums.mjtJoint.mjJNT_HINGE
  slide = mjbindings.enums.mjtJoint.mjJNT_SLIDE
  ball = mjbindings.enums.mjtJoint.mjJNT_BALL
  free = mjbindings.enums.mjtJoint.mjJNT_FREE

  qpos = physics.named.data.qpos

  for joint_id in range(physics.model.njnt):
    joint_name = physics.model.id2name(joint_id, 'joint')
    joint_type = physics.model.jnt_type[joint_id]
    is_limited = physics.model.jnt_limited[joint_id]
    range_min, range_max = physics.model.jnt_range[joint_id]

    if is_limited:
      if joint_type == hinge or joint_type == slide:
        qpos[joint_name] = random.uniform(range_min, range_max)

      elif joint_type == ball:
        qpos[joint_name] = random_limited_quaternion(random, range_max)

    else:
      if joint_type == hinge:
        qpos[joint_name] = random.uniform(-np.pi, np.pi)

      elif joint_type == ball:
        quat = random.randn(4)
        quat /= np.linalg.norm(quat)
        qpos[joint_name] = quat

      elif joint_type == free:
        quat = random.rand(4)
        quat /= np.linalg.norm(quat)
        qpos[joint_name][3:] = quat

