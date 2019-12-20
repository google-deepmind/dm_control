# Copyright 2019 The dm_control Authors.
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

"""Utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.mujoco.wrapper.mjbindings import mjlib

import numpy as np


def euler2quat(ax, ay, az):
  """Converts euler angles to a quaternion.

  Note: rotation order is zyx

  Args:
    ax: Roll angle (deg)
    ay: Pitch angle (deg).
    az: Yaw angle (deg).

  Returns:
    A numpy array representing the rotation as a quaternion.
  """
  r1 = az
  r2 = ay
  r3 = ax

  c1 = np.cos(np.deg2rad(r1 / 2))
  s1 = np.sin(np.deg2rad(r1 / 2))
  c2 = np.cos(np.deg2rad(r2 / 2))
  s2 = np.sin(np.deg2rad(r2 / 2))
  c3 = np.cos(np.deg2rad(r3 / 2))
  s3 = np.sin(np.deg2rad(r3 / 2))

  q0 = c1 * c2 * c3 + s1 * s2 * s3
  q1 = c1 * c2 * s3 - s1 * s2 * c3
  q2 = c1 * s2 * c3 + s1 * c2 * s3
  q3 = s1 * c2 * c3 - c1 * s2 * s3

  return np.array([q0, q1, q2, q3])


def mj_quatprod(q, r):
  quaternion = np.zeros(4)
  mjlib.mju_mulQuat(quaternion, np.ascontiguousarray(q),
                    np.ascontiguousarray(r))
  return quaternion


def mj_quat2vel(q, dt):
  vel = np.zeros(3)
  mjlib.mju_quat2Vel(vel, np.ascontiguousarray(q), dt)
  return vel


def mj_quatneg(q):
  quaternion = np.zeros(4)
  mjlib.mju_negQuat(quaternion, np.ascontiguousarray(q))
  return quaternion


def mj_quatdiff(source, target):
  return mj_quatprod(mj_quatneg(source), np.ascontiguousarray(target))
