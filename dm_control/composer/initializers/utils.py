# Copyright 2018 The dm_control Authors.
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

"""Utilities that are helpful for implementing initializers."""

import collections


def _get_root_model(mjcf_elements):
  root_model = mjcf_elements[0].root.root_model
  for element in mjcf_elements:
    if element.root.root_model != root_model:
      raise ValueError('entities do not all belong to the same root model')
  return root_model


class JointStaticIsolator(object):
  """Helper class that isolates a collection of MuJoCo joints from others.

  An instance of this class is a context manager that caches the positions and
  velocities of all non-isolated joints *upon construction*, and resets them to
  their original state when the context exits.
  """

  def __init__(self, physics, joints):
    """Initializes the joint isolator.

    Args:
      physics: An instance of `mjcf.Physics`.
      joints: An iterable of `mjcf.Element` representing joints that may be
        modified inside the context managed by this isolator.
    """
    if not isinstance(joints, collections.Iterable):
      joints = [joints]
    root_model = _get_root_model(joints)
    other_joints = [joint for joint in root_model.find_all('joint')
                    if joint not in joints]
    if other_joints:
      self._other_joints_mj = physics.bind(other_joints)
      self._initial_qpos = self._other_joints_mj.qpos.copy()
      self._initial_qvel = self._other_joints_mj.qvel.copy()
    else:
      self._other_joints_mj = None

  def __enter__(self):
    pass

  def __exit__(self, exc_type, exc_value, traceback):
    del exc_type, exc_value, traceback  # unused
    if self._other_joints_mj:
      self._other_joints_mj.qpos = self._initial_qpos
      self._other_joints_mj.qvel = self._initial_qvel
