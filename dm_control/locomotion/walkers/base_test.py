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

"""Tests for dm_control.locomotion.walkers.base."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control import mjcf
from dm_control.locomotion.walkers import base

import numpy as np


class FakeWalker(base.Walker):

  def _build(self):
    self._mjcf_root = mjcf.RootElement(model='walker')
    self._egocentric_camera = self._mjcf_root.worldbody.add(
        'camera', name='egocentric', xyaxes=[0, -1, 0, 0, 0, 1])
    self._torso_body = self._mjcf_root.worldbody.add(
        'body', name='torso', xyaxes=[0, 1, 0, -1, 0, 0])

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def actuators(self):
    return []

  @property
  def root_body(self):
    return self._torso_body

  @property
  def end_effectors(self):
    return []

  @property
  def observable_joints(self):
    return []

  @property
  def ground_contact_geoms(self):
    return []

  @property
  def egocentric_camera(self):
    return self._egocentric_camera


class BaseWalkerTest(absltest.TestCase):

  def testTransformVectorToEgocentricFrame(self):
    walker = FakeWalker()
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)

    # 3D vectors
    np.testing.assert_allclose(
        walker.transform_vec_to_egocentric_frame(physics, [0, 1, 0]), [1, 0, 0],
        atol=1e-10)
    np.testing.assert_allclose(
        walker.transform_vec_to_egocentric_frame(physics, [-1, 0, 0]),
        [0, 1, 0],
        atol=1e-10)
    np.testing.assert_allclose(
        walker.transform_vec_to_egocentric_frame(physics, [0, 0, 1]), [0, 0, 1],
        atol=1e-10)

    # 2D vectors; z-component is ignored
    np.testing.assert_allclose(
        walker.transform_vec_to_egocentric_frame(physics, [0, 1]), [1, 0],
        atol=1e-10)
    np.testing.assert_allclose(
        walker.transform_vec_to_egocentric_frame(physics, [-1, 0]), [0, 1],
        atol=1e-10)

  def testTransformMatrixToEgocentricFrame(self):
    walker = FakeWalker()
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)

    rotation_atob = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]])
    ego_rotation_atob = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

    np.testing.assert_allclose(
        walker.transform_xmat_to_egocentric_frame(physics, rotation_atob),
        ego_rotation_atob, atol=1e-10)

    flat_rotation_atob = np.reshape(rotation_atob, -1)
    flat_rotation_ego_atob = np.reshape(ego_rotation_atob, -1)
    np.testing.assert_allclose(
        walker.transform_xmat_to_egocentric_frame(physics, flat_rotation_atob),
        flat_rotation_ego_atob, atol=1e-10)


if __name__ == '__main__':
  absltest.main()
