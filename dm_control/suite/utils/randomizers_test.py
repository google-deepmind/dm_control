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

"""Tests for randomizers.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.suite.utils import randomizers
import numpy as np
from six.moves import range

mjlib = mjbindings.mjlib


class RandomizeUnlimitedJointsTest(parameterized.TestCase):

  def setUp(self):
    self.rand = np.random.RandomState(100)

  def test_single_joint_of_each_type(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <default>
            <joint range="0 90" />
          </default>
          <worldbody>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="free" type="free"/>
            </body>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="limited_hinge" type="hinge" limited="true"/>
              <joint name="slide" type="slide"/>
              <joint name="limited_slide" type="slide" limited="true"/>
              <joint name="hinge" type="hinge"/>
            </body>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="ball" type="ball"/>
            </body>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="limited_ball" type="ball" limited="true"/>
            </body>
          </worldbody>
        </mujoco>""")

    randomizers.randomize_limited_and_rotational_joints(physics, self.rand)
    self.assertNotEqual(0., physics.named.data.qpos['hinge'])
    self.assertNotEqual(0., physics.named.data.qpos['limited_hinge'])
    self.assertNotEqual(0., physics.named.data.qpos['limited_slide'])

    self.assertNotEqual(0., np.sum(physics.named.data.qpos['ball']))
    self.assertNotEqual(0., np.sum(physics.named.data.qpos['limited_ball']))

    self.assertNotEqual(0., np.sum(physics.named.data.qpos['free'][3:]))

    # Unlimited slide and the positional part of the free joint remains
    # uninitialized.
    self.assertEqual(0., physics.named.data.qpos['slide'])
    self.assertEqual(0., np.sum(physics.named.data.qpos['free'][:3]))

  def test_multiple_joints_of_same_type(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <worldbody>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="hinge_1" type="hinge"/>
              <joint name="hinge_2" type="hinge"/>
              <joint name="hinge_3" type="hinge"/>
            </body>
          </worldbody>
        </mujoco>""")

    randomizers.randomize_limited_and_rotational_joints(physics, self.rand)
    self.assertNotEqual(0., physics.named.data.qpos['hinge_1'])
    self.assertNotEqual(0., physics.named.data.qpos['hinge_2'])
    self.assertNotEqual(0., physics.named.data.qpos['hinge_3'])

    self.assertNotEqual(physics.named.data.qpos['hinge_1'],
                        physics.named.data.qpos['hinge_2'])

    self.assertNotEqual(physics.named.data.qpos['hinge_2'],
                        physics.named.data.qpos['hinge_3'])

    self.assertNotEqual(physics.named.data.qpos['hinge_1'],
                        physics.named.data.qpos['hinge_3'])

  def test_unlimited_hinge_randomization_range(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <worldbody>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="hinge" type="hinge"/>
            </body>
          </worldbody>
        </mujoco>""")

    for _ in range(10):
      randomizers.randomize_limited_and_rotational_joints(physics, self.rand)
      self.assertBetween(physics.named.data.qpos['hinge'], -np.pi, np.pi)

  def test_limited_1d_joint_limits_are_respected(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <default>
            <joint limited="true"/>
          </default>
          <worldbody>
            <body>
              <geom type="box" size="1 1 1"/>
              <joint name="hinge" type="hinge" range="0 10"/>
              <joint name="slide" type="slide" range="30 50"/>
            </body>
          </worldbody>
        </mujoco>""")

    for _ in range(10):
      randomizers.randomize_limited_and_rotational_joints(physics, self.rand)
      self.assertBetween(physics.named.data.qpos['hinge'],
                         np.deg2rad(0), np.deg2rad(10))
      self.assertBetween(physics.named.data.qpos['slide'], 30, 50)

  def test_limited_ball_joint_are_respected(self):
    physics = mujoco.Physics.from_xml_string("""<mujoco>
          <worldbody>
            <body name="body" zaxis="1 0 0">
              <geom type="box" size="1 1 1"/>
              <joint name="ball" type="ball" limited="true" range="0 60"/>
            </body>
          </worldbody>
        </mujoco>""")

    body_axis = np.array([1., 0., 0.])
    joint_axis = np.zeros(3)
    for _ in range(10):
      randomizers.randomize_limited_and_rotational_joints(physics, self.rand)

      quat = physics.named.data.qpos['ball']
      mjlib.mju_rotVecQuat(joint_axis, body_axis, quat)
      angle_cos = np.dot(body_axis, joint_axis)
      self.assertGreater(angle_cos, 0.5)  # cos(60) = 0.5


if __name__ == '__main__':
  absltest.main()
