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

"""Tests for `dm_control.mjcf.traversal_utils`."""

from absl.testing import absltest
from dm_control import mjcf
import numpy as np


class TraversalUtilsTest(absltest.TestCase):

  def assert_same_attributes(self, element, expected_attributes):
    actual_attributes = element.get_attributes()
    self.assertEqual(set(actual_attributes.keys()),
                     set(expected_attributes.keys()))
    for name in actual_attributes:
      actual_value = actual_attributes[name]
      expected_value = expected_attributes[name]
      np.testing.assert_array_equal(actual_value, expected_value, name)

  def test_resolve_root_defaults(self):
    root = mjcf.RootElement()
    root.default.geom.type = 'box'
    root.default.geom.pos = [0, 1, 0]
    root.default.joint.ref = 2
    root.default.joint.pos = [0, 0, 1]

    # Own attribute overriding default.
    body = root.worldbody.add('body')
    geom1 = body.add('geom', type='sphere')
    mjcf.commit_defaults(geom1)
    self.assert_same_attributes(geom1, dict(
        type='sphere',
        pos=[0, 1, 0]))

    # No explicit attributes.
    geom2 = body.add('geom')
    mjcf.commit_defaults(geom2)
    self.assert_same_attributes(geom2, dict(
        type='box',
        pos=[0, 1, 0]))

    # Attributes mutually exclusive with those defined in default.
    joint1 = body.add('joint', margin=3)
    mjcf.commit_defaults(joint1)
    self.assert_same_attributes(joint1, dict(
        pos=[0, 0, 1],
        ref=2,
        margin=3))

  def test_resolve_defaults_for_some_attributes(self):
    root = mjcf.RootElement()
    root.default.geom.type = 'box'
    root.default.geom.pos = [0, 1, 0]
    geom1 = root.worldbody.add('geom')
    mjcf.commit_defaults(geom1, attributes=['pos'])
    self.assert_same_attributes(geom1, dict(
        pos=[0, 1, 0]))

  def test_resolve_hierarchies_of_defaults(self):
    root = mjcf.RootElement()
    root.default.geom.type = 'box'
    root.default.joint.pos = [0, 1, 0]

    top1 = root.default.add('default', dclass='top1')
    top1.geom.pos = [0.1, 0, 0]
    top1.joint.pos = [1, 0, 0]
    top1.joint.axis = [0, 0, 1]
    sub1 = top1.add('default', dclass='sub1')
    sub1.geom.size = [0.5]

    top2 = root.default.add('default', dclass='top2')
    top2.joint.pos = [0, 0, 1]
    top2.joint.axis = [0, 1, 0]
    top2.geom.type = 'sphere'

    body = root.worldbody.add('body')
    geom1 = body.add('geom', dclass=sub1)
    mjcf.commit_defaults(geom1)
    self.assert_same_attributes(geom1, dict(
        dclass=sub1,
        type='box',
        size=[0.5],
        pos=[0.1, 0, 0]))

    geom2 = body.add('geom', dclass=top1)
    mjcf.commit_defaults(geom2)
    self.assert_same_attributes(geom2, dict(
        dclass=top1,
        type='box',
        pos=[0.1, 0, 0]))

    geom3 = body.add('geom', dclass=top2)
    mjcf.commit_defaults(geom3)
    self.assert_same_attributes(geom3, dict(
        dclass=top2,
        type='sphere'))

    geom4 = body.add('geom')
    mjcf.commit_defaults(geom4)
    self.assert_same_attributes(geom4, dict(
        type='box'))

    joint1 = body.add('joint', dclass=sub1)
    mjcf.commit_defaults(joint1)
    self.assert_same_attributes(joint1, dict(
        dclass=sub1,
        pos=[1, 0, 0],
        axis=[0, 0, 1]))

    joint2 = body.add('joint', dclass=top2)
    mjcf.commit_defaults(joint2)
    self.assert_same_attributes(joint2, dict(
        dclass=top2,
        pos=[0, 0, 1],
        axis=[0, 1, 0]))

    joint3 = body.add('joint')
    mjcf.commit_defaults(joint3)
    self.assert_same_attributes(joint3, dict(
        pos=[0, 1, 0]))

  def test_resolve_actuator_defaults(self):
    root = mjcf.RootElement()
    root.default.general.forcelimited = 'true'
    root.default.motor.forcerange = [-2, 3]
    root.default.position.kp = 0.1
    root.default.velocity.kv = 0.2

    body = root.worldbody.add('body')
    joint = body.add('joint')

    motor = root.actuator.add('motor', joint=joint)
    mjcf.commit_defaults(motor)
    self.assert_same_attributes(motor, dict(
        joint=joint,
        forcelimited='true',
        forcerange=[-2, 3]))

    position = root.actuator.add('position', joint=joint)
    mjcf.commit_defaults(position)
    self.assert_same_attributes(position, dict(
        joint=joint,
        kp=0.1,
        kv=0.2,
        forcelimited='true',
        forcerange=[-2, 3]))

    velocity = root.actuator.add('velocity', joint=joint)
    mjcf.commit_defaults(velocity)
    self.assert_same_attributes(velocity, dict(
        joint=joint,
        kv=0.2,
        forcelimited='true',
        forcerange=[-2, 3]))

  def test_resolve_childclass(self):
    root = mjcf.RootElement()
    root.default.geom.type = 'capsule'
    top = root.default.add('default', dclass='top')
    top.geom.pos = [0, 1, 0]
    sub = top.add('default', dclass='sub')
    sub.geom.pos = [0, 0, 1]

    # Element only affected by the childclass of immediate parent.
    body = root.worldbody.add('body', childclass=sub)
    geom1 = body.add('geom')
    mjcf.commit_defaults(geom1)
    self.assert_same_attributes(geom1, dict(
        type='capsule',
        pos=[0, 0, 1]))

    # Element overrides parent's childclass.
    geom2 = body.add('geom', dclass=top)
    mjcf.commit_defaults(geom2)
    self.assert_same_attributes(geom2, dict(
        dclass=top,
        type='capsule',
        pos=[0, 1, 0]))

    # Element's parent overrides grandparent's childclass.
    subbody1 = body.add('body', childclass=top)
    geom3 = subbody1.add('geom')
    mjcf.commit_defaults(geom3)
    self.assert_same_attributes(geom3, dict(
        type='capsule',
        pos=[0, 1, 0]))

    # Element's grandparent does not specify a childclass, but grandparent does.
    subbody2 = body.add('body')
    geom4 = subbody2.add('geom')
    mjcf.commit_defaults(geom4)
    self.assert_same_attributes(geom4, dict(
        type='capsule',
        pos=[0, 0, 1]))

    # Direct child of worldbody, not affected by any childclass.
    geom5 = root.worldbody.add('geom')
    mjcf.commit_defaults(geom5)
    self.assert_same_attributes(geom5, dict(
        type='capsule'))


if __name__ == '__main__':
  absltest.main()
