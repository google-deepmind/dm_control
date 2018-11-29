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

"""Tests for observable."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from dm_control import mujoco
from dm_control.composer.observation import fake_physics
from dm_control.composer.observation.observable import base
import numpy as np
import six


_MJCF = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <body name="body" pos="0 0 0">
      <joint name="my_hinge" type="hinge" pos="-.1 -.2 -.3" axis="1 -1 0"/>
      <geom name="my_box" type="box" size=".1 .2 .3" rgba="0 0 1 1"/>
      <geom name="small_sphere" type="sphere" size=".12" pos=".1 .2 .3"/>
    </body>
    <camera name="world" mode="targetbody" target="body" pos="1 1 1" />
  </worldbody>
</mujoco>
"""


class _FakeBaseObservable(base.Observable):

  def _callable(self, physics):
    pass


class ObservableTest(absltest.TestCase):

  def testBaseProperties(self):
    fake_observable = _FakeBaseObservable(update_interval=42,
                                          buffer_size=5,
                                          delay=10,
                                          aggregator=None,
                                          corruptor=None)
    self.assertEqual(fake_observable.update_interval, 42)
    self.assertEqual(fake_observable.buffer_size, 5)
    self.assertEqual(fake_observable.delay, 10)

    fake_observable.update_interval = 48
    self.assertEqual(fake_observable.update_interval, 48)

    fake_observable.buffer_size = 7
    self.assertEqual(fake_observable.buffer_size, 7)

    fake_observable.delay = 13
    self.assertEqual(fake_observable.delay, 13)

    enabled = not fake_observable.enabled
    fake_observable.enabled = not fake_observable.enabled
    self.assertEqual(fake_observable.enabled, enabled)

  def testGeneric(self):
    physics = fake_physics.FakePhysics()
    repeated_observable = base.Generic(
        fake_physics.FakePhysics.repeated, update_interval=42)
    repeated_observation = repeated_observable.observation_callable(physics)()
    self.assertEqual(repeated_observable.update_interval, 42)
    np.testing.assert_array_equal(repeated_observation, [0, 0])

  def testMujocoFeature(self):
    physics = mujoco.Physics.from_xml_string(_MJCF)

    hinge_observable = base.MujocoFeature(
        kind='qpos', feature_name='my_hinge')
    hinge_observation = hinge_observable.observation_callable(physics)()
    np.testing.assert_array_equal(
        hinge_observation, physics.named.data.qpos['my_hinge'])

    box_observable = base.MujocoFeature(
        kind='geom_xpos', feature_name='small_sphere', update_interval=5)
    box_observation = box_observable.observation_callable(physics)()
    self.assertEqual(box_observable.update_interval, 5)
    np.testing.assert_array_equal(
        box_observation, physics.named.data.geom_xpos['small_sphere'])

    observable_from_callable = base.MujocoFeature(
        kind='geom_xpos', feature_name=lambda: ['my_box', 'small_sphere'])
    observation_from_callable = (
        observable_from_callable.observation_callable(physics)())
    np.testing.assert_array_equal(
        observation_from_callable,
        physics.named.data.geom_xpos[['my_box', 'small_sphere']])

  def testMujocoCamera(self):
    physics = mujoco.Physics.from_xml_string(_MJCF)

    camera_observable = base.MujocoCamera(
        camera_name='world', height=480, width=640, update_interval=7)
    self.assertEqual(camera_observable.update_interval, 7)
    camera_observation = camera_observable.observation_callable(physics)()
    np.testing.assert_array_equal(
        camera_observation, physics.render(480, 640, 'world'))
    self.assertEqual(camera_observation.shape,
                     camera_observable.array_spec.shape)
    self.assertEqual(camera_observation.dtype,
                     camera_observable.array_spec.dtype)

    camera_observable.height = 300
    camera_observable.width = 400
    camera_observation = camera_observable.observation_callable(physics)()
    self.assertEqual(camera_observable.height, 300)
    self.assertEqual(camera_observable.width, 400)
    np.testing.assert_array_equal(
        camera_observation, physics.render(300, 400, 'world'))
    self.assertEqual(camera_observation.shape,
                     camera_observable.array_spec.shape)
    self.assertEqual(camera_observation.dtype,
                     camera_observable.array_spec.dtype)

  def testCorruptor(self):
    physics = fake_physics.FakePhysics()
    def add_twelve(old_value, random_state):
      del random_state  # Unused.
      return [x + 12 for x in old_value]
    repeated_observable = base.Generic(
        fake_physics.FakePhysics.repeated, corruptor=add_twelve)
    corrupted = repeated_observable.observation_callable(
        physics=physics, random_state=None)()
    np.testing.assert_array_equal(corrupted, [12, 12])

  def testInvalidAggregatorName(self):
    name = 'invalid_name'
    with six.assertRaisesRegex(self, KeyError, 'Unrecognized aggregator name'):
      _ = _FakeBaseObservable(update_interval=3, buffer_size=2, delay=1,
                              aggregator=name, corruptor=None)

if __name__ == '__main__':
  absltest.main()
