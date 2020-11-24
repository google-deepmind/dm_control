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

"""Tests for dm_control.composer.props.primitive."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.entities.props import primitive
import numpy as np


class PrimitiveTest(parameterized.TestCase):

  def _make_free_prop(self, geom_type='sphere', size=(0.1,), **kwargs):
    prop = primitive.Primitive(geom_type=geom_type, size=size, **kwargs)
    arena = composer.Arena()
    arena.add_free_entity(prop)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    return prop, physics

  @parameterized.parameters([
      dict(geom_type='sphere', size=[0.1]),
      dict(geom_type='capsule', size=[0.1, 0.2]),
      dict(geom_type='cylinder', size=[0.1, 0.2]),
      dict(geom_type='box', size=[0.1, 0.2, 0.3]),
      dict(geom_type='ellipsoid', size=[0.1, 0.2, 0.3]),
  ])
  def test_instantiation(self, geom_type, size):
    name = 'foo'
    rgba = [1., 0., 1., 0.5]
    prop, physics = self._make_free_prop(
        geom_type=geom_type, size=size, name=name, rgba=rgba)
    # Check that the name and other kwargs are set correctly.
    self.assertEqual(prop.mjcf_model.model, name)
    np.testing.assert_array_equal(physics.bind(prop.geom).rgba, rgba)
    # Check that we can step without anything breaking.
    physics.step()

  @parameterized.parameters([
      dict(position=[0., 0., 0.]),
      dict(position=[0.1, -0.2, 0.3]),
  ])
  def test_position_observable(self, position):
    prop, physics = self._make_free_prop()
    prop.set_pose(physics, position=position)
    observation = prop.observables.position(physics)
    np.testing.assert_array_equal(position, observation)

  @parameterized.parameters([
      dict(quat=[1., 0., 0., 0.]),
      dict(quat=[0., -1., 1., 0.]),
  ])
  def test_orientation_observable(self, quat):
    prop, physics = self._make_free_prop()
    normalized_quat = np.array(quat) / np.linalg.norm(quat)
    prop.set_pose(physics, quaternion=normalized_quat)
    observation = prop.observables.orientation(physics)
    np.testing.assert_array_almost_equal(normalized_quat, observation)

  @parameterized.parameters([
      dict(velocity=[0., 0., 0.]),
      dict(velocity=[0.1, -0.2, 0.3]),
  ])
  def test_linear_velocity_observable(self, velocity):
    prop, physics = self._make_free_prop()
    prop.set_velocity(physics, velocity=velocity)
    observation = prop.observables.linear_velocity(physics)
    np.testing.assert_array_almost_equal(velocity, observation)

  @parameterized.parameters([
      dict(angular_velocity=[0., 0., 0.]),
      dict(angular_velocity=[0.1, -0.2, 0.3]),
  ])
  def test_angular_velocity_observable(self, angular_velocity):
    prop, physics = self._make_free_prop()
    prop.set_velocity(physics, angular_velocity=angular_velocity)
    observation = prop.observables.angular_velocity(physics)
    np.testing.assert_array_almost_equal(angular_velocity, observation)


if __name__ == '__main__':
  absltest.main()
