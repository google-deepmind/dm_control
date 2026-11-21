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
"""Checks that built-in bounded aggregators retain observation limits."""

import functools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.composer.observation import fake_physics
from dm_control.composer.observation import observable
from dm_control.composer.observation import updater
from dm_env import specs
import numpy as np


class _BoundedObservable(observable.Generic):

  def __init__(self, source, minimum, maximum, aggregator):
    super().__init__(lambda _: source, buffer_size=3, aggregator=aggregator)
    self._spec = specs.BoundedArray(
        shape=source.shape, dtype=source.dtype, minimum=minimum, maximum=maximum
    )
    self.enabled = True

  @property
  def array_spec(self):
    return self._spec


class AggregatorBoundsTest(parameterized.TestCase):

  @parameterized.parameters('min', 'max', 'mean', 'median', 'sum')
  def test_builtin_aggregators_declare_the_attribute_consumed_by_updater(
      self, name
  ):
    aggregator = observable.base.AGGREGATORS[name]
    self.assertIs(aggregator.preserves_bounds, name != 'sum')

  @parameterized.product(
      name=('min', 'max', 'mean', 'median'),
      dtype=(np.float32, np.int16),
      vector_bounds=(False, True),
  )
  def test_builtin_reductions_preserve_bounds_dtype_and_observations(
      self, name, dtype, vector_bounds
  ):
    source = np.array([2, 4], dtype=dtype)
    minimum = np.array([0, 1], dtype=dtype) if vector_bounds else 0
    maximum = np.array([8, 10], dtype=dtype) if vector_bounds else 10
    obs = _BoundedObservable(source, minimum, maximum, name)
    observer = updater.Updater({'value': obs}, pad_with_initial_value=True)
    observer.reset(fake_physics.FakePhysics(), random_state=None)
    with mock.patch.object(updater.logging, 'warning') as warning:
      result_spec = observer.observation_spec()['value']
    self.assertIsInstance(result_spec, specs.BoundedArray)
    warning.assert_not_called()
    np.testing.assert_array_equal(result_spec.minimum, obs.array_spec.minimum)
    np.testing.assert_array_equal(result_spec.maximum, obs.array_spec.maximum)
    self.assertEqual(result_spec.shape, (2,))
    self.assertEqual(result_spec.name, 'value')
    observed = observer.get_observation()['value']
    result_spec.validate(observed)
    expected = getattr(np, name)(np.tile(source, (3, 1)), axis=0)
    self.assertEqual(result_spec.dtype, expected.dtype)
    np.testing.assert_array_equal(observed, expected)
    bad = np.full(result_spec.shape, 11, dtype=result_spec.dtype)
    with self.assertRaises(ValueError):
      result_spec.validate(bad)
    np.testing.assert_array_equal(source, [2, 4])

  @parameterized.parameters(np.float32, np.int16)
  def test_sum_still_discards_bounds_and_accepts_aggregates_outside_them(
      self, dtype
  ):
    source = np.array([2, 4], dtype=dtype)
    obs = _BoundedObservable(source, 0, 4, 'sum')
    observer = updater.Updater({'value': obs}, pad_with_initial_value=True)
    observer.reset(fake_physics.FakePhysics(), random_state=None)
    result_spec = observer.observation_spec()['value']
    self.assertIs(type(result_spec), specs.Array)
    actual = observer.get_observation()['value']
    np.testing.assert_array_equal(actual, [6, 12])
    result_spec.validate(actual)

  @parameterized.parameters(None, False, True)
  def test_custom_aggregator_bounds_contract_is_unchanged(self, declaration):
    aggregator = functools.partial(np.mean, axis=0)
    if declaration is not None:
      aggregator.preserves_bounds = declaration
    obs = _BoundedObservable(np.array([2.0, 3.0]), 0.0, 4.0, aggregator)
    observer = updater.Updater({'value': obs}, pad_with_initial_value=True)
    observer.reset(fake_physics.FakePhysics(), random_state=None)
    result_spec = observer.observation_spec()['value']
    expected_type = specs.BoundedArray if declaration else specs.Array
    self.assertIs(type(result_spec), expected_type)
    result_spec.validate(observer.get_observation()['value'])

  @parameterized.parameters('min', 'max', 'mean', 'median', 'sum')
  def test_unbounded_input_does_not_gain_bounds(self, name):
    source = np.array([2.0, 3.0])
    obs = observable.Generic(lambda _: source, buffer_size=3, aggregator=name)
    obs.enabled = True
    observer = updater.Updater({'value': obs}, pad_with_initial_value=True)
    observer.reset(fake_physics.FakePhysics(), random_state=None)
    result_spec = observer.observation_spec()['value']
    self.assertIs(type(result_spec), specs.Array)
    result_spec.validate(observer.get_observation()['value'])

  def test_actual_physics_observations_keep_limits_after_mean_aggregation(self):
    xml = """<mujoco><option timestep="0.01" gravity="0 0 0"/>
      <worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>"""
    physics = mujoco.Physics.from_xml_string(xml)
    self.addCleanup(physics.free)
    physics.data.qpos[:] = 0.1
    physics.data.qvel[:] = 0.2
    physics.forward()
    obs = _BoundedObservable(physics.data.qpos, -1.0, 1.0, 'mean')
    observer = updater.Updater({'position': obs}, pad_with_initial_value=False)
    observer.reset(physics, random_state=None)
    result_spec = observer.observation_spec()['position']
    self.assertIsInstance(result_spec, specs.BoundedArray)
    for _ in range(5):
      observer.prepare_for_next_control_step()
      physics.step()
      observer.update()
      result_spec.validate(observer.get_observation()['position'])
    np.testing.assert_array_equal(result_spec.minimum, -1.0)
    np.testing.assert_array_equal(result_spec.maximum, 1.0)


if __name__ == '__main__':
  absltest.main()
