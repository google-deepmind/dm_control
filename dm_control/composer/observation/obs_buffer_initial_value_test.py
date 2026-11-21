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
"""Regression tests for independent initial-value observation padding."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.composer.observation import obs_buffer
from dm_control.composer.observation import observable
from dm_control.composer.observation import updater
import numpy as np


class InitialPaddingSnapshotTest(parameterized.TestCase):

  @parameterized.product(
      delay=(0, 2), buffer_size=(2, 4), dtype=(np.float32, np.int32, np.bool_)
  )
  def test_mutating_input_cannot_change_initial_padding(
      self, delay, buffer_size, dtype
  ):
    source = np.array([0, 1], dtype=dtype)
    initial = source.copy()
    buffer = obs_buffer.Buffer(
        buffer_size, source.shape, source.dtype, pad_with_initial_value=True
    )
    buffer.insert(timestamp=0, delay=delay, value=source)
    source[:] = 1 - source
    expected = np.repeat(initial[None, :], buffer_size, axis=0)
    for timestamp in (0, 1, 2):
      np.testing.assert_array_equal(buffer.read(timestamp), expected)
    returned = buffer.read(2)
    returned[:] = 9
    np.testing.assert_array_equal(buffer.read(2), expected)
    np.testing.assert_array_equal(source, 1 - initial)

  @parameterized.parameters('list', 'scalar', 'strided', 'fortran')
  def test_delayed_initial_value_has_an_independent_array_snapshot(self, kind):
    if kind == 'list':
      source = [2.0, 3.0]
    elif kind == 'scalar':
      source = 2.0
    elif kind == 'strided':
      source = np.arange(8, dtype=float)[::2]
    else:
      source = np.asfortranarray(np.arange(6, dtype=float).reshape(2, 3))
    expected = np.array(source)
    buffer = obs_buffer.Buffer(
        1,
        expected.shape,
        expected.dtype,
        pad_with_initial_value=True,
        strip_singleton_buffer_dim=True,
    )
    buffer.insert(timestamp=0, delay=2, value=source)
    if isinstance(source, list):
      source[:] = [-1.0] * len(source)
    elif isinstance(source, np.ndarray):
      source[...] = -1.0
    actual = buffer.read(0)
    self.assertIsInstance(actual, np.ndarray)
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.dtype, expected.dtype)
    np.testing.assert_array_equal(actual, expected)
    actual[...] = -5.0
    np.testing.assert_array_equal(buffer.read(1), expected)
    np.testing.assert_array_equal(buffer.read(2), expected)

  @parameterized.parameters(0, 2)
  def test_reused_source_keeps_old_padding_and_observation_values(self, delay):
    source = np.array([3.0, 7.0])
    buffer = obs_buffer.Buffer(3, (2,), float, pad_with_initial_value=True)
    for timestamp, value in enumerate(([3.0, 7.0], [11.0, 13.0], [17.0, 19.0])):
      source[:] = value
      buffer.insert(timestamp, delay, source)
    source[:] = -1.0
    expected = np.array([[3.0, 7.0], [11.0, 13.0], [17.0, 19.0]])
    if delay:
      np.testing.assert_array_equal(
          buffer.read(2), np.repeat(expected[:1], 3, 0)
      )
    np.testing.assert_array_equal(buffer.read(2 + delay), expected)

  def test_zero_padding_and_existing_entries_keep_their_behavior(self):
    source = np.array([3.0, 7.0])
    buffer = obs_buffer.Buffer(3, (2,), float, pad_with_initial_value=False)
    buffer.insert(0, 2, source)
    source[:] = 99.0
    np.testing.assert_array_equal(buffer.read(0), np.zeros((3, 2)))
    np.testing.assert_array_equal(
        buffer.read(2), [[0.0, 0.0], [0.0, 0.0], [3.0, 7.0]]
    )

  @parameterized.parameters(None, 'mean')
  def test_real_physics_updater_preserves_initial_state_while_padding(
      self, aggregator
  ):
    xml = """<mujoco><option timestep="0.01" gravity="0 0 0"/>
      <worldbody><body><joint type="slide" axis="1 0 0"/>
      <geom type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>"""
    physics = mujoco.Physics.from_xml_string(xml)
    self.addCleanup(physics.free)
    physics.data.qpos[:] = 2.0
    physics.data.qvel[:] = 1.0
    physics.forward()
    obs = observable.Generic(
        lambda p: p.data.qpos, buffer_size=3, aggregator=aggregator
    )
    obs.enabled = True
    observer = updater.Updater({'position': obs}, pad_with_initial_value=True)
    observer.reset(physics, random_state=None)
    initial = physics.data.qpos.copy()
    samples = [initial.copy()] * 3
    for _ in range(4):
      observer.prepare_for_next_control_step()
      physics.step()
      samples = samples[1:] + [physics.data.qpos.copy()]
      observer.update()
      expected = np.stack(samples)
      if aggregator:
        expected = np.mean(expected, axis=0)
      actual = observer.get_observation()['position']
      np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
      observer.observation_spec()['position'].validate(actual)
    self.assertGreater(physics.data.qpos[0], initial[0])


if __name__ == '__main__':
  absltest.main()
