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

"""Tests for observation.obs_buffer."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.composer.observation import obs_buffer
import numpy as np


def _generate_constant_schedule(update_timestep, delay,
                                control_timestep, n_observed_steps):
  first = update_timestep
  last = control_timestep * n_observed_steps + 1
  return [(i, delay) for i in range(first, last, update_timestep)]


class BufferTest(parameterized.TestCase):

  def testOutOfOrderArrival(self):
    buf = obs_buffer.Buffer(buffer_size=3, shape=(), dtype=np.float)
    buf.insert(timestamp=0, delay=4, value=1)
    buf.insert(timestamp=1, delay=2, value=2)
    buf.insert(timestamp=2, delay=3, value=3)
    np.testing.assert_array_equal(buf.read(current_time=2), [0., 0., 0.])
    np.testing.assert_array_equal(buf.read(current_time=3), [0., 0., 2.])
    np.testing.assert_array_equal(buf.read(current_time=4), [0., 2., 1.])
    np.testing.assert_array_equal(buf.read(current_time=5), [2., 1., 3.])
    np.testing.assert_array_equal(buf.read(current_time=6), [2., 1., 3.])

  @parameterized.parameters(((3, 3),), ((),))
  def testStripSingletonDimension(self, shape):
    buf = obs_buffer.Buffer(buffer_size=1, shape=shape, dtype=np.float,
                            strip_singleton_buffer_dim=True)
    expected_value = np.full(shape, 42, dtype=np.float)
    buf.insert(timestamp=0, delay=0, value=expected_value)
    np.testing.assert_array_equal(buf.read(current_time=1), expected_value)

  def testPlanToSingleUndelayedObservation(self):
    buf = obs_buffer.Buffer(
        buffer_size=1, shape=(), dtype=np.float)
    control_timestep = 20
    observation_schedule = _generate_constant_schedule(
        update_timestep=1, delay=0,
        control_timestep=control_timestep, n_observed_steps=1)
    buf.drop_unobserved_upcoming_items(
        observation_schedule, read_interval=control_timestep)
    self.assertEqual(observation_schedule, [(20, 0)])

  def testPlanTwoStepsAhead(self):
    buf = obs_buffer.Buffer(
        buffer_size=1, shape=(), dtype=np.float)
    control_timestep = 5
    observation_schedule = _generate_constant_schedule(
        update_timestep=2, delay=3,
        control_timestep=control_timestep, n_observed_steps=2)
    buf.drop_unobserved_upcoming_items(
        observation_schedule, read_interval=control_timestep)
    self.assertEqual(observation_schedule, [(2, 3), (6, 3), (10, 3)])


if __name__ == '__main__':
  absltest.main()
