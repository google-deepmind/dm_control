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

"""Declared observation specs must agree with flattened environment output."""

import collections
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mujoco
from dm_control.rl import control
from dm_env import specs
import numpy as np


class DeclaredSpecTask(control.Task):
  """Small task exercised with actual MuJoCo stepping."""

  def initialize_episode(self, physics):
    physics.data.qpos[:] = 0.2

  def before_step(self, action, physics):
    physics.set_control(action)

  def action_spec(self, physics):
    return specs.BoundedArray((1,), np.float64, -1.0, 1.0)

  def get_observation(self, physics):
    return collections.OrderedDict(
        position=physics.data.qpos.copy(),
        velocity=physics.data.qvel.copy(),
        time=np.asarray(physics.time()),
    )

  def observation_spec(self, physics):
    return collections.OrderedDict(
        position=specs.Array((1,), np.float64),
        velocity=specs.Array((1,), np.float64),
        time=specs.Array((), np.float64),
    )

  def get_reward(self, physics):
    return -float(physics.data.qpos[0] ** 2)


class FlatObservationSpecTest(parameterized.TestCase):

  @parameterized.product(
      declared=(False, True),
      ordered=(False, True),
      dtype_pair=(
          (np.float32, np.float32),
          (np.float32, np.int32),
          (np.float64, bool),
      ),
  )
  def test_spec_matches_reset_and_step(self, declared, ordered, dtype_pair):
    mapping = collections.OrderedDict if ordered else dict
    observation = mapping(
        [
            ('z', np.arange(6, dtype=dtype_pair[0]).reshape(2, 3)),
            ('a', np.asarray(1, dtype=dtype_pair[1])),
            ('empty', np.empty((0, 2), dtype=dtype_pair[0])),
        ]
    )
    declared_spec = mapping(
        (key, specs.Array(value.shape, value.dtype))
        for key, value in observation.items()
    )
    physics = mock.Mock(spec=control.Physics)
    physics.reset_context = mock.MagicMock()
    task = mock.Mock(spec=control.Task)
    task.get_observation.return_value = observation
    task.get_reward.return_value = 0.0
    task.get_termination.return_value = None
    if declared:
      task.observation_spec.return_value = declared_spec
    else:
      task.observation_spec.side_effect = NotImplementedError()
    env = control.Environment(physics, task, flat_observation=True)
    actual_spec = env.observation_spec()
    self.assertEqual(set(actual_spec), {control.FLAT_OBSERVATION_KEY})
    self.assertIsInstance(
        actual_spec, mapping if declared else collections.OrderedDict
    )
    if declared:
      task.get_observation.assert_not_called()
    expected = control.flatten_observation(observation)[
        control.FLAT_OBSERVATION_KEY
    ]
    for timestep in (env.reset(), env.step([0.0])):
      actual = timestep.observation[control.FLAT_OBSERVATION_KEY]
      actual_spec[control.FLAT_OBSERVATION_KEY].validate(actual)
      np.testing.assert_array_equal(actual, expected)
    self.assertEqual(
        actual_spec[control.FLAT_OBSERVATION_KEY].name,
        control.FLAT_OBSERVATION_KEY,
    )
    self.assertEqual(set(declared_spec), {'z', 'a', 'empty'})
    self.assertEqual(declared_spec['z'].shape, (2, 3))

  def test_declared_specs_do_not_materialize_observation_arrays(self):
    physics = mock.Mock(spec=control.Physics)
    task = mock.Mock(spec=control.Task)
    # Computing the spec must not allocate this image or invoke the task.
    task.observation_spec.return_value = {
        'image': specs.Array((100_000, 100_000, 3), np.uint8),
        'scalar': specs.Array((), np.float32),
    }
    task.get_observation.side_effect = AssertionError('unexpected observation')
    env = control.Environment(physics, task, flat_observation=True)
    flat = env.observation_spec()[control.FLAT_OBSERVATION_KEY]
    self.assertEqual(flat.shape, (30_000_000_001,))
    self.assertEqual(flat.dtype, np.result_type(np.uint8, np.float32))
    task.get_observation.assert_not_called()

  def test_unflattened_declared_spec_is_returned_unchanged(self):
    task = mock.Mock(spec=control.Task)
    declared = {'bounded': specs.BoundedArray((2,), np.float32, -1.0, 1.0)}
    task.observation_spec.return_value = declared
    env = control.Environment(mock.Mock(spec=control.Physics), task)
    self.assertIs(env.observation_spec(), declared)
    task.get_observation.assert_not_called()

  def test_actual_physics_reset_and_steps_match_the_declared_flat_spec(self):
    physics = mujoco.Physics.from_xml_string("""
      <mujoco><option timestep="0.01"/>
        <worldbody><body><joint name="hinge" type="hinge"/>
          <geom type="sphere" size="0.1" pos="0 0 -0.2"/>
        </body></worldbody><actuator><motor joint="hinge"/></actuator>
      </mujoco>""")
    self.addCleanup(physics.free)
    env = control.Environment(
        physics, DeclaredSpecTask(), time_limit=0.03, flat_observation=True
    )
    spec = env.observation_spec()[control.FLAT_OBSERVATION_KEY]
    timesteps = [env.reset()] + [env.step([0.1]) for _ in range(3)]
    self.assertTrue(timesteps[0].first())
    self.assertTrue(timesteps[-1].last())
    for timestep in timesteps:
      spec.validate(timestep.observation[control.FLAT_OBSERVATION_KEY])
    np.testing.assert_allclose(
        timesteps[-1].observation[control.FLAT_OBSERVATION_KEY],
        [physics.data.qpos[0], physics.data.qvel[0], physics.time()],
    )
    self.assertGreater(physics.time(), 0.0)


if __name__ == '__main__':
  absltest.main()
