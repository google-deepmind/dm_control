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

"""Tests for scaled actuators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control import mjcf
from dm_control.locomotion.walkers import scaled_actuators
import numpy as np
from six.moves import range


class ScaledActuatorsTest(absltest.TestCase):

  def setUp(self):
    super(ScaledActuatorsTest, self).setUp()
    self._mjcf_model = mjcf.RootElement()
    self._min = -1.4
    self._max = 2.3
    self._gain = 1.7
    self._scaled_min = -0.8
    self._scaled_max = 1.3
    self._range = self._max - self._min
    self._scaled_range = self._scaled_max - self._scaled_min
    self._joints = []
    for _ in range(2):
      body = self._mjcf_model.worldbody.add('body')
      body.add('geom', type='sphere', size=[1])
      self._joints.append(body.add('joint', type='hinge'))
    self._scaled_actuator_joint = self._joints[0]
    self._standard_actuator_joint = self._joints[1]
    self._random_state = np.random.RandomState(3474)

  def _set_actuator_controls(self, physics, normalized_ctrl,
                             scaled_actuator=None, standard_actuator=None):
    if scaled_actuator is not None:
      physics.bind(scaled_actuator).ctrl = (
          normalized_ctrl * self._scaled_range + self._scaled_min)
    if standard_actuator is not None:
      physics.bind(standard_actuator).ctrl = (
          normalized_ctrl * self._range + self._min)

  def _assert_same_qfrc_actuator(self, physics, joint1, joint2):
    np.testing.assert_allclose(physics.bind(joint1).qfrc_actuator,
                               physics.bind(joint2).qfrc_actuator)

  def test_position_actuator(self):
    scaled_actuator = scaled_actuators.add_position_actuator(
        target=self._scaled_actuator_joint, kp=self._gain,
        qposrange=(self._min, self._max),
        ctrlrange=(self._scaled_min, self._scaled_max))
    standard_actuator = self._mjcf_model.actuator.add(
        'position', joint=self._standard_actuator_joint, kp=self._gain,
        ctrllimited=True, ctrlrange=(self._min, self._max))
    physics = mjcf.Physics.from_mjcf_model(self._mjcf_model)

    # Zero torque.
    physics.bind(self._scaled_actuator_joint).qpos = (
        0.2345 * self._range + self._min)
    self._set_actuator_controls(physics, 0.2345, scaled_actuator)
    np.testing.assert_allclose(
        physics.bind(self._scaled_actuator_joint).qfrc_actuator, 0, atol=1e-15)

    for _ in range(100):
      normalized_ctrl = self._random_state.uniform()
      physics.bind(self._joints).qpos = (
          self._random_state.uniform() * self._range + self._min)
      self._set_actuator_controls(physics, normalized_ctrl,
                                  scaled_actuator, standard_actuator)
      self._assert_same_qfrc_actuator(
          physics, self._scaled_actuator_joint, self._standard_actuator_joint)

  def test_velocity_actuator(self):
    scaled_actuator = scaled_actuators.add_velocity_actuator(
        target=self._scaled_actuator_joint, kv=self._gain,
        qvelrange=(self._min, self._max),
        ctrlrange=(self._scaled_min, self._scaled_max))
    standard_actuator = self._mjcf_model.actuator.add(
        'velocity', joint=self._standard_actuator_joint, kv=self._gain,
        ctrllimited=True, ctrlrange=(self._min, self._max))
    physics = mjcf.Physics.from_mjcf_model(self._mjcf_model)

    # Zero torque.
    physics.bind(self._scaled_actuator_joint).qvel = (
        0.5432 * self._range + self._min)
    self._set_actuator_controls(physics, 0.5432, scaled_actuator)
    np.testing.assert_allclose(
        physics.bind(self._scaled_actuator_joint).qfrc_actuator, 0, atol=1e-15)

    for _ in range(100):
      normalized_ctrl = self._random_state.uniform()
      physics.bind(self._joints).qvel = (
          self._random_state.uniform() * self._range + self._min)
      self._set_actuator_controls(physics, normalized_ctrl,
                                  scaled_actuator, standard_actuator)
      self._assert_same_qfrc_actuator(
          physics, self._scaled_actuator_joint, self._standard_actuator_joint)

  def test_invalid_kwargs(self):
    invalid_kwargs = dict(joint=self._scaled_actuator_joint, ctrllimited=False)
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        scaled_actuators._GOT_INVALID_KWARGS.format(sorted(invalid_kwargs))):
      scaled_actuators.add_position_actuator(
          target=self._scaled_actuator_joint,
          qposrange=(self._min, self._max),
          **invalid_kwargs)

  def test_invalid_target(self):
    invalid_target = self._mjcf_model.worldbody
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        scaled_actuators._GOT_INVALID_TARGET.format(invalid_target)):
      scaled_actuators.add_position_actuator(
          target=invalid_target, qposrange=(self._min, self._max))


if __name__ == '__main__':
  absltest.main()
