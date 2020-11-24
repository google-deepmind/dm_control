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

"""Tests for the CMU humanoid."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base
from dm_control.locomotion.walkers import cmu_humanoid
import numpy as np
from six.moves import range
from six.moves import zip


class CMUHumanoidTest(parameterized.TestCase):

  @parameterized.parameters([
      cmu_humanoid.CMUHumanoid,
      cmu_humanoid.CMUHumanoidPositionControlled,
  ])
  def test_can_compile_and_step_simulation(self, walker_type):
    walker = walker_type()
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    for _ in range(100):
      physics.step()

  @parameterized.parameters([
      cmu_humanoid.CMUHumanoid,
      cmu_humanoid.CMUHumanoidPositionControlled,
  ])
  def test_actuators_sorted_alphabetically(self, walker_type):
    walker = walker_type()
    actuator_names = [
        actuator.name for actuator in walker.mjcf_model.find_all('actuator')]
    np.testing.assert_array_equal(actuator_names, sorted(actuator_names))

  def test_actuator_to_mocap_joint_mapping(self):
    walker = cmu_humanoid.CMUHumanoid()

    with self.subTest('Forward mapping'):
      for actuator_num, cmu_mocap_joint_num in enumerate(walker.actuator_order):
        self.assertEqual(walker.actuator_to_joint_order[cmu_mocap_joint_num],
                         actuator_num)

    with self.subTest('Inverse mapping'):
      for cmu_mocap_joint_num, actuator_num in enumerate(
          walker.actuator_to_joint_order):
        self.assertEqual(walker.actuator_order[actuator_num],
                         cmu_mocap_joint_num)

  def test_cmu_humanoid_position_controlled_has_correct_actuators(self):
    walker_torque = cmu_humanoid.CMUHumanoid()
    walker_pos = cmu_humanoid.CMUHumanoidPositionControlled()

    actuators_torque = walker_torque.mjcf_model.find_all('actuator')
    actuators_pos = walker_pos.mjcf_model.find_all('actuator')

    actuator_pos_params = {
        params.name: params for params in cmu_humanoid._POSITION_ACTUATORS}

    self.assertEqual(len(actuators_torque), len(actuators_pos))

    for actuator_torque, actuator_pos in zip(actuators_torque, actuators_pos):
      self.assertEqual(actuator_pos.name, actuator_torque.name)
      self.assertEqual(actuator_pos.joint.full_identifier,
                       actuator_torque.joint.full_identifier)
      self.assertEqual(actuator_pos.tag, 'general')
      self.assertEqual(actuator_pos.ctrllimited, 'true')
      np.testing.assert_array_equal(actuator_pos.ctrlrange, (-1, 1))

      expected_params = actuator_pos_params[actuator_pos.name]
      self.assertEqual(actuator_pos.biasprm[1], -expected_params.kp)
      np.testing.assert_array_equal(actuator_pos.forcerange,
                                    expected_params.forcerange)

  @parameterized.parameters([
      'body_camera',
      'egocentric_camera',
      'head',
      'left_arm_root',
      'right_arm_root',
      'root_body',
  ])
  def test_get_element_property(self, name):
    attribute_value = getattr(cmu_humanoid.CMUHumanoid(), name)
    self.assertIsInstance(attribute_value, mjcf.Element)

  @parameterized.parameters([
      'actuators',
      'bodies',
      'end_effectors',
      'marker_geoms',
      'mocap_joints',
      'observable_joints',
  ])
  def test_get_element_tuple_property(self, name):
    attribute_value = getattr(cmu_humanoid.CMUHumanoid(), name)
    self.assertNotEmpty(attribute_value)
    for item in attribute_value:
      self.assertIsInstance(item, mjcf.Element)

  def test_set_name(self):
    name = 'fred'
    walker = cmu_humanoid.CMUHumanoid(name=name)
    self.assertEqual(walker.mjcf_model.model, name)

  def test_set_marker_rgba(self):
    marker_rgba = (1., 0., 1., 0.5)
    walker = cmu_humanoid.CMUHumanoid(marker_rgba=marker_rgba)
    for marker_geom in walker.marker_geoms:
      np.testing.assert_array_equal(marker_geom.rgba, marker_rgba)

  @parameterized.parameters(
      'actuator_activation',
      'appendages_pos',
      'body_camera',
      'head_height',
      'sensors_torque',
  )
  def test_evaluate_observable(self, name):
    walker = cmu_humanoid.CMUHumanoid()
    observable = getattr(walker.observables, name)
    physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model)
    observation = observable(physics)
    self.assertIsInstance(observation, (float, np.ndarray))

  def test_proprioception(self):
    walker = cmu_humanoid.CMUHumanoid()
    for item in walker.observables.proprioception:
      self.assertIsInstance(item, observable_base.Observable)

  def test_cmu_pose_to_actuation(self):
    walker = cmu_humanoid.CMUHumanoidPositionControlled()
    random_state = np.random.RandomState(123)

    expected_actuation = random_state.uniform(-1, 1, len(walker.actuator_order))

    cmu_limits = zip(*(joint.range for joint in walker.mocap_joints))
    cmu_lower, cmu_upper = (np.array(limit) for limit in cmu_limits)
    cmu_pose = cmu_lower + (cmu_upper - cmu_lower) * (
        1 + expected_actuation[walker.actuator_to_joint_order]) / 2

    actual_actuation = walker.cmu_pose_to_actuation(cmu_pose)

    np.testing.assert_allclose(actual_actuation, expected_actuation)


if __name__ == '__main__':
  absltest.main()
