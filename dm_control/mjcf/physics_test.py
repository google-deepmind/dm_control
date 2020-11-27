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

"""Tests for `dm_control.mjcf.physics`."""

import copy
import os
import pickle

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.mjcf import physics as mjcf_physics
from dm_control.mujoco.wrapper import mjbindings
import mock
import numpy as np

mjlib = mjbindings.mjlib

ARM_MODEL = os.path.join(os.path.dirname(__file__), 'test_assets/robot_arm.xml')


class PhysicsTest(parameterized.TestCase):
  """Tests for `mjcf.Physics`."""

  def setUp(self):
    super().setUp()
    self.model = mjcf.from_path(ARM_MODEL)
    self.physics = mjcf.Physics.from_xml_string(
        self.model.to_xml_string(), assets=self.model.get_assets())
    self.random = np.random.RandomState(0)

  def sample_elements(self, namespace, single_element):
    all_elements = self.model.find_all(namespace)
    if single_element:
      # A single randomly chosen element from this namespace.
      elements = self.random.choice(all_elements)
      full_identifiers = elements.full_identifier
    else:
      # A random permutation of all elements in this namespace.
      elements = self.random.permutation(all_elements)
      full_identifiers = [element.full_identifier for element in elements]
    return elements, full_identifiers

  def test_construct_and_reload_from_mjcf_model(self):
    physics = mjcf.Physics.from_mjcf_model(self.model)
    physics.data.time = 1.
    physics.reload_from_mjcf_model(self.model)
    self.assertEqual(physics.data.time, 0.)

  @parameterized.parameters(
      # namespace, single_element
      ('geom', True),
      ('geom', False),
      ('joint', True),
      ('joint', False))
  def test_id(self, namespace, single_element):
    elements, full_identifiers = self.sample_elements(namespace, single_element)
    actual = self.physics.bind(elements).element_id
    if single_element:
      expected = self.physics.model.name2id(full_identifiers, namespace)
    else:
      expected = [self.physics.model.name2id(name, namespace)
                  for name in full_identifiers]
    np.testing.assert_array_equal(expected, actual)

  def assertCanGetAndSetBindingArray(
      self, binding, attribute_name, named_indexer, full_identifiers):

    # Read the values using the binding attribute.
    actual = getattr(binding, attribute_name)

    # Read them using the normal named indexing machinery.
    expected = named_indexer[full_identifiers]
    np.testing.assert_array_equal(expected, actual)

    # Assign an array of unique values to the attribute.
    expected = np.arange(actual.size, dtype=actual.dtype).reshape(actual.shape)
    setattr(binding, attribute_name, expected)

    # Read the values back using the normal named indexing machinery.
    actual = named_indexer[full_identifiers]
    np.testing.assert_array_equal(expected, actual)

  @parameterized.parameters(
      # namespace, attribute_name, model_or_data, field_name, single_element
      ('geom', 'xpos', 'data', 'geom_xpos', True),
      ('geom', 'xpos', 'data', 'geom_xpos', False),
      ('joint', 'qpos', 'data', 'qpos', True),
      ('joint', 'qpos', 'data', 'qpos', False),
      ('site', 'rgba', 'model', 'site_rgba', True),
      ('site', 'rgba', 'model', 'site_rgba', False),
      ('sensor', 'sensordata', 'data', 'sensordata', True),
      ('sensor', 'sensordata', 'data', 'sensordata', False))
  def test_attribute_access(self, namespace, attribute_name, model_or_data,
                            field_name, single_element):

    elements, full_identifiers = self.sample_elements(namespace, single_element)
    named_indexer = getattr(getattr(self.physics.named, model_or_data),
                            field_name)
    binding = self.physics.bind(elements)
    self.assertCanGetAndSetBindingArray(
        binding, attribute_name, named_indexer, full_identifiers)

  @parameterized.parameters(
      # namespace, attribute_name, model_or_data, field_name, single_element,
      # column_index
      ('geom', 'pos', 'model', 'geom_pos', True, None),
      ('geom', 'pos', 'model', 'geom_pos', False, None),
      ('geom', 'pos', 'model', 'geom_pos', True, 1),
      ('geom', 'pos', 'model', 'geom_pos', False, 1),
      ('geom', 'pos', 'model', 'geom_pos', True, 'y'),
      ('geom', 'pos', 'model', 'geom_pos', False, 'y'),
      ('geom', 'pos', 'model', 'geom_pos', True, slice(0, None, 2)),
      ('geom', 'pos', 'model', 'geom_pos', False, slice(0, None, 2)),
      ('geom', 'pos', 'model', 'geom_pos', True, [0, 2]),
      ('geom', 'pos', 'model', 'geom_pos', False, [0, 2]),
      ('geom', 'pos', 'model', 'geom_pos', True, ['x', 'z']),
      ('geom', 'pos', 'model', 'geom_pos', False, ['x', 'z']),
      ('joint', 'qpos', 'data', 'qpos', True, None),
      ('joint', 'qpos', 'data', 'qpos', False, None))
  def test_indexing(self, namespace, attribute_name, model_or_data,
                    field_name, single_element, column_index):

    elements, full_identifiers = self.sample_elements(namespace, single_element)
    named_indexer = getattr(getattr(self.physics.named, model_or_data),
                            field_name)
    binding = self.physics.bind(elements)

    if column_index is not None:
      binding_index = (attribute_name, column_index)
      try:
        named_index = np.ix_(full_identifiers, column_index)
      except ValueError:
        named_index = (full_identifiers, column_index)
    else:
      binding_index = attribute_name
      named_index = full_identifiers

    # Read the values by indexing the binding.
    actual = binding[binding_index]

    # Read them using the normal named indexing machinery.
    expected = named_indexer[named_index]
    np.testing.assert_array_equal(expected, actual)

    # Write an array of unique values into the binding.
    expected = np.arange(actual.size, dtype=actual.dtype).reshape(actual.shape)
    binding[binding_index] = expected

    # Read the values back using the normal named indexing machinery.
    actual = named_indexer[named_index]
    np.testing.assert_array_equal(expected, actual)

  def test_bind_mocap_body(self):
    pos = [1, 2, 3]
    quat = [1, 0, 0, 0]
    model = mjcf.RootElement()
    # Bodies are non-mocap by default.
    normal_body = model.worldbody.add('body', pos=pos, quat=quat)
    mocap_body = model.worldbody.add('body', pos=pos, quat=quat, mocap='true')
    physics = mjcf.Physics.from_xml_string(model.to_xml_string())

    binding = physics.bind(mocap_body)
    np.testing.assert_array_equal(pos, binding.mocap_pos)
    np.testing.assert_array_equal(quat, binding.mocap_quat)

    new_pos = [4, 5, 6]
    new_quat = [0, 1, 0, 0]
    binding.mocap_pos = new_pos
    binding.mocap_quat = new_quat
    np.testing.assert_array_equal(
        new_pos, physics.named.data.mocap_pos[mocap_body.full_identifier])
    np.testing.assert_array_equal(
        new_quat, physics.named.data.mocap_quat[mocap_body.full_identifier])

    with self.assertRaises(AttributeError):
      _ = physics.bind(normal_body).mocap_pos

    with self.assertRaisesRegex(
        ValueError,
        'Cannot bind to a collection containing multiple element types'):
      physics.bind([mocap_body, normal_body])

  def test_bind_worldbody(self):
    expected_mass = 10
    model = mjcf.RootElement()
    child = model.worldbody.add('body')
    child.add('geom', type='sphere', size=[0.1], mass=expected_mass)
    physics = mjcf.Physics.from_mjcf_model(model)
    mass = physics.bind(model.worldbody).subtreemass
    self.assertEqual(mass, expected_mass)

  def test_caching(self):
    all_joints = self.model.find_all('joint')

    original = self.physics.bind(all_joints)
    cached = self.physics.bind(all_joints)
    self.assertIs(cached, original)

    different_order = self.physics.bind(all_joints[::-1])
    self.assertIsNot(different_order, original)

    # Reloading the `Physics` instance should clear the cache.
    self.physics.reload_from_xml_string(
        self.model.to_xml_string(), assets=self.model.get_assets())
    after_reload = self.physics.bind(all_joints)
    self.assertIsNot(after_reload, original)

  def test_exceptions(self):
    joint = self.model.find_all('joint')[0]
    geom = self.model.find_all('geom')[0]
    with self.assertRaisesRegex(
        ValueError,
        'Cannot bind to a collection containing multiple element types'):
      self.physics.bind([joint, geom])

    with self.assertRaisesRegex(ValueError, 'cannot be bound to physics'):
      mjcf.physics.Binding(self.physics, 'invalid_namespace', 'whatever')

    binding = self.physics.bind(joint)
    with self.assertRaisesRegex(AttributeError, 'does not have attribute'):
      getattr(binding, 'invalid_attribute')

  def test_dirty(self):
    self.physics.forward()
    self.assertFalse(self.physics.is_dirty)

    joints, _ = self.sample_elements('joint', single_element=False)
    sites, _ = self.sample_elements('site', single_element=False)

    # Accessing qpos shouldn't trigger a recalculation.
    _ = self.physics.bind(joints).qpos
    self.assertFalse(self.physics.is_dirty)

    # Reassignments to qpos should cause the physics to become dirty.
    site_xpos_before = copy.deepcopy(self.physics.bind(sites).xpos)
    self.physics.bind(joints).qpos += 0.5
    self.assertTrue(self.physics.is_dirty)

    # Accessing stuff in mjModel shouldn't trigger a recalculation.
    _ = self.physics.bind(sites).pos
    self.assertTrue(self.physics.is_dirty)

    # Accessing stuff in mjData should trigger a recalculation.
    actual_sites_xpos_after = copy.deepcopy(self.physics.bind(sites).xpos)
    self.assertFalse(self.physics.is_dirty)
    self.assertFalse((actual_sites_xpos_after == site_xpos_before).all())

    # Automatic recalculation should render `forward` a no-op here.
    self.physics.forward()
    expected_sites_xpos_after = self.physics.bind(sites).xpos
    np.testing.assert_array_equal(actual_sites_xpos_after,
                                  expected_sites_xpos_after)

    # `forward` should not be called on subsequent queries to xpos.
    with mock.patch.object(
        self.physics, 'forward',
        side_effect=self.physics.forward) as mock_forward:
      _ = self.physics.bind(sites).xpos
      mock_forward.assert_not_called()

  @parameterized.parameters(True, False)
  def test_assign_while_dirty(self, assign_via_slice):
    actuators = self.model.find_all('actuator')
    if assign_via_slice:
      self.physics.bind(actuators).ctrl[:] = 0.75
    else:
      self.physics.bind(actuators).ctrl = 0.75
    self.assertTrue(self.physics.is_dirty)
    self.physics.step()
    self.assertTrue(self.physics.is_dirty)
    sensors = self.model.find_all('sensor')
    if assign_via_slice:
      self.physics.bind(sensors).sensordata[:] = 12345
    else:
      self.physics.bind(sensors).sensordata = 12345
    self.assertFalse(self.physics.is_dirty)
    np.testing.assert_array_equal(
        self.physics.bind(sensors).sensordata,
        [12345] * len(self.physics.bind(sensors).sensordata))

  def test_setitem_on_binding_attr(self):
    bodies, _ = self.sample_elements('body', single_element=False)
    xfrc_binding = self.physics.bind(bodies).xfrc_applied

    xfrc_binding[:, 1] = list(range(len(bodies)))
    for i, body in enumerate(bodies):
      self.assertEqual(xfrc_binding[i, 1], i)
      self.assertEqual(
          self.physics.named.data.xfrc_applied[body.full_identifier][1], i)

    xfrc_binding[:, 1] *= 2
    for i, body in enumerate(bodies):
      self.assertEqual(xfrc_binding[i, 1], 2 * i)
      self.assertEqual(
          self.physics.named.data.xfrc_applied[body.full_identifier][1], 2 * i)

    xfrc_binding[[1, 3, 5], 2] = 42
    for i, body in enumerate(bodies):
      actual_value = (
          self.physics.named.data.xfrc_applied[body.full_identifier][2])
      if i in [1, 3, 5]:
        self.assertEqual(actual_value, 42)
      else:
        self.assertNotEqual(actual_value, 42)

    # Bind to a single element.
    single_binding = self.physics.bind(bodies[0]).xfrc_applied
    single_binding[:2] = 55
    np.testing.assert_array_equal(single_binding[:2], [55, 55])
    np.testing.assert_array_equal(
        self.physics.named.data.xfrc_applied[bodies[0].full_identifier][:2],
        [55, 55])

  def test_empty_binding(self):
    binding = self.physics.bind([])
    self.assertEqual(binding.xpos.shape, (0,))
    with self.assertRaisesWithLiteralMatch(
        ValueError, 'Cannot assign a value to an empty binding.'):
      binding.xpos = 5

  @parameterized.parameters([('data', 'act'), ('data', 'act_dot')])
  def test_actuator_state_binding(self, model_or_data, attribute_name):

    def make_model_with_mixed_actuators():
      actuators = []
      is_stateful = []
      root = mjcf.RootElement()
      body = root.worldbody.add('body')
      body.add('geom', type='sphere', size=[0.1])
      slider = body.add('joint', type='slide', name='slide_joint')
      # Third-order `general` actuator.
      actuators.append(
          root.actuator.add(
              'general', dyntype='integrator', biastype='affine',
              dynprm=[1, 0, 0], joint=slider, name='general_act'))
      is_stateful.append(True)
      # Cylinder actuators are also third-order.
      actuators.append(
          root.actuator.add('cylinder', joint=slider, name='cylinder_act'))
      is_stateful.append(True)
      # A second-order actuator, added after the third-order actuators. The
      # actuators will be automatically reordered in the generated XML so that
      # the second-order actuator comes first.
      actuators.append(
          root.actuator.add('velocity', joint=slider, name='velocity_act'))
      is_stateful.append(False)
      return root, actuators, is_stateful

    model, actuators, is_stateful = make_model_with_mixed_actuators()
    physics = mjcf.Physics.from_mjcf_model(model)
    binding = physics.bind(actuators)
    named_indexer = getattr(
        getattr(physics.named, model_or_data), attribute_name)
    stateful_actuator_names = [
        actuator.full_identifier
        for actuator, stateful in zip(actuators, is_stateful) if stateful]
    self.assertCanGetAndSetBindingArray(
        binding, attribute_name, named_indexer, stateful_actuator_names)

  def test_bind_stateless_actuators_only(self):
    actuators = []
    root = mjcf.RootElement()
    body = root.worldbody.add('body')
    body.add('geom', type='sphere', size=[0.1])
    slider = body.add('joint', type='slide', name='slide_joint')
    actuators.append(
        root.actuator.add('velocity', joint=slider, name='velocity_act'))
    actuators.append(
        root.actuator.add('motor', joint=slider, name='motor_act'))
    # `act` should be an empty array if there are no stateful actuators.
    physics = mjcf.Physics.from_mjcf_model(root)
    self.assertEqual(physics.bind(actuators).act.shape, (0,))

  def make_simple_model(self):
    def add_submodel(root):
      body = root.worldbody.add('body')
      geom = body.add('geom', type='ellipsoid', size=[0.1, 0.2, 0.3])
      site = body.add('site', type='sphere', size=[0.1])
      return body, geom, site
    root = mjcf.RootElement()
    add_submodel(root)
    add_submodel(root)
    return root

  def quat2mat(self, quat):
    result = np.empty(9, dtype=np.double)
    mjlib.mju_quat2Mat(result, np.asarray(quat))
    return result

  @parameterized.parameters(['body', 'geom', 'site'])
  def test_write_to_pos(self, entity_type):
    root = self.make_simple_model()
    entity1, entity2 = root.find_all(entity_type)
    physics = mjcf.Physics.from_mjcf_model(root)
    first = physics.bind(entity1)
    second = physics.bind(entity2)

    # Initially both entities should be 'sameframe'
    self.assertEqual(first.sameframe, 1)
    self.assertEqual(second.sameframe, 1)

    # Assigning to `pos` should disable 'sameframe' only for that entity.
    new_pos = (0., 0., 0.1)
    first.pos = new_pos
    self.assertEqual(first.sameframe, 0)
    self.assertEqual(second.sameframe, 1)
    # `xpos` should reflect the new position.
    np.testing.assert_array_equal(first.xpos, new_pos)

    # Writing into the `pos` array should also disable 'sameframe'.
    new_x = -0.1
    pos_array = second.pos
    pos_array[0] = new_x
    self.assertEqual(second.sameframe, 0)
    # `xpos` should reflect the new position.
    self.assertEqual(second.xpos[0], new_x)

  @parameterized.parameters(['body', 'geom', 'site'])
  def test_write_to_quat(self, entity_type):
    root = self.make_simple_model()
    entity1, entity2 = root.find_all(entity_type)
    physics = mjcf.Physics.from_mjcf_model(root)
    first = physics.bind(entity1)
    second = physics.bind(entity2)

    # Initially both entities should be 'sameframe'
    self.assertEqual(first.sameframe, 1)
    self.assertEqual(second.sameframe, 1)

    # Assigning to `quat` should disable 'sameframe' only for that entity.
    new_quat = (0., 0., 0., 1.)
    first.quat = new_quat
    self.assertEqual(first.sameframe, 0)
    self.assertEqual(second.sameframe, 1)
    # `xmat` should reflect the new quaternion.
    np.testing.assert_allclose(first.xmat, self.quat2mat(new_quat))

    # Writing into the `quat` array should also disable 'sameframe'.
    new_w = -1.
    quat_array = second.quat
    quat_array[0] = new_w
    self.assertEqual(second.sameframe, 0)
    # `xmat` should reflect the new quaternion.
    np.testing.assert_allclose(second.xmat, self.quat2mat(quat_array))

  def test_write_to_ipos(self):
    root = self.make_simple_model()
    entity1, entity2 = root.find_all('body')
    physics = mjcf.Physics.from_mjcf_model(root)
    first = physics.bind(entity1)
    second = physics.bind(entity2)

    # Initially both bodies should be 'simple' and 'sameframe'
    self.assertEqual(first.simple, 1)
    self.assertEqual(first.sameframe, 1)
    self.assertEqual(second.simple, 1)
    self.assertEqual(second.sameframe, 1)

    # Assigning to `ipos` should disable 'simple' and 'sameframe' only for that
    # body.
    new_ipos = (0., 0., 0.1)
    first.ipos = new_ipos
    self.assertEqual(first.simple, 0)
    self.assertEqual(first.sameframe, 0)
    self.assertEqual(second.simple, 1)
    self.assertEqual(second.sameframe, 1)
    # `xipos` should reflect the new position.
    np.testing.assert_array_equal(first.xipos, new_ipos)

    # Writing into the `ipos` array should also disable 'simple' and
    # 'sameframe'.
    new_x = -0.1
    ipos_array = second.ipos
    ipos_array[0] = new_x
    self.assertEqual(second.simple, 0)
    self.assertEqual(second.sameframe, 0)
    # `xipos` should reflect the new position.
    self.assertEqual(second.xipos[0], new_x)

  def test_write_to_iquat(self):
    root = self.make_simple_model()
    entity1, entity2 = root.find_all('body')
    physics = mjcf.Physics.from_mjcf_model(root)
    first = physics.bind(entity1)
    second = physics.bind(entity2)

    # Initially both bodies should be 'simple' and 'sameframe'
    self.assertEqual(first.simple, 1)
    self.assertEqual(first.sameframe, 1)
    self.assertEqual(second.simple, 1)
    self.assertEqual(second.sameframe, 1)

    # Assigning to `iquat` should disable 'simple' and 'sameframe' only for that
    # body.
    new_iquat = (0., 0., 0., 1.)
    first.iquat = new_iquat
    self.assertEqual(first.simple, 0)
    self.assertEqual(first.sameframe, 0)
    self.assertEqual(second.simple, 1)
    self.assertEqual(second.sameframe, 1)
    # `ximat` should reflect the new quaternion.
    np.testing.assert_allclose(first.ximat, self.quat2mat(new_iquat))

    # Writing into the `iquat` array should also disable 'simple' and
    # 'sameframe'.
    new_w = -0.1
    iquat_array = second.iquat
    iquat_array[0] = new_w
    self.assertEqual(second.simple, 0)
    self.assertEqual(second.sameframe, 0)
    # `ximat` should reflect the new quaternion.
    np.testing.assert_allclose(second.ximat, self.quat2mat(iquat_array))

  @parameterized.parameters([dict(order='C'), dict(order='F')])
  def test_copy_synchronizing_array_wrapper(self, order):
    root = self.make_simple_model()
    physics = mjcf.Physics.from_mjcf_model(root)
    xpos_view = physics.bind(root.find_all('body')).xpos
    xpos_view_copy = xpos_view.copy(order=order)

    np.testing.assert_array_equal(xpos_view, xpos_view_copy)
    self.assertFalse(np.may_share_memory(xpos_view, xpos_view_copy),
                     msg='Original and copy should not share memory.')
    self.assertIs(type(xpos_view_copy), np.ndarray)

    # Check that `order=` is respected.
    if order == 'C':
      self.assertTrue(xpos_view_copy.flags.c_contiguous)
      self.assertFalse(xpos_view_copy.flags.f_contiguous)
    elif order == 'F':
      self.assertFalse(xpos_view_copy.flags.c_contiguous)
      self.assertTrue(xpos_view_copy.flags.f_contiguous)

    # The copy should be writeable.
    self.assertTrue(xpos_view_copy.flags.writeable)
    new_value = 99.
    xpos_view_copy[0, -1] = new_value
    self.assertEqual(xpos_view_copy[0, -1], new_value)

  def test_error_when_pickling_synchronizing_array_wrapper(self):
    root = self.make_simple_model()
    physics = mjcf.Physics.from_mjcf_model(root)
    xpos_view = physics.bind(root.find_all('body')).xpos
    with self.assertRaisesWithLiteralMatch(
        NotImplementedError,
        mjcf_physics._PICKLING_NOT_SUPPORTED.format(type=type(xpos_view))):
      pickle.dumps(xpos_view)

if __name__ == '__main__':
  absltest.main()
