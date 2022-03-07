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

"""Tests for `mjcf.debugging`."""

import contextlib
import os
import re
import shutil
import sys

from absl.testing import absltest
from dm_control import mjcf
from dm_control.mjcf import code_for_debugging_test as test_code
from dm_control.mjcf import debugging

ORIGINAL_DEBUG_MODE = debugging.debug_mode()


class DebuggingTest(absltest.TestCase):

  def tearDown(self):
    super().tearDown()
    if ORIGINAL_DEBUG_MODE:
      debugging.enable_debug_mode()
    else:
      debugging.disable_debug_mode()

  def setup_debug_mode(self, debug_mode_enabled, full_dump_enabled=False):
    if debug_mode_enabled:
      debugging.enable_debug_mode()
    else:
      debugging.disable_debug_mode()
    if full_dump_enabled:
      base_dir = absltest.get_default_test_tmpdir()
      self.dump_dir = os.path.join(base_dir, 'mjcf_debugging_test')
      shutil.rmtree(self.dump_dir, ignore_errors=True)
      os.mkdir(self.dump_dir)
    else:
      self.dump_dir = ''
    debugging.set_full_dump_dir(self.dump_dir)

  def assertStackFromTestCode(self, stack, function_name, line_ref):
    self.assertEqual(stack[-1].function_name, function_name)
    self.assertStartsWith(test_code.__file__, stack[-1].filename)
    line_info = test_code.LINE_REF['.'.join([function_name, line_ref])]
    self.assertEqual(stack[-1].line_number, line_info.line_number)
    self.assertEqual(stack[-1].text, line_info.text)

  @contextlib.contextmanager
  def assertRaisesTestCodeRef(self, line_ref):
    filename, _ = os.path.splitext(test_code.__file__)
    expected_message = (
        filename + '.py:' + str(test_code.LINE_REF[line_ref].line_number))
    print(expected_message)
    with self.assertRaisesRegex(ValueError, expected_message):
      yield

  def test_get_current_stack_trace(self):
    self.setup_debug_mode(debug_mode_enabled=True)
    stack_trace = debugging.get_current_stack_trace()
    self.assertStartsWith(
        sys.modules[__name__].__file__, stack_trace[-1].filename)
    self.assertEqual(stack_trace[-1].function_name,
                     'test_get_current_stack_trace')
    self.assertEqual(stack_trace[-1].text,
                     'stack_trace = debugging.get_current_stack_trace()')

  def test_disable_debug_mode(self):
    self.setup_debug_mode(debug_mode_enabled=False)
    mjcf_model = test_code.make_valid_model()
    test_code.break_valid_model(mjcf_model)
    self.assertFalse(mjcf_model.get_init_stack())

    my_actuator = mjcf_model.find('actuator', 'my_actuator')
    my_actuator_attrib_stacks = (
        my_actuator.get_last_modified_stacks_for_all_attributes())
    for stack in my_actuator_attrib_stacks.values():
      self.assertFalse(stack)

  def test_element_and_attribute_stacks(self):
    self.setup_debug_mode(debug_mode_enabled=True)
    mjcf_model = test_code.make_valid_model()
    test_code.break_valid_model(mjcf_model)

    self.assertStackFromTestCode(mjcf_model.get_init_stack(),
                                 'make_valid_model', 'mjcf_model')

    my_actuator = mjcf_model.find('actuator', 'my_actuator')
    self.assertStackFromTestCode(my_actuator.get_init_stack(),
                                 'make_valid_model', 'my_actuator')

    my_actuator_attrib_stacks = (
        my_actuator.get_last_modified_stacks_for_all_attributes())
    # `name` attribute was assigned at the same time as the element was created.
    self.assertEqual(my_actuator_attrib_stacks['name'],
                     my_actuator.get_init_stack())
    # `joint` attribute was modified later on.
    self.assertStackFromTestCode(my_actuator_attrib_stacks['joint'],
                                 'break_valid_model', 'my_actuator.joint')

  def test_valid_physics(self):
    self.setup_debug_mode(debug_mode_enabled=True)
    mjcf_model = test_code.make_valid_model()
    mjcf.Physics.from_mjcf_model(mjcf_model)  # Should not raise

  def test_physics_error_message_outside_of_debug_mode(self):
    self.setup_debug_mode(debug_mode_enabled=False)
    mjcf_model = test_code.make_broken_model()
    # Make sure that we advertise debug mode if it's currently disabled.
    with self.assertRaisesRegex(ValueError, '--pymjcf_debug'):
      mjcf.Physics.from_mjcf_model(mjcf_model)

  def test_physics_error_message_in_debug_mode(self):
    self.setup_debug_mode(debug_mode_enabled=True)
    mjcf_model_1 = test_code.make_broken_model()
    with self.assertRaisesTestCodeRef('make_broken_model.my_actuator'):
      mjcf.Physics.from_mjcf_model(mjcf_model_1)
    mjcf_model_2 = test_code.make_valid_model()
    physics = mjcf.Physics.from_mjcf_model(mjcf_model_2)  # Should not raise.
    test_code.break_valid_model(mjcf_model_2)
    with self.assertRaisesTestCodeRef('break_valid_model.my_actuator.joint'):
      physics.reload_from_mjcf_model(mjcf_model_2)

  def test_full_debug_dump(self):
    self.setup_debug_mode(debug_mode_enabled=True, full_dump_enabled=False)
    mjcf_model = test_code.make_valid_model()
    test_code.break_valid_model(mjcf_model)
    # Make sure that we advertise full dump mode if it's currently disabled.
    with self.assertRaisesRegex(ValueError, '--pymjcf_debug_full_dump_dir'):
      mjcf.Physics.from_mjcf_model(mjcf_model)
    self.setup_debug_mode(debug_mode_enabled=True, full_dump_enabled=True)
    with self.assertRaises(ValueError):
      mjcf.Physics.from_mjcf_model(mjcf_model)

    with open(os.path.join(self.dump_dir, 'model.xml')) as f:
      dumped_xml = f.read()
    dumped_xml = [line.strip() for line in dumped_xml.strip().split('\n')]

    xml_line_pattern = re.compile(r'^(.*)<!--pymjcfdebug:(\d+)-->$')
    uninstrumented_pattern = re.compile(r'({})'.format(
        '|'.join([
            r'<mujoco model=".*">',
            r'</mujoco>',
            r'<default class=".*/"/?>',
            r'</default>'
        ])))

    for xml_line in dumped_xml:
      print(xml_line)
      xml_line_match = xml_line_pattern.match(xml_line)
      if not xml_line_match:
        # Only uninstrumented lines are allowed to have no metadata.
        self.assertIsNotNone(uninstrumented_pattern.match(xml_line))
      else:
        xml_element = xml_line_match.group(1)
        debug_id = int(xml_line_match.group(2))
        with open(os.path.join(self.dump_dir, str(debug_id) + '.dump')) as f:
          element_dump = f.read()
        self.assertIn(xml_element, element_dump)

if __name__ == '__main__':
  absltest.main()
