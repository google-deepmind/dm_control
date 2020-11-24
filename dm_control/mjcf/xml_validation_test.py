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

"""Tests that generated XML string is valid."""

import os

from absl.testing import absltest
from dm_control.mjcf import parser
from dm_control.mujoco import wrapper

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
_ARENA_XML = os.path.join(ASSETS_DIR, 'arena.xml')
_LEGO_BRICK_XML = os.path.join(ASSETS_DIR, 'lego_brick.xml')
_ROBOT_XML = os.path.join(ASSETS_DIR, 'robot_arm.xml')


def validate(xml_string):
  """Validates that an XML string is a valid MJCF.

  Validation is performed by constructing Mujoco model from the string.
  The construction process contains compilation and validation phases by Mujoco
  engine, the best validation tool we have access to.

  Args:
    xml_string: XML string to validate
  """

  mjmodel = wrapper.MjModel.from_xml_string(xml_string)
  wrapper.MjData(mjmodel)


class XMLValidationTest(absltest.TestCase):

  def testXmlAttach(self):
    robot_arm = parser.from_file(_ROBOT_XML)
    arena = parser.from_file(_ARENA_XML)
    lego = parser.from_file(_LEGO_BRICK_XML)

    # validate MJCF strings before changing them
    validate(robot_arm.to_xml_string())
    validate(arena.to_xml_string())
    validate(lego.to_xml_string())

    # combine objects in complex scene
    robot_arm.find('site', 'fingertip1').attach(lego)
    arena.worldbody.attach(robot_arm)

    # validate
    validate(arena.to_xml_string())


if __name__ == '__main__':
  absltest.main()
