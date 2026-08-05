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
_ZIPPED_MODEL = os.path.join(ASSETS_DIR, 'model_with_assetdir.zip')


def validate(xml_string, assets=None):
  """Validates that an XML string is a valid MJCF.

  Validation is performed by constructing Mujoco model from the string.
  The construction process contains compilation and validation phases by Mujoco
  engine, the best validation tool we have access to.

  Args:
    xml_string: XML string to validate
    assets: Optional dict of assets to use for the model.
  """

  mjmodel = wrapper.MjModel.from_xml_string(xml_string, assets)
  wrapper.MjData(mjmodel)


class XMLValidationTest(absltest.TestCase):

  def testDeformableFlexRoundTrip(self):
    # A flex needs three floats per vertex, so even the smallest one exceeds
    # the five entries the schema used to allow. `dim` and `radius` are
    # omitted here because MuJoCo supplies defaults for both.
    model = parser.from_xml_string("""
<mujoco model="test">
  <worldbody>
    <body name="v0"><freejoint/><geom type="sphere" size="0.01"/></body>
    <body name="v1" pos="0.1 0 0"><freejoint/><geom type="sphere" size="0.01"/></body>
    <body name="v2" pos="0 0.1 0"><freejoint/><geom type="sphere" size="0.01"/></body>
  </worldbody>
  <deformable>
    <flex name="f" group="2" flatskin="true" body="v0 v1 v2"
          vertex="0 0 0  0.1 0 0  0 0.1 0" element="0 1 2"/>
  </deformable>
</mujoco>
""")
    xml_string = model.to_xml_string()
    validate(xml_string)
    mjmodel = wrapper.MjModel.from_xml_string(xml_string)
    self.assertEqual(mjmodel.flex_vertnum[0], 3)
    self.assertEqual(mjmodel.flex_group[0], 2)
    self.assertTrue(mjmodel.flex_flatskin[0])

  def testFlexcompFlatskinIsBoolean(self):
    for parent_open, parent_close in (('<body name="b">', '</body>'), ('', '')):
      model = parser.from_xml_string("""
<mujoco model="test">
  <worldbody>
    %s
    <flexcomp name="fc" type="grid" dim="2" count="3 3 1" spacing="0.1 0.1 0.1"
              radius="0.001" flatskin="true"/>
    %s
  </worldbody>
</mujoco>
""" % (parent_open, parent_close))
      xml_string = model.to_xml_string()
      validate(xml_string)
      mjmodel = wrapper.MjModel.from_xml_string(xml_string)
      self.assertTrue(mjmodel.flex_flatskin[0])

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

  def testXmlFromZip(self):
    model = parser.from_zip(_ZIPPED_MODEL)
    validate(model.to_xml_string(), model.get_assets())


if __name__ == '__main__':
  absltest.main()
