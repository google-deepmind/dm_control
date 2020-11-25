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

"""Tests for utils.xml_tools."""

import io

from absl.testing import absltest
from dm_control.utils import xml_tools
import lxml

etree = lxml.etree


class XmlHelperTest(absltest.TestCase):

  def test_nested(self):
    element = etree.Element('inserted')
    xml_tools.nested_element(element, depth=2)
    level_1 = element.find('inserted')
    self.assertIsNotNone(level_1)
    level_2 = level_1.find('inserted')
    self.assertIsNotNone(level_2)

  def test_tostring(self):
    xml_str = """
    <root>
      <head>
        <content></content>
      </head>
    </root>"""
    tree = xml_tools.parse(io.StringIO(xml_str))
    self.assertEqual(b'<root>\n  <head>\n    <content/>\n  </head>\n</root>\n',
                     etree.tostring(tree, pretty_print=True))

  def test_find_element(self):
    xml_str = """
    <root>
      <option name='option_name'>
        <content></content>
      </option>
      <world name='world_name'>
        <geom name='geom_name'/>
      </world>
    </root>"""
    tree = xml_tools.parse(io.StringIO(xml_str))
    world = xml_tools.find_element(root=tree, tag='world', name='world_name')
    self.assertEqual(world.tag, 'world')
    self.assertEqual(world.attrib['name'], 'world_name')

    geom = xml_tools.find_element(root=tree, tag='geom', name='geom_name')
    self.assertEqual(geom.tag, 'geom')
    self.assertEqual(geom.attrib['name'], 'geom_name')

    with self.assertRaisesRegex(ValueError, 'Element with tag'):
      xml_tools.find_element(root=tree, tag='does_not_exist', name='name')

    with self.assertRaisesRegex(ValueError, 'Element with tag'):
      xml_tools.find_element(root=tree, tag='world', name='does_not_exist')


if __name__ == '__main__':
  absltest.main()
