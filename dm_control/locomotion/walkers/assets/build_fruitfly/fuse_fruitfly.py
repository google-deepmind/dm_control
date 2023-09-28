# Copyright 2023 The dm_control Authors.
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
"""Fuse fruitfly model."""

import os
from typing import Sequence

from absl import app

from dm_control import mujoco
from lxml import etree

ASSET_RELPATH = 'assets/'
ASSET_DIR = os.path.dirname(__file__) + '/' + ASSET_RELPATH
BASE_MODEL = 'drosophila_defaults.xml'
FLY_MODEL = 'drosophila.xml'  # Raw model as exported from Blender.
FUSED_MODEL = ASSET_DIR + 'drosophila_fused.xml'


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  print('Load base model.')
  with open(os.path.join(ASSET_DIR, FLY_MODEL), 'r') as f:
    tree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True))

  print('Remove lights.')
  lights = tree.xpath('.//light')
  for light in lights:
    light.getparent().remove(light)

  print('Set fusestatic option.')
  compiler = tree.find('compiler')
  compiler.attrib['fusestatic'] = 'true'
  del compiler.attrib['boundmass']
  del compiler.attrib['boundinertia']

  print('Add freejoint.')
  root = tree.find('worldbody').find('body')
  root.getchildren()[0].addprevious(etree.Element('freejoint'))

  print('Load physics, fuse.')
  physics = mujoco.Physics.from_xml_string(etree.tostring(tree,
                                                          pretty_print=True))

  print('Save fused model.')
  mujoco.mj_saveLastXML(os.path.join(FUSED_MODEL), physics.model.ptr)


if __name__ == '__main__':
  app.run(main)
