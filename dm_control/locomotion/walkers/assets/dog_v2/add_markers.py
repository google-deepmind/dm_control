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

"""Add markers to the dog model, used for motion capture tracking."""

import collections
import os

from absl import app
from absl import flags

from dm_control import mjcf
from dm_control.utils import io as resources

import editdistance
import ipdb  # pylint: disable=unused-import
from lxml import etree

import numpy as np

BASE_MODEL = "./dog_muscles_-1.xml"

print('Load base model.')
with open(BASE_MODEL, 'r') as f:
  model = mjcf.from_file(f)

markers_per_body = {
  'torso': [[-0.25, 0, 0.13], [-0.1, 0, 0.13], [0.05, 0, 0.15],
            [-0.25, -0.06, -0.01], [-0.1, -0.08, -0.02], [0.05, -0.09, -0.02],
            [-0.25, 0.06, -0.01], [-0.1, 0.08, -0.02], [0.05, 0.09, -0.02]],
  'C_6': [[0.05, -0.05, 0], [0.05, 0.05, 0]],
  'upper_arm': [[0.04, 0, 0]],
  'lower_arm': [[0.025, 0, 0], [0.025, 0, -0.1]],
  'hand': [[0, -0.02, 0], [0, 0.02, 0], [0.02, -0.02, -0.08],
           [0.02, 0.02, -0.08]],
  'finger': [[0.065, 0, 0]],
  'pelvis': [[0.04, -0.07, 0.02], [0.04, 0.07, 0.02]],
  'upper_leg': [[-0.03, 0.04, 0]],
  'lower_leg': [[0.04, 0, 0], [-0.04, 0.035, 0]],
  'foot': [[0, -0.02, 0], [0, 0.02, 0], [0, -0.02, -0.1], [0, 0.03, -0.1]],
  'toe': [[0.055, 0, 0]],
  'Ca_10': [[0, 0, 0.01]],
  'skull': [[0.01, 0, 0.04], [0, -0.05, 0], [0, 0.05, 0], [-0.1, 0, 0.02]]}

bodies = model.find_all('body')

TOT_MARKERS = 0
for body in bodies:
  for name, markers_pos in markers_per_body.items():
    if name in body.name and 'anchor' not in body.name:
      marker_idx = 0
      for pos in markers_pos:
        if '_R' in body.name:
          pos[1] *= -1
        body.add("site", name='marker_' + body.name + '_' + str(marker_idx),
                 rgba=[1, 0.5, 0, 0.5], size=[0.01, 0.01, 0.01], pos=pos)

        marker_idx += 1
        TOT_MARKERS += 1
      print(body.name)

for i in range(50):
  marker_body = model.worldbody.add('body', name="marker_" + str(i), mocap=True)
  marker_body.add('site', name="marker_" + str(i), rgba=[1, 0, 0, 0.5],
                  size=[0.01, 0.01, 0.01])

print("TOT_MARKERS:", TOT_MARKERS)

xml_string = model.to_xml_string('float', precision=4, zero_threshold=1e-7)
root = etree.XML(xml_string, etree.XMLParser(remove_blank_text=True))
print('Remove hashes from filenames')
assets = list(root.find('asset').iter())
for asset in assets:
  asset_filename = asset.get('file')
  if asset_filename is not None:
    name = asset_filename[:-4]
    extension = asset_filename[-4:]
    asset.set('file', name[:-41] + extension)

ASSET_RELPATH = '/dog_assets'
print('Add <compiler meshdir/>, for locally-loadable model')
compiler = etree.Element(
  'compiler', meshdir="." + ASSET_RELPATH, texturedir="." + ASSET_RELPATH)
root.insert(0, compiler)

print('Remove class="/"')
default_elem = root.find('default')
root.insert(6, default_elem[0])
root.remove(default_elem)
xml_string = etree.tostring(root, pretty_print=True)
xml_string = xml_string.replace(b' class="/"', b'')

print('Insert spaces between top level elements')
lines = xml_string.splitlines()
newlines = []
for line in lines:
  newlines.append(line)
  if line.startswith(b'  <'):
    if line.startswith(b'  </') or line.endswith(b'/>'):
      newlines.append(b'')
newlines.append(b'')
xml_string = b'\n'.join(newlines)
name = "dog_muscles_-1_markers.xml"
f = open(name, 'wb')
f.write(xml_string)
f.close()


