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

"""Saves Mujoco models with relevant assets."""

import os
from dm_control.mujoco.wrapper import util


def export_with_assets(mjcf_model, out_dir, out_file_name=None):
  """Saves mjcf.model in the given directory in MJCF (XML) format.

  Creates an MJCF XML file named `out_file_name` in the specified `out_dir`, and
  writes all of its assets into the same directory.

  Args:
    mjcf_model: `mjcf.RootElement` instance to export.
    out_dir: Directory to save the model and assets. Will be created if it does
      not already exist.
    out_file_name: (Optional) Name of the XML file to create. Defaults to the
      model name (`mjcf_model.model`) suffixed with '.xml'.

  Raises:
    ValueError: If `out_file_name` is a string that does not end with '.xml'.
  """
  if out_file_name is None:
    out_file_name = mjcf_model.model + '.xml'
  elif not out_file_name.lower().endswith('.xml'):
    raise ValueError('If `out_file_name` is specified it must end with '
                     '\'.xml\': got {}'.format(out_file_name))
  assets = mjcf_model.get_assets()
  # This should never happen because `mjcf` does not support `.xml` assets.
  assert out_file_name not in assets
  assets[out_file_name] = mjcf_model.to_xml_string()
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  for filename, contents in assets.items():
    with open(os.path.join(out_dir, filename), 'wb') as f:
      f.write(util.to_binary_string(contents))
