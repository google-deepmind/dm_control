# Copyright 2020 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Saves Mujoco models with relevant assets in a .zip file."""

import os
import zipfile

from dm_control.mjcf import constants


def export_with_assets_as_zip(mjcf_model, out_dir, model_name=None,
                              *,
                              precision=constants.XML_DEFAULT_PRECISION,
                              zero_threshold=0):
  """Saves mjcf_model and all its assets as a .zip file in the given directory.

  Creates a .zip file named `model_name`.zip in the specified `out_dir`, and a
  directory inside of this file named `model_name`. The MJCF XML is written into
  this directory with the name `model_name`.xml, and all the assets are also
  written into this directory without changing their names.

  Args:
    mjcf_model: `mjcf.RootElement` instance to export.
    out_dir: Directory to save the .zip file. Will be created if it does not
      already exist.
    model_name: (Optional) Name of the .zip file, the name of the directory
      inside the .zip root containing the model and assets, and name of the XML
      file inside this directory. Defaults to the MJCF model name
      (`mjcf_model.model`).
    precision: (optional) Number of digits to output for floating point
      quantities.
    zero_threshold: (optional) When outputting XML, floating point quantities
      whose absolute value falls below this threshold will be treated as zero.

  Returns:
    The full path to the .zip file.
  """

  if model_name is None:
    model_name = mjcf_model.model

  xml_name = model_name + '.xml'
  zip_name = model_name + '.zip'

  files_to_zip = mjcf_model.get_assets()
  files_to_zip[xml_name] = mjcf_model.to_xml_string(
      precision=precision, zero_threshold=zero_threshold)

  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  file_name = os.path.join(out_dir, zip_name)
  with zipfile.ZipFile(file_name, 'w') as zip_file:
    for filename, contents in files_to_zip.items():
      zip_file.writestr(os.path.join(model_name, filename), contents)
  return file_name
