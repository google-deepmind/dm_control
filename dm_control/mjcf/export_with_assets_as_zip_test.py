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
"""Tests for `dm_control.mjcf.export_with_assets`."""

import os
import zipfile

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.mujoco.wrapper import util
import six

FLAGS = flags.FLAGS

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
_TEST_MODEL_WITH_ASSETS = os.path.join(_ASSETS_DIR, 'model_with_assets.xml')
_TEST_MODEL_WITHOUT_ASSETS = os.path.join(_ASSETS_DIR, 'lego_brick.xml')


def setUpModule():
  # Flags are not parsed when this test is invoked by `nosetests`, so we fall
  # back on using the default value for `--test_tmpdir`.
  if not FLAGS.is_parsed():
    FLAGS.test_tmpdir = absltest.get_default_test_tmpdir()
    FLAGS.mark_as_parsed()


class ExportWithAssetsAsZipTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('with_assets', _TEST_MODEL_WITH_ASSETS, 'mujoco_with_assets'),
      ('without_assets', _TEST_MODEL_WITHOUT_ASSETS, 'mujoco'),
  )
  def test_export_model(self, xml_path, model_name):
    """Save processed MJCF model."""
    out_dir = self.create_tempdir().full_path
    mjcf_model = mjcf.from_path(xml_path)
    mjcf.export_with_assets_as_zip(
        mjcf_model, out_dir=out_dir, model_name=model_name)

    # Read the .zip file in the output directory.
    # Check that the only directory is named `model_name`/, and put all the
    # contents under any directory in a dict a directory in a dict.
    zip_file_contents = {}
    zip_filename = os.path.join(out_dir, (model_name + '.zip'))
    self.assertTrue(zipfile.is_zipfile(zip_filename))
    with zipfile.ZipFile(zip_filename, 'r') as zip_file:
      for zip_info in zip_file.infolist():
        # Note: zip_info.is_dir() is not Python 2 compatible, but directories
        # inside a ZipFile are guaranteed to end with '/'.
        if not zip_info.filename.endswith(os.path.sep):
          with zip_file.open(zip_info.filename) as f:
            zip_file_contents[zip_info.filename] = f.read()
        else:
          self.assertEqual(os.path.join(model_name), zip_info.filename)

    # Check that the output directory contains an XML file of the correct name.
    xml_filename = os.path.join(model_name, model_name) + '.xml'
    self.assertIn(xml_filename, zip_file_contents)

    # Check that its contents match the output of `mjcf_model.to_xml_string()`.
    xml_contents = util.to_native_string(zip_file_contents.pop(xml_filename))
    expected_xml_contents = mjcf_model.to_xml_string()
    self.assertEqual(xml_contents, expected_xml_contents)

    # Check that the other files in the directory match the contents of the
    # model's `assets` dict.
    assets = mjcf_model.get_assets()
    for asset_name, asset_contents in six.iteritems(assets):
      self.assertEqual(asset_contents,
                       zip_file_contents[os.path.join(model_name, asset_name)])

  def test_default_model_filename(self):
    out_dir = self.create_tempdir().full_path
    mjcf_model = mjcf.from_path(_TEST_MODEL_WITH_ASSETS)
    mjcf.export_with_assets_as_zip(mjcf_model, out_dir, model_name=None)
    expected_name = mjcf_model.model + '.zip'
    self.assertTrue(os.path.isfile(os.path.join(out_dir, expected_name)))


if __name__ == '__main__':
  absltest.main()
