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

"""Tests for `dm_control.mjcf.export_with_assets`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper import util
import six

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
_TEST_MODEL_WITH_ASSETS = os.path.join(_ASSETS_DIR, 'model_with_assets.xml')
_TEST_MODEL_WITHOUT_ASSETS = os.path.join(_ASSETS_DIR, 'lego_brick.xml')
_OUT_DIR = os.path.join(absltest.get_default_test_tmpdir(), 'export')


class ExportWithAssetsTest(parameterized.TestCase):

  def setUp(self):
    super(ExportWithAssetsTest, self).setUp()
    # Remove any existing export directory and its contents between tests.
    shutil.rmtree(_OUT_DIR, ignore_errors=True)

  @parameterized.named_parameters(
      ('with_assets', _TEST_MODEL_WITH_ASSETS, 'mujoco_with_assets.xml'),
      ('without_assets', _TEST_MODEL_WITHOUT_ASSETS, 'mujoco.xml'),)
  def test_export_model(self, xml_path, out_xml_name):
    """Save processed MJCF model."""
    mjcf_model = mjcf.from_path(xml_path)
    mjcf.export_with_assets(
        mjcf_model, out_dir=_OUT_DIR, out_file_name=out_xml_name)

    # Read the files in the output directory and put their contents in a dict.
    out_dir_contents = {}
    for filename in os.listdir(_OUT_DIR):
      with open(os.path.join(_OUT_DIR, filename), 'rb') as f:
        out_dir_contents[filename] = f.read()

    # Check that the output directory contains an XML file of the correct name.
    self.assertIn(out_xml_name, out_dir_contents)

    # Check that its contents match the output of `mjcf_model.to_xml_string()`.
    xml_contents = util.to_native_string(out_dir_contents.pop(out_xml_name))
    expected_xml_contents = mjcf_model.to_xml_string()
    self.assertEqual(xml_contents, expected_xml_contents)

    # Check that the other files in the directory match the contents of the
    # model's `assets` dict.
    assets = mjcf_model.get_assets()
    self.assertDictEqual(out_dir_contents, assets)

    # Check that we can construct an MjModel instance using the path to the
    # output file.
    from_exported = wrapper.MjModel.from_xml_path(
        os.path.join(_OUT_DIR, out_xml_name))

    # Check that this model is identical to one constructed directly from
    # `mjcf_model`.
    from_mjcf = wrapper.MjModel.from_xml_string(
        expected_xml_contents, assets=assets)
    self.assertEqual(from_exported.to_bytes(), from_mjcf.to_bytes())

  def test_default_model_filename(self):
    mjcf_model = mjcf.from_path(_TEST_MODEL_WITH_ASSETS)
    mjcf.export_with_assets(mjcf_model, _OUT_DIR, out_file_name=None)
    expected_name = mjcf_model.model + '.xml'
    self.assertTrue(os.path.isfile(os.path.join(_OUT_DIR, expected_name)))

  def test_exceptions(self):
    mjcf_model = mjcf.from_path(_TEST_MODEL_WITH_ASSETS)
    with six.assertRaisesRegex(self, ValueError, 'must end with \'.xml\''):
      mjcf.export_with_assets(mjcf_model, _OUT_DIR,
                              out_file_name='invalid_extension.png')


if __name__ == '__main__':
  absltest.main()
