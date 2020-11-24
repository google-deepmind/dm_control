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

"""Tests for `dm_control.mjcf.copier`."""

import os

from absl.testing import absltest
from dm_control import mjcf
from dm_control.mjcf import parser
import numpy as np
import six

_ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
_TEST_MODEL_XML = os.path.join(_ASSETS_DIR, 'test_model.xml')
_MODEL_WITH_ASSETS_XML = os.path.join(_ASSETS_DIR, 'model_with_assets.xml')


class CopierTest(absltest.TestCase):

  def testSimpleCopy(self):
    mjcf_model = parser.from_path(_TEST_MODEL_XML)
    mixin = mjcf.RootElement(model='test_mixin')
    mixin.compiler.boundmass = 1
    mjcf_model.include_copy(mixin)
    self.assertEqual(mjcf_model.model, 'test')  # Model name should not change
    self.assertEqual(mjcf_model.compiler.boundmass, mixin.compiler.boundmass)
    mixin.compiler.boundinertia = 2
    mjcf_model.include_copy(mixin)
    self.assertEqual(mjcf_model.compiler.boundinertia,
                     mixin.compiler.boundinertia)
    mixin.compiler.boundinertia = 1
    with six.assertRaisesRegex(self, ValueError, 'Conflicting values'):
      mjcf_model.include_copy(mixin)
    mixin.worldbody.add('body', name='b_0', pos=[0, 1, 2])
    mjcf_model.include_copy(mixin, override_attributes=True)
    self.assertEqual(mjcf_model.compiler.boundmass, mixin.compiler.boundmass)
    self.assertEqual(mjcf_model.compiler.boundinertia,
                     mixin.compiler.boundinertia)
    np.testing.assert_array_equal(mjcf_model.worldbody.body['b_0'].pos,
                                  [0, 1, 2])

  def testCopyingWithReference(self):
    sensor_mixin = mjcf.RootElement('sensor_mixin')
    touch_site = sensor_mixin.worldbody.add('site', name='touch_site')
    sensor_mixin.sensor.add('touch', name='touch_sensor', site=touch_site)

    mjcf_model = mjcf.RootElement('model')
    mjcf_model.include_copy(sensor_mixin)

    # Copied reference should be updated to the copied site.
    self.assertIs(mjcf_model.find('sensor', 'touch_sensor').site,
                  mjcf_model.find('site', 'touch_site'))

  def testCopyingWithAssets(self):
    mjcf_model = parser.from_path(_MODEL_WITH_ASSETS_XML)
    copied = mjcf.RootElement()
    copied.include_copy(mjcf_model)

    original_assets = (mjcf_model.find_all('mesh')
                       + mjcf_model.find_all('texture')
                       + mjcf_model.find_all('hfield'))
    copied_assets = (copied.find_all('mesh')
                     + copied.find_all('texture')
                     + copied.find_all('hfield'))

    self.assertLen(copied_assets, len(original_assets))
    for original_asset, copied_asset in zip(original_assets, copied_assets):
      self.assertIs(copied_asset.file, original_asset.file)

if __name__ == '__main__':
  absltest.main()
