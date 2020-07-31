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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Tests for dm_control.mjcf.skin."""

import os

from absl.testing import absltest
from dm_control.mjcf import skin

from dm_control.utils import io as resources

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
SKIN_FILE_PATH = os.path.join(ASSETS_DIR, 'skins/test_skin.skn')


class FakeMJCFBody:

  def __init__(self, full_identifier):
    self.full_identifier = full_identifier


class SkinTest(absltest.TestCase):

  def test_can_parse_and_write_back(self):
    contents = resources.GetResource(SKIN_FILE_PATH, mode='rb')
    parsed = skin.parse(contents, body_getter=FakeMJCFBody)
    reconstructed = skin.serialize(parsed)
    self.assertEqual(reconstructed, contents)


if __name__ == '__main__':
  absltest.main()
