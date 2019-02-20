# Copyright 2019 The dm_control Authors.
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

"""Tests for dm_control.locomotion.soccer.boxhead."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.locomotion.soccer import boxhead


class BoxheadTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(camera_control=True, walker_id=None),
      dict(camera_control=False, walker_id=None),
      dict(camera_control=True, walker_id=0),
      dict(camera_control=False, walker_id=10))
  def test_instantiation(self, camera_control, walker_id):
    boxhead.BoxHead(marker_rgba=[.8, .1, .1, 1.],
                    camera_control=camera_control,
                    walker_id=walker_id)

  @parameterized.parameters(-1, 11)
  def test_invalid_walker_id(self, walker_id):
    with self.assertRaisesWithLiteralMatch(
        ValueError, boxhead._INVALID_WALKER_ID.format(walker_id)):
      boxhead.BoxHead(walker_id=walker_id)


if __name__ == '__main__':
  absltest.main()
