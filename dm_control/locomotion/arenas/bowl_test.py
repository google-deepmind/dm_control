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
"""Tests for locomotion.arenas.bowl."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from dm_control import mjcf
from dm_control.locomotion.arenas import bowl


class BowlTest(absltest.TestCase):

  def test_can_compile_mjcf(self):

    arena = bowl.Bowl()
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)


if __name__ == '__main__':
  absltest.main()
