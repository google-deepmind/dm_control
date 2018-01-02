# Copyright 2017 The dm_control Authors.
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

"""Tests for mjbindings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from absl.testing import parameterized

from dm_control.mujoco.wrapper.mjbindings import constants
from dm_control.mujoco.wrapper.mjbindings import sizes


class MjbindingsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('mjdata', 'xpos', ('nbody', 3)),
      ('mjmodel', 'geom_type', ('ngeom',)),
      # Fields with identifiers in mjxmacro that are resolved at compile-time.
      ('mjmodel', 'actuator_dynprm', ('nu', constants.mjNDYN)),
      ('mjdata', 'efc_solref', ('njmax', constants.mjNREF)),
      # Fields with multiple named indices.
      ('mjmodel', 'key_qpos', ('nkey', 'nq')),
  )
  def testIndexDict(self, struct_name, field_name, expected_metadata):
    self.assertEqual(expected_metadata,
                     sizes.array_sizes[struct_name][field_name])


if __name__ == '__main__':
  absltest.main()
