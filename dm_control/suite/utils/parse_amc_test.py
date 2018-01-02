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

"""Tests for parse_amc utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

# Internal dependencies.

from absl.testing import absltest
from dm_control.suite import humanoid_CMU
from dm_control.suite.utils import parse_amc

from dm_control.utils import resources

_TEST_AMC_PATH = resources.GetResourceFilename(
    os.path.join(os.path.dirname(__file__), '../demos/zeros.amc'))


class ParseAMCTest(absltest.TestCase):

  def test_sizes_of_parsed_data(self):

    # Instantiate the humanoid environment.
    env = humanoid_CMU.stand()

    # Parse and convert specified clip.
    converted = parse_amc.convert(
        _TEST_AMC_PATH, env.physics, env.control_timestep())

    self.assertEqual(converted.qpos.shape[0], 63)
    self.assertEqual(converted.qvel.shape[0], 62)
    self.assertEqual(converted.time.shape[0], converted.qpos.shape[1])
    self.assertEqual(converted.qpos.shape[1],
                     converted.qvel.shape[1] + 1)

    # Parse and convert specified clip -- WITH SMALLER TIMESTEP
    converted2 = parse_amc.convert(
        _TEST_AMC_PATH, env.physics, 0.5 * env.control_timestep())

    self.assertEqual(converted2.qpos.shape[0], 63)
    self.assertEqual(converted2.qvel.shape[0], 62)
    self.assertEqual(converted2.time.shape[0], converted2.qpos.shape[1])
    self.assertEqual(converted.qpos.shape[1],
                     converted.qvel.shape[1] + 1)

    # Compare sizes of parsed objects for different timesteps
    self.assertEqual(converted.qpos.shape[1] * 2, converted2.qpos.shape[1])


if __name__ == '__main__':
  absltest.main()
