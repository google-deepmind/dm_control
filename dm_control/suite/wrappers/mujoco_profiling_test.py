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
"""Tests for the mujoco_profiling wrapper."""

import collections

from absl.testing import absltest
from dm_control.suite import cartpole
from dm_control.suite.wrappers import mujoco_profiling
import numpy as np


class MujocoProfilingTest(absltest.TestCase):

  def test_dict_observation(self):
    obs_key = 'mjprofile'

    env = cartpole.swingup()

    # Make sure we are testing the right environment for the test.
    observation_spec = env.observation_spec()
    self.assertIsInstance(observation_spec, collections.OrderedDict)

    # The wrapper should only add one observation.
    wrapped = mujoco_profiling.Wrapper(env, observation_key=obs_key)

    wrapped_observation_spec = wrapped.observation_spec()
    self.assertIsInstance(wrapped_observation_spec, collections.OrderedDict)

    expected_length = len(observation_spec) + 1
    self.assertLen(wrapped_observation_spec, expected_length)
    expected_keys = list(observation_spec.keys()) + [obs_key]
    self.assertEqual(expected_keys, list(wrapped_observation_spec.keys()))

    # Check that the added spec item is consistent with the added observation.
    time_step = wrapped.reset()
    profile_observation = time_step.observation[obs_key]
    wrapped_observation_spec[obs_key].validate(profile_observation)

    self.assertEqual(profile_observation.shape, (2,))
    self.assertEqual(profile_observation.dtype, np.double)


if __name__ == '__main__':
  absltest.main()
