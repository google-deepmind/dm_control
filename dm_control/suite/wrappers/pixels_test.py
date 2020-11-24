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

"""Tests for the pixel wrapper."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.suite import cartpole
from dm_control.suite.wrappers import pixels
import dm_env
from dm_env import specs
import numpy as np


class FakePhysics(object):

  def render(self, *args, **kwargs):
    del args
    del kwargs
    return np.zeros((4, 5, 3), dtype=np.uint8)


class FakeArrayObservationEnvironment(dm_env.Environment):

  def __init__(self):
    self.physics = FakePhysics()

  def reset(self):
    return dm_env.restart(np.zeros((2,)))

  def step(self, action):
    del action
    return dm_env.transition(0.0, np.zeros((2,)))

  def action_spec(self):
    pass

  def observation_spec(self):
    return specs.Array(shape=(2,), dtype=np.float)


class PixelsTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_dict_observation(self, pixels_only):
    pixel_key = 'rgb'

    env = cartpole.swingup()

    # Make sure we are testing the right environment for the test.
    observation_spec = env.observation_spec()
    self.assertIsInstance(observation_spec, collections.OrderedDict)

    width = 320
    height = 240

    # The wrapper should only add one observation.
    wrapped = pixels.Wrapper(env,
                             observation_key=pixel_key,
                             pixels_only=pixels_only,
                             render_kwargs={'width': width, 'height': height})

    wrapped_observation_spec = wrapped.observation_spec()
    self.assertIsInstance(wrapped_observation_spec, collections.OrderedDict)

    if pixels_only:
      self.assertLen(wrapped_observation_spec, 1)
      self.assertEqual([pixel_key], list(wrapped_observation_spec.keys()))
    else:
      expected_length = len(observation_spec) + 1
      self.assertLen(wrapped_observation_spec, expected_length)
      expected_keys = list(observation_spec.keys()) + [pixel_key]
      self.assertEqual(expected_keys, list(wrapped_observation_spec.keys()))

    # Check that the added spec item is consistent with the added observation.
    time_step = wrapped.reset()
    rgb_observation = time_step.observation[pixel_key]
    wrapped_observation_spec[pixel_key].validate(rgb_observation)

    self.assertEqual(rgb_observation.shape, (height, width, 3))
    self.assertEqual(rgb_observation.dtype, np.uint8)

  @parameterized.parameters(True, False)
  def test_single_array_observation(self, pixels_only):
    pixel_key = 'depth'

    env = FakeArrayObservationEnvironment()
    observation_spec = env.observation_spec()
    self.assertIsInstance(observation_spec, specs.Array)

    wrapped = pixels.Wrapper(env, observation_key=pixel_key,
                             pixels_only=pixels_only)
    wrapped_observation_spec = wrapped.observation_spec()
    self.assertIsInstance(wrapped_observation_spec, collections.OrderedDict)

    if pixels_only:
      self.assertLen(wrapped_observation_spec, 1)
      self.assertEqual([pixel_key], list(wrapped_observation_spec.keys()))
    else:
      self.assertLen(wrapped_observation_spec, 2)
      self.assertEqual([pixels.STATE_KEY, pixel_key],
                       list(wrapped_observation_spec.keys()))

    time_step = wrapped.reset()

    depth_observation = time_step.observation[pixel_key]
    wrapped_observation_spec[pixel_key].validate(depth_observation)

    self.assertEqual(depth_observation.shape, (4, 5, 3))
    self.assertEqual(depth_observation.dtype, np.uint8)

if __name__ == '__main__':
  absltest.main()
