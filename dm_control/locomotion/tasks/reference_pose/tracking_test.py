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
"""Tests for mocap tracking."""

import os
from absl.testing import absltest
from absl.testing import parameterized

from dm_control import composer
from dm_control.locomotion import arenas
from dm_control.locomotion import walkers
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import types

import numpy as np

from dm_control.utils import io as resources

TEST_FILE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../mocap'))
TEST_FILE_PATH = os.path.join(TEST_FILE_DIR, 'test_trajectories.h5')


class MultiClipMocapTrackingTest(parameterized.TestCase):

  def setUp(self):
    super(MultiClipMocapTrackingTest, self).setUp()

    self.walker = walkers.CMUHumanoidPositionControlled
    def _make_wrong_walker(name):
      return walkers.CMUHumanoidPositionControlled(
          include_face=False, model_version='2020', scale_default=True,
          name=name)
    self.wrong_walker = _make_wrong_walker
    self.arena = arenas.Floor()
    self.test_data = resources.GetResourceFilename(TEST_FILE_PATH)

  @parameterized.named_parameters(('termination_reward', 'termination_reward'),
                                  ('comic', 'comic'))
  def test_initialization_and_step(self, reward):

    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=1,
        reward_type=reward,
    )

    env = composer.Environment(task=task)

    env.reset()

    # check no task error after episode init before first step
    self.assertLess(task._termination_error, 1e-3)

    action_spec = env.action_spec()
    env.step(np.zeros(action_spec.shape))

  @parameterized.named_parameters(('first_clip', 0), ('second_clip', 1))
  def test_clip_weights(self, clip_number):
    # test whether clip weights work correctly if ids are not specified.

    clip_weights = (1, 0) if clip_number == 0 else (0, 1)
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=1,
        dataset=types.ClipCollection(
            ids=('cmuv2019_001', 'cmuv2019_002'), weights=clip_weights),
        reward_type='comic',
    )

    env = composer.Environment(task=task)

    env.reset()

    self.assertEqual(task._current_clip.identifier,
                     task._dataset.ids[clip_number])

  @parameterized.named_parameters(
      ('start_step_id_length_mismatch_explicit_id', (0,), (10, 10), (1, 1)),
      ('end_step_id_length_mismatch_explicit_id', (0, 0), (10,), (1, 1)),
      ('clip_weights_id_length_mismatch_explicit_id', (0, 0), (10, 10), (1,)),
  )
  def test_task_validation(self, clip_start_steps, clip_end_steps,
                           clip_weights):
    # test whether task construction fails with invalid arguments.
    with self.assertRaisesRegex(ValueError, 'ClipCollection'):
      unused_task = tracking.MultiClipMocapTracking(
          walker=self.walker,
          arena=self.arena,
          ref_path=self.test_data,
          ref_steps=(1, 2, 3, 4, 5),
          min_steps=1,
          dataset=types.ClipCollection(
              ids=('cmuv2019_001', 'cmuv2019_002'),
              start_steps=clip_start_steps,
              end_steps=clip_end_steps,
              weights=clip_weights),
          reward_type='comic',
          )

  def test_init_at_clip_start(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(
            ids=('cmuv2019_001', 'cmuv2019_002'),
            start_steps=(2, 0),
            end_steps=(10, 10)),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=1,
        reward_type='termination_reward',
        always_init_at_clip_start=True,
    )
    self.assertEqual(task._possible_starts, [(0, 2), (1, 0)])

  def test_failure_with_wrong_walker(self):
    with self.assertRaisesRegex(ValueError, 'proto/walker'):
      task = tracking.MultiClipMocapTracking(
          walker=self.wrong_walker,
          arena=self.arena,
          ref_path=self.test_data,
          ref_steps=(1, 2, 3, 4, 5),
          min_steps=1,
          dataset=types.ClipCollection(
              ids=('cmuv2019_001', 'cmuv2019_002'),
              start_steps=(0, 0),
              end_steps=(10, 10)),
          reward_type='comic',
      )

      env = composer.Environment(task=task)

      env.reset()


if __name__ == '__main__':
  absltest.main()
