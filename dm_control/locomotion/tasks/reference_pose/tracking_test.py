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
from dm_control.locomotion.mocap import props
from dm_control.locomotion.tasks.reference_pose import tracking
from dm_control.locomotion.tasks.reference_pose import types

import numpy as np

from dm_control.utils import io as resources

TEST_FILE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '../../mocap'))
TEST_FILE_PATH = os.path.join(TEST_FILE_DIR, 'test_trajectories.h5')

REFERENCE_PROP_KEYS = [
    f'reference_props_{key}_global' for key in ['pos', 'quat']
]
PROP_OBSERVATION_KEYS = [
    f'cmuv2019_box/{key}' for key in ['position', 'orientation']
]
N_PROPS = 1
GHOST_OFFSET = np.array((0, 0, 0.1))


class MultiClipMocapTrackingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

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

  def test_enabled_reference_observables(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(1, 2, 3, 4, 5),
        min_steps=1,
        reward_type='comic',
        enabled_reference_observables=('walker/reference_rel_joints',)
    )

    env = composer.Environment(task=task)

    timestep = env.reset()

    self.assertIn('walker/reference_rel_joints', timestep.observation.keys())
    self.assertNotIn('walker/reference_rel_root_pos_local',
                     timestep.observation.keys())

    # check that all desired observables are enabled.
    desired_observables = []
    desired_observables += task._walker.observables.proprioception
    desired_observables += task._walker.observables.kinematic_sensors
    desired_observables += task._walker.observables.dynamic_sensors

    for observable in desired_observables:
      self.assertTrue(observable.enabled)

  def test_prop_factory(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(0,),
        min_steps=1,
        disable_props=False,
        prop_factory=props.Prop,
    )
    env = composer.Environment(task=task)

    observation = env.reset().observation
    # Test the expected prop observations exist and have the expected size.
    dims = [3, 4]
    for key, dim in zip(REFERENCE_PROP_KEYS, dims):
      self.assertIn(key, task.observables)
      self.assertSequenceEqual(observation[key].shape, (N_PROPS, dim))

    # Since no ghost offset was specified, test that there are no ghost props.
    self.assertEmpty(task._ghost_props)

    # Test that props go to the expected location on reset.
    for ref_key, obs_key in zip(REFERENCE_PROP_KEYS, PROP_OBSERVATION_KEYS):
      np.testing.assert_array_almost_equal(
          observation[ref_key], observation[obs_key]
      )

  def test_ghost_prop(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(0,),
        min_steps=1,
        disable_props=False,
        prop_factory=props.Prop,
        ghost_offset=GHOST_OFFSET,
    )
    env = composer.Environment(task=task)

    # Test that the ghost props are present when ghost_offset specified.
    self.assertLen(task._ghost_props, N_PROPS)

    # Test that the ghost prop tracks the goal trajectory after step.
    env.reset()
    observation = env.step(env.action_spec().generate_value()).observation
    ghost_pos, ghost_quat = task._ghost_props[0].get_pose(env.physics)
    goal_pos, goal_quat = (
        np.squeeze(observation[key]) for key in REFERENCE_PROP_KEYS)

    np.testing.assert_array_equal(np.array(ghost_pos), goal_pos + GHOST_OFFSET)
    np.testing.assert_array_almost_equal(ghost_quat, goal_quat)

  def test_disable_props(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(0,),
        min_steps=1,
        prop_factory=props.Prop,
        disable_props=True,
    )
    env = composer.Environment(task=task)

    observation = env.reset().observation
    # Test that the prop observations are empty.
    for key in REFERENCE_PROP_KEYS:
      self.assertIn(key, task.observables)
      self.assertSequenceEqual(observation[key].shape, (1, 0))
    # Test that the props and ghost props are not constructed.
    self.assertEmpty(task._props)
    self.assertEmpty(task._ghost_props)

  def test_prop_termination(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(0,),
        min_steps=1,
        disable_props=False,
        prop_factory=props.Prop,
    )
    env = composer.Environment(task=task)
    observation = env.reset().observation

    # Test that prop position contributes to prop termination error.
    task._set_walker(env.physics)
    wrong_position = observation[REFERENCE_PROP_KEYS[0]] + np.ones(3)
    task._props[0].set_pose(env.physics, wrong_position)
    task.after_step(env.physics, 0)
    task._compute_termination_error()
    self.assertGreater(task._prop_termination_error, 0.)
    task.get_reward(env.physics)
    self.assertEqual(task._should_truncate, True)

  def test_ghost_walker(self):
    task = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(0,),
        min_steps=1,
        ghost_offset=None,
    )
    env = composer.Environment(task=task)
    task_with_ghost = tracking.MultiClipMocapTracking(
        walker=self.walker,
        arena=self.arena,
        ref_path=self.test_data,
        dataset=types.ClipCollection(ids=('cmuv2019_001', 'cmuv2019_002')),
        ref_steps=(0,),
        min_steps=1,
        ghost_offset=GHOST_OFFSET,
    )
    env_with_ghost = composer.Environment(task=task_with_ghost)
    # Test that the ghost does not introduce additional actions.
    self.assertEqual(env_with_ghost.action_spec(), env.action_spec())


if __name__ == '__main__':
  absltest.main()
