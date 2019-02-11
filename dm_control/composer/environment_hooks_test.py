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

"""Tests for Entity and Task hooks in an Environment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

# Internal dependencies.
from absl.testing import absltest

from dm_control import composer
from dm_control import mjcf
from six.moves import range


class HooksTracker(object):

  def __init__(self, test_case, physics_timestep, control_timestep,
               *args, **kwargs):
    super(HooksTracker, self).__init__(*args, **kwargs)
    self.tracked = False
    self._test_case = test_case
    self._call_count = collections.defaultdict(lambda: 0)
    self._physics_timestep = physics_timestep
    self._physics_steps_per_control_step = (
        round(int(control_timestep / physics_timestep)))

  def assertEqual(self, actual, expected, msg=''):
    msg = '{}: {}: {!r} != {!r}'.format(type(self), msg, actual, expected)
    self._test_case.assertEqual(actual, expected, msg)

  def assertHooksNotCalled(self, *hook_names):
    for hook_name in hook_names:
      self.assertEqual(
          self._call_count[hook_name], 0,
          'assertHooksNotCalled: hook_name = {!r}'.format(hook_name))

  def assertHooksCalledOnce(self, *hook_names):
    for hook_name in hook_names:
      self.assertEqual(
          self._call_count[hook_name], 1,
          'assertHooksCalledOnce: hook_name = {!r}'.format(hook_name))

  def assertCompleteEpisode(self, control_steps):
    self.assertHooksCalledOnce('initialize_episode_mjcf',
                               'after_compile',
                               'initialize_episode')
    physics_steps = control_steps * self._physics_steps_per_control_step
    self.assertEqual(self._call_count['before_step'], control_steps)
    self.assertEqual(self._call_count['before_substep'], physics_steps)
    self.assertEqual(self._call_count['after_substep'], physics_steps)
    self.assertEqual(self._call_count['after_step'], control_steps)

  def assertPhysicsStepCountEqual(self, physics, expected_count):
    actual_count = int(round(physics.time() / self._physics_timestep))
    self.assertEqual(actual_count, expected_count)

  def reset_call_counts(self):
    self._call_count = collections.defaultdict(lambda: 0)

  def initialize_episode_mjcf(self, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksNotCalled('after_compile',
                              'initialize_episode',
                              'before_step',
                              'before_substep',
                              'after_substep',
                              'after_step')
    self._call_count['initialize_episode_mjcf'] += 1

  def after_compile(self, physics, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksCalledOnce('initialize_episode_mjcf')
    self.assertHooksNotCalled('initialize_episode',
                              'before_step',
                              'before_substep',
                              'after_substep',
                              'after_step')
    # Number of physics steps is always consistent with `before_substep`.
    self.assertPhysicsStepCountEqual(physics,
                                     self._call_count['before_substep'])
    self._call_count['after_compile'] += 1

  def initialize_episode(self, physics, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksCalledOnce('initialize_episode_mjcf',
                               'after_compile')
    self.assertHooksNotCalled('before_step',
                              'before_substep',
                              'after_substep',
                              'after_step')
    # Number of physics steps is always consistent with `before_substep`.
    self.assertPhysicsStepCountEqual(physics,
                                     self._call_count['before_substep'])
    self._call_count['initialize_episode'] += 1

  def before_step(self, physics, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksCalledOnce('initialize_episode_mjcf',
                               'after_compile',
                               'initialize_episode')

    # `before_step` is only called in between complete control steps.
    self.assertEqual(
        self._call_count['after_step'], self._call_count['before_step'])

    # Complete control steps imply complete physics steps.
    self.assertEqual(
        self._call_count['after_substep'], self._call_count['before_substep'])

    # Number of physics steps is always consistent with `before_substep`.
    self.assertPhysicsStepCountEqual(physics,
                                     self._call_count['before_substep'])

    self._call_count['before_step'] += 1

  def before_substep(self, physics, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksCalledOnce('initialize_episode_mjcf',
                               'after_compile',
                               'initialize_episode')

    # We are inside a partial control step, so `after_step` should lag behind.
    self.assertEqual(
        self._call_count['after_step'], self._call_count['before_step'] - 1)

    # `before_substep` is only called in between complete physics steps.
    self.assertEqual(
        self._call_count['after_substep'], self._call_count['before_substep'])

    # Number of physics steps is always consistent with `before_substep`.
    self.assertPhysicsStepCountEqual(
        physics, self._call_count['before_substep'])

    self._call_count['before_substep'] += 1

  def after_substep(self, physics, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksCalledOnce('initialize_episode_mjcf',
                               'after_compile',
                               'initialize_episode')

    # We are inside a partial control step, so `after_step` should lag behind.
    self.assertEqual(
        self._call_count['after_step'], self._call_count['before_step'] - 1)

    # We are inside a partial physics step, so `after_substep` should be behind.
    self.assertEqual(self._call_count['after_substep'],
                     self._call_count['before_substep'] - 1)

    # Number of physics steps is always consistent with `before_substep`.
    self.assertPhysicsStepCountEqual(
        physics, self._call_count['before_substep'])

    self._call_count['after_substep'] += 1

  def after_step(self, physics, random_state):
    del random_state  # Unused.
    if not self.tracked:
      return
    self.assertHooksCalledOnce('initialize_episode_mjcf',
                               'after_compile',
                               'initialize_episode')

    # We are inside a partial control step, so `after_step` should lag behind.
    self.assertEqual(
        self._call_count['after_step'], self._call_count['before_step'] - 1)

    # `after_step` is only called in between complete physics steps.
    self.assertEqual(
        self._call_count['after_substep'], self._call_count['before_substep'])

    # Number of physics steps is always consistent with `before_substep`.
    self.assertPhysicsStepCountEqual(
        physics, self._call_count['before_substep'])

    # Check that the number of physics steps is consistent with control steps.
    self.assertEqual(
        self._call_count['before_substep'],
        self._call_count['before_step'] * self._physics_steps_per_control_step)

    self._call_count['after_step'] += 1


class TrackedEntity(HooksTracker, composer.Entity):

  def _build(self, name):
    self._mjcf_root = mjcf.RootElement(model=name)

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def name(self):
    return self._mjcf_root.model


class TrackedTask(HooksTracker, composer.NullTask):

  def __init__(self, physics_timestep, control_timestep, *args, **kwargs):
    super(TrackedTask, self).__init__(physics_timestep=physics_timestep,
                                      control_timestep=control_timestep,
                                      *args, **kwargs)
    self.set_timesteps(physics_timestep=physics_timestep,
                       control_timestep=control_timestep)

  def before_step(self, physics, unused_actions, random_state):
    super(TrackedTask, self).before_step(physics, random_state)

  def before_substep(self, physics, unused_actions, random_state):
    super(TrackedTask, self).before_substep(physics, random_state)


class TrackedExtraHooks(HooksTracker):

  def before_step(self, physics, unused_actions, random_state):
    super(TrackedExtraHooks, self).before_step(physics, random_state)

  def before_substep(self, physics, unused_actions, random_state):
    super(TrackedExtraHooks, self).before_substep(physics, random_state)


class HooksTestMixin(object):

  def setUp(self):
    self.num_episodes = 5
    self.steps_per_episode = 100

    self.control_timestep = 0.05
    self.physics_timestep = 0.002

    self.extra_hooks = TrackedExtraHooks(physics_timestep=self.physics_timestep,
                                         control_timestep=self.control_timestep,
                                         test_case=self)

    self.entities = []
    for i in range(9):
      self.entities.append(TrackedEntity(name='entity_{}'.format(i),
                                         physics_timestep=self.physics_timestep,
                                         control_timestep=self.control_timestep,
                                         test_case=self))

    ########################################
    # Make the following entity hierarchy  #
    #                  0                   #
    #       1          2          3        #
    #     4   5      6   7                 #
    #     8                                #
    ########################################

    self.entities[4].attach(self.entities[8])
    self.entities[1].attach(self.entities[4])
    self.entities[1].attach(self.entities[5])
    self.entities[0].attach(self.entities[1])

    self.entities[2].attach(self.entities[6])
    self.entities[2].attach(self.entities[7])
    self.entities[0].attach(self.entities[2])

    self.entities[0].attach(self.entities[3])

    self.task = TrackedTask(root_entity=self.entities[0],
                            physics_timestep=self.physics_timestep,
                            control_timestep=self.control_timestep,
                            test_case=self)

  @contextlib.contextmanager
  def track_episode(self):
    tracked_objects = [self.task, self.extra_hooks] + self.entities
    for obj in tracked_objects:
      obj.reset_call_counts()
      obj.tracked = True
    yield
    for obj in tracked_objects:
      obj.assertCompleteEpisode(self.steps_per_episode)
      obj.tracked = False


class EnvironmentHooksTest(HooksTestMixin, absltest.TestCase):

  def testEnvironmentHooksScheduling(self):
    env = composer.Environment(self.task)
    for hook_name in composer.HOOK_NAMES:
      env.add_extra_hook(hook_name, getattr(self.extra_hooks, hook_name))
    for _ in range(self.num_episodes):
      with self.track_episode():
        env.reset()
        for _ in range(self.steps_per_episode):
          env.step([])


if __name__ == '__main__':
  absltest.main()
