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

"""Tests for locomotion.tasks.go_to_target."""


from absl.testing import absltest

from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.tasks import go_to_target
from dm_control.locomotion.walkers import cmu_humanoid
import numpy as np
from six.moves import range


class GoToTargetTest(absltest.TestCase):

  def test_observables(self):
    walker = cmu_humanoid.CMUHumanoid()
    arena = floors.Floor()
    task = go_to_target.GoToTarget(
        walker=walker, arena=arena, moving_target=False)

    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    timestep = env.reset()

    self.assertIn('walker/target', timestep.observation)

  def test_target_position_randomized_on_reset(self):
    walker = cmu_humanoid.CMUHumanoid()
    arena = floors.Floor()
    task = go_to_target.GoToTarget(
        walker=walker, arena=arena, moving_target=False)
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()
    first_target_position = task.target_position(env.physics)
    env.reset()
    second_target_position = task.target_position(env.physics)
    self.assertFalse(np.all(first_target_position == second_target_position),
                     'Target positions are unexpectedly identical.')

  def test_reward_fixed_target(self):
    walker = cmu_humanoid.CMUHumanoid()
    arena = floors.Floor()
    task = go_to_target.GoToTarget(
        walker=walker, arena=arena, moving_target=False)

    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()

    target_position = task.target_position(env.physics)
    zero_action = np.zeros_like(env.physics.data.ctrl)
    for _ in range(2):
      timestep = env.step(zero_action)
      self.assertEqual(timestep.reward, 0)
    walker_pos = env.physics.bind(walker.root_body).xpos
    walker.set_pose(
        env.physics,
        position=[target_position[0], target_position[1], walker_pos[2]])
    env.physics.forward()

    # Receive reward while the agent remains at that location.
    timestep = env.step(zero_action)
    self.assertEqual(timestep.reward, 1)

    # Target position should not change.
    np.testing.assert_array_equal(target_position,
                                  task.target_position(env.physics))

  def test_reward_moving_target(self):
    walker = cmu_humanoid.CMUHumanoid()
    arena = floors.Floor()

    steps_before_moving_target = 2
    task = go_to_target.GoToTarget(
        walker=walker,
        arena=arena,
        moving_target=True,
        steps_before_moving_target=steps_before_moving_target)
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()

    target_position = task.target_position(env.physics)
    zero_action = np.zeros_like(env.physics.data.ctrl)
    for _ in range(2):
      timestep = env.step(zero_action)
      self.assertEqual(timestep.reward, 0)

    walker_pos = env.physics.bind(walker.root_body).xpos
    walker.set_pose(
        env.physics,
        position=[target_position[0], target_position[1], walker_pos[2]])
    env.physics.forward()

    # Receive reward while the agent remains at that location.
    for _ in range(steps_before_moving_target):
      timestep = env.step(zero_action)
      self.assertEqual(timestep.reward, 1)
      np.testing.assert_array_equal(target_position,
                                    task.target_position(env.physics))

    # After taking > steps_before_moving_target, the target should move and
    # reward should be 0.
    timestep = env.step(zero_action)
    self.assertEqual(timestep.reward, 0)

  def test_termination_and_discount(self):
    walker = cmu_humanoid.CMUHumanoid()
    arena = floors.Floor()
    task = go_to_target.GoToTarget(walker=walker, arena=arena)

    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()

    zero_action = np.zeros_like(env.physics.data.ctrl)

    # Walker starts in upright position.
    # Should not trigger failure termination in the first few steps.
    for _ in range(5):
      env.step(zero_action)
      self.assertFalse(task.should_terminate_episode(env.physics))
      np.testing.assert_array_equal(task.get_discount(env.physics), 1)

    # Rotate the walker upside down and run the physics until it makes contact.
    current_time = env.physics.data.time
    walker.shift_pose(env.physics, position=(0, 0, 10), quaternion=(0, 1, 0, 0))
    env.physics.forward()
    while env.physics.data.ncon == 0:
      env.physics.step()
    env.physics.data.time = current_time

    # Should now trigger a failure termination.
    env.step(zero_action)
    self.assertTrue(task.should_terminate_episode(env.physics))
    np.testing.assert_array_equal(task.get_discount(env.physics), 0)


if __name__ == '__main__':
  absltest.main()
