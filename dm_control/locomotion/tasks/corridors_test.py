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

"""Tests for dm_control.locomotion.tasks.corridors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import rotations
from dm_control.locomotion.arenas import corridors as corridor_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks
from dm_control.locomotion.walkers import cmu_humanoid
import numpy as np
from six.moves import range


class CorridorsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(position_offset=(0, 0, 0),
           rotate_180_degrees=False,
           use_variations=False),
      dict(position_offset=(1, 2, 3),
           rotate_180_degrees=True,
           use_variations=True))
  def test_walker_is_correctly_reinitialized(
      self, position_offset, rotate_180_degrees, use_variations):
    walker_spawn_position = position_offset

    if not rotate_180_degrees:
      walker_spawn_rotation = None
    else:
      walker_spawn_rotation = np.pi

    if use_variations:
      walker_spawn_position = deterministic.Constant(position_offset)
      walker_spawn_rotation = deterministic.Constant(walker_spawn_rotation)

    walker = cmu_humanoid.CMUHumanoid()
    arena = corridor_arenas.EmptyCorridor()
    task = corridor_tasks.RunThroughCorridor(
        walker=walker,
        arena=arena,
        walker_spawn_position=walker_spawn_position,
        walker_spawn_rotation=walker_spawn_rotation)

    # Randomize the initial pose and joint positions in order to check that they
    # are set correctly by `initialize_episode`.
    random_state = np.random.RandomState(12345)
    task.initialize_episode_mjcf(random_state)
    physics = mjcf.Physics.from_mjcf_model(task.root_entity.mjcf_model)

    walker_joints = walker.mjcf_model.find_all('joint')
    physics.bind(walker_joints).qpos = random_state.uniform(
        size=len(walker_joints))
    walker.set_pose(physics,
                    position=random_state.uniform(size=3),
                    quaternion=rotations.UniformQuaternion()(random_state))

    task.initialize_episode(physics, random_state)
    physics.forward()

    with self.subTest('Correct joint positions'):
      walker_qpos = physics.bind(walker_joints).qpos
      if walker.upright_pose.qpos is not None:
        np.testing.assert_array_equal(walker_qpos, walker.upright_pose.qpos)
      else:
        walker_qpos0 = physics.bind(walker_joints).qpos0
        np.testing.assert_array_equal(walker_qpos, walker_qpos0)

    walker_xpos, walker_xquat = walker.get_pose(physics)

    with self.subTest('Correct position'):
      expected_xpos = walker.upright_pose.xpos + np.array(position_offset)
      np.testing.assert_array_equal(walker_xpos, expected_xpos)

    with self.subTest('Correct orientation'):
      upright_xquat = walker.upright_pose.xquat.copy()
      upright_xquat /= np.linalg.norm(walker.upright_pose.xquat)
      if rotate_180_degrees:
        expected_xquat = (-upright_xquat[3], -upright_xquat[2],
                          upright_xquat[1], upright_xquat[0])
      else:
        expected_xquat = upright_xquat
      np.testing.assert_allclose(walker_xquat, expected_xquat)

  def test_termination_and_discount(self):
    walker = cmu_humanoid.CMUHumanoid()
    arena = corridor_arenas.EmptyCorridor()
    task = corridor_tasks.RunThroughCorridor(walker, arena)

    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    env.reset()

    zero_action = np.zeros_like(env.physics.data.ctrl)

    # Walker starts in upright position.
    # Should not trigger failure termination in the first few steps.
    for _ in range(5):
      env.step(zero_action)
      self.assertFalse(task.should_terminate_episode(env.physics))
      self.assertEqual(task.get_discount(env.physics), 1)

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
    self.assertEqual(task.get_discount(env.physics), 0)


if __name__ == '__main__':
  absltest.main()
