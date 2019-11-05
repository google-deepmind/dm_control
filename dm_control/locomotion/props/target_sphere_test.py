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
"""Tests for props.target_sphere."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from dm_control import composer

from dm_control.entities.props import primitive
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.props import target_sphere


class TargetSphereTest(absltest.TestCase):

  def testActivation(self):
    target_radius = 0.6
    prop_radius = 0.1
    target_height = 1

    arena = floors.Floor()
    target = target_sphere.TargetSphere(radius=target_radius,
                                        height_above_ground=target_height)
    prop = primitive.Primitive(geom_type='sphere', size=[prop_radius])
    arena.attach(target)
    arena.add_free_entity(prop)

    task = composer.NullTask(arena)
    task.initialize_episode = (
        lambda physics, random_state: prop.set_pose(physics, [0, 0, 2]))

    env = composer.Environment(task)
    env.reset()

    max_activated_height = target_height + target_radius + prop_radius

    while env.physics.bind(prop.geom).xpos[2] > max_activated_height:
      self.assertFalse(target.activated)
      self.assertEqual(env.physics.bind(target.material).rgba[-1], 1)
      env.step([])

    while env.physics.bind(prop.geom).xpos[2] > 0.2:
      self.assertTrue(target.activated)
      self.assertEqual(env.physics.bind(target.material).rgba[-1], 0)
      env.step([])

    # Target should be reset when the environment is reset.
    env.reset()
    self.assertFalse(target.activated)
    self.assertEqual(env.physics.bind(target.material).rgba[-1], 1)


if __name__ == '__main__':
  absltest.main()
