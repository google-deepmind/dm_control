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
"""Tests for the Rodent."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.observation.observable import base as observable_base
from dm_control.locomotion.arenas import corridors as corr_arenas
from dm_control.locomotion.tasks import corridors as corr_tasks
from dm_control.locomotion.walkers import rodent

import numpy as np
from six.moves import range

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.001


def _get_rat_corridor_physics():
  walker = rodent.Rat()
  arena = corr_arenas.EmptyCorridor()
  task = corr_tasks.RunThroughCorridor(
      walker=walker,
      arena=arena,
      walker_spawn_position=(5, 0, 0),
      walker_spawn_rotation=0,
      physics_timestep=_PHYSICS_TIMESTEP,
      control_timestep=_CONTROL_TIMESTEP)

  env = composer.Environment(
      time_limit=30,
      task=task,
      strip_singleton_obs_buffer_dim=True)

  return walker, env


class RatTest(parameterized.TestCase):

  def test_can_compile_and_step_simulation(self):
    _, env = _get_rat_corridor_physics()
    physics = env.physics
    for _ in range(100):
      physics.step()

  @parameterized.parameters([
      'egocentric_camera',
      'head',
      'left_arm_root',
      'right_arm_root',
      'root_body',
      'pelvis_body',
  ])
  def test_get_element_property(self, name):
    attribute_value = getattr(rodent.Rat(), name)
    self.assertIsInstance(attribute_value, mjcf.Element)

  @parameterized.parameters([
      'actuators',
      'bodies',
      'mocap_tracking_bodies',
      'end_effectors',
      'mocap_joints',
      'observable_joints',
  ])
  def test_get_element_tuple_property(self, name):
    attribute_value = getattr(rodent.Rat(), name)
    self.assertNotEmpty(attribute_value)
    for item in attribute_value:
      self.assertIsInstance(item, mjcf.Element)

  def test_set_name(self):
    name = 'fred'
    walker = rodent.Rat(name=name)
    self.assertEqual(walker.mjcf_model.model, name)

  @parameterized.parameters(
      'tendons_pos',
      'tendons_vel',
      'actuator_activation',
      'appendages_pos',
      'head_height',
      'sensors_torque',
  )
  def test_evaluate_observable(self, name):
    walker, env = _get_rat_corridor_physics()
    physics = env.physics
    observable = getattr(walker.observables, name)
    observation = observable(physics)
    self.assertIsInstance(observation, (float, np.ndarray))

  def test_proprioception(self):
    walker = rodent.Rat()
    for item in walker.observables.proprioception:
      self.assertIsInstance(item, observable_base.Observable)

  def test_can_create_two_rats(self):
    rat1 = rodent.Rat(name='rat1')
    rat2 = rodent.Rat(name='rat2')
    arena = corr_arenas.EmptyCorridor()
    arena.add_free_entity(rat1)
    arena.add_free_entity(rat2)
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)  # Should not raise an error.

    rat1.mjcf_model.model = 'rat3'
    rat2.mjcf_model.model = 'rat4'
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)  # Should not raise an error.

if __name__ == '__main__':
  absltest.main()
