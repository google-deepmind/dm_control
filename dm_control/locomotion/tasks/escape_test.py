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
"""Tests for locomotion.tasks.escape."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest

from dm_control import composer
from dm_control.locomotion.arenas import bowl
from dm_control.locomotion.tasks import escape
from dm_control.locomotion.walkers import rodent

import numpy as np
from six.moves import range

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.001


class EscapeTest(absltest.TestCase):

  def test_observables(self):
    walker = rodent.Rat()

    # Build a corridor-shaped arena that is obstructed by walls.
    arena = bowl.Bowl(
        size=(20., 20.),
        aesthetic='outdoor_natural')

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = escape.Escape(
        walker=walker,
        arena=arena,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    timestep = env.reset()

    self.assertIn('walker/joints_pos', timestep.observation)

  def test_contact(self):
    walker = rodent.Rat()

    # Build a corridor-shaped arena that is obstructed by walls.
    arena = bowl.Bowl(
        size=(20., 20.),
        aesthetic='outdoor_natural')

    # Build a task that rewards the agent for running down the corridor at a
    # specific velocity.
    task = escape.Escape(
        walker=walker,
        arena=arena,
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP)

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


if __name__ == '__main__':
  absltest.main()
