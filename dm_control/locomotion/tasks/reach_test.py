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
"""Tests for locomotion.tasks.reach."""

import functools
from absl.testing import absltest

from dm_control import composer
from dm_control.locomotion.arenas import floors
from dm_control.locomotion.props import target_sphere
from dm_control.locomotion.tasks import reach
from dm_control.locomotion.walkers import rodent
import numpy as np

_CONTROL_TIMESTEP = .02
_PHYSICS_TIMESTEP = 0.001


class ReachTest(absltest.TestCase):

  def test_observables(self):
    walker = rodent.Rat()

    arena = floors.Floor(
        size=(10., 10.),
        aesthetic='outdoor_natural')

    task = reach.TwoTouch(
        walker=walker,
        arena=arena,
        target_builders=[
            functools.partial(target_sphere.TargetSphereTwoTouch, radius=0.025),
        ],
        randomize_spawn_rotation=True,
        target_type_rewards=[25.],
        shuffle_target_builders=False,
        target_area=(1.5, 1.5),
        physics_timestep=_PHYSICS_TIMESTEP,
        control_timestep=_CONTROL_TIMESTEP,
    )
    random_state = np.random.RandomState(12345)
    env = composer.Environment(task, random_state=random_state)
    timestep = env.reset()

    self.assertIn('walker/joints_pos', timestep.observation)


if __name__ == '__main__':
  absltest.main()
