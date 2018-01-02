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

"""Tests to check whether methods of `mujoco.Physics` are threadsafe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest

from dm_control.mujoco import engine
from dm_control.mujoco.testing import assets
from dm_control.mujoco.testing import decorators

MODEL = assets.get_contents('cartpole.xml')
NUM_STEPS = 10


class ThreadSafetyTest(absltest.TestCase):

  @decorators.run_threaded()
  def test_load_physics_from_string(self):
    engine.Physics.from_xml_string(MODEL)

  @decorators.run_threaded()
  def test_load_and_reload_physics_from_string(self):
    physics = engine.Physics.from_xml_string(MODEL)
    physics.reload_from_xml_string(MODEL)

  @decorators.run_threaded()
  def test_load_and_step_physics(self):
    physics = engine.Physics.from_xml_string(MODEL)
    for _ in xrange(NUM_STEPS):
      physics.step()

  @decorators.run_threaded()
  def test_load_and_step_multiple_physics_parallel(self):
    physics1 = engine.Physics.from_xml_string(MODEL)
    physics2 = engine.Physics.from_xml_string(MODEL)
    for _ in xrange(NUM_STEPS):
      physics1.step()
      physics2.step()

  @decorators.run_threaded()
  def test_load_and_step_multiple_physics_sequential(self):
    physics1 = engine.Physics.from_xml_string(MODEL)
    for _ in xrange(NUM_STEPS):
      physics1.step()
    del physics1
    physics2 = engine.Physics.from_xml_string(MODEL)
    for _ in xrange(NUM_STEPS):
      physics2.step()

  @decorators.run_threaded(calls_per_thread=5)
  def test_load_physics_and_render(self):
    physics = engine.Physics.from_xml_string(MODEL)

    # Check that frames aren't repeated - make the cartpole move.
    physics.set_control([1.0])

    unique_frames = set()
    for _ in xrange(NUM_STEPS):
      physics.step()
      frame = physics.render(width=320, height=240, camera_id=0)
      unique_frames.add(frame.tostring())

    self.assertEqual(NUM_STEPS, len(unique_frames))

  @decorators.run_threaded(calls_per_thread=5)
  def test_render_multiple_physics_instances_per_thread_parallel(self):
    physics1 = engine.Physics.from_xml_string(MODEL)
    physics2 = engine.Physics.from_xml_string(MODEL)
    for _ in xrange(NUM_STEPS):
      physics1.step()
      physics1.render(width=320, height=240, camera_id=0)
      physics2.step()
      physics2.render(width=320, height=240, camera_id=0)


if __name__ == '__main__':
  absltest.main()
