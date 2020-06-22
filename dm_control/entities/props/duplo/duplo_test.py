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

"""Tests for the Duplo prop."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.entities.props import duplo
from dm_control.entities.props.duplo import utils
from dm_control.mujoco.wrapper import mjbindings
import numpy as np
from six.moves import range

mjlib = mjbindings.mjlib

# Expected separation force when `variation == 0`
EXPECTED_FIXED_FORCE = 10.0
EXPECTED_FIXED_FORCE_TOL = 0.5

# Bounds and median are based on empirical distribution of separation forces
# for real Duplo blocks.
EXPECTED_MIN_FORCE = 6.
EXPECTED_MAX_FORCE = 18.
EXPECTED_MEDIAN_FORCE = 12.
EXPECTED_MEDIAN_FORCE_TOL = 2.


class DuploTest(parameterized.TestCase):
  """Tests for the Duplo prop."""

  def make_bricks(self, seed, *args, **kwargs):
    top_brick = duplo.Duplo(*args, **kwargs)
    bottom_brick = duplo.Duplo(*args, **kwargs)
    # This sets the radius of the studs. NB: we do this for both bricks because
    # the stud radius has a (tiny!) effect on the mass of the top brick.
    top_brick.initialize_episode_mjcf(np.random.RandomState(seed))
    bottom_brick.initialize_episode_mjcf(np.random.RandomState(seed))
    return top_brick, bottom_brick

  def measure_separation_force(self, seed, *args, **kwargs):
    top_brick, bottom_brick = self.make_bricks(seed=seed, *args, **kwargs)
    return utils.measure_separation_force(top_brick, bottom_brick)

  @parameterized.parameters([p._asdict() for p in duplo._STUD_SIZE_PARAMS])
  def test_separation_force_fixed(self, easy_align, flanges):
    forces = []
    for seed in range(3):
      forces.append(self.measure_separation_force(
          seed=seed, easy_align=easy_align, flanges=flanges, variation=0.0))

    # Separation forces should all be identical since variation == 0.0.
    np.testing.assert_array_equal(forces[0], forces[1:])

    # Separation forces should be close to the reference value.
    self.assertAlmostEqual(forces[0], EXPECTED_FIXED_FORCE,
                           delta=EXPECTED_FIXED_FORCE_TOL)

  @parameterized.parameters([p._asdict() for p in duplo._STUD_SIZE_PARAMS])
  def test_separation_force_distribution(self, easy_align, flanges):
    forces = []
    for seed in range(10):
      forces.append(self.measure_separation_force(
          seed=seed, easy_align=easy_align, flanges=flanges, variation=1.0))

    self.assertGreater(min(forces), EXPECTED_MIN_FORCE)
    self.assertLess(max(forces), EXPECTED_MAX_FORCE)
    median_force = np.median(forces)
    median_force_delta = median_force - EXPECTED_MEDIAN_FORCE
    self.assertLess(
        abs(median_force_delta), EXPECTED_MEDIAN_FORCE_TOL,
        msg=('Expected median separation force to be {}+/-{} N, got {} N.'
             .format(EXPECTED_MEDIAN_FORCE, EXPECTED_MEDIAN_FORCE_TOL,
                     median_force)))

  @parameterized.parameters([p._asdict() for p in duplo._STUD_SIZE_PARAMS])
  def test_separation_force_identical_with_same_seed(self, easy_align, flanges):
    def measure(seed):
      return self.measure_separation_force(
          seed=seed, easy_align=easy_align, flanges=flanges, variation=1.0)

    first = measure(seed=0)
    second = measure(seed=0)
    third = measure(seed=1)

    self.assertEqual(first, second)
    self.assertNotEqual(first, third)

  def test_exception_if_color_out_of_range(self):
    invalid_color = (1., 0., 2.)
    expected_message = duplo._COLOR_NOT_BETWEEN_0_AND_1.format(invalid_color)
    with self.assertRaisesWithLiteralMatch(ValueError, expected_message):
      _ = duplo.Duplo(color=invalid_color)

  @parameterized.parameters([p._asdict() for p in duplo._STUD_SIZE_PARAMS])
  def test_stud_and_hole_sites_align_when_stacked(self, easy_align, flanges):
    top_brick, bottom_brick = self.make_bricks(
        easy_align=easy_align, flanges=flanges, seed=0)
    arena, _ = utils.stack_bricks(top_brick, bottom_brick)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    # Step the physics a few times to allow it to settle.
    for _ in range(10):
      physics.step()
    # When two bricks are stacked, the studs on the bottom brick should align
    # precisely with the holes on the top brick.
    bottom_stud_pos = physics.bind(bottom_brick.studs.ravel()).xpos
    top_hole_pos = physics.bind(top_brick.holes.ravel()).xpos
    np.testing.assert_allclose(bottom_stud_pos, top_hole_pos, atol=1e-6)

  # TODO(b/120829077): Extend this test to other brick configurations.
  def test_correct_stud_contacts(self):
    top_brick, bottom_brick = self.make_bricks(seed=0)
    arena, _ = utils.stack_bricks(top_brick, bottom_brick)
    physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
    # Step the physics a few times to allow it to settle.
    for _ in range(10):
      physics.step()

    # Each stud should make 3 contacts - two with flanges, one with a tube.
    expected_contacts_per_stud = 3

    for stud_site in bottom_brick.studs.flat:
      stud_geom = bottom_brick.mjcf_model.find('geom', stud_site.name)
      geom_id = physics.bind(stud_geom).element_id

      # Check that this stud participates in the expected number of contacts.
      stud_contacts = ((physics.data.contact.geom1 == geom_id) ^
                       (physics.data.contact.geom2 == geom_id))
      self.assertEqual(stud_contacts.sum(), expected_contacts_per_stud)

      # The normal forces should be roughly equal across contacts.
      normal_forces = []
      for contact_id in np.where(stud_contacts)[0]:
        all_forces = np.empty(6)
        mjlib.mj_contactForce(physics.model.ptr, physics.data.ptr,
                              contact_id, all_forces)
        # all_forces is [normal, tangent, tangent, torsion, rolling, rolling]
        normal_forces.append(all_forces[0])
      np.testing.assert_allclose(
          normal_forces[0], normal_forces[1:], rtol=0.05)

if __name__ == '__main__':
  absltest.main()
