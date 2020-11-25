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

"""Tests for locomotion.arenas.corridors."""


from absl.testing import absltest
from absl.testing import parameterized
from dm_control import mjcf
from dm_control.composer.variation import deterministic
from dm_control.locomotion.arenas import corridors


class CorridorsTest(parameterized.TestCase):

  @parameterized.parameters([
      corridors.EmptyCorridor,
      corridors.GapsCorridor,
      corridors.WallsCorridor,
  ])
  def test_can_compile_mjcf(self, arena_type):
    arena = arena_type()
    mjcf.Physics.from_mjcf_model(arena.mjcf_model)

  @parameterized.parameters([
      corridors.EmptyCorridor,
      corridors.GapsCorridor,
      corridors.WallsCorridor,
  ])
  def test_can_regenerate_corridor_size(self, arena_type):
    width_sequence = [5.2, 3.8, 7.4]
    length_sequence = [21.1, 19.4, 16.3]

    arena = arena_type(
        corridor_width=deterministic.Sequence(width_sequence),
        corridor_length=deterministic.Sequence(length_sequence))

    # Add a probe geom that will generate contacts with the side walls.
    probe_body = arena.mjcf_model.worldbody.add('body', name='probe')
    probe_joint = probe_body.add('freejoint')
    probe_geom = probe_body.add('geom', name='probe', type='box')

    for expected_width, expected_length in zip(width_sequence, length_sequence):
      # No random_state is required since we are using deterministic variations.
      arena.regenerate(random_state=None)

      def resize_probe_geom_and_assert_num_contacts(
          delta_size, expected_num_contacts,
          expected_width=expected_width, expected_length=expected_length):
        probe_geom.size = [
            (expected_length / 2 + corridors._CORRIDOR_X_PADDING) + delta_size,
            expected_width / 2 + delta_size, 0.1]
        physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
        probe_geomid = physics.bind(probe_geom).element_id
        physics.bind(probe_joint).qpos[:3] = [expected_length / 2, 0, 100]
        physics.forward()
        probe_contacts = [c for c in physics.data.contact
                          if c.geom1 == probe_geomid or c.geom2 == probe_geomid]
        self.assertLen(probe_contacts, expected_num_contacts)

      epsilon = 1e-7

      # If the probe geom is epsilon-smaller than the expected corridor size,
      # then we expect to detect no contact.
      resize_probe_geom_and_assert_num_contacts(-epsilon, 0)

      # If the probe geom is epsilon-larger than the expected corridor size,
      # then we expect to generate 4 contacts with each side wall, so 16 total.
      resize_probe_geom_and_assert_num_contacts(epsilon, 16)


if __name__ == '__main__':
  absltest.main()
