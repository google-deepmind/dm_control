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

"""Tests for prop_initializer."""

from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.composer.initializers import prop_initializer
from dm_control.composer.variation import deterministic
from dm_control.composer.variation import distributions
from dm_control.entities import props
from dm_control.rl import control
import numpy as np


class _SequentialChoice(distributions.Distribution):
  """Helper class to return samples in order for deterministic testing."""
  __slots__ = ()

  def __init__(self, choices, single_sample=False):
    super().__init__(choices, single_sample=single_sample)
    self._idx = 0

  def _callable(self, random_state):
    def next_item(*args, **kwargs):
      del args, kwargs  # Unused.
      result = self._args[0][self._idx]
      self._idx = (self._idx + 1) % len(self._args[0])
      return result

    return next_item


def _make_spheres(num_spheres, radius, nconmax):
  spheres = []
  arena = composer.Arena()
  arena.mjcf_model.worldbody.add('geom', type='plane', size=[1, 1, 0.1],
                                 pos=[0., 0., -2 * radius], name='ground')
  for i in range(num_spheres):
    sphere = props.Primitive(
        geom_type='sphere', size=[radius], name='sphere_{}'.format(i))
    arena.add_free_entity(sphere)
    spheres.append(sphere)
  arena.mjcf_model.size.nconmax = nconmax
  physics = mjcf.Physics.from_mjcf_model(arena.mjcf_model)
  return physics, spheres


class PropPlacerTest(parameterized.TestCase):
  """Tests for PropPlacer."""

  def assertNoContactsInvolvingEntities(self, physics, entities):
    all_colliding_geoms = set()
    for contact in physics.data.contact:
      all_colliding_geoms.add(contact.geom1)
      all_colliding_geoms.add(contact.geom2)
    for entity in entities:
      entity_geoms = physics.bind(entity.mjcf_model.find_all('geom')).element_id
      colliding_entity_geoms = all_colliding_geoms.intersection(entity_geoms)
      if colliding_entity_geoms:
        names = ', '.join(
            physics.model.id2name(i, 'geom') for i in colliding_entity_geoms)
        self.fail('Entity {} has colliding geoms: {}'
                  .format(entity.mjcf_model.model, names))

  def assertPositionsWithinBounds(self, physics, entities, lower, upper):
    for entity in entities:
      position, _ = entity.get_pose(physics)
      if np.any(position < lower) or np.any(position > upper):
        self.fail('Entity {} is out of bounds: position={}, bounds={}'
                  .format(entity.mjcf_model.model, position, (lower, upper)))

  def test_sample_non_colliding_positions(self):
    halfwidth = 0.05
    radius = halfwidth / 4.
    offset = np.array([0, 0, halfwidth + radius*1.1])
    lower = -np.full(3, halfwidth) + offset
    upper = np.full(3, halfwidth) + offset
    position_variation = distributions.Uniform(lower, upper)
    physics, spheres = _make_spheres(num_spheres=8, radius=radius, nconmax=1000)
    prop_placer = prop_initializer.PropPlacer(
        props=spheres,
        position=position_variation,
        ignore_collisions=False,
        settle_physics=False)
    prop_placer(physics, random_state=np.random.RandomState(0))
    self.assertNoContactsInvolvingEntities(physics, spheres)
    self.assertPositionsWithinBounds(physics, spheres, lower, upper)

  def test_rejection_sampling_failure(self):
    max_attempts_per_prop = 2
    fixed_position = (0, 0, 0.1)  # Guaranteed to always have collisions.
    physics, spheres = _make_spheres(num_spheres=2, radius=0.01, nconmax=1000)
    prop_placer = prop_initializer.PropPlacer(
        props=spheres,
        position=fixed_position,
        ignore_collisions=False,
        max_attempts_per_prop=max_attempts_per_prop)
    expected_message = prop_initializer._REJECTION_SAMPLING_FAILED.format(
        model_name=spheres[1].mjcf_model.model,  # Props are placed in order.
        max_attempts=max_attempts_per_prop)
    with self.assertRaisesWithLiteralMatch(RuntimeError, expected_message):
      prop_placer(physics, random_state=np.random.RandomState(0))

  def test_ignore_contacts_with_entities(self):
    physics, spheres = _make_spheres(num_spheres=2, radius=0.01, nconmax=1000)

    # Target position of both spheres (non-colliding).
    fixed_positions = [(0, 0, 0.1), (0, 0.1, 0.1)]

    # Placer that initializes both spheres to (0, 0, 0.1), ignoring contacts.
    prop_placer_init = prop_initializer.PropPlacer(
        props=spheres,
        position=fixed_positions[0],
        ignore_collisions=True,
        max_attempts_per_prop=1)

    # Sequence of placers that will move the spheres to their target positions.
    prop_placer_seq = []
    for prop, target_position in zip(spheres, fixed_positions):
      placer = prop_initializer.PropPlacer(
          props=[prop],
          position=target_position,
          ignore_collisions=False,
          max_attempts_per_prop=1)
      prop_placer_seq.append(placer)

    # We expect the first placer in the sequence to fail without
    # `ignore_contacts_with_entities` because the second sphere is already at
    # the same location.
    prop_placer_init(physics, random_state=np.random.RandomState(0))
    expected_message = prop_initializer._REJECTION_SAMPLING_FAILED.format(
        model_name=spheres[0].mjcf_model.model, max_attempts=1)
    with self.assertRaisesWithLiteralMatch(RuntimeError, expected_message):
      prop_placer_seq[0](physics, random_state=np.random.RandomState(0))

    # Placing the first sphere should succeed if we ignore contacts involving
    # the second sphere.
    prop_placer_init(physics, random_state=np.random.RandomState(0))
    prop_placer_seq[0](physics, random_state=np.random.RandomState(0),
                       ignore_contacts_with_entities=[spheres[1]])
    # Now place the second sphere with all collisions active.
    prop_placer_seq[1](physics, random_state=np.random.RandomState(0),
                       ignore_contacts_with_entities=None)
    self.assertNoContactsInvolvingEntities(physics, spheres)

  def test_exception_if_contact_buffer_always_full(self):
    max_attempts_per_prop = 2
    radius = 0.1
    num_spheres = 5
    physics, spheres = _make_spheres(num_spheres=num_spheres,
                                     radius=radius, nconmax=1)

    candidate_positions = np.multiply.outer(
        np.arange(num_spheres * max_attempts_per_prop), [radius * 2.01, 0, 0])

    # If we only place the first sphere then the others will all be overlapping
    # at the origin, so we get an error due to filling the contact buffer.
    prop_placer_failure = prop_initializer.PropPlacer(
        props=[spheres[0]],
        position=deterministic.Sequence(candidate_positions),
        ignore_collisions=False,
        max_attempts_per_prop=max_attempts_per_prop)
    with self.assertRaises(control.PhysicsError):
      prop_placer_failure(physics, random_state=np.random.RandomState(0))

    physics, spheres = _make_spheres(num_spheres=num_spheres,
                                     radius=radius, nconmax=1)

    # If we place all of the spheres then we can find a configuration where they
    # are non-colliding, so the contact buffer is not full when the initializer
    # returns.
    prop_placer = prop_initializer.PropPlacer(
        props=spheres,
        position=deterministic.Sequence(candidate_positions),
        ignore_collisions=False,
        max_attempts_per_prop=max_attempts_per_prop)
    prop_placer(physics, random_state=np.random.RandomState(0))

  def test_no_exception_if_contact_buffer_transiently_full(self):
    max_attempts_per_prop = 2
    radius = 0.1
    num_spheres = 3
    physics, spheres = _make_spheres(num_spheres=num_spheres,
                                     radius=radius, nconmax=1)
    fixed_positions = [[-radius * 1.01, 0., 0],
                       [radius * 1.01, 0., 0.]]
    for sphere, position in zip(spheres[:2], fixed_positions):
      sphere.set_pose(physics, position=position)

    candidate_positions = [
        [0., 0., 0.],  # Collides with both fixed spheres.
        [5 * radius, 0., 0.]]  # Does not collide with either sphere.

    # The first candidate position transiently fills the contact buffer.
    prop_placer = prop_initializer.PropPlacer(
        props=spheres[2:],
        position=deterministic.Sequence(candidate_positions),
        ignore_collisions=False,
        max_attempts_per_prop=max_attempts_per_prop)
    prop_placer(physics, random_state=np.random.RandomState(0))

  @parameterized.parameters([False, True])
  def test_settle_physics(self, settle_physics):
    radius = 0.1
    physics, spheres = _make_spheres(num_spheres=2, radius=radius, nconmax=1)

    # Only place the first sphere.
    prop_placer = prop_initializer.PropPlacer(
        props=spheres[:1],
        position=np.array([2.01 * radius, 0., 0.]),
        settle_physics=settle_physics)
    prop_placer(physics, random_state=np.random.RandomState(0))

    first_position, first_quaternion = spheres[0].get_pose(physics)
    del first_quaternion  # Unused.

    # If we allowed the physics to settle then the first sphere should be
    # resting on the ground, otherwise it should be at the target height.
    expected_first_z_pos = -radius if settle_physics else 0.
    self.assertAlmostEqual(first_position[2], expected_first_z_pos, places=3)

    second_position, second_quaternion = spheres[1].get_pose(physics)
    del second_quaternion  # Unused.

    # The sphere that we were not placing should not have moved.
    self.assertEqual(second_position[2], 0.)

  @parameterized.parameters([0, 1, 2, 3])
  def test_settle_physics_multiple_attempts(self, max_settle_physics_attempts):
    # Tests the multiple-reset mechanism for `settle_physics`.
    # Rather than testing the mechanic itself, which is tested above, we instead
    # test that the mechanism correctly makes several attempts when it fails
    # to settle.  We force it to fail by making the settling time short, and
    # test that the position is repeatedly called using a deterministic
    # sequential pose distribution.

    radius = 0.1
    physics, spheres = _make_spheres(num_spheres=1, radius=radius, nconmax=1)

    # Generate sequence of positions that will be sampled in order.
    positions = [
        np.array([2.01 * radius, 1., 0.]),
        np.array([2.01 * radius, 2., 0.]),
        np.array([2.01 * radius, 3., 0.]),
    ]
    positions_dist = _SequentialChoice(positions)

    def build_placer():
      return prop_initializer.PropPlacer(
          props=spheres[:1],
          position=positions_dist,
          settle_physics=True,
          max_settle_physics_time=1e-6,  # To ensure that settling FAILS.
          max_settle_physics_attempts=max_settle_physics_attempts)

    if max_settle_physics_attempts == 0:
      with self.assertRaises(ValueError):
        build_placer()
    else:
      prop_placer = build_placer()

      prop_placer(physics, random_state=np.random.RandomState(0))

      first_position, first_quaternion = spheres[0].get_pose(physics)
      del first_quaternion  # Unused.

      # If we allowed the physics to settle then the first sphere should be
      # resting on the ground, otherwise it should be at the target height.
      expected_first_y_pos = max_settle_physics_attempts
      self.assertAlmostEqual(first_position[1], expected_first_y_pos, places=3)


if __name__ == '__main__':
  absltest.main()
