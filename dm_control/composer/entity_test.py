# Copyright 2018 The dm_control Authors.
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

"""Tests for composer.Entity."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer import entity
from dm_control.composer.observation.observable import base as observable
import six
from six.moves import range


class TestEntity(entity.Entity):
  """Simple test entity that does nothing but declare some observables."""

  def _build(self, name='test_entity'):
    self._mjcf_root = mjcf.element.RootElement(model=name)

  def _build_observables(self):
    return TestEntityObservables(self)

  @property
  def mjcf_model(self):
    return self._mjcf_root


class TestEntityObservables(entity.Observables):
  """Trivial observables for the test entity."""

  @define.observable
  def observable0(self):
    return observable.Generic(lambda phys: 0.0)

  @define.observable
  def observable1(self):
    return observable.Generic(lambda phys: 1.0)


class EntityTest(absltest.TestCase):

  def setUp(self):
    super(EntityTest, self).setUp()
    self.entity = TestEntity()

  def testNumObservables(self):
    """Tests that the observables dict has the right number of entries."""
    self.assertLen(self.entity.observables.as_dict(), 2)

  def testObservableNames(self):
    """Tests that the observables dict keys correspond to the observable names.
    """
    obs = self.entity.observables.as_dict()
    self.assertIn('observable0', obs)
    self.assertIn('observable1', obs)

    subentity = TestEntity(name='subentity')
    self.entity.attach(subentity)
    self.assertIn('subentity/observable0', subentity.observables.as_dict())
    self.assertEqual(subentity.observables.dict_keys.observable0,
                     'subentity/observable0')
    self.assertIn('observable0', dir(subentity.observables.dict_keys))
    self.assertIn('subentity/observable1', subentity.observables.as_dict())
    self.assertEqual(subentity.observables.dict_keys.observable1,
                     'subentity/observable1')
    self.assertIn('observable1', dir(subentity.observables.dict_keys))

  def testEnableDisableObservables(self):
    """Test the enabling and disable functionality for observables."""
    all_obs = self.entity.observables.as_dict()

    self.entity.observables.enable_all()
    for obs in all_obs.values():
      self.assertTrue(obs.enabled)

    self.entity.observables.disable_all()
    for obs in all_obs.values():
      self.assertFalse(obs.enabled)

    self.entity.observables.observable0.enabled = True
    self.assertTrue(all_obs['observable0'].enabled)

  def testObservableDefaultOptions(self):
    corruptor = lambda x: x
    options = {
        'update_interval': 2,
        'buffer_size': 10,
        'delay': 1,
        'aggregator': 'max',
        'corruptor': corruptor,
        'enabled': True
    }
    self.entity.observables.set_options(options)

    for obs in self.entity.observables.as_dict().values():
      self.assertEqual(obs.update_interval, 2)
      self.assertEqual(obs.delay, 1)
      self.assertEqual(obs.buffer_size, 10)
      self.assertEqual(obs.aggregator, observable.AGGREGATORS['max'])
      self.assertEqual(obs.corruptor, corruptor)
      self.assertTrue(obs.enabled)

  def testObservablePartialDefaultOptions(self):
    options = {'update_interval': 2, 'delay': 1}
    self.entity.observables.set_options(options)

    for obs in self.entity.observables.as_dict().values():
      self.assertEqual(obs.update_interval, 2)
      self.assertEqual(obs.delay, 1)
      self.assertEqual(obs.buffer_size, None)
      self.assertEqual(obs.aggregator, None)
      self.assertEqual(obs.corruptor, None)

  def testObservableOptionsInvalidName(self):
    options = {'asdf': None}
    with six.assertRaisesRegex(
        self, KeyError, 'No observable with name \'asdf\''):
      self.entity.observables.set_options(options)

  def testObservableInvalidOptions(self):
    options = {'observable0': {'asdf': 2}}
    with six.assertRaisesRegex(self, AttributeError,
                               'Cannot add attribute asdf in configure.'):
      self.entity.observables.set_options(options)

  def testObservableOptions(self):
    options = {
        'observable0': {
            'update_interval': 2,
            'delay': 3
        },
        'observable1': {
            'update_interval': 4,
            'delay': 5
        }
    }
    self.entity.observables.set_options(options)
    observables = self.entity.observables.as_dict()
    self.assertEqual(observables['observable0'].update_interval, 2)
    self.assertEqual(observables['observable0'].delay, 3)
    self.assertEqual(observables['observable0'].buffer_size, None)
    self.assertEqual(observables['observable0'].aggregator, None)
    self.assertEqual(observables['observable0'].corruptor, None)
    self.assertFalse(observables['observable0'].enabled)

    self.assertEqual(observables['observable1'].update_interval, 4)
    self.assertEqual(observables['observable1'].delay, 5)
    self.assertEqual(observables['observable1'].buffer_size, None)
    self.assertEqual(observables['observable1'].aggregator, None)
    self.assertEqual(observables['observable1'].corruptor, None)
    self.assertFalse(observables['observable1'].enabled)

  def testObservableOptionsEntityConstructor(self):
    options = {
        'observable0': {
            'update_interval': 2,
            'delay': 3
        },
        'observable1': {
            'update_interval': 4,
            'delay': 5
        }
    }
    ent = TestEntity(observable_options=options)
    observables = ent.observables.as_dict()
    self.assertEqual(observables['observable0'].update_interval, 2)
    self.assertEqual(observables['observable0'].delay, 3)
    self.assertEqual(observables['observable0'].buffer_size, None)
    self.assertEqual(observables['observable0'].aggregator, None)
    self.assertEqual(observables['observable0'].corruptor, None)
    self.assertFalse(observables['observable0'].enabled)

    self.assertEqual(observables['observable1'].update_interval, 4)
    self.assertEqual(observables['observable1'].delay, 5)
    self.assertEqual(observables['observable1'].buffer_size, None)
    self.assertEqual(observables['observable1'].aggregator, None)
    self.assertEqual(observables['observable1'].corruptor, None)
    self.assertFalse(observables['observable1'].enabled)

  def testObservablePartialOptions(self):
    options = {'observable0': {'update_interval': 2, 'delay': 3}}
    self.entity.observables.set_options(options)
    observables = self.entity.observables.as_dict()
    self.assertEqual(observables['observable0'].update_interval, 2)
    self.assertEqual(observables['observable0'].delay, 3)
    self.assertEqual(observables['observable0'].buffer_size, None)
    self.assertEqual(observables['observable0'].aggregator, None)
    self.assertEqual(observables['observable0'].corruptor, None)
    self.assertFalse(observables['observable0'].enabled)

    self.assertEqual(observables['observable1'].update_interval, 1)
    self.assertEqual(observables['observable1'].delay, None)
    self.assertEqual(observables['observable1'].buffer_size, None)
    self.assertEqual(observables['observable1'].aggregator, None)
    self.assertEqual(observables['observable1'].corruptor, None)
    self.assertFalse(observables['observable1'].enabled)

  def testAttach(self):
    entities = [TestEntity() for _ in range(4)]
    entities[0].attach(entities[1])
    entities[1].attach(entities[2])
    entities[0].attach(entities[3])

    self.assertIsNone(entities[0].parent)
    self.assertIs(entities[1].parent, entities[0])
    self.assertIs(entities[2].parent, entities[1])
    self.assertIs(entities[3].parent, entities[0])

    self.assertIsNone(entities[0].mjcf_model.parent_model)
    self.assertIs(entities[1].mjcf_model.parent_model, entities[0].mjcf_model)
    self.assertIs(entities[2].mjcf_model.parent_model, entities[1].mjcf_model)
    self.assertIs(entities[3].mjcf_model.parent_model, entities[0].mjcf_model)

    self.assertEqual(list(entities[0].iter_entities()), entities)

  def testDetach(self):
    entities = [TestEntity() for _ in range(4)]
    entities[0].attach(entities[1])
    entities[1].attach(entities[2])
    entities[0].attach(entities[3])

    entities[1].detach()
    with six.assertRaisesRegex(self, RuntimeError, 'not attached'):
      entities[1].detach()

    self.assertIsNone(entities[0].parent)
    self.assertIsNone(entities[1].parent)
    self.assertIs(entities[2].parent, entities[1])
    self.assertIs(entities[3].parent, entities[0])

    self.assertIsNone(entities[0].mjcf_model.parent_model)
    self.assertIsNone(entities[1].mjcf_model.parent_model)
    self.assertIs(entities[2].mjcf_model.parent_model, entities[1].mjcf_model)
    self.assertIs(entities[3].mjcf_model.parent_model, entities[0].mjcf_model)

    self.assertEqual(list(entities[0].iter_entities()),
                     [entities[0], entities[3]])

  def testIterEntitiesExcludeSelf(self):
    entities = [TestEntity() for _ in range(4)]
    entities[0].attach(entities[1])
    entities[1].attach(entities[2])
    entities[0].attach(entities[3])
    self.assertEqual(
        list(entities[0].iter_entities(exclude_self=True)), entities[1:])


if __name__ == '__main__':
  absltest.main()
