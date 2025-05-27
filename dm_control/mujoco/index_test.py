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

"""Tests for index."""

import collections

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mujoco import index
from dm_control.mujoco import wrapper
from dm_control.mujoco.testing import assets
from dm_control.mujoco.wrapper.mjbindings import sizes
import mujoco
import numpy as np

MODEL = assets.get_contents('cartpole.xml')
MODEL_NO_NAMES = assets.get_contents('cartpole_no_names.xml')
MODEL_3RD_ORDER_ACTUATORS = assets.get_contents(
    'model_with_third_order_actuators.xml')

FIELD_REPR = {
    'act': ('FieldIndexer(act):\n'
            '(empty)'),
    'qM': ('FieldIndexer(qM):\n'
           '0  [ 0       ]\n'
           '1  [ 1       ]\n'
           '2  [ 2       ]'),
    'sensordata': ('FieldIndexer(sensordata):\n'
                   '0 accelerometer [ 0       ]\n'
                   '1 accelerometer [ 1       ]\n'
                   '2 accelerometer [ 2       ]\n'
                   '3     collision [ 3       ]'),
    'xpos': ('FieldIndexer(xpos):\n'
             '           x         y         z         \n'
             '0  world [ 0         1         2       ]\n'
             '1   cart [ 3         4         5       ]\n'
             '2   pole [ 6         7         8       ]\n'
             '3 mocap1 [ 9         10        11      ]\n'
             '4 mocap2 [ 12        13        14      ]'),
}

# A test below accesses all attributes of np.ndarray, but accessing some of them
# triggers an exception under numpy>=2.0.
_DEPRECATED_NDARRAY_ATTRIBUTES = {
    'itemset', 'newbyteorder', 'ptp'
}


class MujocoIndexTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._model = wrapper.MjModel.from_xml_string(MODEL)
    self._data = wrapper.MjData(self._model)
    mujoco.mj_forward(self._model.ptr, self._data.ptr)

    self._size_to_axis_indexer = index.make_axis_indexers(self._model)

    self._model_indexers = index.struct_indexer(self._model, 'mjmodel',
                                                self._size_to_axis_indexer)

    self._data_indexers = index.struct_indexer(self._data, 'mjdata',
                                               self._size_to_axis_indexer)

  def assertIndexExpressionEqual(self, expected, actual):
    try:
      if isinstance(expected, tuple):
        self.assertLen(actual, len(expected))
        for expected_item, actual_item in zip(expected, actual):
          self.assertIndexExpressionEqual(expected_item, actual_item)
      elif isinstance(expected, (list, np.ndarray)):
        np.testing.assert_array_equal(expected, actual)
      else:
        self.assertEqual(expected, actual)
    except AssertionError:
      self.fail('Indexing expressions are not equal.\n'
                'expected: {!r}\nactual: {!r}'.format(expected, actual))

  @parameterized.parameters(
      # (field name, named index key, expected integer index key)
      ('actuator_gear', 'slide', 0),
      ('geom_rgba', ('mocap_sphere', 'g'), (6, 1)),
      ('dof_armature', 'slider', slice(0, 1, None)),
      ('dof_armature', ['slider', 'hinge'], [0, 1]),
      ('numeric_data', 'three_numbers', slice(1, 4, None)),
      ('numeric_data', ['three_numbers', 'control_timestep'], [1, 2, 3, 0]))
  def testModelNamedIndexing(self, field_name, key, numeric_key):

    indexer = getattr(self._model_indexers, field_name)
    field = getattr(self._model, field_name)

    converted_key = indexer._convert_key(key)

    # Explicit check that the converted key matches the numeric key.
    converted_key = indexer._convert_key(key)
    self.assertIndexExpressionEqual(numeric_key, converted_key)

    # This writes unique values to the underlying buffer to prevent false
    # negatives.
    field.flat[:] = np.arange(field.size)

    # Check that the result of named indexing matches the result of numeric
    # indexing.
    np.testing.assert_array_equal(field[numeric_key], indexer[key])

  @parameterized.parameters(
      # (field name, named index key, expected integer index key)
      ('xpos', 'pole', 2),
      ('xpos', ['pole', 'cart'], [2, 1]),
      ('sensordata', 'accelerometer', slice(0, 3, None)),
      ('sensordata', 'collision', slice(3, 4, None)),
      ('sensordata', ['accelerometer', 'collision'], [0, 1, 2, 3]),
      # Slices.
      ('xpos', (slice(None), 0), (slice(None), 0)),
      # Custom fixed-size columns.
      ('xpos', ('pole', 'y'), (2, 1)),
      ('xmat', ('cart', ['yy', 'zz']), (1, [4, 8])),
      # Custom indexers for mocap bodies.
      ('mocap_quat', 'mocap1', 0),
      ('mocap_pos', (['mocap2', 'mocap1'], 'z'), ([1, 0], 2)),
      # Two-dimensional named indexing.
      ('xpos', (['pole', 'cart'], ['x', 'z']), ([2, 1], [0, 2])),
      ('xpos', ([['pole'], ['cart']], ['x', 'z']), ([[2], [1]], [0, 2])))
  def testDataNamedIndexing(self, field_name, key, numeric_key):

    indexer = getattr(self._data_indexers, field_name)
    field = getattr(self._data, field_name)

    # Explicit check that the converted key matches the numeric key.
    converted_key = indexer._convert_key(key)
    self.assertIndexExpressionEqual(numeric_key, converted_key)

    # This writes unique values to the underlying buffer to prevent false
    # negatives.
    field.flat[:] = np.arange(field.size)

    # Check that the result of named indexing matches the result of numeric
    # indexing.
    np.testing.assert_array_equal(field[numeric_key], indexer[key])

  @parameterized.parameters(
      # (field name, named index key, expected integer index key)
      ('act', 'cylinder', 0),
      ('act_dot', 'general', 1),
      ('act', ['general', 'cylinder', 'general'], [1, 0, 1]))
  def testIndexThirdOrderActuators(self, field_name, key, numeric_key):
    model = wrapper.MjModel.from_xml_string(MODEL_3RD_ORDER_ACTUATORS)
    data = wrapper.MjData(model)
    size_to_axis_indexer = index.make_axis_indexers(model)
    data_indexers = index.struct_indexer(data, 'mjdata', size_to_axis_indexer)

    indexer = getattr(data_indexers, field_name)
    field = getattr(data, field_name)

    # Explicit check that the converted key matches the numeric key.
    converted_key = indexer._convert_key(key)
    self.assertIndexExpressionEqual(numeric_key, converted_key)

    # This writes unique values to the underlying buffer to prevent false
    # negatives.
    field.flat[:] = np.arange(field.size)

    # Check that the result of named indexing matches the result of numeric
    # indexing.
    np.testing.assert_array_equal(field[numeric_key], indexer[key])

  @parameterized.parameters(
      # (field name, named index key)
      ('xpos', 'pole'),
      ('xpos', ['pole', 'cart']),
      ('xpos', (['pole', 'cart'], 'y')),
      ('xpos', (['pole', 'cart'], ['x', 'z'])),
      ('qpos', 'slider'),
      ('qvel', ['slider', 'hinge']),)
  def testDataAssignment(self, field_name, key):

    indexer = getattr(self._data_indexers, field_name)
    field = getattr(self._data, field_name)

    # The result of the indexing expression is either an array or a scalar.
    index_result = indexer[key]
    try:
      # Write a sequence of unique values to prevent false negatives.
      new_values = np.arange(index_result.size).reshape(index_result.shape)
    except AttributeError:
      new_values = 99
    indexer[key] = new_values

    # Check that the new value(s) can be read back from the underlying buffer.
    converted_key = indexer._convert_key(key)
    np.testing.assert_array_equal(new_values, field[converted_key])

  @parameterized.parameters(
      # (field name, first index key, second index key)
      ('sensordata', 'accelerometer', 0),
      ('sensordata', 'accelerometer', [0, 2]),
      ('sensordata', 'accelerometer', slice(None)),)
  def testChainedAssignment(self, field_name, first_key, second_key):

    indexer = getattr(self._data_indexers, field_name)
    field = getattr(self._data, field_name)

    # The result of the indexing expression is either an array or a scalar.
    index_result = indexer[first_key][second_key]
    try:
      # Write a sequence of unique values to prevent false negatives.
      new_values = np.arange(index_result.size).reshape(index_result.shape)
    except AttributeError:
      new_values = 99
    indexer[first_key][second_key] = new_values

    # Check that the new value(s) can be read back from the underlying buffer.
    converted_key = indexer._convert_key(first_key)
    np.testing.assert_array_equal(new_values, field[converted_key][second_key])

  def testNamedColumnFieldNames(self):

    all_fields = set()
    for struct in sizes.array_sizes.values():
      all_fields.update(struct.keys())

    named_col_fields = set()
    for field_set in index._COLUMN_ID_TO_FIELDS.values():
      named_col_fields.update(field_set)

    # Check that all of the "named column" fields specified in index are
    # also found in mjbindings.sizes.
    self.assertContainsSubset(named_col_fields, all_fields)

  @parameterized.parameters('xpos', 'xmat')  # field name
  def testTooManyIndices(self, field_name):
    indexer = getattr(self._data_indexers, field_name)
    with self.assertRaisesRegex(IndexError, 'Index tuple'):
      _ = indexer[:, :, :, 'too', 'many', 'elements']

  @parameterized.parameters(
      # bad item, exception regexp
      (Ellipsis, 'Ellipsis'),
      (None, 'None'),
      (np.newaxis, 'None'),
      (b'', 'Empty string'),
      (u'', 'Empty string'))
  def testBadIndexItems(self, bad_index_item, exception_regexp):
    indexer = getattr(self._data_indexers, 'xpos')
    expressions = [
        bad_index_item,
        (0, bad_index_item),
        [bad_index_item],
        [[bad_index_item]],
        (0, [bad_index_item]),
        (0, [[bad_index_item]]),
        np.array([bad_index_item]),
        (0, np.array([bad_index_item])),
        (0, np.array([[bad_index_item]]))
    ]
    for expression in expressions:
      with self.assertRaisesRegex(IndexError, exception_regexp):
        _ = indexer[expression]

  @parameterized.parameters('act', 'qM', 'sensordata', 'xpos')  # field name
  def testFieldIndexerRepr(self, field_name):

    indexer = getattr(self._data_indexers, field_name)
    field = getattr(self._data, field_name)

    # Write a sequence of unique values to prevent false negatives.
    field.flat[:] = np.arange(field.size)

    # Check that the string representation is as expected.
    self.assertEqual(FIELD_REPR[field_name], repr(indexer))

  @parameterized.parameters(MODEL, MODEL_NO_NAMES)
  def testBuildIndexersForEdgeCases(self, xml_string):
    model = wrapper.MjModel.from_xml_string(xml_string)
    data = wrapper.MjData(model)

    size_to_axis_indexer = index.make_axis_indexers(model)

    index.struct_indexer(model, 'mjmodel', size_to_axis_indexer)
    index.struct_indexer(data, 'mjdata', size_to_axis_indexer)

  # pylint: disable=undefined-variable
  @parameterized.named_parameters([
      (name, name) for name in dir(np.ndarray)
      if not name.startswith('_')  # Exclude 'private' attributes
      and name not in ('ctypes', 'flat')  # Can't compare via identity/equality
      and name not in _DEPRECATED_NDARRAY_ATTRIBUTES
  ])
  # pylint: enable=undefined-variable
  def testFieldIndexerDelegatesNDArrayAttributes(self, name):
    field = self._data.xpos
    field_indexer = self._data_indexers.xpos
    actual = getattr(field_indexer, name)
    expected = getattr(field, name)
    if isinstance(expected, np.ndarray):
      np.testing.assert_array_equal(actual, expected)
    else:
      self.assertEqual(actual, expected)

    # FieldIndexer attributes should be read-only
    with self.assertRaisesRegex(AttributeError, name):
      setattr(field_indexer, name, expected)

  def testFieldIndexerDir(self):
    expected_subset = dir(self._data.xpos)
    actual_set = dir(self._data_indexers.xpos)
    self.assertContainsSubset(expected_subset, actual_set)


def _iter_indexers(model, data):
  mujoco.mj_forward(model.ptr, data.ptr)
  size_to_axis_indexer = index.make_axis_indexers(model)
  all_fields = collections.OrderedDict()
  for struct, struct_name in ((model, 'mjmodel'), (data, 'mjdata')):
    indexer = index.struct_indexer(struct, struct_name, size_to_axis_indexer)
    for field_name, field_indexer in indexer._asdict().items():
      if field_name not in all_fields:
        all_fields[field_name] = field_indexer
  for field_name, field_indexer in all_fields.items():
    yield field_name, field_indexer


class AllFieldsTest(parameterized.TestCase):
  """Generic tests covering each FieldIndexer in model and data."""

  # NB: the class must hold references to the model and data instances or they
  # may be garbage-collected before any indexing is attempted.
  model = wrapper.MjModel.from_xml_string(MODEL)
  data = wrapper.MjData(model)

  # Iterates over ('field_name', FieldIndexer) pairs
  @parameterized.named_parameters(_iter_indexers(model, data))
  def testReadWrite_(self, field):
    # Read the contents of the FieldIndexer as a numpy array.
    old_contents = field[:]
    # Write unique values to the FieldIndexer and read them back again.
    # Don't write to non-float fields since these might contain pointers.
    if np.issubdtype(old_contents.dtype, np.floating):
      new_contents = np.arange(old_contents.size, dtype=old_contents.dtype)
      new_contents.shape = old_contents.shape
      field[:] = new_contents
      np.testing.assert_array_equal(new_contents, field[:])


if __name__ == '__main__':
  absltest.main()
