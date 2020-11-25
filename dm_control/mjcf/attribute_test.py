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

"""Tests for `dm_control.mjcf.attribute`."""

import contextlib
import hashlib
import os

from absl.testing import absltest
from absl.testing import parameterized
from dm_control.mjcf import attribute
from dm_control.mjcf import constants
from dm_control.mjcf import element
from dm_control.mjcf import namescope
from dm_control.mjcf import schema
from dm_control.mujoco.wrapper import mjbindings
from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import types
import numpy as np

mjlib = mjbindings.mjlib

ASSETS_DIR = os.path.join(os.path.dirname(__file__), 'test_assets')
FAKE_SCHEMA_FILENAME = 'attribute_test_schema.xml'

ORIGINAL_SCHEMA_PATH = os.path.join(os.path.dirname(__file__), 'schema.xml')


class AttributeTest(parameterized.TestCase):
  """Test for Attribute classes.

  Our tests here reflect actual usages of the Attribute classes, namely that we
  never directly create attributes but instead access them through Elements.
  """

  def setUp(self):
    super(AttributeTest, self).setUp()
    schema.override_schema(os.path.join(ASSETS_DIR, FAKE_SCHEMA_FILENAME))
    self._alpha = namescope.NameScope('alpha', None)
    self._beta = namescope.NameScope('beta', None)
    self._beta.parent = self._alpha
    self._mujoco = element.RootElement()
    self._mujoco.namescope.parent = self._beta

  def tearDown(self):
    super(AttributeTest, self).tearDown()
    schema.override_schema(ORIGINAL_SCHEMA_PATH)

  def assertXMLStringIsNone(self, mjcf_element, attribute_name):
    for prefix_root in (self._alpha, self._beta, self._mujoco.namescope, None):
      self.assertIsNone(
          mjcf_element.get_attribute_xml_string(attribute_name, prefix_root))

  def assertXMLStringEqual(self, mjcf_element, attribute_name, expected):
    for prefix_root in (self._alpha, self._beta, self._mujoco.namescope, None):
      self.assertEqual(
          mjcf_element.get_attribute_xml_string(attribute_name, prefix_root),
          expected)

  def assertXMLStringIsCorrectlyScoped(
      self, mjcf_element, attribute_name, expected):
    for prefix_root in (self._alpha, self._beta, self._mujoco.namescope, None):
      self.assertEqual(
          mjcf_element.get_attribute_xml_string(attribute_name, prefix_root),
          self._mujoco.namescope.full_prefix(prefix_root) + expected)

  def assertCorrectXMLStringForDefaultsClass(
      self, mjcf_element, attribute_name, expected):
    for prefix_root in (self._alpha, self._beta, self._mujoco.namescope, None):
      self.assertEqual(
          mjcf_element.get_attribute_xml_string(attribute_name, prefix_root),
          (self._mujoco.namescope.full_prefix(prefix_root) + expected) or '/')

  def assertElementIsIdentifiedByName(self, mjcf_element, expected):
    self.assertEqual(mjcf_element.name, expected)
    self.assertEqual(self._mujoco.find(mjcf_element.spec.namespace, expected),
                     mjcf_element)

  @contextlib.contextmanager
  def assertAttributeIsNoneWhenDone(self, mjcf_element, attribute_name):
    yield
    self.assertIsNone(getattr(mjcf_element, attribute_name))
    self.assertXMLStringIsNone(mjcf_element, attribute_name)

  def assertCorrectClearBehavior(self, mjcf_element, attribute_name, required):
    if required:
      return self.assertRaisesRegex(AttributeError, 'is required')
    else:
      return self.assertAttributeIsNoneWhenDone(mjcf_element, attribute_name)

  def assertCorrectClearBehaviorByAllMethods(
      self, mjcf_element, attribute_name, required):
    original_value = getattr(mjcf_element, attribute_name)
    def reset_value():
      setattr(mjcf_element, attribute_name, original_value)
      if original_value is not None:
        self.assertIsNotNone(getattr(mjcf_element, attribute_name))

    # clear by using del
    with self.assertCorrectClearBehavior(
        mjcf_element, attribute_name, required):
      delattr(mjcf_element, attribute_name)

    # clear by assigning None
    reset_value()
    with self.assertCorrectClearBehavior(
        mjcf_element, attribute_name, required):
      setattr(mjcf_element, attribute_name, None)

    if isinstance(original_value, str):
      # clear by assigning empty string
      reset_value()
      with self.assertCorrectClearBehavior(
          mjcf_element, attribute_name, required):
        setattr(mjcf_element, attribute_name, '')

  def assertCanBeCleared(self, mjcf_element, attribute_name):
    self.assertCorrectClearBehaviorByAllMethods(
        mjcf_element, attribute_name, required=False)

  def assertCanNotBeCleared(self, mjcf_element, attribute_name):
    self.assertCorrectClearBehaviorByAllMethods(
        mjcf_element, attribute_name, required=True)

  def testFloatScalar(self):
    mujoco = self._mujoco
    mujoco.optional.float = 5
    self.assertEqual(mujoco.optional.float, 5)
    self.assertEqual(type(mujoco.optional.float), float)
    with self.assertRaisesRegex(ValueError, 'Expect a float value'):
      mujoco.optional.float = 'five'
    # failed assignment should not change the value
    self.assertEqual(mujoco.optional.float, 5)
    self.assertXMLStringEqual(mujoco.optional, 'float', '5.0')
    self.assertCanBeCleared(mujoco.optional, 'float')

  def testIntScalar(self):
    mujoco = self._mujoco
    mujoco.optional.int = 12345
    self.assertEqual(mujoco.optional.int, 12345)
    self.assertEqual(type(mujoco.optional.int), int)
    with self.assertRaisesRegex(ValueError, 'Expect an integer value'):
      mujoco.optional.int = 10.5
    # failed assignment should not change the value
    self.assertEqual(mujoco.optional.int, 12345)
    self.assertXMLStringEqual(mujoco.optional, 'int', '12345')
    self.assertCanBeCleared(mujoco.optional, 'int')

  def testStringScalar(self):
    mujoco = self._mujoco
    mujoco.optional.string = 'foobar'
    self.assertEqual(mujoco.optional.string, 'foobar')
    self.assertXMLStringEqual(mujoco.optional, 'string', 'foobar')
    with self.assertRaisesRegex(ValueError, 'Expect a string value'):
      mujoco.optional.string = mujoco.optional
    self.assertCanBeCleared(mujoco.optional, 'string')

  def testFloatArray(self):
    mujoco = self._mujoco
    mujoco.optional.float_array = [3, 2, 1]
    np.testing.assert_array_equal(mujoco.optional.float_array, [3, 2, 1])
    self.assertEqual(mujoco.optional.float_array.dtype, np.float)
    with self.assertRaisesRegex(ValueError, 'no more than 3 entries'):
      mujoco.optional.float_array = [0, 0, 0, -10]
    with self.assertRaisesRegex(ValueError, 'one-dimensional array'):
      mujoco.optional.float_array = np.array([[1, 2], [3, 4]])
    # failed assignments should not change the value
    np.testing.assert_array_equal(mujoco.optional.float_array, [3, 2, 1])
    # XML string should not be affected by global print options
    np.set_printoptions(precision=3, suppress=True)
    mujoco.optional.float_array = [np.pi, 2, 1e-16]
    self.assertXMLStringEqual(mujoco.optional, 'float_array',
                              '3.1415926535897931 2 9.9999999999999998e-17')
    self.assertCanBeCleared(mujoco.optional, 'float_array')

  def testFormatVeryLargeArray(self):
    mujoco = self._mujoco
    array = np.arange(2000, dtype=np.double)
    mujoco.optional.huge_float_array = array
    xml_string = mujoco.optional.get_attribute_xml_string('huge_float_array')
    self.assertNotIn('...', xml_string)
    # Check that array <--> string conversion is a round trip.
    mujoco.optional.huge_float_array = None
    self.assertIsNone(mujoco.optional.huge_float_array)
    mujoco.optional.huge_float_array = xml_string
    np.testing.assert_array_equal(mujoco.optional.huge_float_array, array)

  def testIntArray(self):
    mujoco = self._mujoco
    mujoco.optional.int_array = [2, 2]
    np.testing.assert_array_equal(mujoco.optional.int_array, [2, 2])
    self.assertEqual(mujoco.optional.int_array.dtype, np.int)
    with self.assertRaisesRegex(ValueError, 'no more than 2 entries'):
      mujoco.optional.int_array = [0, 0, 10]
    # failed assignment should not change the value
    np.testing.assert_array_equal(mujoco.optional.int_array, [2, 2])
    self.assertXMLStringEqual(mujoco.optional, 'int_array', '2 2')
    self.assertCanBeCleared(mujoco.optional, 'int_array')

  def testKeyword(self):
    mujoco = self._mujoco

    valid_values = ['Alpha', 'Beta', 'Gamma']
    for value in valid_values:
      mujoco.optional.keyword = value.lower()
      self.assertEqual(mujoco.optional.keyword, value)
      self.assertXMLStringEqual(mujoco.optional, 'keyword', value)

      mujoco.optional.keyword = value.upper()
      self.assertEqual(mujoco.optional.keyword, value)
      self.assertXMLStringEqual(mujoco.optional, 'keyword', value)

    with self.assertRaisesRegex(ValueError, str(valid_values)):
      mujoco.optional.keyword = 'delta'
    # failed assignment should not change the value
    self.assertXMLStringEqual(mujoco.optional, 'keyword', valid_values[-1])
    self.assertCanBeCleared(mujoco.optional, 'keyword')

  def testIdentifier(self):
    mujoco = self._mujoco

    entity = mujoco.worldentity.add('entity')
    subentity_1 = entity.add('subentity', name='foo')
    subentity_2 = entity.add('subentity_alias', name='bar')

    self.assertIsNone(entity.name)
    self.assertElementIsIdentifiedByName(subentity_1, 'foo')
    self.assertElementIsIdentifiedByName(subentity_2, 'bar')
    self.assertXMLStringIsCorrectlyScoped(subentity_1, 'name', 'foo')
    self.assertXMLStringIsCorrectlyScoped(subentity_2, 'name', 'bar')

    with self.assertRaisesRegex(ValueError, 'Expect a string value'):
      subentity_2.name = subentity_1
    with self.assertRaisesRegex(ValueError, 'reserved for scoping'):
      subentity_2.name = 'foo/bar'
    with self.assertRaisesRegex(ValueError, 'Duplicated identifier'):
      subentity_2.name = 'foo'
    # failed assignment should not change the value
    self.assertElementIsIdentifiedByName(subentity_2, 'bar')

    with self.assertRaisesRegex(ValueError, 'cannot be named \'world\''):
      mujoco.worldentity.add('body', name='world')

    subentity_1.name = 'baz'
    self.assertElementIsIdentifiedByName(subentity_1, 'baz')
    self.assertIsNone(mujoco.find('subentity', 'foo'))

    # 'foo' is now unused, so we should be allowed to use it
    subentity_2.name = 'foo'
    self.assertElementIsIdentifiedByName(subentity_2, 'foo')

    # duplicate name should be allowed when in different namespaces
    entity.name = 'foo'
    self.assertElementIsIdentifiedByName(entity, 'foo')
    self.assertCanBeCleared(entity, 'name')

  def testStringReference(self):
    mujoco = self._mujoco
    mujoco.optional.reference = 'foo'
    self.assertEqual(mujoco.optional.reference, 'foo')
    self.assertXMLStringIsCorrectlyScoped(mujoco.optional, 'reference', 'foo')
    self.assertCanBeCleared(mujoco.optional, 'reference')

  def testElementReferenceWithFixedNamespace(self):
    mujoco = self._mujoco
    # `mujoco.optional.fixed_type_ref` must be an element in the 'optional'
    # namespace. 'identified' elements are part of the 'optional' namespace.
    bar = mujoco.add('identified', identifier='bar')
    mujoco.optional.fixed_type_ref = bar
    self.assertXMLStringIsCorrectlyScoped(
        mujoco.optional, 'fixed_type_ref', 'bar')
    # Removing the referenced entity should cause the `fixed_type_ref` to be set
    # to None.
    bar.remove()
    self.assertIsNone(mujoco.optional.fixed_type_ref)

  def testElementReferenceWithVariableNamespace(self):
    mujoco = self._mujoco

    # `mujoco.optional.reference` can be an element in either the 'entity' or
    # or 'optional' namespaces. First we assign an 'identified' element to the
    # reference attribute. These are part of the 'optional' namespace.
    bar = mujoco.add('identified', identifier='bar')
    mujoco.optional.reftype = 'optional'
    mujoco.optional.reference = bar
    self.assertXMLStringIsCorrectlyScoped(mujoco.optional, 'reference', 'bar')

    # Assigning to `mujoco.optional.reference` should also change the value of
    # `mujoco.optional.reftype` to match the namespace of the element that was
    # assigned to `mujoco.optional.reference`
    self.assertXMLStringEqual(mujoco.optional, 'reftype', 'optional')

    # Now assign an 'entity' element to the reference attribute. These are part
    # of the 'entity' namespace.
    baz = mujoco.worldentity.add('entity', name='baz')
    mujoco.optional.reftype = 'entity'
    mujoco.optional.reference = baz
    self.assertXMLStringIsCorrectlyScoped(mujoco.optional, 'reference', 'baz')
    # The `reftype` should change to 'entity' accordingly.
    self.assertXMLStringEqual(mujoco.optional, 'reftype', 'entity')

    # Removing the referenced entity should cause the `reference` and `reftype`
    # to be set to None.
    baz.remove()
    self.assertIsNone(mujoco.optional.reference)
    self.assertIsNone(mujoco.optional.reftype)

  def testInvalidReference(self):
    mujoco = self._mujoco
    bar = mujoco.worldentity.add('entity', name='bar')
    baz = bar.add('subentity', name='baz')
    mujoco.optional.reftype = 'entity'
    with self.assertRaisesWithLiteralMatch(
        ValueError, attribute._INVALID_REFERENCE_TYPE.format(
            valid_type='entity', actual_type='subentity')):
      mujoco.optional.reference = baz
    with self.assertRaisesWithLiteralMatch(
        ValueError, attribute._INVALID_REFERENCE_TYPE.format(
            valid_type='optional', actual_type='subentity')):
      mujoco.optional.fixed_type_ref = baz

  def testDefaults(self):
    mujoco = self._mujoco

    # Unnamed global defaults class should become a properly named and scoped
    # class with a trailing slash
    self.assertIsNone(mujoco.default.dclass)
    self.assertCorrectXMLStringForDefaultsClass(mujoco.default, 'class', '')

    # An element without an explicit dclass should be assigned to the properly
    # scoped global defaults class
    entity = mujoco.worldentity.add('entity')
    subentity = entity.add('subentity')
    self.assertIsNone(subentity.dclass)
    self.assertCorrectXMLStringForDefaultsClass(subentity, 'class', '')

    # Named global defaults class should gain scoping prefix
    mujoco.default.dclass = 'main'
    self.assertEqual(mujoco.default.dclass, 'main')
    self.assertCorrectXMLStringForDefaultsClass(mujoco.default, 'class', 'main')
    self.assertCorrectXMLStringForDefaultsClass(subentity, 'class', 'main')

    # Named subordinate defaults class should gain scoping prefix
    sub_default = mujoco.default.add('default', dclass='sub')
    self.assertEqual(sub_default.dclass, 'sub')
    self.assertCorrectXMLStringForDefaultsClass(sub_default, 'class', 'sub')

    # An element without an explicit dclass but belongs to a childclassed
    # parent should be left alone
    entity.childclass = 'sub'
    self.assertEqual(entity.childclass, 'sub')
    self.assertCorrectXMLStringForDefaultsClass(entity, 'childclass', 'sub')
    self.assertXMLStringIsNone(subentity, 'class')

    # An element WITH an explicit dclass should be left alone have it properly
    # scoped regardless of whether it belongs to a childclassed parent or not.
    subentity.dclass = 'main'
    self.assertCorrectXMLStringForDefaultsClass(subentity, 'class', 'main')

  @parameterized.named_parameters(
      ('NoBasepath', '', os.path.join(ASSETS_DIR, FAKE_SCHEMA_FILENAME)),
      ('WithBasepath', ASSETS_DIR, FAKE_SCHEMA_FILENAME))
  def testFileFromPath(self, basepath, value):
    mujoco = self._mujoco
    full_path = os.path.join(basepath, value)
    with open(full_path, 'rb') as f:
      contents = f.read()
    _, basename = os.path.split(value)
    prefix, extension = os.path.splitext(basename)
    expected_xml = prefix + '-' + hashlib.sha1(contents).hexdigest() + extension
    mujoco.files.text_path = basepath
    text_file = mujoco.files.add('text', file=value)
    expected_value = attribute.Asset(
        contents=contents, extension=extension, prefix=prefix)
    self.assertEqual(text_file.file, expected_value)
    self.assertXMLStringEqual(text_file, 'file', expected_xml)
    self.assertCanBeCleared(text_file, 'file')
    self.assertCanBeCleared(mujoco.files, 'text_path')

  def testFileNameTrimming(self):
    original_filename = (
        'THIS_IS_AN_EXTREMELY_LONG_FILENAME_THAT_WOULD_CAUSE_MUJOCO_TO_COMPLAIN'
        '_THAT_ITS_INTERNAL_LENGTH_LIMIT_IS_EXCEEDED_IF_NOT_TRIMMED_DOWN')
    extension = '.some_extension'
    asset = attribute.Asset(
        contents='', extension=extension, prefix=original_filename)
    vfs_filename = asset.get_vfs_filename()
    self.assertLen(vfs_filename, constants.MAX_VFS_FILENAME_LENGTH)

    vfs = types.MJVFS()
    mjlib.mj_defaultVFS(vfs)
    success_code = 0
    retval = mjlib.mj_makeEmptyFileVFS(
        vfs, util.to_binary_string(vfs_filename), 1)
    self.assertEqual(retval, success_code)
    mjlib.mj_deleteVFS(vfs)

  def testFileFromPlaceholder(self):
    mujoco = self._mujoco
    contents = b'Fake contents'
    extension = '.whatever'
    expected_xml = hashlib.sha1(contents).hexdigest() + extension
    placeholder = attribute.Asset(contents=contents, extension=extension)
    text_file = mujoco.files.add('text', file=placeholder)
    self.assertEqual(text_file.file, placeholder)
    self.assertXMLStringEqual(text_file, 'file', expected_xml)
    self.assertCanBeCleared(text_file, 'file')

  def testFileFromAssetsDict(self):
    prefix = 'fake_filename'
    extension = '.whatever'
    path = 'invalid/path/' + prefix + extension
    contents = 'Fake contents'
    assets = {path: contents}
    mujoco = element.RootElement(assets=assets)
    text_file = mujoco.files.add('text', file=path)
    expected_value = attribute.Asset(
        contents=contents, extension=extension, prefix=prefix)
    self.assertEqual(text_file.file, expected_value)

  def testFileExceptions(self):
    mujoco = self._mujoco
    text_file = mujoco.files.add('text')
    with self.assertRaisesRegex(ValueError,
                                'Expect either a string or `Asset` value'):
      text_file.file = mujoco.optional

  def testBasePathExceptions(self):
    mujoco = self._mujoco
    with self.assertRaisesRegex(ValueError, 'Expect a string value'):
      mujoco.files.text_path = mujoco.optional

  def testRequiredAttributes(self):
    mujoco = self._mujoco
    attributes = (
        ('float', 1.0), ('int', 2), ('string', 'foobar'),
        ('float_array', [1.5, 2.5, 3.5]), ('int_array', [4, 5]),
        ('keyword', 'alpha'), ('identifier', 'thing'),
        ('reference', 'other_thing'), ('basepath', ASSETS_DIR),
        ('file', FAKE_SCHEMA_FILENAME)
    )

    # Removing any one of the required attributes should cause initialization
    # of a new element to fail
    for name, _ in attributes:
      attributes_dict = {key: value for key, value in attributes if key != name}
      with self.assertRaisesRegex(AttributeError, name + '.+ is required'):
        mujoco.add('required', **attributes_dict)

    attributes_dict = {key: value for key, value in attributes}
    mujoco.add('required', **attributes_dict)
    # Should not be allowed to clear each required attribute after the fact
    for name, _ in attributes:
      self.assertCanNotBeCleared(mujoco.required, name)


if __name__ == '__main__':
  absltest.main()
