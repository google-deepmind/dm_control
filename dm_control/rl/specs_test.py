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
"""Tests for specs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Internal dependencies.

from absl.testing import absltest
from dm_control.rl import specs as array_spec
import numpy as np
import six


class ArraySpecTest(absltest.TestCase):

  def testShapeTypeError(self):
    with self.assertRaises(TypeError):
      array_spec.ArraySpec(32, np.int32)

  def testDtypeTypeError(self):
    with self.assertRaises(TypeError):
      array_spec.ArraySpec((1, 2, 3), "32")

  def testStringDtype(self):
    array_spec.ArraySpec((1, 2, 3), "int32")

  def testNumpyDtype(self):
    array_spec.ArraySpec((1, 2, 3), np.int32)

  def testDtype(self):
    spec = array_spec.ArraySpec((1, 2, 3), np.int32)
    self.assertEqual(np.int32, spec.dtype)

  def testShape(self):
    spec = array_spec.ArraySpec([1, 2, 3], np.int32)
    self.assertEqual((1, 2, 3), spec.shape)

  def testEqual(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int32)
    spec_2 = array_spec.ArraySpec((1, 2, 3), np.int32)
    self.assertEqual(spec_1, spec_2)

  def testNotEqualDifferentShape(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int32)
    spec_2 = array_spec.ArraySpec((1, 3, 3), np.int32)
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualDifferentDtype(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int64)
    spec_2 = array_spec.ArraySpec((1, 2, 3), np.int32)
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualOtherClass(self):
    spec_1 = array_spec.ArraySpec((1, 2, 3), np.int32)
    spec_2 = None
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = ()
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

  def testIsUnhashable(self):
    spec = array_spec.ArraySpec(shape=(1, 2, 3), dtype=np.int32)
    with self.assertRaisesRegexp(TypeError, "unhashable type"):
      hash(spec)

  def testValidateDtype(self):
    spec = array_spec.ArraySpec((1, 2), np.int32)
    spec.validate(np.zeros((1, 2), dtype=np.int32))
    with self.assertRaises(ValueError):
      spec.validate(np.zeros((1, 2), dtype=np.float32))

  def testValidateShape(self):
    spec = array_spec.ArraySpec((1, 2), np.int32)
    spec.validate(np.zeros((1, 2), dtype=np.int32))
    with self.assertRaises(ValueError):
      spec.validate(np.zeros((1, 2, 3), dtype=np.int32))

  def testGenerateValue(self):
    spec = array_spec.ArraySpec((1, 2), np.int32)
    test_value = spec.generate_value()
    spec.validate(test_value)


class BoundedArraySpecTest(absltest.TestCase):

  def testInvalidMinimum(self):
    with six.assertRaisesRegex(self, ValueError, "not compatible"):
      array_spec.BoundedArraySpec((3, 5), np.uint8, (0, 0, 0), (1, 1))

  def testInvalidMaximum(self):
    with six.assertRaisesRegex(self, ValueError, "not compatible"):
      array_spec.BoundedArraySpec((3, 5), np.uint8, 0, (1, 1, 1))

  def testMinMaxAttributes(self):
    spec = array_spec.BoundedArraySpec((1, 2, 3), np.float32, 0, (5, 5, 5))
    self.assertEqual(type(spec.minimum), np.ndarray)
    self.assertEqual(type(spec.maximum), np.ndarray)

  def testNotWriteable(self):
    spec = array_spec.BoundedArraySpec((1, 2, 3), np.float32, 0, (5, 5, 5))
    with six.assertRaisesRegex(self, ValueError, "read-only"):
      spec.minimum[0] = -1
    with six.assertRaisesRegex(self, ValueError, "read-only"):
      spec.maximum[0] = 100

  def testEqualBroadcastingBounds(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=0.0, maximum=1.0)
    spec_2 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertEqual(spec_1, spec_2)

  def testNotEqualDifferentMinimum(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, -0.6], maximum=[1.0, 1.0])
    spec_2 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertNotEqual(spec_1, spec_2)

  def testNotEqualOtherClass(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, -0.6], maximum=[1.0, 1.0])
    spec_2 = array_spec.ArraySpec((1, 2), np.int32)
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = None
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

    spec_2 = ()
    self.assertNotEqual(spec_1, spec_2)
    self.assertNotEqual(spec_2, spec_1)

  def testNotEqualDifferentMaximum(self):
    spec_1 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=0.0, maximum=2.0)
    spec_2 = array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=[0.0, 0.0], maximum=[1.0, 1.0])
    self.assertNotEqual(spec_1, spec_2)

  def testIsUnhashable(self):
    spec = array_spec.BoundedArraySpec(
        shape=(1, 2), dtype=np.int32, minimum=0.0, maximum=2.0)
    with self.assertRaisesRegexp(TypeError, "unhashable type"):
      hash(spec)

  def testRepr(self):
    as_string = repr(array_spec.BoundedArraySpec(
        (1, 2), np.int32, minimum=101.0, maximum=73.0))
    self.assertIn("101", as_string)
    self.assertIn("73", as_string)

  def testValidateBounds(self):
    spec = array_spec.BoundedArraySpec((2, 2), np.int32, minimum=5, maximum=10)
    spec.validate(np.array([[5, 6], [8, 10]], dtype=np.int32))
    with self.assertRaises(ValueError):
      spec.validate(np.array([[5, 6], [8, 11]], dtype=np.int32))
    with self.assertRaises(ValueError):
      spec.validate(np.array([[4, 6], [8, 10]], dtype=np.int32))

  def testGenerateValue(self):
    spec = array_spec.BoundedArraySpec((2, 2), np.int32, minimum=5, maximum=10)
    test_value = spec.generate_value()
    spec.validate(test_value)

  def testScalarBounds(self):
    spec = array_spec.BoundedArraySpec((), np.float, minimum=0.0, maximum=1.0)

    self.assertIsInstance(spec.minimum, np.ndarray)
    self.assertIsInstance(spec.maximum, np.ndarray)

    # Sanity check that numpy compares correctly to a scalar for an empty shape.
    self.assertEqual(0.0, spec.minimum)
    self.assertEqual(1.0, spec.maximum)

    # Check that the spec doesn't fail its own input validation.
    _ = array_spec.BoundedArraySpec(
        spec.shape, spec.dtype, spec.minimum, spec.maximum)


if __name__ == "__main__":
  absltest.main()
