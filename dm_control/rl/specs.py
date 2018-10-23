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

"""Classes that describe the shape and dtype of numpy arrays."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ArraySpec(object):
  """Describes a numpy array or scalar shape and dtype.

  An `ArraySpec` allows an API to describe the arrays that it accepts or
  returns, before that array exists.
  The equivalent version describing a `tf.Tensor` is `TensorSpec`.
  """
  __slots__ = ('_shape', '_dtype', '_name')

  def __init__(self, shape, dtype, name=None):
    """Initializes a new `ArraySpec`.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    self._shape = tuple(shape)
    self._dtype = np.dtype(dtype)
    self._name = name

  @property
  def shape(self):
    """Returns a `tuple` specifying the array shape."""
    return self._shape

  @property
  def dtype(self):
    """Returns a numpy dtype specifying the array dtype."""
    return self._dtype

  @property
  def name(self):
    """Returns the name of the ArraySpec."""
    return self._name

  def __repr__(self):
    return 'ArraySpec(shape={}, dtype={}, name={})'.format(self.shape,
                                                           repr(self.dtype),
                                                           repr(self.name))

  def __eq__(self, other):
    """Checks if the shape and dtype of two specs are equal."""
    if not isinstance(other, ArraySpec):
      return False
    return self.shape == other.shape and self.dtype == other.dtype

  def __ne__(self, other):
    return not self == other

  def _fail_validation(self, message, *args):
    message %= args
    if self.name:
      message += ' for spec %s' % self.name
    raise ValueError(message)

  def validate(self, value):
    """Checks if value conforms to this spec.

    Args:
      value: a numpy array or value convertible to one via `np.asarray`.

    Returns:
      value, converted if necessary to a numpy array.

    Raises:
      ValueError: if value doesn't conform to this spec.
    """
    value = np.asarray(value)
    if value.shape != self.shape:
      self._fail_validation(
          'Expected shape %r but found %r', self.shape, value.shape)
    if value.dtype != self.dtype:
      self._fail_validation(
          'Expected dtype %s but found %s', self.dtype, value.dtype)

  def generate_value(self):
    """Generate a test value which conforms to this spec."""
    return np.zeros(shape=self.shape, dtype=self.dtype)


class BoundedArraySpec(ArraySpec):
  """An `ArraySpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  # Specifying the same minimum and maximum for every element.
  spec = BoundedArraySpec((3, 4), np.float64, minimum=0.0, maximum=1.0)

  # Specifying a different minimum and maximum for each element.
  spec = BoundedArraySpec(
      (2,), np.float64, minimum=[0.1, 0.2], maximum=[0.9, 0.9])

  # Specifying the same minimum and a different maximum for each element.
  spec = BoundedArraySpec(
      (3,), np.float64, minimum=-10.0, maximum=[4.0, 5.0, 3.0])
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by arrays
  with values in the set {0, 1, 2}:
  ```python
  spec = BoundedArraySpec((3, 4), np.int, minimum=0, maximum=2)
  ```
  """

  __slots__ = ('_minimum', '_maximum')

  def __init__(self, shape, dtype, minimum, maximum, name=None):
    """Initializes a new `BoundedArraySpec`.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      minimum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not broadcastable to `shape`.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    super(BoundedArraySpec, self).__init__(shape, dtype, name)

    try:
      np.broadcast_to(minimum, shape=shape)
    except ValueError as numpy_exception:
      raise ValueError('minimum is not compatible with shape. '
                       'Message: {!r}.'.format(numpy_exception))

    try:
      np.broadcast_to(maximum, shape=shape)
    except ValueError as numpy_exception:
      raise ValueError('maximum is not compatible with shape. '
                       'Message: {!r}.'.format(numpy_exception))

    self._minimum = np.array(minimum)
    self._minimum.setflags(write=False)

    self._maximum = np.array(maximum)
    self._maximum.setflags(write=False)

  @property
  def minimum(self):
    """Returns a NumPy array specifying the minimum bounds (inclusive)."""
    return self._minimum

  @property
  def maximum(self):
    """Returns a NumPy array specifying the maximum bounds (inclusive)."""
    return self._maximum

  def __repr__(self):
    template = ('BoundedArraySpec(shape={}, dtype={}, name={}, '
                'minimum={}, maximum={})')
    return template.format(self.shape, repr(self.dtype), repr(self.name),
                           self._minimum, self._maximum)

  def __eq__(self, other):
    if not isinstance(other, BoundedArraySpec):
      return False
    return (super(BoundedArraySpec, self).__eq__(other) and
            (self.minimum == other.minimum).all() and
            (self.maximum == other.maximum).all())

  def validate(self, value):
    value = np.asarray(value)
    super(BoundedArraySpec, self).validate(value)
    if (value < self.minimum).any() or (value > self.maximum).any():
      self._fail_validation(
          'Values were not all within bounds %s <= value <= %s',
          self.minimum, self.maximum)

  def generate_value(self):
    return (np.ones(shape=self.shape, dtype=self.dtype) *
            self.dtype.type(self.minimum))
