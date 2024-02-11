# Copyright 2023 The dm_control Authors.
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

"""Fake Blender mathutils module."""
# pylint: disable=invalid-name

import numpy as np


class Color:
  """Fake color class."""

  def __init__(self, coords):
    self._coords = coords

  @property
  def r(self) -> float:
    return self._coords[0]

  @property
  def g(self) -> float:
    return self._coords[1]

  @property
  def b(self) -> float:
    return self._coords[2]

  @property
  def a(self) -> float:
    return self._coords[3]


class Vector:
  """Fake vector class."""

  def __init__(self, coords):
    self._coords = np.asarray(coords)

  @property
  def x(self) -> float:
    return self._coords[0]

  @property
  def y(self) -> float:
    return self._coords[1]

  @property
  def z(self) -> float:
    return self._coords[2]

  @property
  def w(self) -> float:
    return self._coords[3]

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Vector):
      return False
    return np.linalg.norm(self._coords - rhs._coords) < 1e-6

  def __str__(self) -> str:
    return 'Vector({:.2f}, {:.2f}, {:.2f})'.format(self.x, self.y, self.z)

  def __repr__(self) -> str:
    return 'Vector({:.2f}, {:.2f}, {:.2f})'.format(self.x, self.y, self.z)


class Quaternion:
  """Fake quaternion class."""

  def __init__(self, coords):
    self._coords = coords

  @property
  def x(self) -> float:
    return self._coords[1]

  @property
  def y(self) -> float:
    return self._coords[2]

  @property
  def z(self) -> float:
    return self._coords[3]

  @property
  def w(self) -> float:
    return self._coords[0]

  def __matmul__(self, rhs: Vector) -> Vector:
    return Vector((1, 0, 0))


class Matrix:
  """Fake matrix class."""

  def __init__(self, coords):
    self._coords = coords

  @classmethod
  def Diagonal(cls, _):
    return cls((1,))

  @property
  def translation(self) -> Vector:
    return Vector((0.0, 0.0, 0.0))

  def to_quaternion(self) -> Quaternion:
    return Quaternion((1.0, 0.0, 0.0, 0.0))

  def to_scale(self) -> Vector:
    return Vector((0.0, 0.0, 0.0))
