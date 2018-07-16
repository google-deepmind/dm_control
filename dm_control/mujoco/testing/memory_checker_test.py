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


"""Tests for memory_checker."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes

# Internal dependencies.

from absl.testing import absltest

from dm_control.mujoco.testing import memory_checker
from dm_control.mujoco.wrapper.mjbindings import mjlib

from six.moves import range


class MemoryCheckingContextTest(absltest.TestCase):

  n_arrays = 3
  bytes_per_array = 13
  bytes_per_array_padded = 16

  def test_enter_and_exit(self):
    """Check that pointers are set and reset correctly on entry and exit."""
    mju_user_malloc = mjlib.mju_user_malloc
    mju_user_free = mjlib.mju_user_free
    ptr_value = lambda func: ctypes.cast(func, ctypes.c_void_p).value

    old_mju_user_malloc_ptr_value = ptr_value(mju_user_malloc)
    old_mju_user_free_ptr_value = ptr_value(mju_user_free)

    class DummyError(RuntimeError):
      pass

    try:
      with memory_checker.assert_mujoco_memory_freed():
        self.assertNotEqual(
            ptr_value(mju_user_malloc), old_mju_user_malloc_ptr_value)
        self.assertNotEqual(
            ptr_value(mju_user_free), old_mju_user_free_ptr_value)
        raise DummyError("Simulating an exception inside the context manager")
    except DummyError:
      pass

    # Check that the pointers to the custom memory handlers were reset as we
    # exited, even though an exception occurred inside the context.
    self.assertEqual(ptr_value(mju_user_malloc), old_mju_user_malloc_ptr_value)
    self.assertEqual(ptr_value(mju_user_free), old_mju_user_free_ptr_value)

  def test_allocate_and_free_inside(self):
    """Allocating and freeing inside shouldn't raise any exceptions."""
    with memory_checker.assert_mujoco_memory_freed():
      allocated = [
          mjlib.mju_malloc(self.bytes_per_array)
          for _ in range(self.n_arrays)
      ]
      for ptr in allocated:
        mjlib.mju_free(ptr)

  def test_allocate_outside_free_inside(self):
    """Allocating outside and freeing inside shouldn't raise any exceptions."""
    allocated = [
        mjlib.mju_malloc(self.bytes_per_array)
        for _ in range(self.n_arrays)
    ]
    with memory_checker.assert_mujoco_memory_freed():
      for ptr in allocated:
        mjlib.mju_free(ptr)

  def test_allocate_inside_free_outside(self):
    """Allocating inside and freeing outside should raise an AssertionError."""
    with self.assertRaisesRegexp(
        AssertionError,
        "MuJoCo failed to free {} arrays with a total size of {} B"
        .format(self.n_arrays, self.n_arrays * self.bytes_per_array_padded)):
      with memory_checker.assert_mujoco_memory_freed():
        allocated = [
            mjlib.mju_malloc(self.bytes_per_array)
            for _ in range(self.n_arrays)
        ]
    for ptr in allocated:
      mjlib.mju_free(ptr)


if __name__ == "__main__":
  absltest.main()
