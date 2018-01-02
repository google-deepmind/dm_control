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

"""A context manager for checking that MuJoCo memory is freed."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import ctypes

# Internal dependencies.

from dm_control.mujoco.wrapper.mjbindings import mjlib
import six


# Used for overriding MuJoCo's memory handlers
_LIBC = ctypes.cdll.LoadLibrary("libc.so.6")
_LIBC.aligned_alloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
_LIBC.aligned_alloc.restype = ctypes.c_void_p
_LIBC.free.argtypes = [ctypes.c_void_p]
_LIBC.free.restype = None

# MuJoCo normally pads and aligns memory to multiples of 8 bytes
_BYTE_ALIGNMENT = 8

# Expose pointers to custom memory handlers.
mjlib.mju_user_malloc = ctypes.c_void_p.in_dll(mjlib, "mju_user_malloc")
mjlib.mju_user_free = ctypes.c_void_p.in_dll(mjlib, "mju_user_free")


@contextlib.contextmanager
def assert_mujoco_memory_freed():
  """Context manager for debugging memory leaks in MuJoCo.

  Yields:
    None

  Raises:
    AssertionError: If MuJoCo heap-allocated any memory inside the context
      manager without freeing it.
  """

  # NB: The custom memory handlers need to use libc's `aligned_alloc` and `free`
  #     rather than `mju_malloc` and `mju_free`, since these will delegate to
  #     `mju_user_malloc` and `mju_user_free` if they are not NULL.

  remaining_pointers = {}

  @ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_size_t)
  def debug_malloc(size):
    if size % _BYTE_ALIGNMENT:
      size += _BYTE_ALIGNMENT - (size % _BYTE_ALIGNMENT)
    address = _LIBC.aligned_alloc(_BYTE_ALIGNMENT, size)
    remaining_pointers[address] = size
    return address

  @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
  def debug_free(address):
    _LIBC.free(address)
    # Allow freeing of arrays that were allocated outside of the context.
    remaining_pointers.pop(address, None)

  # Keep the old pointer addresses in case there were already custom memory
  # handling callbacks defined.
  old_user_malloc_ptr_value = mjlib.mju_user_malloc.value
  old_user_free_ptr_value = mjlib.mju_user_free.value

  # Set the new callbacks.
  mjlib.mju_user_malloc.value = ctypes.cast(debug_malloc, ctypes.c_void_p).value
  mjlib.mju_user_free.value = ctypes.cast(debug_free, ctypes.c_void_p).value

  try:
    yield
  finally:
    # Make sure we reset the memory handlers, even if an exception is raised.
    mjlib.mju_user_malloc.value = old_user_malloc_ptr_value
    mjlib.mju_user_free.value = old_user_free_ptr_value

  if remaining_pointers:
    n_not_freed = len(remaining_pointers)
    n_bytes_leaked = sum(six.itervalues(remaining_pointers))
    details_str = "\n".join(
        "address: {} size: {} B".format(address, size)
        for address, size in six.iteritems(remaining_pointers))
    raise AssertionError(
        "MuJoCo failed to free {} arrays with a total size of {} B:\n{}"
        .format(n_not_freed, n_bytes_leaked, details_str))
