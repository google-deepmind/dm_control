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

"""Various helper functions and classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ctypes
import ctypes.util
import functools
import os
import platform
import sys
import threading
# Internal dependencies.
import numpy as np
import six

from dm_control.utils import io as resources

# Environment variables that can be used to override the default paths to the
# MuJoCo shared library and key file.
ENV_MJLIB_PATH = "MJLIB_PATH"
ENV_MJKEY_PATH = "MJKEY_PATH"

MJLIB_NAME = "mujoco150"


def _get_shared_library_filename():
  """Get platform-dependent prefix and extension of MuJoCo shared library."""
  platform_name = platform.system()
  if platform_name == "Linux":
    prefix = "lib"
    extension = "so"
  elif platform_name == "Darwin":
    prefix = "lib"
    extension = "dylib"
  elif platform_name == "Windows":
    prefix = ""
    extension = "dll"
  else:
    raise OSError("Unsupported platform: {}".format(platform_name))
  return "{}{}.{}".format(prefix, MJLIB_NAME, extension)


DEFAULT_MJLIB_PATH = os.path.join(
    "~/.mujoco/mjpro150/bin", _get_shared_library_filename())
DEFAULT_MJKEY_PATH = "~/.mujoco/mjkey.txt"


DEFAULT_ENCODING = sys.getdefaultencoding()


def to_binary_string(s):
  """Convert text string to binary."""
  if isinstance(s, six.binary_type):
    return s
  return s.encode(DEFAULT_ENCODING)


def to_native_string(s):
  """Convert a text or binary string to the native string format."""
  if six.PY3 and isinstance(s, six.binary_type):
    return s.decode(DEFAULT_ENCODING)
  elif six.PY2 and isinstance(s, six.text_type):
    return s.encode(DEFAULT_ENCODING)
  else:
    return s


def _get_full_path(path):
  expanded_path = os.path.expanduser(os.path.expandvars(path))
  return resources.GetResourceFilename(expanded_path)


def get_mjlib():
  """Loads `libmujoco.so` and returns it as a `ctypes.CDLL` object."""
  try:
    # Use the MJLIB_PATH environment variable if it has been set.
    raw_path = os.environ[ENV_MJLIB_PATH]
  except KeyError:
    paths_to_try = [
        # If libmujoco is in LD_LIBRARY_PATH then ctypes only needs its name.
        os.path.basename(DEFAULT_MJLIB_PATH),
        _get_full_path(DEFAULT_MJLIB_PATH),
    ]
    for library_path in paths_to_try:
      try:
        return ctypes.cdll.LoadLibrary(library_path)
      except OSError as e:
        if "undefined symbol" in str(e) and platform.system() == "Linux":
          # This means that we've found MuJoCo but haven't loaded GLEW.
          ctypes.CDLL(ctypes.util.find_library("GL"), ctypes.RTLD_GLOBAL)
          ctypes.CDLL(ctypes.util.find_library("GLEW"), ctypes.RTLD_GLOBAL)
          return ctypes.cdll.LoadLibrary(library_path)
    raw_path = DEFAULT_MJLIB_PATH
  return ctypes.cdll.LoadLibrary(_get_full_path(raw_path))


def get_mjkey_path():
  """Returns a path to the MuJoCo key file."""
  raw_path = os.environ.get(ENV_MJKEY_PATH, DEFAULT_MJKEY_PATH)
  return _get_full_path(raw_path)


class WrapperBase(object):
  """Base class for wrappers that provide getters/setters for ctypes structs."""

  # This is needed so that the __del__ methods of MjModel and MjData can still
  # succeed in cases where an exception occurs during __init__() before the _ptr
  # attribute has been assigned.
  _ptr = None

  def __init__(self, ptr, model=None):
    """Constructs a wrapper instance from a `ctypes.Structure`.

    Args:
      ptr: `ctypes.POINTER` to the struct to be wrapped.
      model: `MjModel` instance; needed by `MjDataWrapper` in order to get the
        dimensions of dynamically-sized arrays at runtime.
    """
    self._ptr = ptr
    self._model = model

  @property
  def ptr(self):
    """Pointer to the underlying `ctypes.Structure` instance."""
    return self._ptr


class CachedProperty(property):
  """A property that is evaluated only once per object instance."""

  def __init__(self, func, doc=None):
    super(CachedProperty, self).__init__(fget=func, doc=doc)
    self.lock = threading.RLock()

  def __get__(self, obj, cls):
    if obj is None:
      return self
    name = self.fget.__name__
    obj_dict = obj.__dict__
    with self.lock:
      try:
        # Return cached result if it was computed before the lock was acquired
        return obj_dict[name]
      except KeyError:
        # Otherwise call the function, cache the result, and return it
        return obj_dict.setdefault(name, self.fget(obj))


def _as_array(src, shape):
  """Converts a native `src` array to a managed numpy buffer.

  Args:
    src: A ctypes pointer or array.
    shape: A tuple specifying the dimensions of the output array.

  Returns:
    A numpy array.
  """

  # To work around a memory leak in numpy, we have to go through this
  # frombuffer method instead of calling ctypeslib.as_array.  See
  # https://github.com/numpy/numpy/issues/6511
  # return np.ctypeslib.as_array(src, shape)

  # This is part of the public API.  See
  # http://git.net/ml/python.ctypes/2008-02/msg00014.html
  ctype = src._type_  # pylint: disable=protected-access

  size = np.product(shape)
  ptr = ctypes.cast(src, ctypes.POINTER(ctype * size))
  buf = np.frombuffer(ptr.contents, dtype=ctype)
  buf.shape = shape
  return buf


def buf_to_npy(src, shape, np_dtype=None):
  """Returns a numpy array view of the contents of a ctypes pointer or array.

  Args:
    src: A ctypes pointer or array.
    shape: A tuple specifying the dimensions of the output array.
    np_dtype: A string or `np.dtype` object specifying the dtype of the output
      array. If None, the dtype is inferred from the type of `src`.

  Returns:
    A numpy array.
  """
  # This causes a harmless RuntimeWarning about mismatching buffer format
  # strings due to a bug in ctypes: http://stackoverflow.com/q/4964101/1461210
  arr = _as_array(src, shape)
  if np_dtype is not None:
    arr.dtype = np_dtype
  return arr


@functools.wraps(np.ctypeslib.ndpointer)
def ndptr(*args, **kwargs):
  """Wraps `np.ctypeslib.ndpointer` to allow passing None for NULL pointers."""
  base = np.ctypeslib.ndpointer(*args, **kwargs)

  def from_param(_, obj):
    if obj is None:
      return obj
    else:
      return base.from_param(obj)

  return type(base.__name__, (base,), {"from_param": classmethod(from_param)})
