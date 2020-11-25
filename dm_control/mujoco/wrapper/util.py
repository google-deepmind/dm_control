# Copyright 2017-2018 The dm_control Authors.
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

import ctypes
import ctypes.util
import functools
import os
import platform
import sys
from dm_control import _render
import numpy as np

from dm_control.utils import io as resources

_PLATFORM = platform.system()
try:
  _PLATFORM_SUFFIX = {
      "Linux": "linux",
      "Darwin": "macos",
      "Windows": "win64"
  }[_PLATFORM]
except KeyError:
  raise OSError("Unsupported platform: {}".format(_PLATFORM))

# Environment variables that can be used to override the default paths to the
# MuJoCo shared library and key file.
ENV_MJLIB_PATH = "MJLIB_PATH"
ENV_MJKEY_PATH = "MJKEY_PATH"


MJLIB_NAME = "mujoco200"


def _get_shared_library_filename():
  """Get platform-dependent prefix and extension of MuJoCo shared library."""
  if _PLATFORM == "Linux":
    prefix = "lib"
    extension = "so"
  elif _PLATFORM == "Darwin":
    prefix = "lib"
    extension = "dylib"
  elif _PLATFORM == "Windows":
    prefix = ""
    extension = "dll"
  else:
    raise OSError("Unsupported platform: {}".format(_PLATFORM))
  return "{}{}.{}".format(prefix, MJLIB_NAME, extension)


DEFAULT_MJLIB_DIR = "~/.mujoco/mujoco200_{}/bin".format(_PLATFORM_SUFFIX)
DEFAULT_MJLIB_PATH = os.path.join(
    DEFAULT_MJLIB_DIR, _get_shared_library_filename())
DEFAULT_MJKEY_PATH = "~/.mujoco/mjkey.txt"


DEFAULT_ENCODING = sys.getdefaultencoding()


def to_binary_string(s):
  """Convert text string to binary."""
  if isinstance(s, bytes):
    return s
  return s.encode(DEFAULT_ENCODING)


def to_native_string(s):
  """Convert a text or binary string to the native string format."""
  if isinstance(s, bytes):
    return s.decode(DEFAULT_ENCODING)
  else:
    return s


def _get_full_path(path):
  expanded_path = os.path.expanduser(os.path.expandvars(path))
  return resources.GetResourceFilename(expanded_path)


def _maybe_load_linux_dynamic_deps(library_dir):
  """Ensures that GL and GLEW symbols are available on Linux."""
  interpreter_symbols = ctypes.cdll.LoadLibrary("")
  if not hasattr(interpreter_symbols, "glewInit"):
    # This means our interpreter is not yet linked against GLEW.
    if _render.BACKEND == "osmesa":
      libglew_path = os.path.join(library_dir, "libglewosmesa.so")
    elif _render.BACKEND == "egl":
      libglew_path = os.path.join(library_dir, "libglewegl.so")
    else:
      libglew_path = ctypes.util.find_library("GLEW")
    ctypes.CDLL(libglew_path, ctypes.RTLD_GLOBAL)  # Also loads GL implicitly.


def get_mjlib():
  """Loads `libmujoco.so` and returns it as a `ctypes.CDLL` object."""
  try:
    # Use the MJLIB_PATH environment variable if it has been set.
    library_path = _get_full_path(os.environ[ENV_MJLIB_PATH])
  except KeyError:
    library_path = ctypes.util.find_library(MJLIB_NAME)
    if not library_path:
      library_path = _get_full_path(DEFAULT_MJLIB_PATH)
  if not os.path.isfile(library_path):
    raise OSError("Cannot find MuJoCo library at {}.".format(library_path))
  if platform.system() == "Linux":
    _maybe_load_linux_dynamic_deps(os.path.dirname(library_path))
  return ctypes.cdll.LoadLibrary(library_path)


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
    self._name = func.__name__

  def __get__(self, obj, cls):
    if obj is None:
      return self
    obj_dict = obj.__dict__
    try:
      return obj_dict[self._name]
    except KeyError:
      return obj_dict.setdefault(self._name, self.fget(obj))


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

  # If we are wrapping an array of ctypes structs, return a `numpy.recarray`.
  # This allows the fields of the struct to be accessed as attributes.
  if issubclass(ctype, ctypes.Structure):
    buf = buf.view(np.recarray)

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


_INVALID_CALLBACK_TYPE = "value must be callable, c_void_p, or None: got {!r}"


def cast_func_to_c_void_p(func, cfunctype):
  """Casts a native function pointer or a Python callable into `c_void_p`.

  Args:
    func: A callable, or a `c_void_p` pointing to a native function, or `None`.
    cfunctype: A `CFUNCTYPE` prototype that is used to wrap `func` if it is
      a Python callable.

  Returns:
    A tuple `(func_ptr, wrapped_pyfunc)`, where `func_ptr` is a `c_void_p`
    object, and `wrapped_pyfunc` is a `CFUNCTYPE` object that wraps `func` if
    it is a Python callable. (If `func` is not a Python callable then
    `wrapped_pyfunc` is `None`.)
  """
  if not (callable(func) or isinstance(func, ctypes.c_void_p) or func is None):
    raise TypeError(_INVALID_CALLBACK_TYPE.format(func))
  try:
    new_func_ptr = ctypes.cast(func, ctypes.c_void_p)
    wrapped_pyfunc = None
  except ctypes.ArgumentError:
    wrapped_pyfunc = cfunctype(func)
    new_func_ptr = ctypes.cast(wrapped_pyfunc, ctypes.c_void_p)
  return new_func_ptr, wrapped_pyfunc
