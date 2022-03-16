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

"""Main user-facing classes and utility functions for loading MuJoCo models."""

import contextlib
import copy
import ctypes
import threading
import weakref

from absl import logging

from dm_control.mujoco.wrapper import util
# Some clients explicitly import core.mjlib.
from dm_control.mujoco.wrapper.mjbindings import mjlib  # pylint: disable=unused-import
import mujoco
import numpy as np

# Unused internal import: resources.

_NULL = b"\00"
_FAKE_BINARY_FILENAME = "model.mjb"

_CONTACT_ID_OUT_OF_RANGE = (
    "`contact_id` must be between 0 and {max_valid} (inclusive), got: {actual}."
)

# Global cache used to store finalizers for freeing ctypes pointers.
# Contains {pointer_address: weakref_object} pairs.
_FINALIZERS = {}


class Error(Exception):
  """Base class for MuJoCo exceptions."""
  pass


if mujoco.mjVERSION_HEADER != mujoco.mj_version():
  raise Error("MuJoCo library version ({0}) does not match header version "
              "({1})".format(mujoco.mjVERSION_HEADER, mujoco.mj_version()))

_REGISTERED = False
_REGISTRATION_LOCK = threading.Lock()

# This is used to keep track of the `MJMODEL` pointer that was most recently
# loaded by `_get_model_ptr_from_xml`. Only this model can be saved to XML.
_LAST_PARSED_MODEL_PTR = None

_NOT_LAST_PARSED_ERROR = (
    "Only the model that was most recently loaded from an XML file or string "
    "can be saved to an XML file.")

import time

# NB: Python functions that are called from C are defined at module-level to
# ensure they won't be garbage-collected before they are called.
@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _warning_callback(message):
  logging.warning(util.to_native_string(message))


@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def _error_callback(message):
  logging.fatal(util.to_native_string(message))


# Override MuJoCo's callbacks for handling warnings
mujoco.set_mju_user_warning(_warning_callback)


def enable_timer(enabled=True):
  if enabled:
    set_callback("mjcb_time", time.time)
  else:
    set_callback("mjcb_time", None)


def _str2type(type_str):
  type_id = mujoco.mju_str2Type(util.to_binary_string(type_str))
  if not type_id:
    raise Error("{!r} is not a valid object type name.".format(type_str))
  return type_id


def _type2str(type_id):
  type_str_ptr = mujoco.mju_type2Str(type_id)
  if not type_str_ptr:
    raise Error("{!r} is not a valid object type ID.".format(type_id))
  return type_str_ptr


def set_callback(name, new_callback=None):
  """Sets a user-defined callback function to modify MuJoCo's behavior.

  Callback functions should have the following signature:
    func(const_mjmodel_ptr, mjdata_ptr) -> None

  Args:
    name: Name of the callback to set. Must be a field in
      `functions.function_pointers`.
    new_callback: The new callback. This can be one of the following:
      * A Python callable
      * A C function exposed by a `ctypes.CDLL` object
      * An integer specifying the address of a callback function
      * None, in which case any existing callback of that name is removed
  """
  getattr(mujoco, "set_" + name)(new_callback)


@contextlib.contextmanager
def callback_context(name, new_callback=None):
  """Context manager that temporarily overrides a MuJoCo callback function.

  On exit, the callback will be restored to its original value (None if the
  callback was not already overridden when the context was entered).

  Args:
    name: Name of the callback to set. Must be a field in
      `mjbindings.function_pointers`.
    new_callback: The new callback. This can be one of the following:
      * A Python callable
      * A C function exposed by a `ctypes.CDLL` object
      * An integer specifying the address of a callback function
      * None, in which case any existing callback of that name is removed

  Yields:
    None
  """
  old_callback = getattr(mujoco, "get_" + name)()
  set_callback(name, new_callback)
  try:
    yield
  finally:
    # Ensure that the callback is reset on exit, even if an exception is raised.
    set_callback(name, old_callback)


def get_schema():
  """Returns a string containing the schema used by the MuJoCo XML parser."""
  buf = ctypes.create_string_buffer(100000)
  mujoco.mj_printSchema(None, buf, len(buf), 0, 0)
  return buf.value


def _get_model_ptr_from_xml(xml_path=None, xml_string=None, assets=None):
  """Parses a model XML file, compiles it, and returns a pointer to an mjModel.

  Args:
    xml_path: None or a path to a model XML file in MJCF or URDF format.
    xml_string: None or an XML string containing an MJCF or URDF model
      description.
    assets: None or a dict containing external assets referenced by the model
      (such as additional XML files, textures, meshes etc.), in the form of
      `{filename: contents_string}` pairs. The keys should correspond to the
      filenames specified in the model XML. Ignored if `xml_string` is None.

    One of `xml_path` or `xml_string` must be specified.

  Returns:
    A `ctypes.POINTER` to a new `mjbindings.types.MJMODEL` instance.

  Raises:
    TypeError: If both or neither of `xml_path` and `xml_string` are specified.
    Error: If the model is not created successfully.
  """
  if xml_path is None and xml_string is None:
    raise TypeError(
        "At least one of `xml_path` or `xml_string` must be specified.")
  elif xml_path is not None and xml_string is not None:
    raise TypeError(
        "Only one of `xml_path` or `xml_string` may be specified.")

  if xml_string is not None:
    ptr = mujoco.MjModel.from_xml_string(xml_string, assets or {})
  else:
    ptr = mujoco.MjModel.from_xml_path(xml_path, assets or {})

  global _LAST_PARSED_MODEL_PTR
  _LAST_PARSED_MODEL_PTR = ptr

  return ptr


def save_last_parsed_model_to_xml(xml_path, check_model=None):
  """Writes a description of the most recently loaded model to an MJCF XML file.

  Args:
    xml_path: Path to the output XML file.
    check_model: Optional `MjModel` instance. If specified, this model will be
      checked to see if it is the most recently parsed one, and a ValueError
      will be raised otherwise.
  Raises:
    Error: If MuJoCo encounters an error while writing the XML file.
    ValueError: If `check_model` was passed, and this model is not the most
      recently parsed one.
  """
  if check_model and check_model.ptr is not _LAST_PARSED_MODEL_PTR:
    raise ValueError(_NOT_LAST_PARSED_ERROR)
  mujoco.mj_saveLastXML(xml_path, _LAST_PARSED_MODEL_PTR)


def _get_model_ptr_from_binary(binary_path=None, byte_string=None):
  """Returns a pointer to an mjModel from the contents of a MuJoCo model binary.

  Args:
    binary_path: Path to an MJB file (as produced by MjModel.save_binary).
    byte_string: String of bytes (as returned by MjModel.to_bytes).

    One of `binary_path` or `byte_string` must be specified.

  Returns:
    A `ctypes.POINTER` to a new `mjbindings.types.MJMODEL` instance.

  Raises:
    TypeError: If both or neither of `byte_string` and `binary_path`
      are specified.
  """
  if binary_path is None and byte_string is None:
    raise TypeError(
        "At least one of `byte_string` or `binary_path` must be specified.")
  elif binary_path is not None and byte_string is not None:
    raise TypeError(
        "Only one of `byte_string` or `binary_path` may be specified.")

  if byte_string is not None:
    assets = {_FAKE_BINARY_FILENAME: byte_string}
    return mujoco.MjModel.from_binary_path(_FAKE_BINARY_FILENAME, assets)
  return mujoco.MjModel.from_binary_path(binary_path, {})


class _MjModelMeta(type):
  """Metaclass which allows MjModel below to delegate to mujoco.MjModel."""

  def __new__(cls, name, bases, dct):
    for attr in dir(mujoco.MjModel):
      if not attr.startswith("_"):
        if attr not in dct:
          # pylint: disable=protected-access
          fget = lambda self, attr=attr: getattr(self._model, attr)
          fset = (
              lambda self, value, attr=attr: setattr(self._model, attr, value))
          # pylint: enable=protected-access
          dct[attr] = property(fget, fset)
    return super().__new__(cls, name, bases, dct)


class MjModel(metaclass=_MjModelMeta):
  """Wrapper class for a MuJoCo 'mjModel' instance.

  MjModel encapsulates features of the model that are expected to remain
  constant. It also contains simulation and visualization options which may be
  changed occasionally, although this is done explicitly by the user.
  """
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, model_ptr):
    """Creates a new MjModel instance from a mujoco.MjModel."""
    self._model = model_ptr

  @property
  def ptr(self):
    """The lower level MjModel instance."""
    return self._model

  def __getstate__(self):
    return self._model

  def __setstate__(self, state):
    self._model = state

  def __copy__(self):
    new_model_ptr = copy.copy(self._model)
    return self.__class__(new_model_ptr)

  @classmethod
  def from_xml_string(cls, xml_string, assets=None):
    """Creates an `MjModel` instance from a model description XML string.

    Args:
      xml_string: String containing an MJCF or URDF model description.
      assets: Optional dict containing external assets referenced by the model
        (such as additional XML files, textures, meshes etc.), in the form of
        `{filename: contents_string}` pairs. The keys should correspond to the
        filenames specified in the model XML.

    Returns:
      An `MjModel` instance.
    """
    model_ptr = _get_model_ptr_from_xml(xml_string=xml_string, assets=assets)
    return cls(model_ptr)

  @classmethod
  def from_byte_string(cls, byte_string):
    """Creates an MjModel instance from a model binary as a string of bytes."""
    model_ptr = _get_model_ptr_from_binary(byte_string=byte_string)
    return cls(model_ptr)

  @classmethod
  def from_xml_path(cls, xml_path):
    """Creates an MjModel instance from a path to a model XML file."""
    model_ptr = _get_model_ptr_from_xml(xml_path=xml_path)
    return cls(model_ptr)

  @classmethod
  def from_binary_path(cls, binary_path):
    """Creates an MjModel instance from a path to a compiled model binary."""
    model_ptr = _get_model_ptr_from_binary(binary_path=binary_path)
    return cls(model_ptr)

  def save_binary(self, binary_path):
    """Saves the MjModel instance to a binary file."""
    mujoco.mj_saveModel(self.ptr, binary_path, None)

  def to_bytes(self):
    """Serialize the model to a string of bytes."""
    bufsize = mujoco.mj_sizeModel(self.ptr)
    buf = np.zeros(shape=(bufsize,), dtype=np.uint8)
    mujoco.mj_saveModel(self.ptr, None, buf)
    return buf.tobytes()

  def copy(self):
    """Returns a copy of this MjModel instance."""
    return self.__copy__()

  def free(self):
    """Frees the native resources held by this MjModel.

    This is an advanced feature for use when manual memory management is
    necessary. This MjModel object MUST NOT be used after this function has
    been called.
    """
    del self._ptr

  def name2id(self, name, object_type):
    """Returns the integer ID of a specified MuJoCo object.

    Args:
      name: String specifying the name of the object to query.
      object_type: The type of the object. Can be either a lowercase string
        (e.g. 'body', 'geom') or an `mjtObj` enum value.

    Returns:
      An integer object ID.

    Raises:
      Error: If `object_type` is not a valid MuJoCo object type, or if no object
        with the corresponding name and type was found.
    """
    if isinstance(object_type, str):
      object_type = _str2type(object_type)
    obj_id = mujoco.mj_name2id(self.ptr, object_type, name)
    if obj_id == -1:
      raise Error("Object of type {!r} with name {!r} does not exist.".format(
          _type2str(object_type), name))
    return obj_id

  def id2name(self, object_id, object_type):
    """Returns the name associated with a MuJoCo object ID, if there is one.

    Args:
      object_id: Integer ID.
      object_type: The type of the object. Can be either a lowercase string
        (e.g. 'body', 'geom') or an `mjtObj` enum value.

    Returns:
      A string containing the object name, or an empty string if the object ID
      either doesn't exist or has no name.

    Raises:
      Error: If `object_type` is not a valid MuJoCo object type.
    """
    if isinstance(object_type, str):
      object_type = _str2type(object_type)
    return mujoco.mj_id2name(self.ptr, object_type, object_id) or ""

  @contextlib.contextmanager
  def disable(self, *flags):
    """Context manager for temporarily disabling MuJoCo flags.

    Args:
      *flags: Positional arguments specifying flags to disable. Can be either
        lowercase strings (e.g. 'gravity', 'contact') or `mjtDisableBit` enum
        values.

    Yields:
      None

    Raises:
      ValueError: If any item in `flags` is neither a valid name nor a value
        from `mujoco.mjtDisableBit`.
    """
    old_bitmask = self.opt.disableflags
    new_bitmask = old_bitmask
    for flag in flags:
      if isinstance(flag, str):
        try:
          field_name = "mjDSBL_" + flag.upper()
          flag = getattr(mujoco.mjtDisableBit, field_name)
        except AttributeError:
          valid_names = [
              field_name.split("_")[1].lower()
              for field_name in list(mujoco.mjtDisableBit.__members__)[:-1]
          ]
          raise ValueError("'{}' is not a valid flag name. Valid names: {}"
                           .format(flag, ", ".join(valid_names))) from None
      elif isinstance(flag, int):
        flag = mujoco.mjtDisableBit(flag)
      new_bitmask |= flag.value
    self.opt.disableflags = new_bitmask
    try:
      yield
    finally:
      self.opt.disableflags = old_bitmask

  @property
  def name(self):
    """Returns the name of the model."""
    # The model's name is the first null terminated string in _model.names
    return str(self._model.names[:self._model.names.find(b"\0")], "ascii")


class _MjDataMeta(type):
  """Metaclass which allows MjData below to delegate to mujoco.MjData."""

  def __new__(cls, name, bases, dct):
    for attr in dir(mujoco.MjData):
      if not attr.startswith("_"):
        if attr not in dct:
          # pylint: disable=protected-access
          fget = lambda self, attr=attr: getattr(self._data, attr)
          fset = lambda self, value, attr=attr: setattr(self._data, attr, value)
          # pylint: enable=protected-access
          dct[attr] = property(fget, fset)
    return super().__new__(cls, name, bases, dct)


class MjData(metaclass=_MjDataMeta):
  """Wrapper class for a MuJoCo 'mjData' instance.

  MjData contains all of the dynamic variables and intermediate results produced
  by the simulation. These are expected to change on each simulation timestep.
  """

  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self, model):
    """Construct a new MjData instance.

    Args:
      model: An MjModel instance.
    """
    self._model = model
    self._data = mujoco.MjData(model._model)

  def __getstate__(self):
    return (self._model, self._data)

  def __setstate__(self, state):
    self._model, self._data = state

  def __copy__(self):
    # This makes a shallow copy that shares the same parent MjModel instance.
    return self._make_copy(share_model=True)

  def _make_copy(self, share_model):
    # TODO(nimrod): Avoid allocating a new MjData just to replace it.
    new_obj = self.__class__(
        self._model if share_model else copy.copy(self._model))
    super(self.__class__, new_obj).__setattr__("_data", copy.copy(self._data))
    return new_obj

  def copy(self):
    """Returns a copy of this MjData instance with the same parent MjModel."""
    return self.__copy__()

  def object_velocity(self, object_id, object_type, local_frame=False):
    """Returns the 6D velocity (linear, angular) of a MuJoCo object.

    Args:
      object_id: Object identifier. Can be either integer ID or String name.
      object_type: The type of the object. Can be either a lowercase string
        (e.g. 'body', 'geom') or an `mjtObj` enum value.
      local_frame: Boolean specifiying whether the velocity is given in the
        global (worldbody), or local (object) frame.

    Returns:
      2x3 array with stacked (linear_velocity, angular_velocity)

    Raises:
      Error: If `object_type` is not a valid MuJoCo object type, or if no object
        with the corresponding name and type was found.
    """
    if not isinstance(object_type, int):
      object_type = _str2type(object_type)
    velocity = np.empty(6, dtype=np.float64)
    if not isinstance(object_id, int):
      object_id = self.model.name2id(object_id, object_type)
    mujoco.mj_objectVelocity(self._model.ptr, self._data, object_type,
                             object_id, velocity, local_frame)
    #  MuJoCo returns velocities in (angular, linear) order, which we flip here.
    return velocity.reshape(2, 3)[::-1]

  def contact_force(self, contact_id):
    """Returns the wrench of a contact as a 2 x 3 array of (forces, torques).

    Args:
      contact_id: Integer, the index of the contact within the contact buffer
        (`self.contact`).

    Returns:
      2x3 array with stacked (force, torque). Note that the order of dimensions
        is (normal, tangent, tangent), in the contact's frame.

    Raises:
      ValueError: If `contact_id` is negative or bigger than ncon-1.
    """
    if not 0 <= contact_id < self.ncon:
      raise ValueError(_CONTACT_ID_OUT_OF_RANGE
                       .format(max_valid=self.ncon-1, actual=contact_id))
    wrench = np.empty(6, dtype=np.float64)
    mujoco.mj_contactForce(self._model.ptr, self._data, contact_id, wrench)
    return wrench.reshape(2, 3)

  @property
  def ptr(self):
    """The lower level MjData instance."""
    return self._data

  @property
  def model(self):
    """The parent MjModel for this MjData instance."""
    return self._model

  @property
  def contact(self):
    """Variable-length recarray containing all current contacts."""
    return self._data.contact[:self.ncon]


# Docstrings for these subclasses are inherited from their parent class.


class MjvCamera(mujoco.MjvCamera):  # pylint: disable=missing-docstring

  # Provide this alias for the "type" property for backwards compatibility.
  @property
  def type_(self):
    return self.type

  @type_.setter
  def type_(self, t):
    self.type = t

  @property
  def ptr(self):
    return self


class MjvOption(mujoco.MjvOption):  # pylint: disable=missing-docstring

  def __init__(self):
    super().__init__()
    self.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

  # Provided for backwards compatibility
  @property
  def ptr(self):
    return self


class MjrContext:
  """Wrapper for mujoco.MjrContext."""

  def __init__(self,
               model,
               gl_context,
               font_scale=mujoco.mjtFontScale.mjFONTSCALE_150):
    """Initializes this MjrContext instance.

    Args:
      model: An `MjModel` instance.
      gl_context: A `render.ContextBase` instance.
      font_scale: Integer controlling the font size for text. Must be a value
        in `mujoco.mjtFontScale`.

    Raises:
      ValueError: If `font_scale` is invalid.
    """
    if not isinstance(font_scale, mujoco.mjtFontScale):
      font_scale = mujoco.mjtFontScale(font_scale)
    self._gl_context = gl_context
    with gl_context.make_current() as ctx:
      ptr = ctx.call(mujoco.MjrContext, model.ptr, font_scale)
      ctx.call(mujoco.mjr_setBuffer, mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ptr)
    gl_context.keep_alive(ptr)
    gl_context.increment_refcount()
    self._ptr = weakref.ref(ptr)

  @property
  def ptr(self):
    return self._ptr()

  def free(self):
    """Frees the native resources held by this MjrContext.

    This is an advanced feature for use when manual memory management is
    necessary. This MjrContext object MUST NOT be used after this function has
    been called.
    """
    if self._gl_context:
      self._gl_context.decrement_refcount()
      self._gl_context.free()
      self._gl_context = None

  def __del__(self):
    self.free()

# A mapping from human-readable short names to mjtRndFlag enum values, i.e.
# {'shadow': mjtRndFlag.mjRND_SHADOW, 'fog': mjtRndFlag.mjRND_FOG, ...}
_NAME_TO_RENDER_FLAG_ENUM_VALUE = {
    name[len("mjRND_"):].lower(): getattr(mujoco.mjtRndFlag, name).value
    for name in mujoco.mjtRndFlag.__members__
    if name != "mjRND_NUMRNDFLAG"
}


def _estimate_max_renderable_geoms(model):
  """Estimates the maximum number of renderable geoms for a given model."""
  # Only one type of object frame can be rendered at once.
  max_nframes = max(
      [model.nbody, model.ngeom, model.nsite, model.ncam, model.nlight])
  # This is probably an underestimate, but it is unlikely that all possible
  # rendering options will be enabled simultaneously, or that all renderable
  # geoms will be present within the viewing frustum at the same time.
  return (
      3 * max_nframes +  # 1 geom per axis for each frame.
      4 * model.ngeom +  # geom itself + contacts + 2 * split contact forces.
      3 * model.nbody +  # COM + inertia box + perturbation force.
      model.nsite +
      model.ntendon +
      model.njnt +
      model.nu +
      model.nskin +
      model.ncam +
      model.nlight)


class MjvScene(mujoco.MjvScene):  # pylint: disable=missing-docstring

  def __init__(self, model=None, max_geom=None):
    """Initializes a new `MjvScene` instance.

    Args:
      model: (optional) An `MjModel` instance.
      max_geom: (optional) An integer specifying the maximum number of geoms
        that can be represented in the scene. If None, this will be chosen
        automatically based on `model`.
    """
    if model is None:
      super().__init__()
    else:
      if max_geom is None:
        if model is None:
          max_renderable_geoms = 0
        else:
          max_renderable_geoms = _estimate_max_renderable_geoms(model)
        max_geom = max(1000, max_renderable_geoms)

      super().__init__(model.ptr, max_geom)

  @property
  def ptr(self):
    return self

  @contextlib.contextmanager
  def override_flags(self, overrides):
    """Context manager for temporarily overriding rendering flags.

    Args:
      overrides: A mapping specifying rendering flags to override. The keys can
        be either lowercase strings or `mjtRndFlag` enum values, and the values
        are the overridden flag values, e.g. `{'wireframe': True}` or
        `{mujoco.mjtRndFlag.mjRND_WIREFRAME: True}`. See `mujoco.mjtRndFlag` for
        the set of valid flags.

    Yields:
      None
    """
    if not overrides:
      yield
    else:
      original_flags = self.flags.copy()
      for key, value in overrides.items():
        index = _NAME_TO_RENDER_FLAG_ENUM_VALUE.get(key, key)
        self.flags[index] = value
      try:
        yield
      finally:
        np.copyto(self.flags, original_flags)

  def free(self):
    """Frees the native resources held by this MjvScene.

    This is an advanced feature for use when manual memory management is
    necessary. This MjvScene object MUST NOT be used after this function has
    been called.
    """
    pass

  @property
  def geoms(self):
    """Variable-length recarray containing all geoms currently in the buffer."""
    return super().geoms[:super().ngeom]


class MjvPerturb(mujoco.MjvPerturb):  # pylint: disable=missing-docstring

  @property
  def ptr(self):
    return self


class MjvFigure(mujoco.MjvFigure):  # pylint: disable=missing-docstring

  @property
  def ptr(self):
    return self

  @property
  def range_(self):
    return self.range

  @range_.setter
  def range_(self, value):
    self.range = value
