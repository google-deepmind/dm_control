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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import ctypes
import os
import weakref

from absl import logging

from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import constants
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import functions
from dm_control.mujoco.wrapper.mjbindings import mjlib
from dm_control.mujoco.wrapper.mjbindings import types
from dm_control.mujoco.wrapper.mjbindings import wrappers
import numpy as np
import six

# Internal analytics import.
# Unused internal import: resources.

_NULL = b"\00"
_FAKE_XML_FILENAME = b"model.xml"
_FAKE_BINARY_FILENAME = b"model.mjb"

# Although `mjMAXVFSNAME` from `mjmodel.h` specifies a limit of 100 bytes
# (including the terminal null byte), the actual limit seems to be 99 bytes
# (98 characters).
_MAX_VFS_FILENAME_CHARACTERS = 98
_VFS_FILENAME_TOO_LONG = (
    "Filename length {length} exceeds {limit} character limit: {filename}")
_INVALID_FONT_SCALE = ("`font_scale` must be one of {}, got {{}}."
                       .format(enums.mjtFontScale))

# Global cache used to store finalizers for freeing ctypes pointers.
# Contains {pointer_address: weakref_object} pairs.
_FINALIZERS = {}


class Error(Exception):
  """Base class for MuJoCo exceptions."""
  pass


if constants.mjVERSION_HEADER != mjlib.mj_version():
  raise Error("MuJoCo library version ({0}) does not match header version "
              "({1})".format(constants.mjVERSION_HEADER, mjlib.mj_version()))

_REGISTERED = False
_ERROR_BUFSIZE = 1000

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


# Override MuJoCo's callbacks for handling warnings and errors.
mjlib.mju_user_warning = ctypes.c_void_p.in_dll(mjlib, "mju_user_warning")
mjlib.mju_user_error = ctypes.c_void_p.in_dll(mjlib, "mju_user_error")
mjlib.mju_user_warning.value = ctypes.cast(
    _warning_callback, ctypes.c_void_p).value
mjlib.mju_user_error.value = ctypes.cast(
    _error_callback, ctypes.c_void_p).value


def enable_timer(enabled=True):
  if enabled:
    set_callback("mjcb_time", time.time)
  else:
    set_callback("mjcb_time", None)


def _maybe_register_license(path=None):
  """Registers the MuJoCo license if not already registered.

  Args:
    path: Optional custom path to license key file.

  Raises:
    Error: If the license could not be registered.
  """
  global _REGISTERED
  if not _REGISTERED:
    if path is None:
      path = util.get_mjkey_path()
    result = mjlib.mj_activate(util.to_binary_string(path))
    if result == 1:
      _REGISTERED = True
      # Internal analytics of mj_activate.
    elif result == 0:
      raise Error("Could not register license.")
    else:
      raise Error("Unknown registration error (code: {})".format(result))


def _str2type(type_str):
  type_id = mjlib.mju_str2Type(util.to_binary_string(type_str))
  if not type_id:
    raise Error("{!r} is not a valid object type name.".format(type_str))
  return type_id


def _type2str(type_id):
  type_str_ptr = mjlib.mju_type2Str(type_id)
  if not type_str_ptr:
    raise Error("{!r} is not a valid object type ID.".format(type_id))
  return ctypes.string_at(type_str_ptr)


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
  setattr(functions.callbacks, name, new_callback)


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
  old_callback = getattr(functions.callbacks, name)
  set_callback(name, new_callback)
  try:
    yield
  finally:
    # Ensure that the callback is reset on exit, even if an exception is raised.
    set_callback(name, old_callback)


def get_schema():
  """Returns a string containing the schema used by the MuJoCo XML parser."""
  buf = ctypes.create_string_buffer(100000)
  mjlib.mj_printSchema(None, buf, len(buf), 0, 0)
  return buf.value


@contextlib.contextmanager
def _temporary_vfs(filenames_and_contents):
  """Creates a temporary VFS containing one or more files.

  Args:
    filenames_and_contents: A dict containing `{filename: contents}` pairs.
      The length of each filename must not exceed 98 characters.

  Yields:
    A `types.MJVFS` instance.

  Raises:
    Error: If a file cannot be added to the VFS, or if an error occurs when
      looking up the filename.
    ValueError: If the length of a filename exceeds 98 characters.
  """
  vfs = types.MJVFS()
  mjlib.mj_defaultVFS(vfs)
  for filename, contents in six.iteritems(filenames_and_contents):
    if len(filename) > _MAX_VFS_FILENAME_CHARACTERS:
      raise ValueError(
          _VFS_FILENAME_TOO_LONG.format(
              length=len(filename),
              limit=_MAX_VFS_FILENAME_CHARACTERS,
              filename=filename))
    filename = util.to_binary_string(filename)
    contents = util.to_binary_string(contents)
    _, extension = os.path.splitext(filename)
    # For XML files we need to append a NULL byte, otherwise MuJoCo's parser
    # can sometimes read past the end of the string. However, we should *not*
    # do this for other file types (in particular for STL meshes, where this
    # causes MuJoCo's compiler to complain that the file size is incorrect).
    append_null = extension.lower() == b".xml"
    num_bytes = len(contents) + append_null
    retcode = mjlib.mj_makeEmptyFileVFS(vfs, filename, num_bytes)
    if retcode == 1:
      raise Error("Failed to create {!r}: VFS is full.".format(filename))
    elif retcode == 2:
      raise Error("Failed to create {!r}: duplicate filename.".format(filename))
    file_index = mjlib.mj_findFileVFS(vfs, filename)
    if file_index == -1:
      raise Error("Could not find {!r} in the VFS".format(filename))
    vf = vfs.filedata[file_index]
    vf_as_char_arr = ctypes.cast(vf, ctypes.POINTER(ctypes.c_char * num_bytes))
    vf_as_char_arr.contents[:len(contents)] = contents
    if append_null:
      vf_as_char_arr.contents[-1] = _NULL
  try:
    yield vfs
  finally:
    mjlib.mj_deleteVFS(vfs)  # Ensure that we free the VFS afterwards.


def _create_finalizer(ptr, free_func):
  """Creates a finalizer for a ctypes pointer.

  Args:
    ptr: A `ctypes.POINTER` to be freed.
    free_func: A callable that frees the pointer. It will be called with `ptr`
      as its only argument when `ptr` is garbage collected.
  """
  ptr_type = type(ptr)
  address = ctypes.addressof(ptr.contents)

  if address not in _FINALIZERS:  # Only one finalizer needed per address.

    logging.debug("Allocated %s at %x", ptr_type.__name__, address)

    def callback(dead_ptr_ref):
      """A weakref callback that frees the resource held by a pointer."""
      del dead_ptr_ref  # Unused weakref to the dead ctypes pointer object.
      if address not in _FINALIZERS:
        # Someone had already explicitly called `call_finalizer_for_pointer`.
        return
      else:
        # Turn the address back into a pointer to be freed.
        temp_ptr = ctypes.cast(address, ptr_type)
        free_func(temp_ptr)
        logging.debug("Freed %s at %x", ptr_type.__name__, address)
        del _FINALIZERS[address]  # Remove the weakref from the global cache.

    # Store weakrefs in a global cache so that they don't get garbage collected
    # before their referents.
    _FINALIZERS[address] = (weakref.ref(ptr, callback), callback)


def _finalize(ptr):
  """Calls the finalizer for the specified pointer to free allocated memory."""
  address = ctypes.addressof(ptr.contents)
  try:
    ptr_ref, callback = _FINALIZERS[address]
    callback(ptr_ref)
  except KeyError:
    pass


def _load_xml(filename, vfs_or_none):
  """Invokes `mj_loadXML` with logging/error handling."""
  error_buf = ctypes.create_string_buffer(_ERROR_BUFSIZE)
  model_ptr = mjlib.mj_loadXML(
      util.to_binary_string(filename),
      vfs_or_none,
      error_buf,
      _ERROR_BUFSIZE)
  if not model_ptr:
    raise Error(util.to_native_string(error_buf.value))
  elif error_buf.value:
    logging.warning(util.to_native_string(error_buf.value))

  # Free resources when the ctypes pointer is garbage collected.
  _create_finalizer(model_ptr, mjlib.mj_deleteModel)

  return model_ptr


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

  _maybe_register_license()

  if xml_string is not None:
    assets = {} if assets is None else assets.copy()
    # Ensure that the fake XML filename doesn't overwrite an existing asset.
    xml_path = _FAKE_XML_FILENAME
    while xml_path in assets:
      xml_path = "_" + xml_path
    assets[xml_path] = xml_string
    with _temporary_vfs(assets) as vfs:
      ptr = _load_xml(xml_path, vfs)
  else:
    ptr = _load_xml(xml_path, None)

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
  error_buf = ctypes.create_string_buffer(_ERROR_BUFSIZE)
  mjlib.mj_saveLastXML(util.to_binary_string(xml_path),
                       _LAST_PARSED_MODEL_PTR,
                       error_buf,
                       _ERROR_BUFSIZE)
  if error_buf.value:
    raise Error(error_buf.value)


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

  _maybe_register_license()

  if byte_string is not None:
    with _temporary_vfs({_FAKE_BINARY_FILENAME: byte_string}) as vfs:
      ptr = mjlib.mj_loadModel(_FAKE_BINARY_FILENAME, vfs)
  else:
    ptr = mjlib.mj_loadModel(util.to_binary_string(binary_path), None)

  # Free resources when the ctypes pointer is garbage collected.
  _create_finalizer(ptr, mjlib.mj_deleteModel)

  return ptr


# Subclasses implementing constructors/destructors for low-level wrappers.
# ------------------------------------------------------------------------------


class MjModel(wrappers.MjModelWrapper):
  """Wrapper class for a MuJoCo 'mjModel' instance.

  MjModel encapsulates features of the model that are expected to remain
  constant. It also contains simulation and visualization options which may be
  changed occasionally, although this is done explicitly by the user.
  """

  def __init__(self, model_ptr):
    """Creates a new MjModel instance from a ctypes pointer.

    Args:
      model_ptr: A `ctypes.POINTER` to an `mjbindings.types.MJMODEL` instance.
    """
    super(MjModel, self).__init__(ptr=model_ptr)

  def __getstate__(self):
    # All of MjModel's state is assumed to reside within the MuJoCo C struct.
    # However there is no mechanism to prevent users from adding arbitrary
    # Python attributes to an MjModel instance - these would not be serialized.
    return self.to_bytes()

  def __setstate__(self, byte_string):
    model_ptr = _get_model_ptr_from_binary(byte_string=byte_string)
    self.__init__(model_ptr)

  def __copy__(self):
    new_model_ptr = mjlib.mj_copyModel(None, self.ptr)
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
    mjlib.mj_saveModel(self.ptr, util.to_binary_string(binary_path), None, 0)

  def to_bytes(self):
    """Serialize the model to a string of bytes."""
    bufsize = mjlib.mj_sizeModel(self.ptr)
    buf = ctypes.create_string_buffer(bufsize)
    mjlib.mj_saveModel(self.ptr, None, buf, bufsize)
    return buf.raw

  def copy(self):
    """Returns a copy of this MjModel instance."""
    return self.__copy__()

  def free(self):
    """Frees the native resources held by this MjModel.

    This is an advanced feature for use when manual memory management is
    necessary. This MjModel object MUST NOT be used after this function has
    been called.
    """
    _finalize(self._ptr)
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
    if not isinstance(object_type, int):
      object_type = _str2type(object_type)
    obj_id = mjlib.mj_name2id(
        self.ptr, object_type, util.to_binary_string(name))
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
    if not isinstance(object_type, int):
      object_type = _str2type(object_type)
    name_ptr = mjlib.mj_id2name(self.ptr, object_type, object_id)
    if not name_ptr:
      return ""
    return util.to_native_string(ctypes.string_at(name_ptr))

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
        from `enums.mjtDisableBit`.
    """
    old_bitmask = self.opt.disableflags
    new_bitmask = old_bitmask
    for flag in flags:
      if isinstance(flag, six.string_types):
        try:
          field_name = "mjDSBL_" + flag.upper()
          bitmask = getattr(enums.mjtDisableBit, field_name)
        except AttributeError:
          valid_names = [field_name.split("_")[1].lower()
                         for field_name in enums.mjtDisableBit._fields[:-1]]
          raise ValueError("'{}' is not a valid flag name. Valid names: {}"
                           .format(flag, ", ".join(valid_names)))
      else:
        if flag not in enums.mjtDisableBit[:-1]:
          raise ValueError("'{}' is not a value in `enums.mjtDisableBit`. "
                           "Valid values: {}"
                           .format(flag, tuple(enums.mjtDisableBit[:-1])))
        bitmask = flag
      new_bitmask |= bitmask
    self.opt.disableflags = new_bitmask
    try:
      yield
    finally:
      self.opt.disableflags = old_bitmask

  @property
  def name(self):
    """Returns the name of the model."""
    # The model name is the first null-terminated string in the `names` buffer.
    return util.to_native_string(
        ctypes.string_at(ctypes.addressof(self.names.contents)))


class MjData(wrappers.MjDataWrapper):
  """Wrapper class for a MuJoCo 'mjData' instance.

  MjData contains all of the dynamic variables and intermediate results produced
  by the simulation. These are expected to change on each simulation timestep.
  """

  def __init__(self, model):
    """Construct a new MjData instance.

    Args:
      model: An MjModel instance.
    """
    self._model = model

    # Allocate resources for mjData.
    data_ptr = mjlib.mj_makeData(model.ptr)

    # Free resources when the ctypes pointer is garbage collected.
    _create_finalizer(data_ptr, mjlib.mj_deleteData)

    super(MjData, self).__init__(data_ptr, model)

  def __getstate__(self):
    # Note: we can replace this once a `saveData` MJAPI function exists.
    # To reconstruct an MjData instance we need three things:
    #   1. Its parent MjModel instance
    #   2. A subset of its fixed-size fields whose values aren't determined by
    #      the model
    #   3. The contents of its internal buffer (all of its pointer fields point
    #      into this)
    struct_fields = {}
    for name in ["solver", "timer", "warning"]:
      struct_fields[name] = getattr(self, name).copy()
    scalar_field_names = ["ncon", "time", "energy"]
    scalar_fields = {name: getattr(self, name) for name in scalar_field_names}
    static_fields = {"struct_fields": struct_fields,
                     "scalar_fields": scalar_fields}
    buffer_contents = ctypes.string_at(self.buffer_, self.nbuffer)
    return (self._model, static_fields, buffer_contents)

  def __setstate__(self, state_tuple):
    # Replace this once a `loadData` MJAPI function exists.
    self._model, static_fields, buffer_contents = state_tuple
    self.__init__(self.model)
    for name, contents in six.iteritems(static_fields["struct_fields"]):
      getattr(self, name)[:] = contents

    for name, value in six.iteritems(static_fields["scalar_fields"]):
      # Array and scalar values must be handled separately.
      try:
        getattr(self, name)[:] = value
      except TypeError:
        setattr(self, name, value)
    buf_ptr = (ctypes.c_char * self.nbuffer).from_address(self.buffer_)
    buf_ptr[:] = buffer_contents

  def __copy__(self):
    # This makes a shallow copy that shares the same parent MjModel instance.
    new_obj = self.__class__(self.model)
    mjlib.mj_copyData(new_obj.ptr, self.model.ptr, self.ptr)
    return new_obj

  def copy(self):
    """Returns a copy of this MjData instance with the same parent MjModel."""
    return self.__copy__()

  def free(self):
    """Frees the native resources held by this MjData.

    This is an advanced feature for use when manual memory management is
    necessary. This MjData object MUST NOT be used after this function has
    been called.
    """
    _finalize(self._ptr)
    del self._ptr

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
    mjlib.mj_objectVelocity(self.model.ptr, self.ptr,
                            object_type, object_id, velocity, local_frame)
    #  MuJoCo returns velocities in (angular, linear) order, which we flip here.
    return velocity.reshape(2, 3)[::-1]

  @property
  def model(self):
    """The parent MjModel for this MjData instance."""
    return self._model

  @util.CachedProperty
  def _contact_buffer(self):
    """Cached structured array containing the full contact buffer."""
    contact_array = util.buf_to_npy(
        super(MjData, self).contact, shape=(self._model.nconmax,))
    return contact_array

  @property
  def contact(self):
    """Variable-length recarray containing all current contacts."""
    return self._contact_buffer[:self.ncon]


# Docstrings for these subclasses are inherited from their Wrapper parent class.


class MjvCamera(wrappers.MjvCameraWrapper):

  def __init__(self):
    ptr = ctypes.pointer(types.MJVCAMERA())
    mjlib.mjv_defaultCamera(ptr)
    super(MjvCamera, self).__init__(ptr)


class MjvOption(wrappers.MjvOptionWrapper):

  def __init__(self):
    ptr = ctypes.pointer(types.MJVOPTION())
    mjlib.mjv_defaultOption(ptr)
    # Do not visualize rangefinder lines by default:
    ptr.contents.flags[enums.mjtVisFlag.mjVIS_RANGEFINDER] = False
    super(MjvOption, self).__init__(ptr)


class UnmanagedMjrContext(wrappers.MjrContextWrapper):
  """A wrapper for MjrContext that does not manage the native object's lifetime.

  This wrapper is provided for API backward-compatibility reasons, since the
  creating and destruction of an MjrContext requires an OpenGL context to be
  provided.
  """

  def __init__(self):
    ptr = ctypes.pointer(types.MJRCONTEXT())
    mjlib.mjr_defaultContext(ptr)
    super(UnmanagedMjrContext, self).__init__(ptr)


class MjrContext(wrappers.MjrContextWrapper):  # pylint: disable=missing-docstring

  def __init__(self,
               model,
               gl_context,
               font_scale=enums.mjtFontScale.mjFONTSCALE_150):
    """Initializes this MjrContext instance.

    Args:
      model: An `MjModel` instance.
      gl_context: A `render.ContextBase` instance.
      font_scale: Integer controlling the font size for text. Must be a value
        in `mjbindings.enums.mjtFontScale`.

    Raises:
      ValueError: If `font_scale` is invalid.
    """
    if font_scale not in enums.mjtFontScale:
      raise ValueError(_INVALID_FONT_SCALE.format(font_scale))

    ptr = ctypes.pointer(types.MJRCONTEXT())
    mjlib.mjr_defaultContext(ptr)

    with gl_context.make_current() as ctx:
      ctx.call(mjlib.mjr_makeContext, model.ptr, ptr, font_scale)
      ctx.call(mjlib.mjr_setBuffer, enums.mjtFramebuffer.mjFB_OFFSCREEN, ptr)
      gl_context.increment_refcount()

    # Free resources when the ctypes pointer is garbage collected.
    def finalize_mjr_context(ptr):
      if not gl_context.terminated:
        with gl_context.make_current() as ctx:
          ctx.call(mjlib.mjr_freeContext, ptr)
          gl_context.decrement_refcount()

    _create_finalizer(ptr, finalize_mjr_context)

    super(MjrContext, self).__init__(ptr)

  def free(self):
    """Frees the native resources held by this MjrContext.

    This is an advanced feature for use when manual memory management is
    necessary. This MjrContext object MUST NOT be used after this function has
    been called.
    """
    _finalize(self._ptr)
    del self._ptr


class MjvScene(wrappers.MjvSceneWrapper):  # pylint: disable=missing-docstring

  def __init__(self, model=None, max_geom=1000):
    """Initializes a new `MjvScene` instance.

    Args:
      model: (optional) An `MjModel` instance.
      max_geom: (optional) An integer specifying the maximum number of geoms
        that can be represented in the scene.
    """
    model_ptr = model.ptr if model is not None else None
    scene_ptr = ctypes.pointer(types.MJVSCENE())

    # Allocate and initialize resources for the abstract scene.
    mjlib.mjv_makeScene(model_ptr, scene_ptr, max_geom)

    # Free resources when the ctypes pointer is garbage collected.
    _create_finalizer(scene_ptr, mjlib.mjv_freeScene)

    super(MjvScene, self).__init__(scene_ptr)

  @contextlib.contextmanager
  def override_flags(self, new_flags):
    """Context manager for temporarily overriding rendering flags.

    Args:
      new_flags: A mapping from `enums.mjtRndFlag` values to the corresponding
        overridden flag values.

    Yields:
      None
    """
    if not new_flags:
      yield
    else:
      original_flags = self.flags.copy()
      for index, value in new_flags.items():
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
    _finalize(self._ptr)
    del self._ptr

  @util.CachedProperty
  def _geoms_buffer(self):
    """Cached recarray containing the full geom buffer."""
    return util.buf_to_npy(super(MjvScene, self).geoms, shape=(self.maxgeom,))

  @property
  def geoms(self):
    """Variable-length recarray containing all geoms currently in the buffer."""
    return self._geoms_buffer[:self.ngeom]


class MjvPerturb(wrappers.MjvPerturbWrapper):

  def __init__(self):
    ptr = ctypes.pointer(types.MJVPERTURB())
    mjlib.mjv_defaultPerturb(ptr)
    super(MjvPerturb, self).__init__(ptr)


class MjvFigure(wrappers.MjvFigureWrapper):

  def __init__(self):
    ptr = ctypes.pointer(types.MJVFIGURE())
    mjlib.mjv_defaultFigure(ptr)
    super(MjvFigure, self).__init__(ptr)
