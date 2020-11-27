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

"""Classes representing various MJCF attribute data types."""

import abc
import collections
import hashlib
import io
import os

from dm_control.mjcf import base
from dm_control.mjcf import constants
from dm_control.mjcf import debugging
from dm_control.mjcf import skin
from dm_control.mujoco.wrapper import util
import numpy as np

# Copybara placeholder for internal file handling dependency.

from dm_control.utils import io as resources


_INVALID_REFERENCE_TYPE = (
    'Reference should be an MJCF Element whose type is {valid_type!r}: '
    'got {actual_type!r}.')

_MESH_EXTENSIONS = ('.stl', '.msh')

# MuJoCo's compiler enforces this.
_INVALID_MESH_EXTENSION = (
    'Mesh files must have one of the following extensions: {}, got {{}}.'
    .format(_MESH_EXTENSIONS))


class _Attribute(metaclass=abc.ABCMeta):
  """Abstract base class for MJCF attribute data types."""

  def __init__(self, name, required, parent, value,
               conflict_allowed, conflict_behavior):
    self._name = name
    self._required = required
    self._parent = parent
    self._value = None
    self._conflict_allowed = conflict_allowed
    self._conflict_behavior = conflict_behavior
    self._check_and_assign(value)

  def _check_and_assign(self, new_value):
    if new_value is None:
      self.clear()
    elif isinstance(new_value, str):
      self._assign_from_string(new_value)
    else:
      self._assign(new_value)
    if debugging.debug_mode():
      self._last_modified_stack = debugging.get_current_stack_trace()

  @property
  def last_modified_stack(self):
    if debugging.debug_mode():
      return self._last_modified_stack

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, new_value):
    self._check_and_assign(new_value)

  @abc.abstractmethod
  def _assign(self, value):
    raise NotImplementedError  # pragma: no cover

  def clear(self):
    if self._required:
      raise AttributeError(
          'Attribute {!r} of element <{}> is required'
          .format(self._name, self._parent.tag))
    else:
      self._force_clear()

  def _force_clear(self):
    self._before_clear()
    self._value = None
    if debugging.debug_mode():
      self._last_modified_stack = debugging.get_current_stack_trace()

  def _before_clear(self):
    pass

  def _assign_from_string(self, string):
    self._assign(string)

  def to_xml_string(self, prefix_root):  # pylint: disable=unused-argument
    if self._value is None:
      return None
    else:
      return str(self._value)

  @property
  def conflict_allowed(self):
    return self._conflict_allowed

  @property
  def conflict_behavior(self):
    return self._conflict_behavior


class String(_Attribute):
  """A string MJCF attribute."""

  def _assign(self, value):
    if not isinstance(value, str):
      raise ValueError('Expect a string value: got {}'.format(value))
    elif not value:
      self.clear()
    else:
      self._value = value


class Integer(_Attribute):
  """An integer MJCF attribute."""

  def _assign(self, value):
    try:
      float_value = float(value)
      int_value = int(float(value))
      if float_value != int_value:
        raise ValueError
    except ValueError:
      raise ValueError('Expect an integer value: got {}'.format(value))
    self._value = int_value


class Float(_Attribute):
  """An float MJCF attribute."""

  def _assign(self, value):
    try:
      float_value = float(value)
    except ValueError:
      raise ValueError('Expect a float value: got {}'.format(value))
    self._value = float_value


class Keyword(_Attribute):
  """A keyword MJCF attribute."""

  def __init__(self, name, required, parent, value,
               conflict_allowed, conflict_behavior, valid_values):
    self._valid_values = collections.OrderedDict(
        (value.lower(), value) for value in valid_values)
    super(Keyword, self).__init__(
        name, required, parent, value, conflict_allowed, conflict_behavior)

  def _assign(self, value):
    if not value:
      self.clear()
    else:
      try:
        self._value = self._valid_values[str(value).lower()]
      except KeyError:
        raise ValueError('Expect keyword to be one of {} but got: {}'.format(
            list(self._valid_values.values()), value))

  @property
  def valid_values(self):
    return list(self._valid_values.keys())


class Array(_Attribute):
  """An array MJCF attribute."""

  def __init__(self, name, required, parent, value,
               conflict_allowed, conflict_behavior, length, dtype):
    self._length = length
    self._dtype = dtype
    super(Array, self).__init__(name, required, parent, value,
                                conflict_allowed, conflict_behavior)

  def _assign(self, value):
    self._value = self._check_shape(np.array(value, dtype=self._dtype))

  def _assign_from_string(self, string):
    self._assign(np.fromstring(string, dtype=self._dtype, sep=' '))

  def to_xml_string(self, prefix_root=None):  # pylint: disable=unused-argument
    if self._value is None:
      return None
    else:
      out = io.BytesIO()
      # 17 decimal digits is sufficient to represent a double float without loss
      # of precision.
      # https://en.wikipedia.org/wiki/IEEE_754#Character_representation
      np.savetxt(out, self._value, fmt='%.17g', newline=' ')
      return util.to_native_string(out.getvalue())[:-1]  # Strip trailing space.

  def _check_shape(self, array):
    actual_length = array.shape[0]
    if len(array.shape) > 1:
      raise ValueError('Expect one-dimensional array: got {}'.format(array))
    if self._length and actual_length > self._length:
      raise ValueError('Expect array with no more than {} entries: got {}'
                       .format(self._length, array))
    return array


class Identifier(_Attribute):
  """A string attribute that represents a unique identifier of an element."""

  def _assign(self, value):
    if not isinstance(value, str):
      raise ValueError('Expect a string value: got {}'.format(value))
    elif not value:
      self.clear()
    elif self._parent.spec.namespace == 'body' and value == 'world':
      raise ValueError('A body cannot be named \'world\'. '
                       'The name \'world\' is used by MuJoCo to refer to the '
                       '<worldbody>.')
    elif constants.PREFIX_SEPARATOR in value:
      raise ValueError(
          'An identifier cannot contain a {!r}, '
          'as this is reserved for scoping purposes: got {!r}'
          .format(constants.PREFIX_SEPARATOR, value))
    else:
      old_value = self._value
      if value != old_value:
        self._parent.namescope.add(
            self._parent.spec.namespace, value, self._parent)
        if old_value:
          self._parent.namescope.remove(self._parent.spec.namespace, old_value)
      self._value = value

  def _before_clear(self):
    if self._value:
      self._parent.namescope.remove(self._parent.spec.namespace, self._value)

  def _defaults_string(self, prefix_root):
    prefix = self._parent.namescope.full_prefix(prefix_root, as_list=True)
    prefix.append(self._value or '')
    return constants.PREFIX_SEPARATOR.join(prefix) or constants.PREFIX_SEPARATOR

  def to_xml_string(self, prefix_root=None):
    if self._parent.tag == constants.DEFAULT:
      return self._defaults_string(prefix_root)
    elif self._value:
      prefix = self._parent.namescope.full_prefix(prefix_root, as_list=True)
      prefix.append(self._value)
      return constants.PREFIX_SEPARATOR.join(prefix)
    else:
      return self._value


class Reference(_Attribute):
  """A string attribute that represents a reference to an identifier."""

  def __init__(self, name, required, parent, value,
               conflict_allowed, conflict_behavior, reference_namespace):
    self._reference_namespace = reference_namespace
    super(Reference, self).__init__(
        name, required, parent, value, conflict_allowed, conflict_behavior)

  def _check_dead_reference(self):
    if isinstance(self._value, base.Element) and self._value.is_removed:
      self.clear()

  @property
  def value(self):
    self._check_dead_reference()
    return super(Reference, self).value

  @value.setter
  def value(self, new_value):
    super(Reference, self.__class__).value.fset(self, new_value)

  @property
  def reference_namespace(self):
    if isinstance(self._reference_namespace, _Attribute):
      return constants.INDIRECT_REFERENCE_ATTRIB.get(
          self._reference_namespace.value, self._reference_namespace.value)
    else:
      return self._reference_namespace

  def _assign(self, value):
    if not isinstance(value, (base.Element, str)):
      raise ValueError(
          'Expect a string or `mjcf.Element` value: got {}'.format(value))
    elif not value:
      self.clear()
    else:
      if isinstance(value, base.Element):
        value_namespace = (
            value.spec.namespace.split(constants.NAMESPACE_SEPARATOR)[0])
        if value_namespace != self.reference_namespace:
          raise ValueError(_INVALID_REFERENCE_TYPE.format(
              valid_type=self.reference_namespace,
              actual_type=value_namespace))
      self._value = value

  def _before_clear(self):
    if isinstance(self._value, base.Element):
      if isinstance(self._reference_namespace, _Attribute):
        self._reference_namespace._force_clear()  # pylint: disable=protected-access

  def _defaults_string(self, prefix_root):
    """Generates the XML string if this is a reference to a defaults class.

    To prevent global defaults from clashing, we turn all global defaults
    into a properly named defaults class. Therefore, care must be taken when
    this attribute is not explicitly defined. If the parent element can be
    traced up to a body with a nontrivial 'childclass' then must continue to
    leave this attribute undefined.

    Args:
      prefix_root: A `NameScope` object to be treated as root
        for the purpose of calculating the prefix.

    Returns:
      A string to be used in the generated XML.
    """
    self._check_dead_reference()
    prefix = self._parent.namescope.full_prefix(prefix_root)
    if not self._value:
      defaults_root = self._parent.parent
      while defaults_root is not None:
        if (hasattr(defaults_root, constants.CHILDCLASS)
            and defaults_root.childclass):
          break
        defaults_root = defaults_root.parent
      if defaults_root is None:
        # This element doesn't belong to a childclass'd body.
        global_class = self._parent.root.default.dclass or ''
        out_string = (prefix + global_class) or constants.PREFIX_SEPARATOR
      else:
        out_string = None
    else:
      out_string = prefix + self._value
    return out_string

  def to_xml_string(self, prefix_root):
    self._check_dead_reference()
    if isinstance(self._value, base.Element):
      return self._value.prefixed_identifier(prefix_root)
    elif (self.reference_namespace == constants.DEFAULT
          and self._name != constants.CHILDCLASS):
      return self._defaults_string(prefix_root)
    elif self._value:
      return self._parent.namescope.full_prefix(prefix_root) + self._value
    else:
      return None


class BasePath(_Attribute):
  """A string attribute that represents a base path for an asset type."""

  def __init__(self, name, required, parent, value,
               conflict_allowed, conflict_behavior, path_namespace):
    self._path_namespace = path_namespace
    super(BasePath, self).__init__(
        name, required, parent, value, conflict_allowed, conflict_behavior)

  def _assign(self, value):
    if not isinstance(value, str):
      raise ValueError('Expect a string value: got {}'.format(value))
    elif not value:
      self.clear()
    else:
      self._parent.namescope.replace(
          constants.BASEPATH, self._path_namespace, value)
      self._value = value

  def _before_clear(self):
    if self._value:
      self._parent.namescope.remove(constants.BASEPATH, self._path_namespace)

  def to_xml_string(self, prefix_root=None):
    return None


class BaseAsset:
  """Base class for binary assets."""

  __slots__ = ('extension', 'prefix')

  def __init__(self, extension, prefix=''):
    self.extension = extension
    self.prefix = prefix

  def __eq__(self, other):
    return self.get_vfs_filename() == other.get_vfs_filename()

  def get_vfs_filename(self):
    """Returns the name of the asset file as registered in MuJoCo's VFS."""
    # Hash the contents of the asset to get a unique identifier.
    hash_string = hashlib.sha1(util.to_binary_string(self.contents)).hexdigest()
    # Prepend the prefix, if one exists.
    if self.prefix:
      prefix = self.prefix
      raw_length = len(prefix) + len(hash_string) + len(self.extension) + 1
      if raw_length > constants.MAX_VFS_FILENAME_LENGTH:
        trim_amount = raw_length - constants.MAX_VFS_FILENAME_LENGTH
        prefix = prefix[:-trim_amount]
      filename = '-'.join([prefix, hash_string])
    else:
      filename = hash_string

    # An extension is needed because MuJoCo's compiler looks at this when
    # deciding how to load meshes and heightfields.
    return filename + self.extension


class Asset(BaseAsset):
  """Class representing a binary asset."""

  __slots__ = ('contents',)

  def __init__(self, contents, extension, prefix=''):
    """Initializes a new `Asset`.

    Args:
      contents: The contents of the file as a bytestring.
      extension: A string specifying the file extension (e.g. '.png', '.stl').
      prefix: (optional) A prefix applied to the filename given in MuJoCo's VFS.
    """
    self.contents = contents
    super(Asset, self).__init__(extension, prefix)


class SkinAsset(BaseAsset):
  """Class representing a binary asset corresponding to a skin."""

  __slots__ = ('skin', 'parent', '_cached_revision', '_cached_contents')

  def __init__(self, contents, parent, extension, prefix=''):
    self.skin = skin.parse(
        contents, lambda body_name: parent.root.find('body', body_name))
    self.parent = parent
    self._cached_revision = -1
    self._cached_contents = None
    super(SkinAsset, self).__init__(extension, prefix)

  @property
  def contents(self):
    if self._cached_revision < self.parent.namescope.revision:
      self._cached_contents = skin.serialize(self.skin)
      self._cached_revision = self.parent.namescope.revision
    return self._cached_contents


class File(_Attribute):
  """Attribute representing an asset file."""

  def __init__(self, name, required, parent, value,
               conflict_allowed, conflict_behavior, path_namespace):
    self._path_namespace = path_namespace
    super(File, self).__init__(name, required, parent, value,
                               conflict_allowed, conflict_behavior)
    parent.namescope.files.add(self)

  def _assign(self, value):
    if not value:
      self.clear()
    else:
      if isinstance(value, str):
        asset = self._get_asset_from_path(value)
      elif isinstance(value, Asset):
        asset = value
      else:
        raise ValueError('Expect either a string or `Asset` value: got {}'
                         .format(value))
      self._validate_extension(asset.extension)
      self._value = asset

  def _get_asset_from_path(self, path):
    """Constructs a `Asset` given a file path."""
    _, basename = os.path.split(path)
    filename, extension = os.path.splitext(basename)

    # Look in the dict of pre-loaded assets before checking the filesystem.
    try:
      contents = self._parent.namescope.assets[path]
    except KeyError:
      # Construct the full path to the asset file, prefixed by the path to the
      # model directory, and by `meshdir` or `texturedir` if appropriate.
      path_parts = []
      if self._parent.namescope.model_dir:
        path_parts.append(self._parent.namescope.model_dir)
      try:
        base_path = self._parent.namescope.get(constants.BASEPATH,
                                               self._path_namespace)
        path_parts.append(base_path)
      except KeyError:
        pass
      path_parts.append(path)
      full_path = os.path.join(*path_parts)  # pylint: disable=no-value-for-parameter
      contents = resources.GetResource(full_path)
    if self._parent.tag == constants.SKIN:
      return SkinAsset(contents=contents, parent=self._parent,
                       extension=extension, prefix=filename)
    else:
      return Asset(contents=contents, extension=extension, prefix=filename)

  def _validate_extension(self, extension):
    if self._parent.tag == constants.MESH:
      if extension.lower() not in _MESH_EXTENSIONS:
        raise ValueError(_INVALID_MESH_EXTENSION.format(extension))

  def get_contents(self):
    """Returns a bytestring representing the contents of the asset."""
    if self._value is None:
      raise RuntimeError('You must assign a value to this attribute before '
                         'querying the contents.')
    return self._value.contents

  def to_xml_string(self, prefix_root=None):
    """Returns the asset filename as it will appear in the generated XML."""
    del prefix_root  # Unused
    if self._value is not None:
      return self._value.get_vfs_filename()
    else:
      return None
