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

"""Mujoco functions to support named indexing.

The Mujoco name structure works as follows:

In mjxmacro.h, each "X" entry denotes a type (a), a field name (b) and a list
of dimension size metadata (c) which may contain both numbers and names, for
example

   X(int,    name_bodyadr, nbody, 1) // or
   X(mjtNum, body_pos,     nbody, 3)
     a       b             c ----->

The second declaration states that the field `body_pos` has type `mjtNum` and
dimension sizes `(nbody, 3)`, i.e. the first axis is indexed by body number.
These and other named dimensions are sized based on the loaded model. This
information is parsed and stored in `mjbindings.sizes`.

In mjmodel.h, the struct mjModel contains an array of element name addresses
for each size name.

   int* name_bodyadr; // body name pointers (nbody x 1)

By iterating over each of these element name address arrays, we first obtain a
mapping from size names to a list of element names.

    {'nbody': ['cart', 'pole'], 'njnt': ['free', 'ball', 'hinge'], ...}

In addition to the element names that are derived from the mjModel struct at
runtime, we also assign hard-coded names to certain dimensions where there is an
established naming convention (e.g. 'x', 'y', 'z' for dimensions that correspond
to Cartesian positions).

For some dimensions, a single element name maps to multiple indices within the
underlying field. For example, a single joint name corresponds to a variable
number of indices within `qpos` that depends on the number of degrees of freedom
associated with that joint type. These are referred to as "ragged" dimensions.

In such cases we determine the size of each named element by examining the
address arrays (e.g. `jnt_qposadr`), and construct a mapping from size name to
element sizes:

    {'nq': [7, 3, 1], 'nv': [6, 3, 1], ...}

Given these two dictionaries, we then create an `Axis` instance for each size
name. These objects have a `convert_key_item` method that handles the conversion
from indexing expressions containing element names to valid numpy indices.
Different implementations of `Axis` are used to handle "ragged" and "non-ragged"
dimensions.

    {'nbody': RegularNamedAxis(names=['cart', 'pole']),
     'nq': RaggedNamedAxis(names=['free', 'ball', 'hinge'], sizes=[7, 4, 1])}

We construct this dictionary once using `make_axis_indexers`.

Finally, for each field we construct a `FieldIndexer` class. A `FieldIndexer`
instance encapsulates a field together with a list of `Axis` instances (one per
dimension), and implements the named indexing logic by calling their respective
`convert_key_item` methods.

Summary of terminology:

* _size name_ or _size_ A dimension size name, e.g. `nbody` or `ngeom`.
* _element name_ or _name_ A named element in a Mujoco model, e.g. 'cart' or
  'pole'.
* _element index_ or _index_ The index of an element name, for a specific size
  name.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import weakref

# Internal dependencies.

from dm_control.mujoco.wrapper import util
from dm_control.mujoco.wrapper.mjbindings import sizes
import numpy as np
import six


# Mapping from {size_name: address_field_name} for ragged dimensions.
_RAGGED_ADDRS = {
    'nq': 'jnt_qposadr',
    'nv': 'jnt_dofadr',
    'nsensordata': 'sensor_adr',
    'nnumericdata': 'numeric_adr',
}

# Names of columns.
_COLUMN_NAMES = {
    'xyz': ['x', 'y', 'z'],
    'quat': ['qw', 'qx', 'qy', 'qz'],
    'mat': ['xx', 'xy', 'xz',
            'yx', 'yy', 'yz',
            'zx', 'zy', 'zz'],
}

# Mapping from keys of _COLUMN_NAMES to sets of field names whose columns are
# addressable using those names.
_COLUMN_ID_TO_FIELDS = {
    'xyz': set([
        'body_pos',
        'body_ipos',
        'body_inertia',
        'jnt_pos',
        'jnt_axis',
        'geom_size',
        'geom_pos',
        'site_size',
        'site_pos',
        'cam_pos',
        'cam_poscom0',
        'cam_pos0',
        'light_pos',
        'light_dir',
        'light_poscom0',
        'light_pos0',
        'light_dir0',
        'mesh_vert',
        'mesh_normal',
        'mocap_pos',
        'xpos',
        'xipos',
        'xanchor',
        'xaxis',
        'geom_xpos',
        'site_xpos',
        'cam_xpos',
        'light_xpos',
        'light_xdir',
        'subtree_com',
        'wrap_xpos',
        'subtree_linvel',
        'subtree_angmom',
    ]),
    'quat': set([
        'body_quat',
        'body_iquat',
        'geom_quat',
        'site_quat',
        'cam_quat',
        'mocap_quat',
        'xquat',
    ]),
    'mat': set([
        'cam_mat0',
        'xmat',
        'ximat',
        'geom_xmat',
        'site_xmat',
        'cam_xmat',
    ])
}


def _get_size_name_to_element_names(model):
  """Returns a dict that maps size names to element names.

  Args:
    model: An instance of `mjbindings.mjModelWrapper`.

  Returns:
    A `dict` mapping from a size name (e.g. `'nbody'`) to a list of element
    names.
  """

  names = model.names[:model.nnames]
  size_name_to_element_names = {}

  for field_name in dir(model):
    if not _is_name_pointer(field_name):
      continue

    # Get addresses of element names in `model.names` array, e.g.
    # field name: `name_nbodyadr` and name_addresses: `[86, 92, 101]`, and skip
    # when there are no elements for this type in the model.
    name_addresses = getattr(model, field_name).ravel()
    if not name_addresses.size:
      continue

    # Get the element names.
    element_names = []
    for start_index in name_addresses:
      name = names[start_index:names.find(b'\0', start_index)]
      element_names.append(util.to_native_string(name))

    # String identifier for the size of the first dimension, e.g. 'nbody'.
    size_name = _get_size_name(field_name)

    size_name_to_element_names[size_name] = element_names

  # Add custom element names for certain columns.
  for size_name, element_names in six.iteritems(_COLUMN_NAMES):
    size_name_to_element_names[size_name] = element_names

  # "Ragged" axes inherit their element names from other "non-ragged" axes.
  # For example, the element names for "nv" axis come from "njnt".
  for size_name, address_field_name in six.iteritems(_RAGGED_ADDRS):
    donor = 'n' + address_field_name.split('_')[0]
    if donor in size_name_to_element_names:
      size_name_to_element_names[size_name] = size_name_to_element_names[donor]

  # Mocap bodies are a special subset of bodies.
  mocap_body_names = [None] * model.nmocap
  for body_id, body_name in enumerate(size_name_to_element_names['nbody']):
    body_mocapid = model.body_mocapid[body_id]
    if body_mocapid != -1:
      mocap_body_names[body_mocapid] = body_name
  assert None not in mocap_body_names
  size_name_to_element_names['nmocap'] = mocap_body_names

  # Arrays with dimension `na` correspond to stateful actuators. MuJoCo's
  # compiler requires that these are always defined after stateless actuators,
  # so we only need the final `na` elements in the list of all actuator names.
  if model.na:
    all_actuator_names = size_name_to_element_names['nu']
    size_name_to_element_names['na'] = all_actuator_names[-model.na:]

  return size_name_to_element_names


def _get_size_name_to_element_sizes(model):
  """Returns a dict that maps size names to element sizes for ragged axes.

  Args:
    model: An instance of `mjbindings.mjModelWrapper`.

  Returns:
    A `dict` mapping from a size name (e.g. `'nv'`) to a numpy array of element
      sizes. Size names corresponding to non-ragged axes are omitted.
  """

  size_name_to_element_sizes = {}

  for size_name, address_field_name in six.iteritems(_RAGGED_ADDRS):
    addresses = getattr(model, address_field_name).ravel()
    total_length = getattr(model, size_name)
    element_sizes = np.diff(np.r_[addresses, total_length])
    size_name_to_element_sizes[size_name] = element_sizes

  return size_name_to_element_sizes


def make_axis_indexers(model):
  """Returns a dict that maps size names to `Axis` indexers.

  Args:
    model: An instance of `mjbindings.MjModelWrapper`.

  Returns:
    A `dict` mapping from a size name (e.g. `'nbody'`) to an `Axis` instance.
  """

  size_name_to_element_names = _get_size_name_to_element_names(model)
  size_name_to_element_sizes = _get_size_name_to_element_sizes(model)

  # Unrecognized size names are treated as unnamed axes.
  axis_indexers = collections.defaultdict(UnnamedAxis)

  for size_name in size_name_to_element_names:
    element_names = size_name_to_element_names[size_name]
    if size_name in _RAGGED_ADDRS:
      element_sizes = size_name_to_element_sizes[size_name]
      indexer = RaggedNamedAxis(element_names, element_sizes)
    else:
      indexer = RegularNamedAxis(element_names)
    axis_indexers[size_name] = indexer

  return axis_indexers


def _is_name_pointer(field_name):
  """Returns True for name pointer field names such as `name_bodyadr`."""
  # Denotes name pointer fields in mjModel.
  prefix, suffix = 'name_', 'adr'
  return field_name.startswith(prefix) and field_name.endswith(suffix)


def _get_size_name(field_name, struct_name='mjmodel'):
  # Look up size name in metadata.
  return sizes.array_sizes[struct_name][field_name][0]


def _validate_key_item(key_item):
  if isinstance(key_item, (list, np.ndarray)):
    for sub in key_item:
      _validate_key_item(sub)  # Recurse into nested arrays and lists.
  elif key_item is Ellipsis:
    raise IndexError('Ellipsis indexing not supported.')
  elif key_item is None:
    raise IndexError('None indexing not supported.')
  elif key_item in (b'', u''):
    raise IndexError('Empty strings are not allowed.')


@six.add_metaclass(abc.ABCMeta)
class Axis(object):
  """Handles the conversion of named indexing expressions into numpy indices."""

  @abc.abstractmethod
  def convert_key_item(self, key_item):
    """Converts a (possibly named) indexing expression to a numpy index."""


class UnnamedAxis(Axis):
  """An object representing an axis where the elements are not named."""

  def convert_key_item(self, key_item):
    """Validate the indexing expression and return it unmodified."""
    _validate_key_item(key_item)
    return key_item


class RegularNamedAxis(Axis):
  """Represents an axis where each named element has a fixed size of 1."""

  def __init__(self, names):
    """Initializes a new `RegularNamedAxis` instance.

    Args:
      names: A list or array of element names.
    """
    self._names = names
    self._names_to_offsets = {name: offset
                              for offset, name in enumerate(names) if name}

  def convert_key_item(self, key_item):
    """Converts a named indexing expression to a numpy-friendly index."""

    _validate_key_item(key_item)

    if isinstance(key_item, six.string_types):
      key_item = self._names_to_offsets[util.to_native_string(key_item)]

    elif isinstance(key_item, (list, np.ndarray)):
      # Cast lists to numpy arrays.
      key_item = np.array(key_item, copy=False)
      original_shape = key_item.shape

      # We assume that either all or none of the items in the array are strings
      # representing names. If there is a mix, we will let NumPy throw an error
      # when trying to index with the returned item.
      if isinstance(key_item.flat[0], six.string_types):
        key_item = np.array([self._names_to_offsets[util.to_native_string(k)]
                             for k in key_item.flat])
        # Ensure the output shape is the same as that of the input.
        key_item.shape = original_shape

    return key_item

  @property
  def names(self):
    """Returns a list of element names."""
    return self._names


class RaggedNamedAxis(Axis):
  """Represents an axis where the named elements may vary in size."""

  def __init__(self, element_names, element_sizes):
    """Initializes a new `RaggedNamedAxis` instance.

    Args:
      element_names: A list or array containing the element names.
      element_sizes: A list or array containing the size of each element.
    """
    names_to_slices = {}
    names_to_indices = {}

    offset = 0
    for name, size in zip(element_names, element_sizes):
      # Don't add unnamed elements to the dicts.
      if name:
        names_to_slices[name] = slice(offset, offset + size)
        names_to_indices[name] = range(offset, offset + size)
      offset += size

    self._names = element_names
    self._sizes = element_sizes
    self._names_to_slices = names_to_slices
    self._names_to_indices = names_to_indices

  def convert_key_item(self, key):
    """Converts a named indexing expression to a numpy-friendly index."""

    _validate_key_item(key)

    if isinstance(key, six.string_types):
      key = self._names_to_slices[util.to_native_string(key)]

    elif isinstance(key, (list, np.ndarray)):
      # We assume that either all or none of the items in the sequence are
      # strings representing names. If there is a mix, we will let NumPy throw
      # an error when trying to index with the returned key.
      if isinstance(key[0], six.string_types):
        new_key = []
        for k in key:
          idx = self._names_to_indices[util.to_native_string(k)]
          if isinstance(idx, int):
            new_key.append(idx)
          else:
            new_key.extend(idx)
        key = new_key

    return key

  @property
  def names(self):
    """Returns a list of element names."""
    return self._names


Axes = collections.namedtuple('Axes', ['row', 'col'])
Axes.__new__.__defaults__ = (None,)  # Default value for optional 'col' field


class FieldIndexer(object):
  """An array-like object providing named access to a field in a MuJoCo struct.

  FieldIndexers expose the same attributes and methods as an `np.ndarray`.

  They may be indexed with strings or lists of strings corresponding to element
  names. They also support standard numpy indexing expressions, with the
  exception of indices containing `Ellipsis` or `None`.
  """

  __slots__ = ('_field_name', '_field', '_axes')

  def __init__(self,
               parent_struct,
               field_name,
               axis_indexers):
    """Initializes a new `FieldIndexer`.

    Args:
      parent_struct: Wrapped ctypes structure, as generated by `mjbindings`.
      field_name: String containing field name in `parent_struct`.
      axis_indexers: A list of `Axis` instances, one per dimension.
    """
    self._field_name = field_name
    self._field = weakref.proxy(getattr(parent_struct, field_name))
    self._axes = Axes(*axis_indexers)

  def __dir__(self):
    # Enables IPython tab completion
    return sorted(set(dir(type(self)) + dir(self._field)))

  def __getattr__(self, name):
    return getattr(self._field, name)

  def _convert_key(self, key):
    """Convert a (possibly named) indexing expression to a valid numpy index."""
    return_tuple = isinstance(key, tuple)
    if not return_tuple:
      key = (key,)
    if len(key) > self._field.ndim:
      raise IndexError('Index tuple has {} elements, but array has only {} '
                       'dimensions.'.format(len(key), self._field.ndim))
    new_key = tuple(axis.convert_key_item(key_item)
                    for axis, key_item in zip(self._axes, key))
    if not return_tuple:
      new_key = new_key[0]
    return new_key

  def __getitem__(self, key):
    """Converts the key to a numeric index and returns the indexed array.

    Args:
      key: Indexing expression.

    Raises:
      IndexError: If an indexing tuple has too many elements, or if it contains
        `Ellipsis`, `None`, or an empty string.

    Returns:
      The indexed array.
    """
    return self._field[self._convert_key(key)]

  def __setitem__(self, key, value):
    """Converts the key and assigns to the indexed array.

    Args:
      key: Indexing expression.
      value: Value to assign.

    Raises:
      IndexError: If an indexing tuple has too many elements, or if it contains
        `Ellipsis`, `None`, or an empty string.
    """
    self._field[self._convert_key(key)] = value

  @property
  def axes(self):
    """A namedtuple containing the row and column indexers for this field."""
    return self._axes

  def __repr__(self):
    """Returns a pretty string representation of the `FieldIndexer`."""

    def get_name_arr_and_len(dim_idx):
      """Returns a string array of element names and the max name length."""
      axis = self._axes[dim_idx]
      size = self._field.shape[dim_idx]
      try:
        name_len = max(len(name) for name in axis.names)
        name_arr = np.zeros(size, dtype='S{}'.format(name_len))
        for name in axis.names:
          if name:
            # Use the `Axis` object to convert the name into a numpy index, then
            # use this index to write into name_arr.
            name_arr[axis.convert_key_item(name)] = name
      except AttributeError:
        name_arr = np.zeros(size, dtype='S0')  # An array of zero-length strings
        name_len = 0
      return name_arr, name_len

    row_name_arr, row_name_len = get_name_arr_and_len(0)
    if self._field.ndim > 1:
      col_name_arr, col_name_len = get_name_arr_and_len(1)
    else:
      col_name_arr, col_name_len = np.zeros(1, dtype='S0'), 0

    idx_len = int(np.log10(max(self._field.shape[0], 1))) + 1

    cls_template = '{class_name:}({field_name:}):'
    col_template = '{padding:}{col_names:}'
    row_template = '{idx:{idx_len:}} {row_name:>{row_name_len:}} {row_vals:}'

    lines = []

    # Write the class name and field name.
    lines.append(cls_template.format(class_name=self.__class__.__name__,
                                     field_name=self._field_name))

    # Write a header line containing the column names (if there are any).
    if col_name_len:
      col_width = max(col_name_len, 9) + 1
      extra_indent = 4
      padding = ' ' * (idx_len + row_name_len + extra_indent)
      col_names = ''.join(
          '{name:<{width:}}'
          .format(name=util.to_native_string(name), width=col_width)
          for name in col_name_arr)
      lines.append(col_template.format(padding=padding, col_names=col_names))

    # Write the row names (if there are any) and the formatted array values.
    if not self._field.shape[0]:
      lines.append('(empty)')
    else:
      for idx, row in enumerate(self._field):
        row_vals = np.array2string(
            np.atleast_1d(row),
            suppress_small=True,
            formatter={'float_kind': '{: < 9.3g}'.format})
        lines.append(row_template.format(
            idx=idx,
            idx_len=idx_len,
            row_name=util.to_native_string(row_name_arr[idx]),
            row_name_len=row_name_len,
            row_vals=row_vals))
    return '\n'.join(lines)


def struct_indexer(struct, struct_name, size_to_axis_indexer):
  """Returns a namedtuple with a `FieldIndexer` for each dynamic array field.

  Usage example

  ```python
  named_data = struct_indexer(mjdata, 'mjdata', size_to_axis_indexer)
  fingertip_xpos = named_data.xpos['fingertip']
  elbow_qvel = named_data.qvel['elbow']
  ```

  Args:
    struct: Wrapped ctypes structure as generated by `mjbindings`.
    struct_name: String containing corresponding Mujoco name of struct.
    size_to_axis_indexer: dict that maps size names to `Axis` instances.

  Returns:
    A `namedtuple` with a field for every dynamically sized array field mapping
      to a `FieldIndexer`.

  Raises:
    ValueError: If `struct_name` is not recognized.
  """
  struct_name = struct_name.lower()
  if struct_name not in sizes.array_sizes:
    raise ValueError('Unrecognized struct name ' + struct_name)

  array_sizes = sizes.array_sizes[struct_name]

  # Used to create the namedtuple.
  field_names = []
  field_indexers = {}

  for field_name in array_sizes:

    # Skip over structured arrays and fields that have sizes but aren't numpy
    # arrays, such as text fields and contacts (b/34805932).
    attr = getattr(struct, field_name)
    if not isinstance(attr, np.ndarray) or attr.dtype.fields:
      continue

    size_names = sizes.array_sizes[struct_name][field_name]

    # Here we override the size name in order to enable named column indexing
    # for certain fields, e.g. 3 becomes "xyz" for field name "xpos".
    for new_col_size, field_set in six.iteritems(_COLUMN_ID_TO_FIELDS):
      if field_name in field_set:
        size_names = (size_names[0], new_col_size)
        break

    axis_indexers = []
    for size_name in size_names:
      axis_indexers.append(size_to_axis_indexer[size_name])

    field_indexers[field_name] = FieldIndexer(
        parent_struct=struct,
        field_name=field_name,
        axis_indexers=axis_indexers)

    field_names.append(field_name)

  struct_indexer_ = collections.namedtuple(struct_name + '_indexer',
                                           field_names)

  return struct_indexer_(**field_indexers)
