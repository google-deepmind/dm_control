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

"""Helpers for MJCF elements to interact with `dm_control.mujoco.Physics`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import weakref

from absl import flags
from absl import logging
from dm_control import mujoco
from dm_control.mjcf import constants
from dm_control.mjcf import debugging
from dm_control.mujoco import wrapper as mujoco_wrapper
from dm_control.mujoco.wrapper.mjbindings import sizes
import numpy as np
import six
from six.moves import range

FLAGS = flags.FLAGS
flags.DEFINE_boolean('pymjcf_log_xml', False,
                     'Whether to log the generated XML on model compilation.')

_XML_PRINT_SHARD_SIZE = 100
_PICKLING_NOT_SUPPORTED = 'Objects of type {type} cannot be pickled.'

_Attribute = collections.namedtuple(
    'Attribute',
    ('name', 'get_named_indexer', 'triggers_dirty', 'disable_on_write'))


def _pymjcf_log_xml():
  """Returns True if the generated XML should be logged on model compilation."""
  if FLAGS.is_parsed():
    return FLAGS.pymjcf_log_xml
  else:
    return FLAGS['pymjcf_log_xml'].default


def _get_attributes(size_names, strip_prefixes):
  """Creates a dict of valid attribute from Mujoco array size name."""
  strip_regex = re.compile(r'\A({})_'.format('|'.join(strip_prefixes)))
  strip = lambda string: strip_regex.sub('', string)
  out = {}
  for name, size in six.iteritems(sizes.array_sizes['mjdata']):
    if size[0] in size_names:
      attrib_name = strip(name)
      named_indexer_getter = (
          lambda physics, name=name: getattr(physics.named.data, name))
      triggers_dirty = attrib_name in constants.MJDATA_TRIGGERS_DIRTY
      out[attrib_name] = _Attribute(
          name=attrib_name,
          get_named_indexer=named_indexer_getter,
          triggers_dirty=triggers_dirty,
          disable_on_write=())
  for name, size in six.iteritems(sizes.array_sizes['mjmodel']):
    if size[0] in size_names:
      attrib_name = strip(name)
      named_indexer_getter = (
          lambda physics, name=name: getattr(physics.named.model, name))
      triggers_dirty = attrib_name not in constants.MJMODEL_DOESNT_TRIGGER_DIRTY
      disable_on_write = constants.MJMODEL_DISABLE_ON_WRITE.get(name, ())
      out[attrib_name] = _Attribute(
          name=attrib_name,
          get_named_indexer=named_indexer_getter,
          triggers_dirty=triggers_dirty,
          disable_on_write=disable_on_write)
  return out


# Fields related to the internal states of actuators (i.e. with a leading
# dimension of 'na') require special treatment.
def _get_actuator_state_fields():
  actuator_state_fields = []
  for sizes_dict in six.itervalues(sizes.array_sizes):
    for field_name, dimensions in six.iteritems(sizes_dict):
      if dimensions[0] == 'na':
        actuator_state_fields.append(field_name)
  return frozenset(actuator_state_fields)

_ACTUATOR_STATE_FIELDS = _get_actuator_state_fields()


def _filter_stateful_actuators(physics, actuator_names):
  """Removes any stateless actuators from the list of actuator names."""
  if physics.model.na:
    # MuJoCo requires that stateful actuators always come after stateless
    # actuators in the model, so we keep actuator names only if their
    # corresponding IDs are >= to the total number of stateless actuators.
    num_stateless_actuators = physics.model.nu - physics.model.na
    return [
        name for name in actuator_names
        if physics.model.name2id(name, 'actuator') >= num_stateless_actuators]
  else:
    return []


_ATTRIBUTES = {
    'actuator': _get_attributes(['na', 'nu'], strip_prefixes=['actuator']),
    'body': _get_attributes(['nbody'], strip_prefixes=['body']),
    'mocap_body': _get_attributes(['nbody', 'nmocap'], strip_prefixes=['body']),
    'camera': _get_attributes(['ncam'], strip_prefixes=['cam']),
    'equality': _get_attributes(['neq'], strip_prefixes=['eq']),
    'geom': _get_attributes(['ngeom'], strip_prefixes=['geom']),
    'hfield': _get_attributes(['nhfield'], strip_prefixes=['hfield']),
    'joint': _get_attributes(['nq', 'nv', 'njnt'],
                             strip_prefixes=['jnt', 'dof']),
    'light': _get_attributes(['nlight'], strip_prefixes=['light']),
    'material': _get_attributes(['nmat'], strip_prefixes=['mat']),
    'mesh': _get_attributes(['nmesh'], strip_prefixes=['mesh']),
    'numeric': _get_attributes(['nnumeric', 'nnumericdata'],
                               strip_prefixes=['numeric']),
    'sensor': _get_attributes(['nsensor', 'nsensordata'],
                              strip_prefixes=['sensor']),
    'site': _get_attributes(['nsite'], strip_prefixes=['site']),
    'tendon': _get_attributes(['ntendon'], strip_prefixes=['tendon', 'ten']),
    'text': _get_attributes(['ntext', 'ntextdata'], strip_prefixes=['text']),
    'texture': _get_attributes(['ntex'], strip_prefixes=['tex']),
}


def names_from_elements(mjcf_elements):
  """Returns `namespace` and `named_index` for `mjcf_elements`.

  Args:
    mjcf_elements: Either an `mjcf.Element`, or an iterable of `mjcf.Element`
        of the same kind.

  Returns:
    A tuple of `(namespace, named_indices)` where
      -`namespace` is the Mujoco element type (eg: 'geom', 'body', etc.)
      -`named_indices` are the names of `mjcf_elements`, either as a single
        string or an iterable of strings depending on whether `mjcf_elements`
        was an `mjcf.Element` or an iterable of `mjcf_Element`s.

  Raises:
      ValueError: If `mjcf_elements` cannot be bound to this Physics.
  """
  if isinstance(mjcf_elements, collections.Iterable):
    elements_tuple = tuple(mjcf_elements)
    if elements_tuple:
      namespace = _get_namespace(elements_tuple[0])
    else:
      return None, None
    for element in elements_tuple:
      element_namespace = _get_namespace(element)
      if element_namespace != namespace:
        raise ValueError('Cannot bind to a collection containing multiple '
                         'element types ({!r} != {!r}).'
                         .format(element_namespace, namespace))
    named_index = [element.full_identifier for element in elements_tuple]
  else:
    namespace = _get_namespace(mjcf_elements)
    named_index = mjcf_elements.full_identifier

  return namespace, named_index


class SynchronizingArrayWrapper(np.ndarray):
  """A non-contiguous view of an ndarray that synchronizes with the original.

  Note: this class should not be instantiated directly.
  """
  __slots__ = (
      '_backing_array',
      '_backing_index',
      '_physics',
      '_triggers_dirty',
      '_disable_on_write',
  )

  def __new__(cls,
              backing_array,
              backing_index,
              physics,
              triggers_dirty,
              disable_on_write):
    obj = backing_array[backing_index].view(SynchronizingArrayWrapper)
    # pylint: disable=protected-access
    obj._backing_array = backing_array
    obj._backing_index = backing_index
    obj._physics = physics
    obj._triggers_dirty = triggers_dirty
    obj._disable_on_write = disable_on_write
    # pylint: enable=protected-access
    return obj

  def _synchronize_from_backing_array(self):
    if self._physics.is_dirty and not self._triggers_dirty:
      self._physics.forward()
    super(SynchronizingArrayWrapper, self).__setitem__(
        slice(None, None, None), self._backing_array[self._backing_index])

  def copy(self, order='C'):
    return np.copy(self, order=order)

  def __copy__(self):
    return self.copy()

  def __deepcopy__(self, memo):
    return self.copy()

  def __reduce__(self):
    raise NotImplementedError(_PICKLING_NOT_SUPPORTED.format(type=type(self)))

  def __setitem__(self, index, value):
    if self._physics.is_dirty and not self._triggers_dirty:
      self._physics.forward()
    super(SynchronizingArrayWrapper, self).__setitem__(index, value)
    if isinstance(self._backing_index, collections.Iterable):
      if isinstance(index, tuple):
        resolved_index = (self._backing_index[index[0]],) + index[1:]
      else:
        resolved_index = self._backing_index[index]
      self._backing_array[resolved_index] = value

    for backing_array, backing_index in self._disable_on_write:
      if isinstance(index, collections.Iterable):
        # We only need the row component of the index.
        if isinstance(index, tuple):
          resolved_index = backing_index[index[0]]
        else:
          resolved_index = backing_index[index]
      else:
        # If it is only an index into the columns of the backing array then we
        # just discard it and use the backing index.
        resolved_index = backing_index
      backing_array[resolved_index] = 0

    if self._triggers_dirty:
      self._physics.mark_as_dirty()

  def __setslice__(self, start, stop, value):
    self.__setitem__(slice(start, stop, None), value)


class Binding(object):
  """Binding between a mujoco.Physics and an mjcf.Element or a list of Elements.

  This object should normally be created by calling `physics.bind(element)`
  where `physics` is an instance of `mjcf.Physics`. See docstring for that
  function for details.
  """
  __slots__ = (
      '_attributes',
      '_physics',
      '_namespace',
      '_named_index',
      '_named_indexers',
      '_getattr_cache',
      '_array_index_cache',
  )

  def __init__(self, physics, namespace, named_index):
    try:
      self._attributes = _ATTRIBUTES[namespace]
    except KeyError:
      raise ValueError('elements of type {!r} cannot be bound to physics'
                       .format(namespace))
    self._physics = physics
    self._namespace = namespace
    self._named_index = named_index
    self._named_indexers = {}
    self._getattr_cache = {}
    self._array_index_cache = {}

  def __dir__(self):
    return sorted(set(dir(type(self))).union(self._attributes))

  def _get_cached_named_indexer(self, name):
    named_indexer = self._named_indexers.get(name)
    if named_indexer is None:
      try:
        named_indexer = self._attributes[name].get_named_indexer(self._physics)
        self._named_indexers[name] = named_indexer
      except KeyError:
        raise AttributeError('bound element <{}> does not have attribute {!r}'
                             .format(self._namespace, name))
    return named_indexer

  def _get_cached_array_and_index(self, name):
    """Returns `(array, index)` for a given field name."""
    named_indexer = self._get_cached_named_indexer(name)
    array = named_indexer._field  # pylint: disable=protected-access
    try:
      index = self._array_index_cache[name]
    except KeyError:
      # If we are indexing into a field relating to actuator internal states
      # then we must first remove the names of any stateless actuators.
      if name in _ACTUATOR_STATE_FIELDS:
        named_index = _filter_stateful_actuators(
            self._physics, self._named_index)
      else:
        named_index = self._named_index
      index = named_indexer._convert_key(named_index)  # pylint: disable=protected-access
      self._array_index_cache[name] = index
    return array, index

  @property
  def element_id(self):
    if isinstance(self._named_index, list):
      return np.array([self._physics.model.name2id(item_name, self._namespace)
                       for item_name in self._named_index])
    else:
      return self._physics.model.name2id(self._named_index, self._namespace)

  def __getattr__(self, name):
    if name in Binding.__slots__:
      return super(Binding, self).__getattr__(name)
    else:
      try:
        out = self._getattr_cache[name]
        out._synchronize_from_backing_array()  # pylint: disable=protected-access
      except KeyError:
        array, index = self._get_cached_array_and_index(name)
        triggers_dirty = self._attributes[name].triggers_dirty

        # A list of (array, index) tuples specifying other addresses that need
        # to be zeroed out when this array attribute is written to.
        disable_on_write = []
        for name_to_disable in self._attributes[name].disable_on_write:
          array_to_disable, index_to_disable = self._get_cached_array_and_index(
              name_to_disable)
          # Ensure that the result of indexing is a `SynchronizingArrayWrapper`
          # rather than a scalar, otherwise we won't be able to write into it.
          if array_to_disable.ndim == 1:
            if isinstance(index_to_disable, np.ndarray):
              index_to_disable = index_to_disable.copy().reshape(-1, 1)
            else:
              index_to_disable = [index_to_disable]
          disable_on_write.append((array_to_disable, index_to_disable))

        if self._physics.is_dirty and not triggers_dirty:
          self._physics.forward()
        if isinstance(index, int) and array.ndim == 1:
          # Case where indexing results in a scalar.
          out = array[index]
        else:
          # Case where indexing results in an array.
          out = SynchronizingArrayWrapper(
              backing_array=array,
              backing_index=index,
              physics=self._physics,
              triggers_dirty=triggers_dirty,
              disable_on_write=disable_on_write)
          self._getattr_cache[name] = out
      return out

  def __setattr__(self, name, value):
    if name in Binding.__slots__:
      super(Binding, self).__setattr__(name, value)
    else:
      if self._physics.is_dirty and not self._attributes[name].triggers_dirty:
        self._physics.forward()
      array, index = self._get_cached_array_and_index(name)
      array[index] = value
      for name_to_disable in self._attributes[name].disable_on_write:
        disable_array, disable_index = self._get_cached_array_and_index(
            name_to_disable)
        disable_array[disable_index] = 0
      if self._attributes[name].triggers_dirty:
        self._physics.mark_as_dirty()

  def _get_name_and_indexer_and_expression(self, index):
    """Returns (name, indexer, expression) for a given input to __getitem__."""
    if isinstance(index, tuple):
      name, column_index = index
      try:
        # If named_index and column_index are both array-like, reshape
        # named_index to (n, 1) so that it can be broadcasted against
        # column_index.
        expression = np.ix_(self._named_index, column_index)
      except ValueError:
        expression = (self._named_index, column_index)
    else:
      name = index
      expression = self._named_index
    indexer = self._get_cached_named_indexer(name)
    return name, indexer, expression

  def __getitem__(self, index):
    name, indexer, expression = self._get_name_and_indexer_and_expression(index)
    if self._physics.is_dirty and not self._attributes[name].triggers_dirty:
      self._physics.forward()
    return indexer[expression]

  def __setitem__(self, index, value):
    name, indexer, expression = self._get_name_and_indexer_and_expression(index)
    if self._physics.is_dirty and not self._attributes[name].triggers_dirty:
      self._physics.forward()
    indexer[expression] = value
    if self._attributes[name].triggers_dirty:
      self._physics.mark_as_dirty()


class _EmptyBinding(object):
  """The result of binding no `mjcf.Elements` to an `mjcf.Physics` instance."""

  __slots__ = ('_arr',)

  def __init__(self):
    self._arr = np.empty((0))

  def __getattr__(self, name):
    return self._arr

  def __setattr__(self, name, value):
    if name in self.__slots__:
      super(_EmptyBinding, self).__setattr__(name, value)
    else:
      raise ValueError('Cannot assign a value to an empty binding.')

_EMPTY_BINDING = _EmptyBinding()


def _log_xml(xml_string):
  xml_lines = xml_string.split('\n')
  for start_line in range(0, len(xml_lines), _XML_PRINT_SHARD_SIZE):
    end_line = min(start_line + _XML_PRINT_SHARD_SIZE, len(xml_lines))
    template = 'XML lines %d-%d of %d:\n%s'
    if start_line == 0:
      template = 'PyMJCF: compiling generated XML:\n' + template
    logging.info(template, start_line + 1, end_line, len(xml_lines),
                 '\n'.join(xml_lines[start_line:end_line]))


class Physics(mujoco.Physics):
  """A specialized `mujoco.Physics` that supports binding to MJCF elements."""

  @classmethod
  def from_mjcf_model(cls, mjcf_model):
    """Constructs a new `mjcf.Physics` from an `mjcf.RootElement`.

    Args:
      mjcf_model: An `mjcf.RootElement` instance.

    Returns:
      A new `mjcf.Physics` instance.
    """
    debug_context = debugging.DebugContext()
    xml_string = mjcf_model.to_xml_string(debug_context=debug_context)
    if _pymjcf_log_xml():
      if debug_context.debug_mode and debug_context.default_dump_dir:
        logging.info('Full debug info is dumped to disk at %s',
                     debug_context.default_dump_dir)
        debug_context.dump_full_debug_info_to_disk()
      else:
        logging.info('Full debug info is not yet dumped to disk. If this is '
                     'needed, pass all three flags: --pymjcf_log_xml '
                     '--pymjcf_debug --pymjcf_debug_full_dump_dir=/path/dir/')
      _log_xml(xml_string)
    assets = mjcf_model.get_assets()
    try:
      return cls.from_xml_string(xml_string=xml_string, assets=assets)
    except mujoco_wrapper.Error:
      debug_context.process_and_raise_last_exception()

  def reload_from_mjcf_model(self, mjcf_model):
    """Reloads this `mjcf.Physics` from an `mjcf.RootElement`.

    After calling this method, the state of this `Physics` instance is the same
    as a new `Physics` instance created with the `from_mjcf_model` named
    constructor.

    Args:
      mjcf_model: An `mjcf.RootElement` instance.
    """
    debug_context = debugging.DebugContext()
    xml_string = mjcf_model.to_xml_string(debug_context=debug_context)
    if _pymjcf_log_xml():
      _log_xml(xml_string)
    assets = mjcf_model.get_assets()
    try:
      self.reload_from_xml_string(xml_string=xml_string, assets=assets)
    except mujoco_wrapper.Error:
      debug_context.process_and_raise_last_exception()

  def _reload_from_data(self, data):
    """Initializes a new or existing `Physics` instance from a `core.MjData`.

    Assigns all attributes and sets up rendering contexts and named indexing.

    The default constructor as well as the other `reload_from` methods should
    delegate to this method.

    Args:
      data: Instance of `core.MjData`.
    """
    super(Physics, self)._reload_from_data(data)
    self._bindings = {}
    self._bindings[()] = _EMPTY_BINDING
    self._dirty = False

  @property
  def is_dirty(self):
    """Whether this physics' internal state needs to be recalculated."""
    return self._dirty

  def mark_as_dirty(self):
    """Marks this physics as dirty, thus requiring recalculation."""
    self._dirty = True

  def forward(self):
    """Recomputes the forward dynamics without advancing the simulation."""
    super(Physics, self).forward()
    self._dirty = False

  def bind(self, mjcf_elements):
    """Creates a binding between this `Physics` instance and `mjcf.Element`s.

    The binding allows for easier interaction with the `Physics` data structures
    related to an MJCF element. For example, in order to access the Cartesian
    position of a geom, we can use:

    ```python
    physics.bind(geom_element).pos
    ```

    instead of the more cumbersome:

    ```python
    physics.named.model.geom_pos[geom_element.full_identifier]
    ```

    Note that the binding takes into account the type of element. This allows us
    to remove prefixes from certain common attributes in order to unify access.
    For example, we can use:

    ```python
    physics.bind(geom_element).pos = [1, 2, 3]
    physics.bind(site_element).pos = [4, 5, 6]
    ```

    instead of:

    ```python
    physics.named.model.geom_pos[geom_element.full_identifier] = [1, 2, 3]
    physics.named.model.site_pos[site_element.full_identifier] = [4, 5, 6]
    ```

    This in turn allows for the creation of common algorithms that can operate
    across a wide range of element type.

    When attribute values are modified through the binding, future queries of
    derived values are automatically recalculated if necessary. For example,
    if a joint's `qpos` is modified and a site's `xpos` is later read, the value
    of the `xpos` is updated according to the new joint configuration. This is
    done lazily when an updated value is required, so repeated value
    modifications do not incur a performance penalty.

    It is also possible to bind a sequence containing one or more elements,
    provided they are all of the same type. In this case the binding exposes
    `SynchronizingArrayWrapper`s, which are array-like objects that provide
    writeable views onto the corresponding memory addresses in MuJoCo. Writing
    into a `SynchronizingArrayWrapper` causes the underlying values in MuJoCo
    to be updated, and if necessary causes derived values to be recalculated.
    Note that in order to trigger recalculation it is necessary to reference a
    derived attribute of a binding.

    ```python
    bound_joints = physics.bind([joint1, joint2])
    bound_bodies = physics.bind([body1, body2])
    # `qpos_view` and `xpos_view` are `SynchronizingArrayWrapper`s providing
    # views onto `physics.data.qpos` and `physics.data.xpos` respectively.
    qpos_view = bound_joints.qpos
    xpos_view = bound_bodies.xpos
    # This updates the corresponding values in `physics.data.qpos`, and marks
    # derived values (such as `physics.data.xpos`) as needing recalculation.
    qpos_view[0] += 1.
    # Note: at this point `xpos_view` still contains the old values, since we
    # need to actually read the value of a derived attribute in order to
    # trigger recalculation.
    another_xpos_view = bound_bodies.xpos  # Triggers recalculation of `xpos`.
    # Now both `xpos_view` and `another_xpos_view` will contain the updated
    # values.
    ```

    Note that `SynchronizingArrayWrapper`s cannot be pickled. We also do not
    recommend holding references to them - instead hold a reference to the
    binding object, or call `physics.bind` again.

    Bindings also support numpy-style square bracket indexing. The first element
    in the indexing expression should be an attribute name, and the second
    element (if present) is used to index into the columns of the underlying
    array. Named indexing into columns is also allowed, provided that the
    corresponding field in `physics.named` supports it.

    ```python
    physics.bind([geom1, geom2])['pos'] = [[1, 2, 3], [4, 5, 6]]
    physics.bind([geom1, geom2])['pos', ['x', 'z']] = [[1, 3], [4, 6]]
    ```

    Args:
      mjcf_elements: Either an `mjcf.Element`, or an iterable of `mjcf.Element`
        of the same kind.

    Returns:
      A binding between this Physics instance an `mjcf_elements`, as described
      above.

    Raises:
      ValueError: If `mjcf_elements` cannot be bound to this Physics.
    """
    if mjcf_elements is None:
      return None

    # To reduce overhead from processing MJCF elements and making new bindings,
    # we cache and reuse existing Binding objects. The cheapest version of
    # caching is when we can use `mjcf_elements` as key directly. However, this
    # is not always possible since `mjcf_elements` may contain weak references.
    # In this case, we fallback to using the elements' namespace and identifiers
    # as cache keys instead.

    # Checking for iterability in this way is cheaper than using `isinstance`.
    try:
      cache_key = tuple(mjcf_elements)
    except TypeError:
      # `mjcf_elements` is not iterable.
      cache_key = mjcf_elements

    needs_new_binding = False
    try:
      binding = self._bindings[cache_key]
    except KeyError:
      # This means `mjcf_elements` is hashable, so we use it as cache key.
      namespace, named_index = names_from_elements(mjcf_elements)
      needs_new_binding = True
    except TypeError:
      # This means `mjcf_elements` is unhashable, fallback to caching by name.
      namespace, named_index = names_from_elements(mjcf_elements)

      # Checking for iterability in this way is cheaper than using `isinstance`.
      try:
        cache_key = (namespace, tuple(named_index))
      except TypeError:
        # `named_index` is not iterable.
        cache_key = (namespace, named_index)

      try:
        binding = self._bindings[cache_key]
      except KeyError:
        needs_new_binding = True

    if needs_new_binding:
      binding = Binding(weakref.proxy(self), namespace, named_index)
      self._bindings[cache_key] = binding

    return binding


def _get_namespace(element):
  """Returns the element namespace string."""
  # The worldbody is treated as a member of the `body` namespace.
  if element.tag == 'worldbody':
    namespace = 'body'
  else:
    namespace = element.spec.namespace.split(constants.NAMESPACE_SEPARATOR)[0]
    # Mocap bodies have distinct attributes so we use a dummy namespace for
    # them.
    if namespace == 'body' and element.mocap == 'true':
      namespace = 'mocap_body'
  return namespace
