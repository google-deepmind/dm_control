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

"""Classes to represent MJCF elements in the object model."""

import collections
import copy
import os
import sys

from dm_control.mjcf import attribute as attribute_types
from dm_control.mjcf import base
from dm_control.mjcf import constants
from dm_control.mjcf import copier
from dm_control.mjcf import debugging
from dm_control.mjcf import namescope
from dm_control.mjcf import schema
from dm_control.mujoco.wrapper import util
from lxml import etree
import numpy as np


_raw_property = property  # pylint: disable=invalid-name


_CONFLICT_BEHAVIOR_FUNC = {'min': min, 'max': max}


def property(method):  # pylint: disable=redefined-builtin
  """Modifies `@property` to keep track of any `AttributeError` raised.

  Our `Element` implementations overrides the `__getattr__` method. This does
  not interact well with `@property`: if a `property`'s code is buggy so as to
  raise an `AttributeError`, then Python would silently discard it and redirect
  to our `__getattr__` instead, leading to an uninformative stack trace. This
  makes it very difficult to debug issues that involve properties.

  To remedy this, we modify `@property` within this module to store any
  `AttributeError` raised within the respective `Element` object. Then, in our
  `__getattr__` logic, we could re-raise it to preserve the original stack
  trace.

  The reason that this is not implemented as a different decorator is that we
  could accidentally use @property on a new method. This would work fine until
  someone triggers a subtle bug. This is when a proper trace would be most
  useful, but we would still end up with a strange undebuggable stack trace
  anyway.

  Note that at the end of this module, we have a `del property` to prevent this
  override from being broadcasted externally.

  Args:
    method: The method that is being decorated.

  Returns:
    A `property` corresponding to the decorated method.
  """
  def _mjcf_property(self):
    try:
      return method(self)
    except:
      _, err, tb = sys.exc_info()
      err_with_next_tb = err.with_traceback(tb.tb_next)
      if isinstance(err, AttributeError):
        self._last_attribute_error = err_with_next_tb  # pylint: disable=protected-access
      raise err_with_next_tb
  return _raw_property(_mjcf_property)


def _make_element(spec, parent, attributes=None):
  """Helper function to generate the right kind of Element given a spec."""
  if (spec.name == constants.WORLDBODY
      or (spec.name == constants.SITE
          and (parent.tag == constants.BODY
               or parent.tag == constants.WORLDBODY))):
    return _AttachableElement(spec, parent, attributes)
  elif isinstance(parent, _AttachmentFrame):
    return _AttachmentFrameChild(spec, parent, attributes)
  elif spec.name == constants.DEFAULT:
    return _DefaultElement(spec, parent, attributes)
  elif spec.name == constants.ACTUATOR:
    return _ActuatorElement(spec, parent, attributes)
  else:
    return _ElementImpl(spec, parent, attributes)


_DEFAULT_NAME_FROM_FILENAME = frozenset(['mesh', 'hfield', 'texture'])


class _ElementImpl(base.Element):
  """Actual implementation of a generic MJCF element object."""
  __slots__ = ['__weakref__', '_spec', '_parent', '_attributes', '_children',
               '_own_attributes', '_attachments', '_is_removed', '_init_stack',
               '_is_worldbody', '_cached_namescope', '_cached_root',
               '_cached_full_identifier', '_cached_revision',
               '_last_attribute_error']

  def __init__(self, spec, parent, attributes=None):
    attributes = attributes or {}

    # For certain `asset` elements the `name` attribute can be omitted, in which
    # case the name will be the filename without the leading path and extension.
    # See http://www.mujoco.org/book/XMLreference.html#asset.
    if ('name' not in attributes
        and 'file' in attributes
        and spec.name in _DEFAULT_NAME_FROM_FILENAME):
      _, filename = os.path.split(attributes['file'])
      basename, _ = os.path.splitext(filename)
      attributes['name'] = basename

    self._spec = spec
    self._parent = parent
    self._attributes = collections.OrderedDict()
    self._own_attributes = None
    self._children = []
    self._attachments = collections.OrderedDict()
    self._is_removed = False
    self._is_worldbody = (self.tag == 'worldbody')

    if self._parent:
      self._cached_namescope = self._parent.namescope
      self._cached_root = self._parent.root
    self._cached_full_identifier = ''
    self._cached_revision = -1

    self._last_attribute_error = None

    if debugging.debug_mode():
      self._init_stack = debugging.get_current_stack_trace()

    with debugging.freeze_current_stack_trace():
      for child_spec in self._spec.children.values():
        if not (child_spec.repeated or child_spec.on_demand):
          self._children.append(_make_element(spec=child_spec, parent=self))

      if constants.DCLASS in attributes:
        attributes[constants.CLASS] = attributes[constants.DCLASS]
        del attributes[constants.DCLASS]

      for attribute_name in attributes.keys():
        self._check_valid_attribute(attribute_name)

      for attribute_spec in self._spec.attributes.values():
        value = None
        # Some Reference attributes refer to a namespace that is specified
        # via another attribute. We therefore have to set things up for
        # the additional indirection.
        if attribute_spec.type is attribute_types.Reference:
          reference_namespace = (
              attribute_spec.other_kwargs['reference_namespace'])
          if reference_namespace.startswith(
              constants.INDIRECT_REFERENCE_NAMESPACE_PREFIX):
            attribute_spec = copy.deepcopy(attribute_spec)
            namespace_attrib_name = reference_namespace[
                len(constants.INDIRECT_REFERENCE_NAMESPACE_PREFIX):]
            attribute_spec.other_kwargs['reference_namespace'] = (
                self._attributes[namespace_attrib_name])
        if attribute_spec.name in attributes:
          value = attributes[attribute_spec.name]
        try:
          self._attributes[attribute_spec.name] = attribute_spec.type(
              name=attribute_spec.name,
              required=attribute_spec.required,
              conflict_allowed=attribute_spec.conflict_allowed,
              conflict_behavior=attribute_spec.conflict_behavior,
              parent=self, value=value, **attribute_spec.other_kwargs)
        except:
          # On failure, clear attributes already created
          for attribute_obj in self._attributes.values():
            attribute_obj._force_clear()  # pylint: disable=protected-access
          # Then raise a meaningful error
          err_type, err, tb = sys.exc_info()
          raise err_type(
              f'during initialization of attribute {attribute_spec.name!r} of '
              f'element <{self._spec.name}>: {err}').with_traceback(tb)

  def get_init_stack(self):
    """Gets the stack trace where this element was first initialized."""
    if debugging.debug_mode():
      return self._init_stack

  def get_last_modified_stacks_for_all_attributes(self):
    """Gets a dict of stack traces where each attribute was last modified."""
    return collections.OrderedDict(
        [(name, self._attributes[name].last_modified_stack)
         for name in self._spec.attributes])

  def is_same_as(self, other):
    """Checks whether another element is semantically equivalent to this one.

    Two elements are considered equivalent if they have the same
    specification (i.e. same tag appearing in the same context), the same
    attribute values, and all of their children are equivalent. The ordering
    of non-repeated children is not important for this comparison, while
    the ordering of repeated children are important only amongst the same
    type* of children. In other words, for two bodies to be considered
    equivalent, their child sites must appear in the same order, and their
    child geoms must appear in the same order, but permutations between sites
    and geoms are disregarded. (The only exception is in tendon definition,
    where strict ordering of all children is necessary for equivalence.)

    *Note that the notion of "same type" in this function is very loose:
    for example different actuator element subtypes are treated as separate
    types when children ordering is considered. Therefore, two <actuator>
    elements might be considered equivalent even though they result in different
    orderings of `mjData.ctrl` when compiled. As it stands, this function
    is designed primarily as a testing aid and should not be used to guarantee
    that models are actually identical.

    Args:
      other: An `mjcf.Element`

    Returns:
      `True` if `other` element is semantically equivalent to this one.
    """
    if other is None or other.spec != self._spec:
      return False

    for attribute_name in self._spec.attributes.keys():
      attribute = self._attributes[attribute_name]
      other_attribute = getattr(other, attribute_name)
      if isinstance(attribute.value, base.Element):
        if attribute.value.full_identifier != other_attribute.full_identifier:
          return False
      elif not np.all(attribute.value == other_attribute):
        return False

    if (self._parent and
        self._parent.tag == constants.TENDON and
        self._parent.parent == self.root):
      return self._tendon_has_same_children_as(other)
    else:
      return self._has_same_children_as(other)

  def _has_same_children_as(self, other):
    """Helper function to check whether another element has the same children.

    See docstring for `is_same_as` for explanation about the treatment of
    children ordering.

    Args:
      other: An `mjcf.Element`

    Returns:
      A boolean
    """
    for child_name, child_spec in self._spec.children.items():
      child = self.get_children(child_name)
      other_child = getattr(other, child_name)
      if not child_spec.repeated:
        if ((child is None and other_child is not None) or
            (child is not None and not child.is_same_as(other_child))):
          return False
      else:
        if len(child) != len(other_child):
          return False
        else:
          for grandchild, other_grandchild in zip(child, other_child):
            if not grandchild.is_same_as(other_grandchild):
              return False
    return True

  def _tendon_has_same_children_as(self, other):
    return all(child.is_same_as(other_child)
               for child, other_child
               in zip(self.all_children(), other.all_children()))

  def _alias_attributes_dict(self, other):
    if self._own_attributes is None:
      self._own_attributes = self._attributes
    self._attributes = other

  def _restore_attributes_dict(self):
    if self._own_attributes is not None:
      for attribute_name, attribute in self._attributes.items():
        self._own_attributes[attribute_name].value = attribute.value
      self._attributes = self._own_attributes
      self._own_attributes = None

  @property
  def tag(self):
    return self._spec.name

  @property
  def spec(self):
    return self._spec

  @property
  def parent(self):
    return self._parent

  @property
  def namescope(self):
    return self._cached_namescope

  @property
  def root(self):
    return self._cached_root

  def prefixed_identifier(self, prefix_root):
    if not self._spec.identifier and not self._is_worldbody:
      return None
    elif self._is_worldbody:
      prefix = self.namescope.full_prefix(prefix_root=prefix_root)
      return prefix or 'world'
    else:
      full_identifier = (
          self._attributes[self._spec.identifier].to_xml_string(
              prefix_root=prefix_root))
      if full_identifier:
        return full_identifier
      else:
        prefix = self.namescope.full_prefix(prefix_root=prefix_root)
        prefix = prefix or constants.PREFIX_SEPARATOR
        return prefix + self._default_identifier

  @property
  def full_identifier(self):
    """Fully-qualified identifier used for this element in the generated XML."""
    if self.namescope.revision > self._cached_revision:
      self._cached_full_identifier = self.prefixed_identifier(
          prefix_root=self.namescope.root)
      self._cached_revision = self.namescope.revision
    return self._cached_full_identifier

  @property
  def _default_identifier(self):
    """The default identifier used if this element is not named by the user."""
    if not self._spec.identifier:
      return None
    else:
      siblings = self.root.find_all(self._spec.namespace,
                                    exclude_attachments=True)
      return '{separator}unnamed_{namespace}_{index}'.format(
          separator=constants.PREFIX_SEPARATOR,
          namespace=self._spec.namespace,
          index=siblings.index(self))

  def __dir__(self):
    out_dir = set()
    classes = (type(self),)
    while classes:
      super_classes = set()
      for klass in classes:
        out_dir.update(klass.__dict__)
        super_classes.update(klass.__bases__)
      classes = super_classes
    out_dir.update(self._spec.children)
    out_dir.update(self._spec.attributes)
    if constants.CLASS in out_dir:
      out_dir.remove(constants.CLASS)
      out_dir.add(constants.DCLASS)
    return sorted(out_dir)

  def find(self, namespace, identifier):
    """Finds an element with a particular identifier.

    This function allows the direct access to an arbitrarily deeply nested
    child element by name, without the need to manually traverse through the
    object tree. The `namespace` argument specifies the kind of element to
    find. In most cases, this corresponds to the element's XML tag name.
    However, if an element has multiple specialized tags, then the namespace
    corresponds to the tag name of the most general element of that kind.
    For example, `namespace='joint'` would search for `<joint>` and
    `<freejoint>`, while `namespace='actuator'` would search for `<general>`,
    `<motor>`, `<position>`, `<velocity>`, and `<cylinder>`.

    Args:
      namespace: A string specifying the namespace being searched. See the
        docstring above for explanation.
      identifier: The identifier string of the desired element.

    Returns:
      An `mjcf.Element` object, or `None` if an element with the specified
      identifier is not found.

    Raises:
      ValueError: if either `namespace` or `identifier` is not a string, or if
        `namespace` is not a valid namespace.
    """
    if not isinstance(namespace, str):
      raise ValueError(
          '`namespace` should be a string: got {!r}'.format(namespace))
    if not isinstance(identifier, str):
      raise ValueError(
          '`identifier` should be a string: got {!r}'.format(identifier))
    if namespace not in schema.FINDABLE_NAMESPACES:
      raise ValueError('{!r} is not a valid namespace. Available: {}.'.format(
          namespace, schema.FINDABLE_NAMESPACES))
    if constants.PREFIX_SEPARATOR in identifier:
      scope_name = identifier.split(constants.PREFIX_SEPARATOR)[0]
      try:
        attachment = self.namescope.get('attached_model', scope_name)
        found_element = attachment.find(
            namespace, identifier[(len(scope_name) + 1):])
      except (KeyError, ValueError):
        found_element = None
    else:
      try:
        found_element = self.namescope.get(namespace, identifier)
      except KeyError:
        found_element = None
      if found_element and self._parent:
        next_parent = found_element.parent
        while next_parent and next_parent != self:
          next_parent = next_parent.parent
        if not next_parent:
          found_element = None
    return found_element

  def find_all(self, namespace,
               immediate_children_only=False, exclude_attachments=False):
    """Finds all elements of a particular kind.

    The `namespace` argument specifies the kind of element to
    find. In most cases, this corresponds to the element's XML tag name.
    However, if an element has multiple specialized tags, then the namespace
    corresponds to the tag name of the most general element of that kind.
    For example, `namespace='joint'` would search for `<joint>` and
    `<freejoint>`, while `namespace='actuator'` would search for `<general>`,
    `<motor>`, `<position>`, `<velocity>`, and `<cylinder>`.

    Args:
      namespace: A string specifying the namespace being searched. See the
        docstring above for explanation.
      immediate_children_only: (optional) A boolean, if `True` then only
        the immediate children of this element are returned.
      exclude_attachments: (optional) A boolean, if `True` then elements
        belonging to attached models are excluded.

    Returns:
      A list of `mjcf.Element`.

    Raises:
      ValueError: if `namespace` is not a valid namespace.
    """
    if namespace not in schema.FINDABLE_NAMESPACES:
      raise ValueError('{!r} is not a valid namespace. Available: {}'.format(
          namespace, schema.FINDABLE_NAMESPACES))
    out = []
    children = self._children if exclude_attachments else self.all_children()
    for child in children:
      if (namespace == child.spec.namespace or
          # Direct children of attachment frames have custom namespaces of the
          # form "joint@attachment_frame_<id>".
          child.spec.namespace and child.spec.namespace.startswith(
              namespace + constants.NAMESPACE_SEPARATOR) or
          # Attachment frames are considered part of the "body" namespace.
          namespace == constants.BODY and isinstance(child, _AttachmentFrame)):
        out.append(child)
      if not immediate_children_only:
        out.extend(child.find_all(namespace,
                                  exclude_attachments=exclude_attachments))
    return out

  def enter_scope(self, scope_identifier):
    """Finds the root element of the given scope and returns it.

    This function allows the access to a nested scope that is a child of this
    element. The `scope_identifier` argument specifies the path to the child
    scope element.

    Args:
      scope_identifier: The path of the desired scope element.

    Returns:
      An `mjcf.Element` object, or `None` if a scope element with the
      specified path is not found.
    """
    if constants.PREFIX_SEPARATOR in scope_identifier:
      scope_name = scope_identifier.split(constants.PREFIX_SEPARATOR)[0]
      try:
        attachment = self.namescope.get('attached_model', scope_name)
      except KeyError:
        return None

      scope_suffix = scope_identifier[(len(scope_name) + 1):]
      if scope_suffix:
        return attachment.enter_scope(scope_suffix)
      else:
        return attachment
    else:
      try:
        return self.namescope.get('attached_model', scope_identifier)
      except KeyError:
        return None

  def _check_valid_attribute(self, attribute_name):
    if attribute_name not in self._spec.attributes:
      raise AttributeError(
          '{!r} is not a valid attribute for <{}>'.format(
              attribute_name, self._spec.name))

  def _get_attribute(self, attribute_name):
    self._check_valid_attribute(attribute_name)
    return self._attributes[attribute_name].value

  def get_attribute_xml_string(self, attribute_name, prefix_root=None):
    self._check_valid_attribute(attribute_name)
    return self._attributes[attribute_name].to_xml_string(prefix_root)

  def get_attributes(self):
    fix_attribute_name = (
        lambda name: constants.DCLASS if name == constants.CLASS else name)
    return collections.OrderedDict(
        [(fix_attribute_name(name), self._get_attribute(name))
         for name in self._spec.attributes.keys()
         if self._get_attribute(name) is not None])

  def _set_attribute(self, attribute_name, value):
    self._check_valid_attribute(attribute_name)
    self._attributes[attribute_name].value = value
    self.namescope.increment_revision()

  def set_attributes(self, **kwargs):
    if constants.DCLASS in kwargs:
      kwargs[constants.CLASS] = kwargs[constants.DCLASS]
      del kwargs[constants.DCLASS]
    old_values = []
    with debugging.freeze_current_stack_trace():
      for attribute_name, new_value in kwargs.items():
        old_value = self._get_attribute(attribute_name)
        try:
          self._set_attribute(attribute_name, new_value)
          old_values.append((attribute_name, old_value))
        except:
          # On failure, restore old attribute values for those already set.
          for name, old_value in old_values:
            self._set_attribute(name, old_value)
          # Then raise a meaningful error.
          err_type, err, tb = sys.exc_info()
          raise err_type(
              f'during assignment to attribute {attribute_name!r} of '
              f'element <{self._spec.name}>: {err}').with_traceback(tb)

  def _remove_attribute(self, attribute_name):
    self._check_valid_attribute(attribute_name)
    self._attributes[attribute_name].clear()
    self.namescope.increment_revision()

  def _check_valid_child(self, element_name):
    try:
      return self._spec.children[element_name]
    except KeyError:
      raise AttributeError(
          '<{}> is not a valid child of <{}>'
          .format(element_name, self._spec.name))

  def get_children(self, element_name):
    child_spec = self._check_valid_child(element_name)
    if child_spec.repeated:
      return _ElementListView(spec=child_spec, parent=self)
    else:
      for child in self._children:
        if child.tag == element_name:
          return child
      if child_spec.on_demand:
        return None
      else:
        raise RuntimeError(
            'Cannot find the non-repeated child <{}> of <{}>. '
            'This should never happen, as we pre-create these in __init__. '
            'Please file an bug report. Thank you.'
            .format(element_name, self._spec.name))

  def add(self, element_name, **kwargs):
    """Add a new child element to this element.

    Args:
      element_name: The tag of the element to add.
      **kwargs: Attributes of the new element being created.

    Raises:
      ValueError: If the 'element_name' is not a valid child, or if an invalid
        attribute is specified in `kwargs`.

    Returns:
      An `mjcf.Element` corresponding to the newly created child element.
    """
    child_spec = self._check_valid_child(element_name)
    if child_spec.on_demand:
      need_new_on_demand = self.get_children(element_name) is None
    else:
      need_new_on_demand = False
    if not (child_spec.repeated or need_new_on_demand):
      raise ValueError('A <{}> child already exists, please access it directly.'
                       .format(element_name))
    new_element = _make_element(child_spec, self, attributes=kwargs)
    self._children.append(new_element)
    self.namescope.increment_revision()
    return new_element

  def __getattr__(self, name):
    if self._last_attribute_error:
      # This means that we got here through a @property raising AttributeError.
      # We therefore just re-raise the last AttributeError back to the user.
      # Note that self._last_attribute_error was set by our specially
      # instrumented @property decorator.
      exc = self._last_attribute_error
      self._last_attribute_error = None
      raise exc  # pylint: disable=raising-bad-type
    elif name in self._spec.children:
      return self.get_children(name)
    elif name in self._spec.attributes:
      return self._get_attribute(name)
    elif name == constants.DCLASS and constants.CLASS in self._spec.attributes:
      return self._get_attribute(constants.CLASS)
    else:
      raise AttributeError('object has no attribute: {}'.format(name))

  def __setattr__(self, name, value):
    # If this name corresponds to a descriptor for a slotted attribute or
    # settable property then try to invoke the descriptor to set the attribute
    # and return if successful.
    klass_attr = getattr(type(self), name, None)
    if klass_attr is not None:
      try:
        return klass_attr.__set__(self, value)
      except AttributeError:
        pass
    # If we did not find a settable descriptor then we look in the attribute
    # spec to see if there is a MuJoCo attribute matching this name.
    attribute_name = name if name != constants.DCLASS else constants.CLASS
    if attribute_name in self._spec.attributes:
      self._set_attribute(attribute_name, value)
    else:
      raise AttributeError('can\'t set attribute: {}'.format(name))

  def __delattr__(self, name):
    if name in self._spec.children:
      if self._spec.children[name].repeated:
        raise AttributeError(
            '`{0}` is a collection of child elements, '
            'which cannot be deleted. Did you mean to call `{0}.clear()`?'
            .format(name))
      else:
        return self.get_children(name).remove()
    elif name in self._spec.attributes:
      return self._remove_attribute(name)
    else:
      raise AttributeError('object has no attribute: {}'.format(name))

  def _check_attachments_on_remove(self, affect_attachments):
    if not affect_attachments and self._attachments:
      raise ValueError(
          'please use remove(affect_attachments=True) as this will affect some '
          'attributes and/or children belonging to an attached model')
    for child in self._children:
      child._check_attachments_on_remove(affect_attachments)  # pylint: disable=protected-access

  def remove(self, affect_attachments=False):
    """Removes this element from the model."""
    self._check_attachments_on_remove(affect_attachments)
    if affect_attachments:
      for attachment in self._attachments.values():
        attachment.remove(affect_attachments=True)
    for child in list(self._children):
      child.remove(affect_attachments)
    if self._spec.repeated or self._spec.on_demand:
      self._parent._children.remove(self)  # pylint: disable=protected-access
      for attribute in self._attributes.values():
        attribute._force_clear()  # pylint: disable=protected-access
      self._parent = None
      self._is_removed = True
    else:
      for attribute in self._attributes.values():
        attribute._force_clear()  # pylint: disable=protected-access
    self.namescope.increment_revision()

  @property
  def is_removed(self):
    return self._is_removed

  def all_children(self):
    all_children = [child for child in self._children]
    for attachment in self._attachments.values():
      all_children += [child for child in attachment.all_children()
                       if child.spec.repeated]
    return all_children

  def to_xml(self, prefix_root=None, debug_context=None):
    """Generates an etree._Element corresponding to this MJCF element.

    Args:
      prefix_root: (optional) A `NameScope` object to be treated as root
        for the purpose of calculating the prefix.
        If `None` then no prefix is included.
      debug_context: (optional) A `debugging.DebugContext` object to which
        the debugging information associated with the generated XML is written.
        This is intended for internal use within PyMJCF; users should never need
        manually pass this argument.

    Returns:
      An etree._Element object.
    """
    prefix_root = prefix_root or self.namescope
    xml_element = etree.Element(self._spec.name)
    self._attributes_to_xml(xml_element, prefix_root, debug_context)
    self._children_to_xml(xml_element, prefix_root, debug_context)
    return xml_element

  def _attributes_to_xml(self, xml_element, prefix_root, debug_context=None):
    del debug_context  # Unused.
    for attribute_name, attribute in self._attributes.items():
      attribute_value = attribute.to_xml_string(prefix_root)
      if attribute_name == self._spec.identifier and attribute_value is None:
        xml_element.set(attribute_name, self.full_identifier)
      elif attribute_value is None:
        continue
      else:
        xml_element.set(attribute_name, attribute_value)

  def _children_to_xml(self, xml_element, prefix_root, debug_context=None):
    for child in self.all_children():
      child_xml = child.to_xml(prefix_root, debug_context)
      if (child_xml.attrib or len(child_xml)  # pylint: disable=g-explicit-length-test
          or child.spec.repeated or child.spec.on_demand):
        xml_element.append(child_xml)
        if debugging.debug_mode() and debug_context:
          debug_comment = debug_context.register_element_for_debugging(child)
          xml_element.append(debug_comment)
          if len(child_xml) > 0:  # pylint: disable=g-explicit-length-test
            child_xml.insert(0, copy.deepcopy(debug_comment))

  def to_xml_string(self, prefix_root=None,
                    self_only=False, pretty_print=True, debug_context=None):
    """Generates an XML string corresponding to this MJCF element.

    Args:
      prefix_root: (optional) A `NameScope` object to be treated as root
        for the purpose of calculating the prefix.
        If `None` then no prefix is included.
      self_only: (optional) A boolean, whether to generate an XML corresponding
        only to this element without any children.
      pretty_print: (optional) A boolean, whether to the XML string should be
        properly indented.
      debug_context: (optional) A `debugging.DebugContext` object to which
        the debugging information associated with the generated XML is written.
        This is intended for internal use within PyMJCF; users should never need
        manually pass this argument.

    Returns:
      A string.
    """
    xml_element = self.to_xml(prefix_root, debug_context)
    if self_only and len(xml_element) > 0:  # pylint: disable=g-explicit-length-test
      etree.strip_elements(xml_element, '*')
      xml_element.text = '...'
    if (self_only and self._spec.identifier and
        not self._attributes[self._spec.identifier].to_xml_string(prefix_root)):
      del xml_element.attrib[self._spec.identifier]
    xml_string = util.to_native_string(
        etree.tostring(xml_element, pretty_print=pretty_print))
    if pretty_print and debug_context:
      return debug_context.commit_xml_string(xml_string)
    else:
      return xml_string

  def __str__(self):
    return self.to_xml_string(self_only=True, pretty_print=False)

  def __repr__(self):
    return 'MJCF Element: ' + str(self)

  def _check_valid_attachment(self, other):
    self_spec = self._spec
    if self_spec.name == constants.WORLDBODY:
      self_spec = self._spec.children[constants.BODY]

    other_spec = other.spec
    if other_spec.name == constants.WORLDBODY:
      other_spec = other_spec.children[constants.BODY]

    if other_spec != self_spec:
      raise ValueError(
          'The attachment must have the same spec as this element.')

  def _attach(self, other, exclude_worldbody=False, dry_run=False):
    """Attaches another element of the same spec to this element.

    All children of `other` will be treated as children of this element.
    All XML attributes which are defined in `other` but not defined in this
    element will be copied over, and any conflicting XML attribute value causes
    an error. After the attachment, any XML attribute modified in this element
    will also affect `other` and vice versa.

    Children of this element which are not a repeated elements will also be
    attached by the corresponding children of `other`.

    Args:
      other: Another Element with the same spec.
      exclude_worldbody: (optional) A boolean. If `True`, then don't do anything
        if `other` is a worldbody.
      dry_run: (optional) A boolean, if `True` only verify that the operation
        is valid without actually committing any change.

    Raises:
      ValueError: If `other` has a different spec, or if there are conflicting
        XML attribute values.
    """
    self._check_valid_attachment(other)
    if exclude_worldbody and other.tag == constants.WORLDBODY:
      return
    if dry_run:
      self._check_conflicting_attributes(other, copying=False)
    else:
      self._attachments[other.namescope] = other
      self._sync_attributes(other, copying=False)
    self._attach_children(other, exclude_worldbody, dry_run)
    if other.tag != constants.WORLDBODY and not dry_run:
      other._alias_attributes_dict(self._attributes)  # pylint: disable=protected-access

  def _detach(self, other_namescope):
    """Detaches a model with the specified namescope."""
    attached_element = self._attachments.get(other_namescope)
    if attached_element:
      attached_element._restore_attributes_dict()  # pylint: disable=protected-access
      del self._attachments[other_namescope]
    for child in self._children:
      child._detach(other_namescope)  # pylint: disable=protected-access

  def _check_conflicting_attributes(self, other, copying):
    for attribute_name, other_attribute in other.get_attributes().items():
      if attribute_name == constants.DCLASS:
        attribute_name = constants.CLASS
      if ((not self._attributes[attribute_name].conflict_allowed)
          and self._attributes[attribute_name].value is not None
          and other_attribute is not None
          and np.asarray(
              self._attributes[attribute_name].value != other_attribute).any()):
        raise ValueError(
            'Conflicting values for attribute `{}`: {} vs {}'
            .format(attribute_name,
                    self._attributes[attribute_name].value,
                    other_attribute))

  def _sync_attributes(self, other, copying):
    self._check_conflicting_attributes(other, copying)
    for attribute_name, other_attribute in other.get_attributes().items():
      if attribute_name == constants.DCLASS:
        attribute_name = constants.CLASS

      self_attribute = self._attributes[attribute_name]
      if other_attribute is not None:
        if self_attribute.conflict_behavior in _CONFLICT_BEHAVIOR_FUNC:
          if self_attribute.value is not None:
            self_attribute.value = (
                _CONFLICT_BEHAVIOR_FUNC[self_attribute.conflict_behavior](
                    self_attribute.value, other_attribute))
          else:
            self_attribute.value = other_attribute
        elif copying or not self_attribute.conflict_allowed:
          self_attribute.value = other_attribute

  def _attach_children(self, other, exclude_worldbody, dry_run=False):
    for other_child in other.all_children():
      if not other_child.spec.repeated:
        self_child = self.get_children(other_child.spec.name)
        self_child._attach(other_child, exclude_worldbody, dry_run)  # pylint: disable=protected-access

  def resolve_references(self):
    for attribute in self._attributes.values():
      if isinstance(attribute, attribute_types.Reference):
        if attribute.value and isinstance(attribute.value, str):
          referred = self.root.find(
              attribute.reference_namespace, attribute.value)
          if referred:
            attribute.value = referred
    for child in self.all_children():
      child.resolve_references()

  def _update_references(self, reference_dict):
    for attribute in self._attributes.values():
      if isinstance(attribute, attribute_types.Reference):
        if attribute.value in reference_dict:
          attribute.value = reference_dict[attribute.value]
    for child in self.all_children():
      child._update_references(reference_dict)  # pylint: disable=protected-access


class _AttachableElement(_ElementImpl):
  """Specialized object representing a <site> or <worldbody> element.

  This element defines a frame to which another MJCF model can be attached.
  """
  __slots__ = []

  def attach(self, attachment):
    """Attaches another MJCF model at this site.

    An empty <body> will be created as an attachment frame. All children of
    `attachment`'s <worldbody> will be treated as children of this frame.
    Furthermore, all other elements in `attachment` are merged into the root
    of the MJCF model to which this element belongs.

    Args:
      attachment: An MJCF `RootElement`

    Returns:
      An `mjcf.Element` corresponding to the attachment frame. A joint can be
      added directly to this frame to give degrees of freedom to the attachment.

    Raises:
      ValueError: If `other` is not a valid attachment to this element.
    """
    if not isinstance(attachment, RootElement):
      raise ValueError('Expected a mjcf.RootElement: got {}'
                       .format(attachment))
    if attachment.namescope.parent is not None:
      raise ValueError('The model specified is already attached elsewhere')
    if attachment.namescope == self.namescope:
      raise ValueError('Cannot merge a model to itself')
    self.root._attach(attachment, exclude_worldbody=True, dry_run=True)  # pylint: disable=protected-access

    if self.namescope.has_identifier('namescope', attachment.model):
      id_number = 1
      while self.namescope.has_identifier(
          'namescope', '{}_{}'.format(attachment.model, id_number)):
        id_number += 1
      attachment.model = '{}_{}'.format(attachment.model, id_number)
    attachment.namescope.parent = self.namescope

    if self.tag == constants.WORLDBODY:
      frame_parent = self
      frame_siblings = self._children
      index = len(frame_siblings)
    else:
      frame_parent = self._parent
      frame_siblings = self._parent._children  # pylint: disable=protected-access
      index = frame_siblings.index(self) + 1
      while (index < len(frame_siblings)
             and isinstance(frame_siblings[index], _AttachmentFrame)):
        index += 1
    frame = _AttachmentFrame(frame_parent, self, attachment)
    frame_siblings.insert(index, frame)
    self.root._attach(attachment, exclude_worldbody=True)  # pylint: disable=protected-access
    return frame


class _AttachmentFrame(_ElementImpl):
  """An specialized <body> representing a frame holding an external attachment.
  """
  __slots__ = ['_site', '_attachment']

  def __init__(self, parent, site, attachment):
    if parent.tag == constants.WORLDBODY:
      spec = schema.WORLD_ATTACHMENT_FRAME
    else:
      spec = schema.ATTACHMENT_FRAME

    spec_is_copied = False
    for child_name, child_spec in spec.children.items():
      if child_spec.namespace:
        if not spec_is_copied:
          spec = copy.deepcopy(spec)
          spec_is_copied = True
        spec_as_dict = child_spec._asdict()
        spec_as_dict['namespace'] = '{}{}attachment_frame_{}'.format(
            child_spec.namespace, constants.NAMESPACE_SEPARATOR, id(self))
        spec.children[child_name] = type(child_spec)(**spec_as_dict)

    attributes = {}
    with debugging.freeze_current_stack_trace():
      for attribute_name in spec.attributes.keys():
        if hasattr(site, attribute_name):
          attributes[attribute_name] = getattr(site, attribute_name)
    super(_AttachmentFrame, self).__init__(spec, parent, attributes)
    self._site = site
    self._attachment = attachment
    self._attachments[attachment.namescope] = attachment.worldbody
    self.namescope.add('attachment_frame', attachment.namescope.name, self)
    self.namescope.add('attached_model', attachment.namescope.name, attachment)

  def prefixed_identifier(self, prefix_root=None):
    prefix = self.namescope.full_prefix(prefix_root)
    return prefix + self._attachment.namescope.name + constants.PREFIX_SEPARATOR

  def to_xml(self, prefix_root=None, debug_context=None):
    xml_element = (
        super(_AttachmentFrame, self).to_xml(prefix_root, debug_context))
    xml_element.set('name', self.prefixed_identifier(prefix_root))
    return xml_element

  @property
  def full_identifier(self):
    return self.prefixed_identifier(self.namescope.root)

  def _detach(self, other_namescope):
    super(_AttachmentFrame, self)._detach(other_namescope)
    if other_namescope is self._attachment.namescope:
      self.namescope.remove('attachment_frame', self._attachment.namescope.name)
      self.namescope.remove('attached_model', self._attachment.namescope.name)
      self.remove()


class _AttachmentFrameChild(_ElementImpl):
  """A child element of an attachment frame.

  Right now, this is always a <joint> or a <freejoint>. The name of the joint
  is not freely specifiable, but instead just inherits from the parent frame.
  This ensures uniqueness, as attachment frame identifiers always end in '/'.
  """
  __slots__ = []

  def to_xml(self, prefix_root=None, debug_context=None):
    xml_element = (
        super(_AttachmentFrameChild, self).to_xml(prefix_root, debug_context))
    if self.spec.namespace is not None:
      if self.name:
        name = (self._parent.prefixed_identifier(prefix_root) +
                self.name + constants.PREFIX_SEPARATOR)
      else:
        name = self._parent.prefixed_identifier(prefix_root)
      xml_element.set('name', name)
    return xml_element

  def prefixed_identifier(self, prefix_root=None):
    if self.name:
      return (self._parent.prefixed_identifier(prefix_root) +
              self.name + constants.PREFIX_SEPARATOR)
    else:
      return self._parent.prefixed_identifier(prefix_root)


class _DefaultElement(_ElementImpl):
  """Specialized object representing a <default> element.

  This is necessary for the proper handling of global defaults.
  """
  __slots__ = []

  def _attach(self, other, exclude_worldbody=False, dry_run=False):
    self._check_valid_attachment(other)
    if ((not isinstance(self._parent, RootElement))
        or (not isinstance(other.parent, RootElement))):
      raise ValueError('Only global <{}> can be attached'
                       .format(constants.DEFAULT))
    if not dry_run:
      self._attachments[other.namescope] = other

  def all_children(self):
    return [child for child in self._children]

  def to_xml(self, prefix_root=None, debug_context=None):
    prefix_root = prefix_root or self.namescope
    xml_element = (
        super(_DefaultElement, self).to_xml(prefix_root, debug_context))
    if isinstance(self._parent, RootElement):
      root_default = etree.Element(self._spec.name)
      root_default.append(xml_element)
      for attachment in self._attachments.values():
        attachment_xml = attachment.to_xml(prefix_root, debug_context)
        for attachment_child_xml in attachment_xml:
          root_default.append(attachment_child_xml)
      xml_element = root_default
    return xml_element


class _ActuatorElement(_ElementImpl):
  """Specialized object representing an <actuator> element.

  This is necessary because MuJoCo requires that all 3rd-order actuators (i.e.
  those with internal dynamics) come after all 2nd-order actuators in the
  generated XML.
  """
  __slots__ = ()

  def _is_third_order_actuator(self, child):
    if child.tag == 'general':
      return child.dyntype and child.dyntype != 'none'
    elif child.tag == 'cylinder':
      return True  # The `<cylinder>` shortcut has 'filter' dynamics.
    else:
      return False  # No other actuator shortcuts have internal dynamics.

  def _children_to_xml(self, xml_element, prefix_root, debug_context=None):
    second_order = []
    third_order = []
    debug_comments = {}
    for child in self.all_children():
      child_xml = child.to_xml(prefix_root, debug_context)
      if (child_xml.attrib or len(child_xml)  # pylint: disable=g-explicit-length-test
          or child.spec.repeated or child.spec.on_demand):
        if self._is_third_order_actuator(child):
          third_order.append(child_xml)
        else:
          second_order.append(child_xml)
        if debugging.debug_mode() and debug_context:
          debug_comment = debug_context.register_element_for_debugging(child)
          debug_comments[child_xml] = debug_comment
          if len(child_xml) > 0:  # pylint: disable=g-explicit-length-test
            child_xml.insert(0, copy.deepcopy(debug_comment))
    # Ensure that all second-order actuators come before third-order actuators
    # in the XML.
    for child_xml in second_order + third_order:
      xml_element.append(child_xml)
      if debugging.debug_mode() and debug_context:
        xml_element.append(debug_comments[child_xml])


class RootElement(_ElementImpl):
  """The root `<mujoco>` element of an MJCF model."""
  __slots__ = ['_namescope']

  def __init__(self, model=None, model_dir='', assets=None):
    model = model or 'unnamed_model'
    self._namescope = namescope.NameScope(
        model, self, model_dir=model_dir, assets=assets)
    super(RootElement, self).__init__(
        spec=schema.MUJOCO, parent=None, attributes={'model': model})

  def _attach(self, other, exclude_worldbody=False, dry_run=False):
    self._check_valid_attachment(other)
    if not dry_run:
      self._attachments[other.namescope] = other
    self._attach_children(other, exclude_worldbody, dry_run)
    self.namescope.increment_revision()

  @property
  def namescope(self):
    return self._namescope

  @property
  def root(self):
    return self

  @property
  def model(self):
    return self._namescope.name

  @model.setter
  def model(self, new_name):
    old_name = self._namescope.name
    self._namescope.name = new_name
    self._attributes['model'].value = new_name
    if self.parent_model:
      self.parent_model.namescope.rename('attachment_frame', old_name, new_name)
      self.parent_model.namescope.rename('attached_model', old_name, new_name)

  def attach(self, other):
    return self.worldbody.attach(other)

  def detach(self):
    parent_model = self.parent_model
    if not parent_model:
      raise RuntimeError(
          'Cannot `detach` a model that is not attached to some other model.')
    else:
      parent_model._detach(self.namescope)  # pylint: disable=protected-access
      self.namescope.parent = None

  def include_copy(self, other, override_attributes=False):
    other_copier = copier.Copier(other)
    new_elements = other_copier.copy_into(self, override_attributes)
    self._update_references(new_elements)
    self.namescope.increment_revision()

  @property
  def parent_model(self):
    """The RootElement of the MJCF model to which this one is attached."""
    namescope_parent = self._namescope.parent
    return namescope_parent.mjcf_model if namescope_parent else None

  @property
  def root_model(self):
    return self.parent_model.root_model if self.parent_model else self

  def get_assets(self):
    """Returns a dict containing the binary assets referenced in this model.

    This will contain `{vfs_filename: contents}` pairs. `vfs_filename` will be
    the name of the asset in MuJoCo's Virtual File System, which corresponds to
    the filename given in the XML returned by `to_xml_string()`. `contents` is a
    bytestring.

    This dict can be used together with the result of `to_xml_string()` to
    construct a `mujoco.Physics` instance:

    ```python
    physics = mujoco.Physics.from_xml_string(
        xml_string=mjcf_model.to_xml_string(),
        assets=mjcf_model.get_assets())
    ```
    """
    # Get the assets referenced within this `RootElement`'s namescope.
    assets = {file_obj.to_xml_string(): file_obj.get_contents()
              for file_obj in self.namescope.files
              if file_obj.value}

    # Recursively add assets belonging to attachments.
    for attached_model in self._attachments.values():
      assets.update(attached_model.get_assets())

    return assets

  @property
  def full_identifier(self):
    return self._namescope.full_prefix(self._namescope.root)

  def __copy__(self):
    new_model = RootElement(model=self._namescope.name,
                            model_dir=self.namescope.model_dir)
    new_model.include_copy(self)
    return new_model

  def __deepcopy__(self, _):
    return self.__copy__()

  def is_same_as(self, other):
    if other is None or other.spec != self._spec:
      return False
    return self._has_same_children_as(other)


class _ElementListView(object):
  """A hybrid list/dict-like view to a group of repeated MJCF elements."""

  def __init__(self, spec, parent):
    self._spec = spec
    self._parent = parent
    self._elements = self._parent._children  # pylint: disable=protected-access
    self._scoped_elements = collections.OrderedDict(
        [(scope_namescope.name, getattr(scoped_parent, self._spec.name))
         for scope_namescope, scoped_parent
         in self._parent._attachments.items()])

  @property
  def spec(self):
    return self._spec

  @property
  def tag(self):
    return self._spec.name

  @property
  def namescope(self):
    return self._parent.namescope

  @property
  def parent(self):
    return self._parent

  def __len__(self):
    return len(self._full_list())

  def __iter__(self):
    return iter(self._full_list())

  def _identifier_not_found_error(self, index):
    return KeyError('An element <{}> with {}={!r} does not exist'
                    .format(self._spec.name, self._spec.identifier, index))

  def _find_index(self, index):
    """Locates an element given the index among siblings with the same tag."""
    if isinstance(index, str) and self._spec.identifier:
      for i, element in enumerate(self._elements):
        if (element.tag == self._spec.name
            and getattr(element, self._spec.identifier) == index):
          return i
      raise self._identifier_not_found_error(index)
    else:
      count = 0
      for i, element in enumerate(self._elements):
        if element.tag == self._spec.name:
          if index == count:
            return i
          else:
            count += 1
      raise IndexError('list index out of range')

  def _full_list(self):
    out_list = [element for element in self._elements
                if element.tag == self._spec.name]
    for scoped_elements in self._scoped_elements.values():
      out_list += scoped_elements[:]
    return out_list

  def clear(self):
    for child in self._full_list():
      child.remove()

  def __getitem__(self, index):
    if (isinstance(index, str) and self._spec.identifier
        and constants.PREFIX_SEPARATOR in index):
      scope_name = index.split(constants.PREFIX_SEPARATOR)[0]
      scoped_elements = self._scoped_elements[scope_name]
      try:
        return scoped_elements[index[(len(scope_name) + 1):]]
      except KeyError:
        # Re-raise so that the error shows the full, un-stripped index string
        raise self._identifier_not_found_error(index)
    elif isinstance(index, slice) or (isinstance(index, int) and index < 0):
      return self._full_list()[index]
    else:
      return self._elements[self._find_index(index)]

  def __delitem__(self, index):
    found_index = self._find_index(index)
    self._elements[found_index].remove()

  def __str__(self):
    return str(
        [element.to_xml_string(
            prefix_root=self.namescope, self_only=True, pretty_print=False)
         for element in self._full_list()])

  def __repr__(self):
    return 'MJCF Elements List: ' + str(self)


# This restores @property back to Python's built-in one.
del property
del _raw_property
