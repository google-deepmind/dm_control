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

"""Base class for all MJCF elements in the object model."""

import abc

from dm_control.mjcf import constants


class Element(metaclass=abc.ABCMeta):
  """Abstract base class for an MJCF element.

  This class is provided so that `isinstance(foo, Element)` is `True` for all
  Element-like objects. We do not implement the actual element here because
  the actual object returned from traversing the object hierarchy is a
  weakproxy-like proxy to an actual element. This is because we do not allow
  orphaned non-root elements, so when a particular element is removed from the
  tree, all references held automatically become invalid.
  """
  __slots__ = []

  @abc.abstractmethod
  def get_init_stack(self):
    """Gets the stack trace where this element was first initialized."""

  @abc.abstractmethod
  def get_last_modified_stacks_for_all_attributes(self):
    """Gets a dict of stack traces where each attribute was last modified."""

  @abc.abstractmethod
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

  @property
  @abc.abstractmethod
  def tag(self):
    pass

  @property
  @abc.abstractmethod
  def spec(self):
    pass

  @property
  @abc.abstractmethod
  def parent(self):
    pass

  @property
  @abc.abstractmethod
  def namescope(self):
    pass

  @property
  @abc.abstractmethod
  def root(self):
    pass

  @abc.abstractmethod
  def prefixed_identifier(self, prefix_root):
    pass

  @property
  @abc.abstractmethod
  def full_identifier(self):
    """Fully-qualified identifier used for this element in the generated XML."""

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
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

  @abc.abstractmethod
  def get_attribute_xml_string(self, attribute_name, prefix_root=None):
    pass

  @abc.abstractmethod
  def get_attributes(self):
    pass

  @abc.abstractmethod
  def set_attributes(self, **kwargs):
    pass

  @abc.abstractmethod
  def get_children(self, element_name):
    pass

  @abc.abstractmethod
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

  @abc.abstractmethod
  def remove(self, affect_attachments=False):
    """Removes this element from the model."""

  @property
  @abc.abstractmethod
  def is_removed(self):
    pass

  @abc.abstractmethod
  def all_children(self):
    pass

  @abc.abstractmethod
  def to_xml(self, prefix_root=None, debug_context=None,
             *, precision=constants.XML_DEFAULT_PRECISION):
    """Generates an etree._Element corresponding to this MJCF element.

    Args:
      prefix_root: (optional) A `NameScope` object to be treated as root
        for the purpose of calculating the prefix.
        If `None` then no prefix is included.
      debug_context: (optional) A `debugging.DebugContext` object to which
        the debugging information associated with the generated XML is written.
        This is intended for internal use within PyMJCF; users should never need
        manually pass this argument.
      precision: (optional) Number of digits to output for floating point
        quantities.

    Returns:
      An etree._Element object.
    """

  @abc.abstractmethod
  def to_xml_string(self, prefix_root=None,
                    self_only=False, pretty_print=True, debug_context=None,
                    *, precision=constants.XML_DEFAULT_PRECISION):
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
      precision: (optional) Number of digits to output for floating point
        quantities.

    Returns:
      A string.
    """

  @abc.abstractmethod
  def resolve_references(self):
    pass
