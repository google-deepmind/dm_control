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

"""An object to manage the scoping of identifiers in MJCF models."""

import collections

from dm_control.mjcf import constants
import six


class NameScope(object):
  """A name scoping context for an MJCF model.

  This object maintains the uniqueness of identifiers within each MJCF
  namespace. Examples of MJCF namespaces include 'body', 'joint', and 'geom'.
  Each namescope also carries a name, and can have a parent namescope.
  When MJCF models are merged, all identifiers gain a hierarchical prefix
  separated by '/', which is the concatenation of all scope names up to
  the root namescope.
  """

  def __init__(self, name, mjcf_model, model_dir='', assets=None):
    """Initializes a scope with the given name.

    Args:
      name: The scope's name
      mjcf_model: The RootElement of the MJCF model associated with this scope.
      model_dir: (optional) Path to the directory containing the model XML file.
        This is used to prefix the paths of all asset files.
      assets: (optional) A dictionary of pre-loaded assets, of the form
        `{filename: bytestring}`. If present, PyMJCF will search for assets in
        this dictionary before attempting to load them from the filesystem.
    """
    self._parent = None
    self._name = name
    self._mjcf_model = mjcf_model
    self._namespaces = collections.defaultdict(dict)
    self._model_dir = model_dir
    self._files = set()
    self._assets = assets or {}
    self._revision = 0

  @property
  def revision(self):
    return self._revision

  def increment_revision(self):
    self._revision += 1
    for namescope in six.itervalues(self._namespaces['namescope']):
      namescope.increment_revision()

  @property
  def name(self):
    """This scope's name."""
    return self._name

  @property
  def files(self):
    """A set containing the `File` attributes registered in this scope."""
    return self._files

  @property
  def assets(self):
    """A dictionary containing pre-loaded assets."""
    return self._assets

  @property
  def model_dir(self):
    """Path to the directory containing the model XML file."""
    return self._model_dir

  @name.setter
  def name(self, new_name):
    if self._parent:
      self._parent.add('namescope', new_name, self)
      self._parent.remove('namescope', self._name)
    self._name = new_name
    self.increment_revision()

  @property
  def mjcf_model(self):
    return self._mjcf_model

  @property
  def parent(self):
    """This parent `NameScope`, or `None` if this is a root scope."""
    return self._parent

  @parent.setter
  def parent(self, new_parent):
    if self._parent:
      self._parent.remove('namescope', self._name)
    self._parent = new_parent
    if self._parent:
      self._parent.add('namescope', self._name, self)
    self.increment_revision()

  @property
  def root(self):
    if self._parent is None:
      return self
    else:
      return self._parent.root

  def full_prefix(self, prefix_root=None, as_list=False):
    """The prefix for identifiers belonging to this scope.

    Args:
      prefix_root: (optional) A `NameScope` object to be treated as root
        for the purpose of calculating the prefix. If `None` then no prefix
        is produced.
      as_list: (optional) A boolean, if `True` return the list of prefix
        components. If `False`, return the full prefix string separated by
        `mjcf.constants.PREFIX_SEPARATOR`.

    Returns:
      The prefix string.
    """
    prefix_root = prefix_root or self
    if prefix_root != self and self._parent:
      prefix_list = self._parent.full_prefix(prefix_root, as_list=True)
      prefix_list.append(self._name)
    else:
      prefix_list = []
    if as_list:
      return prefix_list
    else:
      if prefix_list:
        prefix_list.append('')
      return constants.PREFIX_SEPARATOR.join(prefix_list)

  def _assign(self, namespace, identifier, obj):
    """Checks a proposed identifier's validity before assigning to an object."""
    namespace_dict = self._namespaces[namespace]
    if not isinstance(identifier, str):
      raise ValueError(
          'Identifier must be a string: got {}'.format(type(identifier)))
    elif constants.PREFIX_SEPARATOR in identifier:
      raise ValueError(
          'Identifier cannot contain {!r}: got {}'
          .format(constants.PREFIX_SEPARATOR, identifier))
    else:
      namespace_dict[identifier] = obj

  def add(self, namespace, identifier, obj):
    """Add an identifier to this name scope.

    Args:
      namespace: A string specifying the namespace to which the
        identifier belongs.
      identifier: The identifier string.
      obj: The object referred to by the identifier.

    Raises:
      ValueError: If `identifier` not valid.
    """
    namespace_dict = self._namespaces[namespace]
    if identifier in namespace_dict:
      raise ValueError('Duplicated identifier {!r} in namespace <{}>'
                       .format(identifier, namespace))
    else:
      self._assign(namespace, identifier, obj)
    self.increment_revision()

  def replace(self, namespace, identifier, obj):
    """Reassociates an identifier with a different object.

    Args:
      namespace: A string specifying the namespace to which the
        identifier belongs.
      identifier: The identifier string.
      obj: The object referred to by the identifier.

    Raises:
      ValueError: If `identifier` not valid.
    """
    self._assign(namespace, identifier, obj)
    self.increment_revision()

  def remove(self, namespace, identifier):
    """Removes an identifier from this name scope.

    Args:
      namespace: A string specifying the namespace to which the
        identifier belongs.
      identifier: The identifier string.

    Raises:
      KeyError: If `identifier` does not exist in this scope.
    """
    del self._namespaces[namespace][identifier]
    self.increment_revision()

  def rename(self, namespace, old_identifier, new_identifier):
    obj = self.get(namespace, old_identifier)
    self.add(namespace, new_identifier, obj)
    self.remove(namespace, old_identifier)

  def get(self, namespace, identifier):
    return self._namespaces[namespace][identifier]

  def has_identifier(self, namespace, identifier):
    return identifier in self._namespaces[namespace]
