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

"""Helper object for keeping track of new elements created when copying MJCF."""


from dm_control.mjcf import constants


class Copier(object):
  """Helper for keeping track of new elements created when copying MJCF."""

  def __init__(self, source):
    if source._attachments:  # pylint: disable=protected-access
      raise NotImplementedError('Cannot copy from elements with attachments')
    self._source = source

  def copy_into(self, destination, override_attributes=False):
    """Copies this copier's element into a destination MJCF element."""
    newly_created_elements = {}
    destination._check_valid_attachment(self._source)  # pylint: disable=protected-access
    if override_attributes:
      destination.set_attributes(**self._source.get_attributes())
    else:
      destination._sync_attributes(self._source, copying=True)  # pylint: disable=protected-access
    for source_child in self._source.all_children():
      dest_child = None
      # First, if source_child has an identifier, we look for an existing child
      # element of self with the same identifier to override.
      if source_child.spec.identifier and override_attributes:
        identifier_attr = source_child.spec.identifier
        if identifier_attr == constants.CLASS:
          identifier_attr = constants.DCLASS
        identifier = getattr(source_child, identifier_attr)
        if identifier:
          dest_child = destination.find(source_child.spec.namespace, identifier)
        if dest_child is not None and dest_child.parent is not destination:
          raise ValueError(
              '<{}> with identifier {!r} is already a child of another element'
              .format(source_child.spec.namespace, identifier))
      # Next, we cover the case where either the child is not a repeated element
      # or if source_child has an identifier attribute but it isn't set.
      if not source_child.spec.repeated and dest_child is None:
        dest_child = destination.get_children(source_child.tag)

      # Add a new element if dest_child doesn't exist, either because it is
      # supposed to be a repeated child, or because it's an uncreated on-demand.
      if dest_child is None:
        dest_child = destination.add(
            source_child.tag, **source_child.get_attributes())
        newly_created_elements[source_child] = dest_child
        override_child_attributes = True
      else:
        override_child_attributes = override_attributes

      # Finally, copy attributes into dest_child.
      child_copier = Copier(source_child)
      newly_created_elements.update(
          child_copier.copy_into(dest_child, override_child_attributes))
    return newly_created_elements
