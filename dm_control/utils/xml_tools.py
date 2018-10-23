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

"""Helper functions for model xml creation and modification."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from lxml import etree


def find_element(root, tag, name):
  """Finds and returns the first element of specified tag and name.

  Args:
    root: `etree.Element` to be searched recursively.
    tag: The `tag` property of the sought element.
    name: The `name` attribute of the sought element.

  Returns:
    An `etree.Element` with the specified properties.

  Raises:
    ValueError: If no matching element is found.
  """
  result = root.find('.//{}[@name={!r}]'.format(tag, name))
  if result is None:
    raise ValueError(
        'Element with tag {!r} and name {!r} not found'.format(tag, name))
  return result


def nested_element(element, depth):
  """Makes a nested `tree.Element` given a single element.

  If `depth=2`, the new tree will look like

  ```xml
  <element>
    <element>
      <element>
      </element>
    </element>
  </element>
  ```

  Args:
    element: The `etree.Element` used to create a nested structure.
    depth: An `int` denoting the nesting depth. The resulting will contain
      `element` nested `depth` times.


  Returns:
    A nested `etree.Element`.
  """
  if depth > 0:
    child = nested_element(copy.deepcopy(element), depth=(depth - 1))
    element.append(child)
  return element


def parse(file_obj):
  """Reads xml from a file and returns an `etree.Element`.

  Compared to the `etree.fromstring()`, this function removes the whitespace in
  the xml file. This means later on, a user can pretty print the `etree.Element`
  with `etree.tostring(element, pretty_print=True)`.

  Args:
    file_obj: A file or file-like object.

  Returns:
    `etree.Element` of the xml file.
  """
  parser = etree.XMLParser(remove_blank_text=True)
  return etree.parse(file_obj, parser)
