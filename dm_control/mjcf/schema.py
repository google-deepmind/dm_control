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

"""A Python object representation of Mujoco's MJCF schema.

The root schema is provided as a module-level constant `schema.MUJOCO`.
"""

import collections
import copy
import os

from dm_control.mjcf import attribute
from lxml import etree

from dm_control.utils import io as resources

_SCHEMA_XML_PATH = os.path.join(os.path.dirname(__file__), 'schema.xml')

_ARRAY_DTYPE_MAP = {
    'int': int,
    'float': float,
    'string': str
}

_SCALAR_TYPE_MAP = {
    'int': attribute.Integer,
    'float': attribute.Float,
    'string': attribute.String
}

ElementSpec = collections.namedtuple(
    'ElementSpec', ('name', 'repeated', 'on_demand', 'identifier', 'namespace',
                    'attributes', 'children'))

AttributeSpec = collections.namedtuple(
    'AttributeSpec', ('name', 'type', 'required',
                      'conflict_allowed', 'conflict_behavior', 'other_kwargs'))

# Additional namespaces that are not present in the MJCF schema but can
# be used in `find` and `find_all`.
_ADDITIONAL_FINDABLE_NAMESPACES = frozenset(['attachment_frame'])


def _str2bool(string):
  """Converts either 'true' or 'false' (not case-sensitively) into a boolean."""
  if string is None:
    return False
  else:
    string = string.lower()

  if string == 'true':
    return True
  elif string == 'false':
    return False
  else:
    raise ValueError(
        'String should either be `true` or `false`: got {}'.format(string))


def parse_schema(schema_path):
  """Parses the schema XML.

  Args:
    schema_path: Path to the schema XML file.

  Returns:
    An `ElementSpec` for the root element in the schema.
  """
  with resources.GetResourceAsFile(schema_path) as file_handle:
    schema_xml = etree.parse(file_handle).getroot()
  return _parse_element(schema_xml)


def _parse_element(element_xml):
  """Parses an <element> element in the schema."""
  name = element_xml.get('name')
  if not name:
    raise ValueError('Element must always have a name')
  repeated = _str2bool(element_xml.get('repeated'))
  on_demand = _str2bool(element_xml.get('on_demand'))

  attributes = collections.OrderedDict()
  attributes_xml = element_xml.find('attributes')
  if attributes_xml is not None:
    for attribute_xml in attributes_xml.findall('attribute'):
      attributes[attribute_xml.get('name')] = _parse_attribute(attribute_xml)

  identifier = None
  namespace = None
  for attribute_spec in attributes.values():
    if attribute_spec.type == attribute.Identifier:
      identifier = attribute_spec.name
      namespace = element_xml.get('namespace') or name

  children = collections.OrderedDict()
  children_xml = element_xml.find('children')
  if children_xml is not None:
    for child_xml in children_xml.findall('element'):
      children[child_xml.get('name')] = _parse_element(child_xml)

  element_spec = ElementSpec(
      name, repeated, on_demand, identifier, namespace, attributes, children)

  recursive = _str2bool(element_xml.get('recursive'))
  if recursive:
    element_spec.children[name] = element_spec

  common_keys = set(element_spec.attributes).intersection(element_spec.children)
  if common_keys:
    raise RuntimeError(
        'Element \'{}\' contains the following attributes and children with '
        'the same name: \'{}\'. This violates the design assumptions of '
        'this library. Please file a bug report. Thank you.'
        .format(name, sorted(common_keys)))

  return element_spec


def _parse_attribute(attribute_xml):
  """Parses an <attribute> element in the schema."""
  name = attribute_xml.get('name')
  required = _str2bool(attribute_xml.get('required'))
  conflict_allowed = _str2bool(attribute_xml.get('conflict_allowed'))
  conflict_behavior = attribute_xml.get('conflict_behavior', 'replace')
  attribute_type = attribute_xml.get('type')
  other_kwargs = {}
  if attribute_type == 'keyword':
    attribute_callable = attribute.Keyword
    other_kwargs['valid_values'] = attribute_xml.get('valid_values').split(' ')
  elif attribute_type == 'array':
    array_size_str = attribute_xml.get('array_size')
    attribute_callable = attribute.Array
    other_kwargs['length'] = int(array_size_str) if array_size_str else None
    other_kwargs['dtype'] = _ARRAY_DTYPE_MAP[attribute_xml.get('array_type')]
  elif attribute_type == 'identifier':
    attribute_callable = attribute.Identifier
  elif attribute_type == 'reference':
    attribute_callable = attribute.Reference
    other_kwargs['reference_namespace'] = (
        attribute_xml.get('reference_namespace') or name)
  elif attribute_type == 'basepath':
    attribute_callable = attribute.BasePath
    other_kwargs['path_namespace'] = attribute_xml.get('path_namespace')
  elif attribute_type == 'file':
    attribute_callable = attribute.File
    other_kwargs['path_namespace'] = attribute_xml.get('path_namespace')
  else:
    try:
      attribute_callable = _SCALAR_TYPE_MAP[attribute_type]
    except KeyError:
      raise ValueError('Invalid attribute type: {}'.format(attribute_type))

  return AttributeSpec(
      name=name, type=attribute_callable, required=required,
      conflict_allowed=conflict_allowed, conflict_behavior=conflict_behavior,
      other_kwargs=other_kwargs)


def collect_namespaces(root_spec):
  """Constructs a set of namespaces in a given ElementSpec.

  Args:
    root_spec: An `ElementSpec` for the root element in the schema.

  Returns:
    A set of strings specifying the names of all the namespaces that are present
    in the spec.
  """
  findable_namespaces = set()
  def update_namespaces_from_spec(spec):
    findable_namespaces.add(spec.namespace)
    for child_spec in spec.children.values():
      if child_spec is not spec:
        update_namespaces_from_spec(child_spec)
  update_namespaces_from_spec(root_spec)
  return findable_namespaces


MUJOCO = parse_schema(_SCHEMA_XML_PATH)
FINDABLE_NAMESPACES = frozenset(
    collect_namespaces(MUJOCO).union(_ADDITIONAL_FINDABLE_NAMESPACES))


def _attachment_frame_spec(is_world_attachment):
  """Create specs for attachment frames.

  Attachment frames are specialized <body> without an identifier.
  The only allowed children are joints which also don't have identifiers.

  Args:
    is_world_attachment: Whether we are creating a spec for attachments to
      worldbody. If `True`, allow <freejoint> as child.

  Returns:
    An `ElementSpec`.
  """
  frame_spec = ElementSpec(
      'body', repeated=True, on_demand=False, identifier=None, namespace='body',
      attributes=collections.OrderedDict(),
      children=collections.OrderedDict())

  body_spec = MUJOCO.children['worldbody'].children['body']
  # 'name' and 'childclass' attributes are excluded.
  for attrib_name in (
      'mocap', 'pos', 'quat', 'axisangle', 'xyaxes', 'zaxis', 'euler', 'user'):
    frame_spec.attributes[attrib_name] = copy.deepcopy(
        body_spec.attributes[attrib_name])

  inertial_spec = body_spec.children['inertial']
  frame_spec.children['inertial'] = copy.deepcopy(inertial_spec)
  joint_spec = body_spec.children['joint']
  frame_spec.children['joint'] = ElementSpec(
      'joint', repeated=True, on_demand=False,
      identifier=None, namespace='joint',
      attributes=copy.deepcopy(joint_spec.attributes),
      children=collections.OrderedDict())

  if is_world_attachment:
    freejoint_spec = (MUJOCO.children['worldbody']
                      .children['body'].children['freejoint'])
    frame_spec.children['freejoint'] = ElementSpec(
        'freejoint', repeated=False, on_demand=True,
        identifier=None, namespace='joint',
        attributes=copy.deepcopy(freejoint_spec.attributes),
        children=collections.OrderedDict())

  return frame_spec

ATTACHMENT_FRAME = _attachment_frame_spec(is_world_attachment=False)
WORLD_ATTACHMENT_FRAME = _attachment_frame_spec(is_world_attachment=True)


def override_schema(schema_xml_path):
  """Override the schema with a custom xml.

  This method updates several global variables and care should be taken not to
  call it if the pre-update values have already been used.

  Args:
    schema_xml_path: Path to schema xml file.
  """
  global MUJOCO
  global FINDABLE_NAMESPACES

  MUJOCO = parse_schema(schema_xml_path)
  FINDABLE_NAMESPACES = frozenset(
      collect_namespaces(MUJOCO).union(_ADDITIONAL_FINDABLE_NAMESPACES))

