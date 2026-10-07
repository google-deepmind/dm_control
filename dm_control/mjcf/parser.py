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

"""Functions for parsing XML into an MJCF object model."""

import io
import os
import sys
import zipfile

from dm_control.mjcf import constants
from dm_control.mjcf import debugging
from dm_control.mjcf import element
from lxml import etree
# Copybara placeholder for internal file handling dependency.
from dm_control.utils import io as resources


def from_xml_string(xml_string, escape_separators=False,
                    model_dir='', resolve_references=True, assets=None):
  """Parses an XML string into an MJCF object model.

  Args:
    xml_string: An XML string representing an MJCF model.
    escape_separators: (optional) A boolean, whether to replace '/' characters
      in element identifiers. If `False`, any '/' present in the XML causes
      a ValueError to be raised.
    model_dir: (optional) Path to the directory containing the model XML file.
      This is used to prefix the paths of all asset files.
    resolve_references: (optional) A boolean indicating whether the parser
      should attempt to resolve reference attributes to a corresponding element.
    assets: (optional) A dictionary of pre-loaded assets, of the form
      `{filename: bytestring}`. If present, PyMJCF will search for assets in
      this dictionary before attempting to load them from the filesystem.

  Returns:
    An `mjcf.RootElement`.
  """
  xml_root = etree.fromstring(xml_string)
  return _parse(xml_root, escape_separators,
                model_dir=model_dir,
                resolve_references=resolve_references,
                assets=assets)


def from_file(file_handle, escape_separators=False,
              model_dir='', resolve_references=True, assets=None):
  """Parses an XML file into an MJCF object model.

  Args:
    file_handle: A Python file-like handle.
    escape_separators: (optional) A boolean, whether to replace '/' characters
      in element identifiers. If `False`, any '/' present in the XML causes
      a ValueError to be raised.
    model_dir: (optional) Path to the directory containing the model XML file.
      This is used to prefix the paths of all asset files.
    resolve_references: (optional) A boolean indicating whether the parser
      should attempt to resolve reference attributes to a corresponding element.
    assets: (optional) A dictionary of pre-loaded assets, of the form
      `{filename: bytestring}`. If present, PyMJCF will search for assets in
      this dictionary before attempting to load them from the filesystem.

  Returns:
    An `mjcf.RootElement`.
  """
  xml_root = etree.parse(file_handle).getroot()
  return _parse(xml_root, escape_separators,
                model_dir=model_dir,
                resolve_references=resolve_references,
                assets=assets)


def from_path(path, escape_separators=False, resolve_references=True,
              assets=None):
  """Parses an XML file into an MJCF object model.

  Args:
    path: A path to an XML file. This path should be loadable using
      `resources.GetResource`.
    escape_separators: (optional) A boolean, whether to replace '/' characters
      in element identifiers. If `False`, any '/' present in the XML causes
      a ValueError to be raised.
    resolve_references: (optional) A boolean indicating whether the parser
      should attempt to resolve reference attributes to a corresponding element.
    assets: (optional) A dictionary of pre-loaded assets, of the form
      `{filename: bytestring}`. If present, PyMJCF will search for assets in
      this dictionary before attempting to load them from the filesystem.

  Returns:
    An `mjcf.RootElement`.
  """
  model_dir, _ = os.path.split(path)
  contents = resources.GetResource(path)
  xml_root = etree.fromstring(contents)
  return _parse(xml_root, escape_separators,
                model_dir=model_dir, resolve_references=resolve_references,
                assets=assets)


def from_zip(path, model_file='model.xml', escape_separators=False,
             resolve_references=True):
  """Parses a zipped XML file into an MJCF object model.

  Args:
    path: A path to a zip file containing an MJCF model and its assets.
    model_file: If the zip contains multiple XML files, specify the name of the
      main model file. Ignored if the zip only contains one XML file.
    escape_separators: (optional) A boolean, whether to replace '/' characters
      in element identifiers. If `False`, any '/' present in the XML causes a
      ValueError to be raised.
    resolve_references: (optional) A boolean indicating whether the parser
      should attempt to resolve reference attributes to a corresponding element.

  Returns:
    An `mjcf.RootElement`.

  Raises:
    ValueError: If:
      - the path does not point to a zip file
      - the zip file contains no XML files
      - the zip file contains more than one XML file and none of them have the
        name specified in `model_file`.
  """
  contents = resources.GetResource(path)
  if not zipfile.is_zipfile(io.BytesIO(contents)):
    raise ValueError(f'File {path} is not a zip file.')
  with zipfile.ZipFile(io.BytesIO(contents), 'r') as zf:
    xml_files = [f for f in zf.namelist() if f.endswith('.xml')]
    if not xml_files:
      raise ValueError(f'No XML file found in {path}.')
    elif len(xml_files) > 1:
      model_files = [f for f in xml_files if f == model_file]
      if not model_files:
        raise ValueError(
            f'Multiple XML files found in {path}, but none named {model_file}.'
        )
      xml_path = model_files[0]
    else:
      xml_path = xml_files[0]
    xml_string = zf.read(xml_path)

    model_dir = os.path.dirname(xml_path)
    assets = {
        os.path.relpath(name, model_dir): zf.read(name)
        for name in zf.namelist()
        if not (name.endswith(os.path.sep) or name == xml_path)
    }

  xml_root = etree.fromstring(xml_string)
  return _parse(xml_root, escape_separators,
                resolve_references=resolve_references, assets=assets)


def _parse(xml_root, escape_separators=False,
           model_dir='', resolve_references=True, assets=None):
  """Parses a complete MJCF model from an XML.

  Args:
    xml_root: An `etree.Element` object.
    escape_separators: (optional) A boolean, whether to replace '/' characters
      in element identifiers. If `False`, any '/' present in the XML causes
      a ValueError to be raised.
    model_dir: (optional) Path to the directory containing the model XML file.
      This is used to prefix the paths of all asset files.
    resolve_references: (optional) A boolean indicating whether the parser
      should attempt to resolve reference attributes to a corresponding element.
    assets: (optional) A dictionary of pre-loaded assets, of the form
      `{filename: bytestring}`. If present, PyMJCF will search for assets in
      this dictionary before attempting to load them from the filesystem.

  Returns:
    An `mjcf.RootElement`.

  Raises:
    ValueError: If `xml_root`'s tag is not 'mujoco.*'.
  """

  assets = assets or {}

  if not xml_root.tag.startswith('mujoco'):
    raise ValueError('Root element of the XML should be <mujoco.*>: got <{}>'
                     .format(xml_root.tag))

  with debugging.freeze_current_stack_trace():
    # Recursively parse any included XML files.
    to_include = []
    # Unsloth/User Fix: Use .iter() to find ALL include tags, not just root ones.
    # We collect them first to avoid issues while modifying the tree.
    include_tags = list(xml_root.iter('include'))
    
    for include_tag in include_tags:
      try:
        # First look for the path to the included XML file in the assets dict.
        path_or_xml_string = assets[include_tag.attrib['file']]
        parsing_func = from_xml_string
      except KeyError:
        # If it's not present in the assets dict then attempt to load the XML
        # from the filesystem.
        path_or_xml_string = os.path.join(model_dir, include_tag.attrib['file'])
        parsing_func = from_path
      included_mjcf = parsing_func(
          path_or_xml_string,
          escape_separators=escape_separators,
          resolve_references=resolve_references,
          assets=assets)
      to_include.append(included_mjcf)
      
      # Remove the <include/> tag.
      # Since we are iterating deeper, we need to find the parent to remove it.
      parent = include_tag.getparent()
      if parent is not None:
          parent.remove(include_tag)
      else:
          # Fallback for root (though unlikely via iter unless it is self)
          xml_root.remove(include_tag)

    # Parse the main XML file.
    try:
      model = xml_root.attrib.pop('model')
    except KeyError:
      model = None
    mjcf_root = element.RootElement(
        model=model, model_dir=model_dir, assets=assets)
    _parse_children(xml_root, mjcf_root, escape_separators)

    # Merge in the included XML files.
    for included_mjcf in to_include:
      # The included MJCF might have been automatically assigned a model name
      # that conficts with that of `mjcf_root`, so we override it here.
      included_mjcf.model = mjcf_root.model
      mjcf_root.include_copy(included_mjcf)

    if resolve_references:
      mjcf_root.resolve_references()
    return mjcf_root


def _parse_children(xml_element, mjcf_element, escape_separators=False):
  """Parses all children of a given XML element into an MJCF element.

  Args:
    xml_element: The source `etree.Element` object.
    mjcf_element: The target `mjcf.Element` object.
    escape_separators: (optional) A boolean, whether to replace '/' characters
      in element identifiers. If `False`, any '/' present in the XML causes
      a ValueError to be raised.
  """
  for xml_child in xml_element:
    if xml_child.tag is etree.Comment or xml_child.tag is etree.PI:
      continue
    try:
      child_spec = mjcf_element.spec.children[xml_child.tag]
      if escape_separators:
        attributes = {}
        for name, value in xml_child.attrib.items():
          # skip flipping the slash for fields that may contain paths, like
          # custom text and asset file.
          if name in ['data', 'file', 'meshdir', 'assetdir', 'texturedir',
                      'content_type', 'fileleft', 'fileright', 'fileback',
                      'filefront', 'plugin', 'key', 'value']:
            attributes[name] = value
          else:
            new_value = value.replace(
                constants.PREFIX_SEPARATOR_ESCAPE,
                constants.PREFIX_SEPARATOR_ESCAPE * 2)
            new_value = new_value.replace(
                constants.PREFIX_SEPARATOR, constants.PREFIX_SEPARATOR_ESCAPE)
            attributes[name] = new_value
      else:
        attributes = dict(xml_child.attrib)
      if child_spec.repeated or child_spec.on_demand:
        mjcf_child = mjcf_element.add(xml_child.tag, **attributes)
      else:
        mjcf_child = getattr(mjcf_element, xml_child.tag)
        mjcf_child.set_attributes(**attributes)
    except:  # pylint: disable=bare-except
      err_type, err, traceback = sys.exc_info()
      raise err_type(  # pylint: disable=raise-missing-from
          f'Line {xml_child.sourceline}: error while parsing element '
          f'<{xml_child.tag}>: {err}').with_traceback(traceback)
    _parse_children(xml_child, mjcf_child, escape_separators)
