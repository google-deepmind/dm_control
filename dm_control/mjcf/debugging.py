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

"""Implements PyMJCF debug mode.

PyMJCF debug mode stores a stack trace each time the MJCF object is modified.
If Mujoco raises a compile error on the generated XML model, we would then be
able to find the original source line that created the offending element.
"""

import collections
import contextlib
import copy
import os
import re
import sys
import traceback

from absl import flags
from lxml import etree
import six

FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    'pymjcf_debug', False,
    'Enables PyMJCF debug mode (SLOW!). In this mode, a stack trace is logged '
    'each the MJCF object is modified. This may be helpful in locating the '
    'Python source line corresponding to a problematic element in the '
    'generated XML.')
flags.DEFINE_string(
    'pymjcf_debug_full_dump_dir', '',
    'Path to dump full debug info when Mujoco error is encountered.')

StackTraceEntry = collections.namedtuple(
    'StackTraceEntry', ('filename', 'line_number', 'function_name', 'text'))

ElementDebugInfo = collections.namedtuple(
    'ElementDebugInfo', ('element', 'init_stack', 'attribute_stacks'))

MODULE_PATH = os.path.dirname(sys.modules[__name__].__file__)
DEBUG_METADATA_PREFIX = 'pymjcfdebug'

_DEBUG_METADATA_TAG_PREFIX = '<!--' + DEBUG_METADATA_PREFIX
_DEBUG_METADATA_SEARCH_PATTERN = re.compile(
    r'<!--{}:(\d+)-->'.format(DEBUG_METADATA_PREFIX))

# Modified by `freeze_current_stack_trace`.
_CURRENT_FROZEN_STACK = None

# These globals will take their default values from the `--pymjcf_debug` and
# `--pymjcf_debug_full_dump_dir` flags respectively. We cannot use `FLAGS` as
# global variables because flag parsing might not have taken place (e.g. when
# running `nosetests`).
_DEBUG_MODE_ENABLED = None
_DEBUG_FULL_DUMP_DIR = None


def debug_mode():
  """Returns a boolean that indicates whether PyMJCF debug mode is enabled."""
  global _DEBUG_MODE_ENABLED
  if _DEBUG_MODE_ENABLED is None:
    if FLAGS.is_parsed():
      _DEBUG_MODE_ENABLED = FLAGS.pymjcf_debug
    else:
      _DEBUG_MODE_ENABLED = FLAGS['pymjcf_debug'].default
  return _DEBUG_MODE_ENABLED


def enable_debug_mode():
  """Enables PyMJCF debug mode."""
  global _DEBUG_MODE_ENABLED
  _DEBUG_MODE_ENABLED = True


def disable_debug_mode():
  """Disables PyMJCF debug mode."""
  global _DEBUG_MODE_ENABLED
  _DEBUG_MODE_ENABLED = False


def get_full_dump_dir():
  """Gets the directory to dump full debug info files."""
  global _DEBUG_FULL_DUMP_DIR
  if _DEBUG_FULL_DUMP_DIR is None:
    if FLAGS.is_parsed():
      _DEBUG_FULL_DUMP_DIR = FLAGS.pymjcf_debug_full_dump_dir
    else:
      _DEBUG_FULL_DUMP_DIR = FLAGS['pymjcf_debug_full_dump_dir'].default
  return _DEBUG_FULL_DUMP_DIR


def set_full_dump_dir(dump_path):
  """Sets the directory to dump full debug info files."""
  global _DEBUG_FULL_DUMP_DIR
  _DEBUG_FULL_DUMP_DIR = dump_path


def get_current_stack_trace():
  """Returns the stack trace of the current execution frame.

  Returns:
    A list of `StackTraceEntry` named tuples corresponding to the current stack
    trace of the process, truncated to immediately before entry into
    PyMJCF internal code.
  """
  if _CURRENT_FROZEN_STACK:
    return copy.deepcopy(_CURRENT_FROZEN_STACK)
  else:
    return _get_actual_current_stack_trace()


def _get_actual_current_stack_trace():
  """Returns the stack trace of the current execution frame.

  Returns:
    A list of `StackTraceEntry` named tuples corresponding to the current stack
    trace of the process, truncated to immediately before entry into
    PyMJCF internal code.
  """
  raw_stack = traceback.extract_stack()
  processed_stack = []
  for raw_stack_item in raw_stack:
    stack_item = StackTraceEntry(*raw_stack_item)
    if (stack_item.filename.startswith(MODULE_PATH)
        and not stack_item.filename.endswith('_test.py')):
      break
    else:
      processed_stack.append(stack_item)
  return processed_stack


@contextlib.contextmanager
def freeze_current_stack_trace():
  """A context manager that freezes the stack trace.

  AVOID USING THIS CONTEXT MANAGER OUTSIDE OF INTERNAL PYMJCF IMPLEMENTATION,
  AS IT REDUCES THE USEFULNESS OF DEBUG MODE.

  If PyMJCF debug mode is enabled, calls to `debugging.get_current_stack_trace`
  within this context will always return the stack trace from when this context
  was entered.

  The frozen stack is global to this debugging module. That is, if the context
  is entered while another one is still active, then the stack trace of the
  outermost one is returned.

  This context significantly speeds up bulk operations in debug mode, e.g.
  parsing an existing XML string or creating a deeply-nested element, as it
  prevents the same stack trace from being repeatedly constructed.

  Yields:
    `None`
  """
  global _CURRENT_FROZEN_STACK
  if debug_mode() and _CURRENT_FROZEN_STACK is None:
    _CURRENT_FROZEN_STACK = _get_actual_current_stack_trace()
    yield
    _CURRENT_FROZEN_STACK = None
  else:
    yield


class DebugContext(object):
  """A helper object to store debug information for a generated XML string.

  This class is intended for internal use within the PyMJCF implementation.
  """

  def __init__(self):
    self._xml_string = None
    self._debug_info_for_element_ids = {}

  def register_element_for_debugging(self, elem):
    """Registers an `Element` and returns debugging metadata for the XML.

    Args:
      elem: An `mjcf.Element`.

    Returns:
      An `lxml.etree.Comment` that represents debugging metadata in the
      generated XML.
    """
    if not debug_mode():
      return None
    else:
      self._debug_info_for_element_ids[id(elem)] = ElementDebugInfo(
          elem,
          copy.deepcopy(elem.get_init_stack()),
          copy.deepcopy(elem.get_last_modified_stacks_for_all_attributes()))
      return etree.Comment('{}:{}'.format(DEBUG_METADATA_PREFIX, id(elem)))

  def commit_xml_string(self, xml_string):
    """Commits the XML string associated with this debug context.

    This function also formats the XML string to make sure that the debugging
    metadata appears on the same line as the corresponding XML element.

    Args:
      xml_string: A pretty-printed XML string.

    Returns:
      A reformatted XML string where all debugging metadata appears on the same
      line as the corresponding XML element.
    """
    formatted = re.sub(r'\n\s*' + _DEBUG_METADATA_TAG_PREFIX,
                       _DEBUG_METADATA_TAG_PREFIX, xml_string)
    self._xml_string = formatted
    return formatted

  def process_and_raise_last_exception(self):
    """Processes and re-raises the last mujoco.wrapper.Error caught.

    This function will insert the relevant line from the source XML to the error
    message. If debug mode is enabled, additional debugging information is
    appended to the error message. If debug mode is not enabled, the error
    message instructs the user to enable it by rerunning the executable with an
    appropriate flag.
    """
    err_type, err, stack = sys.exc_info()
    line_number_match = re.search(r'[Ll][Ii][Nn][Ee]\s*[:=]?\s*(\d+)', str(err))
    if line_number_match:
      xml_line_number = int(line_number_match.group(1))
      xml_line = self._xml_string.split('\n')[xml_line_number - 1]
      stripped_xml_line = xml_line.strip()
      comment_match = re.search(_DEBUG_METADATA_TAG_PREFIX, stripped_xml_line)
      if comment_match:
        stripped_xml_line = stripped_xml_line[:comment_match.start()]
    else:
      xml_line = ''
      stripped_xml_line = ''

    message_lines = []
    if debug_mode():
      if get_full_dump_dir():
        self.dump_full_debug_info_to_disk()
      message_lines.extend([
          'Compile error raised by Mujoco.',
          str(err)])
      if xml_line:
        message_lines.extend([
            stripped_xml_line,
            self._generate_debug_message_from_xml_line(xml_line)])
    else:
      message_lines.extend([
          'Compile error raised by Mujoco; '
          'run again with --pymjcf_debug for additional debug information.',
          str(err)])
      if xml_line:
        message_lines.append(stripped_xml_line)

    message = '\n'.join(message_lines)
    six.reraise(err_type, err_type(message), stack)

  @property
  def default_dump_dir(self):
    return get_full_dump_dir()

  @property
  def debug_mode(self):
    return debug_mode()

  def dump_full_debug_info_to_disk(self, dump_dir=None):
    """Dumps full debug information to disk.

    Full debug information consists of an XML file whose elements are tagged
    with a unique ID, and a stack trace file for each element ID. Each stack
    trace file consists of a stack trace for when the element was created, and
    when each attribute was last modified.

    Args:
      dump_dir: Full path to the directory in which dump files are created.

    Raises:
      ValueError: If neither `dump_dir` nor the global dump path is given. The
        global dump path can be specified either via the
        --pymjcf_debug_full_dump_dir flag or via `debugging.set_full_dump_dir`.
    """
    dump_dir = dump_dir or self.default_dump_dir
    if not dump_dir:
      raise ValueError('`dump_dir` is not specified')
    section_separator = '\n' + ('=' * 80) + '\n'
    def dump_stack(header, stack, f):
      indent = '    '
      f.write(header + '\n')
      for stack_entry in stack:
        f.write(indent + '`{}` at {}:{}\n'
                .format(stack_entry.function_name,
                        stack_entry.filename, stack_entry.line_number))
        f.write((indent * 2) + str(stack_entry.text) + '\n')
      f.write(section_separator)
    with open(os.path.join(dump_dir, 'model.xml'), 'w') as f:
      f.write(self._xml_string)
    for elem_id, debug_info in self._debug_info_for_element_ids.items():
      with open(os.path.join(dump_dir, str(elem_id) + '.dump'), 'w') as f:
        f.write('{}:{}\n'.format(DEBUG_METADATA_PREFIX, elem_id))
        f.write(str(debug_info.element) + '\n')
        dump_stack('Element creation', debug_info.init_stack, f)
        for attrib_name, stack in debug_info.attribute_stacks.items():
          attrib_value = (
              debug_info.element.get_attribute_xml_string(attrib_name))
          if stack[-1] == debug_info.init_stack[-1]:
            if attrib_value is not None:
              f.write('Attribute {}="{}"\n'.format(attrib_name, attrib_value))
              f.write('    was set when the element was created\n')
              f.write(section_separator)
          else:
            if attrib_value is not None:
              dump_stack('Attribute {}="{}"'.format(attrib_name, attrib_value),
                         stack, f)
            else:
              dump_stack(
                  'Attribute {} was CLEARED'.format(attrib_name), stack, f)

  def _generate_debug_message_from_xml_line(self, xml_line):
    """Generates a debug message by parsing the metadata on an XML line."""
    metadata_match = _DEBUG_METADATA_SEARCH_PATTERN.search(xml_line)
    if metadata_match:
      elem_id = int(metadata_match.group(1))
      return self._generate_debug_message_from_element_id(elem_id)
    else:
      return ''

  def _generate_debug_message_from_element_id(self, elem_id):
    """Generates a debug message for the specified Element."""
    out = []
    debug_info = self._debug_info_for_element_ids[elem_id]

    out.append('Debug summary for element:')
    if not get_full_dump_dir():
      out.append(
          '  * Full debug info can be dumped to disk by setting the '
          'flag --pymjcf_debug_full_dump_dir=path/to/dump>')
    out.append('  * Element object was created by `{}` at {}:{}'
               .format(debug_info.init_stack[-1].function_name,
                       debug_info.init_stack[-1].filename,
                       debug_info.init_stack[-1].line_number))

    for attrib_name, stack in debug_info.attribute_stacks.items():
      attrib_value = debug_info.element.get_attribute_xml_string(attrib_name)
      if stack[-1] == debug_info.init_stack[-1]:
        if attrib_value is not None:
          out.append('  * {}="{}" was set when the element was created'
                     .format(attrib_name, attrib_value))
      else:
        if attrib_value is not None:
          out.append('  * {}="{}" was set by `{}` at `{}:{}`'
                     .format(attrib_name, attrib_value,
                             stack[-1].function_name, stack[-1].filename,
                             stack[-1].line_number))
        else:
          out.append('  * {} was CLEARED by `{}` at {}:{}'
                     .format(attrib_name, stack[-1].function_name,
                             stack[-1].filename, stack[-1].line_number))

    return '\n'.join(out)
