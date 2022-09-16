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

"""Misc helper functions needed by autowrap.py."""

import collections

_MJXMACRO_SUFFIX = "_POINTERS"


class Indenter:
  r"""Callable context manager for tracking string indentation levels.

  Example usage:

  ```python
  idt = Indenter()
  s = idt("level 0\n")
  with idt:
    s += idt("level 1\n")
    with idt:
      s += idt("level 2\n")
    s += idt("level 1 again\n")
  s += idt("back to level 0\n")
  print(s)
  ```
  """

  def __init__(self, level=0, indent_str="  "):
    """Initializes an Indenter.

    Args:
      level: The initial indentation level.
      indent_str: The string used to indent each line.
    """
    self.indent_str = indent_str
    self.level = level

  def __enter__(self):
    self.level += 1
    return self

  def __exit__(self, type_, value, traceback):
    self.level -= 1

  def __call__(self, string):
    return indent(string, self.level, self.indent_str)


def indent(s, n=1, indent_str="  "):
  """Inserts `n * indent_str` at the start of each non-empty line in `s`."""
  p = n * indent_str
  return "".join((p + l) if l.lstrip() else l for l in s.splitlines(True))


class UniqueOrderedDict(collections.OrderedDict):
  """Subclass of `OrderedDict` that enforces the uniqueness of keys."""

  def __setitem__(self, k, v):
    existing_v = self.get(k)
    if existing_v is None:
      super().__setitem__(k, v)
    elif v != existing_v:
      raise ValueError("Key '{}' already exists.".format(k))


def macro_struct_name(name, suffix=None):
  """Converts mjxmacro struct names, e.g. "MJDATA_POINTERS" to "mjdata"."""
  if suffix is None:
    suffix = _MJXMACRO_SUFFIX
  return name[:-len(suffix)].lower()


def is_macro_pointer(name):
  """Returns True if the mjxmacro struct name contains pointer sizes."""
  return name.endswith(_MJXMACRO_SUFFIX)


def try_coerce_to_num(s, try_types=(int, float)):
  """Try to coerce string to Python numeric type, return None if empty."""
  if not s:
    return None
  for try_type in try_types:
    try:
      return try_type(s.rstrip("UuFf"))
    except (ValueError, AttributeError):
      continue
  return s


def recursive_dict_lookup(key, try_dict, max_depth=10):
  """Recursively map dictionary keys to values."""
  if max_depth < 0:
    raise KeyError("Maximum recursion depth exceeded")
  while key in try_dict:
    key = try_dict[key]
    return recursive_dict_lookup(key, try_dict, max_depth - 1)
  return key


def comment_line(string, width=79, fill_char="-"):
  """Wraps `string` in a padded comment line."""
  return "# {0:{2}^{1}}\n".format(string, width - 2, fill_char)
