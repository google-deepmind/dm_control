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

"""Parses MuJoCo header files and generates Python bindings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pprint
import textwrap

from absl import logging
from dm_control.autowrap import c_declarations
from dm_control.autowrap import codegen_util
from dm_control.autowrap import header_parsing
import pyparsing
import six

# Absolute path to the top-level module.
_MODULE = "dm_control.mujoco.wrapper"

# Imports used in all generated source files.
_BOILERPLATE_IMPORTS = [
    "from __future__ import absolute_import",
    "from __future__ import division",
    "from __future__ import print_function\n",
]


class Error(Exception):
  pass


class BindingGenerator(object):
  """Parses declarations from MuJoCo headers and generates Python bindings."""

  def __init__(self,
               enums_dict=None,
               consts_dict=None,
               typedefs_dict=None,
               hints_dict=None,
               structs_dict=None,
               funcs_dict=None,
               strings_dict=None,
               func_ptrs_dict=None,
               index_dict=None):
    """Constructs a new HeaderParser instance.

    The optional arguments listed below can be used to passing in dict-like
    objects specifying pre-defined declarations. By default empty
    UniqueOrderedDicts will be instantiated and then populated according to the
    contents of the headers.

    Args:
      enums_dict: Nested mappings from {enum_name: {member_name: value}}.
      consts_dict: Mapping from {const_name: value}.
      typedefs_dict: Mapping from {type_name: ctypes_typename}.
      hints_dict: Mapping from {var_name: shape_tuple}.
      structs_dict: Mapping from {struct_name: Struct_instance}.
      funcs_dict: Mapping from {func_name: Function_instance}.
      strings_dict: Mapping from {var_name: StaticStringArray_instance}.
      func_ptrs_dict: Mapping from {var_name: FunctionPtr_instance}.
      index_dict: Mapping from {lowercase_struct_name: {var_name: shape_tuple}}.
    """
    self.enums_dict = (enums_dict if enums_dict is not None
                       else codegen_util.UniqueOrderedDict())
    self.consts_dict = (consts_dict if consts_dict is not None
                        else codegen_util.UniqueOrderedDict())
    self.typedefs_dict = (typedefs_dict if typedefs_dict is not None
                          else codegen_util.UniqueOrderedDict())
    self.hints_dict = (hints_dict if hints_dict is not None
                       else codegen_util.UniqueOrderedDict())
    self.structs_dict = (structs_dict if structs_dict is not None
                         else codegen_util.UniqueOrderedDict())
    self.funcs_dict = (funcs_dict if funcs_dict is not None
                       else codegen_util.UniqueOrderedDict())
    self.strings_dict = (strings_dict if strings_dict is not None
                         else codegen_util.UniqueOrderedDict())
    self.func_ptrs_dict = (func_ptrs_dict if func_ptrs_dict is not None
                           else codegen_util.UniqueOrderedDict())
    self.index_dict = (index_dict if index_dict is not None
                       else codegen_util.UniqueOrderedDict())

  def get_consts_and_enums(self):
    consts_and_enums = self.consts_dict.copy()
    for enum in six.itervalues(self.enums_dict):
      consts_and_enums.update(enum)
    return consts_and_enums

  def resolve_size(self, old_size):
    """Resolves an array size identifier.

    The following conversions will be attempted:

      * If `old_size` is an integer it will be returned as-is.
      * If `old_size` is a string of the form `"3"` it will be cast to an int.
      * If `old_size` is a string in `self.consts_dict` then the value of the
        constant will be returned.
      * If `old_size` is a string of the form `"3*constant_name"` then the
        result of `3*constant_value` will be returned.
      * If `old_size` is a string that does not specify an int constant and
        cannot be cast to an int (e.g. an identifier for a dynamic dimension,
        such as `"ncontact"`) then it will be returned as-is.

    Args:
      old_size: An int or string.

    Returns:
      An int or string.
    """
    if isinstance(old_size, int):
      return old_size  # If it's already an int then there's nothing left to do
    elif "*" in old_size:
      # If it's a string specifying a product (such as "2*mjMAXLINEPNT"),
      # recursively resolve the components to ints and calculate the result.
      size = 1
      for part in old_size.split("*"):
        dim = self.resolve_size(part)
        assert isinstance(dim, int)
        size *= dim
      return size
    else:
      # Recursively dereference any sizes declared in header macros
      size = codegen_util.recursive_dict_lookup(old_size,
                                                self.get_consts_and_enums())
      # Try to coerce the result to an int, return a string if this fails
      return codegen_util.try_coerce_to_num(size, try_types=(int,))

  def get_shape_tuple(self, old_size, squeeze=False):
    """Generates a shape tuple from parser results.

    Args:
      old_size: Either a `pyparsing.ParseResults`, or a valid int or string
       input to `self.resolve_size` (see method docstring for further details).
      squeeze: If True, any dimensions that are statically defined as 1 will be
        removed from the shape tuple.

    Returns:
      A shape tuple containing ints for dimensions that are statically defined,
      and string size identifiers for dimensions that can only be determined at
      runtime.
    """
    if isinstance(old_size, pyparsing.ParseResults):
      # For multi-dimensional arrays, convert each dimension separately
      shape = tuple(self.resolve_size(dim) for dim in old_size)
    else:
      shape = (self.resolve_size(old_size),)
    if squeeze:
      shape = tuple(d for d in shape if d != 1)  # Remove singleton dimensions
    return shape

  def resolve_typename(self, old_ctypes_typename):
    """Gets a qualified ctypes typename from typedefs_dict and C_TO_CTYPES."""

    # Recursively dereference any typenames declared in self.typedefs_dict
    new_ctypes_typename = codegen_util.recursive_dict_lookup(
        old_ctypes_typename, self.typedefs_dict)

    # Try to convert to a ctypes native typename
    new_ctypes_typename = header_parsing.C_TO_CTYPES.get(
        new_ctypes_typename, new_ctypes_typename)

    if new_ctypes_typename == old_ctypes_typename:
      logging.warn("Could not resolve typename '%s'", old_ctypes_typename)

    return new_ctypes_typename

  def get_type_from_token(self, token, parent=None):
    """Accepts a token returned by a parser, returns a subclass of CDeclBase."""

    comment = codegen_util.mangle_comment(token.comment)
    is_const = token.is_const == "const"

    # An anonymous union declaration
    if token.anonymous_union:
      if not parent and parent.name:
        raise Error(
            "Anonymous unions must be members of a named struct or union.")

      # Generate a name based on the name of the parent.
      name = codegen_util.mangle_varname(parent.name + "_anon_union")

      members = codegen_util.UniqueOrderedDict()
      sub_structs = codegen_util.UniqueOrderedDict()
      out = c_declarations.AnonymousUnion(
          name, members, sub_structs, comment, parent)

      # Add members
      for sub_token in token.members:

        # Recurse into nested structs
        member = self.get_type_from_token(sub_token, parent=out)
        out.members[member.name] = member

        # Nested sub-structures need special treatment
        if isinstance(member, c_declarations.Struct):
          out.sub_structs[member.name] = member

      # Add to dict of unions
      self.structs_dict[out.ctypes_typename] = out

    # A struct declaration
    elif token.members:

      name = token.name

      # If the name is empty, see if there is a type declaration that matches
      # this struct's typename
      if not name:
        for k, v in six.iteritems(self.typedefs_dict):
          if v == token.typename:
            name = k

      # Anonymous structs need a dummy typename
      typename = token.typename
      if not typename:
        if parent:
          typename = token.name
        else:
          raise Error(
              "Anonymous structs that aren't members of a named struct are not "
              "supported (name = '{token.name}').".format(token=token))

      # Mangle the name if it contains any protected keywords
      name = codegen_util.mangle_varname(name)

      members = codegen_util.UniqueOrderedDict()
      sub_structs = codegen_util.UniqueOrderedDict()
      out = c_declarations.Struct(name, typename, members, sub_structs, comment,
                                  parent, is_const)

      # Map the old typename to the mangled typename in typedefs_dict
      self.typedefs_dict[typename] = out.ctypes_typename

      # Add members
      for sub_token in token.members:

        # Recurse into nested structs
        member = self.get_type_from_token(sub_token, parent=out)
        out.members[member.name] = member

        # Nested sub-structures need special treatment
        if isinstance(member, c_declarations.Struct):
          out.sub_structs[member.name] = member

      # Add to dict of structs
      self.structs_dict[out.ctypes_typename] = out

    else:

      name = codegen_util.mangle_varname(token.name)
      typename = self.resolve_typename(token.typename)

      # 1D array with size defined at compile time
      if token.size:
        shape = self.get_shape_tuple(token.size)
        if typename in {header_parsing.NONE, header_parsing.CTYPES_CHAR}:
          out = c_declarations.StaticPtrArray(
              name, typename, shape, comment, parent, is_const)
        else:
          out = c_declarations.StaticNDArray(
              name, typename, shape, comment, parent, is_const)

      elif token.ptr:

        # Pointer to a numpy-compatible type, could be an array or a scalar
        if typename in header_parsing.CTYPES_TO_NUMPY:

          # Multidimensional array (one or more dimensions might be undefined)
          if name in self.hints_dict:

            # Dynamically-sized dimensions have string identifiers
            shape = self.hints_dict[name]
            if any(isinstance(d, six.string_types) for d in shape):
              out = c_declarations.DynamicNDArray(name, typename, shape,
                                                  comment, parent, is_const)
            else:
              out = c_declarations.StaticNDArray(name, typename, shape, comment,
                                                 parent, is_const)

          # This must be a pointer to a scalar primitive
          else:
            out = c_declarations.ScalarPrimitivePtr(name, typename, comment,
                                                    parent, is_const)

        # Pointer to struct or other arbitrary type
        else:
          out = c_declarations.ScalarPrimitivePtr(name, typename, comment,
                                                  parent, is_const)

      # A struct we've already encountered
      elif typename in self.structs_dict:
        s = self.structs_dict[typename]
        out = c_declarations.Struct(name, s.typename, s.members, s.sub_structs,
                                    comment, parent)

      # Presumably this is a scalar primitive
      else:
        out = c_declarations.ScalarPrimitive(name, typename, comment, parent,
                                             is_const)

    return out

  # Parsing functions.
  # ----------------------------------------------------------------------------

  def parse_hints(self, xmacro_src):
    """Parses mjxmacro.h, update self.hints_dict."""
    parser = header_parsing.XMACRO
    for tokens, _, _ in parser.scanString(xmacro_src):
      for xmacro in tokens:
        for member in xmacro.members:
          # "Squeeze out" singleton dimensions.
          shape = self.get_shape_tuple(member.dims, squeeze=True)
          self.hints_dict.update({member.name: shape})

          if codegen_util.is_macro_pointer(xmacro.name):
            struct_name = codegen_util.macro_struct_name(xmacro.name)
            if struct_name not in self.index_dict:
              self.index_dict[struct_name] = {}

            self.index_dict[struct_name].update({member.name: shape})

  def parse_enums(self, src):
    """Parses mj*.h, update self.enums_dict."""
    parser = header_parsing.ENUM_DECL
    for tokens, _, _ in parser.scanString(src):
      for enum in tokens:
        members = codegen_util.UniqueOrderedDict()
        value = 0
        for member in enum.members:
          # Leftward bitshift
          if member.bit_lshift_a:
            value = int(member.bit_lshift_a) << int(member.bit_lshift_b)
          # Assignment
          elif member.value:
            value = int(member.value)
          # Implicit count
          else:
            value += 1
          members.update({member.name: value})
        self.enums_dict.update({enum.name: members})

  def parse_consts_typedefs(self, src):
    """Updates self.consts_dict, self.typedefs_dict."""
    parser = (header_parsing.COND_DECL |
              header_parsing.UNCOND_DECL |
              header_parsing.FUNCTION_PTR_TYPE_DECL)
    for tokens, _, _ in parser.scanString(src):
      self.recurse_into_conditionals(tokens)

  def recurse_into_conditionals(self, tokens):
    """Called recursively within nested #if(n)def... #else... #endif blocks."""
    for token in tokens:
      # Another nested conditional block
      if token.predicate:
        if (token.predicate in self.get_consts_and_enums()
            and self.get_consts_and_enums()[token.predicate]):
          self.recurse_into_conditionals(token.if_true)
        else:
          self.recurse_into_conditionals(token.if_false)
      # One or more declarations
      else:
        # A type declaration for a function pointer.
        if token.arguments:
          self.typedefs_dict.update(
              {token.typename: header_parsing.CTYPES_FUNCTION_PTR})
        elif token.typename:
          self.typedefs_dict.update({token.name: token.typename})
        elif token.value:
          value = codegen_util.try_coerce_to_num(token.value)
          # Avoid adding function aliases.
          if isinstance(value, six.string_types):
            continue
          else:
            self.consts_dict.update({token.name: value})
        else:
          self.consts_dict.update({token.name: True})

  def parse_structs(self, src):
    """Updates self.structs_dict."""
    parser = header_parsing.NESTED_STRUCTS
    for tokens, _, _ in parser.scanString(src):
      for token in tokens:
        self.get_type_from_token(token)

  def parse_functions(self, src):
    """Updates self.funcs_dict."""
    parser = header_parsing.MJAPI_FUNCTION_DECL
    for tokens, _, _ in parser.scanString(src):
      for token in tokens:
        name = codegen_util.mangle_varname(token.name)
        comment = codegen_util.mangle_comment(token.comment)
        if token.arguments:
          args = codegen_util.UniqueOrderedDict()
          for arg in token.arguments:
            a = self.get_type_from_token(arg)
            args[a.name] = a
        else:
          args = None
        if token.return_value:
          ret_val = self.get_type_from_token(token.return_value)
        else:
          ret_val = None
        func = c_declarations.Function(name, args, ret_val, comment)
        self.funcs_dict[func.name] = func

  def parse_global_strings(self, src):
    """Updates self.strings_dict."""
    parser = header_parsing.MJAPI_STRING_ARRAY
    for token, _, _ in parser.scanString(src):
      name = codegen_util.mangle_varname(token.name)
      shape = self.get_shape_tuple(token.dims)
      self.strings_dict[name] = c_declarations.StaticStringArray(
          name, shape, symbol_name=token.name)

  def parse_function_pointers(self, src):
    """Updates self.func_ptrs_dict."""
    parser = header_parsing.MJAPI_FUNCTION_PTR
    for token, _, _ in parser.scanString(src):
      name = codegen_util.mangle_varname(token.name)
      self.func_ptrs_dict[name] = c_declarations.FunctionPtr(
          name, symbol_name=token.name)

  # Code generation methods
  # ----------------------------------------------------------------------------

  def make_header(self, imports=()):
    """Returns a header string for an auto-generated Python source file."""
    docstring = textwrap.dedent("""
    \"\"\"Automatically generated by {scriptname:}.

    MuJoCo header version: {mujoco_version:}
    \"\"\"
    """.format(scriptname=os.path.split(__file__)[-1],
               mujoco_version=self.consts_dict["mjVERSION_HEADER"]))
    docstring = docstring[1:]  # Strip the leading line break.
    return "\n".join(
        [docstring] + _BOILERPLATE_IMPORTS + list(imports) + ["\n"])

  def write_consts(self, fname):
    """Write constants."""
    imports = [
        "# pylint: disable=invalid-name",
    ]
    with open(fname, "w") as f:
      f.write(self.make_header(imports))
      f.write(codegen_util.comment_line("Constants") + "\n")
      for name, value in six.iteritems(self.consts_dict):
        f.write("{0} = {1}\n".format(name, value))
      f.write("\n" + codegen_util.comment_line("End of generated code"))

  def write_enums(self, fname):
    """Write enum definitions."""
    with open(fname, "w") as f:
      imports = [
          "import collections",
          "# pylint: disable=invalid-name",
          "# pylint: disable=line-too-long",
      ]
      f.write(self.make_header(imports))
      f.write(codegen_util.comment_line("Enums"))
      for enum_name, members in six.iteritems(self.enums_dict):
        fields = ["\"{}\"".format(name) for name in six.iterkeys(members)]
        values = [str(value) for value in six.itervalues(members)]
        s = textwrap.dedent("""
        {0} = collections.namedtuple(
            "{0}",
            [{1}]
        )({2})
        """).format(enum_name, ",\n     ".join(fields), ", ".join(values))
        f.write(s)
      f.write("\n" + codegen_util.comment_line("End of generated code"))

  def write_types(self, fname):
    """Write ctypes struct declarations."""
    imports = [
        "import ctypes",
    ]
    with open(fname, "w") as f:
      f.write(self.make_header(imports))
      f.write(codegen_util.comment_line("ctypes struct and union declarations"))
      for struct in six.itervalues(self.structs_dict):
        f.write("\n" + struct.ctypes_decl)
      f.write("\n" + codegen_util.comment_line("End of generated code"))

  def write_wrappers(self, fname):
    """Write wrapper classes for ctypes structs."""
    with open(fname, "w") as f:
      imports = [
          "import ctypes",
          "# pylint: disable=undefined-variable",
          "# pylint: disable=wildcard-import",
          "from {} import util".format(_MODULE),
          "from {}.mjbindings.types import *".format(_MODULE),
      ]
      f.write(self.make_header(imports))
      f.write(codegen_util.comment_line("Low-level wrapper classes"))
      for struct_or_union in six.itervalues(self.structs_dict):
        if isinstance(struct_or_union, c_declarations.Struct):
          f.write("\n" + struct_or_union.wrapper_class)
      f.write("\n" + codegen_util.comment_line("End of generated code"))

  def write_funcs_and_globals(self, fname):
    """Write ctypes declarations for functions and global data."""
    imports = [
        "import collections",
        "import ctypes",
        "# pylint: disable=undefined-variable",
        "# pylint: disable=wildcard-import",
        "from {} import util".format(_MODULE),
        "from {}.mjbindings.types import *".format(_MODULE),
        "import numpy as np",
        "# pylint: disable=line-too-long",
        "# pylint: disable=invalid-name",
        "# common_typos_disable",
    ]
    with open(fname, "w") as f:
      f.write(self.make_header(imports))
      f.write("mjlib = util.get_mjlib()\n")

      f.write("\n" + codegen_util.comment_line("ctypes function declarations"))
      for function in six.itervalues(self.funcs_dict):
        f.write("\n" + function.ctypes_func_decl(cdll_name="mjlib"))

      # Only require strings for UI purposes.
      f.write("\n" + codegen_util.comment_line("String arrays") + "\n")
      for string_arr in six.itervalues(self.strings_dict):
        f.write(string_arr.ctypes_var_decl(cdll_name="mjlib"))

      f.write("\n" + codegen_util.comment_line("Function pointers"))

      fields = [repr(name) for name in self.func_ptrs_dict.keys()]
      values = [func_ptr.ctypes_var_decl(cdll_name="mjlib")
                for func_ptr in self.func_ptrs_dict.values()]
      f.write(textwrap.dedent("""
        function_pointers = collections.namedtuple(
            'FunctionPointers',
            [{0}]
        )({1})
        """).format(",\n     ".join(fields), ",\n  ".join(values)))

      f.write("\n" + codegen_util.comment_line("End of generated code"))

  def write_index_dict(self, fname):
    """Write file containing array shape information for indexing."""
    pp = pprint.PrettyPrinter()
    output_string = pp.pformat(dict(self.index_dict))
    indent = codegen_util.Indenter()
    imports = [
        "# pylint: disable=bad-continuation",
        "# pylint: disable=line-too-long",
    ]
    with open(fname, "w") as f:
      f.write(self.make_header(imports))
      f.write("array_sizes = (\n")
      with indent:
        f.write(output_string)
      f.write("\n)")
      f.write("\n" + codegen_util.comment_line("End of generated code"))
