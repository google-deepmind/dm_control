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

"""pyparsing definitions and helper functions for parsing MuJoCo headers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyparsing as pp
import six

# NB: Don't enable parser memoization (`pp.ParserElement.enablePackrat()`),
#     since this results in a ~6x slowdown.


NONE = "None"
CTYPES_CHAR = "ctypes.c_char"

C_TO_CTYPES = {
    # integers
    "int": "ctypes.c_int",
    "unsigned int": "ctypes.c_uint",
    "char": CTYPES_CHAR,
    "unsigned char": "ctypes.c_ubyte",
    "size_t": "ctypes.c_size_t",
    # floats
    "float": "ctypes.c_float",
    "double": "ctypes.c_double",
    # pointers
    "void": NONE,
}

CTYPES_PTRS = {NONE: "ctypes.c_void_p"}

CTYPES_TO_NUMPY = {
    # integers
    "ctypes.c_int": "np.intc",
    "ctypes.c_uint": "np.uintc",
    "ctypes.c_ubyte": "np.ubyte",
    # floats
    "ctypes.c_float": "np.float32",
    "ctypes.c_double": "np.float64",
}

# Helper functions for constructing recursive parsers.
# ------------------------------------------------------------------------------


def _nested_scopes(opening, closing, body):
  """Constructs a parser for (possibly nested) scopes."""
  scope = pp.Forward()
  scope << pp.Group(  # pylint: disable=expression-not-assigned
      opening +
      pp.ZeroOrMore(body | scope)("members") +
      closing)
  return scope


def _nested_if_else(if_, pred, else_, endif, match_if_true, match_if_false):
  """Constructs a parser for (possibly nested) if...(else)...endif blocks."""
  ifelse = pp.Forward()
  ifelse << pp.Group(  # pylint: disable=expression-not-assigned
      if_ +
      pred("predicate") +
      pp.ZeroOrMore(match_if_true | ifelse)("if_true") +
      pp.Optional(else_ +
                  pp.ZeroOrMore(match_if_false | ifelse)("if_false")) +
      endif)
  return ifelse


# Some common string patterns to suppress.
# ------------------------------------------------------------------------------
(X, LPAREN, RPAREN, LBRACK, RBRACK, LBRACE, RBRACE, SEMI, COMMA, EQUAL, FSLASH,
 BSLASH) = map(pp.Suppress, "X()[]{};,=/\\")
EOL = pp.LineEnd().suppress()

# Comments, continuation.
# ------------------------------------------------------------------------------
COMMENT = pp.Combine(
    pp.Suppress("//") +
    pp.Optional(pp.White()).suppress() +
    pp.SkipTo(pp.LineEnd()))

MULTILINE_COMMENT = pp.delimitedList(
    COMMENT.copy().setWhitespaceChars(" \t"), delim=EOL)

CONTINUATION = (BSLASH + pp.LineEnd()).suppress()

# Preprocessor directives.
# ------------------------------------------------------------------------------
DEFINE = pp.Keyword("#define").suppress()
IFDEF = pp.Keyword("#ifdef").suppress()
IFNDEF = pp.Keyword("#ifndef").suppress()
ELSE = pp.Keyword("#else").suppress()
ENDIF = pp.Keyword("#endif").suppress()

# Variable names, types, literals etc.
# ------------------------------------------------------------------------------
NAME = pp.Word(pp.alphanums + "_")
INT = pp.Word(pp.nums + "UuLl")
FLOAT = pp.Word(pp.nums + ".+-EeFf")
NUMBER = FLOAT | INT

# Dimensions can be of the form `[3]`, `[constant_name]` or `[2*constant_name]`
ARRAY_DIM = pp.Combine(
    LBRACK +
    (INT | NAME) +
    pp.Optional(pp.Literal("*")) +
    pp.Optional(INT | NAME) +
    RBRACK)

PTR = pp.Literal("*")
EXTERN = pp.Keyword("extern")
NATIVE_TYPENAME = pp.MatchFirst(
    [pp.Keyword(n) for n in six.iterkeys(C_TO_CTYPES)])

# Macros.
# ------------------------------------------------------------------------------

HDR_GUARD = DEFINE + "THIRD_PARTY_MUJOCO_HDRS_"

# e.g. "#define mjUSEDOUBLE"
DEF_FLAG = pp.Group(
    DEFINE +
    NAME("name") +
    (COMMENT("comment") | EOL)).ignore(HDR_GUARD)

# e.g. "#define mjMINVAL    1E-14       // minimum value in any denominator"
DEF_CONST = pp.Group(
    DEFINE +
    NAME("name") +
    (NUMBER | NAME)("value") +
    (COMMENT("comment") | EOL))

# e.g. "X( mjtNum*, name_textadr, ntext, 1 )"
XMEMBER = pp.Group(
    X +
    LPAREN +
    (NATIVE_TYPENAME | NAME)("typename") +
    pp.Optional(PTR("ptr")) +
    COMMA +
    NAME("name") +
    COMMA +
    pp.delimitedList((INT | NAME), delim=COMMA)("dims") +
    RPAREN)

XMACRO = pp.Group(
    pp.Optional(COMMENT("comment")) +
    DEFINE +
    NAME("name") +
    CONTINUATION +
    pp.delimitedList(XMEMBER, delim=CONTINUATION)("members"))


# Type/variable declarations.
# ------------------------------------------------------------------------------
TYPEDEF = pp.Keyword("typedef").suppress()
STRUCT = pp.Keyword("struct")
UNION = pp.Keyword("union")
ENUM = pp.Keyword("enum").suppress()

# e.g. "typedef unsigned char mjtByte;      // used for true/false"
TYPE_DECL = pp.Group(
    TYPEDEF +
    pp.Optional(STRUCT) +
    (NATIVE_TYPENAME | NAME)("typename") +
    pp.Optional(PTR("ptr")) +
    NAME("name") +
    SEMI +
    pp.Optional(COMMENT("comment")))

# Declarations of flags/constants/types.
UNCOND_DECL = DEF_FLAG | DEF_CONST | TYPE_DECL

# Declarations inside (possibly nested) #if(n)def... #else... #endif... blocks.
COND_DECL = _nested_if_else(IFDEF, NAME, ELSE, ENDIF, UNCOND_DECL, UNCOND_DECL)
# Note: this doesn't work for '#if defined(FLAG)' blocks

# e.g. "mjtNum gravity[3];              // gravitational acceleration"
STRUCT_MEMBER = pp.Group(
    pp.Optional(STRUCT("struct")) +
    (NATIVE_TYPENAME | NAME)("typename") +
    pp.Optional(PTR("ptr")) +
    NAME("name") +
    pp.ZeroOrMore(ARRAY_DIM)("size") +
    SEMI +
    pp.Optional(COMMENT("comment")))

# Struct declaration within a union (non-nested).
UNION_STRUCT_DECL = pp.Group(
    STRUCT("struct") +
    pp.Optional(NAME("typename")) +
    pp.Optional(COMMENT("comment")) +
    LBRACE +
    pp.OneOrMore(STRUCT_MEMBER)("members") +
    RBRACE +
    pp.Optional(NAME("name")) +
    SEMI)

ANONYMOUS_UNION_DECL = pp.Group(
    pp.Optional(MULTILINE_COMMENT("comment")) +
    UNION("anonymous_union") +
    LBRACE +
    pp.OneOrMore(
        UNION_STRUCT_DECL |
        STRUCT_MEMBER |
        COMMENT.suppress())("members") +
    RBRACE +
    SEMI)

# Multiple (possibly nested) struct declarations.
NESTED_STRUCTS = _nested_scopes(
    opening=(STRUCT +
             pp.Optional(NAME("typename")) +
             pp.Optional(COMMENT("comment")) +
             LBRACE),
    closing=(RBRACE + pp.Optional(NAME("name")) + SEMI),
    body=pp.OneOrMore(
        STRUCT_MEMBER |
        ANONYMOUS_UNION_DECL |
        COMMENT.suppress())("members"))

BIT_LSHIFT = INT("bit_lshift_a") + pp.Suppress("<<") + INT("bit_lshift_b")

ENUM_LINE = pp.Group(
    NAME("name") +
    pp.Optional(EQUAL + (INT("value") ^ BIT_LSHIFT)) +
    pp.Optional(COMMA) +
    pp.Optional(COMMENT("comment")))

ENUM_DECL = pp.Group(
    TYPEDEF +
    ENUM +
    NAME("typename") +
    pp.Optional(COMMENT("comment")) +
    LBRACE +
    pp.OneOrMore(ENUM_LINE | COMMENT.suppress())("members") +
    RBRACE +
    pp.Optional(NAME("name")) +
    SEMI)

# Function declarations.
# ------------------------------------------------------------------------------
MJAPI = pp.Keyword("MJAPI").suppress()
CONST = pp.Keyword("const")
VOID = pp.Group(pp.Keyword("void") + ~PTR).suppress()

ARG = pp.Group(
    pp.Optional(CONST("is_const")) +
    (NATIVE_TYPENAME | NAME)("typename") +
    pp.Optional(PTR("ptr")) +
    NAME("name") +
    pp.Optional(ARRAY_DIM("size")))

RET = pp.Group(
    pp.Optional(CONST("is_const")) +
    (NATIVE_TYPENAME | NAME)("typename") +
    pp.Optional(PTR("ptr")))

FUNCTION_DECL = (
    (VOID | RET("return_value")) +
    NAME("name") +
    LPAREN +
    (VOID | pp.delimitedList(ARG, delim=COMMA)("arguments")) +
    RPAREN +
    SEMI)

MJAPI_FUNCTION_DECL = pp.Group(
    pp.Optional(MULTILINE_COMMENT("comment")) +
    pp.LineStart() +
    MJAPI +
    FUNCTION_DECL)

# e.g.
# // predicate function: set enable/disable based on item category
# typedef int (*mjfItemEnable)(int category, void* data);
FUNCTION_PTR_TYPE_DECL = pp.Group(
    pp.Optional(MULTILINE_COMMENT("comment")) +
    TYPEDEF +
    RET("return_type") +
    LPAREN +
    PTR +
    NAME("typename") +
    RPAREN +
    LPAREN +
    (VOID | pp.delimitedList(ARG, delim=COMMA)("arguments")) +
    RPAREN +
    SEMI)

# Global variables.
# ------------------------------------------------------------------------------

MJAPI_STRING_ARRAY = (
    MJAPI +
    EXTERN +
    CONST +
    pp.Keyword("char") +
    PTR +
    NAME("name") +
    pp.OneOrMore(ARRAY_DIM)("dims") +
    SEMI)

MJAPI_FUNCTION_PTR = MJAPI + EXTERN + NAME("typename") + NAME("name") + SEMI
