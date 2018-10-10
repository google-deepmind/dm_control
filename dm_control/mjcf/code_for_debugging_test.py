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

"""Constructs models for debugging_test.py.

The purpose of this file is to provide "golden" source line numbers for test
cases in debugging_test.py. When this module is loaded, it inspects its own
source code to look for lines that begin with `# !!LINE_REF`, and stores the
following line number in a dict. This allows test cases to look up the line
number by name, rather than brittly hard-coding in the line number.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import re

from dm_control import mjcf

SourceLine = collections.namedtuple(
    'SourceLine', ('line_number', 'text'))

LINE_REF = {}


def make_valid_model():
  # !!LINE_REF make_valid_model.mjcf_model
  mjcf_model = mjcf.RootElement()
  # !!LINE_REF make_valid_model.my_body
  my_body = mjcf_model.worldbody.add('body', name='my_body')
  my_body.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])
  # !!LINE_REF make_valid_model.my_joint
  my_joint = my_body.add('joint', name='my_joint', type='hinge')
  # !!LINE_REF make_valid_model.my_actuator
  mjcf_model.actuator.add('velocity', name='my_actuator', joint=my_joint)
  return mjcf_model


def make_broken_model():
  # !!LINE_REF make_broken_model.mjcf_model
  mjcf_model = mjcf.RootElement()
  # !!LINE_REF make_broken_model.my_body
  my_body = mjcf_model.worldbody.add('body', name='my_body')
  my_body.add('inertial', mass=1, pos=[0, 0, 0], diaginertia=[1, 1, 1])
  # !!LINE_REF make_broken_model.my_joint
  my_body.add('joint', name='my_joint', type='hinge')
  # !!LINE_REF make_broken_model.my_actuator
  mjcf_model.actuator.add('velocity', name='my_actuator', joint='invalid_joint')
  return mjcf_model


def break_valid_model(mjcf_model):
  # !!LINE_REF break_valid_model.my_actuator.joint
  mjcf_model.find('actuator', 'my_actuator').joint = 'invalid_joint'
  return mjcf_model


def _parse_line_refs():
  line_ref_pattern = re.compile(r'\s*# !!LINE_REF\s*([^\s]+)')
  filename, _ = os.path.splitext(__file__)  # __file__ can be `.pyc`.
  with open(filename + '.py') as f:
    src = f.read()
  src_lines = src.split('\n')
  for line_number, line in enumerate(src_lines):
    match = line_ref_pattern.match(line)
    if match:
      LINE_REF[match.group(1)] = SourceLine(
          line_number + 2, src_lines[line_number + 1].strip())

_parse_line_refs()
