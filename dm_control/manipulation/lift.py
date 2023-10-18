# Copyright 2019 The dm_control Authors.
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

"""Tasks where the goal is to elevate a prop."""

import collections
import io
import os

from absl import app
from absl import flags
from absl import logging
from dm_control.autowrap import binding_generator
from dm_control.autowrap import codegen_util

_MUJOCO_HEADER_PATHS = {
    "mjmodel": "/path/to/mjmodel.h",
    "mjxmacro": "/path/to/mjxmacro.h",
}

FLAGS = flags.FLAGS

flags.DEFINE_string("output_dir", None,
                    "Path to output directory for wrapper source files.")


def main(unused_argv):
  """Automatically generates ctypes Python bindings for MuJoCo.

  Parses the following MuJoCo header files:

    mjdata.h
    mjmodel.h
    mjrender.h
    mjui.h
    mjvisualize.h
    mjxmacro.h
    mujoco.h;

  generates the following Python source files:

    constants.py:  constants
    enums.py:      enums
    sizes.py:      size information for dynamically-shaped arrays
  """

  binding_generator = binding_generator.BindingGenerator()

  # Parse enums.
  for header_name, header_path in _MUJOCO_HEADER_PATHS.items():
    with io.open(header_path, "r", errors="ignore") as f:
      binding_generator.parse_enums(f.read())

  # Parse constants and type declarations.
  for header_name, header_path in _MUJOCO_HEADER_PATHS.items():
    with io.open(header_path, "r", errors="ignore") as f:
      binding_generator.parse_consts_typedefs(f.read())

  # Create the output directory if it doesn't already exist.
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)

  # Generate Python source files and write them to the output directory.
  binding_generator.write_consts(os.path.join(FLAGS.output_dir, "constants.py"))
  binding_generator.write_enums(os.path.join(FLAGS.output_dir, "enums.py"))
  binding_generator.write_index_dict(os.path.join(FLAGS.output_dir, "sizes.py"))


if __name__ == "__main__":
  flags.mark_flag_as_required("output_dir")
  app.run(main)
