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

"""Utilities for handling nested structures of callables or constants."""

import tree


def evaluate(structure, *args, **kwargs):
  """Evaluates a arbitrarily nested structure of callables or constant values.

  Args:
    structure: An arbitrarily nested structure of callables or constant values.
      By "structures", we mean lists, tuples, namedtuples, or dicts.
    *args: Positional arguments passed to each callable in `structure`.
    **kwargs: Keyword arguments passed to each callable in `structure`.

  Returns:
    The same nested structure, with each callable replaced by the value returned
    by calling it.
  """
  return tree.map_structure(
      lambda x: x(*args, **kwargs) if callable(x) else x, structure)
