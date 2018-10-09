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

"""Utility function for evaluating callables or constants."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def evaluate(x, *args, **kwargs):
  """Evaluates a callable or constant value.

  Args:
    x: Either a callable or a constant value.
    *args: Positional arguments passed to `x` if `x` is callable.
    **kwargs: Keyword arguments passed to `x` if `x` is callable.

  Returns:
    Either the result of calling `x` if `x` is callable or else `x`.
  """
  return x(*args, **kwargs) if callable(x) else x
