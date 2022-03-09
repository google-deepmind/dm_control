# Copyright 2017-2018 The dm_control Authors.
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

"""Various helper functions and classes."""

import functools
import sys
import mujoco
import numpy as np

# Environment variable that can be used to override the default path to the
# MuJoCo shared library.
ENV_MJLIB_PATH = "MJLIB_PATH"

DEFAULT_ENCODING = sys.getdefaultencoding()


def to_binary_string(s):
  """Convert text string to binary."""
  if isinstance(s, bytes):
    return s
  return s.encode(DEFAULT_ENCODING)


def to_native_string(s):
  """Convert a text or binary string to the native string format."""
  if isinstance(s, bytes):
    return s.decode(DEFAULT_ENCODING)
  else:
    return s


def get_mjlib():
  return mujoco


@functools.wraps(np.ctypeslib.ndpointer)
def ndptr(*args, **kwargs):
  """Wraps `np.ctypeslib.ndpointer` to allow passing None for NULL pointers."""
  base = np.ctypeslib.ndpointer(*args, **kwargs)

  def from_param(_, obj):
    if obj is None:
      return obj
    else:
      return base.from_param(obj)

  return type(base.__name__, (base,), {"from_param": classmethod(from_param)})
