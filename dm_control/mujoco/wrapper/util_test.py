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

"""Tests for util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import resource

# Internal dependencies.

from absl.testing import absltest

from dm_control.mujoco.wrapper import core
from dm_control.mujoco.wrapper import util

from six.moves import range

_NUM_CALLS = 10000
_RSS_GROWTH_TOLERANCE = 300  # Bytes


class UtilTest(absltest.TestCase):

  def test_buf_to_npy_no_memory_leak(self):
    """Ensures we can call buf_to_npy without leaking memory."""
    model = core.MjModel.from_xml_string("<mujoco/>")
    src = model._ptr.contents.name_geomadr
    shape = (model.ngeom,)

    # This uses high water marks to find memory leaks in native code.
    old_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(_NUM_CALLS):
      buf = util.buf_to_npy(src, shape)
    del buf
    new_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    growth = new_max - old_max

    self.assertLessEqual(
        growth, _RSS_GROWTH_TOLERANCE,
        msg=("The Resident Set Size (RSS) of this process grew by {} bytes, "
             "exceeding the tolerance of {} bytes."
             .format(growth, _RSS_GROWTH_TOLERANCE)))

if __name__ == "__main__":
  absltest.main()
