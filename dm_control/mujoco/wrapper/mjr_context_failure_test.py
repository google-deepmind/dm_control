# Copyright 2026 The dm_control Authors.
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

"""Tests cleanup of partially initialized MuJoCo rendering contexts."""

from unittest import mock

from absl.testing import absltest
from dm_control.mujoco.wrapper import core


class MjrContextFailureTest(absltest.TestCase):

  def testFreePartiallyInitializedContextDoesNotReleaseGlContext(self):
    gl_context = mock.MagicMock()
    gl_context.terminated = False
    mjr_context = core.MjrContext.__new__(core.MjrContext)
    mjr_context._gl_context = gl_context
    mjr_context._ptr = None
    mjr_context._gl_context_refcounted = False

    mjr_context.free()

    gl_context.decrement_refcount.assert_not_called()
    gl_context.free.assert_not_called()
    self.assertIsNone(mjr_context.ptr)


if __name__ == "__main__":
  absltest.main()
