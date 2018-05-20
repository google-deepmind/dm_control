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

"""Tests for the base rendering module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
# Internal dependencies.
from absl.testing import absltest
from dm_control.render import base
from dm_control.render import executor

WIDTH = 1024
HEIGHT = 768


class ContextBaseTests(absltest.TestCase):

  class ContextMock(base.ContextBase):

    def _platform_init(self, max_width, max_height):
      self.init_thread = threading.current_thread()
      self.make_current_count = 0
      self.max_width = max_width
      self.max_height = max_height
      self.free_thread = None

    def _platform_make_current(self):
      self.make_current_count += 1
      self.make_current_thread = threading.current_thread()

    def _platform_free(self):
      self.free_thread = threading.current_thread()

  def setUp(self):
    self.context = ContextBaseTests.ContextMock(WIDTH, HEIGHT)

  def test_init(self):
    self.assertIs(self.context.init_thread, self.context.thread)
    self.assertEqual(self.context.max_width, WIDTH)
    self.assertEqual(self.context.max_height, HEIGHT)

  def test_make_current(self):
    self.assertEqual(self.context.make_current_count, 0)

    with self.context.make_current():
      pass
    self.assertEqual(self.context.make_current_count, 1)
    self.assertIs(self.context.make_current_thread, self.context.thread)

    # Already current, shouldn't trigger a call to `_platform_make_current`.
    with self.context.make_current():
      pass
    self.assertEqual(self.context.make_current_count, 1)
    self.assertIs(self.context.make_current_thread, self.context.thread)

  def test_thread_sharing(self):
    first_context = ContextBaseTests.ContextMock(
        WIDTH, HEIGHT, executor.PassthroughRenderExecutor)
    second_context = ContextBaseTests.ContextMock(
        WIDTH, HEIGHT, executor.PassthroughRenderExecutor)

    with first_context.make_current():
      pass
    self.assertEqual(first_context.make_current_count, 1)

    with first_context.make_current():
      pass
    self.assertEqual(first_context.make_current_count, 1)

    with second_context.make_current():
      pass
    self.assertEqual(second_context.make_current_count, 1)

    with second_context.make_current():
      pass
    self.assertEqual(second_context.make_current_count, 1)

    with first_context.make_current():
      pass
    self.assertEqual(first_context.make_current_count, 2)

    with second_context.make_current():
      pass
    self.assertEqual(second_context.make_current_count, 2)

  def test_free(self):
    with self.context.make_current():
      pass

    thread = self.context.thread
    self.assertIn(id(self.context), base._CURRENT_THREAD_FOR_CONTEXT)
    self.assertIn(thread, base._CURRENT_CONTEXT_FOR_THREAD)

    self.context.free()
    self.assertIs(self.context.free_thread, thread)
    self.assertIsNone(self.context.thread)

    self.assertNotIn(id(self.context), base._CURRENT_THREAD_FOR_CONTEXT)
    self.assertNotIn(thread, base._CURRENT_CONTEXT_FOR_THREAD)

  def test_refcounting(self):
    thread = self.context.thread

    self.assertEqual(self.context._refcount, 0)
    self.context.increment_refcount()
    self.assertEqual(self.context._refcount, 1)

    # Context should not be freed yet, since its refcount is still positive.
    self.context.free()
    self.assertIsNone(self.context.free_thread)
    self.assertIs(self.context.thread, thread)

    # Decrement the refcount to zero.
    self.context.decrement_refcount()
    self.assertEqual(self.context._refcount, 0)

    # Now the context can be freed.
    self.context.free()
    self.assertIs(self.context.free_thread, thread)
    self.assertIsNone(self.context.thread)

  def test_del(self):
    self.assertIsNone(self.context.free_thread)
    self.context.__del__()
    self.assertIsNotNone(self.context.free_thread)


if __name__ == '__main__':
  absltest.main()
