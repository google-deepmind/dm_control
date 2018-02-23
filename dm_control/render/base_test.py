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
      self.resize_count = 0

    def _platform_make_current(self):
      self.make_current_count += 1
      self.make_current_thread = threading.current_thread()

    def _platform_resize_framebuffer(self, width, height):
      self.resize_count += 1
      self.resize_thread = threading.current_thread()

    def _platform_free(self):
      self.free_thread = threading.current_thread()

  def setUp(self):
    self.context = ContextBaseTests.ContextMock(WIDTH, HEIGHT)

  def test_init(self):
    self.assertIs(self.context.init_thread, self.context.thread)
    self.assertEqual(self.context._max_width, WIDTH)
    self.assertEqual(self.context._max_height, HEIGHT)

  def test_make_current(self):
    self.assertEqual(self.context.make_current_count, 0)
    self.assertEqual(self.context.resize_count, 0)

    with self.context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(self.context.make_current_count, 1)
    self.assertIs(self.context.make_current_thread, self.context.thread)
    self.assertEqual(self.context.resize_count, 1)
    self.assertIs(self.context.resize_thread, self.context.thread)
    self.assertEqual(self.context._current_width, WIDTH)
    self.assertEqual(self.context._current_height, HEIGHT)

    # Same size, shouldn't trigger a resize.
    with self.context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(self.context.resize_count, 1)
    self.assertIs(self.context.resize_thread, self.context.thread)
    self.assertEqual(self.context._current_width, WIDTH)
    self.assertEqual(self.context._current_height, HEIGHT)

    # New size, should trigger a resize.
    with self.context.make_current(WIDTH // 2, HEIGHT // 2):
      pass
    self.assertEqual(self.context.resize_count, 2)
    self.assertIs(self.context.resize_thread, self.context.thread)
    self.assertEqual(self.context._current_width, WIDTH // 2)
    self.assertEqual(self.context._current_height, HEIGHT // 2)

  def test_thread_sharing(self):
    first_context = ContextBaseTests.ContextMock(
        WIDTH, HEIGHT, executor.PassthroughRenderExecutor)
    second_context = ContextBaseTests.ContextMock(
        WIDTH, HEIGHT, executor.PassthroughRenderExecutor)

    with first_context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(first_context.make_current_count, 1)

    with first_context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(first_context.make_current_count, 1)

    with second_context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(second_context.make_current_count, 1)

    with second_context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(second_context.make_current_count, 1)

    with first_context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(first_context.make_current_count, 2)

    with second_context.make_current(WIDTH, HEIGHT):
      pass
    self.assertEqual(second_context.make_current_count, 2)

  def test_free(self):
    with self.context.make_current(WIDTH, HEIGHT):
      pass

    thread = self.context.thread
    self.assertIn(id(self.context), base._CURRENT_THREAD_FOR_CONTEXT)
    self.assertIn(thread, base._CURRENT_CONTEXT_FOR_THREAD)

    self.context.free()
    self.assertIs(self.context.free_thread, thread)
    self.assertIsNone(self.context.thread)

    self.assertNotIn(id(self.context), base._CURRENT_THREAD_FOR_CONTEXT)
    self.assertNotIn(thread, base._CURRENT_CONTEXT_FOR_THREAD)


if __name__ == '__main__':
  absltest.main()
