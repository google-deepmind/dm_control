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
import mock
import six

WIDTH = 1024
HEIGHT = 768


class ContextBaseTests(absltest.TestCase):

  class ContextMock(base.ContextBase):

    def __init__(self):
      super(ContextBaseTests.ContextMock, self).__init__()

    def activate(self, width, height):
      pass

    def deactivate(self):
      pass

    def _free(self):
      pass

  def setUp(self):
    self.original_manager = base.policy_manager

    base.policy_manager = mock.MagicMock()
    self.context = ContextBaseTests.ContextMock()
    self.context._policy = base.policy_manager

  def tearDown(self):
    base.policy_manager = self.original_manager

  def test_activating_context(self):
    with self.context.make_current(WIDTH, HEIGHT):
      base.policy_manager.activate.assert_called_once_with(
          self.context, WIDTH, HEIGHT)
    base.policy_manager.deactivate.assert_called_once_with(self.context)


class ContextPolicyManagerTests(absltest.TestCase):

  def setUp(self):
    self.context = mock.MagicMock()
    self.policy = mock.MagicMock()
    base.policy_manager._policy = self.policy

  def test_activation(self):
    base.policy_manager.activate(self.context, WIDTH, HEIGHT)
    self.policy.activate.assert_called_once_with(self.context, WIDTH, HEIGHT)

  def test_deactivation(self):
    base.policy_manager.deactivate(self.context)
    self.policy.deactivate.assert_called_once_with(self.context)

  def test_selecting_policy(self):
    base.policy_manager.enable_debug_mode(True)
    self.assertIsInstance(
        base.policy_manager._policy, base._DebugContextPolicy)
    base.policy_manager.enable_debug_mode(False)
    self.assertIsInstance(
        base.policy_manager._policy, base._OptimizedContextPolicy)


class OptimizedContextPolicyTests(absltest.TestCase):

  def setUp(self):
    self.policy = base._OptimizedContextPolicy()

  def test_activating_same_context_multiple_times(self):
    context = mock.MagicMock(spec=base.ContextBase)
    for _ in six.moves.xrange(3):
      self.policy.activate(context, WIDTH, HEIGHT)
      self.policy.deactivate(context)
    context.activate.assert_called_once_with(WIDTH, HEIGHT)
    self.assertEqual(0, context.deactivate.call_count)

  def test_switching_contexts(self):
    contexts = [mock.MagicMock(spec=base.ContextBase)
                for _ in six.moves.xrange(3)]
    for context in contexts:
      self.policy.activate(context, WIDTH, HEIGHT)
      self.policy.deactivate(context)
    self.policy.activate(None, WIDTH, HEIGHT)
    for context in contexts:
      context.activate.assert_called_once_with(WIDTH, HEIGHT)
      context.deactivate.assert_called_once()

  def test_context_are_tracked_separately_for_each_thread(self):
    parent_context = mock.MagicMock(spec=base.ContextBase)
    child_context = mock.MagicMock(spec=base.ContextBase)

    def run():
      # Record the context that was active on this thread prior to activation
      # call.
      self.child_thread_context_before = self.policy._active_context

      # Activate and record the activated context.
      self.policy.activate(child_context, WIDTH, HEIGHT)
      self.child_thread_context_after = self.policy._active_context

    thread = threading.Thread(target=run)

    # Main thread activates 'parent_context'
    self.policy.activate(parent_context, WIDTH, HEIGHT)
    self.assertEqual(parent_context, self.policy._active_context)

    # The child thread activates 'child_context'
    thread.start()
    thread.join()

    # Activation from separate threads shouldn't affect one another
    self.assertIsNone(self.child_thread_context_before)
    self.assertEqual(parent_context, self.policy._active_context)
    self.assertEqual(child_context, self.child_thread_context_after)


class DebugContextPolicyTests(absltest.TestCase):

  def setUp(self):
    self.policy = base._DebugContextPolicy()

  def test_activating_same_context_multiple_times(self):
    context = mock.MagicMock()
    for _ in six.moves.xrange(3):
      self.policy.activate(context, WIDTH, HEIGHT)
      self.policy.deactivate(context)
    self.assertEqual(3, context.activate.call_count)
    self.assertEqual(3, context.deactivate.call_count)

  def test_switching_contexts(self):
    contexts = [mock.MagicMock() for _ in six.moves.xrange(3)]
    for context in contexts:
      self.policy.activate(context, WIDTH, HEIGHT)
      self.policy.deactivate(context)
    for context in contexts:
      context.activate.assert_called_once_with(WIDTH, HEIGHT)
      context.deactivate.assert_called_once()


if __name__ == '__main__':
  absltest.main()
