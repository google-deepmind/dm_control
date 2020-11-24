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

"""Tests for dm_control.utils.render_executor."""

import threading
import time
import unittest

from absl.testing import absltest
from absl.testing import parameterized
from dm_control._render import executor
import mock
from six.moves import range


def enforce_timeout(timeout):
  def wrap(test_func):
    def wrapped_test(self, *args, **kwargs):
      thread = threading.Thread(
          target=test_func, args=((self,) + args), kwargs=kwargs)
      thread.daemon = True
      thread.start()
      thread.join(timeout=timeout)
      self.assertFalse(
          thread.is_alive(),
          msg='Test timed out after {} seconds.'.format(timeout))
    return wrapped_test
  return wrap


class RenderExecutorTest(parameterized.TestCase):

  def _make_executor(self, executor_type):
    if (executor_type == executor.NativeMutexOffloadingRenderExecutor and
        executor_type is None):
      raise unittest.SkipTest(
          'NativeMutexOffloadingRenderExecutor is not available.')
    else:
      return executor_type()

  def test_passthrough_executor_thread(self):
    render_executor = self._make_executor(executor.PassthroughRenderExecutor)
    self.assertIs(render_executor.thread, threading.current_thread())
    render_executor.terminate()

  @parameterized.parameters(executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_offloading_executor_thread(self, executor_type):
    render_executor = self._make_executor(executor_type)
    self.assertIsNot(render_executor.thread, threading.current_thread())
    render_executor.terminate()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_call_on_correct_thread(self, executor_type):
    render_executor = self._make_executor(executor_type)
    with render_executor.execution_context() as ctx:
      actual_executed_thread = ctx.call(threading.current_thread)
    self.assertIs(actual_executed_thread, render_executor.thread)
    render_executor.terminate()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_multithreaded(self, executor_type):
    render_executor = self._make_executor(executor_type)
    list_length = 5
    shared_list = [None] * list_length

    def fill_list(thread_idx):
      def assign_value(i):
        shared_list[i] = thread_idx
      for _ in range(1000):
        with render_executor.execution_context() as ctx:
          for i in range(list_length):
            ctx.call(assign_value, i)
          # Other threads should be prevented from calling `assign_value` while
          # this thread is inside the `execution_context`.
          self.assertEqual(shared_list, [thread_idx] * list_length)

    threads = [threading.Thread(target=fill_list, args=(i,)) for i in range(9)]
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    render_executor.terminate()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_exception(self, executor_type):
    render_executor = self._make_executor(executor_type)
    message = 'fake error'
    def raise_value_error():
      raise ValueError(message)
    with render_executor.execution_context() as ctx:
      with self.assertRaisesWithLiteralMatch(ValueError, message):
        ctx.call(raise_value_error)
    render_executor.terminate()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_terminate(self, executor_type):
    render_executor = self._make_executor(executor_type)
    cleanup = mock.MagicMock()
    render_executor.terminate(cleanup)
    cleanup.assert_called_once_with()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_call_outside_of_context(self, executor_type):
    render_executor = self._make_executor(executor_type)
    func = mock.MagicMock()
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, executor.render_executor._NOT_IN_CONTEXT):
      render_executor.call(func)
    # Also test that the locked flag is properly cleared when leaving a context.
    with render_executor.execution_context():
      render_executor.call(lambda: None)
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, executor.render_executor._NOT_IN_CONTEXT):
      render_executor.call(func)
    func.assert_not_called()
    render_executor.terminate()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_call_after_terminate(self, executor_type):
    render_executor = self._make_executor(executor_type)
    render_executor.terminate()
    func = mock.MagicMock()
    with self.assertRaisesWithLiteralMatch(
        RuntimeError, executor.render_executor._ALREADY_TERMINATED):
      with render_executor.execution_context() as ctx:
        ctx.call(func)
    func.assert_not_called()

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  def test_locking(self, executor_type):
    render_executor = self._make_executor(executor_type)
    other_thread_context_entered = threading.Condition()
    other_thread_context_done = [False]
    def other_thread_func():
      with render_executor.execution_context():
        with other_thread_context_entered:
          other_thread_context_entered.notify()
        time.sleep(1)
        other_thread_context_done[0] = True
    other_thread = threading.Thread(target=other_thread_func)
    with other_thread_context_entered:
      other_thread.start()
      other_thread_context_entered.wait()
    with render_executor.execution_context():
      self.assertTrue(
          other_thread_context_done[0],
          msg=('Main thread should not be able to enter the execution context '
               'until the other thread is done.'))

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  @enforce_timeout(timeout=5.)
  def test_reentrant_locking(self, executor_type):
    render_executor = self._make_executor(executor_type)
    def triple_lock(render_executor):
      with render_executor.execution_context():
        with render_executor.execution_context():
          with render_executor.execution_context():
            pass
    triple_lock(render_executor)

  @parameterized.parameters(executor.PassthroughRenderExecutor,
                            executor.OffloadingRenderExecutor,
                            executor.NativeMutexOffloadingRenderExecutor)
  @enforce_timeout(timeout=5.)
  def test_no_deadlock_in_callbacks(self, executor_type):
    render_executor = self._make_executor(executor_type)
    # This test times out in the event of a deadlock.
    def callback():
      with render_executor.execution_context() as ctx:
        ctx.call(lambda: None)
    with render_executor.execution_context() as ctx:
      ctx.call(callback)

if __name__ == '__main__':
  absltest.main()
