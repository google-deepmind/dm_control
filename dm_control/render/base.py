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

"""Base class for OpenGL context handlers.

The module lays foundation for defining various rendering contexts in a uniform
manner.

ContextBase defines a common interface rendering contexts should fulfill. In
addition, it provides a context activation method that can be used in 'with'
statements to ensure symmetrical context activation and deactivation.

The problem of optimizing context swaps falls to ContextPolicyManager and the
accompanying policy classes. OptimizedContextPolicy will attempt to reduce
the number of context swaps, increasing application's performance.
DebugContextPolicy, on the other, hand will rigorously keep activating and
deactivating contexts for each request, providing a reliable framework for
functional tests of the new context implementations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import atexit
import collections
import contextlib
import weakref

from dm_control.render import executor
import six

_CURRENT_CONTEXT_FOR_THREAD = collections.defaultdict(lambda: None)
_CURRENT_THREAD_FOR_CONTEXT = collections.defaultdict(lambda: None)


@six.add_metaclass(abc.ABCMeta)
class ContextBase(object):
  """Base class for managing OpenGL contexts."""

  def __init__(self,
               max_width,
               max_height,
               render_executor_class=executor.RenderExecutor):
    """Initializes this context."""
    self._render_executor = render_executor_class()

    self_weakref = weakref.ref(self)
    def _free_at_exit():
      if self_weakref():
        self_weakref().free()
    atexit.register(_free_at_exit)

    with self._render_executor.execution_context() as ctx:
      ctx.call(self._platform_init, max_width, max_height)

  @property
  def thread(self):
    return self._render_executor.thread

  def _free_on_executor_thread(self):
    if _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread] == id(self):
      del _CURRENT_THREAD_FOR_CONTEXT[id(self)]
      del _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread]
    self._platform_free()

  def free(self):
    """Frees resources associated with this context."""
    self._render_executor.terminate(self._free_on_executor_thread)

  def __del__(self):
    self.free()

  @contextlib.contextmanager
  def make_current(self):
    """Context manager that makes this Renderer's OpenGL context current.

    Yields:
      An object that exposes a `call` method that can be used to call a
      function on the dedicated rendering thread.

    Raises:
      RuntimeError: If this context is already current on another thread.
    """

    with self._render_executor.execution_context() as ctx:
      if _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread] != id(self):
        if _CURRENT_THREAD_FOR_CONTEXT[id(self)]:
          raise RuntimeError(
              'Cannot make context {!r} current on thread {!r}: '
              'this context is already current on another thread {!r}.'
              .format(self, self._render_executor.thread,
                      _CURRENT_THREAD_FOR_CONTEXT[id(self)]))
        else:
          current_context = (
              _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread])
          if current_context:
            del _CURRENT_THREAD_FOR_CONTEXT[current_context]
          _CURRENT_THREAD_FOR_CONTEXT[id(self)] = self._render_executor.thread
          _CURRENT_CONTEXT_FOR_THREAD[self._render_executor.thread] = id(self)
          ctx.call(self._platform_make_current)
      yield ctx

  @abc.abstractmethod
  def _platform_init(self, max_width, max_height):
    """Performs an implementation-specific context initialization."""

  @abc.abstractmethod
  def _platform_make_current(self):
    """Make the OpenGL context current on the executing thread."""

  @abc.abstractmethod
  def _platform_free(self):
    """Performs an implementation-specific context cleanup."""
