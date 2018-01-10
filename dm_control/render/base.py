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
import contextlib
import threading

# Internal dependencies.
import six

_ACTIVE_CONTEXT_PARAM = '_active_context'

# A storage for thread local data.
_thread_local_data = threading.local()


class _ContextPolicyManager(object):
  """Manages a context switching policy."""

  def __init__(self):
    """Instance initializer."""
    self._policy = None
    self.enable_debug_mode(False)

  def enable_debug_mode(self, flag):
    """Enables/disables a debug context management policy.

    For details, please see DebugContextPolicy docstring.

    Args:
      flag: A boolean value.
    """
    if flag:
      self._policy = _DebugContextPolicy()
    else:
      self._policy = _OptimizedContextPolicy()

  def activate(self, context, width, height):
    """Forwards the call to policy method that handles context activation.

    Args:
      context: Render context to activate, an instance of ContextBase.
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.
    """
    self._policy.activate(context, width, height)

  def deactivate(self, context):
    """Forwards the call to policy method that handles context deactivation.

    Args:
      context: Render context to deactivate, an instance of ContextBase.
    """
    self._policy.deactivate(context)

  def release_context(self, context):
    """Forwards the call to policy method that handles context tracking.

    Args:
      context: Render context to deactivate, an instance of ContextBase.
    """
    self._policy.release_context(context)


class _OptimizedContextPolicy(object):
  """Context management policy that performs lazy context activation.

  It performs context activations only when the context or the viewport size
  change. If an application uses only a single context with a fixed-size
  viewport, the policy will have it activated only once.

  Moreover, the policy makes sure that each context is activated and then used
  from the same thread of execution.
  """

  def __init__(self):
    """Instance initializer."""
    self._context_stamp = (0, -1, -1)

  def activate(self, context, width, height):
    """Performs a lazy context activation.

    Checks if the context has changed since the last call, and if it has, it
    proceeds with the activation procedure.
    Activation consists of deactivating the previously active context, if any,
    and then activating the new context.

    Args:
      context: Render context to activate, an instance of ContextBase.
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.
    """
    context_stamp = (id(context), width, height)
    if self._context_stamp == context_stamp:
      return
    else:
      if self._active_context:
        self._active_context.deactivate()
      self._active_context = context
      self._context_stamp = context_stamp
      if context:
        context.activate(width, height)

  def deactivate(self, context):
    """Performs a lazy context deactivation.

    Actual deactivation is deferred to the activation procedure.

    Args:
      context: Render context to deactivate, an instance of ContextBase.
    """
    pass

  def release_context(self, context):
    """Stops tracking the specified context, releasing references to it.

    Args:
      context: Render context to deactivate, an instance of ContextBase.
    """
    if self._active_context is context:
      self._active_context = None

  @property
  def _active_context(self):
    value = getattr(_thread_local_data, _ACTIVE_CONTEXT_PARAM, None)
    return value

  @_active_context.setter
  def _active_context(self, value):
    setattr(_thread_local_data, _ACTIVE_CONTEXT_PARAM, value)


class _DebugContextPolicy(object):
  """Context management policy used for debugging rendering problems.

  It always activates and then symmetrically deactivates the rendering context,
  for every 'make_current' call made.
  """

  def activate(self, context, width, height):
    """Activates the specified context.

    Args:
      context: Render context to activate, an instance of ContextBase.
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.
    """
    context.activate(width, height)

  def deactivate(self, context):
    """Deactivates the specified context.

    Args:
      context: Render context to deactivate, an instance of ContextBase.
    """
    context.deactivate()

  def release_context(self, context):
    """The call is ignored by this policy.

    Args:
      context: Render context to deactivate, an instance of ContextBase.
    """
    pass


# A singleton instance of the context policy manager.
policy_manager = _ContextPolicyManager()


@six.add_metaclass(abc.ABCMeta)
class ContextBase(object):
  """Base class for managing OpenGL contexts."""

  def __init__(self):
    """Initializes this context."""

  @abc.abstractmethod
  def activate(self, width, height):
    """Called when entering the `make_current` context manager.

    Args:
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.
    """

  @abc.abstractmethod
  def deactivate(self):
    """Called when exiting the `make_current` context manager."""

  @abc.abstractmethod
  def _free(self):
    """Performs an implementation specific context cleanup."""

  def free(self):
    """Frees resources associated with this context."""
    policy_manager.release_context(self)
    self._free()

  def __del__(self):
    self.free()

  @contextlib.contextmanager
  def make_current(self, width, height):
    """Context manager that makes this Renderer's OpenGL context current.

    Args:
      width: Integer specifying the new framebuffer width in pixels.
      height: Integer specifying the new framebuffer height in pixels.

    Yields:
      None
    """
    policy_manager.activate(self, width, height)
    try:
      yield
    finally:
      policy_manager.deactivate(self)

