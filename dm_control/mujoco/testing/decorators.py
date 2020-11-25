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

"""Decorators used in MuJoCo tests."""

import functools
import sys
import threading
import six


def run_threaded(num_threads=4, calls_per_thread=10):
  """A decorator that executes the same test repeatedly in multiple threads.

  Note: `setUp` and `tearDown` methods will only be called once from the main
        thread, so all thread-local setup must be done within the test method.

  Args:
    num_threads: Number of concurrent threads to spawn. If None then the wrapped
      method will be executed in the main thread instead.
    calls_per_thread: Number of times each thread should call the test method.
  Returns:
    Decorated test method.
  """
  def decorator(test_method):
    """Decorator around the test method."""
    @functools.wraps(test_method)  # Needed for `named_parameters` to work.
    def decorated_method(self, *args, **kwargs):
      """Actual method this factory will return."""
      exceptions = []
      def worker():
        try:
          for _ in range(calls_per_thread):
            test_method(self, *args, **kwargs)
        except:  # pylint: disable=bare-except
          # Appending to Python list is thread-safe:
          # http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
          exceptions.append(sys.exc_info())
      if num_threads is not None:
        threads = [threading.Thread(target=worker, name='thread_{}'.format(i))
                   for i in range(num_threads)]
        for thread in threads:
          thread.start()
        for thread in threads:
          thread.join()
      else:
        worker()
      for exc_class, old_exc, tb in exceptions:
        six.reraise(exc_class, old_exc, tb)
    return decorated_method
  return decorator
