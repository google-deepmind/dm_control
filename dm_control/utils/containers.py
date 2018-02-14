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

"""Container classes used in control domains."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections


class TaggedTasks(collections.Mapping):
  """Maps task names to their corresponding factory functions with tags.

  To store a function in a `TaggedTasks` container, we can use its `.add`
  decorator:

  ```python
  tasks = TaggedTasks()

  @tasks.add('easy', 'stable')
  def example_task():
    ...
    return environment

  environment_factory = tasks['example_task']

  # Or to restrict to a given tag:
  environment_factory = tasks.tagged('easy')['example_task']
  ```
  """

  def __init__(self):
    self._tasks = collections.OrderedDict()
    self._tags = collections.defaultdict(dict)

  def add(self, *tags):
    """Decorator that adds a factory function to the container with tags.

    Args:
      *tags: Strings specifying the tags for this function.

    Returns:
      The same function.

    Raises:
      ValueError: if a function with the same name already exists within the
        container.
    """
    def wrap(factory_func):
      name = factory_func.__name__
      if name in self:
        raise ValueError("Function named {!r} already exists in the container."
                         "".format(name))
      self._tasks[name] = factory_func
      for tag in tags:
        self._tags[tag][name] = factory_func
      return factory_func
    return wrap

  def tagged(self, tag):
    """Returns a (possibly empty) dict of all items that match the given tag."""
    if tag not in self._tags:
      return {}
    else:
      return self._tags[tag]

  def tags(self):
    """Returns a list of all the tags in this container."""
    return list(self._tags.keys())

  def __getitem__(self, k):
    return self._tasks[k]

  def __iter__(self):
    return iter(self._tasks)

  def __len__(self):
    return len(self._tasks)

  def __repr__(self):
    return "{}({})".format(self.__class__.__name__, str(self._tasks))
