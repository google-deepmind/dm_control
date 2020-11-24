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

import collections
import six

_NAME_ALREADY_EXISTS = (
    "A function named {name!r} already exists in the container and "
    "`allow_overriding_keys` is False.")


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

  def __init__(self, allow_overriding_keys=False):
    """Initializes a new `TaggedTasks` container.

    Args:
      allow_overriding_keys: Boolean, whether `add` can override existing keys
        within the container. If False (default), calling `add` multiple times
        with the same function name will result in a `ValueError`.
    """
    self._tasks = collections.OrderedDict()
    self._tags = collections.defaultdict(dict)
    self.allow_overriding_keys = allow_overriding_keys

  def add(self, *tags):
    """Decorator that adds a factory function to the container with tags.

    Args:
      *tags: Strings specifying the tags for this function.

    Returns:
      The same function.

    Raises:
      ValueError: if a function with the same name already exists within the
        container and `allow_overriding_keys` is False.
    """
    def wrap(factory_func):
      name = factory_func.__name__
      if name in self and not self.allow_overriding_keys:
        raise ValueError(_NAME_ALREADY_EXISTS.format(name=name))
      self._tasks[name] = factory_func
      for tag in tags:
        self._tags[tag][name] = factory_func
      return factory_func
    return wrap

  def tagged(self, *tags):
    """Returns a (possibly empty) dict of functions matching all the given tags.

    Args:
      *tags: Strings specifying tags to query by.

    Returns:
      A dict of `{name: function}` containing all the functions that are tagged
      by all of the strings in `tags`.
    """
    if not tags:
      return {}
    tags = set(tags)
    if not tags.issubset(six.viewkeys(self._tags)):
      return {}
    names = six.viewkeys(self._tags[tags.pop()])
    while tags:
      names &= six.viewkeys(self._tags[tags.pop()])
    return {name: self._tasks[name] for name in names}

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
