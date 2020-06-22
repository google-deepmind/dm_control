# Copyright 2018 The dm_control Authors.
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

"""A structured set of manipulation tasks with a single entry point."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from dm_control import composer as _composer
from dm_control.manipulation import bricks as _bricks
from dm_control.manipulation import lift as _lift
from dm_control.manipulation import place as _place
from dm_control.manipulation import reach as _reach
from dm_control.manipulation.shared import registry as _registry

_registry.done_importing_tasks()

_TIME_LIMIT = 10.
_TIMEOUT = None

ALL = tuple(_registry.get_all_names())
TAGS = tuple(_registry.get_tags())

flags.DEFINE_bool('timeout', True, 'Whether episodes should have a time limit.')
FLAGS = flags.FLAGS


def _get_timeout():
  global _TIMEOUT
  if _TIMEOUT is None:
    if FLAGS.is_parsed():
      _TIMEOUT = FLAGS.timeout
    else:
      _TIMEOUT = FLAGS['timeout'].default
  return _TIMEOUT


def get_environments_by_tag(tag):
  """Returns the names of all environments matching a given tag.

  Args:
    tag: A string from `TAGS`.

  Returns:
    A tuple of environment names.
  """
  return tuple(_registry.get_names_by_tag(tag))


def load(environment_name, seed=None):
  """Loads a manipulation environment.

  Args:
    environment_name: String, the name of the environment to load. Must be in
      `ALL`.
    seed: An optional integer used to seed the task's random number generator.
      If None (default), the random number generator will self-seed from a
      platform-dependent source of entropy.

  Returns:
    An instance of `composer.Environment`.
  """
  task = _registry.get_constructor(environment_name)()
  time_limit = _TIME_LIMIT if _get_timeout() else float('inf')
  return _composer.Environment(task, time_limit=time_limit, random_state=seed)
