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

"""A collection of MuJoCo-based Reinforcement Learning environments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import inspect
import itertools

from dm_control.rl import control

from dm_control.suite import acrobot
from dm_control.suite import ball_in_cup
from dm_control.suite import cartpole
from dm_control.suite import cheetah
from dm_control.suite import finger
from dm_control.suite import fish
from dm_control.suite import hopper
from dm_control.suite import humanoid
from dm_control.suite import humanoid_CMU
from dm_control.suite import lqr
from dm_control.suite import manipulator
from dm_control.suite import pendulum
from dm_control.suite import point_mass
from dm_control.suite import quadruped
from dm_control.suite import reacher
from dm_control.suite import stacker
from dm_control.suite import swimmer
from dm_control.suite import walker
from dm_control.suite import cloth_v0
from dm_control.suite import cloth_v3
from dm_control.suite import cloth_v4
from dm_control.suite import cloth_v7
from dm_control.suite import cloth_v8
from dm_control.suite import cloth_gripper
from dm_control.suite import cloth_sim
from dm_control.suite import cloth_sim_state
from dm_control.suite import cloth_corner
from dm_control.suite import cloth_point
from dm_control.suite import cloth_point_state
from dm_control.suite import cloth_two_hand
from dm_control.suite import rope_v1
from dm_control.suite import rope_v2
from dm_control.suite import rope_sac
from dm_control.suite import rope_two_hand
from dm_control.suite import rope_colored

# Find all domains imported.
_DOMAINS = {name: module for name, module in locals().items()
            if inspect.ismodule(module) and hasattr(module, 'SUITE')}


def _get_tasks(tag):
  """Returns a sequence of (domain name, task name) pairs for the given tag."""
  result = []

  for domain_name in sorted(_DOMAINS.keys()):
    domain = _DOMAINS[domain_name]

    if tag is None:
      tasks_in_domain = domain.SUITE
    else:
      tasks_in_domain = domain.SUITE.tagged(tag)

    for task_name in tasks_in_domain.keys():
      result.append((domain_name, task_name))

  return tuple(result)


def _get_tasks_by_domain(tasks):
  """Returns a dict mapping from task name to a tuple of domain names."""
  result = collections.defaultdict(list)

  for domain_name, task_name in tasks:
    result[domain_name].append(task_name)

  return {k: tuple(v) for k, v in result.items()}


# A sequence containing all (domain name, task name) pairs.
ALL_TASKS = _get_tasks(tag=None)

# Subsets of ALL_TASKS, generated via the tag mechanism.
BENCHMARKING = _get_tasks('benchmarking')
EASY = _get_tasks('easy')
HARD = _get_tasks('hard')
EXTRA = tuple(sorted(set(ALL_TASKS) - set(BENCHMARKING)))

# A mapping from each domain name to a sequence of its task names.
TASKS_BY_DOMAIN = _get_tasks_by_domain(ALL_TASKS)


def load(domain_name, task_name, task_kwargs=None, environment_kwargs=None,
         visualize_reward=False):
  """Returns an environment from a domain name, task name and optional settings.

  ```python
  env = suite.load('cartpole', 'balance')
  ```

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` of keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.

  Returns:
    The requested environment.
  """
  return build_environment(domain_name, task_name, task_kwargs,
                           environment_kwargs, visualize_reward)


def build_environment(domain_name, task_name, task_kwargs=None,
                      environment_kwargs=None, visualize_reward=False):
  """Returns an environment from the suite given a domain name and a task name.

  Args:
    domain_name: A string containing the name of a domain.
    task_name: A string containing the name of a task.
    task_kwargs: Optional `dict` specifying keyword arguments for the task.
    environment_kwargs: Optional `dict` specifying keyword arguments for the
      environment.
    visualize_reward: Optional `bool`. If `True`, object colours in rendered
      frames are set to indicate the reward at each step. Default `False`.

  Raises:
    ValueError: If the domain or task doesn't exist.

  Returns:
    An instance of the requested environment.
  """
  if domain_name not in _DOMAINS:
    raise ValueError('Domain {!r} does not exist.'.format(domain_name))

  domain = _DOMAINS[domain_name]

  if task_name not in domain.SUITE:
    raise ValueError('Level {!r} does not exist in domain {!r}.'.format(
        task_name, domain_name))

  task_kwargs = task_kwargs or {}
  if environment_kwargs is not None:
    task_kwargs = task_kwargs.copy()
    task_kwargs['environment_kwargs'] = environment_kwargs
  env = domain.SUITE[task_name](**task_kwargs)
  env.task.visualize_reward = visualize_reward
  return env
