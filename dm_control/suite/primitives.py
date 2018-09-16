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

"""Primitives domain (for testing purposes)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

# Internal dependencies.

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers

import numpy as np

_DEFAULT_TIME_LIMIT = 40
SUITE = containers.TaggedTasks()


def get_model_and_assets():
  """Returns a tuple containing the model XML string and a dict of assets."""
  return common.read_model('primitives.xml'), common.ASSETS

@SUITE.add('benchmarking', 'test')
def test(time_limit=_DEFAULT_TIME_LIMIT, random=None):
  """Returns the easy point_mass task."""
  physics = mujoco.Physics.from_xml_string(*get_model_and_assets())
  task = Primitives(random=random)
  return control.Environment(physics, task, time_limit=time_limit)

class Primitives(base.Task):
  """A test task for mujoco primitives"""

  def __init__(self, random=None):
    super(Primitives, self).__init__(random=random)

  def initialize_episode(self, physics):
    pass

  def get_observation(self, physics):
    obs = collections.OrderedDict()
    return obs

  def get_reward(self, physics):
    return 0.0
