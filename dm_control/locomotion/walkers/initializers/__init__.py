# Copyright 2019 The dm_control Authors.
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

"""Initializers for the locomotion walkers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class WalkerInitializer(object):
  """The abstract base class for a walker initializer."""

  @abc.abstractmethod
  def initialize_pose(self, physics, walker, random_state):
    raise NotImplementedError


class UprightInitializer(WalkerInitializer):
  """An initializer that uses the walker-declared upright pose."""

  def initialize_pose(self, physics, walker, random_state):
    all_joints_binding = physics.bind(walker.mjcf_model.find_all('joint'))
    qpos, xpos, xquat = walker.upright_pose
    if qpos is None:
      all_joints_binding.qpos = all_joints_binding.qpos0
    else:
      all_joints_binding.qpos = qpos
    walker.set_pose(physics, position=xpos, quaternion=xquat)
    walker.set_velocity(
        physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))


class RandomlySampledInitializer(WalkerInitializer):
  """Initializer that random selects between many initializers."""

  def __init__(self, initializers):
    self._initializers = initializers
    self.num_initializers = len(initializers)

  def initialize_pose(self, physics, walker, random_state):
    random_initalizer_idx = np.random.randint(0, self.num_initializers)
    self._initializers[random_initalizer_idx].initialize_pose(
        physics, walker, random_state)
