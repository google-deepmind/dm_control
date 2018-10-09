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

"""Module defining the abstract robot class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from dm_control.composer import entity
import numpy as np
import six

DOWN_QUATERNION = np.array([0., 0.70710678118, 0.70710678118, 0.])


@six.add_metaclass(abc.ABCMeta)
class Robot(entity.Entity):
  """The abstract base class for robots."""

  @abc.abstractproperty
  def actuators(self):
    """Returns the actuator elements of the robot."""
    raise NotImplementedError
