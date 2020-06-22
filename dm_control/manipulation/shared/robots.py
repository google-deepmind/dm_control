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

"""Custom robot constructors with manipulation-specific defaults."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control.entities.manipulators import kinova
from dm_control.manipulation.shared import observations


# The default position of the base of the arm relative to the origin.
ARM_OFFSET = (0., 0.4, 0.)


def make_arm(obs_settings):
  """Constructs a robot arm with manipulation-specific defaults.

  Args:
    obs_settings: `observations.ObservationSettings` instance.

  Returns:
    An instance of `manipulators.base.RobotArm`.
  """
  return kinova.JacoArm(
      observable_options=observations.make_options(
          obs_settings, observations.JACO_ARM_OBSERVABLES))


def make_hand(obs_settings):
  """Constructs a robot hand with manipulation-specific defaults.

  Args:
    obs_settings: `observations.ObservationSettings` instance.

  Returns:
    An instance of `manipulators.base.RobotHand`.
  """
  return kinova.JacoHand(
      use_pinch_site_as_tcp=True,
      observable_options=observations.make_options(
          obs_settings, observations.JACO_HAND_OBSERVABLES))
