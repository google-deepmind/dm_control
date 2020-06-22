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

"""Tools for defining and visualizing workspaces for manipulation tasks.

Workspaces define distributions from which the initial positions and/or
orientations of the hand and prop(s) are sampled, plus other task-specific
spatial parameters such as target sizes.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from dm_control.composer.variation import distributions
from dm_control.composer.variation import rotations
from dm_control.entities.manipulators import base
from dm_control.manipulation.shared import constants
import numpy as np


_MIN_SITE_DIMENSION = 1e-6  # Ensures that all site dimensions are positive.
_VISIBLE_GROUP = 0
_INVISIBLE_GROUP = 3  # Invisible sensor sites live in group 4 by convention.

DOWN_QUATERNION = base.DOWN_QUATERNION

BoundingBox = collections.namedtuple('BoundingBox', ['lower', 'upper'])

uniform_z_rotation = rotations.QuaternionFromAxisAngle(
    axis=(0., 0., 1.),
    # NB: We must specify `single_sample=True` here otherwise we will sample a
    #     length-4 array of angles rather than a scalar. This happens because
    #     `PropPlacer` passes in the previous quaternion as `initial_value`,
    #     and by default `distributions.Distribution` assumes that the shape
    #     of the output array should be the same as that of `initial_value`.
    angle=distributions.Uniform(-np.pi, np.pi, single_sample=True))


def add_bbox_site(body, lower, upper, visible=False, **kwargs):
  """Adds a site for visualizing a bounding box to an MJCF model.

  Args:
    body: An `mjcf.Element`, the (world)body to which the site should be added.
    lower: A sequence of lower x,y,z bounds.
    upper: A sequence of upper x,y,z bounds.
    visible: Whether the site should be visible by default.
    **kwargs: Keyword arguments used to set other attributes of the newly
      created site.

  Returns:
    An `mjcf.Element` representing the newly created site.
  """
  upper = np.array(upper)
  lower = np.array(lower)
  pos = (upper + lower) / 2.
  size = np.maximum((upper - lower) / 2., _MIN_SITE_DIMENSION)
  group = None if visible else constants.TASK_SITE_GROUP
  return body.add(
      'site', type='box', pos=pos, size=size, group=group, **kwargs)


def add_target_site(body, radius, visible=False, **kwargs):
  """Adds a site for visualizing a target location.

  Args:
    body: An `mjcf.Element`, the (world)body to which the site should be added.
    radius: The radius of the target.
    visible: Whether the site should be visible by default.
    **kwargs: Keyword arguments used to set other attributes of the newly
      created site.

  Returns:
    An `mjcf.Element` representing the newly created site.
  """
  group = None if visible else constants.TASK_SITE_GROUP
  return body.add(
      'site', type='sphere', size=[radius], group=group, **kwargs)
