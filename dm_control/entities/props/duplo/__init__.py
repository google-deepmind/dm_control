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

"""A 2x4 Duplo brick."""

import collections
import os

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable
import numpy as np
from six.moves import range

_DUPLO_XML_PATH = os.path.join(os.path.dirname(__file__), 'duplo2x4.xml')

# Stud radii are drawn from a uniform distribution. The `variation` argument
# scales the minimum and maximum whilst keeping the lower quartile constant.
_StudSize = collections.namedtuple(
    '_StudSize', ['minimum', 'lower_quartile', 'maximum'])
_StudParams = collections.namedtuple('_StudParams', ['easy_align', 'flanges'])

_STUD_SIZE_PARAMS = {
    _StudParams(easy_align=False, flanges=False):
        _StudSize(minimum=0.004685, lower_quartile=0.004781, maximum=0.004898),
    _StudParams(easy_align=False, flanges=True):
        _StudSize(minimum=0.004609, lower_quartile=0.004647, maximum=0.004716),
    _StudParams(easy_align=True, flanges=False):
        _StudSize(minimum=0.004754, lower_quartile=0.004844, maximum=0.004953),
    _StudParams(easy_align=True, flanges=True):
        _StudSize(minimum=0.004695, lower_quartile=0.004717, maximum=0.004765)
}

_COLOR_NOT_BETWEEN_0_AND_1 = (
    'All values in `color` must be between 0 and 1, got {!r}.')


class Duplo(composer.Entity):
  """A 2x4 Duplo brick."""

  def _build(self, easy_align=False, flanges=True, variation=0.0,
             color=(1., 0., 0.)):
    """Initializes a new `Duplo` instance.

    Args:
      easy_align: If True, the studs on the top of the brick will be capsules
        rather than cylinders. This makes alignment easier.
      flanges: Whether to use flanges on the bottom of the brick. These make the
        dynamics more expensive, but allow the bricks to be clicked together in
        partially overlapping configurations.
      variation: A float that controls the amount of variation in stud size (and
        therefore separation force). A value of 1.0 results in a distribution of
        separation forces that approximately matches the empirical distribution
        measured for real Duplo bricks. A value of 0.0 yields a deterministic
        separation force approximately equal to the mode of the empirical
        distribution.
      color: An optional tuple of (R, G, B) values specifying the color of the
        Duplo brick. These should be floats between 0 and 1. The default is red.

    Raises:
      ValueError: If `color` contains any value that is not between 0 and 1.
    """
    self._mjcf_root = mjcf.from_path(_DUPLO_XML_PATH)

    stud = self._mjcf_root.default.find('default', 'stud')
    if easy_align:
      # Make cylindrical studs invisible and disable contacts.
      stud.geom.group = 3
      stud.geom.contype = 9
      stud.geom.conaffinity = 8
      # Make capsule studs visible and enable contacts.
      stud_cap = self._mjcf_root.default.find('default', 'stud-capsule')
      stud_cap.geom.group = 0
      stud_cap.geom.contype = 0
      stud_cap.geom.conaffinity = 4
      self._active_stud_dclass = stud_cap
    else:
      self._active_stud_dclass = stud

    if flanges:
      flange_dclass = self._mjcf_root.default.find('default', 'flange')
      flange_dclass.geom.contype = 4  # Enable contact with flanges.

    stud_size = _STUD_SIZE_PARAMS[(easy_align, flanges)]
    offset = (1 - variation) * stud_size.lower_quartile
    self._lower = offset + variation * stud_size.minimum
    self._upper = offset + variation * stud_size.maximum

    self._studs = np.ndarray((2, 4), dtype=object)
    self._holes = np.ndarray((2, 4), dtype=object)

    for row in range(2):
      for column in range(4):
        self._studs[row, column] = self._mjcf_root.find(
            'site', 'stud_{}{}'.format(row, column))
        self._holes[row, column] = self._mjcf_root.find(
            'site', 'hole_{}{}'.format(row, column))

    if not all(0 <= value <= 1 for value in color):
      raise ValueError(_COLOR_NOT_BETWEEN_0_AND_1.format(color))
    self._mjcf_root.default.geom.rgba[:3] = color

  def initialize_episode_mjcf(self, random_state):
    """Randomizes the stud radius (and therefore the separation force)."""
    radius = random_state.uniform(self._lower, self._upper)
    self._active_stud_dclass.geom.size[0] = radius

  def _build_observables(self):
    return DuploObservables(self)

  @property
  def studs(self):
    """A (2, 4) numpy array of `mjcf.Elements` corresponding to stud sites."""
    return self._studs

  @property
  def holes(self):
    """A (2, 4) numpy array of `mjcf.Elements` corresponding to hole sites."""
    return self._holes

  @property
  def mjcf_model(self):
    return self._mjcf_root


class DuploObservables(composer.Observables, composer.FreePropObservableMixin):
  """Observables for the `Duplo` prop."""

  @define.observable
  def position(self):
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.find('sensor', 'position'))

  @define.observable
  def orientation(self):
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.find('sensor', 'orientation'))

  @define.observable
  def linear_velocity(self):
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.find('sensor', 'linear_velocity'))

  @define.observable
  def angular_velocity(self):
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.find('sensor', 'angular_velocity'))

  @define.observable
  def force(self):
    return observable.MJCFFeature(
        'sensordata',
        self._entity.mjcf_model.find('sensor', 'force'))
