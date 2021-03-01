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
"""Props made of a single primitive MuJoCo geom."""
import itertools

from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable
import numpy as np
_DEFAULT_HALF_LENGTHS = [0.05, 0.1, 0.15]


class Primitive(composer.Entity):
  """A primitive MuJoCo geom prop."""

  def _build(self, geom_type, size, mass=None, name=None):
    """Initializes this prop.

    Args:
      geom_type: a string, one of the types supported by MuJoCo.
      size: a list or numpy array of up to 3 numbers, depending on the type.
      mass: The mass for the primitive geom.
      name: (optional) A string, the name of this prop.
    """
    size = np.reshape(np.asarray(size), -1)
    self._mjcf_root = mjcf.element.RootElement(model=name)

    self._geom = self._mjcf_root.worldbody.add(
        'geom', name='body_geom', type=geom_type, size=size, mass=mass)

    touch_sensor = self._mjcf_root.worldbody.add(
        'site', type=geom_type, name='touch_sensor', size=size*1.05,
        rgba=[1, 1, 1, 0.1],  # touch sensor site is almost transparent
        group=composer.SENSOR_SITES_GROUP)

    self._touch = self._mjcf_root.sensor.add(
        'touch', site=touch_sensor)

    self._position = self._mjcf_root.sensor.add(
        'framepos', name='position', objtype='geom', objname=self.geom)

    self._orientation = self._mjcf_root.sensor.add(
        'framequat', name='orientation', objtype='geom',
        objname=self.geom)

    self._linear_velocity = self._mjcf_root.sensor.add(
        'framelinvel', name='linear_velocity', objtype='geom',
        objname=self.geom)

    self._angular_velocity = self._mjcf_root.sensor.add(
        'frameangvel', name='angular_velocity', objtype='geom',
        objname=self.geom)

    self._name = name

  def _build_observables(self):
    return PrimitiveObservables(self)

  @property
  def geom(self):
    """Returns the primitive's geom, e.g., to change color or friction."""
    return self._geom

  @property
  def touch(self):
    """Exposing the touch sensor for observations and reward."""
    return self._touch

  @property
  def position(self):
    """Ground truth pos sensor."""
    return self._position

  @property
  def orientation(self):
    """Ground truth angular position sensor."""
    return self._orientation

  @property
  def linear_velocity(self):
    """Ground truth velocity sensor."""
    return self._linear_velocity

  @property
  def angular_velocity(self):
    """Ground truth angular velocity sensor."""
    return self._angular_velocity

  @property
  def mjcf_model(self):
    return self._mjcf_root

  @property
  def name(self):
    return self._name


class PrimitiveObservables(composer.Observables,
                           composer.FreePropObservableMixin):
  """Primitive entity's observables."""

  @define.observable
  def position(self):
    return observable.MJCFFeature('sensordata', self._entity.position)

  @define.observable
  def orientation(self):
    return observable.MJCFFeature('sensordata', self._entity.orientation)

  @define.observable
  def linear_velocity(self):
    return observable.MJCFFeature('sensordata', self._entity.linear_velocity)

  @define.observable
  def angular_velocity(self):
    return observable.MJCFFeature('sensordata', self._entity.angular_velocity)

  @define.observable
  def touch(self):
    return observable.MJCFFeature('sensordata', self._entity.touch)


class Sphere(Primitive):
  """A class representing a sphere prop."""

  def _build(self, radius=0.05, mass=None, name='sphere'):
    super(Sphere, self)._build(
        geom_type='sphere', size=radius, mass=mass, name=name)


class Box(Primitive):
  """A class representing a box prop."""

  def _build(self, half_lengths=None, mass=None, name='box'):
    half_lengths = half_lengths or _DEFAULT_HALF_LENGTHS
    super(Box, self)._build(geom_type='box',
                            size=half_lengths,
                            mass=mass,
                            name=name)


class BoxWithSites(Box):
  """A class representing a box prop with sites on the corners."""

  def _build(self, half_lengths=None, mass=None, name='box'):
    half_lengths = half_lengths or _DEFAULT_HALF_LENGTHS
    super(BoxWithSites, self)._build(half_lengths=half_lengths, mass=mass,
                                     name=name)

    corner_positions = itertools.product([half_lengths[0], -half_lengths[0]],
                                         [half_lengths[1], -half_lengths[1]],
                                         [half_lengths[2], -half_lengths[2]])
    corner_sites = []
    for i, corner_pos in enumerate(corner_positions):
      corner_sites.append(
          self._mjcf_root.worldbody.add(
              'site',
              type='sphere',
              name='corner_{}'.format(i),
              size=[0.1],
              pos=corner_pos,
              rgba=[1, 0, 0, 1.0],
              group=composer.SENSOR_SITES_GROUP))
    self._corner_sites = tuple(corner_sites)

  @property
  def corner_sites(self):
    return self._corner_sites


class Ellipsoid(Primitive):
  """A class representing an ellipsoid prop."""

  def _build(self, radii=None, mass=None, name='ellipsoid'):
    radii = radii or _DEFAULT_HALF_LENGTHS
    super(Ellipsoid, self)._build(geom_type='ellipsoid',
                                  size=radii,
                                  mass=mass,
                                  name=name)


class Cylinder(Primitive):
  """A class representing a cylinder prop."""

  def _build(self, radius=0.05, half_length=0.15, mass=None, name='cylinder'):
    super(Cylinder, self)._build(geom_type='cylinder',
                                 size=[radius, half_length],
                                 mass=mass,
                                 name=name)


class Capsule(Primitive):
  """A class representing a capsule prop."""

  def _build(self, radius=0.05, half_length=0.15, mass=None, name='capsule'):
    super(Capsule, self)._build(geom_type='capsule',
                                size=[radius, half_length],
                                mass=mass,
                                name=name)
