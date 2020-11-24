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

"""Prop consisting of a single geom with position and velocity sensors."""


from dm_control import composer
from dm_control import mjcf
from dm_control.composer import define
from dm_control.composer.observation import observable


class Primitive(composer.Entity):
  """A prop consisting of a single geom with position and velocity sensors."""

  def _build(self, geom_type, size, name=None, **kwargs):
    """Initializes the prop.

    Args:
      geom_type: String specifying the geom type.
      size: List or numpy array of up to 3 numbers, depending on `geom_type`:
        geom_type='box', size=[x_half_length, y_half_length, z_half_length]
        geom_type='capsule', size=[radius, half_length]
        geom_type='cylinder', size=[radius, half_length]
        geom_type='ellipsoid', size=[x_radius, y_radius, z_radius]
        geom_type='sphere', size=[radius]
      name: (optional) A string, the name of this prop.
      **kwargs: Additional geom parameters. Please see the MuJoCo documentation
        for further details: http://www.mujoco.org/book/XMLreference.html#geom.
    """
    self._mjcf_root = mjcf.element.RootElement(model=name)
    self._geom = self._mjcf_root.worldbody.add(
        'geom', name='geom', type=geom_type, size=size, **kwargs)
    self._position = self._mjcf_root.sensor.add(
        'framepos', name='position', objtype='geom', objname=self.geom)
    self._orientation = self._mjcf_root.sensor.add(
        'framequat', name='orientation', objtype='geom', objname=self.geom)
    self._linear_velocity = self._mjcf_root.sensor.add(
        'framelinvel', name='linear_velocity', objtype='geom',
        objname=self.geom)
    self._angular_velocity = self._mjcf_root.sensor.add(
        'frameangvel', name='angular_velocity', objtype='geom',
        objname=self.geom)

  def _build_observables(self):
    return PrimitiveObservables(self)

  @property
  def geom(self):
    """The geom belonging to this prop."""
    return self._geom

  @property
  def position(self):
    """Sensor that returns the prop position."""
    return self._position

  @property
  def orientation(self):
    """Sensor that returns the prop orientation (as a quaternion)."""
    # TODO(b/120829807): Consider returning a rotation matrix instead.
    return self._orientation

  @property
  def linear_velocity(self):
    """Sensor that returns the linear velocity of the prop."""
    return self._linear_velocity

  @property
  def angular_velocity(self):
    """Sensor that returns the angular velocity of the prop."""
    return self._angular_velocity

  @property
  def mjcf_model(self):
    return self._mjcf_root


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
