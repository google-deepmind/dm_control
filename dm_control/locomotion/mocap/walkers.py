# Copyright 2020 The dm_control Authors.
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
"""Helpers for modifying a walker to match mocap data."""

from dm_control import composer
from dm_control import mjcf
from dm_control.locomotion.mocap import mocap_pb2
from dm_control.locomotion.walkers import rescale
import numpy as np


class WalkerInfo(object):
  """Encapsulates routines that modify a walker to match mocap data."""

  def __init__(self, proto):
    """Initializes this object.

    Args:
      proto: A `mocap_pb2.Walker` protocol buffer.
    """
    self._proto = proto

  def check_walker_is_compatible(self, walker):
    """Checks whether a given walker is compatible with this `WalkerInfo`."""
    mocap_model = getattr(walker, 'mocap_walker_model', None)
    if mocap_model is not None and mocap_model != self._proto.model:
      model_type_name = list(mocap_pb2.Walker.Model.keys())[list(
          mocap_pb2.Walker.Model.values()).index(self._proto.model)]
      raise ValueError('Walker is not compatible with model type {!r}: got {}'
                       .format(model_type_name, walker))

  def rescale_walker(self, walker):
    """Rescales a given walker to match the data in this `WalkerInfo`."""
    self.check_walker_is_compatible(walker)
    for subtree_info in self._proto.scaling.subtree:
      body = walker.mjcf_model.find('body', subtree_info.body_name)
      subtree_root = body.parent
      if subtree_info.parent_length:
        position_factor = subtree_info.parent_length / np.linalg.norm(body.pos)
      else:
        position_factor = subtree_info.size_factor
      rescale.rescale_subtree(
          subtree_root, position_factor, subtree_info.size_factor)

    if self._proto.mass:
      physics = mjcf.Physics.from_mjcf_model(walker.mjcf_model.root_model)
      current_mass = physics.bind(walker.root_body).subtreemass
      mass_factor = self._proto.mass / current_mass
      for body in walker.root_body.find_all('body'):
        inertial = getattr(body, 'inertial', None)
        if inertial:
          inertial.mass *= mass_factor
      for geom in walker.root_body.find_all('geom'):
        if geom.mass is not None:
          geom.mass *= mass_factor
        else:
          current_density = geom.density if geom.density is not None else 1000
          geom.density = current_density * mass_factor

  def add_marker_sites(self, walker, size=0.01, rgba=(0., 0., 1., .3),
                       default_to_random_position=True, random_state=None):
    """Adds sites corresponding to mocap tracking markers."""
    self.check_walker_is_compatible(walker)
    random_state = random_state or np.random
    sites = []
    if self._proto.markers:
      mocap_class = walker.mjcf_model.default.add('default', dclass='mocap')
      mocap_class.site.set_attributes(type='sphere', size=(size,), rgba=rgba,
                                      group=composer.SENSOR_SITES_GROUP)
    for marker_info in self._proto.markers.marker:
      body = walker.mjcf_model.find('body', marker_info.parent)
      if not body:
        raise ValueError('Walker model does not contain a body named {!r}'
                         .format(str(marker_info.parent)))
      pos = marker_info.position
      if not pos:
        if default_to_random_position:
          pos = random_state.uniform(-0.005, 0.005, size=3)
        else:
          pos = np.zeros(3)
      sites.append(
          body.add(
              'site', name=str(marker_info.name), pos=pos, dclass=mocap_class))
    walker.list_of_site_names = [site.name for site in sites]
    return sites
