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

"""Simple floor arenas."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dm_control import composer
from dm_control.locomotion.arenas import assets as locomotion_arenas_assets
import numpy as np

_GROUNDPLANE_QUAD_SIZE = 0.25


class Floor(composer.Arena):
  """A simple floor arena with a checkered pattern."""

  def _build(self, size=(8, 8), reflectance=.2, aesthetic='default',
             name='floor', top_camera_y_padding_factor=1.1,
             top_camera_distance=100):
    super(Floor, self)._build(name=name)
    self._size = size
    self._top_camera_y_padding_factor = top_camera_y_padding_factor
    self._top_camera_distance = top_camera_distance

    self._mjcf_root.visual.headlight.set_attributes(
        ambient=[.4, .4, .4], diffuse=[.8, .8, .8], specular=[.1, .1, .1])

    if aesthetic != 'default':
      ground_info = locomotion_arenas_assets.get_ground_texture_info(aesthetic)
      sky_info = locomotion_arenas_assets.get_sky_texture_info(aesthetic)
      texturedir = locomotion_arenas_assets.get_texturedir(aesthetic)
      self._mjcf_root.compiler.texturedir = texturedir

      self._ground_texture = self._mjcf_root.asset.add(
          'texture', name='aesthetic_texture', file=ground_info.file,
          type=ground_info.type)
      self._ground_material = self._mjcf_root.asset.add(
          'material', name='aesthetic_material', texture=self._ground_texture,
          texuniform='true')
      self._skybox = self._mjcf_root.asset.add(
          'texture', name='aesthetic_skybox', file=sky_info.file,
          type='skybox', gridsize=sky_info.gridsize,
          gridlayout=sky_info.gridlayout)
    else:
      self._ground_texture = self._mjcf_root.asset.add(
          'texture',
          rgb1=[.2, .3, .4],
          rgb2=[.1, .2, .3],
          type='2d',
          builtin='checker',
          name='groundplane',
          width=200,
          height=200,
          mark='edge',
          markrgb=[0.8, 0.8, 0.8])
      self._ground_material = self._mjcf_root.asset.add(
          'material',
          name='groundplane',
          texrepeat=[2, 2],  # Makes white squares exactly 1x1 length units.
          texuniform=True,
          reflectance=reflectance,
          texture=self._ground_texture)

    # Build groundplane.
    self._ground_geom = self._mjcf_root.worldbody.add(
        'geom',
        type='plane',
        name='groundplane',
        material=self._ground_material,
        size=list(size) + [_GROUNDPLANE_QUAD_SIZE])

    # Choose the FOV so that the floor always fits nicely within the frame
    # irrespective of actual floor size.
    fovy_radians = 2 * np.arctan2(top_camera_y_padding_factor * size[1],
                                  top_camera_distance)
    self._top_camera = self._mjcf_root.worldbody.add(
        'camera',
        name='top_camera',
        pos=[0, 0, top_camera_distance],
        quat=[1, 0, 0, 0],
        fovy=np.rad2deg(fovy_radians))

  @property
  def ground_geoms(self):
    return (self._ground_geom,)

  def regenerate(self, random_state):
    pass

  @property
  def size(self):
    return self._size
